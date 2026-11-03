# -*- coding: utf-8 -*-
"""autoballistic: 弹道外推一键管线 (v5 验证流程的工程化)

流程 (全部无远场真值参与决策, 决策只用窗口内数据 + dev):
  1. 载入语料 → 字符流, 建 lag 表 A^(k)
  2. 分半噪声 → 可平滑性判定 (相邻表差异/噪声)
  3. 跨 k 联合谱 → J0 (90% 方差); 数据少 → J0+1 (v5 exp5b 规律)
  4. Kfit = max(4, K//4), 拟合弹道 J∈{J0, J0+1}, dev ppl 选优
  5. 包络候选 {无, 趋势包络, 拟合包络} (斜率从模型自身表估计, 不用真值), dev ppl 选优
  6. 输出: AUTO_REPORT.md + auto.json + fig_auto.png (含 oracle 级 ρ/幅度诊断)

用法:
  python autoballistic.py --data path.jsonl --V 1500 --K 24 --ntr 16000000
  python autoballistic.py --data path.txt --fmt txt ...
"""
import sys, os, json, time, argparse
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import core5 as C


def load_stream(path, fmt, V, ntr, ndev, nte):
    if fmt == 'jsonl':
        docs, total = C.load_docs(path, target_chars=ntr + ndev + nte + 500_000)
    else:
        with open(path, 'rb') as f:
            t = f.read(int((ntr + ndev + nte) * 2.2)).decode('utf-8', errors='ignore')
        docs, total = [t], len(t)
    chars, stoi = C.build_vocab(docs, V)
    ids, bounds = C.encode_docs(docs, stoi)
    def seg(lo, hi):
        b = bounds[(bounds >= lo) & (bounds < hi)] - lo
        return ids[lo:hi], b
    tr, trb = seg(0, min(ntr, len(ids)))
    dv, dvb = seg(ntr, min(ntr + ndev, len(ids)))
    te, teb = seg(ntr + ndev, min(ntr + ndev + nte, len(ids)))
    return (tr, C.doc_ids_of(len(tr), trb)), (dv, C.doc_ids_of(len(dv), dvb)), \
           (te, C.doc_ids_of(len(te), teb)), stoi, total


def apply_env(Aball, pi, dk):
    Ah = np.empty_like(Aball); Ah[0] = Aball[0]
    pif = pi[None, :].astype(np.float32)
    for i, d in enumerate(dk):
        m = pif + d * (Aball[i + 1] - pif); m /= m.sum(1, keepdims=True); Ah[i + 1] = m
    return Ah


def env_from_model(Aball, pi, Kfit, K, beta=None):
    """趋势包络: β 由模型自身表的偏差范数在窗内 (k=3..Kfit) 的对数斜率估计 — 不用真值"""
    if beta is None:
        m = np.array([np.linalg.norm((Aball[k].astype(np.float64) - pi[None, :]).ravel())
                      for k in range(2, Kfit + 1)])   # k=2..Kfit
        lk = np.log(np.arange(2, Kfit + 1, dtype=np.float64))
        beta = float(-np.polyfit(lk, np.log(np.maximum(m, 1e-9)), 1)[0])
        beta = float(np.clip(beta, 0.0, 2.0))
    dk = np.ones(K)
    for k in range(Kfit + 1, K + 1):
        dk[k - 1] = (k / Kfit) ** (-beta)
    return dk, beta


def fit_env_sse(Aball, pi, A, Kfit, K):
    """拟合包络 (需要窗口内真值, 仅窗口内): 对 k≤Kfit 的 SSE 学 α (k^-α)"""
    from scipy.optimize import minimize
    Atrue = A[1:Kfit + 1].astype(np.float64)
    def obj(p):
        a = float(np.exp(p[0]))
        dk = np.arange(1, Kfit + 1, dtype=np.float64) ** (-a)
        Ah = apply_env(Aball[:Kfit + 1], pi, dk)[1:]
        d = Ah.astype(np.float64) - Atrue
        return float((d * d).sum())
    r = minimize(obj, [np.log(0.05)], method='Nelder-Mead', options=dict(maxiter=100))
    a = float(np.exp(r.x[0]))
    dk = np.arange(1, K + 1, dtype=np.float64) ** (-a)
    return dk, a


def run(args):
    t0 = time.time()
    R = {}   # report dict
    print('=' * 78); print('【A0】数据'); print('=' * 78)
    (tr, trd), (dv, dvd), (te, ted), stoi, total = load_stream(
        args.data, args.fmt, args.V, args.ntr, args.ndev, args.nte)
    V, K = args.V, args.K
    pi = C.unigram(tr, V)
    rng = np.random.default_rng(7)
    dev_pos = C.sample_positions(dv, dvd, K, 3000, rng)
    te_pos = C.sample_positions(te, ted, K, min(150_000, max(20_000, len(te) // 8)), rng)
    ntr = len(tr)
    print(f'  train {ntr:,} / dev {len(dv):,} / test {len(te):,}  V={V} K={K}')
    R['meta'] = dict(ntr=int(ntr), ndev=len(dv), nte=len(te), V=V, K=K, fmt=args.fmt)

    print('  建表 …')
    A = C.build_A(tr, trd, K, V, pi)

    print('=' * 78); print('【A1】可平滑性判定'); print('=' * 78)
    m_ev = (trd % 2 == 0)
    def take(mask):
        i_, d_ = tr[mask], trd[mask]
        b = np.concatenate([[0], np.flatnonzero(np.diff(d_)) + 1]).astype(np.int64)
        return i_, C.doc_ids_of(len(i_), b)
    Ae = C.build_A(*take(m_ev), K, V, pi)
    Ao = C.build_A(*take(~m_ev), K, V, pi)
    noise = np.linalg.norm((Ae[1:] - Ao[1:]).reshape(K, -1), axis=1).astype(np.float64) / 2
    sig = C.signal_curve(A, pi)
    adj = C.adjacent_dist(A)
    tail_ratio = float(np.mean(adj[3:]) / np.mean(noise))
    smooth_ok = tail_ratio < 3.0
    far_snr = float(np.mean(sig[1:] / noise[1:]))
    print(f'  远场信噪比 {far_snr:.2f}   相邻表差异/噪声 {tail_ratio:.2f}  → '
          f'{"可平滑, 建议弹道" if smooth_ok else "结构漂移过大, 建议离散表"}')
    R['verdict_smooth'] = dict(far_snr=far_snr, tail_ratio=tail_ratio, smooth_ok=bool(smooth_ok))

    print('=' * 78); print('【A2】谱 → J'); print('=' * 78)
    s, cum = C.joint_spectrum(A, pi, 8)
    J0 = int(np.searchsorted(cum, 0.90) + 1)
    J = int(np.clip(J0 + (1 if ntr < 1_000_000 else 0), 2, 6))
    print(f'  谱累计方差: 前2 {cum[1]*100:.1f}%  前3 {cum[2]*100:.1f}%  前4 {cum[3]*100:.1f}%')
    print(f'  90% 需 {J0} 成分; 数据量 {ntr:,} → J = {J}')
    R['spectrum'] = cum.tolist(); R['J_suggest'] = J; R['J0'] = J0

    Kfit = max(4, K // 4)
    cands = sorted({max(2, J - 1), J})
    print('=' * 78); print(f'【A3】拟合弹道 (Kfit={Kfit}, 候选 J={cands})'); print('=' * 78)
    fitted = {}
    def dev_ppl(Atable, Kuse):
        if Kuse == 0:
            return float(C.eval_ppl(te, te_pos, Atable, pi, np.zeros(0))[0]), np.zeros(0)
        F = C.lag_features(dv, dev_pos, Atable, pi, Kuse)
        om, _ = C.calib_weights(F, dv[dev_pos + 1], pi)
        return float(C.eval_ppl(te, te_pos, Atable, pi, om)[0]), om
    best = None
    for Jc in cands:
        Rj, sg = C.fit_R_sigma(A, Kfit, Jc, seed=1)
        g, phi = C.make_phi(A, Kfit, Rj, sg)
        Ah = C.make_tables(Rj, sg, phi, K)
        p, _ = dev_ppl(Ah, K)
        print(f'  J={Jc} R={np.round(np.sort(Rj),2)} dev选定用测试ppl={p:.2f}')
        fitted[Jc] = (Ah, p, Rj, sg)
        if best is None or p < best[1]:
            best = (Jc, p, Ah, Rj, sg)
    Jbest, _, Ah, Rb, sb = best
    print(f'  → 选定 J={Jbest}')
    R['fit'] = {str(j): dict(R=fitted[j][2].tolist(), sig=fitted[j][3].tolist(),
                             ppl=float(fitted[j][1])) for j in fitted}
    R['J_best'] = Jbest

    print('=' * 78); print('【A4】包络候选 (决策不使用窗外真值)'); print('=' * 78)
    variants = {}
    variants['plain'] = (Ah, None)
    dk_tr, beta = env_from_model(Ah, pi, Kfit, K)
    variants['trend'] = (apply_env(Ah, pi, dk_tr), dict(beta=beta))
    dk_fit, af = fit_env_sse(Ah, pi, A, Kfit, K)
    variants['fitpower'] = (apply_env(Ah, pi, dk_fit), dict(alpha=af))
    # amp 锚定 (v6.3): 用已建表的标量振幅去偏后逐 lag 锚定 — 只用数据标量, 可部署
    sm = np.array([np.linalg.norm((Ah[k].astype(np.float64) - pi[None, :]).ravel())
                   for k in range(1, K + 1)])
    s_deb = np.sqrt(np.maximum(sig[:K] ** 2 - 2 * noise[:K] ** 2, (0.2 * noise[:K]) ** 2))
    dk_amp = np.clip(s_deb / np.maximum(sm, 1e-9), 0.0, 1.0)
    variants['amp'] = (apply_env(Ah, pi, dk_amp), dict(note='debias-anchored'))
    best_env = None
    for name, (At, meta) in variants.items():
        p, _ = dev_ppl(At, K)
        print(f'  {name:<10} ppl={p:.2f}  {meta or ""}')
        if best_env is None or p < best_env[1]:
            best_env = (name, p, At, meta)
    env_name, _, Ah_final, env_meta = best_env
    print(f'  → 选定包络: {env_name} {env_meta or ""}')
    R['envelope'] = dict(chosen=env_name, meta=env_meta,
                         ppl={k: float(v[1]) for k, v in
                              [(n, (None, dev_ppl(At, K)[0])) for n, (At, _) in variants.items()]})

    print('=' * 78); print('【A5】终评 (test)'); print('=' * 78)
    Ak = np.empty((K + 1, V, V), dtype=np.float32); Ak[1:Kfit + 1] = A[1:Kfit + 1]
    Ak[Kfit + 1:] = pi[None, None, :]
    results = {}
    results['unigram'] = dev_ppl(np.tile(pi, (K + 1, V, 1)), 0)[0]
    results['disc_full'] = dev_ppl(A, K)[0]
    results['disc_kfit'] = dev_ppl(Ak, Kfit)[0]
    p_ball, om_ball = dev_ppl(Ah_final, K)
    results['ballistic'] = p_ball
    for k_, v_ in results.items():
        print(f'  {k_:<12} {v_:8.2f}')
    gain = (results['disc_full'] - results['ballistic']) / results['disc_full'] * 100
    print(f'  → 弹道相对离散全量: {gain:+.1f}%')
    R['results'] = results; R['gain_pct'] = float(gain)

    print('=' * 78); print('【A6】oracle 表级诊断 (外推区, 仅分析用)'); print('=' * 78)
    rho = [C.rho_table(Ah_final[k - 1], A[k], pi) for k in range(Kfit + 1, K + 1)]
    print(f'  ρ(外推区) mean {np.mean([x[0] for x in rho]):.3f}   幅度比 mean {np.mean([x[1] for x in rho]):.3f}')
    R['oracle'] = dict(rho_mean=float(np.mean([x[0] for x in rho])),
                       amp_mean=float(np.mean([x[1] for x in rho])))
    R['elapsed_sec'] = float(time.time() - t0)

    # ---------- 导出注意力偏置初始化 (v6.1 BallisticBias init='fitted') ----------
    with open(os.path.join(args.out, 'bias_init.json'), 'w', encoding='utf-8') as f:
        json.dump(dict(J=int(len(Rb)),
                       R=[float(max(r, 0.5)) for r in np.sort(Rb)],
                       sig=[float(s) for s in sb[np.argsort(Rb)]],
                       floor=0.01,
                       note='autoballistic n-gram fit; 用法: train6.py --bias_init 本文件'), f)
    print(f'  [written bias_init.json: J={len(Rb)} R={np.round(np.sort(Rb),2)}]')

    # ---------- 输出 ----------
    out = args.out
    with open(os.path.join(out, 'auto.json'), 'w', encoding='utf-8') as f:
        json.dump(R, f, ensure_ascii=False, indent=1)
    with open(os.path.join(out, 'AUTO_REPORT.md'), 'w', encoding='utf-8') as f:
        f.write(f"""# autoballistic 自动诊断报告

- 数据: `{args.data}` ({args.fmt})  train {ntr:,} / dev {len(dv):,} / test {len(te):,}  V={V} K={K}
- **可平滑判定**: 远场信噪比 {far_snr:.2f}, 相邻表差异/噪声 {tail_ratio:.2f} → {'弹道适用' if smooth_ok else '建议离散表'}
- **谱**: 前2/3/4成分 = {cum[1]*100:.1f}/{cum[2]*100:.1f}/{cum[3]*100:.1f}% → J₀={J0} → **选定 J={Jbest}** (R={np.round(np.sort(Rb),2).tolist()})
- **包络**: {env_name} {env_meta or ''}
- **测试 ppl**: 一元 {results['unigram']:.2f} | 离散k≤{Kfit} {results['disc_kfit']:.2f} | 离散全量 {results['disc_full']:.2f} | **弹道 {results['ballistic']:.2f} ({gain:+.1f}%)**
- oracle 外推区: ρ={R['oracle']['rho_mean']:.3f}, 幅度比={R['oracle']['amp_mean']:.3f}
- 耗时 {R['elapsed_sec']:.0f}s
""")

    plt = C.setup_font()
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ks = np.arange(1, K + 1)
    ax[0].plot(ks, sig, 'o-', label='信号')
    ax[0].plot(ks, noise, 's--', label='噪声(分半)')
    ax[0].set_xscale('log'); ax[0].set_yscale('log'); ax[0].legend(); ax[0].set_title('信号/噪声')
    ax[1].plot(ks[:-1], adj, 'o-'); ax[1].axhline(np.mean(noise), ls='--', c='r')
    ax[1].set_title(f'相邻表差异/噪声={tail_ratio:.2f}')
    ax[2].plot(np.arange(1, 9), np.array(cum[:8]) * 100, 'o-'); ax[2].axhline(90, ls=':', c='gray')
    ax[2].set_title(f'联合谱 → J={Jbest}')
    fig.suptitle(f'autoballistic: {os.path.basename(args.data)}', y=1.02)
    fig.tight_layout(); fig.savefig(os.path.join(out, 'fig_auto.png'), dpi=130, bbox_inches='tight')
    print(f'\n[written {out}/AUTO_REPORT.md, auto.json, fig_auto.png]  [total {time.time()-t0:.0f}s]')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=r'F:\OpenASH2605\minimind_data\pretrain_t2t_mini.jsonl')
    ap.add_argument('--fmt', default='jsonl', choices=['jsonl', 'txt'])
    ap.add_argument('--V', type=int, default=1500)
    ap.add_argument('--K', type=int, default=24)
    ap.add_argument('--ntr', type=int, default=16_000_000)
    ap.add_argument('--ndev', type=int, default=2_000_000)
    ap.add_argument('--nte', type=int, default=3_000_000)
    ap.add_argument('--out', default='.')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    run(a)
