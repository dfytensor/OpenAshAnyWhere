# -*- coding: utf-8 -*-
"""【E】minimind 真实语料上的弹道外推主实验
  E1 基线 (π / bigram / 离散k≤6 / 离散全量k≤24)
  E2 弹道 J∈{1,2,3,4,6}: 平滑-only vs 外推到 k=1..24; 表级外推恢复率 ρ(k)
  E3 Kfit∈{3,6,9,12} 外推压力测试
  E4 训练量消融 {0.25M,1M,4M,16M}
  E5 硬共享远场基线 (A^(k≥2)=平均表) vs 弹道连续核
  E6 ω 权重剖面
"""
import sys, json, time
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import core5 as C

V, K = 1500, 24
NTR, NDEV, NTE = 16_000_000, 2_000_000, 3_000_000
NDEVPOS, NTEPOS = 3000, 150_000
rng = np.random.default_rng(7)
OUT, t0 = {}, time.time()

print("=" * 78); print("【E0】数据"); print("=" * 78)
docs, total = C.load_docs(target_chars=NTR + NDEV + NTE + 500_000)
chars, stoi = C.build_vocab(docs, V)
ids_all, bounds_all = C.encode_docs(docs, stoi)
def seg(lo, hi):
    b = bounds_all[(bounds_all >= lo) & (bounds_all < hi)] - lo
    return ids_all[lo:hi], b
tr_ids, tr_b = seg(0, NTR)
dv_ids, dv_b = seg(NTR, NTR + NDEV)
te_ids, te_b = seg(NTR + NDEV, min(len(ids_all), NTR + NDEV + NTE))
tr_did = C.doc_ids_of(len(tr_ids), tr_b)
dv_did = C.doc_ids_of(len(dv_ids), dv_b)
te_did = C.doc_ids_of(len(te_ids), te_b)
pi = C.unigram(tr_ids, V)
lpi = np.log(pi)
print(f"  train {len(tr_ids):,} / dev {len(dv_ids):,} / test {len(te_ids):,}  V={V} K={K}")

dev_pos = C.sample_positions(dv_ids, dv_did, K, NDEVPOS, rng)
te_pos  = C.sample_positions(te_ids, te_did, K, NTEPOS, rng)
print(f"  dev 位置 {len(dev_pos)}   test 位置 {len(te_pos)}   [elapsed {time.time()-t0:.0f}s]")

print("\n" + "=" * 78); print("【E1】离散表基线"); print("=" * 78)
A = C.build_A(tr_ids, tr_did, K, V, pi)

def run_model(Atable, Kuse, tag):
    if Kuse == 0:
        ppl, nats = C.eval_ppl(te_ids, te_pos, Atable, pi, np.zeros(0))
        print(f"  {tag:<38} ppl={ppl:8.2f}  ({nats:.3f} nats/字符)  Kuse=0")
        return float(ppl), np.zeros(0)
    F = C.lag_features(dv_ids, dev_pos, Atable, pi, Kuse)
    om, _ = C.calib_weights(F, dv_ids[dev_pos + 1], pi)
    ppl, nats = C.eval_ppl(te_ids, te_pos, Atable, pi, om)
    print(f"  {tag:<38} ppl={ppl:8.2f}  ({nats:.3f} nats/字符)  Kuse={Kuse}")
    return ppl, om

P = {}
P['unigram'] = run_model(np.tile(pi, (K + 1, V, 1)), 0, '一元模型 π')
A_bg = np.empty((K + 1, V, V), dtype=np.float32); A_bg[1:] = A[1]
P['bigram'] = run_model(A_bg, 1, 'bigram (k=1)')
A6 = np.empty((K + 1, V, V), dtype=np.float32); A6[1:7] = A[1:7]
A6[7:] = pi[None, None, :]
P['disc6'] = run_model(A6, 6, '离散 k≤6 (其余 π 兜底)')
P['disc12'] = run_model(np.concatenate([A[:13], np.tile(pi[None, None, :], (K - 12, V, 1)).astype(np.float32)]), 12, '离散 k≤12 (其余 π 兜底)')
P['disc24'] = run_model(A, K, '离散 全量 k≤24')

print("\n" + "=" * 78); print("【E2】弹道: 拟合 k≤6, 外推到 k=1..24"); print("=" * 78)
Kfit = 6
Jres = {}
for J in [1, 2, 3, 4, 6]:
    R, sg = C.fit_R_sigma(A, Kfit, J, seed=1)
    g, phi = C.make_phi(A, Kfit, R, sg)
    Ah = C.make_tables(R, sg, phi, K)
    p6, _ = run_model(Ah, Kfit, f'弹道 J={J} R={np.round(np.sort(R),1)} 仅用k≤6')
    pex, omx = run_model(Ah, K, f'弹道 J={J} 外推到 k≤24')
    rho_fit = [C.rho_table(Ah[k-1], A[k], pi)[0] for k in range(1, Kfit + 1)]
    rho_new, amp_new = {}, {}
    for k in range(Kfit + 1, K + 1):
        r, a = C.rho_table(Ah[k-1], A[k], pi)
        rho_new[k], amp_new[k] = r, a
    print(f"    ρ(拟合区k≤{Kfit}) = {[round(x,3) for x in rho_fit]}")
    print(f"    ρ(外推区k>{Kfit}) = mean {np.mean(list(rho_new.values())):.3f}  min {min(rho_new.values()):.3f}  max {max(rho_new.values()):.3f}")
    print(f"    幅度比 ||Â−π||/||A−π|| (外推区) = mean {np.mean(list(amp_new.values())):.3f}")
    Jres[J] = dict(R=R.tolist(), sig=sg.tolist(), ppl_only6=float(p6), ppl_extrap=float(pex),
                   rho_fit=rho_fit, rho_new=[rho_new[k] for k in sorted(rho_new)],
                   amp_new=[amp_new[k] for k in sorted(amp_new)])
    OUT[f'ballistic_J{J}'] = Jres[J]

best_J = min(Jres, key=lambda j: Jres[j]['ppl_extrap'])
print(f"  -> 最佳 J={best_J}  (外推 ppl {Jres[best_J]['ppl_extrap']:.2f} vs 离散全量 {P['disc24'][0]:.2f})")

print("\n" + "=" * 78); print("【E3】Kfit 压力测试: 短拟合 → 远外推 (J=2 与 J=best)"); print("=" * 78)
for Kf in [3, 6, 9, 12]:
    for J in sorted({2, best_J}):
        R, sg = C.fit_R_sigma(A, Kf, J, seed=1)
        g, phi = C.make_phi(A, Kf, R, sg)
        Ah = C.make_tables(R, sg, phi, K)
        pex, _ = run_model(Ah, K, f'Kfit={Kf} J={J} 外推k≤{K} R={np.round(np.sort(R),1)}')
        rhos = [C.rho_table(Ah[k-1], A[k], pi)[0] for k in range(Kf + 1, K + 1)]
        print(f"    ρ(外推区) mean {np.mean(rhos):.3f}")
        OUT.setdefault('kf_sweep', []).append(dict(Kfit=Kf, J=J, ppl=float(pex), rho=float(np.mean(rhos)), R=R.tolist()))

print("\n" + "=" * 78); print("【E4】训练量消融"); print("=" * 78)
print(f"  {'训练字符':>10}{'离散全量24':>12}{'离散k≤6':>10}{'弹道J2外推':>12}{'增益':>8}")
for ntr in [250_000, 1_000_000, 4_000_000, 16_000_000]:
    sub = tr_ids[:ntr]
    sb = tr_b[tr_b < ntr]
    sd = C.doc_ids_of(len(sub), sb)
    pis = C.unigram(sub, V)
    As = C.build_A(sub, sd, K, V, pis)
    As6 = np.empty((K + 1, V, V), dtype=np.float32); As6[1:7] = As[1:7]; As6[7:] = pis[None, None, :]
    R, sg = C.fit_R_sigma(As, 6, 2, seed=1)
    g, phi = C.make_phi(As, 6, R, sg)
    Ah = C.make_tables(R, sg, phi, K)
    def q(At, ku):
        F = C.lag_features(dv_ids, dev_pos, At, pis, ku)
        om, _ = C.calib_weights(F, dv_ids[dev_pos + 1], pis)
        return C.eval_ppl(te_ids, te_pos, At, pis, om)[0]
    pf, p6, pb = q(As, K), q(As6, 6), q(Ah, K)
    print(f"  {ntr:>10,}{pf:>12.2f}{p6:>10.2f}{pb:>12.2f}{(p6-pb)/p6*100:>7.1f}%")
    OUT.setdefault('ablation', []).append(dict(ntr=int(ntr), disc24=float(pf), disc6=float(p6), ball=float(pb)))

print("\n" + "=" * 78); print("【E5】硬共享远场基线 vs 弹道连续核"); print("=" * 78)
Ash = np.empty((K + 1, V, V), dtype=np.float32)
Ash[1] = A[1]
far_mean = A[2:7].astype(np.float64).mean(0)
far_mean = (far_mean / far_mean.sum(1, keepdims=True)).astype(np.float32)
Ash[2:] = far_mean[None, :, :]
P['shared'] = run_model(Ash, K, '硬共享: k=1原表, k≥2=平均表(A2..A6)')

print("\n" + "=" * 78); print("【E6】ω 权重剖面 (dev 标定)"); print("=" * 78)
_, om_full = run_model(A, K, '离散全量 (重跑取ω)')
R, sg = C.fit_R_sigma(A, 6, best_J, seed=1)
g, phi = C.make_phi(A, 6, R, sg)
Ah = C.make_tables(R, sg, phi, K)
_, om_ball = run_model(Ah, K, '弹道外推 (重跑取ω)')
print(f"  离散 ω_k : {np.round(om_full, 2)}")
print(f"  弹道 ω_k : {np.round(om_ball, 2)}")
OUT['omega_disc'] = om_full.tolist(); OUT['omega_ball'] = om_ball.tolist()

OUT['baselines'] = {k: float(v[0]) for k, v in P.items()}
OUT['best_J'] = int(best_J)
with open('exp5.json', 'w', encoding='utf-8') as f:
    json.dump(OUT, f, ensure_ascii=False, indent=1)

# ---------------- 图 ----------------
plt = C.setup_font()
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.4))
ks_all = np.arange(1, K + 1)
b = OUT['ballistic_J2']
ax[0].plot(ks_all, [1.0] * Kfit + [0.0] * (K - Kfit), 's--', label='离散k≤6兜底 (ρ≡0)', ms=4)
ax[0].plot(ks_all, b['rho_fit'] + b['rho_new'], 'o-', label='弹道 J=2 外推')
ax[0].axvline(Kfit + 0.5, ls=':', c='gray'); ax[0].text(Kfit + 0.8, 0.15, '外推区→')
ax[0].set_xlabel('lag k'); ax[0].set_ylabel('ρ = cos(Â−π, A−π)'); ax[0].set_ylim(0, 1.05)
ax[0].set_title('表级外推恢复率 (真表 vs 模型表)')
names = ['离散全量24', '离散k≤6', f'弹道J={best_J}外推', '硬共享远场']
vals = [P['disc24'][0], P['disc6'][0], Jres[best_J]['ppl_extrap'], P['shared'][0]]
bars = ax[1].bar(names, vals, color=['#888', '#bbb', '#d62728', '#7fb3d5'])
for r_, v in zip(bars, vals):
    ax[1].text(r_.get_x() + r_.get_width() / 2, v + 0.05, f'{v:.2f}', ha='center')
ax[1].set_ylabel('测试困惑度'); ax[1].set_title('minimind 真实语料: 只用 k≤6 vs 全量')
fig.suptitle('弹道外推在真实中文语料上的表现', y=1.02)
fig.tight_layout()
fig.savefig('fig5_extrap.png', dpi=130, bbox_inches='tight')
print(f"\n[saved fig5_extrap.png]  [total {time.time()-t0:.0f}s]")
