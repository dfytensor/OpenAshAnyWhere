# -*- coding: utf-8 -*-
"""【X】理论扩展: 弹道 + 衰减包络 (decay envelope)

诊断发现: 真实语料的远场结构不是常数 (合成语料C那样), 而是幂律式缓慢单调衰减
  ||A^k−π|| ~ k^(−0.5) 量级。v3 弹道的远场是「最后一个原型的平坦平台」→ 幅度高估。

扩展思路: 把「形状」和「强度」分离——
  - 形状 (偏差方向 A−π 的相对结构): 弹道原型负责, 沿距离外推
  - 强度 (偏差幅度): 包络 d(k;α) 负责, 只加 1-2 个参数
  Â^(k) = 归一化[ π + d(k;α)·(A_ball^(k) − π) ]
  d(k) = exp(−α(k−1))  或  d(k) = k^(−β)
合成语料 (远场常数) 上 α,β→0 退化为 v3; 真实语料上应修幅度、提升 ρ 与 ppl。
"""
import sys, json, time
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from scipy.optimize import minimize
import core5 as C

V, K = 1500, 24
NTR, NDEV, NTE = 16_000_000, 2_000_000, 3_000_000
NDEVPOS, NTEPOS = 3000, 150_000
OUT, t0 = {}, time.time()

print("=" * 78); print("【X0】数据与表"); print("=" * 78)
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
A = C.build_A(tr_ids, tr_did, K, V, pi)
dev_pos = C.sample_positions(dv_ids, dv_did, K, NDEVPOS, np.random.default_rng(7))
te_pos = C.sample_positions(te_ids, te_did, K, NTEPOS, np.random.default_rng(7))
sig_true = C.signal_curve(A, pi)
print(f"  train {len(tr_ids):,}  [elapsed {time.time()-t0:.0f}s]")

def run_model(Atable, Kuse, tag):
    F = C.lag_features(dv_ids, dev_pos, Atable, pi, Kuse)
    om, _ = C.calib_weights(F, dv_ids[dev_pos + 1], pi)
    ppl, nats = C.eval_ppl(te_ids, te_pos, Atable, pi, om)
    print(f"  {tag:<44} ppl={ppl:8.2f}")
    return ppl

def apply_env(Aball, dk):
    """Â[k] = normalize(π + d_k·(Aball[k]−π)); dk 从 lag=1 开始, Aball 遵循 (K+1) 约定"""
    Ah = np.empty_like(Aball)
    Ah[0] = Aball[0]
    pif = pi[None, :].astype(np.float32)
    for i, d in enumerate(dk):
        m = pif + d * (Aball[i + 1] - pif)
        m /= m.sum(1, keepdims=True)
        Ah[i + 1] = m
    return Ah

def fit_env(Aball, Kfit, family, seed=1):
    """固定弹道形状, 在 k≤Kfit 上拟合包络参数 (对全表 SSE)"""
    Atrue = A[1:Kfit + 1].astype(np.float64)
    def obj(p):
        if family == 'exp':
            dk = np.exp(-np.exp(p[0]) * np.arange(1, Kfit + 1, dtype=np.float64) + 0.0)
            dk = dk / dk[0]
        else:
            dk = np.arange(1, Kfit + 1, dtype=np.float64) ** (-np.exp(p[0]))
        Ah = apply_env(Aball[:Kfit + 1], dk)
        d = Ah[1:].astype(np.float64) - Atrue
        return float((d * d).sum())
    r = minimize(obj, [np.log(0.05)], method='Nelder-Mead', options=dict(maxiter=120))
    a = float(np.exp(r.x[0]))
    if family == 'exp':
        dk_full = np.exp(-a * np.arange(1, K + 1, dtype=np.float64)); dk_full /= dk_full[0]
    else:
        dk_full = np.arange(1, K + 1, dtype=np.float64) ** (-a)
    return a, dk_full

Kfit = 6
print("\n" + "=" * 78); print("【X1】真实衰减形状"); print("=" * 78)
lk = np.log(np.arange(2, K + 1)); ls = np.log(sig_true[1:])
beta_true = -np.polyfit(lk, ls, 1)[0]
print(f"  信号 s_k 幂律拟合: s_k ≈ s_2·(k/2)^{beta_true:.3f}   (合成语料C 上 ≈ 0)")
print(f"  {'k':>4}{'真实s_k':>10}")
for k in [1, 2, 4, 8, 12, 16, 20, 24]:
    print(f"  {k:>4}{sig_true[k-1]:>10.3f}")

print("\n" + "=" * 78); print("【X2】普通弹道 vs 弹道+包络 (拟合 k≤6, 用 k=1..24)"); print("=" * 78)
res = {}
for J in [2, 4]:
    R, sg = C.fit_R_sigma(A, Kfit, J, seed=1)
    g, phi = C.make_phi(A, Kfit, R, sg)
    Aball = C.make_tables(R, sg, phi, K)
    p0 = run_model(Aball, K, f'弹道 J={J} (无包络)')
    rho0 = [C.rho_table(Aball[k-1], A[k], pi) for k in range(Kfit + 1, K + 1)]
    amp0 = [x[1] for x in rho0]
    row = dict(R=R.tolist(), sig=sg.tolist(), ppl_plain=float(p0),
               rho_plain=[x[0] for x in rho0], amp_plain=amp0)

    for fam in ['exp', 'power']:
        a, dk = fit_env(Aball, Kfit, fam)
        Ah = apply_env(Aball, dk)
        p1 = run_model(Ah, K, f'弹道 J={J} + 包络({fam}, α={a:.3f})')
        rho1 = [C.rho_table(Ah[k-1], A[k], pi) for k in range(Kfit + 1, K + 1)]
        print(f"    [{fam}] 包络参数={a:.3f}  ρ外推 mean {np.mean([x[0] for x in rho1]):.3f} (无包络 {np.mean([x[0] for x in rho0]):.3f})"
              f"  幅度比 mean {np.mean([x[1] for x in rho1]):.3f} (无包络 {np.mean(amp0):.3f})")
        row[f'ppl_{fam}'] = float(p1); row[f'env_{fam}'] = a
        row[f'rho_{fam}'] = [x[0] for x in rho1]; row[f'amp_{fam}'] = [x[1] for x in rho1]
        row[f'dk_{fam}'] = dk.tolist()

    # X2b: 趋势外推包络 — 窗口内 (k=3..6) 幂律斜率外推到窗外, 窗内保持 1 (不污染已拟合区)
    lk6 = np.log(np.arange(3, Kfit + 1, dtype=np.float64))
    ls6 = np.log(sig_true[2:Kfit])
    beta = -np.polyfit(lk6, ls6, 1)[0]
    dk_tr = np.ones(K)
    for k in range(Kfit + 1, K + 1):
        dk_tr[k - 1] = (k / Kfit) ** (-beta)
    Ah = apply_env(Aball, dk_tr)
    p2 = run_model(Ah, K, f'弹道 J={J} + 趋势包络(β={beta:.3f})')
    rho2 = [C.rho_table(Ah[k-1], A[k], pi) for k in range(Kfit + 1, K + 1)]
    print(f"    [trend] β={beta:.3f}  ρ外推 mean {np.mean([x[0] for x in rho2]):.3f}  幅度比 mean {np.mean([x[1] for x in rho2]):.3f} (无包络 {np.mean(amp0):.3f})")
    row['ppl_trend'] = float(p2); row['beta_trend'] = float(beta)
    row['rho_trend'] = [x[0] for x in rho2]; row['amp_trend'] = [x[1] for x in rho2]
    row['dk_trend'] = dk_tr.tolist()
    res[J] = row
    OUT[f'J{J}'] = row

print("\n" + "=" * 78); print("【X3】小训练量下包络的增益 (NTR=1M)"); print("=" * 78)
ntr = 1_000_000
sub, sb = tr_ids[:ntr], tr_b[tr_b < ntr]
sd = C.doc_ids_of(len(sub), sb)
pis = C.unigram(sub, V)
As = C.build_A(sub, sd, K, V, pis)
for J in [2, 4]:
    R, sg = C.fit_R_sigma(As, Kfit, J, seed=1)
    g, phi = C.make_phi(As, Kfit, R, sg)
    Aball = C.make_tables(R, sg, phi, K)
    def q(At):
        F = C.lag_features(dv_ids, dev_pos, At, pis, K)
        om, _ = C.calib_weights(F, dv_ids[dev_pos + 1], pis)
        return C.eval_ppl(te_ids, te_pos, At, pis, om)[0]
    p0 = q(Aball)
    a, dk = fit_env(Aball, Kfit, 'power')
    p1 = q(apply_env(Aball, dk))
    print(f"  J={J}: 无包络 {p0:.2f}  +包络(power,{a:.3f}) {p1:.2f}  Δ={p0-p1:+.2f}")
    # 趋势包络 (1M 下同样适用)
    lk6 = np.log(np.arange(3, 7, dtype=np.float64))
    ls6 = np.log(C.signal_curve(As, pis)[2:6])
    bt = -np.polyfit(lk6, ls6, 1)[0]
    dkt = np.ones(K)
    for k in range(7, K + 1):
        dkt[k - 1] = (k / 6.0) ** (-bt)
    p2 = q(apply_env(Aball, dkt))
    print(f"        +趋势包络(β={bt:.3f}) {p2:.2f}  Δ={p0-p2:+.2f}")
    OUT.setdefault('small', []).append(dict(J=J, ppl_plain=float(p0), ppl_env=float(p1), env=a, ppl_trend=float(p2), beta=float(bt)))

with open('decay5.json', 'w', encoding='utf-8') as f:
    json.dump(OUT, f, ensure_ascii=False, indent=1)

plt = C.setup_font()
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.4))
ks = np.arange(1, K + 1)
ax[0].plot(ks, sig_true, 'k-o', ms=4, label='真实 ||A^k−π||')
for J, c in [(2, '#d62728'), (4, '#1f77b4')]:
    amp_p = np.array([1.0] * Kfit + res[J]['amp_plain']) * sig_true[0]
    amp_e = np.array([1.0] * Kfit + res[J]['amp_trend']) * sig_true[0]
    ax[0].plot(ks, amp_p * sig_true[0], 's--', color=c, alpha=0.55, label=f'弹道J={J} 幅度(无包络)')
    ax[0].plot(ks, amp_e * sig_true[0], '^-', color=c, label=f'弹道J={J} 幅度(+power包络)')
ax[0].axvline(Kfit + 0.5, ls=':', c='gray')
ax[0].set_yscale('log'); ax[0].set_xlabel('lag k'); ax[0].set_ylabel('幅度'); ax[0].legend(fontsize=8)
ax[0].set_title('远场幅度: 平坦平台 vs 幂律衰减')
names = ['弹道J2', 'J2+趋势包络', '弹道J4', 'J4+趋势包络']
vals = [res[2]['ppl_plain'], res[2]['ppl_trend'], res[4]['ppl_plain'], res[4]['ppl_trend']]
bars = ax[1].bar(names, vals, color=['#aaa', '#d62728', '#aaa', '#1f77b4'])
for r_, v in zip(bars, vals):
    ax[1].text(r_.get_x() + r_.get_width() / 2, v + 0.05, f'{v:.2f}', ha='center')
ax[1].set_ylabel('测试困惑度'); ax[1].set_title('包络对困惑度的影响 (只用 k≤6 拟合)')
fig.suptitle('弹道 + 衰减包络: 适配真实语料的幂律远场', y=1.02)
fig.tight_layout()
fig.savefig('fig5_decay.png', dpi=130, bbox_inches='tight')
print(f"\n[saved fig5_decay.png]  [total {time.time()-t0:.0f}s]")
