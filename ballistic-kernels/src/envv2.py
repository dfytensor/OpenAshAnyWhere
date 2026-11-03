# -*- coding: utf-8 -*-
"""【Y】包络 v2: 锚定式衰减 (amplitude-anchored envelope)

洞察: 方向 (V² 维) 在远 lag 不可靠, 但振幅是 1 维标量——
  建全量表很便宜 (bincount 秒级), 噪声对范数的膨胀可用分半估计去偏:
  ||A−π||² ≈ s² + 2·noise²  →  s = sqrt(max(||A−π||² − 2·noise², ε))
包络 v2: d(k) = s_debias(k) / s_model(k)  → 方向来自弹道外推, 振幅来自逐 lag 实测
对照: plain / trend(窗内斜率) / amp_raw(不去偏) / amp(去偏)
"""
import sys, json, time
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from scipy.optimize import minimize
import core5 as C

V, K = 1500, 24
NTR, NDEV, NTE = 16_000_000, 2_000_000, 3_000_000
NDEVPOS, NTEPOS = 3000, 150_000
Kfit = 6
OUT, t0 = {}, time.time()

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
rng = np.random.default_rng(7)
dev_pos = C.sample_positions(dv_ids, dv_did, K, NDEVPOS, rng)
te_pos = C.sample_positions(te_ids, te_did, K, NTEPOS, rng)

# 分半噪声 (逐 k)
m_ev = (tr_did % 2 == 0)
def take(mask):
    i_, d_ = tr_ids[mask], tr_did[mask]
    b = np.concatenate([[0], np.flatnonzero(np.diff(d_)) + 1]).astype(np.int64)
    return i_, C.doc_ids_of(len(i_), b)
Ae = C.build_A(*take(m_ev), K, V, pi)
Ao = C.build_A(*take(~m_ev), K, V, pi)
noise = np.linalg.norm((Ae[1:] - Ao[1:]).reshape(K, -1), axis=1).astype(np.float64) / 2
sig_true = C.signal_curve(A, pi)
print(f'数据就绪 [{time.time()-t0:.0f}s]')

# 弹道拟合 (鲁棒版)
R, sg = C.fit_R_sigma(A, Kfit, 4, seed=1)
g, phi = C.make_phi(A, Kfit, R, sg)
Aball = C.make_tables(R, sg, phi, K)
print(f'J=4 拟合: R={np.round(np.sort(R),2)} sig={np.round(np.sort(sg),2)} [{time.time()-t0:.0f}s]')

def run_model(Atable, Kuse, tag):
    F = C.lag_features(dv_ids, dev_pos, Atable, pi, Kuse)
    om, _ = C.calib_weights(F, dv_ids[dev_pos + 1], pi)
    ppl, _ = C.eval_ppl(te_ids, te_pos, Atable, pi, om)
    print(f'  {tag:<30} ppl={ppl:8.2f}')
    return float(ppl)

def apply_env(Aball, dk):
    Ah = np.empty_like(Aball); Ah[0] = Aball[0]
    pif = pi[None, :].astype(np.float32)
    for i, d in enumerate(dk):
        m = pif + d * (Aball[i + 1] - pif)
        m /= m.sum(1, keepdims=True)
        Ah[i + 1] = m
    return Ah

# --- 包络候选 ---
sm = np.array([np.linalg.norm((Aball[k].astype(np.float64) - pi[None, :]).ravel())
               for k in range(1, K + 1)])          # 模型表自身幅度

dk_plain = np.ones(K)

lk = np.log(np.arange(3, Kfit + 1, dtype=np.float64))
beta = -np.polyfit(lk, np.log(sm[2:Kfit]), 1)[0]
dk_trend = np.ones(K)
for k in range(Kfit + 1, K + 1):
    dk_trend[k - 1] = (k / Kfit) ** (-beta)

# amp_raw: 不去偏 (噪声使远 lag 幅度虚高)
dk_amp_raw = np.ones(K)
# amp: 去偏 s = sqrt(max(||A−π||² − 2·noise²))
s_deb = np.sqrt(np.maximum(sig_true ** 2 - 2 * noise ** 2, (0.2 * noise) ** 2))
for k in range(1, K + 1):
    d = s_deb[k - 1] / max(sm[k - 1], 1e-9)
    dk_amp_raw[k - 1] = float(np.clip(d, 0.0, 1.0))

print('\n' + '=' * 78)
print('【Y1】包络对照 (弹道 J=4, 拟合 k≤6, 用 k=1..24)')
print('=' * 78)
res = {}
for name, dk, note in [
    ('plain', dk_plain, '无包络'),
    ('trend', dk_trend, f'趋势包络 β={beta:.3f}'),
    ('amp_raw', dk_amp_raw, '振幅锚定(不去偏)'),
    ('amp', None, '振幅锚定(去偏)')]:
    if name == 'amp':
        dk = np.ones(K)
        for k in range(1, K + 1):
            pass
        # amp 与 amp_raw 同形, 但 s_true 用去偏值 — 直接构造
        dk = np.array(dk_amp_raw)
        # 修正: 用 s_deb 而非 raw 范数
        for k in range(1, K + 1):
            dk[k - 1] = float(np.clip(s_deb[k - 1] / max(sm[k - 1], 1e-9), 0.0, 1.0))
        note = f'振幅锚定(去偏) β≈{(np.log(s_deb[-1]/s_deb[5])/np.log(24/6)):.3f}'
    Ah = apply_env(Aball, dk)
    p = run_model(Ah, K, f'{name} ({note})')
    rho = [C.rho_table(Ah[k - 1], A[k], pi) for k in range(Kfit + 1, K + 1)]
    print(f'      ρ外推 mean {np.mean([x[0] for x in rho]):.3f}   幅度比 mean {np.mean([x[1] for x in rho]):.3f}')
    res[name] = dict(ppl=p, rho=float(np.mean([x[0] for x in rho])),
                     amp=float(np.mean([x[1] for x in rho])), note=note)
    OUT[name] = res[name]

print('\n' + '=' * 78)
print('【Y2】对照: 离散全量')
print('=' * 78)
p_disc = run_model(A, K, '离散 全量 k≤24')
res['disc_full'] = p_disc
OUT['disc_full'] = p_disc
best_env = min(res, key=lambda n: res[n]['ppl'] if isinstance(res[n], dict) else 9e9)
print(f"  → 最佳包络: {best_env}  (弹道 {res[best_env]['ppl']:.2f} vs 离散全量 {p_disc:.2f})")

OUT['beta_trend'] = float(beta)
OUT['R'] = R.tolist(); OUT['sig'] = sg.tolist()
with open('envv2.json', 'w', encoding='utf-8') as f:
    json.dump(OUT, f, ensure_ascii=False, indent=1)

plt = C.setup_font()
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.4))
ks = np.arange(1, K + 1)
ax[0].plot(ks, sig_true, 'k-o', ms=4, label='真实幅度')
ax[0].plot(ks, sm, 's--', color='#888', label='弹道表幅度(无包络)')
deb = sm * np.clip(s_deb / np.maximum(sm, 1e-9), 0.0, 1.0)
ax[0].plot(ks, deb, '^-', color='#d62728', label='弹道+去偏包络')
ax[0].axvline(Kfit + 0.5, ls=':', c='gray')
ax[0].set_yscale('log'); ax[0].set_xlabel('lag k'); ax[0].set_ylabel('||A−π||')
ax[0].legend(fontsize=9); ax[0].set_title('振幅: 平台高估 → 去偏锚定修正')
names = list(res.keys())
bvals = [res[n]['ppl'] for n in names if isinstance(res[n], dict)]
ax[1].bar(names, bvals, color=['#aaa', '#7fb3d5', '#f4a582', '#d62728'])
for i, v in enumerate(bvals):
    ax[1].text(i, v + 0.05, f'{v:.2f}', ha='center')
ax[1].set_ylabel('测试 ppl'); ax[1].set_title('包络版本对照 (只用 k≤6 拟合)')
fig.suptitle('包络 v2: 方向外推 + 振幅去偏实测', y=1.02)
fig.tight_layout()
fig.savefig('fig_envv2.png', dpi=130, bbox_inches='tight')
print(f'[saved fig_envv2.png]  [total {time.time()-t0:.0f}s]')
