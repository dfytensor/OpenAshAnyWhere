# -*- coding: utf-8 -*-
"""【D】在 minimind 真实语料上测弹道外推的三个前提条件 (P1 信号/噪声, P2 平滑性, P3 低秩) + P4 周期性探针"""
import sys, json, time
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import core5 as C

V, K = 1500, 24
NTR, NDEV, NTE = 16_000_000, 2_000_000, 3_000_000
OUT = {}
t0 = time.time()

print("=" * 78)
print("【D0】minimind 真实语料 (字符级)")
print("=" * 78)
docs, total = C.load_docs(target_chars=NTR + NDEV + NTE + 500_000)
print(f"  文档数 {len(docs)}   总字符 {total:,}")
chars, stoi = C.build_vocab(docs[:len(docs)], V)
ids_all, bounds_all = C.encode_docs(docs, stoi)
print(f"  词表 V={len(chars)+1} (含UNK)   流长 {len(ids_all):,}")
cov = np.mean(ids_all != 0)
print(f"  top-{V-1} 字符覆盖率 {cov:.3f}")

# 固定切分: 训练 = 前 NTR; dev/test = 后续 (边界处切断文档)
cut1, cut2 = NTR, NTR + NDEV
def seg(lo, hi):
    b = bounds_all[(bounds_all >= lo) & (bounds_all < hi)] - lo
    return ids_all[lo:hi], b

tr_ids, tr_b = seg(0, cut1)
dv_ids, dv_b = seg(cut1, cut2)
te_ids, te_b = seg(cut2, min(len(ids_all), cut2 + NTE))
tr_did = C.doc_ids_of(len(tr_ids), tr_b)
dv_did = C.doc_ids_of(len(dv_ids), dv_b)
te_did = C.doc_ids_of(len(te_ids), te_b)
print(f"  train {len(tr_ids):,} / dev {len(dv_ids):,} / test {len(te_ids):,}")
print(f"  [elapsed {time.time()-t0:.0f}s]")

pi = C.unigram(tr_ids, V)
print("\n" + "=" * 78)
print("【D1】P1: 信号强度 ||A^k−π|| vs 估计噪声 (分半)")
print("=" * 78)
A = C.build_A(tr_ids, tr_did, K, V, pi)
sig = C.signal_curve(A, pi)

# 分半 (按文档奇偶) 估噪声: 拆成两个流, 连续段重新编号成新文档
m_even = (tr_did % 2 == 0)
def take(mask):
    ids, did = tr_ids[mask], tr_did[mask]
    b = np.concatenate([[0], np.flatnonzero(np.diff(did)) + 1]).astype(np.int64)
    return ids, C.doc_ids_of(len(ids), b)
ids_e, did_e = take(m_even)
ids_o, did_o = take(~m_even)
A_e = C.build_A(ids_e, did_e, K, V, pi)
A_o = C.build_A(ids_o, did_o, K, V, pi)
noise = np.linalg.norm((A_e[1:] - A_o[1:]).reshape(K, -1), axis=1).astype(np.float64) / 2.0

print(f"  {'k':>4}{'信号':>12}{'噪声':>12}{'信噪比':>10}")
for k in range(K):
    print(f"  {k+1:>4}{sig[k]:>12.3f}{noise[k]:>12.3f}{sig[k]/noise[k]:>10.2f}")
far = sig[1:]
print(f"  -> k=2..{K} 信号均值 {far.mean():.3f}  标准差 {far.std():.3f}  变异系数 {far.std()/far.mean()*100:.1f}%")
print(f"     k=1 相对 far-field 均值高 {(sig[0]-far.mean())/far.mean()*100:.1f}%")
print(f"  -> 噪声几乎恒定 ~{noise.mean():.3f}; 远场信号 {'高于' if far.mean()>noise.mean()*2 else '接近'} 2×噪声")
OUT['signal'] = sig.tolist(); OUT['noise'] = noise.tolist()
print(f"  [elapsed {time.time()-t0:.0f}s]")

print("\n" + "=" * 78)
print("【D2】P2: 平滑性 ||A^k − A^(k+1)||")
print("=" * 78)
adj = C.adjacent_dist(A)
print(f"  {'k':>4}{'相邻差异':>12}{'隔2差异':>12}")
for k in range(K - 1):
    x2 = np.linalg.norm((A[1+k].astype(np.float64) - A[1+k+2].astype(np.float64)).ravel()) if k + 2 < K else float('nan')
    print(f"  {k+1:>4}{adj[k]:>12.3f}{x2:>12.3f}")
tail = adj[3:]
print(f"  -> k>=4 后相邻差异均值 {tail.mean():.3f}  (噪声水平 {noise.mean():.3f})")
print(f"     相邻差异/噪声 = {tail.mean()/noise.mean():.2f}  → {'结构漂移可观, 外推有风险' if tail.mean()/noise.mean()>3 else '表间变化主要由噪声构成 → 平滑外推合法'}")
OUT['adjacent'] = adj.tolist()

print("\n" + "=" * 78)
print("【D3】P3: 跨 k 联合谱 (居中后)")
print("=" * 78)
s, cum = C.joint_spectrum(A, pi, ncomp=8)
print(f"  前1成分 {cum[0]*100:.1f}%   前2 {cum[1]*100:.1f}%   前3 {cum[2]*100:.1f}%   前4 {cum[3]*100:.1f}%   前6 {cum[5]*100:.1f}%")
J90 = int(np.searchsorted(cum, 0.90) + 1)
print(f"  达到 90% 方差所需成分数: {J90}  → 弹道 J 的谱建议值")
OUT['spectrum'] = cum.tolist(); OUT['J_suggest'] = J90

print("\n" + "=" * 78)
print("【D4】P4: 周期性探针 (标点/句子节奏是否造成 lag 周期结构)")
print("=" * 78)
sk = sig
d2 = np.diff(sk, 2)
peaks = [k + 2 for k in range(1, K - 2) if d2[k] < 0 and d2[k - 1] > 0 and sk[k + 1] > sk[1:4].min() * 1.02]
print(f"  信号曲线局部峰 (lag): {peaks}")
punct = {c: float(pi[stoi[c]]) for c in '，。、；：？！\n' if c in stoi}
print(f"  高频标点占比: " + "  ".join(f"{repr(c)}={p*100:.2f}%" for c, p in punct.items()))
print(f"  -> 若信号在特定 lag 出现 >5% 的峰, 说明存在节奏周期; 弹道需周期核, 否则高斯核足够")

# 字频 top20 供报告
top = np.argsort(-pi)[:20]
inv = {v: k for k, v in stoi.items()}
print("\n  top20 字符: " + " ".join(inv.get(i, 'UNK') if i else 'UNK' for i in top))

OUT['meta'] = dict(V=V, K=K, ntr=len(tr_ids), ndev=len(dv_ids), nte=len(te_ids),
                   n_docs=len(docs), total_chars=int(total), coverage=float(cov), J_suggest=J90)
with open(__file__.rsplit('\\', 1)[0] + r"\diag5.json", 'w', encoding='utf-8') as f:
    json.dump(OUT, f, ensure_ascii=False, indent=1)

# ---------------- 图 ----------------
plt = C.setup_font()
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
ks = np.arange(1, K + 1)
ax[0].plot(ks, sig, 'o-', label='信号 ||A^k−π||')
ax[0].plot(ks, noise, 's--', label='噪声(分半)')
ax[0].set_xlabel('lag k'); ax[0].set_title('信号/噪声 vs 距离'); ax[0].legend(); ax[0].set_yscale('log')
ax[1].plot(ks[:-1], adj, 'o-')
ax[1].axhline(noise.mean(), ls='--', c='r', label=f'噪声水平 {noise.mean():.2f}')
ax[1].set_xlabel('lag k'); ax[1].set_title('相邻表差异 ||A^k−A^(k+1)||'); ax[1].legend()
ax[2].plot(np.arange(1, len(cum) + 1), cum * 100, 'o-')
ax[2].axhline(90, ls=':', c='gray'); ax[2].set_xlabel('成分数'); ax[2].set_ylabel('累计方差 %')
ax[2].set_title(f'跨k联合谱 (90%需 {J90} 个成分)')
fig.suptitle('minimind 真实语料: 弹道外推三前提诊断', y=1.02)
fig.tight_layout()
fig.savefig(__file__.rsplit('\\', 1)[0] + r"\fig5_diag.png", dpi=130, bbox_inches='tight')
print(f"\n[saved fig5_diag.png]  [total {time.time()-t0:.0f}s]")
