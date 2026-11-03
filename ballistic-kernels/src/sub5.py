# -*- coding: utf-8 -*-
"""【S】子词级对照: 用 tokenized 缓存 (免 torch 解析) 检查弹道外推结论是否依赖字符级切分
  规模: 前 18000 个缓存样本 (≈7M token), K=16, V=1200 (高频子词), 其余→UNK
"""
import sys, json, time
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import core5 as C

V, K = 1200, 16
NTR_S, NDEV_S, NTE_S = 16000, 1000, 1000
NDEVPOS, NTEPOS = 3000, 100_000
OUT, t0 = {}, time.time()

print("=" * 78); print("【S0】解析 .pt 缓存"); print("=" * 78)
data = C.load_pt(C.PT_SMALL)
print(f"  样本数 {len(data)}  首样本 dtype {data[0].dtype} shape {data[0].shape}  min {data[0].min()} max {data[0].max()}")
lens = np.array([len(x) for x in data[:NTR_S + NDEV_S + NTE_S]])
print(f"  前{len(lens)}样本长度: mean {lens.mean():.0f}  min {lens.min()}  max {lens.max()}")

ids_list = [np.asarray(x, dtype=np.int64).ravel() for x in data[:NTR_S + NDEV_S + NTE_S]]
alltr = np.concatenate(ids_list[:NTR_S])
# 高频子词表 (训练部分统计; 大于训练最大 id 的 token → UNK)
cnt = np.bincount(alltr)
top = np.argsort(-cnt)[:V - 1]
remap = -np.ones(int(cnt.size), dtype=np.int64); remap[top] = np.arange(1, V)
def enc(arr):
    r = remap[np.minimum(arr, cnt.size - 1)]
    r[arr >= cnt.size] = 0
    r[r < 0] = 0
    return r.astype(np.int32)
n0, n1, n2 = NTR_S, NTR_S + NDEV_S, NTR_S + NDEV_S + NTE_S
def pack(lo, hi):
    ids = enc(np.concatenate(ids_list[lo:hi]))
    bounds = np.concatenate([[0], np.cumsum([len(ids_list[i]) for i in range(lo, hi)])[:-1]]).astype(np.int64)
    return ids, bounds
tr_ids, tr_b = pack(0, n0)
dv_ids, dv_b = pack(n0, n1)
te_ids, te_b = pack(n1, n2)
tr_did, dv_did, te_did = (C.doc_ids_of(len(x), b) for x, b in [(tr_ids, tr_b), (dv_ids, dv_b), (te_ids, te_b)])
pi = C.unigram(tr_ids, V)
print(f"  token: train {len(tr_ids):,} / dev {len(dv_ids):,} / test {len(te_ids):,}  V={V}  覆盖率 {np.mean(tr_ids != 0):.3f}")
print(f"  [elapsed {time.time()-t0:.0f}s]")

A = C.build_A(tr_ids, tr_did, K, V, pi)
sig = C.signal_curve(A, pi)
adj = C.adjacent_dist(A)
s, cum = C.joint_spectrum(A, pi, 6)
print("\n" + "=" * 78); print("【S1】诊断 (子词级)"); print("=" * 78)
print(f"  信号 ||A^k−π||: k=1 {sig[0]:.2f} | k=2 {sig[1]:.2f} | k=6 {sig[5]:.2f} | k=16 {sig[15]:.2f}")
far = sig[1:]
print(f"  k=2..{K} 远场: 均值 {far.mean():.3f} CV {far.std()/far.mean()*100:.1f}%  (字符级为 41.9%)")
print(f"  谱: 前1 {cum[0]*100:.1f}%  前2 {cum[1]*100:.1f}%  前3 {cum[2]*100:.1f}%  前4 {cum[3]*100:.1f}%")
OUT['signal'] = sig.tolist(); OUT['adjacent'] = adj.tolist(); OUT['spectrum'] = cum.tolist()

dev_pos = C.sample_positions(dv_ids, dv_did, K, NDEVPOS, np.random.default_rng(7))
te_pos = C.sample_positions(te_ids, te_did, K, NTEPOS, np.random.default_rng(7))

def run_model(Atable, Kuse, tag):
    F = C.lag_features(dv_ids, dev_pos, Atable, pi, Kuse)
    om, _ = C.calib_weights(F, dv_ids[dev_pos + 1], pi)
    ppl, _ = C.eval_ppl(te_ids, te_pos, Atable, pi, om)
    print(f"  {tag:<40} ppl={ppl:8.2f}")
    return ppl

print("\n" + "=" * 78); print("【S2】基线 vs 弹道 (子词级)"); print("=" * 78)
A6 = np.empty((K + 1, V, V), dtype=np.float32); A6[1:7] = A[1:7]; A6[7:] = pi[None, None, :]
p_disc16 = run_model(A, K, '离散 全量 k≤16')
p_disc6 = run_model(A6, 6, '离散 k≤6 (π 兜底)')
best = {}
for J in [2, 4]:
    R, sg = C.fit_R_sigma(A, 6, J, seed=1)
    g, phi = C.make_phi(A, 6, R, sg)
    Ah = C.make_tables(R, sg, phi, K)
    p_only6 = run_model(Ah, 6, f'弹道 J={J} 仅k≤6  R={np.round(np.sort(R),1)}')
    p_ex = run_model(Ah, K, f'弹道 J={J} 外推到k≤16')
    rho = [C.rho_table(Ah[k-1], A[k], pi)[0] for k in range(7, K + 1)]
    print(f"    ρ(外推区 k=7..16) mean {np.mean(rho):.3f}")
    best[J] = p_ex
    OUT[f'J{J}'] = dict(R=R.tolist(), sig=sg.tolist(), ppl_only6=float(p_only6), ppl_extrap=float(p_ex), rho_new=rho)

OUT['ppl'] = dict(disc16=float(p_disc16), disc6=float(p_disc6),
                  ball2=float(best[2]), ball4=float(best[4]))
with open('sub5.json', 'w', encoding='utf-8') as f:
    json.dump(OUT, f, ensure_ascii=False, indent=1)
print(f"\n[total {time.time()-t0:.0f}s]")
