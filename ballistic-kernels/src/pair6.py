# -*- coding: utf-8 -*-
"""【P】二阶交互: 量化 + 兑现 (v6.5, 修正版)
  - pair 表 lag=k 的 backoff = 同 lag 的一阶表 A[k][w2] (不是 A[1])
  - 校准/评估流分离
  - pair 特征 = log(P2_pair / backoff_k), 未知对 = 0 (中性)
"""
import sys, json, time
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import core5 as C

V, K = 1500, 24
K2L = [1, 2]
KP = 12000
MINC = 10
NTR, NDEV, NTE = 16_000_000, 2_000_000, 3_000_000
NDEVPOS, NTEPOS = 2000, 150_000
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
rng = np.random.default_rng(7)
pif64 = pi[None, :].astype(np.float64)
print(f'数据就绪 [{time.time()-t0:.0f}s]')

# ---------- 高频 bigram 对 ----------
keys = tr_ids[:-1].astype(np.int64) * V + tr_ids[1:].astype(np.int64)
same = tr_did[:-1] == tr_did[1:]
keys[~same] = -1
kc = np.bincount(keys[keys >= 0], minlength=V * V)
top = np.argsort(-kc)[:KP]
top = top[kc[top] >= MINC]
KPn = len(top)
pmap = -np.ones(V * V, dtype=np.int64)
pmap[top] = np.arange(KPn)
pair_w2 = (top % V).astype(np.int64)
pid_all = pmap[np.clip(keys, 0, None)]
pid_all[~same] = -1
kd = dv_ids[:-1].astype(np.int64) * V + dv_ids[1:].astype(np.int64)
sd = dv_did[:-1] == dv_did[1:]
pd_all = pmap[np.clip(kd, 0, None)]; pd_all[~sd] = -1
kt = te_ids[:-1].astype(np.int64) * V + te_ids[1:].astype(np.int64)
st = te_did[:-1] == te_did[1:]
pd_te = pmap[np.clip(kt, 0, None)]; pd_te[~st] = -1
print(f'高频对: {KPn} 个 (count≥{MINC}), 训练流覆盖率 {np.mean(pid_all >= 0)*100:.1f}%')

# ---------- 一阶: 弹道 + 锚定包络 ----------
A = C.build_A(tr_ids, tr_did, K, V, pi)
A1 = A[1].astype(np.float64)
R, sg = C.fit_R_sigma(A, Kfit := 6, 4, seed=1)
g, phi = C.make_phi(A, Kfit, R, sg)
Aball = C.make_tables(R, sg, phi, K)
pif = pi[None, :].astype(np.float32)
sm = np.array([np.linalg.norm((Aball[k].astype(np.float64) - pif.astype(np.float64)).ravel()) for k in range(1, K + 1)])
sigc = C.signal_curve(A, pi)
noise = np.full(K, 0.85)
s_deb = np.sqrt(np.maximum(sigc ** 2 - 2 * noise ** 2, (0.2 * noise) ** 2))
dk = np.clip(s_deb / np.maximum(sm, 1e-9), 0.0, 1.0)
Aenv = Aball.copy()
for k in range(1, K + 1):
    m = pif64 + dk[k - 1] * (Aball[k].astype(np.float64) - pif64)
    m /= m.sum(1, keepdims=True)
    Aenv[k] = m.astype(np.float32)
print(f'一阶弹道+包络就绪 [{time.time()-t0:.0f}s]')

dev_pos = C.sample_positions(dv_ids, dv_did, K, NDEVPOS, rng)
te_pos = C.sample_positions(te_ids, te_did, K, NTEPOS, np.random.default_rng(7))

# ---------- pair 表: lag=k, backoff=A[k][w2] ----------
C2 = np.zeros((len(K2L), KPn, V), dtype=np.float32)   # log P2
n = len(tr_ids)
for li, k in enumerate(K2L):
    t = np.arange(1, n - k)
    # 对位置 t (覆盖 (ids[t-1], ids[t])), 目标 ids[t+k]
    kt = tr_ids[np.maximum(t - 1, 0)].astype(np.int64) * V + tr_ids[t].astype(np.int64)
    pv = pmap[np.clip(kt, 0, None)]
    s_ = tr_did[np.maximum(t - 1, 0)] == tr_did[t]
    pv[~s_] = -1
    ok = pv >= 0
    idx = pv[ok] * V + tr_ids[t[ok] + k].astype(np.int64)
    bc = np.bincount(idx, minlength=KPn * V).astype(np.float64).reshape(KPn, V)
    cnt = bc.sum(1, keepdims=True)
    bk = A[k].astype(np.float64)[pair_w2]                    # 同 lag 一阶 backoff
    P2 = (bc + 10.0 * bk) / (cnt + 10.0)
    C2[li] = np.log(np.maximum(P2, 1e-12))
print(f'pair 表就绪 [{time.time()-t0:.0f}s]')

# ---------- Phase 1: 交互信息 (dev, 高频对覆盖位置) ----------
print('\n' + '=' * 78)
print('【P1】交互信息: 已知 w2 时额外知道 w1 的价值 (dev, 高频对位置)')
print('=' * 78)
for li, k in enumerate(K2L):
    tgt = dv_ids[dev_pos + 1]
    pv = pd_all[dev_pos]
    ok = pv >= 0
    P1 = A[k].astype(np.float64)[pair_w2[pv[ok]]]            # 同 lag backoff = 一阶表
    lp1 = np.log(np.maximum(P1, 1e-12))[np.arange(int(ok.sum())), tgt[ok]]
    lp2 = C2[li][pv[ok]][np.arange(int(ok.sum())), tgt[ok]]
    gain = float(np.mean(lp1 - lp2))
    # 分解: 有计数 vs 无计数
    print(f'  lag={k}: 覆盖 {ok.mean()*100:.1f}%   交互信息 = {gain:+.3f} nats (负=对专家更优)')
    OUT[f'info_lag{k}'] = gain

# ---------- Phase 2: PoE ----------
print('\n' + '=' * 78)
print('【P2】一阶 vs 一阶+pair专家 (dev 标定, test 终评)')
print('=' * 78)
lpi = np.log(pi).astype(np.float32)

def feats(ids, did, pos, At, with_pairs):
    Fs = []
    for k in range(1, K + 1):
        w = ids[pos + 1 - k]
        Fs.append(np.log(At[k][w].astype(np.float32) / pif))
    if with_pairs:
        kp_ = ids[pos].astype(np.int64) * V + ids[pos - 1].astype(np.int64)
        s_ = did[pos] == did[pos - 1]
        p_ = pmap[np.clip(kp_, 0, None)]
        p_[~s_] = -1
        m = p_ >= 0
        for li, k in enumerate(K2L):
            F = np.zeros((len(pos), V), dtype=np.float32)
            if m.any():
                F[m] = (C2[li][p_[m]] - np.log(np.maximum(A[k].astype(np.float64)[pair_w2[p_[m]]], 1e-12))).astype(np.float32)
            Fs.append(F)
    return Fs

def fit_and_eval(ids_c, did_c, pos_c, ids_e, did_e, pos_e, At, with_pairs):
    Fd = feats(ids_c, did_c, pos_c, At, with_pairs)
    om, _ = C.calib_weights(Fd, ids_c[pos_c + 1], pi)
    tot, nn = 0.0, 0
    for s in range(0, len(pos_e), 4096):
        p = pos_e[s:s + 4096]
        y = ids_e[p + 1].astype(np.int64)
        cur = np.tile(lpi, (len(p), 1))
        for k in range(1, K + 1):
            cur += om[k - 1] * np.log(At[k][ids_e[p + 1 - k]].astype(np.float32) / pif)
        if with_pairs:
            Fe = feats(ids_e, did_e, p, At, True)[K:]
            for i, F in enumerate(Fe):
                cur += om[K + i] * F
        cur -= cur.max(1, keepdims=True)
        lp = cur[np.arange(len(p)), y] - np.log(np.exp(cur).sum(1))
        tot += -lp.sum(); nn += len(p)
    return float(np.exp(tot / nn)), om

p_base, _ = fit_and_eval(dv_ids, dv_did, dev_pos, te_ids, te_did, te_pos, Aenv, False)
print(f'  一阶弹道+包络 (k≤{K})       ppl={p_base:8.2f}')
p_pair, om_pair = fit_and_eval(dv_ids, dv_did, dev_pos, te_ids, te_did, te_pos, Aenv, True)
print(f'  + pair专家 (lag∈{K2L})     ppl={p_pair:8.2f}')
print(f'  离散全量参照                 ppl= 38.57')
okcov = float(np.mean(pd_te[te_pos] >= 0))
print(f'  test 对覆盖率: {okcov*100:.1f}%   pair权重 ω = {np.round(om_pair[K:], 3)}')
OUT.update(p_base=p_base, p_pair=p_pair, coverage=okcov,
           omega_pair=[float(x) for x in om_pair[K:]],
           info={f'lag{k}': OUT.get(f'info_lag{k}') for k in K2L})
with open('pair6.json', 'w', encoding='utf-8') as f:
    json.dump(OUT, f, ensure_ascii=False, indent=1)
print(f'[total {time.time()-t0:.0f}s]')
