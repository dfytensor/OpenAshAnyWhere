# -*- coding: utf-8 -*-
"""【E4b】训练量消融 (J=4): E4 用了 J=2 (合成语料最优), 真实语料上 J=4 才是合适的原型数"""
import sys, json, time
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import core5 as C

V, K = 1500, 24
NDEVPOS, NTEPOS = 3000, 150_000
OUT, t0 = {}, time.time()

docs, total = C.load_docs(target_chars=21_500_000)
chars, stoi = C.build_vocab(docs, V)
ids_all, bounds_all = C.encode_docs(docs, stoi)
def seg(lo, hi):
    b = bounds_all[(bounds_all >= lo) & (bounds_all < hi)] - lo
    return ids_all[lo:hi], b
tr_ids, tr_b = seg(0, 16_000_000)
dv_ids, dv_b = seg(16_000_000, 18_000_000)
te_ids, te_b = seg(18_000_000, min(len(ids_all), 21_000_000))
dv_did = C.doc_ids_of(len(dv_ids), dv_b)
te_did = C.doc_ids_of(len(te_ids), te_b)
dev_pos = C.sample_positions(dv_ids, dv_did, K, NDEVPOS, np.random.default_rng(7))
te_pos = C.sample_positions(te_ids, te_did, K, NTEPOS, np.random.default_rng(7))

def q(At, pis, ku):
    F = C.lag_features(dv_ids, dev_pos, At, pis, ku)
    om, _ = C.calib_weights(F, dv_ids[dev_pos + 1], pis)
    return float(C.eval_ppl(te_ids, te_pos, At, pis, om)[0])

print(f"  {'训练字符':>10}{'离散全量24':>12}{'离散k≤6':>10}{'弹道J4外推':>12}{'弹道J6外推':>12}{'J4增益':>9}")
OUT['rows'] = []
for ntr in [250_000, 1_000_000, 4_000_000, 16_000_000]:
    sub, sb = tr_ids[:ntr], tr_b[tr_b < ntr]
    sd = C.doc_ids_of(len(sub), sb)
    pis = C.unigram(sub, V)
    As = C.build_A(sub, sd, K, V, pis)
    As6 = np.empty((K + 1, V, V), dtype=np.float32); As6[1:7] = As[1:7]; As6[7:] = pis[None, None, :]
    pf, p6 = q(As, pis, K), q(As6, pis, 6)
    row = dict(ntr=int(ntr), disc24=pf, disc6=p6)
    for J in [4, 6]:
        R, sg = C.fit_R_sigma(As, 6, J, seed=1)
        g, phi = C.make_phi(As, 6, R, sg)
        Ah = C.make_tables(R, sg, phi, K)
        pb = q(Ah, pis, K)
        row[f'ballJ{J}'] = pb
        if J == 4:
            rho = np.mean([C.rho_table(Ah[k-1], As[k], pis)[0] for k in range(7, K + 1)])
            row['rho_J4'] = float(rho)
    print(f"  {ntr:>10,}{pf:>12.2f}{p6:>10.2f}{row['ballJ4']:>12.2f}{row['ballJ6']:>12.2f}{(p6-row['ballJ4'])/p6*100:>8.1f}%", flush=True)
    OUT['rows'].append(row)
    with open('exp5b.json', 'w', encoding='utf-8') as f:
        json.dump(OUT, f, ensure_ascii=False, indent=1)
print(f"  [total {time.time()-t0:.0f}s]")
