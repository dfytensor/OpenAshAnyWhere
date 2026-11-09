# -*- coding: utf-8 -*-
"""subword n-gram fit -> bias_init_sub.json"""
import sys, json, numpy as np
sys.stdout.reconfigure(encoding='utf-8')
V5 = 'F:/' + '\u5938\u514b' + '/token_period_package/v5_minimind'
sys.path.insert(0, V5)
import core5 as C

V, K, Kfit, J = 2048, 12, 6, 4
s = np.load(r'F:\OpenASH2605\v6_attbias\sub_stream.npy')
tr = s[:int(len(s) * 0.95)]
pi = C.unigram(tr, V)
did = np.zeros(len(tr), dtype=np.int64)
A = C.build_A(tr, did, K, V, pi)
R, sg = C.fit_R_sigma(A, Kfit, J, n_rows=400, seed=1)
out = dict(J=J, R=[float(max(r, 0.5)) for r in np.sort(R)],
           sig=[float(x) for x in sg[np.argsort(R)]], floor=0.01,
           note='subword n-gram fit (K=12, Kfit=6, 22M tokens)')
with open(r'F:\OpenASH2605\v6_attbias\bias_init_sub.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=1)
print('R =', out['R'])
print('sig =', [round(x, 2) for x in out['sig']])
