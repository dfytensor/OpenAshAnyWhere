# -*- coding: utf-8 -*-
"""t1024_verdict.py — ①a dme-T1024 vs dme-T384 vs GDN 对照表"""
import json
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
r = json.load(open(r'F:\OpenASH2605\longrange_results.json', encoding='utf-8'))
a = r['A_len_ppl']
ks = list(a.keys())
short = {'full_v53dme_40k_t1024': 'dme-T1024(491M)', 'full_v53dme_40k': 'dme-T384(491M)',
         'full_gdn_80k_fixed': 'GDN-1ep(488M)', 'full_gdn_160k': 'GDN-2ep(982M)'}
print('=== A: ppl by length ===')
print('L'.rjust(6) + ''.join(short[k].rjust(18) for k in ks))
for i, L in enumerate((64, 128, 256, 384, 512, 1024, 2048)):
    print(str(L).rjust(6) + ''.join(('%.2f' % a[k][i][2]).rjust(18) for k in ks))
t = a['full_v53dme_40k_t1024']
d = a['full_v53dme_40k']
g1 = a['full_gdn_80k_fixed']
g2 = a['full_gdn_160k']
print()
print('=== 逐长度差 ===')
for i, L in enumerate((64, 128, 256, 384, 512, 1024, 2048)):
    print('  L%-5d T1024/T384 %+5.1f%% | T1024/GDN-1ep %+5.1f%% | T1024/GDN-2ep %+5.1f%%' % (
        L, (t[i][2] / d[i][2] - 1) * 100, (t[i][2] / g1[i][2] - 1) * 100, (t[i][2] / g2[i][2] - 1) * 100))
b = r['B_recall']
gaps = ['32', '64', '128', '256', '384']
models = list(b[gaps[0]].keys())
print()
print('=== B: paired recall (rep / gain) ===')
print('gap'.rjust(6) + ''.join(m.replace('full_', '').rjust(22) for m in models))
for gap in gaps:
    print(gap.rjust(6) + ''.join(('%6.2f %+0.2f' % (b[gap][m][0], b[gap][m][2])).rjust(22) for m in models))
c = r['C_pos']
print()
print('=== C: pos-bucket @384 (b1..b8) ===')
for m, v in c.items():
    if isinstance(v, list):
        print('  %-22s' % m.replace('full_', ''), ' '.join('%.2f' % x for x in v))
