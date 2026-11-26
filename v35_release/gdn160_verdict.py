# -*- coding: utf-8 -*-
"""gdn160_verdict.py — 星标消除战终审（GDN@160k vs dme@80k）"""
import json
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
r = json.load(open(r'F:\OpenASH2605\longrange_results.json', encoding='utf-8'))
ks = list(r['A_len_ppl'].keys())
short = {'full_v53dme_80k': 'dme@80k(105M,2ep)', 'full_gdn_160k': 'GDN@160k(106M,2ep)',
         'full_gdn_80k_fixed': 'GDN@80k(1ep)', 'full_v53dm_v2_40k': 'dm_v2@40k'}
print('=== A: ppl by length ===')
print('L'.rjust(6) + ''.join(short[k].rjust(20) for k in ks))
for i, L in enumerate((64, 128, 256, 384, 512, 1024, 2048)):
    print(str(L).rjust(6) + ''.join(('%.2f' % r['A_len_ppl'][k][i][2]).rjust(20) for k in ks))
print()
a = r['A_len_ppl']
me = a.get('full_v53dme_80k')
g2 = a.get('full_gdn_160k')
g1 = a.get('full_gdn_80k_fixed')
wins = 0
print('=== 星标消除战终审（同 982M tokens，逐长度） ===')
for i, L in enumerate((64, 128, 256, 384, 512, 1024, 2048)):
    d = me[i][2] - g2[i][2]
    tag = 'WIN' if me[i][2] < g2[i][2] else ('TIE' if abs(d) < 0.01 else 'LOSS')
    if me[i][2] < g2[i][2]:
        wins += 1
    print('  L%-5d dme %.2f vs GDN160 %.2f  (%+.1f%%)  %s' % (
        L, me[i][2], g2[i][2], (me[i][2] / g2[i][2] - 1) * 100, tag))
print()
print('主判据 @384: dme@80k %.2f vs GDN@160k %.2f → %s' % (
    me[3][2], g2[3][2], '无星标胜利 ✅' if me[3][2] < g2[3][2] else 'GDN 守住，差距定格'))
print('GDN 斜率: 8.42(80k) → %.2f(160k) = %+.1f%%（我方同期 -13.6%）' % (
    g2[3][2], (g2[3][2] / 8.42 - 1) * 100))
print('LM 战绩: %d/7 长度胜利' % wins)
print()
b = r['B_recall']
gaps = ['32', '64', '128', '256', '384']
models = list(b[gaps[0]].keys())
print('=== B: paired recall ===')
print('gap'.rjust(6) + ''.join(m.replace('full_', '').rjust(22) for m in models))
for gap in gaps:
    line = gap.rjust(6)
    for m in models:
        e = b[gap][m]
        line += ('%.2f %+0.2f' % (e[0], e[2])).rjust(22)
    print(line)
print()
c = r['C_pos']
print('=== C: pos-bucket ===')
for m, v in c.items():
    if isinstance(v, list):
        print('  %-24s' % m.replace('full_', ''), ' '.join('%.2f' % x for x in v))
