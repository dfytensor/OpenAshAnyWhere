# -*- coding: utf-8 -*-
"""dms_verdict.py — ② 侧记忆门控 delta 终审（预注册判据执行）"""
import json
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
r = json.load(open(r'F:\OpenASH2605\longrange_results.json', encoding='utf-8'))
a = r['A_len_ppl']
ks = list(a.keys())
print('models:', ks)
short = {'full_v53dms_40k': 'dms@40k(gated)', 'full_v53dme_40k': 'dme@40k',
         'full_gdn_80k_fixed': 'GDN-1ep', 'full_gdn_160k': 'GDN-2ep'}
print('=== A: ppl by length ===')
print('L'.rjust(6) + ''.join(short[k].rjust(16) for k in ks))
for i, L in enumerate((64, 128, 256, 384, 512, 1024, 2048)):
    print(str(L).rjust(6) + ''.join(('%.2f' % a[k][i][2]).rjust(16) for k in ks))
s = a['full_v53dms_40k']
d = a['full_v53dme_40k']
g1 = a['full_gdn_80k_fixed']
print()
print('=== 预注册判据 ===')
print('1 守门 @384: dms %.2f vs dme %.2f -> %+.1f%%  %s' % (
    s[3][2], d[3][2], (s[3][2] / d[3][2] - 1) * 100,
    '通过(<=+2%)' if s[3][2] / d[3][2] - 1 <= 0.02 else '未通过'))
print('3 长程榜（相对 dme）:')
for i, L in enumerate((512, 1024, 2048)):
    print('   L%-5d dms %.2f vs dme %.2f (%+.1f%%) | vs GDN-1ep %+.1f%%' % (
        L, s[i + 4][2], d[i + 4][2], (s[i + 4][2] / d[i + 4][2] - 1) * 100,
        (s[i + 4][2] / g1[i + 4][2] - 1) * 100))
print()
print('全长度 dms vs dme:')
for i, L in enumerate((64, 128, 256, 384, 512, 1024, 2048)):
    print('   L%-5d %+.1f%% | vs GDN-1ep %+.1f%%' % (
        L, (s[i][2] / d[i][2] - 1) * 100, (s[i][2] / g1[i][2] - 1) * 100))
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
        print('  %-20s' % m.replace('full_', ''), ' '.join('%.2f' % x for x in v))
