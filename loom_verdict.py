# -*- coding: utf-8 -*-
"""loom_verdict.py — Loom 读端口 A/B 预注册判据执行（dml / dmq vs dms@40k / GDN-1ep）"""
import json
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
r = json.load(open(r'F:\OpenASH2605\longrange_results.json', encoding='utf-8'))
a = r['A_len_ppl']
ks = list(a.keys())
print('models:', ks)
short = {'full_v53dml_40k': 'dml(动态路由)', 'full_v53dmq_40k': 'dmq(q归一化)',
         'full_v53dms_40k': 'dms@40k', 'full_gdn_80k_fixed': 'GDN-1ep'}
print('=== A: ppl by length ===')
print('L'.rjust(6) + ''.join(short[k].rjust(16) for k in ks))
for i, L in enumerate((64, 128, 256, 384, 512, 1024, 2048)):
    print(str(L).rjust(6) + ''.join(('%.2f' % a[k][i][2]).rjust(16) for k in ks))
S = a['full_v53dms_40k']
print()
print('=== 预注册判据 ===')
for k in ('full_v53dml_40k', 'full_v53dmq_40k'):
    T = a[k]
    r384 = T[3][2] / S[3][2] - 1
    gate = r384 <= 0.02
    main = T[3][2] < S[3][2]
    print('%s: @384 %.2f vs dms %.2f → %+.1f%% | 守门(<=+2%%) %s | 主判据(<9.37) %s' % (
        short[k], T[3][2], S[3][2], r384 * 100, '✓' if gate else '✗', '✓' if main else '✗'))
    print('   逐长度 vs dms: ' + ' '.join('L%d %+.1f%%' % (L, (T[i][2] / S[i][2] - 1) * 100)
                                         for i, L in enumerate((64, 128, 256, 384, 512, 1024, 2048))))
b = r['B_recall']
gaps = ['32', '64', '128', '256', '384']
print()
print('=== 4 召回标价 (rep / gain) ===')
print('gap'.rjust(6) + ''.join(short[m].rjust(22) for m in ks))
for gap in gaps:
    print(gap.rjust(6) + ''.join(('%6.2f %+0.2f' % (b[gap][m][0], b[gap][m][2])).rjust(22) for m in ks))
for k in ('full_v53dml_40k', 'full_v53dmq_40k'):
    print('   %s − dms: %s' % (short[k], ' '.join('%+.2f' % (b[g][k][0] - b[g]['full_v53dms_40k'][0]) for g in gaps)))
c = r['C_pos']
print()
print('=== C: pos-bucket @384 (b1..b8) ===')
for m, v in c.items():
    if isinstance(v, list):
        print('  %-16s' % short.get(m, m).replace('full_', ''), ' '.join('%.2f' % x for x in v))
