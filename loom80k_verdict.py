# -*- coding: utf-8 -*-
"""loom80k_verdict.py — dml@80k / dmq@80k 同 token 终局：官方 bench + 预注册判据（每臂独立执行）"""
import json
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
r = json.load(open(r'F:\OpenASH2605\longrange_results.json', encoding='utf-8'))
a = r['A_len_ppl']
Ls = (64, 128, 256, 384, 512, 1024, 2048)
K = {'dml': 'full_v53dml_80k', 'dmq': 'full_v53dmq_80k', 'GDN': 'full_gdn_160k', 'dms': 'full_v53dms_80k'}
A = {k: a[v] for k, v in K.items()}
print('=== A: ppl by length（官方 bench；dml/dmq 各 983M tokens，GDN@160k 982M，dms@80k 983M）===')
print('L'.rjust(6) + 'dml@80k'.rjust(11) + 'dmq@80k'.rjust(11) + 'GDN@160k'.rjust(11) + 'dms@80k'.rjust(11)
      + 'dml/GDN'.rjust(10) + 'dmq/GDN'.rjust(10) + 'dml/dms'.rjust(10) + 'dmq/dms'.rjust(10))
for i, l in enumerate(Ls):
    print(str(l).rjust(6) + ''.join(('%.2f' % A[k][i][2]).rjust(11) for k in ('dml', 'dmq', 'GDN', 'dms'))
          + ''.join(('%+.1f%%' % ((A[x][i][2] / A[y][i][2] - 1) * 100)).rjust(10)
                    for x, y in (('dml', 'GDN'), ('dmq', 'GDN'), ('dml', 'dms'), ('dmq', 'dms'))))
print()
print('=== 预注册判据（每臂独立）===')
for k in ('dml', 'dmq'):
    d = A[k][3][2]
    g = A['GDN'][3][2]
    s = A['dms'][3][2]
    nw = sum(1 for i in range(7) if A[k][i][2] <= A['GDN'][i][2] * 1.003)
    wins = sum(1 for i in range(7) if A[k][i][2] < A['GDN'][i][2] * 0.997)
    print('%s@80k:' % k)
    print('  1 主判据  @384 %.2f < GDN %.2f : %s (%+.1f%%)' % (d, g, 'PASS' if d < g else 'FAIL', (d / g - 1) * 100))
    print('  2 全长度  不劣于 GDN %d/7（其中严格更优 %d）: %s' % (nw, wins, 'PASS' if nw >= 5 else 'FAIL'))
    print('  3 规模存续 vs dms@80k @384 %+.1f%%（需 <= -1.0%%）: %s' % (
        (d / s - 1) * 100, 'PASS' if d / s - 1 <= -0.01 else 'FAIL'))
b = r['B_recall']
gaps = ['32', '64', '128', '256', '384']
print()
print('=== 5 召回 rep / gain ===')
print('gap'.rjust(6) + ''.join(k.rjust(22) for k in ('dml', 'dmq', 'GDN', 'dms')))
for g in gaps:
    print(g.rjust(6) + ''.join(('%6.2f %+0.2f' % (b[g][K[k]][0], b[g][K[k]][2])).rjust(22) for k in ('dml', 'dmq', 'GDN', 'dms')))
for k in ('dml', 'dmq'):
    print('  %s − GDN@160k: %s | %s − dms@80k: %s' % (
        k, ' '.join('%+.2f' % (b[g][K[k]][0] - b[g][K['GDN']][0]) for g in gaps),
        k, ' '.join('%+.2f' % (b[g][K[k]][0] - b[g][K['dms']][0]) for g in gaps)))
c = r['C_pos']
print()
print('=== C: pos-bucket @384 (b1..b8) ===')
for k in ('dml', 'dmq', 'GDN', 'dms'):
    print('  %-8s' % k, ' '.join('%.2f' % x for x in c[K[k]]))
