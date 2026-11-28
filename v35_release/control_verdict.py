# -*- coding: utf-8 -*-
"""control_verdict.py — 协议对照预注册判据 + 两臂相对同协议对照的差"""
import json
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
r = json.load(open(r'F:\OpenASH2605\longrange_results.json', encoding='utf-8'))
a = r['A_len_ppl']
C = a['full_v53dms_40k_bs16x2']
S = a['full_v53dms_40k']
L = a['full_v53dml_40k']
Q = a['full_v53dmq_40k']
G = a['full_gdn_80k_fixed']
Ls = (64, 128, 256, 384, 512, 1024, 2048)

print('=== A: ppl by length ===')
print('L'.rjust(6) + '对照dms(bs16x2)'.rjust(17) + 'dml'.rjust(9) + 'dmq'.rjust(9) + 'dms@40k(bs32)'.rjust(15)
      + 'GDN-1ep'.rjust(10) + '对照/dms-bs32'.rjust(14) + 'dml/对照'.rjust(10) + 'dmq/对照'.rjust(10))
for i, l in enumerate(Ls):
    print(str(l).rjust(6) + ('%.2f' % C[i][2]).rjust(17) + ('%.2f' % L[i][2]).rjust(9) + ('%.2f' % Q[i][2]).rjust(9)
          + ('%.2f' % S[i][2]).rjust(15) + ('%.2f' % G[i][2]).rjust(10)
          + ('%+.1f%%' % ((C[i][2] / S[i][2] - 1) * 100)).rjust(14)
          + ('%+.1f%%' % ((L[i][2] / C[i][2] - 1) * 100)).rjust(10)
          + ('%+.1f%%' % ((Q[i][2] / C[i][2] - 1) * 100)).rjust(10))
print()
print('=== 协议对照预注册判据（锚点：两臂 9.12/9.13，dms-bs32 9.37）===')
c = C[3][2]
print('对照 @384 = %.2f' % c)
if c < 9.25:
    v = '两臂增益与旋钮无关 → 两臂改判未证实，不进 80k'
elif c >= 9.30:
    v = '协议无影响 → 两臂为真，按原预注册进 80k'
else:
    v = '无法区分 → 加跑第二种子'
print('裁决: %s' % v)
print('协议效应（对照 vs dms-bs32）@384: %+.1f%%' % ((c / S[3][2] - 1) * 100))
print('旋钮效应（同协议）@384: dml %+.1f%% | dmq %+.1f%%' % (
    (L[3][2] / C[3][2] - 1) * 100, (Q[3][2] / C[3][2] - 1) * 100))
b = r['B_recall']
gaps = ['32', '64', '128', '256', '384']
ms = [m for m in b[gaps[0]] if 'bs16x2' in m][0]
print()
print('=== 召回 (rep / gain)：对照 | dml | dmq | GDN ===')
for g in gaps:
    print('  gap%-4d' % int(g) + ''.join('  %.2f %+0.2f' % (b[g][x][0], b[g][x][2]) for x in (ms, 'full_v53dml_40k', 'full_v53dmq_40k', 'full_gdn_80k_fixed')))
p = r['C_pos']
print()
print('=== pos-bucket @384 (b1..b8) ===')
for m, v2 in p.items():
    if isinstance(v2, list):
        print('  %-26s' % m.replace('full_', ''), ' '.join('%.2f' % x for x in v2))
