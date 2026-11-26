# -*- coding: utf-8 -*-
"""dmh_verdict.py — 混合侧记忆双端预注册判据执行"""
import json
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
r = json.load(open(r'F:\OpenASH2605\longrange_results.json', encoding='utf-8'))
a = r['A_len_ppl']
H = a['full_v53dmh_40k']
D = a['full_v53dme_40k']
S = a['full_v53dms_40k']
G = a['full_gdn_80k_fixed']
Ls = (64, 128, 256, 384, 512, 1024, 2048)

print('=== A: ppl by length（三方同 491M tokens + GDN-1ep）===')
print('L'.rjust(6) + 'dmh(混合)'.rjust(12) + 'dms(全门控)'.rjust(13) + 'dme(全永久)'.rjust(13)
      + 'GDN-1ep'.rjust(11) + 'dmh/dme'.rjust(10) + 'dmh/GDN'.rjust(10))
for i, L in enumerate(Ls):
    print(str(L).rjust(6) + ('%.2f' % H[i][2]).rjust(12) + ('%.2f' % S[i][2]).rjust(13)
          + ('%.2f' % D[i][2]).rjust(13) + ('%.2f' % G[i][2]).rjust(11)
          + ('%+.1f%%' % ((H[i][2] / D[i][2] - 1) * 100)).rjust(10)
          + ('%+.1f%%' % ((H[i][2] / G[i][2] - 1) * 100)).rjust(10))

print()
print('=== 预注册判据（两端须同时成立）===')
lm384 = H[3][2]
imp2048 = (1 - H[6][2] / D[6][2]) * 100
lm_ok = (lm384 <= 9.50) and (imp2048 >= 3.2)
print('1 LM 端: @384 %.2f <= 9.50 %s | @2048 相对 dme 改善 %.1f%% >= 3.2%% %s → %s' % (
    lm384, '✓' if lm384 <= 9.50 else '✗', imp2048, '✓' if imp2048 >= 3.2 else '✗',
    '✅ 成立' if lm_ok else '❌ 未达'))

b = r['B_recall']
gaps = ['32', '64', '128', '256', '384']
ms = [m for m in b[gaps[0]] if 'dmh' in m][0]
md = [m for m in b[gaps[0]] if 'dme' in m][0]
mss = [m for m in b[gaps[0]] if 'dms' in m][0]
mg = [m for m in b[gaps[0]] if 'gdn' in m][0]
print()
print('=== B: paired recall (rep / gain) ===')
print('gap'.rjust(6) + ''.join(x.replace('full_', '').rjust(22) for x in (ms, mss, md, mg)))
for gap in gaps:
    print(gap.rjust(6) + ''.join(('%6.2f %+0.2f' % (b[gap][x][0], b[gap][x][2])).rjust(22)
                                 for x in (ms, mss, md, mg)))
d256 = b['256'][ms][0] - b['256'][md][0]
d384 = b['384'][ms][0] - b['384'][md][0]
rec_ok = d256 <= 0.10 and d384 <= 0.10
print('2 召回端: @256 相对 dme %+.2f (<=+0.10 %s) | @384 %+.2f (<=+0.10 %s) → %s' % (
    d256, '✓' if d256 <= 0.10 else '✗', d384, '✓' if d384 <= 0.10 else '✗',
    '✅ 成立' if rec_ok else '❌ 未达'))
print()
print('总裁决: %s' % ('双端成立 → 混合设计成立，进 80k' if (lm_ok and rec_ok) else
                     '仅一端/未达 → 按未达端归档（LM %s / 召回 %s）' % (lm_ok, rec_ok)))
c = r['C_pos']
print()
print('=== C: pos-bucket @384 (b1..b8) ===')
for m, v in c.items():
    if isinstance(v, list):
        print('  %-20s' % m.replace('full_', ''), ' '.join('%.2f' % x for x in v))
