# -*- coding: utf-8 -*-
"""dmg_verdict.py — dmg@40k 预注册裁决输出（bench A 表 + 预注册判据）"""
import json
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
r = json.load(open(r'F:\OpenASH2605\longrange_results.json', encoding='utf-8'))
ks = list(r['A_len_ppl'].keys())
short = {'full_v53dmg_40k': 'dmg@40k(105.18M)', 'full_v53dme_40k': 'dme@40k(105.18M)',
         'full_v53dm_v2_40k': 'dm_v2@40k(111.2M)', 'full_gdn_80k_fixed': 'GDN(106.3M)'}
print('=== A: ppl by length ===')
print('L'.rjust(6) + ''.join(short[k].rjust(20) for k in ks))
for i, L in enumerate((64, 128, 256, 384, 512, 1024, 2048)):
    print(str(L).rjust(6) + ''.join(('%.2f' % r['A_len_ppl'][k][i][2]).rjust(20) for k in ks))
print()
dmg = r['A_len_ppl']['full_v53dmg_40k']
dme = r['A_len_ppl']['full_v53dme_40k']
gdn = r['A_len_ppl']['full_gdn_80k_fixed']
print('=== 预注册判据 ===')
print('主判据 dmg@384 = %.2f vs dme 9.50: %s' % (
    dmg[3][2], '先验有效' if dmg[3][2] < 9.50 else '训练覆写先验 → 衰减线关闭'))
for i, L in enumerate((64, 128, 256, 384, 512, 1024, 2048)):
    print('  L%-5d dmg/dme %+.1f%% | dmg/GDN %+.1f%%' % (
        L, (dmg[i][2] / dme[i][2] - 1) * 100, (dmg[i][2] / gdn[i][2] - 1) * 100))
print()
b = r['B_recall']
try:
    print('=== B: recall ===')
    models = list(b.keys())
    gaps = [32, 64, 128, 256, 384]
    print('gap'.rjust(6) + ''.join(m.replace('full_', '').rjust(18) for m in models))
    for gi, gap in enumerate(gaps):
        line = str(gap).rjust(6)
        for m in models:
            rec = b[m]
            entry = rec[gi] if isinstance(rec, list) else rec.get(str(gap))
            line += ('%.2f %+0.2f' % (entry[0], entry[1])).rjust(18)
        print(line)
except Exception as e:
    print('B parse:', e)
print()
c = r['C_pos']
try:
    print('=== C: pos-bucket (b1..b8) ===')
    for m, v in c.items():
        if isinstance(v, list):
            print('  %-24s' % m.replace('full_', ''), ' '.join('%.2f' % x for x in v))
except Exception as e:
    print('C parse:', e)
