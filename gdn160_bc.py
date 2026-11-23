# -*- coding: utf-8 -*-
"""gdn160_bc.py — B/C 段"""
import json
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
r = json.load(open(r'F:\OpenASH2605\longrange_results.json', encoding='utf-8'))
b = r['B_recall']
gaps = ['32', '64', '128', '256', '384']
models = list(b[gaps[0]].keys())
print('=== B: paired recall (rep / gain) ===')
print('gap'.rjust(6) + ''.join(m.replace('full_', '').rjust(24) for m in models))
for gap in gaps:
    line = gap.rjust(6)
    for m in models:
        e = b[gap][m]
        line += ('%.2f %+0.2f' % (e[0], e[2])).rjust(24)
    print(line)
print()
c = r['C_pos']
print('=== C: pos-bucket (b1..b8) ===')
for m, v in c.items():
    if isinstance(v, list):
        print('  %-22s' % m.replace('full_', ''), ' '.join('%.2f' % x for x in v))
