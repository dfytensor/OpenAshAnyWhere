# -*- coding: utf-8 -*-
"""gamma_compare.py — GDN vs DMDA 逐头衰减/半衰期对比（利用曲线差距诊断）"""
import json
import math
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
r = json.load(open(r'F:\OpenASH2605\longrange_results.json', encoding='utf-8'))
dg = r['D_gamma']

def hl(g):
    return math.log(0.5) / math.log(g) if 0 < g < 1 else float('inf')

for model, layers in dg.items():
    print('===', model)
    for rec in layers:
        if isinstance(rec, dict):
            ks = list(rec.keys())
            print('  keys:', ks[:8])
            break
    break
print()
print(json.dumps(dg, ensure_ascii=False)[:800])
