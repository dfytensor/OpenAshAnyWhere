# -*- coding: utf-8 -*-
"""b80k.py — dme@80k 配对召回提取"""
import json
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
r = json.load(open(r'F:\OpenASH2605\longrange_results.json', encoding='utf-8'))
b = r['B_recall']
models = list(b.keys())
print('models:', models)
for m in models:
    rec = b[m]
    print('===', m, type(rec).__name__)
    print(json.dumps(rec, ensure_ascii=False)[:400])
    print()
