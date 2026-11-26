# -*- coding: utf-8 -*-
"""gamma_compare2.py — GDN vs DMDA 半衰期对比完整表"""
import json
import math
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
r = json.load(open(r'F:\OpenASH2605\longrange_results.json', encoding='utf-8'))
dg = r['D_gamma']


def hl(g):
    return math.log(0.5) / math.log(g) if 0 < g < 1 else float('inf')


for model, layers in dg.items():
    gms = [rec[0] for rec in layers]
    hls = [hl(g) for g in gms]
    print('%-24s gamma %.4f-%.4f | half-life %.1f-%.1f tok | mean %.1f' % (
        model, min(gms), max(gms), min(hls), max(hls), sum(hls) / len(hls)))
