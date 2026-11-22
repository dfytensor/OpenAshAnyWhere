# -*- coding: utf-8 -*-
"""gdn_gamma.py — 从 GDN checkpoint 提取衰减基线（A_log/dt_bias）与我方对比"""
import math
import sys

import torch

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sd = torch.load(r'F:\OpenASH2605\full_gdn_80k_fixed\model.pt', map_location='cpu', weights_only=True)
ks = [k for k in sd.keys() if any(s in k.lower() for s in ('a_log', 'dt_bias', 'decay', 'g_low', 'a_proj'))]
print('decay-related keys:')
for k in ks[:12]:
    v = sd[k]
    print('  %-50s %s' % (k, tuple(v.shape)))


def hl(g):
    return math.log(0.5) / math.log(g) if 0 < g < 1 else float('inf')


for k in ks:
    if 'a_log' in k.lower():
        v = sd[k]
        # FLA GDN: g_base = -exp(A_log)（逐头或逐通道）
        if v.dim() == 1:
            g = torch.exp(-torch.exp(v))
            hls = [hl(x) for x in g.tolist()[:16]]
            print('%s -> per-unit gamma: %s' % (k, ' '.join('%.3f' % x for x in g.tolist()[:8])))
            print('  half-life: %s tok' % ' '.join('%.1f' % x for x in hls[:8]))
