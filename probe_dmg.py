# -*- coding: utf-8 -*-
"""probe_dmg.py — dmg@40k 落地机制探针：阶梯保持性（init vs learned 逐层半衰期）"""
import math
import sys

import torch

sys.path.insert(0, r'/mnt/f/OpenASH2605')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sd = torch.load('/mnt/f/OpenASH2605/full_v53dmg_40k/model.pt', map_location='cpu', weights_only=True)

print('=== 深度阶梯：init 先验 vs 训练后 ===')
print('layer | init hl(center) | learned gamma/head | learned hl/head range')
for L in range(7):
    hl_init = 1.5 * (12.0 / 1.5) ** (L / 6)
    loga = sd['decoder_layers.%d.self_attention_linear.kv_loga' % L]
    g = -torch.nn.functional.softplus(loga)
    gamma = g.exp()
    hls = [math.log(0.5) / math.log(x) if 0 < x < 1 else float('inf') for x in gamma.tolist()]
    print('L%d | %.1f tok | %s | %.1f-%.1f tok' % (
        L, hl_init, ' '.join('%.3f' % x for x in gamma.tolist()), min(hls), max(hls)))

print()
print('（对照 GDN 实测剖面：L0 0.599/1.4tok → L1 0.831/3.7 → L2 0.843/4.1 → L3 0.932/9.9 → L4 0.958/16.1）')
print('（对照 dme@40k 学习后：全层 ~0.82 平坦 / 3.7 tok）')
