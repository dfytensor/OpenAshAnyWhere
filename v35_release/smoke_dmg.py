# -*- coding: utf-8 -*-
"""smoke_dmg.py — v53dmg 冒烟：深度分级衰减阶梯验证"""
import math
import sys

import torch

sys.path.insert(0, r'/mnt/f/OpenASH2605')
from train_ab import build

torch.manual_seed(2024)
m = build('v53dmg').cuda()
tot = sum(p.numel() for p in m.parameters())
print('v53dmg params: %.2fM (GDN %.1f%%)' % (tot / 1e6, tot / 106.3e6 * 100))

print('=== 深度分级衰减阶梯（逐层 8 头） ===')
for L in range(7):
    loga = m.decoder_layers[L].self_attention_linear.kv_loga.detach()
    g = -torch.nn.functional.softplus(loga)
    gamma = g.exp()
    hl = [math.log(0.5) / math.log(x) if 0 < x < 1 else float('inf') for x in gamma.tolist()]
    print('L%d gamma %s' % (L, ' '.join('%.3f' % x for x in gamma.tolist())))
    print('   half-life %s tok' % ' '.join('%.1f' % x for x in hl))

x = torch.randint(0, 23000, (2, 128), device='cuda')
y = torch.randint(0, 23000, (2, 128), device='cuda')
with torch.autocast('cuda', dtype=torch.bfloat16):
    loss, _ = m.forward_loss(x, y, fused_ce=True)
loss.backward()
print('loss %.3f OK' % loss.item())
print('DMG SMOKE OK')
