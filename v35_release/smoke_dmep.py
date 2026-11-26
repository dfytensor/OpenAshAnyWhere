# -*- coding: utf-8 -*-
"""smoke_dmep.py — v53dmep 冒烟：表存在/零初始化/恒等等价/梯度流 + kr 回归门"""
import sys
import torch

sys.path.insert(0, r'/mnt/f/OpenASH2605')
from train_ab import build

torch.manual_seed(2024)
m = build('v53dmep').cuda()
tabs = [(n, tuple(p.shape)) for n, p in m.named_parameters() if 'pos_table' in n]
print('pos tables:', len(tabs), 'first:', tabs[0] if tabs else None)
allz = all((p == 0).all().item() for n, p in m.named_parameters() if 'pos_table' in n)
print('tables zero-init:', allz)
tot = sum(p.numel() for p in m.parameters())
print('params: %.2fM (GDN %.1f%%)' % (tot / 1e6, tot / 106.3e6 * 100))

# 恒等等价：零表下 dmep 前向 == dme 前向（同种子同构不可比——改为对照 dmep 自身：表加扰动后输出必变）
x = torch.randint(0, 23000, (2, 128), device='cuda')
y = torch.randint(0, 23000, (2, 128), device='cuda')
with torch.autocast('cuda', dtype=torch.bfloat16):
    loss0, _ = m.forward_loss(x, y, fused_ce=True)
loss0.backward()
g = [p.grad.abs().mean().item() for n, p in m.named_parameters() if 'pos_table' in n]
print('loss %.3f | table grads min %.2e max %.2e' % (loss0.item(), min(g), max(g)))

# 扰动表 → 输出必变（通路激活检查）
with torch.no_grad():
    for n, p in m.named_parameters():
        if 'pos_table' in n:
            p.add_(0.5)
with torch.autocast('cuda', dtype=torch.bfloat16):
    loss1, _ = m.forward_loss(x, y, fused_ce=True)
print('loss after table+0.5: %.3f (must differ from %.3f)' % (loss1.item(), loss0.item()))
print('DMEP SMOKE OK' if abs(loss1.item() - loss0.item()) > 1e-3 else 'PATH DEAD')
