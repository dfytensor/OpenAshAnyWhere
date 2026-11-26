# -*- coding: utf-8 -*-
import sys
import torch

sys.path.insert(0, r'/mnt/f/OpenASH2605')
from train_ab import build

torch.manual_seed(2024)
m = build('v53gw2').cuda()
g = [p.item() for n, p in m.named_parameters() if 'kv_gamma' in n]
print('gw2 gamma init:', g)
print('params: %.1fM' % (sum(p.numel() for p in m.parameters()) / 1e6))
x = torch.randint(0, 23000, (2, 128), device='cuda')
y = torch.randint(0, 23000, (2, 128), device='cuda')
with torch.autocast('cuda', dtype=torch.bfloat16):
    loss, _ = m.forward_loss(x, y, fused_ce=True)
loss.backward()
gg = [p.grad.item() for n, p in m.named_parameters() if 'kv_gamma' in n]
print('loss %.3f | gamma grads:' % loss.item(), ['%.2e' % v for v in gg])
print('GW2 SMOKE OK')
