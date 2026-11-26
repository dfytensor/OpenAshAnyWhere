# -*- coding: utf-8 -*-
import sys
import torch

sys.path.insert(0, r'/mnt/f/OpenASH2605')
from train_ab import build

torch.manual_seed(2024)
m = build('v53dme').cuda()
tot = sum(p.numel() for p in m.parameters())
print('v53dme params: %.2fM (GDN %.1f%%) %s' % (tot / 1e6, tot / 106.3e6 * 100,
                                                'OK' if tot <= 106.3e6 else 'OVER CAP'))
dm = sum(p.numel() for n, p in m.named_parameters() if 'dm_' in n)
print('dm side intact: %.2fM' % (dm / 1e6))
x = torch.randint(0, 23000, (2, 128), device='cuda')
y = torch.randint(0, 23000, (2, 128), device='cuda')
with torch.autocast('cuda', dtype=torch.bfloat16):
    loss, _ = m.forward_loss(x, y, fused_ce=True)
loss.backward()
g = [p.grad.abs().mean().item() for n, p in m.named_parameters() if 'dm_o' in n]
print('loss %.3f | dm_o grads %s' % (loss.item(), ['%.1e' % v for v in g]))
print('DME SMOKE OK')
