# -*- coding: utf-8 -*-
"""smoke_dmep2.py — 强扰动验证通路活性"""
import sys
import torch

sys.path.insert(0, r'/mnt/f/OpenASH2605')
from train_ab import build

torch.manual_seed(2024)
m = build('v53dmep').cuda()
torch.manual_seed(7)
x = torch.randint(0, 23000, (2, 512), device='cuda')
y = torch.randint(0, 23000, (2, 512), device='cuda')

with torch.autocast('cuda', dtype=torch.bfloat16):
    l0, _ = m.forward_loss(x, y, fused_ce=True)
print('T=512 zero-table loss %.4f' % l0.item())

for bump in (0.5, 2.0, 5.0):
    torch.manual_seed(2024)
    m2 = build('v53dmep').cuda()
    with torch.no_grad():
        for n, p in m2.named_parameters():
            if 'decay_pos_table' in n:
                p.add_(bump)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        l1, _ = m2.forward_loss(x, y, fused_ce=True)
    print('decay_table +%.1f -> loss %.4f (delta %.4f)' % (bump, l1.item(), l1.item() - l0.item()))
    del m2
    torch.cuda.empty_cache()

for bump in (2.0, 5.0):
    torch.manual_seed(2024)
    m2 = build('v53dmep').cuda()
    with torch.no_grad():
        for n, p in m2.named_parameters():
            if 'beta_pos_table' in n:
                p.add_(bump)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        l1, _ = m2.forward_loss(x, y, fused_ce=True)
    print('beta_table +%.1f -> loss %.4f (delta %.4f)' % (bump, l1.item(), l1.item() - l0.item()))
    del m2
    torch.cuda.empty_cache()
