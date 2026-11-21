# -*- coding: utf-8 -*-
"""smoke_gw.py — v53gw 冒烟：γ_kv 参数存在、梯度流通、关闭路径位等价"""
import sys
import torch

sys.path.insert(0, r'/mnt/f/OpenASH2605')
from train_ab import build

torch.manual_seed(2024)
m = build('v53gw').cuda()
n_gamma = sum(1 for n, _ in m.named_parameters() if 'kv_gamma' in n)
print('kv_gamma params:', n_gamma)
total = sum(p.numel() for p in m.parameters())
print('total params: %.1fM' % (total / 1e6))

x = torch.randint(0, 23000, (2, 128), device='cuda')
y = torch.randint(0, 23000, (2, 128), device='cuda')
with torch.autocast('cuda', dtype=torch.bfloat16):
    loss, _ = m.forward_loss(x, y, fused_ce=True)
loss.backward()
g = [p.grad.item() for n, p in m.named_parameters() if 'kv_gamma' in n]
print('loss %.4f | kv_gamma grads:' % loss.item(), ['%.2e' % v for v in g])
print('kv_gamma values:', [p.item() for n, p in m.named_parameters() if 'kv_gamma' in n])

# 关闭路径回归：v53k + kr ckpt 位等价（law #5 自食其果检查）
sd = torch.load('/mnt/f/OpenASH2605/full_v53kr_40k/model.pt', map_location='cpu', weights_only=True)
torch.manual_seed(2024)
m2 = build('v53k').cuda()
missing, unexpected = m2.load_state_dict(sd, strict=False)
assert all('kv_gamma' not in k for k in missing + unexpected), 'gamma keys leaked into v53k'
m2.eval()
x2 = torch.randint(0, 23000, (1, 385), device='cuda')
with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
    logits, _ = m2(x2)
lp = torch.log_softmax(logits[0, :-1].float(), -1)
ce = -lp.gather(-1, x2[0, 1:].unsqueeze(-1)).squeeze(-1).mean().item()
print('kr quick CE (expect ~exp(ln 9.6)=2.26): %.4f -> ppl %.2f' % (ce, pow(2.718281828, ce)))
print('SMOKE OK')
