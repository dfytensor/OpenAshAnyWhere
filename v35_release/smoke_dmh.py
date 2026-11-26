# -*- coding: utf-8 -*-
"""smoke_dmh.py — 混合侧记忆冒烟 + dms@80k 回归（掩码全 1 路径必须不变）"""
import math
import sys

import torch

sys.path.insert(0, r'/mnt/f/OpenASH2605')
from train_ab import build

torch.manual_seed(2024)
m = build('v53dmh').cuda()
a = m.decoder_layers[0].self_attention_linear
tot = sum(p.numel() for p in m.parameters())
print('v53dmh params: %.3fM | dm_n_gated=%d mask=%s' % (
    tot / 1e6, a.dm_n_gated, a.dm_gmask.tolist()))
print('mask in state_dict:', any('dm_gmask' in k for k in m.state_dict().keys()))
x = torch.randint(0, 23000, (2, 128), device='cuda')
y = torch.randint(0, 23000, (2, 128), device='cuda')
with torch.autocast('cuda', dtype=torch.bfloat16):
    loss, _ = m.forward_loss(x, y, fused_ce=True)
loss.backward()
g = [p.grad.abs().mean().item() for n, p in m.named_parameters() if 'dm_a_proj' in n]
print('loss %.3f | dm_a_proj grads %s' % (loss.item(), ['%.1e' % v for v in g]))
del m
torch.cuda.empty_cache()

# 掩码 x*1.0 逐位不变性（float32）
t = torch.randn(1000, dtype=torch.float32)
print('x*1.0 bitwise:', bool(torch.equal(t * torch.tensor(1.0), t)))

# dms@80k 回归
torch.manual_seed(2024)
raw = torch.load('/mnt/f/OpenASH2605/minimind_data/pretrain_cached_1270238_384.pt',
                 map_location='cpu', weights_only=True)
stream = torch.cat([d.long() for d in raw]).cuda()
val_stream = stream[-200_000:]
torch.manual_seed(2024)
m2 = build('v53dms')
print('dms mask:', m2.decoder_layers[0].self_attention_linear.dm_gmask.tolist(),
      'n_gated', m2.decoder_layers[0].self_attention_linear.dm_n_gated)
ret = m2.load_state_dict(torch.load('/mnt/f/OpenASH2605/full_v53dms_80k/model.pt',
                                    map_location='cpu', weights_only=True), strict=False)
print('dms@80k load: missing=%d unexpected=%d' % (len(ret.missing_keys), len(ret.unexpected_keys)))
m2.eval().cuda()
n, tot_l, cnt = 128, 0.0, 0
starts = (torch.arange(0, n, device='cuda') * ((val_stream.shape[0] - 385) // n)).tolist()
with torch.no_grad():
    for s in range(0, n, 16):
        win = torch.stack([val_stream[i:i + 385] for i in starts[s:s + 16]]).cuda()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss, _ = m2.forward_loss(win[:, :-1], win[:, 1:], fused_ce=True)
        tot_l += loss.item() * win[:, :-1].numel()
        cnt += win[:, :-1].numel()
print('dms@80k quick val@384 (expect ~8.0-8.3): ppl %.2f' % math.exp(tot_l / cnt))
print('DMH SMOKE OK')
