# -*- coding: utf-8 -*-
"""smoke_dms.py — v53dms 冒烟（门控参数/初始化 γ/前后向）+ dme@80k 旧 ckpt 路径回归"""
import math
import sys

import torch

sys.path.insert(0, r'/mnt/f/OpenASH2605')
from train_ab import build

torch.manual_seed(2024)
m = build('v53dms').cuda()
tot = sum(p.numel() for p in m.parameters())
print('v53dms params: %.3fM (GDN %.1f%%)' % (tot / 1e6, tot / 106.3e6 * 100))
a = m.decoder_layers[0].self_attention_linear
g0 = -torch.exp(a.dm_A_log.detach()) * torch.nn.functional.softplus(a.dm_dt_bias.detach())
print('L0 side-memory init gamma:', ' '.join('%.5f' % x for x in g0.exp().tolist()),
      '| half-life:', ' '.join('%.0f' % (math.log(0.5) / math.log(x)) for x in g0.exp().tolist()))
print('a_proj zero:', bool((a.dm_a_proj.weight == 0).all().item()))
x = torch.randint(0, 23000, (2, 128), device='cuda')
y = torch.randint(0, 23000, (2, 128), device='cuda')
with torch.autocast('cuda', dtype=torch.bfloat16):
    loss, _ = m.forward_loss(x, y, fused_ce=True)
loss.backward()
gg = [p.grad.abs().mean().item() for n, p in m.named_parameters() if 'dm_A_log' in n or 'dm_a_proj' in n]
print('loss %.3f | gate grads min %.2e max %.2e' % (loss.item(), min(gg), max(gg)))
del m
torch.cuda.empty_cache()

# 回归：dme@80k 在新代码下复现 8.20（dm_gate=False 路径必须逐位不变）
torch.manual_seed(2024)
raw = torch.load('/mnt/f/OpenASH2605/minimind_data/pretrain_cached_1270238_384.pt',
                 map_location='cpu', weights_only=True)
stream = torch.cat([d.long() for d in raw]).cuda()
val_stream = stream[-200_000:]
torch.manual_seed(2024)
m2 = build('v53dme')
ret = m2.load_state_dict(torch.load('/mnt/f/OpenASH2605/full_v53dme_80k/model.pt',
                                    map_location='cpu', weights_only=True), strict=False)
print('dme@80k load: missing=%d unexpected=%d' % (len(ret.missing_keys), len(ret.unexpected_keys)))
m2.eval().cuda()
n, tot_l, cnt = 64, 0.0, 0
starts = (torch.arange(0, n, device='cuda') * ((val_stream.shape[0] - 385) // n)).tolist()
with torch.no_grad():
    for s in range(0, n, 16):
        win = torch.stack([val_stream[i:i + 385] for i in starts[s:s + 16]]).cuda()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss, _ = m2.forward_loss(win[:, :-1], win[:, 1:], fused_ce=True)
        tot_l += loss.item() * win[:, :-1].numel()
        cnt += win[:, :-1].numel()
print('dme@80k quick val@384 (expect ~8.1-8.3): ppl %.2f' % math.exp(tot_l / cnt))
print('DMS SMOKE OK')
