# -*- coding: utf-8 -*-
"""posfix_eval.py — 零训练成本假设检验：L≥512 的长程劣势是否源于未训练的 pos_em（位置 385+）
变体：A 原样 / B pos≥384 钳到 384（复用最后训练位）/ C pos≥384 置零 / D 完全去位置编码"""
import math
import sys

import torch

sys.path.insert(0, r'/mnt/f/OpenASH2605')
from train_ab import build

DEV = 'cuda'
torch.manual_seed(2024)
raw = torch.load('/mnt/f/OpenASH2605/minimind_data/pretrain_cached_1270238_384.pt',
                 map_location='cpu', weights_only=True)
stream = torch.cat([d.long() for d in raw]).cuda()
val_stream = stream[-200_000:]

torch.manual_seed(2024)
m = build('v53dme')
m.load_state_dict(torch.load('/mnt/f/OpenASH2605/full_v53dme_80k/model.pt',
                             map_location='cpu', weights_only=True), strict=False)
m.eval().cuda()

W = m.pos_em.weight.detach()
n_tr = W[:385].norm(dim=1).mean().item()
n_un = W[385:].norm(dim=1).mean().item()
print('pos_em row-norm: trained(0-384) %.3f | untrained(385+) %.3f' % (n_tr, n_un))
em_norm = m.em(torch.tensor([100, 5000], device=DEV)).detach().norm(dim=1).mean().item()
print('token em row-norm (reference): %.3f' % em_norm)


@torch.no_grad()
def ppl_at(L, n=96, bs=8):
    stride = max(1, (val_stream.shape[0] - L - 1) // n)
    starts = (torch.arange(0, n, device=DEV) * stride).tolist()
    tot, cnt = 0.0, 0
    for s in range(0, n, bs):
        idx = starts[s:s + bs]
        win = torch.stack([val_stream[i:i + L + 1] for i in idx]).cuda()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss, _ = m.forward_loss(win[:, :-1], win[:, 1:], fused_ce=True)
        tot += loss.item() * win[:, :-1].numel()
        cnt += win[:, :-1].numel()
    return math.exp(tot / cnt)


orig_forward = m.pos_em.forward
variants = {
    'A as-is': lambda p: orig_forward(p),
    'B clamp>=384->384': lambda p: orig_forward(p.clamp(max=384)),
    'C zero>=384': lambda p: orig_forward(p) * (p < 385).float().view(-1, 1),
    'D no_pos': lambda p: orig_forward(p) * 0.0,
}
print()
print('variant'.ljust(22) + ''.join(('L=%d' % L).rjust(10) for L in (384, 512, 1024, 2048)))
for name, fn in variants.items():
    m.pos_em.forward = fn
    row = name.ljust(22)
    for L in (384, 512, 1024, 2048):
        row += ('%.2f' % ppl_at(L)).rjust(10)
    print(row)
m.pos_em.forward = orig_forward
