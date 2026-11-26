# -*- coding: utf-8 -*-
"""mem_extrap.py — 长度外推失败的 MEM 永久头假设检验（评估态，零训练）
变体：as-is / MEM 头改用初始化衰减 / 强制 MEM 衰减 0.99 / 0.999 / 0.9999"""
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


@torch.no_grad()
def pos_curve(m, L, n=64, bs=4, nb=16):
    stride = max(1, (val_stream.shape[0] - L - 1) // n)
    starts = (torch.arange(0, n, device=DEV) * stride).tolist()
    buckets = [[] for _ in range(nb)]
    for s in range(0, n, bs):
        idx = starts[s:s + bs]
        win = torch.stack([val_stream[i:i + L + 1] for i in idx]).cuda()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits, _ = m(win[:, :-1])
        lp = torch.log_softmax(logits.float(), -1)
        ce = -lp.gather(-1, win[:, 1:].unsqueeze(-1)).squeeze(-1)
        blen = ce.shape[1] // nb
        for b in range(nb):
            lo, hi = b * blen, (b + 1) * blen if b < nb - 1 else ce.shape[1]
            buckets[b].append(ce[:, lo:hi].mean().item())
    return [sum(b) / len(b) for b in buckets]


torch.manual_seed(2024)
m = build('v53dme')
sd = torch.load('/mnt/f/OpenASH2605/full_v53dme_80k/model.pt', map_location='cpu', weights_only=True)
m.load_state_dict(sd, strict=False)
m.eval().cuda()

import math
print('MEM heads 0:3 kv_loga -> gamma (init spread, never trained):')
for L in range(2):
    la = m.decoder_layers[L].self_attention_linear.kv_loga.detach()
    g = (-torch.nn.functional.softplus(la)).exp()
    print('  L%d' % L, ' '.join('%.4f' % x for x in g[:4].tolist()))


def set_mem(mode):
    for layer in m.decoder_layers:
        a = layer.self_attention_linear
        if mode == 'asis':
            a.mem_heads = 3
        else:
            a.mem_heads = 0
            if mode != 'init':
                g = float(mode)
                la = math.log(math.expm1(-math.log(g)))
                with torch.no_grad():
                    a.kv_loga[:3] = la


for mode in ('asis', 'init', '0.99', '0.999', '0.9999'):
    set_mem(mode)
    c = pos_curve(m, 2048)
    late = sum(c[8:]) / 8
    early = sum(c[:3]) / 3
    print('MEM=%-7s early(pos<384) %.3f | late(pos>1024) %.3f | curve tail: %s' % (
        mode, early, late, ' '.join('%.3f' % x for x in c[10:])))
