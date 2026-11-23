# -*- coding: utf-8 -*-
"""pos2048.py — L=2048 逐位置损失曲线：dme@80k vs GDN@160k（定位 2048 掉头的位置来源）"""
import sys

import torch

sys.path.insert(0, r'/mnt/f/OpenASH2605')
from train_ab import build
from baseline_fla import build_gdn, FlaLMWrapper

DEV = 'cuda'
torch.manual_seed(2024)
raw = torch.load('/mnt/f/OpenASH2605/minimind_data/pretrain_cached_1270238_384.pt',
                 map_location='cpu', weights_only=True)
stream = torch.cat([d.long() for d in raw]).cuda()
val_stream = stream[-200_000:]


@torch.no_grad()
def pos_curve(m, L, n=64, bs=4, nb=16, gdn=False):
    stride = max(1, (val_stream.shape[0] - L - 1) // n)
    starts = (torch.arange(0, n, device=DEV) * stride).tolist()
    buckets = [[] for _ in range(nb)]
    for s in range(0, n, bs):
        idx = starts[s:s + bs]
        win = torch.stack([val_stream[i:i + L + 1] for i in idx]).cuda()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            if gdn:
                logits = m.model(input_ids=win[:, :-1]).logits
            else:
                logits, _ = m(win[:, :-1])
        lp = torch.log_softmax(logits.float(), -1)
        ce = -lp.gather(-1, win[:, 1:].unsqueeze(-1)).squeeze(-1)
        blen = ce.shape[1] // nb
        for b in range(nb):
            lo, hi = b * blen, (b + 1) * blen if b < nb - 1 else ce.shape[1]
            buckets[b].append(ce[:, lo:hi].mean().item())
    return [sum(b) / len(b) for b in buckets]


torch.manual_seed(2024)
me = build('v53dme')
me.load_state_dict(torch.load('/mnt/f/OpenASH2605/full_v53dme_80k/model.pt',
                              map_location='cpu', weights_only=True), strict=False)
me.eval().cuda()
torch.manual_seed(2024)
g = FlaLMWrapper(build_gdn())
g.load_state_dict(torch.load('/mnt/f/OpenASH2605/full_gdn_160k/model.pt',
                             map_location='cpu', weights_only=True))
g.eval().cuda()

for L in (1024, 2048):
    nb = 16
    c_me = pos_curve(me, L, nb=nb)
    c_g = pos_curve(g, L, nb=nb, gdn=True)
    print('=== L=%d (%d buckets of %d tokens) ===' % (L, nb, L // nb))
    print('  bucket   :', ' '.join(('%5d' % (i * L // nb + L // nb)).rjust(6) for i in range(nb)))
    print('  dme@80k  :', ' '.join('%6.3f' % x for x in c_me))
    print('  GDN@160k :', ' '.join('%6.3f' % x for x in c_g))
    print('  dme-GDN  :', ' '.join('%+6.3f' % (a - b) for a, b in zip(c_me, c_g)))
    print()
