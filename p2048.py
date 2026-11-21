# -*- coding: utf-8 -*-
"""p2048.py — 补测 L=2048（bench A 段 OOM 时的单行补测）"""
import sys
import torch

sys.path.insert(0, r'/mnt/f/OpenASH2605')
from train_ab import build

DEV = 'cuda'
DATA = '/mnt/f/OpenASH2605/minimind_data/pretrain_cached_1270238_384.pt'
torch.manual_seed(2024)
raw = torch.load(DATA, map_location='cpu', weights_only=True)
stream = torch.cat([d.long() for d in raw]).cuda()
val_stream = stream[-200_000:]

CK = [('full_v53dme_40k', 'v53dme'), ('full_v53f1_40k', 'v53f1'),
      ('full_v53dm_v2_40k', 'v53dm'), ('full_gdn_80k_fixed', None)]

for key, arch in CK:
    torch.manual_seed(2024)
    if arch is None:
        from baseline_fla import build_gdn, FlaLMWrapper
        m = FlaLMWrapper(build_gdn())
        m.load_state_dict(torch.load('/mnt/f/OpenASH2605/%s/model.pt' % key,
                                     map_location='cpu', weights_only=True))
        m.eval()
        m = m.cuda()
    else:
        m = build(arch)
        m.load_state_dict(torch.load('/mnt/f/OpenASH2605/%s/model.pt' % key,
                                     map_location='cpu', weights_only=True), strict=False)
        m.eval()
        m = m.cuda()
    L = 2048
    n = min(96, (val_stream.shape[0] - 1) // (L + 1))
    stride = max(1, (val_stream.shape[0] - L - 1) // n)
    starts = (torch.arange(0, n, device=DEV) * stride).tolist()
    tot, cnt = 0.0, 0
    with torch.no_grad():
        for s in range(0, n, 8):
            idx = starts[s:s + 8]
            win = torch.stack([val_stream[i:i + L + 1] for i in idx]).cuda()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss, _ = m.forward_loss(win[:, :-1], win[:, 1:], fused_ce=True)
            tot += loss.item() * win[:, :-1].numel()
            cnt += win[:, :-1].numel()
    import math
    print('%-22s L=2048 ppl %.2f' % (key, math.exp(tot / cnt)))
    del m
    torch.cuda.empty_cache()
