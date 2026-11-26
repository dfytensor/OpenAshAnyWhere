# -*- coding: utf-8 -*-
"""gdn_gprobe.py — 实测 GDN 有效衰减（逐层逐头 mean gamma）vs 我方 0.83/3.7tok"""
import math
import sys

import torch

sys.path.insert(0, r'F:\OpenASH2605')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from baseline_fla import build_gdn, FlaLMWrapper

torch.manual_seed(2024)
raw = torch.load('/mnt/f/OpenASH2605/minimind_data/pretrain_cached_1270238_384.pt',
                 map_location='cpu', weights_only=True)
stream = torch.cat([d.long() for d in raw]).cuda()
val_stream = stream[-200_000:]

w = FlaLMWrapper(build_gdn())
w.load_state_dict(torch.load('/mnt/f/OpenASH2605/full_gdn_80k_fixed/model.pt',
                             map_location='cpu', weights_only=True))
w.eval().cuda()

inner = w.model.model
gats = {}


def mk_hook(i):
    def pre_hook(mod, args, kwargs):
        x = args[0] if args else kwargs.get('hidden_states')
        A_log = mod.A_log.float()
        dt = mod.dt_bias.float()
        g = -torch.exp(A_log) * torch.nn.functional.softplus(
            torch.nn.functional.linear(x.float(), mod.a_proj.weight.float()) + dt)
        gats.setdefault(i, []).append(g.exp().mean().item())
    return pre_hook


hooks = []
for i, layer in enumerate(inner.layers):
    hooks.append(layer.attn.register_forward_pre_hook(mk_hook(i), with_kwargs=True))

with torch.no_grad():
    for s in range(0, 32, 8):
        win = torch.stack([val_stream[i:i + 385] for i in
                           (torch.arange(0, 32) * ((val_stream.shape[0] - 386) // 32)).cuda()[s:s + 8]])
        w.model(input_ids=win)

for i in sorted(gats):
    g = sum(gats[i]) / len(gats[i])
    h = math.log(0.5) / math.log(g) if 0 < g < 1 else float('inf')
    print('GDN L%d effective gamma %.4f -> half-life %.1f tok' % (i, g, h))
