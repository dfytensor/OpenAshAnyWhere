# -*- coding: utf-8 -*-
"""dms_mech.py — ② 机制指纹：L=2048 逐位置曲线 dms vs dme@40k + 学到的侧记忆门控参数"""
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


@torch.no_grad()
def pos_curve(m, L, n=64, bs=8, nb=16):
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


curves = {}
for tag, path, arch in (('dms', '/mnt/f/OpenASH2605/full_v53dms_40k/model.pt', 'v53dms'),
                        ('dme', '/mnt/f/OpenASH2605/full_v53dme_40k/model.pt', 'v53dme')):
    torch.manual_seed(2024)
    m = build(arch)
    m.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=False)
    m.eval().cuda()
    curves[tag] = pos_curve(m, 2048)
    if tag == 'dms':
        print('=== 学到的侧记忆门控（逐层：静态 γ 基线 → 半衰期 tok；a_proj 范数）===')
        for L in range(7):
            a = m.decoder_layers[L].self_attention_linear
            g0 = (-torch.exp(a.dm_A_log.detach().float()) * torch.nn.functional.softplus(a.dm_dt_bias.detach().float())).exp()
            hls = [math.log(0.5) / math.log(x) if 0 < x < 1 else float('inf') for x in g0.tolist()]
            print('  L%d gamma %s | hl %s | a_proj_norm %.3f' % (
                L, ' '.join('%.4f' % x for x in g0.tolist()),
                ' '.join('%6.0f' % x for x in hls), a.dm_a_proj.weight.norm().item()))
    del m
    torch.cuda.empty_cache()

nb = 16
print()
print('=== L=2048 逐位置曲线（16 桶 × 128 tok）===')
print('pos-end :', ' '.join(('%5d' % ((i + 1) * 128)).rjust(6) for i in range(nb)))
print('dms     :', ' '.join('%6.3f' % x for x in curves['dms']))
print('dme     :', ' '.join('%6.3f' % x for x in curves['dme']))
diff = [a - b for a, b in zip(curves['dme'], curves['dms'])]
print('dme-dms :', ' '.join('%+6.3f' % x for x in diff))
second = sum(diff[8:]) / 8
print()
print('预注册机制指纹：后半段(pos>1024) dme−dms 均值 = %+.3f  → %s' % (
    second, '成立(>=+0.03)' if second >= 0.03 else ('无效(<=+0.01)' if second <= 0.01 else '弱')))
