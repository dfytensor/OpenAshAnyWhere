# -*- coding: utf-8 -*-
"""mem_recall.py — MEM 永久头对短距召回的影响（评估态，dme@80k）"""
import math
import sys

import torch

sys.path.insert(0, r'/mnt/f/OpenASH2605')
from train_ab import build

torch.manual_seed(2024)
raw = torch.load('/mnt/f/OpenASH2605/minimind_data/pretrain_cached_1270238_384.pt',
                 map_location='cpu', weights_only=True)
stream = torch.cat([d.long() for d in raw]).cuda()
exec(open('/mnt/f/OpenASH2605/recall_paired_base.py', encoding='utf-8').read().split('gaps = ')[0])

torch.manual_seed(2024)
m = build('v53dme')
m.load_state_dict(torch.load('/mnt/f/OpenASH2605/full_v53dme_80k/model.pt',
                             map_location='cpu', weights_only=True), strict=False)
m.eval().cuda()


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


print('variant'.ljust(10) + ''.join(('gap%d' % g).rjust(14) for g in (32, 64, 128, 256, 384)))
for mode in ('asis', 'init', '0.99', '0.999'):
    set_mem(mode)
    rr = recall_all(m, [32, 64, 128, 256, 384], seed=99)
    print(mode.ljust(10) + ''.join(('%6.2f %+0.2f' % rr[g]).rjust(14) for g in (32, 64, 128, 256, 384)))
