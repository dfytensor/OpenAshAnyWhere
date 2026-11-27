# -*- coding: utf-8 -*-
"""mem_probe2.py — 小批次（每批 ≤2048 token）下的前向峰值；同一形状重复以确认不持续增长"""
import gc
import sys

import torch

torch.cuda.set_per_process_memory_fraction(0.10)
sys.path.insert(0, r'/mnt/f/OpenASH2605')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from train_ab import build

GB = 1e9


def mem(tag):
    torch.cuda.synchronize()
    print('%-28s allocated %.3f GB | peak %.3f GB' % (
        tag, torch.cuda.memory_allocated() / GB, torch.cuda.max_memory_allocated() / GB), flush=True)


torch.manual_seed(2024)
m = build('v53dml')
m.load_state_dict(torch.load('/mnt/e/OpenASH_loom/full_v53dml_80k/model.pt', map_location='cpu',
                             weights_only=True), strict=False)
m.eval().cuda()
mem('after load')
ok = True
for L, bs in ((64, 32), (256, 8), (512, 4), (1024, 2), (2048, 1)):
    for rep in (1, 2, 3):
        torch.cuda.reset_peak_memory_stats()
        ids = torch.randint(0, 23000, (bs, L), device='cuda')
        tgt = torch.randint(0, 23000, (bs, L), device='cuda')
        try:
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                h, st = m._hidden(ids)
                loss, _ = m.forward_loss(ids, tgt, fused_ce=True)
            del h, st, loss
        except torch.OutOfMemoryError:
            print('L=%d bs=%d rep=%d OOM' % (L, bs, rep), flush=True)
            ok = False
            break
        gc.collect()
        torch.cuda.empty_cache()
        print('L=%-4d bs=%-2d rep=%d  %.0f tok/batch' % (L, bs, rep, L * bs), end='  ', flush=True)
        mem('')
        del ids, tgt
print('ALL OK' if ok else 'FAILED', flush=True)
