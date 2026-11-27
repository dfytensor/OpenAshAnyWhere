# -*- coding: utf-8 -*-
"""mem_probe.py — 轻量评测的显存占用定位（同样 10% 硬上限，超限自己 OOM）"""
import gc
import sys
import time

import torch

torch.cuda.set_per_process_memory_fraction(0.10)
sys.path.insert(0, r'/mnt/f/OpenASH2605')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from train_ab import build

GB = 1e9


def mem(tag):
    torch.cuda.synchronize()
    print('%-34s allocated %.3f GB | reserved %.3f GB | peak %.3f GB' % (
        tag, torch.cuda.memory_allocated() / GB, torch.cuda.memory_reserved() / GB,
        torch.cuda.max_memory_allocated() / GB), flush=True)


mem('start')
torch.manual_seed(2024)
m = build('v53dml')
mem('after build (cpu)')
sd = torch.load('/mnt/e/OpenASH_loom/full_v53dml_80k/model.pt', map_location='cpu', weights_only=True)
m.load_state_dict(sd, strict=False)
del sd
gc.collect()
m.eval().cuda()
mem('after load+cuda')
print('params: %.3f GB' % (sum(p.numel() for p in m.parameters()) * 4 / GB), flush=True)

for L in (64, 256, 512, 1024, 2048):
    bs = 16 if L <= 512 else (8 if L <= 1024 else 4)
    ids = torch.randint(0, 23000, (bs, L), device='cuda')
    tgt = torch.randint(0, 23000, (bs, L), device='cuda')
    mem('L=%d bs=%d before' % (L, bs))
    with torch.no_grad():
        with torch.autocast('cuda', dtype=torch.bfloat16):
            h, st = m._hidden(ids)
            mem('L=%d bs=%d after _hidden' % (L, bs))
            del h, st
            gc.collect()
            try:
                loss, _ = m.forward_loss(ids, tgt, fused_ce=True)
                torch.cuda.synchronize()
                mem('L=%d bs=%d after forward_loss' % (L, bs))
                del loss
            except torch.OutOfMemoryError as e:
                print('L=%d bs=%d OOM in forward_loss' % (L, bs), flush=True)
                torch.cuda.empty_cache()
                break
    gc.collect()
    torch.cuda.empty_cache()
    mem('L=%d bs=%d after cleanup' % (L, bs))
print('MEM PROBE DONE', flush=True)
