# -*- coding: utf-8 -*-
"""decode_leak_probe.py — 逐 token 解码时进程显存是否随步数增长（真正的 O(1) 需要：不增长）"""
import sys

import torch

sys.path.insert(0, r'/mnt/f/OpenASH2605')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from train_ab import build

DEV = 'cuda'
MB = 1e6


def load(arch, path):
    torch.manual_seed(2024)
    m = build(arch)
    m.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=False)
    return m.eval().cuda()


def state_bytes(st):
    tot = [0]

    def rec(x):
        if torch.is_tensor(x):
            tot[0] += x.numel() * x.element_size()
        elif isinstance(x, (tuple, list)):
            for y in x:
                rec(y)
    rec(st)
    return tot[0]


@torch.no_grad()
def probe(m, tag, steps=60):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    AC = torch.autocast('cuda', dtype=torch.bfloat16)
    g = torch.Generator(device=DEV).manual_seed(1)
    ids = torch.randint(0, 23000, (1, 256), device=DEV, generator=g)
    with AC:
        _, st = m(ids)
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    print('%s: 模型+prefill 后 allocated %.1f MB | 携带状态 %.2f MB' % (tag, base / MB, state_bytes(st) / MB))
    for t in range(steps):
        tok = torch.randint(0, 23000, (1, 1), device=DEV, generator=g)
        with AC:
            _, st = m(tok, state=st, pos_offset=256 + t)
        if (t + 1) % 20 == 0:
            torch.cuda.synchronize()
            print('   解码 %2d 步后: allocated %.1f MB（较基线 %+.1f MB）| 状态 %.2f MB' % (
                t + 1, torch.cuda.memory_allocated() / MB, (torch.cuda.memory_allocated() - base) / MB,
                state_bytes(st) / MB))
    del st
    import gc
    gc.collect()
    torch.cuda.synchronize()
    print('   释放状态后: allocated %.1f MB' % (torch.cuda.memory_allocated() / MB))


probe(load('v53dml', '/mnt/e/OpenASH_loom/full_v53dml_80k/model.pt'), 'dml@80k')
probe(load('v53dms', '/mnt/f/OpenASH2605/full_v53dms_80k/model.pt'), 'dms@80k')
probe(load('v53k', '/mnt/f/OpenASH2605/full_v53kr_40k/model.pt'), 'kr@40k（无侧记忆）')
print('DECODE LEAK PROBE DONE')
