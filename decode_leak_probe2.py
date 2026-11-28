# -*- coding: utf-8 -*-
"""decode_leak_probe2.py — 定位"延迟测量循环显存堆积"的触发条件（与 run() 式循环逐项对照）"""
import sys

import torch

sys.path.insert(0, r'/mnt/f/OpenASH2605')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from train_ab import build

DEV = 'cuda'
MB = 1e6
torch.manual_seed(2024)
m = build('v53dms')
m.load_state_dict(torch.load('/mnt/f/OpenASH2605/full_v53dms_80k/model.pt', map_location='cpu',
                             weights_only=True), strict=False)
m.eval().cuda()
AC = torch.autocast('cuda', dtype=torch.bfloat16)
g = torch.Generator(device=DEV).manual_seed(1)
ids = torch.randint(0, 23000, (1, 2128), device=DEV, generator=g)


def mem(tag):
    print('   %-44s allocated %7.1f MB' % (tag, torch.cuda.memory_allocated() / MB), flush=True)


@torch.no_grad()
def style_run(st, t0, n, sync_every):
    inc = []
    for t in range(t0, t0 + n):
        with AC:
            lg, st = m(ids[:, t:t + 1], state=st, pos_offset=t)
        inc.append(lg[0, 0].float())
        if sync_every and (t - t0) % sync_every == sync_every - 1:
            torch.cuda.synchronize()
    return st


@torch.no_grad()
def style_o1(st, t0, n):
    for t in range(t0, t0 + n):
        with AC:
            _, st = m(ids[:, t:t + 1], state=st, pos_offset=t)
    return st


with AC:
    _, st = m(ids[:, :128])
torch.cuda.synchronize()
mem('prefill 128 后')
with torch.no_grad():
    st = style_run(st, 128, 64, sync_every=16)
    mem('A) run 式(保留 logits + 每16步同步) 64 步')
    st = style_run(st, 192, 64, sync_every=0)
    mem('B) run 式(保留 logits, 不同步) 64 步')
    st = style_o1(st, 256, 64)
    mem('C) O(1) 式(丢 logits, 不同步) 64 步')
    torch.cuda.synchronize()
    mem('   C 之后 synchronize')
    st = style_o1(st, 320, 16)
    mem('D) O(1) 式再 16 步')
    with AC:
        _, st = m(ids[:, :64])
    mem('E) 做一次 64-token prefill 后')
    # O(1) 式但每步同步
    with AC:
        _, st = m(ids[:, :128])
    for t in range(128, 192):
        with AC:
            _, st = m(ids[:, t:t + 1], state=st, pos_offset=t)
        torch.cuda.synchronize()
    mem('F) O(1) 式但每步同步 64 步')
    # O(1) 式但保留 logits 到变量 lg（不 append）
    with AC:
        _, st = m(ids[:, :128])
    for t in range(128, 192):
        with AC:
            lg, st = m(ids[:, t:t + 1], state=st, pos_offset=t)
    mem('G) 绑定到 lg（不 append, 不同步）64 步')
print('PROBE2 DONE')
