# -*- coding: utf-8 -*-
"""test_loom_decode.py — dml@80k / dmq@80k 训练后权重的解码一致性（预注册判据 6）+ dml@80k 的 O(1) 证据
口径与 test_dm_decode.py 相同：autocast bf16，T=512，prefill 256 + 逐 token 解码 256 步。
O(1) 证据三项：逐 token 延迟、携带状态字节数、进程显存随解码步数不增长。
教训（2026-11-28）：量延迟的循环曾漏写 no_grad → 每步构建自动求导图且新状态依赖旧状态 → 64 步涨 9GB。
      模型解码路径本身无泄漏（decode_leak_probe：60 步显存逐位不变）。"""
import math
import sys
import time

import torch

torch.cuda.set_per_process_memory_fraction(0.30)
sys.path.insert(0, r'/mnt/f/OpenASH2605')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from train_ab import build
from ckpt_guard import assert_zip_integrity

DEV = 'cuda'
torch.manual_seed(2024)
DATA = '/mnt/f/OpenASH2605/minimind_data/pretrain_cached_1270238_384.pt'
assert_zip_integrity(DATA, 'data cache')
raw = torch.load(DATA, map_location='cpu', weights_only=True)
stream = torch.cat([d.long() for d in raw]).cuda()
val = stream[-200_000:]
del raw
AC = torch.autocast('cuda', dtype=torch.bfloat16)


def load(arch, path):
    assert_zip_integrity(path, path.split('/')[-2])
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
def run(m, ids, P):
    T = ids.shape[1] - 1
    tgt = ids[0, 1:]
    with AC:
        full_raw, _ = m(ids[:, :-1])
    full = full_raw.float()[0]
    del full_raw
    with AC:
        _, st = m(ids[:, :P])
    m0 = torch.cuda.memory_allocated()
    inc = []
    for t in range(P, T):
        with AC:
            lg, st = m(ids[:, t:t + 1], state=st, pos_offset=t)
        inc.append(lg[0, 0].float())
    torch.cuda.synchronize()
    m1 = torch.cuda.memory_allocated()
    inc = torch.stack(inc, 0)
    A, B = full[P:], inc
    ce_a = torch.nn.functional.cross_entropy(A, tgt[P:], reduction='none')
    ce_b = torch.nn.functional.cross_entropy(B, tgt[P:], reduction='none')
    return (A - B).abs().mean().item(), (A - B).abs().max().item(), A.std().item(), \
        (A.argmax(-1) == B.argmax(-1)).float().mean().item() * 100, \
        math.exp(ce_a.mean().item()), math.exp(ce_b.mean().item()), \
        (m1 - m0) / 1e6 - (T - P) * 23004 * 4 / 1e6     # 扣除存下来的 logits


@torch.no_grad()
def o1_evidence(m, long_ids):
    print('  O(1) 证据（batch 1，eager，no_grad）：')
    for C in (128, 1024, 2048):
        torch.cuda.synchronize()
        with AC:
            _, st = m(long_ids[:, :C])
        for _ in range(8):                      # 预热
            with AC:
                _, st = m(long_ids[:, C:C + 1], state=st, pos_offset=C)
        torch.cuda.synchronize()
        a0 = torch.cuda.memory_allocated()
        t0 = time.time()
        for t in range(C + 8, C + 72):
            with AC:
                _, st = m(long_ids[:, t:t + 1], state=st, pos_offset=t)
        torch.cuda.synchronize()
        dt = (time.time() - t0) / 64
        a1 = torch.cuda.memory_allocated()
        print('     @ctx=%-4d %.1f ms/token | 状态 %.2f MB | 64 步解码进程显存变化 %+.1f MB' % (
            C, dt * 1e3, state_bytes(st) / 1e6, (a1 - a0) / 1e6), flush=True)
    # 更长的解码：2048 起再解 256 步，显存应保持不变
    with AC:
        _, st = m(long_ids[:, :2048])
    torch.cuda.synchronize()
    a0 = torch.cuda.memory_allocated()
    for t in range(2048, 2048 + 72):
        with AC:
            _, st = m(torch.randint(0, 23000, (1, 1), device=DEV), state=st, pos_offset=t)
    torch.cuda.synchronize()
    print('     @ctx=2048 连续解码 72 步后进程显存变化 %+.1f MB | 状态 %.2f MB' % (
        (torch.cuda.memory_allocated() - a0) / 1e6, state_bytes(st) / 1e6), flush=True)


MODELS = [('dml@80k', 'v53dml', '/mnt/e/OpenASH_loom/full_v53dml_80k/model.pt'),
          ('dmq@80k', 'v53dmq', '/mnt/e/OpenASH_loom/full_v53dmq_80k/model.pt')]
long_ids_all = val[5000:5000 + 2048 + 80].view(1, -1)
print('=== 6) 解码一致性（全序列前向 vs prefill 256 + 逐 token 256 步；autocast bf16）===')
for tag, arch, path in MODELS:
    m = load(arch, path)
    for off in (1000, 40000):
        ids = val[off:off + 513].view(1, -1)
        dm, dx, sd, ag, pa, pb, dmem = run(m, ids, 256)
        print('  %-8s 窗口@%-5d logits|Δ| mean %.4f（logits std 的 %.1f%%）max %.3f（%.0f%%）| argmax 一致 %.2f%% | ppl 全序列 %.3f vs 解码 %.3f | 256 步显存变化 %+.1f MB' % (
            tag, off, dm, dm / sd * 100, dx, dx / sd * 100, ag, pa, pb, dmem), flush=True)
    o1_evidence(m, long_ids_all)
    del m
    torch.cuda.empty_cache()
print('LOOM DECODE TEST DONE')
