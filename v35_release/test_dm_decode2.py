# -*- coding: utf-8 -*-
"""test_dm_decode2.py — 补测
T3' 改动前代码（git HEAD 版 open_ash_v3）vs 改动后代码：全序列前向 logits 逐位对比（prefill 不变性）
T2b fp32 模式（无 autocast）增量一致性（语义级）
T4b O(1) 证据：不同上下文长度下的逐 token 解码延迟 / 吞吐 / 状态字节；对照全序列前向耗时
M  kernel 输出对 output_final_state=True 不敏感（逐位）"""
import math
import sys
import time

import torch

sys.path.insert(0, r'/mnt/f/OpenASH2605')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import train_ab
import os
# —— 改动前代码模块（解码线程化之前的版本 = 提交 77b414b）：缺失则用 git 原始字节重建 ——
# （勿用 PowerShell 的 > 重定向生成：会写出 UTF-16 null 字节）
_OLD = 'open_ash_v3_pre_decode.py'
if not os.path.exists(_OLD):
    import subprocess
    with open(_OLD, 'wb') as f:
        subprocess.run(['git', 'show', '77b414b:open_ash_v3.py'], stdout=f, check=True)
import open_ash_v3_pre_decode as old_mod

DEV = 'cuda'
torch.manual_seed(2024)
raw = torch.load('/mnt/f/OpenASH2605/minimind_data/pretrain_cached_1270238_384.pt',
                 map_location='cpu', weights_only=True)
stream = torch.cat([d.long() for d in raw]).cuda()
val = stream[-200_000:]


def load(d, arch, use_old):
    train_ab.OpenASH_V3 = old_mod.OpenASH_V3 if use_old else new_mod.OpenASH_V3
    torch.manual_seed(2024)
    m = train_ab.build(arch)
    m.load_state_dict(torch.load('/mnt/f/OpenASH2605/%s/model.pt' % d, map_location='cpu',
                                 weights_only=True), strict=False)
    m.eval().cuda()
    return m


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


import open_ash_v3 as new_mod
AC = torch.autocast('cuda', dtype=torch.bfloat16)

# ---------------- M ----------------
print('=== M: kernel 输出对 output_final_state=True 不敏感 ===')
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
q = torch.randn(1, 200, 4, 96, device=DEV, dtype=torch.bfloat16)
k = (torch.randn(1, 200, 4, 96, device=DEV, dtype=torch.bfloat16))
k = (k / (k.norm(dim=-1, keepdim=True) + 1e-6)).to(torch.bfloat16)
v = torch.randn(1, 200, 4, 96, device=DEV, dtype=torch.bfloat16)
g = (-torch.rand(1, 200, 4, device=DEV) * 0.3).to(torch.bfloat16)
beta = torch.rand(1, 200, 4, device=DEV).to(torch.bfloat16)
o1, _ = chunk_gated_delta_rule(q=q, k=k, v=v, g=g, beta=beta, initial_state=None, output_final_state=False)
o2, h2 = chunk_gated_delta_rule(q=q, k=k, v=v, g=g, beta=beta, initial_state=None, output_final_state=True)
print('  |Δo| max = %g | final_state shape %s dtype %s | |h| max %.4f' % (
    (o1.float() - o2.float()).abs().max().item(), tuple(h2.shape), h2.dtype, h2.float().abs().max().item()))
h0 = torch.randn(1, 4, 96, 96, device=DEV)
o3, h3 = chunk_gated_delta_rule(q=q, k=k, v=v, g=g, beta=beta, initial_state=h0, output_final_state=True)
print('  fp32 initial_state accepted: o shape %s, h3 dtype %s' % (tuple(o3.shape), h3.dtype))

# ---------------- T3' ----------------
print()
print("=== T3': 改动前代码 vs 改动后代码：全序列前向（prefill 路径）===")
ids = val[1000:1512].view(1, -1)
for tag, d, arch in (('dms@80k', 'full_v53dms_80k', 'v53dms'), ('dme@80k', 'full_v53dme_80k', 'v53dme'),
                     ('dmh@80k', 'full_v53dmh_80k', 'v53dmh'), ('kr@40k', 'full_v53kr_40k', 'v53k')):
    mo = load(d, arch, True)
    mn = load(d, arch, False)
    with torch.no_grad():
        with AC:
            lo, _ = mo(ids[:, :-1])
        with AC:
            ln, st = mn(ids[:, :-1])
    dl = (lo.float() - ln.float()).abs().max().item()
    print('  %-8s logits |Δ|max = %g  → %s' % (tag, dl, '逐位相同 ✓' if dl == 0 else '有差异'))
    del mo, mn
    torch.cuda.empty_cache()

# ---------------- T2b ----------------
print()
print('=== T2b: fp32 模式（无 autocast）增量一致性 ===')


def inc_compare(m, ids, P, ac):
    T = ids.shape[1] - 1
    with torch.no_grad():
        with ac:
            full, _ = m(ids[:, :-1])
        full = full.float()[0]
        with ac:
            _, st = m(ids[:, :P])
        inc = []
        for t in range(P, T):
            with ac:
                lg, st = m(ids[:, t:t + 1], state=st, pos_offset=t)
            inc.append(lg[0, 0].float())
        inc = torch.stack(inc, 0)
    A, Bf = full[P:], inc
    dif = (A - Bf).abs()
    return dif.max().item(), dif.mean().item(), (A.argmax(-1) == Bf.argmax(-1)).float().mean().item()


null_ac = torch.autocast('cuda', dtype=torch.bfloat16, enabled=False)
for tag, d, arch in (('dms@80k', 'full_v53dms_80k', 'v53dms'), ('kr@40k', 'full_v53kr_40k', 'v53k')):
    m = load(d, arch, False)
    try:
        mx, mn_, ag = inc_compare(m, ids, 256, null_ac)
        print('  %-8s fp32: logits |Δ| max %.3e mean %.3e | argmax 一致 %.2f%%' % (tag, mx, mn_, ag * 100))
    except Exception as e:
        print('  %-8s fp32 模式不可用: %s' % (tag, repr(e)[:140]))
    del m
    torch.cuda.empty_cache()

# ---------------- T4b ----------------
print()
print('=== T4b: O(1) 证据——逐 token 解码延迟与状态 vs 上下文长度 ===')


def bench(m, tag):
    long_ids = val[5000:5000 + 2048 + 80].view(1, -1)
    print('  %s:' % tag)
    for T in (256, 1024, 2048):
        with torch.no_grad():
            with AC:
                m(long_ids[:, :T])
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(5):
                with AC:
                    m(long_ids[:, :T])
            torch.cuda.synchronize()
            tf = (time.time() - t0) / 5
        print('     全序列前向 T=%-4d %.1f ms（摊薄 %.2f ms/token；batch 1 下被启动开销主导，总耗时近似不随 T 变）' % (T, tf * 1e3, tf * 1e3 / T))
    for C in (128, 1024, 2048):
        with torch.no_grad():
            with AC:
                _, st = m(long_ids[:, :C])
            torch.cuda.synchronize()
            t0 = time.time()
            for t in range(C, C + 64):
                with AC:
                    _, st = m(long_ids[:, t:t + 1], state=st, pos_offset=t)
            torch.cuda.synchronize()
            dt = (time.time() - t0) / 64
        print('     解码 @ctx=%-4d %.2f ms/token（%.0f tok/s）| 状态 %.2f MB' % (
            C, dt * 1e3, 1 / dt, state_bytes(st) / 1e6))


for tag, d, arch in (('dms@80k（旗舰）', 'full_v53dms_80k', 'v53dms'), ('kr@40k（无侧记忆）', 'full_v53kr_40k', 'v53k')):
    m = load(d, arch, False)
    bench(m, tag)
    del m
    torch.cuda.empty_cache()
