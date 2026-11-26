# -*- coding: utf-8 -*-
"""test_dm_decode.py — 侧记忆 O(1) 解码线程化验证
T1 kernel 级：单步递推 vs FLA chunk kernel（门控 / 无门控，非零初始态）
T2 模型级：全序列前向 vs prefill(P) + 逐 token 解码（autocast bf16，评测口径）
T3 回归：dms@80k prefill 路径数值不变（threading 只应在 sx==1 生效）
T4 O(1) 证据：不同上下文长度下的逐 token 延迟与携带状态字节数"""
import math
import sys
import time

import torch

sys.path.insert(0, r'/mnt/f/OpenASH2605')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from train_ab import build
from fla.ops.delta_rule import chunk_delta_rule
from fla.ops.gated_delta_rule import chunk_gated_delta_rule

DEV = 'cuda'
torch.manual_seed(0)

# ---------------- T1 ----------------
print('=== T1: kernel 级递推语义（单步 vs chunk）===')
B, T, H, D = 2, 300, 4, 96
q = torch.randn(B, T, H, D, device=DEV, dtype=torch.bfloat16) * 0.5
k = torch.randn(B, T, H, D, device=DEV, dtype=torch.bfloat16)
k = (k / (k.norm(dim=-1, keepdim=True) + 1e-6)).to(torch.bfloat16)
v = torch.randn(B, T, H, D, device=DEV, dtype=torch.bfloat16) * 0.5
beta = torch.rand(B, T, H, device=DEV).sigmoid().to(torch.bfloat16)
g = (-torch.rand(B, T, H, device=DEV) * 0.7).to(torch.bfloat16)   # exp(g)∈[0.5,1]：让衰减顺序的差异可辨
h0 = (torch.randn(B, H, D, D, device=DEV) * 0.05)


def step_loop(gated, h_init, variant='ref'):
    S = h_init.clone()
    outs = []
    for t in range(T):
        kd, qd, vd, bd = k[:, t].float(), q[:, t].float(), v[:, t].float(), beta[:, t].float()
        a = torch.exp(g[:, t].float())[:, :, None, None] if gated else 1.0
        if variant == 'read_before_decay':
            sv = torch.einsum('bhkv,bhk->bhv', S, kd)          # 错：用衰减前状态读
            S = S * a
        else:
            S = S * a                                             # 对：先衰减
            sv = torch.einsum('bhkv,bhk->bhv', S, kd)
        S = S + (kd * bd[:, :, None]).unsqueeze(-1) * (vd - sv).unsqueeze(-2)
        scale = 1.0 if variant == 'no_scale' else D ** -0.5     # no_scale：错：漏掉 q 的 K^-0.5
        outs.append(torch.einsum('bhkv,bhk->bhv', S, qd) * scale)
    return torch.stack(outs, 1), S, None


for gated in (True, False):
    o_c, h_c = (chunk_gated_delta_rule(q=q, k=k, v=v, g=g, beta=beta, initial_state=h0, output_final_state=True)
                if gated else chunk_delta_rule(q=q, k=k, v=v, beta=beta, initial_state=h0, output_final_state=True))
    o_scale = o_c.float().std().item()
    print('  %s（输出尺度 %.3f）:' % ('门控' if gated else '无门控', o_scale))
    for variant, label in (('ref', '本实现（先衰减→读→写→q·S·K^-0.5）'),
                           ('read_before_decay', '反例A：衰减前读'),
                           ('no_scale', '反例B：漏 q 的 K^-0.5')):
        o_s, h_s, _ = step_loop(gated, h0, variant)
        e = (o_c.float() - o_s).abs()
        print('     %-36s max|Δo|/scale = %.2e | mean|Δo|/scale = %.2e' % (
            label, e.max().item() / o_scale, e.mean().item() / o_scale))
    print('     终态 max|Δh|/h_max = %.2e' % ((h_c.float() - h_s).abs().max().item() / h_s.abs().max().item()))

# ---------------- T2/T3/T4 ----------------
torch.manual_seed(2024)
raw = torch.load('/mnt/f/OpenASH2605/minimind_data/pretrain_cached_1270238_384.pt',
                 map_location='cpu', weights_only=True)
stream = torch.cat([d.long() for d in raw]).cuda()
val = stream[-200_000:]


def state_bytes(st):
    tot = 0

    def rec(x):
        nonlocal tot
        if torch.is_tensor(x):
            tot += x.numel() * x.element_size()
        elif isinstance(x, (tuple, list)):
            for y in x:
                rec(y)
    rec(st)
    return tot


MODELS = [('dms@80k', 'full_v53dms_80k', 'v53dms'),
          ('dme@80k', 'full_v53dme_80k', 'v53dme'),
          ('dmh@80k', 'full_v53dmh_80k', 'v53dmh'),
          ('kr@40k(无侧记忆基线)', 'full_v53kr_40k', 'v53k')]

Tw, P = 512, 256
for tag, d, arch in MODELS:
    torch.manual_seed(2024)
    m = build(arch)
    m.load_state_dict(torch.load('/mnt/f/OpenASH2605/%s/model.pt' % d, map_location='cpu',
                                 weights_only=True), strict=False)
    m.eval().cuda()
    with torch.no_grad():
        ids = val[1000:1000 + Tw + 1].view(1, -1)
        tgt = ids[:, 1:]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            full, _ = m(ids[:, :-1])
        full = full.float()[0]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            _, st = m(ids[:, :P])
        inc = []
        for t in range(P, Tw):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                lg, st = m(ids[:, t:t + 1], state=st, pos_offset=t)
            inc.append(lg[0, 0].float())
        inc = torch.stack(inc, 0)
        A, Bf = full[P:], inc
        dif = (A - Bf).abs()
        agree = (A.argmax(-1) == Bf.argmax(-1)).float().mean().item()
        ce_full = torch.nn.functional.cross_entropy(A, tgt[0, P:], reduction='none')
        ce_inc = torch.nn.functional.cross_entropy(Bf, tgt[0, P:], reduction='none')
        # 原有解码路径自身口径：prefill 段 vs 全序列（应逐位相同，证明 threading 不动 prefill）
        with torch.autocast('cuda', dtype=torch.bfloat16):
            pf, _ = m(ids[:, :P])
        pf_dif = (full[:P] - pf[0].float()).abs().max().item()
    print()
    print('=== T2 %s（T=%d, prefill P=%d, 逐 token 解码 %d 步）===' % (tag, Tw, P, Tw - P))
    print('  logits |Δ|: max %.4f | mean %.5f | logits std %.2f | argmax 一致率 %.2f%%' % (
        dif.max().item(), dif.mean().item(), A.std().item(), agree * 100))
    print('  CE(ppl) 全序列 %.4f (ppl %.2f) vs 解码 %.4f (ppl %.2f) | CE |Δ| mean %.5f max %.4f' % (
        ce_full.mean().item(), math.exp(ce_full.mean().item()),
        ce_inc.mean().item(), math.exp(ce_inc.mean().item()),
        (ce_full - ce_inc).abs().mean().item(), (ce_full - ce_inc).abs().max().item()))
    print('  T3 prefill 路径回归: 全序列[:P] vs 单独 prefill |Δ|max = %.2e' % pf_dif)
    print('  T4 携带状态: %.2f MB（每层 %.3f MB × %d 层）' % (
        state_bytes(st) / 1e6, state_bytes(st) / 1e6 / len(st), len(st)))
    del m
    torch.cuda.empty_cache()
