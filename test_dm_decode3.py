# -*- coding: utf-8 -*-
"""test_dm_decode3.py — 改动前 vs 改动后：逐 token 解码与全序列前向的一致性（同口径对照）
旧代码（git HEAD 版，解码路径无侧记忆）vs 新代码（侧记忆 O(1) 线程化）。
口径：autocast bf16（评测口径），T=512，prefill P=256，逐 token 解码 256 步。"""
import math
import sys

import torch

sys.path.insert(0, r'/mnt/f/OpenASH2605')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import train_ab
import open_ash_v3 as new_mod
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
AC = torch.autocast('cuda', dtype=torch.bfloat16)


def load(d, arch, use_old):
    train_ab.OpenASH_V3 = old_mod.OpenASH_V3 if use_old else new_mod.OpenASH_V3
    torch.manual_seed(2024)
    m = train_ab.build(arch)
    m.load_state_dict(torch.load('/mnt/f/OpenASH2605/%s/model.pt' % d, map_location='cpu',
                                 weights_only=True), strict=False)
    m.eval().cuda()
    return m


@torch.no_grad()
def run(m, ids, P):
    T = ids.shape[1] - 1
    tgt = ids[0, 1:]
    with AC:
        full, _ = m(ids[:, :-1])
    full = full.float()[0]
    with AC:
        _, st = m(ids[:, :P])
    inc = []
    for t in range(P, T):
        with AC:
            lg, st = m(ids[:, t:t + 1], state=st, pos_offset=t)
        inc.append(lg[0, 0].float())
    inc = torch.stack(inc, 0)
    A, Bf = full[P:], inc
    ce_a = torch.nn.functional.cross_entropy(A, tgt[P:], reduction='none')
    ce_b = torch.nn.functional.cross_entropy(Bf, tgt[P:], reduction='none')
    return dict(
        dmax=(A - Bf).abs().max().item(), dmean=(A - Bf).abs().mean().item(),
        agree=(A.argmax(-1) == Bf.argmax(-1)).float().mean().item() * 100,
        ppl_a=math.exp(ce_a.mean().item()), ppl_b=math.exp(ce_b.mean().item()),
        ce_dmax=ce_a.sub(ce_b).abs().max().item())


MODELS = [('dms@80k', 'full_v53dms_80k', 'v53dms'), ('dme@80k', 'full_v53dme_80k', 'v53dme'),
          ('dmh@80k', 'full_v53dmh_80k', 'v53dmh')]
for seed_off in (1000, 40000):
    ids = val[seed_off:seed_off + 513].view(1, -1)
    print('=== 窗口起点 %d（T=512, P=256）===' % seed_off)
    print('  %-6s %-6s | logits|Δ| mean  max | argmax一致 | ppl 全序列 vs 解码 | CE|Δ| max' % ('模型', '版本'))
    for tag, d, arch in MODELS:
        for label, use_old in (('改动前', True), ('改动后', False)):
            m = load(d, arch, use_old)
            r = run(m, ids, 256)
            print('  %-8s %-6s | %.5f  %.4f | %.2f%% | %.3f vs %.3f | %.4f' % (
                tag, label, r['dmean'], r['dmax'], r['agree'], r['ppl_a'], r['ppl_b'], r['ce_dmax']))
            del m
            torch.cuda.empty_cache()
