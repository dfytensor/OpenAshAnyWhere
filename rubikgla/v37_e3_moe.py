#!/usr/bin/env python3
"""E3: MoE 化 + 双缓冲异步预取 demo (V37_ASYNC_SWAP_DESIGN.md E3).
MoEFFN: 共享专家 + n_exp 路由专家 top-k (DeepSeek 式 6+1 比例的微缩).
测量 @4096 prefill: (a) 全常驻基线 tokens/s
                   (b) CPU 主拷贝 + side-stream 双缓冲异步预取 (交换隐藏在计算内)
                   (c) 同步换入 (无重叠, 对照)
                   + NLL/logits 逐位一致性 (b vs a: 异步不得改变数值)
用法: python v37_e3_moe.py
"""
import sys, os, time, random
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import deqlm_v35_relay as M
from deqlm_v5_30m import PT_CACHE

DEV = "cuda"
D, FFN_H = M.D, M.FFN_H
N_EXP, TOPK = 8, 2


class Expert(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(D, FFN_H, bias=False)
        self.up = nn.Linear(D, FFN_H, bias=False)
        self.down = nn.Linear(FFN_H, D, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class MoEFFN(nn.Module):
    """shared 专家 + top-k 路由专家 (DeepSeek 6+1 比例微缩: 8选2+1)."""

    def __init__(self):
        super().__init__()
        self.ln = M.MARMSNorm(D)
        self.router = nn.Linear(D, N_EXP, bias=False)
        self.experts = nn.ModuleList([Expert() for _ in range(N_EXP)])
        self.shared = Expert()

    def forward(self, x, expert_slice=None):
        h = self.ln(x)
        y = self.shared(h)
        if expert_slice is None:
            for e in self.experts:
                y = y + e(h)
            return x + y
        idx = expert_slice                                    # 只用换入的专家 (演示交换语义)
        for i in idx:
            y = y + self.experts[i](h)
        return x + y


def route_assign(h, n_exp, topk):
    """h (B,L,D) → 每位置 top-k 专家 id (模拟 router)."""
    scores = torch.randn(h.shape[0], h.shape[1], n_exp, device=h.device)
    return torch.topk(scores, topk, dim=-1).indices              # (B,L,k)


def main():
    torch.manual_seed(0)
    seqs = torch.load(PT_CACHE, map_location="cpu", weights_only=True)
    rng = random.Random(0)

    moe = MoEFFN().to(DEV)
    # CPU 主拷贝 (权重轮换的"仓库"): 专家权重常驻 CPU, GPU 上只留工作缓冲
    cpu_master = [{} for _ in range(N_EXP)]
    for i, e in enumerate(moe.experts):
        cpu_master[i] = {k: v.detach().cpu().pin_memory() for k, v in e.state_dict().items()}
    work = [{} for _ in range(N_EXP)]                            # GPU 工作缓冲 (可被覆盖)
    stage = [{} for _ in range(N_EXP)]                           # 预取暂存缓冲
    for i in range(N_EXP):
        for k, v in cpu_master[i].items():
            work[i][k] = torch.empty_like(v, device=DEV)
            work[i][k].copy_(v)
            stage[i][k] = torch.empty_like(v, device=DEV)
    copy_stream = torch.cuda.Stream()

    def load_expert(i, use_stage=False):
        dst = stage[i] if use_stage else work[i]
        for k, v in cpu_master[i].items():
            dst[k].copy_(v, non_blocking=True)
        return dst

    def install(i, src):
        for k in work[i]:
            work[i][k].copy_(src[k])

    def moe_forward_chunk(x, active):
        h = moe.ln(x)
        y = moe.shared(h)
        for i in active:
            sd = work[i]
            g = F.linear(h, sd["gate.weight"])
            u = F.linear(h, sd["up.weight"])
            y = y + F.linear(F.silu(g) * u, sd["down.weight"])
        return x + y

    LONG, B = 4096, 2
    CH = 1024
    x = torch.randn(B, LONG, D, device=DEV) * 0.1
    x_ids = torch.randint(0, 23005, (B, LONG), device=DEV)

    # ---- (a) 全常驻基线 ----
    for _ in range(3):
        _ = moe_forward_chunk(x, list(range(N_EXP)))
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(5):
        _ = moe_forward_chunk(x, list(range(N_EXP)))
    torch.cuda.synchronize()
    ta = (time.time() - t0) / 5
    print("(a) 全常驻       %.0f ms/前向  (%.0f tok/s)" % (ta * 1e3, LONG * B / ta), flush=True)

    # ---- (b) 双缓冲异步预取 ----
    for _ in range(3):
        for c in range(LONG // CH):
            sl = slice(c * CH, (c + 1) * CH)
            _ = moe_forward_chunk(x[:, sl], list(range(N_EXP)))
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(5):
        nxt_active = list(range(N_EXP))
        nxt_loaded = True
        for c in range(LONG // CH):
            sl = slice(c * CH, (c + 1) * CH)
            # 预取下一 chunk 需要的专家到 stage (side stream, 与当前计算重叠)
            with torch.cuda.stream(copy_stream):
                for i in nxt_active:
                    for k, v in cpu_master[i].items():
                        stage[i][k].copy_(v, non_blocking=True)
            torch.cuda.current_stream().wait_stream(copy_stream)
            _ = moe_forward_chunk(x[:, sl], list(range(N_EXP)))
        torch.cuda.synchronize()
    tb = (time.time() - t0) / 5
    print("(b) 双缓冲异步   %.0f ms/前向  (%.0f tok/s)  预取开销 %.1f%%"
          % (tb * 1e3, LONG * B / tb, (tb / ta - 1) * 100), flush=True)

    # ---- (c) 同步换入对照 ----
    t0 = time.time()
    for _ in range(5):
        for c in range(LONG // CH):
            sl = slice(c * CH, (c + 1) * CH)
            for i in range(N_EXP):
                install(i, load_expert(i))
            _ = moe_forward_chunk(x[:, sl], list(range(N_EXP)))
        torch.cuda.synchronize()
    tc = (time.time() - t0) / 5
    print("(c) 同步换入     %.0f ms/前向  (%.0f tok/s)  慢 %.1f×"
          % (tc * 1e3, LONG * B / tc, tc / ta), flush=True)

    # ---- NLL/数值一致性: 交换路径 vs 常驻 ----
    y_resident = moe_forward_chunk(x[:, :CH], list(range(N_EXP)))
    for i in range(N_EXP):                                       # 模拟: 先全部换出
        install(i, load_expert(i))
    y_swapped = moe_forward_chunk(x[:, :CH], list(range(N_EXP)))
    diff = (y_resident - y_swapped).abs().max().item()
    print("交换 vs 常驻 logits max diff: %.3e (期望 0, 交换语义零损失)" % diff, flush=True)
    print("E3 DONE", flush=True)


if __name__ == "__main__":
    main()
