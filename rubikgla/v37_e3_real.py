#!/usr/bin/env python3
"""E3 真实规模: D=2048 / FFN=5504 / 8专家 — 测双缓冲预取开销随 chunk 的变化.
预期: 小 chunk 拷贝>计算 (开销显性), 大 chunk 计算>拷贝 (开销隐藏).
三种模式: (a) 全常驻 (b) CPU主拷贝+side-stream双缓冲异步预取 (c) 同步换入对照.
"""
import sys, os, time, random
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEV = "cuda"
D, FFN_H, N_EXP = 2048, 5504, 8


class Expert(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(D, FFN_H, bias=False)
        self.up = nn.Linear(D, FFN_H, bias=False)
        self.down = nn.Linear(FFN_H, D, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class MoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm(D)
        self.experts = nn.ModuleList([Expert() for _ in range(N_EXP)])

    def forward(self, x, active):
        h = self.ln(x)
        y = torch.zeros_like(h)
        for i in active:
            y = y + self.experts[i](h)
        return x + y


def main():
    torch.manual_seed(0)
    moe = MoE().to(DEV)
    for e in moe.experts:                                   # bf16 权重 (真实部署精度)
        e.to(torch.bfloat16)
    moe.ln.to(torch.bfloat16)                               # ln 同步 bf16 (dtype 一致)
    moe.eval()

    # CPU 主拷贝 (pinned)
    cpu_master = [{k: v.detach().cpu().pin_memory() for k, v in e.state_dict().items()}
                  for e in moe.experts]
    work = [{k: torch.empty_like(v, device=DEV) for k, v in cpu_master[i].items()}
            for i in range(N_EXP)]
    for i in range(N_EXP):
        for k in work[i]:
            work[i][k].copy_(cpu_master[i][k])
    stage = [{k: torch.empty_like(v, device=DEV) for k, v in cpu_master[i].items()}
             for i in range(N_EXP)]
    copy_stream = torch.cuda.Stream()
    TOPK = 2                                            # 每窗换入/计算的专家子集 (DeepSeek 6+1 的微缩)
    active = list(range(TOPK))                          # 只换入+计算 top-k 子集: 拷贝量 ÷ (N_EXP/TOPK)

    def fwd(x):
        h = moe.ln(x)
        y = torch.zeros_like(h)
        for i in active:
            sd = work[i]
            g = F.linear(h, sd["gate.weight"])
            u = F.linear(h, sd["up.weight"])
            y = y + F.linear(F.silu(g) * u, sd["down.weight"])
        return x + y

    print("专家参数: %.0fMB/个 (bf16), %d 个 = %.1fGB 常驻当量"
          % (sum(v.numel() * 2 for v in cpu_master[0].values()) / 1e6, N_EXP,
             N_EXP * sum(v.numel() * 2 for v in cpu_master[0].values()) / 1e9), flush=True)

    for CH in (1024, 4096, 16384):
        x = torch.randn(1, CH, D, device=DEV, dtype=torch.bfloat16)

        reps = 3 if CH <= 4096 else 1                    # 大 chunk 逐 token 循环太慢, 降采样

        # (a) 全常驻
        for _ in range(2):
            _ = fwd(x)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(reps):
            _ = fwd(x)
        torch.cuda.synchronize()
        ta = (time.time() - t0) / reps

        # (b/c) 换入粒度扫描: 每 SWAP_EVERY token 换入一次, 测摊销开销
        for SWAP in (128, 1024, 4096):
            # 预热
            for w0 in range(0, min(CH, SWAP)):
                _ = fwd(x[:, w0:w0 + 1])
            torch.cuda.synchronize()

            # (b) 双缓冲异步: 窗口开始异步预取到 stage, install 后整窗计算 (预取与计算重叠)
            t0 = time.time()
            for _ in range(reps):
                for w0 in range(0, CH, SWAP):
                    with torch.cuda.stream(copy_stream):
                        for i in active:
                            for k, v in cpu_master[i].items():
                                stage[i][k].copy_(v, non_blocking=True)
                    torch.cuda.current_stream().wait_stream(copy_stream)
                    for i in active:
                        for k in stage[i]:
                            work[i][k].copy_(stage[i][k])   # install: stage→work (D2D, 快)
                    _ = fwd(x[:, w0:min(w0 + SWAP, CH)])    # 整窗 prefill
                    torch.cuda.synchronize()
            torch.cuda.synchronize()
            tb = (time.time() - t0) / reps

            # (c) 同步换入对照
            t0 = time.time()
            for _ in range(reps):
                for w0 in range(0, CH, SWAP):
                    for i in active:
                        for k, v in cpu_master[i].items():
                            work[i][k].copy_(cpu_master[i][k], non_blocking=False)
                    torch.cuda.synchronize()
                    _ = fwd(x[:, w0:min(w0 + SWAP, CH)])    # 整窗 prefill
                    torch.cuda.synchronize()
            torch.cuda.synchronize()
            tc = (time.time() - t0) / reps

            print("chunk=%6d swap_every=%5d | (a)常驻 %7.1fms | (b)异步 %9.1fms (开销 %8.1f%%) | (c)同步 %9.1fms (慢 %6.1f×)"
                  % (CH, SWAP, ta * 1e3, tb * 1e3, (tb / ta - 1) * 100, tc * 1e3, tc / ta), flush=True)
    print("E3-REAL DONE (预期: chunk 越大, (b) 越接近 (a); (c) 恒慢)", flush=True)


if __name__ == "__main__":
    main()
