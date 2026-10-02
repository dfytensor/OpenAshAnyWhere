#!/usr/bin/env python3
"""v21: Hybrid2 验证 @23M — 滑窗/全注意力混合 + 调优三段循环.
基于 v20 (23M 冠军 2.4284). 结构:
  [pre:full][A:SWA-64][B:full][C:SWA-64][post:full]
注意力遍数: full 12 / SWA 20 (窗64/全长256=25%成本) -> 注意力 FLOP 省 ~47%
判定: v21 ≈ 2.4284±0.02 → hybrid2 在循环架构上成立; 大跌 → 循环块需要全局注意力"""
import sys, os, time, math, json, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deqlm_v19_tuned23m import RMSNorm, rope_cache, apply_rope, SwiGLUFFN, TunedBlock, masked_suffix_eval
from deqlm_v5_30m import get_batch, PT_CACHE

DEV = "cuda"
V, D, H = 23005, 320, 10
KV_H = 2
SEQ = 256
PT_STEPS = 39695
WARMUP = 1000
KS = [10, 10, 10]
WIN = 64               # 滑窗宽度

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "deqlm21.log")
OUT = os.path.join(HERE, "deqlm21_results.json")
CKPT = os.path.join(HERE, "deqlm21_pt_full.pth")


def log(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


class HybridGQAAttn(nn.Module):
    """GQA + RoPE; window=None 全注意力, 否则 |i-j|<window 的因果滑窗."""

    def __init__(self, window=None):
        super().__init__()
        self.H, self.KV, self.dh = H, KV_H, D // H
        self.window = window
        self.ln = RMSNorm(D)
        self.wq = nn.Linear(D, D, bias=False)
        self.wk = nn.Linear(D, KV_H * self.dh, bias=False)
        self.wv = nn.Linear(D, KV_H * self.dh, bias=False)
        self.out = nn.Linear(D, D, bias=False)

    def forward(self, x):
        B, L, _ = x.shape
        h = self.ln(x)
        q = self.wq(h).view(B, L, self.H, self.dh).transpose(1, 2)
        k = self.wk(h).view(B, L, self.KV, self.dh).transpose(1, 2)
        v = self.wv(h).view(B, L, self.KV, self.dh).transpose(1, 2)
        cos, sin = rope_cache(L, self.dh, device=x.device)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        k = k.repeat_interleave(self.H // self.KV, dim=1)
        v = v.repeat_interleave(self.H // self.KV, dim=1)
        att = q @ k.transpose(-1, -2) / self.dh ** 0.5
        i = torch.arange(L, device=x.device)
        causal = i[:, None] < i[None, :]
        if self.window is None:
            mask = causal
        else:
            dist = i[:, None] - i[None, :]
            mask = causal | (dist >= self.window)
        att = att.masked_fill(mask[None, None], float("-inf"))
        o = (torch.softmax(att, -1) @ v).transpose(1, 2).reshape(B, L, D)
        return x + self.out(o)


class HybridBlock(nn.Module):
    def __init__(self, window=None):
        super().__init__()
        self.attn = HybridGQAAttn(window)
        self.ffn = SwiGLUFFN()

    def forward(self, x):
        return self.ffn(self.attn(x))


class HybridTiedBlock(nn.Module):
    def __init__(self, kmax, window=None, alpha_init=0.1):
        super().__init__()
        self.kmax = kmax
        self.attn = HybridGQAAttn(window)
        self.ffn = SwiGLUFFN()
        self.alpha_param = nn.Parameter(torch.tensor(math.log(alpha_init / (1 - alpha_init))))

    def alpha(self):
        return torch.clamp(torch.sigmoid(self.alpha_param), 0.01, 0.9)

    def forward(self, z0):
        alpha = self.alpha()
        z = z0
        for _ in range(self.kmax):
            z = z + alpha * self.ffn(self.attn(z))
        return z


class HybridTriLoopLM(nn.Module):
    """[pre:full][A:SWA][B:full][C:SWA][post:full]"""

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.pre = HybridBlock(window=None)
        self.loops = nn.ModuleList([
            HybridTiedBlock(KS[0], window=WIN),
            HybridTiedBlock(KS[1], window=None),
            HybridTiedBlock(KS[2], window=WIN),
        ])
        self.post = HybridBlock(window=None)
        self.ln_f = RMSNorm(D)
        self.head = nn.Linear(D, V, bias=False)
        nn.init.normal_(self.emb.weight, std=0.02)
        nn.init.normal_(self.head.weight, std=0.02)
        for m in [self.pre, self.post, *self.loops]:
            for p in m.parameters():
                if p.dim() > 1:
                    nn.init.normal_(p, std=0.02)

    def forward(self, ids):
        h = self.emb(ids)
        h = self.pre(h)
        for lp in self.loops:
            h = lp(h)
        h = self.post(h)
        return self.head(self.ln_f(h))


def main():
    torch.manual_seed(0)
    random.seed(0)
    seqs = torch.load(PT_CACHE, map_location="cpu", weights_only=True)
    train, val = seqs[:int(0.95 * len(seqs))], seqs[:5000]
    m = HybridTriLoopLM().to(DEV)
    log("v21: %.2fM, Hybrid2 [full][SWA%d][full][SWA%d][full] (对照 v20 全注意力=2.4284)"
        % (sum(p.numel() for p in m.parameters()) / 1e6, WIN, WIN))
    opt = optim.AdamW(m.parameters(), lr=4e-4, weight_decay=0.01, betas=(0.9, 0.95))

    def lr_at(st):
        if st < WARMUP:
            return 4e-4 * st / WARMUP
        p_ = (st - WARMUP) / max(1, PT_STEPS - WARMUP)
        return 4e-5 + 0.5 * (4e-4 - 4e-5) * (1 + math.cos(math.pi * p_))

    t0 = time.time()
    rng = random.Random(42)
    for st in range(PT_STEPS):
        for g in opt.param_groups:
            g["lr"] = lr_at(st)
        m.train()
        x, y = get_batch(train, bs=32, device=DEV, rng=rng)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = m(x)
        loss = F.cross_entropy(logits.float().reshape(-1, V), y.reshape(-1), ignore_index=0)
        if not torch.isfinite(loss):
            opt.zero_grad(set_to_none=True)
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if st % 2000 == 0:
            alphas = " ".join("%.3f" % lp.alpha().item() for lp in m.loops)
            log("%d/%d loss=%.4f a=[%s] (%.0fms/步, ETA %.1f时)"
                % (st, PT_STEPS, loss.item(), alphas,
                   (time.time() - t0) / (st + 1) * 1000,
                   (PT_STEPS - st) * (time.time() - t0) / (st + 1) / 3600))
    torch.save(m.state_dict(), CKPT)
    a, mk = masked_suffix_eval(m, val)
    log("v21 完成: 含pad=%.4f 真实=%.4f  [v20 全注意力=2.4284]" % (a, mk))
    json.dump(dict(all=a, masked=mk, win=WIN,
                   alphas=[round(lp.alpha().item(), 4) for lp in m.loops]), open(OUT, "w"), indent=2)


if __name__ == "__main__":
    main()
