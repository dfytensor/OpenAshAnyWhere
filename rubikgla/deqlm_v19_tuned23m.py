#!/usr/bin/env python3
"""v19: 实验2 — 调优基线 @23M.
v9 (简化基线): 23.05M, 学习式位置嵌入, LayerNorm, MHA, α门残差, 零初始化头, 真实 2.536
v19 (调优):    RoPE + RMSNorm + GQA(kv=2) + 标准残差(无α门) + 标准初始化(非零头), ~22.9M
判定: v19 vs v9 (同为5层): 调优带来多少; v19 vs v13 (2.4922): 循环优势是否被吃掉"""
import sys, os, time, math, json, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deqlm_v5_30m import get_batch, PT_CACHE

DEV = "cuda"
V, D, H = 23005, 320, 10
KV_H = 2               # GQA: 2 个 kv 头共享给 10 个 query 头
SEQ = 256
PT_STEPS = 39695
WARMUP = 1000
N_LAYERS = 5

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "deqlm19.log")
OUT = os.path.join(HERE, "deqlm19_results.json")
CKPT = os.path.join(HERE, "deqlm19_pt_full.pth")


def log(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return self.w * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


def rope_cache(seq, dh, theta=10000.0, device="cuda"):
    inv = 1.0 / (theta ** (torch.arange(0, dh, 2, device=device).float() / dh))
    t = torch.arange(seq, device=device).float()
    freqs = torch.outer(t, inv)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    # x: (B, H, L, dh); cos/sin: (L, dh/2)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    o1 = x1 * cos - x2 * sin
    o2 = x1 * sin + x2 * cos
    out = torch.empty_like(x)
    out[..., 0::2] = o1
    out[..., 1::2] = o2
    return out


class GQAAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.H, self.KV, self.dh = H, KV_H, D // H
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
        att = att.masked_fill((i[:, None] < i[None, :])[None, None], float("-inf"))
        o = (torch.softmax(att, -1) @ v).transpose(1, 2).reshape(B, L, D)
        return x + self.out(o)


class SwiGLUFFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = RMSNorm(D)
        self.gate = nn.Linear(D, 4 * D, bias=False)
        self.up = nn.Linear(D, 4 * D, bias=False)
        self.down = nn.Linear(4 * D, D, bias=False)

    def forward(self, x):
        h = self.ln(x)
        return x + self.down(F.silu(self.gate(h)) * self.up(h))


class TunedBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = GQAAttn()
        self.ffn = SwiGLUFFN()

    def forward(self, x):
        return self.ffn(self.attn(x))


class TunedLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.layers = nn.ModuleList([TunedBlock() for _ in range(N_LAYERS)])
        self.ln_f = RMSNorm(D)
        self.head = nn.Linear(D, V, bias=False)
        nn.init.normal_(self.emb.weight, std=0.02)
        nn.init.normal_(self.head.weight, std=0.02)
        for p in self.layers.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, std=0.02)

    def forward(self, ids):
        h = self.emb(ids)
        for layer in self.layers:
            h = layer(h)
        return self.head(self.ln_f(h))


@torch.no_grad()
def masked_suffix_eval(m, val, nb=60, seed=777, bs=16):
    m.eval()
    tot_a, tok_a, tot_m, tok_m = 0.0, 0, 0.0, 0
    rng = random.Random(seed)
    for i in range(nb):
        xs = []
        for _ in range(bs):
            s = val[rng.randrange(len(val))][:256]
            xs.append(F.pad(s, (0, 256 - s.numel())))
        x = torch.stack(xs).to(DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lo = m(x).float()
        ys = x[:, 1:]
        l = F.cross_entropy(lo[:, :-1].reshape(-1, V), ys.reshape(-1), reduction="none").view(x.shape[0], -1)
        ls = l[:, 192:255]
        msk = x[:, 193:256] != 0
        tot_a += ls.sum().item(); tok_a += ls.numel()
        tot_m += ls[msk].sum().item(); tok_m += msk.sum().item()
    m.train()
    return round(tot_a / tok_a, 4), round(tot_m / tok_m, 4)


def main():
    torch.manual_seed(0)
    random.seed(0)
    seqs = torch.load(PT_CACHE, map_location="cpu", weights_only=True)
    train, val = seqs[:int(0.95 * len(seqs))], seqs[:5000]
    m = TunedLM().to(DEV)
    log("v19: %.2fM, 调优5层 RoPE+RMSNorm+GQA(kv=2)+标准残差 (对照 v9=2.536, v13=2.4922)"
        % (sum(p.numel() for p in m.parameters()) / 1e6))
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
            log("%d/%d loss=%.4f (%.0fms/步, ETA %.1f时)"
                % (st, PT_STEPS, loss.item(), (time.time() - t0) / (st + 1) * 1000,
                   (PT_STEPS - st) * (time.time() - t0) / (st + 1) / 3600))
    torch.save(m.state_dict(), CKPT)
    a, mk = masked_suffix_eval(m, val)
    log("v19 完成: 含pad=%.4f 真实=%.4f  [v9简化=2.536, v13循环=2.4922]" % (a, mk))
    json.dump(dict(all=a, masked=mk), open(OUT, "w"), indent=2)


if __name__ == "__main__":
    main()
