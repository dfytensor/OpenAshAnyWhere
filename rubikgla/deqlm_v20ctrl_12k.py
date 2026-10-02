#!/usr/bin/env python3
"""v20: 实验2续 — 调优版三段循环 @23M, 完成 naive/tuned x loop/plain 2x2.
       loop  plain
naive  v13   v9     (2.4922 / 2.536 — 循环赢 0.044)
tuned  v20   v19    (v19=2.4682, v20=? 若 v20<v19 则调优下循环仍赢, 否则优势是基线伪影)
结构: [std][A×10][B×10][C×10][std], 组件 = RoPE+RMSNorm+GQA+标准残差(循环内保留α混合)"""
import sys, os, time, math, json, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deqlm_v19_tuned23m import RMSNorm, rope_cache, apply_rope, GQAAttn, SwiGLUFFN, TunedBlock, masked_suffix_eval
from deqlm_v5_30m import get_batch, PT_CACHE

DEV = "cuda"
V, D, H = 23005, 320, 10
KV_H = 2
SEQ = 256
PT_STEPS = 12000
WARMUP = 1000
KS = [10, 10, 10]

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "deqlm20c.log")
OUT = os.path.join(HERE, "deqlm20c_results.json")
CKPT = os.path.join(HERE, "deqlm20c_pt_full.pth")


def log(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


class TunedTiedBlock(nn.Module):
    """调优版绑定块: RoPE+RMSNorm+GQA+SwiGLU, 迭代 z <- z + a*Block(z)."""

    def __init__(self, kmax, alpha_init=0.1):
        super().__init__()
        self.kmax = kmax
        self.attn = GQAAttn()
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


class TunedTriLoopLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.pre = TunedBlock()
        self.loops = nn.ModuleList([TunedTiedBlock(k) for k in KS])
        self.post = TunedBlock()
        self.ln_f = RMSNorm(D)
        self.head = nn.Linear(D, V, bias=False)
        nn.init.normal_(self.emb.weight, std=0.02)
        nn.init.normal_(self.head.weight, std=0.02)
        for p in self.layers_params():
            if p.dim() > 1:
                nn.init.normal_(p, std=0.02)

    def layers_params(self):
        for m in [self.pre, self.post, *self.loops]:
            for p in m.parameters():
                yield p

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
    m = TunedTriLoopLM().to(DEV)
    log("v20ctrl(12k): %.2fM, 调优三段循环 (对照 v19 tuned-plain=2.4682, v13 naive-loop=2.4922)"
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
            alphas = " ".join("%.3f" % lp.alpha().item() for lp in m.loops)
            log("%d/%d loss=%.4f a=[%s] (%.0fms/步, ETA %.1f时)"
                % (st, PT_STEPS, loss.item(), alphas,
                   (time.time() - t0) / (st + 1) * 1000,
                   (PT_STEPS - st) * (time.time() - t0) / (st + 1) / 3600))
    torch.save(m.state_dict(), CKPT)
    a, mk = masked_suffix_eval(m, val)
    log("v20 完成: 含pad=%.4f 真实=%.4f  [v19 tuned-plain=2.4682, v13 naive-loop=2.4922]" % (a, mk))
    json.dump(dict(all=a, masked=mk,
                   alphas=[round(lp.alpha().item(), 4) for lp in m.loops]), open(OUT, "w"), indent=2)


if __name__ == "__main__":
    main()
