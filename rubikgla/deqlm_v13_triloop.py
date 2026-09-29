#!/usr/bin/env python3
"""v13: 三段多明治 — 多样性收益递减趋势检验.
结构: emb → [std] → [tied_A ×10] → [tied_B ×10] → [tied_C ×10] → [std] → head
参数: 2 std + 3 tied = 5 块 = 23.05M (对齐 v9/v10/v12)
趋势预测: v9 (2.536, 0函数) → v10 (2.508, 1函数) → v12 (2.4994, 2函数) → v13 (?, 3函数)
  若递减持续: v13 ≈ 2.495±0.003; 若桥梁层(桥接 std)才是关键: v13 ≥ v12"""
import sys, os, time, math, json, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deqlm_v5_30m import StdBlock, TiedBlock, get_batch, V, D, H, DEV, SEQ, PT_CACHE

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "deqlm13.log")
OUT = os.path.join(HERE, "deqlm13_results.json")
CKPT = os.path.join(HERE, "deqlm13_pt_full.pth")
PT_STEPS = 39695
WARMUP = 1000
K_SEG = 10               # 3 段 × 10 = 30 总迭代, 对齐


def log(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


class TriLoopLM(nn.Module):
    """emb → std → A×10 → B×10 → C×10 → std → head"""

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.pos = nn.Embedding(SEQ + 1, D)
        self.ln_emb = nn.LayerNorm(D)
        self.pre = StdBlock()
        self.loop_a = TiedBlock(kmax=K_SEG)
        self.loop_b = TiedBlock(kmax=K_SEG)
        self.loop_c = TiedBlock(kmax=K_SEG)
        self.post = StdBlock()
        self.ln_f = nn.LayerNorm(D)
        self.head = nn.Linear(D, V)
        nn.init.normal_(self.emb.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, ids):
        B, L = ids.shape
        h = self.emb(ids) + self.pos(torch.arange(L, device=ids.device)).unsqueeze(0)
        h = self.ln_emb(h)
        h = self.pre(h)
        h, ra, _ = self.loop_a(h)
        h, rb, _ = self.loop_b(h)
        h, rc, _ = self.loop_c(h)
        h = self.post(h)
        return self.head(self.ln_f(h)), (ra + rb + rc) / 3, K_SEG * 3


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
            out, _, _ = m(x)
        lo = out.float()
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
    m = TriLoopLM().to(DEV)
    log("v13 三段: %.2fM, [std][A×10][B×10][C×10][std]"
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
            logits, res, _ = m(x)
        loss = F.cross_entropy(logits.float().reshape(-1, V), y.reshape(-1), ignore_index=0)
        if not torch.isfinite(loss):
            opt.zero_grad(set_to_none=True)
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if st % 2000 == 0:
            log("%d/%d loss=%.4f a=[%.3f %.3f %.3f] (%.0fms/步, ETA %.0f分)"
                % (st, PT_STEPS, loss.item(), m.loop_a.alpha().item(), m.loop_b.alpha().item(),
                   m.loop_c.alpha().item(), (time.time() - t0) / (st + 1) * 1000,
                   (PT_STEPS - st) * (time.time() - t0) / (st + 1) / 60))
    torch.save(m.state_dict(), CKPT)
    a, mk = masked_suffix_eval(m, val)
    log("v13 完成: 含pad=%.4f 真实=%.4f  [v12=2.4994, v10=2.508, v9=2.536]" % (a, mk))
    json.dump(dict(all=a, masked=mk,
                   alphas=[round(m.loop_a.alpha().item(), 4), round(m.loop_b.alpha().item(), 4),
                           round(m.loop_c.alpha().item(), 4)]), open(OUT, "w"), indent=2)


if __name__ == "__main__":
    main()
