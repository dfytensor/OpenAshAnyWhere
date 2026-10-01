#!/usr/bin/env python3
"""v18: 实验1 — 等FLOP对照 @23M家族.
v13 (三段循环): 23.05M 参数, 32 个层遍 (2 std + 30 循环), 真实 2.4922
v18 (纯32层):   67.2M 参数, 32 个层遍 — 同步数同每步FLOP, 参数 2.9x
判定: v18 < 2.4922 → 等参数优势只是算力换的; v18 >= v13 → 循环在等FLOP下也站得住"""
import sys, os, time, math, json, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deqlm_v5_30m import StdBlock, get_batch, V, D, H, DEV, SEQ, PT_CACHE

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "deqlm18.log")
OUT = os.path.join(HERE, "deqlm18_results.json")
CKPT = os.path.join(HERE, "deqlm18_pt_full.pth")
PT_STEPS = 39695
WARMUP = 1000
N_LAYERS = 32


def log(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


class Plain32LM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.pos = nn.Embedding(SEQ + 1, D)
        self.ln_emb = nn.LayerNorm(D)
        self.layers = nn.ModuleList([StdBlock() for _ in range(N_LAYERS)])
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
    m = Plain32LM().to(DEV)
    log("v18: %.2fM, 纯 %d 层 (等FLOP对照 v13: 23.05M/32遍/2.4922)"
        % (sum(p.numel() for p in m.parameters()) / 1e6, N_LAYERS))
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
    log("v18 完成: 含pad=%.4f 真实=%.4f  [v13=2.4922 (等FLOP, 1/2.9参数)]" % (a, mk))
    json.dump(dict(all=a, masked=mk), open(OUT, "w"), indent=2)


if __name__ == "__main__":
    main()
