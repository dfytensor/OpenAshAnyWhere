#!/usr/bin/env python3
"""v9: 纯 Transformer 对照 (5 层, 参数对齐 23M), 同数据/同步数/同 SFT 协议.
回答: 绑定深度 (v8: 2+30×tied+2, 23M) 的优势是"绑定"带来的, 还是参数/训练量带来的?
PT: 39,695 步 全序列 loss (ignore_index=0), 同 v7
SFT: 28,304 步 后缀 loss (含 pad), 同 v8"""
import sys, os, time, math, json, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deqlm_v5_30m import StdBlock, get_batch, V, D, H, DEV, SEQ, PT_CACHE

HERE = os.path.dirname(os.path.abspath(__file__))
SFT_CACHE = r"F:\rowcol_llm\sft_cached_256.pt"
PT_VAL = PT_CACHE
LOG = os.path.join(HERE, "deqlm9.log")
OUT = os.path.join(HERE, "deqlm9_results.json")
CKPT_PT = os.path.join(HERE, "deqlm9_pt_full.pth")
CKPT_SFT = os.path.join(HERE, "deqlm9_sft_full.pth")
N_LAYERS = 5                    # 5×1.64M ≈ v8 的 4 std + 1 tied, 参数对齐
PT_STEPS, SFT_STEPS = 39695, 28304
P, Q, B = 192, 64, 32
WARMUP = 1000


def log(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


class VanillaLM(nn.Module):
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
    m = VanillaLM().to(DEV)
    log("v9 纯对照: %d 层, %.2fM 参数 (v8=23.05M)" % (N_LAYERS, sum(p.numel() for p in m.parameters()) / 1e6))

    # ═══ PT (同 v7 协议) ═══
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
        x, y = get_batch(train, bs=B, device=DEV, rng=rng)
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
            log("PT %d/%d loss=%.4f (%.0fms/步)" % (st, PT_STEPS, loss.item(), (time.time() - t0) / (st + 1) * 1000))
    torch.save(m.state_dict(), CKPT_PT)
    a, mk = masked_suffix_eval(m, val)
    log("PT 完成: 含pad=%.4f 真实=%.4f" % (a, mk))
    results = dict(pt=dict(all=a, masked=mk))

    # ═══ SFT (同 v8 协议) ═══
    cache = torch.load(SFT_CACHE, map_location="cpu", weights_only=True)
    G = cache["grids"]
    N_S = G.shape[0]
    opt = optim.AdamW(m.parameters(), lr=1e-4, weight_decay=0.01)
    t1 = time.time()
    for st in range(SFT_STEPS):
        m.train()
        idx = torch.randint(0, N_S, (B,))
        seq_ = (G[idx] - 34).clamp(min=0).to(DEV).reshape(B, 256)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = m(seq_)
        ys = seq_[:, P + 1:P + Q]
        loss = F.cross_entropy(logits.float()[:, P:P + Q - 1].reshape(-1, V), ys.reshape(-1))
        if not torch.isfinite(loss):
            opt.zero_grad(set_to_none=True)
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if st % 2000 == 0:
            log("SFT %d/%d loss=%.4f (%.0fms/步)" % (st, SFT_STEPS, loss.item(), (time.time() - t1) / (st + 1) * 1000))
    torch.save(m.state_dict(), CKPT_SFT)
    a, mk = masked_suffix_eval(m, val)
    log("SFT 完成: 含pad=%.4f 真实=%.4f" % (a, mk))
    results["sft"] = dict(all=a, masked=mk)

    json.dump(results, open(OUT, "w"), indent=2)
    log("v9 完成")


if __name__ == "__main__":
    main()
