#!/usr/bin/env python3
"""v14: 四段循环 — 多样性趋势第 4 点.
结构: emb → [std] → [A×8] → [B×8] → [C×7] → [D×7] → head (1 std + 4 tied = 5 块 = 23.05M)
总迭代 8+8+7+7 = 30, 与 v10/v12/v13 对齐.
趋势: v9 2.536 (0) → v10 2.508 (1) → v12 2.4994 (2) → v13 2.4922 (3) → v14 ? (4)
注意混淆: v14 只剩 1 个 std (无后桥), 若 v14 ≥ v13 则说明桥层比第 4 个函数重要."""
import sys, os, time, math, json, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deqlm_v5_30m import StdBlock, TiedBlock, get_batch, V, D, H, DEV, SEQ, PT_CACHE

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "deqlm14.log")
OUT = os.path.join(HERE, "deqlm14_results.json")
CKPT = os.path.join(HERE, "deqlm14_pt_full.pth")
PT_STEPS = 39695
WARMUP = 1000
KS = [8, 8, 7, 7]          # 4 段, 总 30


def log(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


class QuadLoopLM(nn.Module):
    """emb → std → A×8 → B×8 → C×7 → D×7 → head"""

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.pos = nn.Embedding(SEQ + 1, D)
        self.ln_emb = nn.LayerNorm(D)
        self.pre = StdBlock()
        self.loops = nn.ModuleList([TiedBlock(kmax=k) for k in KS])
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
        res_sum = 0.0
        for lp in self.loops:
            h, r, _ = lp(h)
            res_sum += r
        return self.head(self.ln_f(h)), res_sum / len(self.loops), sum(KS)


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
    m = QuadLoopLM().to(DEV)
    log("v14 四段: %.2fM, [std][A×8][B×8][C×7][D×7]"
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
            alphas = " ".join("%.3f" % lp.alpha().item() for lp in m.loops)
            log("%d/%d loss=%.4f a=[%s] (%.0fms/步, ETA %.0f分)"
                % (st, PT_STEPS, loss.item(), alphas,
                   (time.time() - t0) / (st + 1) * 1000,
                   (PT_STEPS - st) * (time.time() - t0) / (st + 1) / 60))
    torch.save(m.state_dict(), CKPT)
    a, mk = masked_suffix_eval(m, val)
    log("v14 完成: 含pad=%.4f 真实=%.4f  [v13=2.4922, v12=2.4994, v10=2.508, v9=2.536]" % (a, mk))
    json.dump(dict(all=a, masked=mk,
                   alphas=[round(lp.alpha().item(), 4) for lp in m.loops]), open(OUT, "w"), indent=2)


if __name__ == "__main__":
    main()
