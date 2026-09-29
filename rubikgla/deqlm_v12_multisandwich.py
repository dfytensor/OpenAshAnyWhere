#!/usr/bin/env python3
"""v12: 多明治 (双段循环) — 多样性假说检验.
结构: emb → [std] → [tied_A ×15] → [std] → [tied_B ×15] → [std] → head
参数: 3 std + 2 tied = 5 块 = 23.05M (与 v9/v10 精确对齐)
预测: 多样性假说成立则 v12 > v10 (2.508) > v9 (2.536); 否则 v12 ≈ v10
训练: 全反传, 39,695 步, 同 v10 协议"""
import sys, os, time, math, json, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deqlm_v5_30m import StdBlock, TiedBlock, get_batch, V, D, H, DEV, SEQ, PT_CACHE

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "deqlm12.log")
OUT = os.path.join(HERE, "deqlm12_results.json")
CKPT = os.path.join(HERE, "deqlm12_pt_full.pth")
PT_STEPS = 39695
WARMUP = 1000
K_SEG = 15               # 每段迭代次数 (2 段 × 15 = 总 30, 与 v10 对齐)


def log(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


class MultiSandwichLM(nn.Module):
    """emb → std → tied_A×K → std → tied_B×K → std → head"""

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.pos = nn.Embedding(SEQ + 1, D)
        self.ln_emb = nn.LayerNorm(D)
        self.pre = StdBlock()                 # 段前
        self.loop_a = TiedBlock(kmax=K_SEG)   # 循环 A
        self.mid = StdBlock()                 # 段间桥
        self.loop_b = TiedBlock(kmax=K_SEG)   # 循环 B
        self.post = StdBlock()                # 段后
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
        h, res_a, _ = self.loop_a(h)          # ×15, 全反传
        h = self.mid(h)
        h, res_b, _ = self.loop_b(h)          # ×15, 全反传
        h = self.post(h)
        return self.head(self.ln_f(h)), (res_a + res_b) / 2, K_SEG * 2


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
    m = MultiSandwichLM().to(DEV)
    log("v12 多明治: %.2fM (对照 v9/v10 = 23.05M), 结构 [std][A×15][std][B×15][std]"
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
            log("%d/%d loss=%.4f aA=%.3f aB=%.3f (%.0fms/步, ETA %.0f分)"
                % (st, PT_STEPS, loss.item(), m.loop_a.alpha().item(), m.loop_b.alpha().item(),
                   (time.time() - t0) / (st + 1) * 1000, (PT_STEPS - st) * (time.time() - t0) / (st + 1) / 60))
    torch.save(m.state_dict(), CKPT)
    a, mk = masked_suffix_eval(m, val)
    log("v12 完成: 含pad=%.4f 真实=%.4f  [对照: v10=2.508, v9=2.536, v7=3.867]" % (a, mk))
    json.dump(dict(all=a, masked=mk, alpha_a=round(m.loop_a.alpha().item(), 4),
                   alpha_b=round(m.loop_b.alpha().item(), 4)), open(OUT, "w"), indent=2)


if __name__ == "__main__":
    main()
