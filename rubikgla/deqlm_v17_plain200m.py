#!/usr/bin/env python3
"""v17: 横向对照 — 纯 9 层 198.56M (与 v16 精确同参数), 检验多明治优势是否随规模保持.
v16 (三段循环): 真实 1.9644. 若 v17 > 1.9644 → 优势保持; 若 v17 <= → 规模下消失.
同数据/同步数/同 lr/同 B16+accum2 (expandable_segments 禁用)."""
import sys, os, time, math, json, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deqlm_v5_30m import get_batch, PT_CACHE

DEV = "cuda"
V = 23005
D, H = 1024, 16
SEQ = 256
PT_STEPS = 39695
WARMUP = 1000
N_LAYERS = 9

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "deqlm17.log")
OUT = os.path.join(HERE, "deqlm17_results.json")
CKPT = os.path.join(HERE, "deqlm17_pt_full.pth")


def log(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


class Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.H, self.dh = H, D // H
        self.ln = nn.LayerNorm(D)
        self.qkv = nn.Linear(D, 3 * D)
        self.out = nn.Linear(D, D)
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        B, L, _ = x.shape
        h = self.ln(x)
        qkv = self.qkv(h).view(B, L, 3, self.H, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = q @ k.transpose(-1, -2) / self.dh ** 0.5
        i = torch.arange(L, device=x.device)
        att = att.masked_fill((i[:, None] < i[None, :])[None, None], float("-inf"))
        o = (torch.softmax(att, -1) @ v).transpose(1, 2).reshape(B, L, D)
        return x + self.alpha * self.out(o)


class FFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm(D)
        self.gate = nn.Linear(D, 4 * D)
        self.up = nn.Linear(D, 4 * D)
        self.down = nn.Linear(4 * D, D)
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        h = self.ln(x)
        return x + self.alpha * self.down(F.silu(self.gate(h)) * self.up(h))


class StdBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = Attn()
        self.ffn = FFN()

    def forward(self, x):
        return self.ffn(self.attn(x))


class Plain200M(nn.Module):
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
    m = Plain200M().to(DEV)
    log("v17: %.2fM, 纯 %d 层 (对照 v16 三段循环 198.56M)"
        % (sum(p.numel() for p in m.parameters()) / 1e6, N_LAYERS))
    opt = optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.01, betas=(0.9, 0.95))

    def lr_at(st):
        if st < WARMUP:
            return 3e-4 * st / WARMUP
        p_ = (st - WARMUP) / max(1, PT_STEPS - WARMUP)
        return 3e-5 + 0.5 * (3e-4 - 3e-5) * (1 + math.cos(math.pi * p_))

    t0 = time.time()
    rng = random.Random(42)
    cuda_err = 0
    B_MICRO = 16
    for st in range(PT_STEPS):
        for g in opt.param_groups:
            g["lr"] = lr_at(st)
        m.train()
        x, y = get_batch(train, bs=B_MICRO, device=DEV, rng=rng)
        x2, y2 = get_batch(train, bs=B_MICRO, device=DEV, rng=rng)
        ok = False
        for attempt in range(4):
            try:
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = m(x)
                loss = F.cross_entropy(logits.float().reshape(-1, V), y.reshape(-1), ignore_index=0)
                if torch.isfinite(loss):
                    (loss * 0.5).backward()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits2 = m(x2)
                loss2 = F.cross_entropy(logits2.float().reshape(-1, V), y2.reshape(-1), ignore_index=0)
                if torch.isfinite(loss2):
                    (loss2 * 0.5).backward()
                if not (torch.isfinite(loss) and torch.isfinite(loss2)):
                    opt.zero_grad(set_to_none=True)
                    ok = True
                    break
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step()
                loss = 0.5 * (loss.detach() + loss2.detach())
                ok = True
                break
            except RuntimeError as e:
                cuda_err += 1
                log("st%d CUDA瞬态错误 (第%d次): %s -> 等15秒重试" % (st, attempt + 1, str(e)[:120]))
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                time.sleep(15)
        if not ok:
            log("st%d 连续4次失败 -> 存档退出" % st)
            torch.save(m.state_dict(), CKPT)
            json.dump(dict(crashed_at=st, cuda_err=cuda_err), open(OUT + ".crash", "w"))
            return
        if st % 1000 == 0:
            log("%d/%d loss=%.4f (%.0fms/步, ETA %.1f时)"
                % (st, PT_STEPS, loss.item(), (time.time() - t0) / (st + 1) * 1000,
                   (PT_STEPS - st) * (time.time() - t0) / (st + 1) / 3600))
        if (st + 1) % 10000 == 0:
            torch.save(m.state_dict(), CKPT)
    torch.save(m.state_dict(), CKPT)
    a, mk = masked_suffix_eval(m, val)
    log("v17 完成: 含pad=%.4f 真实=%.4f  [v16 三段循环=1.9644]" % (a, mk))
    json.dump(dict(all=a, masked=mk), open(OUT, "w"), indent=2)


if __name__ == "__main__":
    main()
