#!/usr/bin/env python3
"""v16: 横向 — 三段多明治放大到 198.6M (d=1024, H=16).
结构: emb → [std×3] → [A×10][B×10][C×10] → [std×3] → head  (9 块, 有效深度 36)
预算: emb/head 47.1M + 9×16.8M + pos 0.26M = 198.6M ≤ 200M
对照计划: v17 纯 9 层同预算 (之后跑). lr 3e-4 (大模型降档)."""
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
KS = [10, 10, 10]

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "deqlm16.log")
OUT = os.path.join(HERE, "deqlm16_results.json")
CKPT = os.path.join(HERE, "deqlm16_pt_full.pth")


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


class TiedBlock(nn.Module):
    def __init__(self, kmax, alpha_init=0.1):
        super().__init__()
        self.H, self.dh = H, D // H
        self.kmax = kmax
        self.ln1 = nn.LayerNorm(D)
        self.ln2 = nn.LayerNorm(D)
        self.qkv = nn.Linear(D, 3 * D)
        self.attn_out = nn.Linear(D, D)
        self.gate = nn.Linear(D, 4 * D)
        self.up = nn.Linear(D, 4 * D)
        self.down = nn.Linear(4 * D, D)
        self.alpha_param = nn.Parameter(torch.tensor(math.log(alpha_init / (1 - alpha_init))))

    def _block_fn(self, z):
        h = self.ln1(z)
        B, L = h.shape[:2]
        qkv = self.qkv(h).view(B, L, 3, self.H, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = q @ k.transpose(-1, -2) / self.dh ** 0.5
        i = torch.arange(L, device=z.device)
        att = att.masked_fill((i[:, None] < i[None, :])[None, None], float("-inf"))
        o = (torch.softmax(att, -1) @ v).transpose(1, 2).reshape(B, L, D)
        h = z + self.attn_out(o)
        h = self.ln2(h)
        return h + self.down(F.silu(self.gate(h)) * self.up(h))

    def alpha(self):
        return torch.clamp(torch.sigmoid(self.alpha_param), 0.01, 0.9)

    def forward(self, z0):
        alpha = self.alpha()
        z = z0
        for _ in range(self.kmax):
            z = z + alpha * self._block_fn(z)
        return z


class TriLoop200M(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.pos = nn.Embedding(SEQ + 1, D)
        self.ln_emb = nn.LayerNorm(D)
        self.pre = nn.ModuleList([StdBlock() for _ in range(3)])
        self.loops = nn.ModuleList([TiedBlock(k) for k in KS])
        self.post = nn.ModuleList([StdBlock() for _ in range(3)])
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
        for layer in self.pre:
            h = layer(h)
        for lp in self.loops:
            h = lp(h)
        for layer in self.post:
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
    m = TriLoop200M().to(DEV)
    log("v16: %.2fM, [std×3][A×10][B×10][C×10][std×3], d=%d H=%d"
        % (sum(p.numel() for p in m.parameters()) / 1e6, D, H))
    opt = optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.01, betas=(0.9, 0.95))

    def lr_at(st):
        if st < WARMUP:
            return 3e-4 * st / WARMUP
        p_ = (st - WARMUP) / max(1, PT_STEPS - WARMUP)
        return 3e-5 + 0.5 * (3e-4 - 3e-5) * (1 + math.cos(math.pi * p_))

    t0 = time.time()
    rng = random.Random(42)
    cuda_err = 0
    B_MICRO = 16          # B16 x accum2 = 等效 32 (显存 13.9GB, expandable_segments 在 d1024 下有驱动bug 禁用)
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
            alphas = " ".join("%.3f" % lp.alpha().item() for lp in m.loops)
            log("%d/%d loss=%.4f a=[%s] (%.0fms/步, ETA %.1f时)"
                % (st, PT_STEPS, loss.item(), alphas,
                   (time.time() - t0) / (st + 1) * 1000,
                   (PT_STEPS - st) * (time.time() - t0) / (st + 1) / 3600))
        if (st + 1) % 10000 == 0:
            torch.save(m.state_dict(), CKPT)
    torch.save(m.state_dict(), CKPT)
    a, mk = masked_suffix_eval(m, val)
    log("v16 完成: 含pad=%.4f 真实=%.4f  [v13@23M=2.4922]" % (a, mk))
    json.dump(dict(all=a, masked=mk,
                   alphas=[round(lp.alpha().item(), 4) for lp in m.loops]), open(OUT, "w"), indent=2)


if __name__ == "__main__":
    main()
