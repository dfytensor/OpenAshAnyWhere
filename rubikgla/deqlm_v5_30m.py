#!/usr/bin/env python3
"""DEQ-LM v5: 30M 规模验证 (d=320, H=10). 配置同 v3, 只放大."""
import sys, os, time, math, json, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
DEV = "cuda"
V, D, H = 23005, 320, 10
N_BOTTOM, N_TOP = 2, 2
KMAX, ALPHA_INIT = 30, 0.1
B, SEQ = 32, 256
STEPS = 3000
PT_CACHE = r"F:\OpenASH2605\minimind_data\pretrain_cached_1270238_256.pt"
HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "deqlm5.log")
OUT = os.path.join(HERE, "deqlm5_results.json")
CKPT = os.path.join(HERE, "deqlm5_final.pth")

def log(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


class StdAttn(nn.Module):
    def __init__(self, d=D, H=H):
        super().__init__()
        self.d, self.H, self.dh = d, H, d // H
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.ln = nn.LayerNorm(d)

    def forward(self, x):
        B, L, _ = x.shape
        h = self.ln(x)
        qkv = self.qkv(h).view(B, L, 3, self.H, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = q @ k.transpose(-1, -2) / self.dh ** 0.5
        i = torch.arange(L, device=x.device)
        att = att.masked_fill((i[:, None] < i[None, :])[None, None], float("-inf"))
        o = (torch.softmax(att, -1) @ v).transpose(1, 2).reshape(B, L, self.d)
        return x + self.alpha * self.out(o)


class StdFFN(nn.Module):
    def __init__(self, d=D, h=None):
        super().__init__()
        h = h or 4 * d
        self.ln = nn.LayerNorm(d)
        self.gate = nn.Linear(d, h)
        self.up = nn.Linear(d, h)
        self.down = nn.Linear(h, d)
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        h = self.ln(x)
        return x + self.alpha * self.down(F.silu(self.gate(h)) * self.up(h))


class StdBlock(nn.Module):
    def __init__(self, d=D, H=H):
        super().__init__()
        self.attn = StdAttn(d, H)
        self.ffn = StdFFN(d)

    def forward(self, x):
        return self.ffn(self.attn(x))


class TiedBlock(nn.Module):
    def __init__(self, d=D, H=H, alpha_init=ALPHA_INIT, kmax=KMAX):
        super().__init__()
        self.d, self.H, self.dh = d, H, d // H
        self.kmax = kmax
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.attn_out = nn.Linear(d, d)
        self.gate = nn.Linear(d, 4 * d)
        self.up = nn.Linear(d, 4 * d)
        self.down = nn.Linear(4 * d, d)
        self.alpha_param = nn.Parameter(torch.tensor(math.log(alpha_init / (1 - alpha_init))))

    def _block_fn(self, z):
        h = self.ln1(z)
        B, L = h.shape[:2]
        qkv = self.qkv(h).view(B, L, 3, self.H, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = q @ k.transpose(-1, -2) / self.dh ** 0.5
        i = torch.arange(L, device=z.device)
        att = att.masked_fill((i[:, None] < i[None, :])[None, None], float("-inf"))
        o = (torch.softmax(att, -1) @ v).transpose(1, 2).reshape(B, L, -1)
        h = z + self.attn_out(o)
        h = self.ln2(h)
        return h + self.down(F.silu(self.gate(h)) * self.up(h))

    def alpha(self):
        return torch.clamp(torch.sigmoid(self.alpha_param), 0.01, 0.9)

    def forward(self, z0, exact_k=None):
        alpha = self.alpha()
        z = z0
        znorm = z0.norm(dim=-1).clamp(min=1e-6)
        residual = 0.0
        n_iter = exact_k if exact_k is not None else self.kmax
        for k in range(n_iter):
            z_new = z + alpha * self._block_fn(z)
            residual = ((z_new - z).norm(dim=-1) / znorm).max().item()
            z = z_new
        return z, residual, n_iter


class DEQLM30(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.pos = nn.Embedding(SEQ + 1, D)
        self.ln_emb = nn.LayerNorm(D)
        self.bottom = nn.ModuleList([StdBlock() for _ in range(N_BOTTOM)])
        self.deq = TiedBlock()
        self.top = nn.ModuleList([StdBlock() for _ in range(N_TOP)])
        self.ln_f = nn.LayerNorm(D)
        self.head = nn.Linear(D, V)
        nn.init.normal_(self.emb.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def bottom_h(self, ids):
        B, L = ids.shape
        h = self.emb(ids) + self.pos(torch.arange(L, device=ids.device)).unsqueeze(0)
        h = self.ln_emb(h)
        for layer in self.bottom:
            h = layer(h)
        return h

    def forward(self, ids, exact_k=None):
        h = self.bottom_h(ids)
        h, residual, k_used = self.deq(h, exact_k=exact_k)
        for layer in self.top:
            h = layer(h)
        return self.head(self.ln_f(h)), residual, k_used


def get_batch(data, bs=B, sl=SEQ, device=DEV, rng=None):
    if rng is None:
        rng = random
    xs = []
    for _ in range(bs):
        s = data[rng.randrange(len(data))][:sl]
        xs.append(F.pad(s, (0, sl - s.numel())))
    x = torch.stack(xs).to(device)
    y = x.roll(-1, dims=1)
    y[:, -1] = 0
    return x, y


@torch.no_grad()
def eval_nll(m, val, exact_k=None, nb=20, seed=999):
    m.eval()
    tot, tok = 0.0, 0
    for i in range(nb):
        x, y = get_batch(val, bs=B, device=DEV, rng=random.Random(seed + i))
        out, _, _ = m(x, exact_k=exact_k)
        l = F.cross_entropy(out.float().reshape(-1, V), y.reshape(-1), reduction="sum")
        tot += l.item(); tok += y.numel()
    m.train()
    return tot / tok


def main():
    torch.manual_seed(0)
    random.seed(0)
    seqs = torch.load(PT_CACHE, map_location="cpu", weights_only=True)
    train, val = seqs[:int(0.95 * len(seqs))], seqs[int(0.95 * len(seqs)):]
    m = DEQLM30().to(DEV)
    log("DEQ-LM v5 (d320) 参数: %.2fM" % (sum(p.numel() for p in m.parameters()) / 1e6))
    opt = optim.AdamW(m.parameters(), lr=4e-4, weight_decay=0.01)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, STEPS, eta_min=4e-5)

    t0 = time.time()
    rng = random.Random(42)
    for st in range(STEPS):
        m.train()
        x, y = get_batch(train, bs=B, device=DEV, rng=rng)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, residual, k_used = m(x)
        loss = F.cross_entropy(logits.float().reshape(-1, V), y.reshape(-1), ignore_index=0)
        if not torch.isfinite(loss):
            log("st%d non-finite -> skip" % st)
            opt.zero_grad(set_to_none=True)
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        sched.step()
        if st % 100 == 0:
            log("st%d loss=%.4f (%.0fms/step)" % (st, loss.item(), (time.time()-t0)/(st+1)*1000))

    m.eval()
    torch.save(m.state_dict(), CKPT)
    log("\n=== k 扫描 (d320) ===")
    results = dict(nll_by_k={})
    for k in [1, 2, 4, 8, 16, 30]:
        nll = eval_nll(m, val, exact_k=k)
        results["nll_by_k"][str(k)] = round(nll, 4)
        log("  k=%2d: NLL=%.4f" % (k, nll))

    # 残差探针
    import numpy as np
    residuals, nlls = [], []
    with torch.no_grad():
        for i in range(100):
            x, y = get_batch(val, bs=1, device=DEV, rng=random.Random(7000 + i))
            out, res, _ = m(x)
            mask = y != 0
            if mask.sum() < 8:
                continue
            l = F.cross_entropy(out.float().reshape(-1, V), y.reshape(-1), reduction="none").reshape(y.shape)
            nlls.append(l[mask].mean().item()); residuals.append(res)
    r, n_ = np.array(residuals), np.array(nlls)
    corr = float(np.corrcoef(r, n_)[0, 1]) if r.std() > 0 else 0.0
    log("残差-NLL 相关: %.3f" % corr)
    results["probe"] = dict(corr=round(corr, 3), res_min=round(float(r.min()), 4), res_max=round(float(r.max()), 4))
    json.dump(results, open(OUT, "w"), indent=2)
    log("结果已存 " + OUT)


if __name__ == "__main__":
    main()
