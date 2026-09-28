#!/usr/bin/env python3
"""DEQ-LM v2: 修复 v1 的两个问题.
  1) 收敛性: DEQ 块线性层加 spectral_norm, alpha 降到 0.1, DEQ 参数 lr x0.1
  2) 评测: kmax 正确传入 DEQ 块, k 扫描有效
  3) 相对残差 (||dz||/||z||) 替代绝对残差"""
import sys, os, time, math, json, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.parametrizations import spectral_norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
DEV = "cuda"
V, D, H = 23005, 128, 4
N_BOTTOM, N_TOP = 2, 2
KMAX, ALPHA, TOL = 30, 0.1, 1e-3
B, SEQ = 32, 256
STEPS = 3000
PT_CACHE = r"F:\OpenASH2605\minimind_data\pretrain_cached_1270238_256.pt"
LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deqlm2.log")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deqlm2_results.json")

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
        x = self.attn(x)
        x = self.ffn(x)
        return x


class DEQBlock(nn.Module):
    """权重绑定块, 迭代 z -> z + a*Block(z) 到不动点. 线性层谱范数=1 促收缩."""

    def __init__(self, d=D, H=H, alpha=ALPHA, kmax=KMAX, tol=TOL):
        super().__init__()
        self.d, self.H, self.dh = d, H, d // H
        self.alpha_init, self.kmax, self.tol = alpha, kmax, tol
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.qkv = spectral_norm(nn.Linear(d, 3 * d))
        self.attn_out = spectral_norm(nn.Linear(d, d))
        self.gate = spectral_norm(nn.Linear(d, 4 * d))
        self.up = spectral_norm(nn.Linear(d, 4 * d))
        self.down = spectral_norm(nn.Linear(4 * d, d))
        self.alpha_param = nn.Parameter(torch.tensor(math.log(alpha / (1 - alpha))))

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

    def forward(self, z0, kmax=None, tol=None, exact_k=None):
        """exact_k: 评测用, 强制恰好迭代 k 次 (不早退)."""
        alpha = torch.clamp(torch.sigmoid(self.alpha_param), 0.01, 0.5)
        km = self.kmax if kmax is None else kmax
        tol = self.tol if tol is None else tol
        z = z0
        znorm = z0.norm(dim=-1).clamp(min=1e-6)
        residual = float("inf")
        k_used = km
        n_iter = exact_k if exact_k is not None else km
        for k in range(n_iter):
            z_new = z + alpha * self._block_fn(z)
            residual = ((z_new - z).norm(dim=-1) / znorm).max().item()
            z = z_new
            if exact_k is None and residual < tol:
                k_used = k + 1
                break
        if exact_k is not None:
            k_used = exact_k
        return z, residual, k_used


class DEQLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.pos = nn.Embedding(SEQ + 1, D)
        self.ln_emb = nn.LayerNorm(D)
        self.bottom = nn.ModuleList([StdBlock() for _ in range(N_BOTTOM)])
        self.deq = DEQBlock()
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

    def forward(self, ids, kmax=None, exact_k=None):
        B, L = ids.shape
        h = self.bottom_h(ids)
        h, residual, k_used = self.deq(h, kmax=kmax, exact_k=exact_k)
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
def eval_nll(m, val, kmax=None, exact_k=None, nb=20, seed=999):
    m.eval()
    tot, tok = 0.0, 0
    for i in range(nb):
        x, y = get_batch(val, bs=B, device=DEV, rng=random.Random(seed + i))
        out, _, _ = m(x, kmax=kmax, exact_k=exact_k)
        lo = out.float()
        l = F.cross_entropy(lo.reshape(-1, V), y.reshape(-1), reduction="sum")
        tot += l.item(); tok += y.numel()
    m.train()
    return tot / tok


def main():
    torch.manual_seed(0)
    random.seed(0)
    seqs = torch.load(PT_CACHE, map_location="cpu", weights_only=True)
    train, val = seqs[:int(0.95 * len(seqs))], seqs[int(0.95 * len(seqs)):]
    log("train %d val %d" % (len(train), len(val)))

    m = DEQLM().to(DEV)
    n = sum(p.numel() for p in m.parameters())
    log("DEQ-LM v2 参数: %.2fM" % (n / 1e6))

    # DEQ 参数 lr x0.1
    deq_ids = set(id(p) for p in m.deq.parameters())
    deq_params = [p for p in m.parameters() if id(p) in deq_ids]
    other_params = [p for p in m.parameters() if id(p) not in deq_ids]
    opt = optim.AdamW([
        dict(params=other_params, lr=6e-4, weight_decay=0.01),
        dict(params=deq_params, lr=6e-5, weight_decay=0.01),
    ])
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, STEPS, eta_min=6e-5)
    log("训练开始: spectral_norm + alpha~0.1 + DEQ lr x0.1")

    results = dict(losses=[], k_dists=[])
    t0 = time.time()
    train_rng = random.Random(42)
    for st in range(STEPS):
        m.train()
        x, y = get_batch(train, bs=B, device=DEV, rng=train_rng)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, residual, k_used = m(x)
        logits = logits.float()
        loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1), ignore_index=0)
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
            el = time.time() - t0
            log("st%d loss=%.4f (%.0fs, %.0fms/step)" % (st, loss.item(), el, el / (st + 1) * 1000))
        if st % 500 == 0:
            results["losses"].append((st, round(loss.item(), 4)))
            m.eval()
            with torch.no_grad():
                xs, _ = get_batch(train, bs=1, device=DEV, rng=random.Random(100 + st))
                _, residual, k_used = m.deq(m.bottom_h(xs))
                results["k_dists"].append((st, k_used, round(residual, 4)))
                log("  DEQ: rel_residual=%.4f k=%d" % (residual, k_used))

    # ═══ 评估 ═══
    log("\n=== 评估 ===")
    m.eval()
    log("强制恰好 k 次迭代的 val NLL:")
    nlls_by_k = {}
    for k in [1, 2, 4, 8, 16, 30]:
        nll = eval_nll(m, val, exact_k=k)
        nlls_by_k[k] = round(nll, 4)
        log("  k=%2d: NLL=%.4f" % (k, nll))
    results["nll_by_k"] = nlls_by_k

    nll_final = eval_nll(m, val, kmax=KMAX)
    log("自适应早退 val NLL: %.4f" % nll_final)
    results["nll_adaptive"] = round(nll_final, 4)

    # 收敛探针: 逐样本相对残差 vs NLL
    m.eval()
    residuals, nlls = [], []
    with torch.no_grad():
        for i in range(100):
            x, y = get_batch(val, bs=1, device=DEV, rng=random.Random(7000 + i))
            out, res, ku = m(x, kmax=KMAX)
            nll_i = F.cross_entropy(out.float().reshape(-1, V), y.reshape(-1)).item()
            residuals.append(res); nlls.append(nll_i)
    import numpy as np
    res_np, nll_np = np.array(residuals), np.array(nlls)
    corr = float(np.corrcoef(res_np, nll_np)[0, 1]) if res_np.std() > 0 else 0.0
    log("残差-NLL 相关 (100 样本): %.3f  (残差范围 %.4f~%.4f)" % (corr, res_np.min(), res_np.max()))
    results["probe"] = dict(corr=round(corr, 3), res_min=round(float(res_np.min()), 4), res_max=round(float(res_np.max()), 4))

    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    log("结果已存 " + OUT)


if __name__ == "__main__":
    main()
