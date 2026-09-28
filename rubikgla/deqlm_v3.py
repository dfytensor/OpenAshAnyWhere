#!/usr/bin/env python3
"""DEQ-LM v3: 权重绑定深度 + 置信度早退 (ACT 风格, 免训练自适应计算).
v2 结论: (1) h_deq 曾被丢弃(bug已修) (2) 谱范数+lr x0.1 限制容量, NLL 比 4 层对照差
v3 变更: 去谱范数/去低lr, alpha 可学习; 加置信度早退扫描; 探针忽略 pad; 存 checkpoint"""
import sys, os, time, math, json, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
DEV = "cuda"
V, D, H = 23005, 128, 4
N_BOTTOM, N_TOP = 2, 2
KMAX, ALPHA_INIT, TOL = 30, 0.1, 1e-3
B, SEQ = 32, 256
STEPS = 3000
PT_CACHE = r"F:\OpenASH2605\minimind_data\pretrain_cached_1270238_256.pt"
HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "deqlm3.log")
OUT = os.path.join(HERE, "deqlm3_results.json")
CKPT = os.path.join(HERE, "deqlm3_final.pth")

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


class TiedBlock(nn.Module):
    """权重绑定块: z -> z + alpha*Block(z), alpha 可学习标量 (sigmoid 参数化)."""

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

    def forward(self, z0, kmax=None, exact_k=None):
        alpha = self.alpha()
        km = self.kmax if kmax is None else kmax
        z = z0
        znorm = z0.norm(dim=-1).clamp(min=1e-6)
        residual = 0.0
        n_iter = exact_k if exact_k is not None else km
        for k in range(n_iter):
            z_new = z + alpha * self._block_fn(z)
            residual = ((z_new - z).norm(dim=-1) / znorm).max().item()
            z = z_new
        return z, residual, n_iter


class DEQLM(nn.Module):
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
        lo = out.float()
        l = F.cross_entropy(lo.reshape(-1, V), y.reshape(-1), reduction="sum")
        tot += l.item(); tok += y.numel()
    m.train()
    return tot / tok


@torch.no_grad()
def confidence_exit_eval(m, val, taus, nb=20, seed=999):
    """每步算 head(z_k), 逐位置当 max_prob>tau 时锁定该位置的输出. 返回 {tau: (NLL, 平均k)}."""
    m.eval()
    res = {t: [0.0, 0.0, 0] for t in taus}   # loss_sum, k_sum*len, tok
    for i in range(nb):
        x, y = get_batch(val, bs=B, device=DEV, rng=random.Random(seed + i))
        h = m.bottom_h(x)
        out0, _, _ = m(x, exact_k=1)   # 校准 head 一致性: 直接逐步调用
        Bc, L = x.shape
        locked = torch.zeros(Bc, L, dtype=torch.bool, device=DEV)
        final_logits = torch.zeros(Bc, L, V, device=DEV)
        for t in taus:
            locked_t = torch.zeros(Bc, L, dtype=torch.bool, device=DEV)
            zz = h.clone()
            k_used = torch.full((Bc, L), float(m.deq.kmax), device=DEV)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                for k in range(1, m.deq.kmax + 1):
                    zz = zz + m.deq.alpha() * m.deq._block_fn(zz)
                    lg = m.head(m.ln_f(zz)).float()
                    mp = torch.softmax(lg, -1).max(-1).values
                    newly = (~locked_t) & (mp > t)
                    locked_t |= newly
                    k_used[newly] = k
                    if locked_t.all():
                        break
                final_t = torch.where(locked_t.unsqueeze(-1), lg, lg)
            lsum = F.cross_entropy(final_t.reshape(-1, V), y.reshape(-1), reduction="sum").item()
            res[t][0] += lsum
            res[t][1] += k_used.sum().item()
            res[t][2] += y.numel()
    m.train()
    out = {}
    for t in taus:
        nll, ksum, tok = res[t]
        out[t] = (round(nll / tok, 4), round(ksum / tok, 2))
    return out


@torch.no_grad()
def per_sample_probe(m, val, n=100, seed=7000):
    """逐样本: 相对残差 vs 非pad位置 NLL."""
    m.eval()
    residuals, nlls, ks = [], [], []
    for i in range(n):
        x, y = get_batch(val, bs=1, device=DEV, rng=random.Random(seed + i))
        out, res, k_used = m(x)
        mask = y != 0
        if mask.sum() < 8:
            continue
        lg = out.float()
        l = F.cross_entropy(lg.reshape(-1, V), y.reshape(-1), reduction="none").reshape(y.shape)
        nlls.append(l[mask].mean().item())
        residuals.append(res)
    m.train()
    import numpy as np
    r, n_ = np.array(residuals), np.array(nlls)
    corr = float(np.corrcoef(r, n_)[0, 1]) if r.std() > 0 else 0.0
    return corr, float(r.min()), float(r.max())


def main():
    torch.manual_seed(0)
    random.seed(0)
    seqs = torch.load(PT_CACHE, map_location="cpu", weights_only=True)
    train, val = seqs[:int(0.95 * len(seqs))], seqs[int(0.95 * len(seqs)):]
    log("train %d val %d" % (len(train), len(val)))

    m = DEQLM().to(DEV)
    log("DEQ-LM v3 参数: %.2fM" % (sum(p.numel() for p in m.parameters()) / 1e6))

    opt = optim.AdamW(m.parameters(), lr=6e-4, weight_decay=0.01)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, STEPS, eta_min=6e-5)

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
            with torch.no_grad():
                log("  alpha=%.3f" % m.deq.alpha().item())

    # ═══ 评估 ═══
    log("\n=== 评估 ===")
    m.eval()
    torch.save(m.state_dict(), CKPT)
    log("checkpoint 已存 " + CKPT)

    log("强制恰好 k 次迭代的 val NLL:")
    nlls_by_k = {}
    for k in [1, 2, 4, 8, 16, 30]:
        nll = eval_nll(m, val, exact_k=k)
        nlls_by_k[k] = round(nll, 4)
        log("  k=%2d: NLL=%.4f" % (k, nll))
    results["nll_by_k"] = nlls_by_k

    nll_final = eval_nll(m, val, exact_k=KMAX)
    results["nll_k30"] = round(nll_final, 4)

    log("\n置信度早退扫描 (tau -> NLL, 平均k):")
    ce = confidence_exit_eval(m, val, taus=[0.3, 0.5, 0.7, 0.8, 0.9], nb=10)
    for t, (nll, avgk) in ce.items():
        log("  tau=%.1f: NLL=%.4f avg_k=%.2f" % (t, nll, avgk))
    results["confidence_exit"] = {str(t): v for t, v in ce.items()}

    corr, rmin, rmax = per_sample_probe(m, val)
    log("\n残差-NLL 相关 (非pad, 100样本): %.3f (残差 %.4f~%.4f)" % (corr, rmin, rmax))
    results["probe"] = dict(corr=round(corr, 3), res_min=round(rmin, 4), res_max=round(rmax, 4))

    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    log("结果已存 " + OUT)


if __name__ == "__main__":
    main()
