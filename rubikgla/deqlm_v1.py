#!/usr/bin/env python3
"""DEQ-LM Phase 1 原型: 三明治架构 (标准层→DEQ块→标准层), minimind 数据.
验证:
  1. DEQ 块能收敛 (残差范数下降)
  2. minimind PT loss 能下降
  3. 残差范数预测正确性 (与之前 3.2x 对照)
  4. 自适应计算 (不同输入收敛步数不同)"""
import sys, os, time, math, json, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
DEV = "cuda"
V, D, H = 23005, 128, 4
N_BOTTOM, N_DEQ, N_TOP = 2, 2, 2     # 三明治: 2+2+2 = 6 层
KMAX, ALPHA, TOL = 30, 0.5, 1e-4
B, SEQ = 32, 256
PT_CACHE = r"F:\OpenASH2605\minimind_data\pretrain_cached_1270238_256.pt"
LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deqlm.log")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deqlm_results.json")

def log(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


# ── 标准层 ──
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


# ── DEQ 层 (权重绑定, 迭代到不动点) ──
class DEQBlock(nn.Module):
    def __init__(self, d=D, H=H, alpha=0.5, kmax=KMAX, tol=TOL):
        super().__init__()
        self.H, self.dh = H, d // H
        self.alpha = alpha
        self.kmax = kmax
        self.tol = tol
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.attn_out = nn.Linear(d, d)
        self.gate = nn.Linear(d, 4 * d)
        self.up = nn.Linear(d, 4 * d)
        self.down = nn.Linear(4 * d, d)
        self.alpha_param = nn.Parameter(torch.tensor(alpha))
        self.ln_out = nn.LayerNorm(d)

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

    def forward(self, z0):
        """迭代到不动点, 返回 (z*, 残差范数, 迭代次数)."""
        alpha = torch.sigmoid(self.alpha_param)
        z = z0
        residual = float("inf")
        k_used = self.kmax
        for k in range(self.kmax):
            z_new = z + alpha * self._block_fn(z)
            residual = (z_new - z).norm(dim=-1).mean(-1).max().item()
            z = z_new
            if residual < self.tol:
                k_used = k + 1
                break
        return z, residual, k_used


# ── 完整模型 ──
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
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, ids, kmax=None):
        B, L = ids.shape
        h = self.emb(ids) + self.pos(torch.arange(L, device=ids.device)).unsqueeze(0)
        h = self.ln_emb(h)
        for layer in self.bottom:
            h = layer(h)
        h_deq, residual, k_used = self.deq(h)
        for layer in self.top:
            h = layer(h)
        return self.head(self.ln_f(h)), residual, k_used


# ── 数据 ──
def get_batch(data, bs=B, sl=SEQ, device=DEV, rng=None):
    """data: 变长 1-D tensor 列表. 采样 bs 条, pad 到 sl, 目标 = 右移一位."""
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
def eval_model(m, val, nb=20, kmax=KMAX):
    m.eval()
    tot, tok = 0.0, 0
    for i in range(nb):
        x, y = get_batch(val, bs=B, device=DEV, rng=random.Random(1000 + i))
        out, _, _ = m(x, kmax=kmax)
        lo = out.float()
        l = F.cross_entropy(lo.reshape(-1, V), y.reshape(-1), reduction="sum")
        tot += l.item(); tok += y.numel()
    m.train()
    return tot / tok


@torch.no_grad()
def eval_at_k(m, val, k, seed=999, nb=10):
    m.eval()
    tot, tok = 0.0, 0
    for i in range(nb):
        x, y = get_batch(val, bs=B, device=DEV, rng=random.Random(seed + i))
        out, _, _ = m(x, kmax=k)
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
    log("DEQ-LM 参数: %.2fM" % (n / 1e6))

    opt = optim.AdamW(m.parameters(), lr=6e-4, weight_decay=0.01)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, 3000, eta_min=6e-5)
    log("训练开始 (标准 Transformer 层 + DEQ 块 + Phantom Gradient)")

    results = dict(losses=[], k_dists=[], residual_norms=[])
    t0 = time.time()
    train_rng = random.Random(42)
    for st in range(3000):
        m.train()
        x, y = get_batch(train, bs=B, device=DEV, rng=train_rng)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, residual, k_used = m(x)
        logits = logits.float()
        loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1), ignore_index=0)

        if not torch.isfinite(loss):
            log("st%d non-finite → skip" % st)
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
            # 记录 DEQ 迭代次数分布
            m.eval()
            with torch.no_grad():
                xs, _ = get_batch(train, bs=1, device=DEV, rng=random.Random(100 + st))
                _, residual, k_used = m(xs, kmax=KMAX)
                results["k_dists"].append((st, k_used, round(residual, 4)))
                log("  DEQ: residual=%.4f k=%d" % (residual, k_used))

    # ═══ 评估 ═══
    log("\n=== 评估 ===")
    m.eval()

    # 1. 不同 k 值的 val NLL
    log("不同 DEQ 迭代次数的 val NLL:")
    for k in [1, 2, 4, 8, 16, 30]:
        nll = eval_at_k(m, val, k=k)
        log("  k=%2d: NLL=%.4f" % (k, nll))

    # 2. 收敛探针: 逐样本残差范数 vs NLL
    log("\n收敛探针:")
    m.eval()
    with torch.no_grad():
        residuals = []
        nlls = []
        for i in range(50):
            x, y = get_batch(val, bs=1, device=DEV, rng=random.Random(7000 + i))
            out, res, ku = m(x, kmax=KMAX)
            lo = out.float()
            nll_i = F.cross_entropy(lo.reshape(-1, V), y.reshape(-1)).item()
            residuals.append(res)
            nlls.append(nll_i)
    import numpy as np
    res_np = np.array(residuals)
    nll_np = np.array(nlls)
    corr = np.corrcoef(res_np, nll_np)[0, 1] if res_np.std() > 0 else 0
    log("残差-NLL 相关系数: %.3f (残差大→NLL高→预测差)" % corr)

    # 3. 最终评估
    m.eval()
    nll_final = eval_model(m, val, nb=20, kmax=KMAX)
    log("最终 val NLL: %.4f" % nll_final)

    results["final"] = dict(nll=round(nll_final, 4), corr=round(corr, 3))
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    log("结果已存 " + OUT)


if __name__ == "__main__":
    main()
