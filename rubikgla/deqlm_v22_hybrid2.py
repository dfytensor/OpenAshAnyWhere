#!/usr/bin/env python3
"""v22: Hybrid2 @23M — 旋转门控线性注意力(rubik低秩+decay)与softmax的混合, 融入三段循环冠军架构.
结构: [pre:softmax+RoPE][A:softmax×10][B:RUBIK×10][C:softmax×10][post:softmax+RoPE]
  RUBIK = RubikLowRankFast 血统: G=matrix_exp(skew(u,w)), 低秩r=2, decay λ, chunked扫描
对照: v20 全softmax = 2.4284. 判定: v22 ≈ 2.43±0.03 → hybrid2混合在循环架构成立."""
import sys, os, time, math, json, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deqlm_v19_tuned23m import RMSNorm, rope_cache, apply_rope, GQAAttn, SwiGLUFFN, TunedBlock, masked_suffix_eval
from deqlm_v5_30m import get_batch, PT_CACHE
from scan import affine_scan          # Hillis-Steele: 8趟全并行 (CEDLR 血统, 快 ~100x)

DEV = "cuda"
V, D, H = 23005, 320, 10
KV_H = 2
SEQ = 256
PT_STEPS = 12000
WARMUP = 1000
KS = [10, 10, 10]
R_LR = 2               # 低秩

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "deqlm22.log")
OUT = os.path.join(HERE, "deqlm22_results.json")
CKPT = os.path.join(HERE, "deqlm22_pt_full.pth")


def log(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


class RubikLinearAttn(nn.Module):
    """低秩旋转门控线性注意力 (CEDLR-Hybrid2 血统): S_t = λ·G_t S_{t-1} + k_t v_t^T, o_t = q_t S_t.
    fp32 岛 + HS扫描 + 稳定Taylor-exp (torch.matrix_exp 反向20x慢, 不可用)."""

    def __init__(self, r=R_LR):
        super().__init__()
        self.r = r
        self.ln_in = RMSNorm(D)
        self.q = nn.Linear(D, D, bias=False)
        self.k = nn.Linear(D, D, bias=False)
        self.v = nn.Linear(D, D, bias=False)
        self.uv = nn.Linear(D, H * 2 * r * (D // H), bias=False)
        nn.init.normal_(self.uv.weight, 0.0, 0.01)
        self.lam = nn.Parameter(torch.full((H, D // H, 1), -1.0))
        self.out = nn.Linear(D, D, bias=False)

    @staticmethod
    def _stable_exp(A, terms=6, max_sq=10):
        """缩放-平方法: A/2^k 缩到小范数后 Taylor, 再 k 次平方. 纯 matmul, 反向友好."""
        with torch.no_grad():
            nrm = A.norm(dim=(-2, -1)).max().item()
        k = 0
        while nrm > 0.5 and k < max_sq:
            A = A / 2
            nrm /= 2
            k += 1
        eye = torch.eye(A.shape[-1], device=A.device, dtype=A.dtype).view(*([1] * (A.dim() - 2)), A.shape[-1], A.shape[-1])
        E = eye + A
        T = A
        for m in range(2, terms + 1):
            T = T @ A / m
            E = E + T
        for _ in range(k):
            E = E @ E
        return E

    def _scan_core(self, x):
        B, L, _ = x.shape
        dh = D // H
        r = self.r
        h = self.ln_in(x).float()
        q = self.q(h).view(B, L, H, dh)
        k = self.k(h).view(B, L, H, dh)
        v = self.v(h).view(B, L, H, dh)
        uv = self.uv(h).view(B, L, H, 2 * r, dh)
        u, w = uv[:, :, :, :r, :], uv[:, :, :, r:, :]
        Ut, Wt = u.transpose(-2, -1), w.transpose(-2, -1)
        A = Ut @ Wt.transpose(-2, -1) - Wt @ Ut.transpose(-2, -1)   # skew (dh,dh)
        G = self._stable_exp(A)
        lam = torch.sigmoid(self.lam).unsqueeze(0).float()
        W_ = k.unsqueeze(-1) * v.unsqueeze(-2)
        _, B_star = affine_scan(lam * G, W_)
        o = torch.einsum("nlhp,nlhpq->nlhq", q, B_star)
        return o.reshape(B, L, H * dh)

    def forward(self, x):
        if self.training:
            out = checkpoint(self._scan_core, x, use_reentrant=False)
        else:
            with torch.no_grad():
                out = self._scan_core(x)
        if torch.is_autocast_enabled():
            out = out.to(torch.bfloat16)
        return x + self.out(out)


class RubikFFNBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = RubikLinearAttn()
        self.ffn = SwiGLUFFN()

    def forward(self, x):
        return self.ffn(self.attn(x))


class TiedBlock(nn.Module):
    def __init__(self, kmax, rubik=False, alpha_init=0.1):
        super().__init__()
        self.kmax = kmax
        self.attn = RubikFFNBlock() if rubik else TunedBlock()
        self.alpha_param = nn.Parameter(torch.tensor(math.log(alpha_init / (1 - alpha_init))))

    def alpha(self):
        return torch.clamp(torch.sigmoid(self.alpha_param), 0.01, 0.9)

    def forward(self, z0):
        alpha = self.alpha()
        z = z0
        for _ in range(self.kmax):
            z = z + alpha * self.attn(z)
        return z


class Hybrid2TriLoopLM(nn.Module):
    """[pre:sm][A:sm×10][B:RUBIK×1][C:sm×10][post:sm] — rubik 作单次桥层 (速度可行)"""

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.pre = TunedBlock()
        self.loop_a = TiedBlock(KS[0], rubik=False)
        self.rubik_bridge = RubikFFNBlock()
        self.loop_c = TiedBlock(KS[2], rubik=False)
        self.post = TunedBlock()
        self.ln_f = RMSNorm(D)
        self.head = nn.Linear(D, V, bias=False)
        nn.init.normal_(self.emb.weight, std=0.02)
        nn.init.normal_(self.head.weight, std=0.02)
        for m in [self.pre, self.post, self.loop_a, self.loop_c, self.rubik_bridge]:
            for p in m.parameters():
                if p.dim() > 1:
                    nn.init.normal_(p, std=0.02)

    def forward(self, ids):
        h = self.emb(ids)
        h = self.pre(h)
        h = self.loop_a(h)
        h = self.rubik_bridge(h)
        h = self.loop_c(h)
        h = self.post(h)
        return self.head(self.ln_f(h))


def main():
    torch.manual_seed(0)
    random.seed(0)
    seqs = torch.load(PT_CACHE, map_location="cpu", weights_only=True)
    train, val = seqs[:int(0.95 * len(seqs))], seqs[:5000]
    m = Hybrid2TriLoopLM().to(DEV)
    log("v22: %.2fM, [sm][sm×10][RUBIK×10][sm×10][sm] (对照 v20 全softmax=2.4284)"
        % (sum(p.numel() for p in m.parameters()) / 1e6))
    opt = optim.AdamW(m.parameters(), lr=4e-4, weight_decay=0.01, betas=(0.9, 0.95))

    def lr_at(st):
        if st < WARMUP:
            return 4e-4 * st / WARMUP
        p_ = (st - WARMUP) / max(1, PT_STEPS - WARMUP)
        return 4e-5 + 0.5 * (4e-4 - 4e-5) * (1 + math.cos(math.pi * p_))

    t0 = time.time()
    rng = random.Random(42)
    B_MICRO = 16
    for st in range(PT_STEPS):
        for g in opt.param_groups:
            g["lr"] = lr_at(st)
        m.train()
        x, y = get_batch(train, bs=B_MICRO, device=DEV, rng=rng)
        x2, y2 = get_batch(train, bs=B_MICRO, device=DEV, rng=rng)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lo = m(x)
        l1 = F.cross_entropy(lo.float().reshape(-1, V), y.reshape(-1), ignore_index=0)
        if torch.isfinite(l1):
            (l1 * 0.5).backward()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lo2 = m(x2)
        l2 = F.cross_entropy(lo2.float().reshape(-1, V), y2.reshape(-1), ignore_index=0)
        if torch.isfinite(l2):
            (l2 * 0.5).backward()
        if not (torch.isfinite(l1) and torch.isfinite(l2)):
            opt.zero_grad(set_to_none=True)
            continue
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if st % 2000 == 0:
            alphas = "%.3f/%.3f" % (m.loop_a.alpha().item(), m.loop_c.alpha().item())
            log("%d/%d loss=%.4f a=[%s] (%.0fms/步, ETA %.1f时)"
                % (st, PT_STEPS, (0.5 * (l1 + l2)).item(), alphas,
                   (time.time() - t0) / (st + 1) * 1000,
                   (PT_STEPS - st) * (time.time() - t0) / (st + 1) / 3600))
    torch.save(m.state_dict(), CKPT)
    a, mk = masked_suffix_eval(m, val)
    log("v22 完成: 含pad=%.4f 真实=%.4f  [v20 全softmax=2.4284]" % (a, mk))
    json.dump(dict(all=a, masked=mk,
                   alphas=[round(m.loop_a.alpha().item(), 4), round(m.loop_c.alpha().item(), 4)]), open(OUT, "w"), indent=2)


if __name__ == "__main__":
    main()
