#!/usr/bin/env python3
"""v25: 旋转消融 — v24 的桥换成纯 GLA (G:=I, 保留数据依赖λ_t+短卷积).
裁决旋转 G_t 的边际价值:
  v25 ≈ 2.424 (v24) → 旋转不值 2.4x 速度, 换 GLA
  v25 明显差       → 旋转有货, 保留 rubik
其余全部同 v24 (结构/步数/数据/评测)."""
import sys, os, time, math, json, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deqlm_v19_tuned23m import RMSNorm, SwiGLUFFN, TunedBlock, masked_suffix_eval
from deqlm_v5_30m import get_batch, PT_CACHE
from scan import affine_scan

DEV = "cuda"
V, D, H = 23005, 320, 10
KV_H = 2
SEQ = 256
PT_STEPS = 39695
WARMUP = 1000
KS = [10, 10, 10]


HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "deqlm25.log")
OUT = os.path.join(HERE, "deqlm25_results.json")
CKPT = os.path.join(HERE, "deqlm25_pt_full.pth")


def log(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


class GLABridgeAttn(nn.Module):
    """纯 GLA 桥: S_t = diag(lam_t) S_{t-1} + k_t v_t^T (无旋转).
    用同一套 HS 扫描 (A=diag(lam) 稠密化, 为公平同口径; 专用对角核是后续工程)."""

    def __init__(self):
        super().__init__()
        self.ln_in = RMSNorm(D)
        self.q = nn.Linear(D, D, bias=False)
        self.k = nn.Linear(D, D, bias=False)
        self.v = nn.Linear(D, D, bias=False)
        self.w_lam = nn.Linear(D, H * (D // H), bias=True)
        nn.init.zeros_(self.w_lam.weight)
        nn.init.constant_(self.w_lam.bias, -1.0)
        self.dwconv = nn.Conv1d(D, D, 4, groups=D, bias=False)
        nn.init.zeros_(self.dwconv.weight)
        self.out = nn.Linear(D, D, bias=False)
        nn.init.zeros_(self.out.weight)

    def _scan_core(self, x):
        B, L, _ = x.shape
        dh = D // H
        h = self.ln_in(x).float()
        q = self.q(h).view(B, L, H, dh)
        k = self.k(h).view(B, L, H, dh)
        v = self.v(h).view(B, L, H, dh)
        v_flat = v.reshape(B, L, H * dh).transpose(1, 2)
        v_conv = F.conv1d(F.pad(v_flat, (3, 0)), self.dwconv.weight, groups=D)
        v = (v_flat + v_conv).transpose(1, 2).reshape(B, L, H, dh)
        lam = torch.sigmoid(self.w_lam(h)).view(B, L, H, dh)
        # GLA: A_t = diag(lam_t) 稠密化
        eye = torch.eye(dh, device=x.device).view(1, 1, 1, dh, dh)
        Ag = (lam.unsqueeze(-1) * eye).to(torch.bfloat16)            # (B,L,H,dh,dh) 对角
        W_ = (k.unsqueeze(-1) * v.unsqueeze(-2)).to(torch.bfloat16)
        _, S = affine_scan(Ag, W_)
        o = torch.einsum("nlhp,nlhpq->nlhq", q.to(torch.bfloat16), S) / math.sqrt(dh)
        return o.float().reshape(B, L, H * dh)

    def forward(self, x):
        if self.training:
            out = checkpoint(self._scan_core, x, use_reentrant=False)
        else:
            with torch.no_grad():
                out = self._scan_core(x)
        if torch.is_autocast_enabled():
            out = out.to(torch.bfloat16)
        return x + self.out(out)


class GLAFFNBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = GLABridgeAttn()
        self.ffn = SwiGLUFFN()

    def forward(self, x):
        return self.ffn(self.attn(x))


class TiedBlock(nn.Module):
    def __init__(self, kmax, alpha_init=0.1):
        super().__init__()
        self.kmax = kmax
        self.attn = TunedBlock()
        self.alpha_param = nn.Parameter(torch.tensor(math.log(alpha_init / (1 - alpha_init))))

    def alpha(self):
        return torch.clamp(torch.sigmoid(self.alpha_param), 0.01, 0.9)

    def forward(self, z0):
        alpha = self.alpha()
        z = z0
        for _ in range(self.kmax):
            z = z + alpha * self.attn(z)
        return z


class GLAHybridLM(nn.Module):
    """[pre:sm][A:sm×10][GLA×1][C:sm×10][post:sm]"""

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.pre = TunedBlock()
        self.loop_a = TiedBlock(KS[0])
        self.gla_bridge = GLAFFNBlock()
        self.loop_c = TiedBlock(KS[2])
        self.post = TunedBlock()
        self.ln_f = RMSNorm(D)
        self.head = nn.Linear(D, V, bias=False)
        nn.init.normal_(self.emb.weight, std=0.02)
        nn.init.normal_(self.head.weight, std=0.02)
        for m in [self.pre, self.post, self.loop_a, self.loop_c]:
            for p in m.parameters():
                if p.dim() > 1:
                    nn.init.normal_(p, std=0.02)

    def forward(self, ids):
        h = self.emb(ids)
        h = self.pre(h)
        h = self.loop_a(h)
        h = self.gla_bridge(h)
        h = self.loop_c(h)
        h = self.post(h)
        return self.head(self.ln_f(h))


def main():
    torch.manual_seed(0)
    random.seed(0)
    seqs = torch.load(PT_CACHE, map_location="cpu", weights_only=True)
    train, val = seqs[:int(0.95 * len(seqs))], seqs[:5000]
    m = GLAHybridLM().to(DEV)
    log("v25: %.2fM, 纯GLA桥 (无旋转), %d 步 (裁决: v24=2.4241, v20=2.4284)"
        % (sum(p.numel() for p in m.parameters()) / 1e6, PT_STEPS))
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
    log("v25 完成: 含pad=%.4f 真实=%.4f  [v24(带旋转)=2.4241, v20=2.4284]" % (a, mk))
    json.dump(dict(all=a, masked=mk,
                   alphas=[round(m.loop_a.alpha().item(), 4), round(m.loop_c.alpha().item(), 4)]),
              open(OUT, "w"), indent=2)


if __name__ == "__main__":
    main()
