#!/usr/bin/env python3
"""v27: 多尺度速率门 + 时标分层循环 — 修复 v26 视野死因 (诊断: v26 λmean=0.4, 256步保留率1e-14).
零 softmax (全GLA+rubik, 但 rubik 也换速率门).
改动 vs v26:
  1. 速率门: rate = softplus(W h + b), λ=exp(−rate) — 绕开 sigmoid 饱和, 可学到 λ≈0.999+
  2. 多尺度初始化: 32 通道 rate log 均匀 0.002~0.5 — 第0步即全局(0.95保留)到局部(1e-56)多档视野
  3. 时标分层循环: 第j次迭代 rate×ρ^j (ρ= sigmoid(p) 可学习, 每通道) — 浅迭代局部/深迭代全局,
     循环深度=时间尺度阶梯, 10次迭代=10层多分辨率阅读器 (loop 专属机制, softmax 做不到)
  4. gla_fast_neg: 直接吃 −rate, 省 exp/log 往返 (精度+速度)
对照: v24=2.4241, v26(全GLA, 视野死)=2.5506."""
import sys, os, time, math, json, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deqlm_v19_tuned23m import RMSNorm, SwiGLUFFN, masked_suffix_eval
from deqlm_v5_30m import get_batch, PT_CACHE
from deqlm_v24_hybrid2fast import RubikLinearAttnV2
from scan import affine_scan
from gla_fast import gla_fast

DEV = "cuda"
V, D, H = 23005, 320, 10
KV_H = 2
SEQ = 256
PT_STEPS = 39695
WARMUP = 1000
KS = [10, 10, 10]
R_LR = 4

# 多尺度速率初始化: 每通道 rate 目标 log 均匀 0.002~0.5, 每头复制一份 (共 H*dh 个)
def multiscale_inv_softplus():
    r = 10.0 ** torch.linspace(math.log10(0.002), math.log10(0.5), D // H)  # (dh,)
    r = r.repeat(H)                                                           # (H*dh,)
    return torch.log(torch.expm1(r))       # softplus(b)=r 的逆

def gla_fast_neg(k, v, negloglam, q):
    """同 gla_fast, 但直接吃 log(λ) = negloglam ≤ 0 — 省 exp/log 往返."""
    cl = torch.cumsum(negloglam, dim=1)
    diff = cl.unsqueeze(2) - cl.unsqueeze(1)
    M = torch.exp(diff.clamp(max=0.0))
    L_ = k.shape[1]
    mask = torch.tril(torch.ones(L_, L_, device=k.device, dtype=torch.bool))
    M = M * mask[None, :, :, None, None]
    u = (M * q.unsqueeze(2) * k.unsqueeze(1)).sum(-1)
    return torch.einsum("btih,bihc->bthc", u, v)

GLA_FN = torch.compile(gla_fast_neg)

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "deqlm27.log")
OUT = os.path.join(HERE, "deqlm27_results.json")
CKPT = os.path.join(HERE, "deqlm27_pt_full.pth")


def log(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


class RubikRateAttn(nn.Module):
    """v24 RubikLinearAttnV2 + 速率门 (w_lam sigmoid -> softplus 多尺度). 旋转/短卷积/bf16扫描全保留."""

    def __init__(self, r=R_LR):
        super().__init__()
        self.r = r
        self.ln_in = RMSNorm(D)
        self.q = nn.Linear(D, D, bias=False)
        self.k = nn.Linear(D, D, bias=False)
        self.v = nn.Linear(D, D, bias=False)
        self.uv = nn.Linear(D, H * 2 * r * (D // H), bias=False)
        nn.init.normal_(self.uv.weight, 0.0, 0.01)
        self.w_rate = nn.Linear(D, H * (D // H), bias=True)
        nn.init.zeros_(self.w_rate.weight)
        with torch.no_grad():
            self.w_rate.bias.copy_(multiscale_inv_softplus().view(H, -1).reshape(-1))
        self.dwconv = nn.Conv1d(D, D, 4, groups=D, bias=False)
        nn.init.zeros_(self.dwconv.weight)
        self.out = nn.Linear(D, D, bias=False)
        nn.init.zeros_(self.out.weight)

    @staticmethod
    def _stable_exp(A, terms=6, max_sq=10):
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
        v_flat = v.reshape(B, L, H * dh).transpose(1, 2)
        v_conv = F.conv1d(F.pad(v_flat, (3, 0)), self.dwconv.weight, groups=D)
        v = (v_flat + v_conv).transpose(1, 2).reshape(B, L, H, dh)
        rate = F.softplus(self.w_rate(h)).view(B, L, H, dh)          # ≥0, 多尺度初始化
        lam = torch.exp(-rate)                                        # λ ∈ (0,1), 可到 0.999+
        uv = self.uv(h).view(B, L, H, 2 * r, dh)
        u, w = uv[:, :, :, :r, :], uv[:, :, :, r:, :]
        Ut, Wt = u.transpose(-2, -1), w.transpose(-2, -1)
        A = Ut @ Wt.transpose(-2, -1) - Wt @ Ut.transpose(-2, -1)
        G = self._stable_exp(A)
        Ag = (lam.unsqueeze(-1) * G).to(torch.bfloat16)
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


class GLAAttnV2(nn.Module):
    """速率门 GLA 站 + 时标分层 (第j次迭代 rate×ρ^j)."""

    def __init__(self):
        super().__init__()
        self.H, self.KV, self.dh = H, KV_H, D // H
        self.ln = RMSNorm(D)
        self.wq = nn.Linear(D, D, bias=False)
        self.wk = nn.Linear(D, KV_H * self.dh, bias=False)
        self.wv = nn.Linear(D, KV_H * self.dh, bias=False)
        self.out = nn.Linear(D, D, bias=False)
        nn.init.zeros_(self.out.weight)
        self.w_rate = nn.Linear(D, H * self.dh, bias=True)
        nn.init.zeros_(self.w_rate.weight)
        with torch.no_grad():
            self.w_rate.bias.copy_(multiscale_inv_softplus().view(H, -1).reshape(-1))
        self.dwconv = nn.Conv1d(D, D, 4, groups=D, bias=False)
        nn.init.zeros_(self.dwconv.weight)
        # 时标阶梯: ρ = sigmoid(p), 每通道, 初始 0.8
        self.rho_p = nn.Parameter(torch.full((H, self.dh), math.log(0.8 / 0.2)))

    def _scan_core(self, x, it):
        B, L, _ = x.shape
        h = self.ln(x)
        q = self.wq(h).view(B, L, self.H, self.dh)
        k = self.wk(h).view(B, L, self.KV, self.dh).repeat_interleave(self.H // self.KV, dim=2)
        v = self.wv(h).view(B, L, self.KV, self.dh).repeat_interleave(self.H // self.KV, dim=2)
        v_flat = v.reshape(B, L, D).transpose(1, 2)
        v_conv = F.conv1d(F.pad(v_flat, (3, 0)), self.dwconv.weight, groups=D)
        v = (v_flat + v_conv).transpose(1, 2).reshape(B, L, H, self.dh)
        rate = F.softplus(self.w_rate(h)).view(B, L, H, self.dh)      # ≥0 多尺度
        if it > 0:
            rho = torch.sigmoid(self.rho_p)                            # (H,dh) in (0,1)
            rate = rate * (rho ** it)                                  # 深迭代=慢时标=大视野
        o = GLA_FN(k, v, -rate, q) / math.sqrt(self.dh)
        return o.reshape(B, L, D)

    def forward(self, x, it=0):
        if self.training:
            out = checkpoint(self._scan_core, x, it, use_reentrant=False)
        else:
            with torch.no_grad():
                out = self._scan_core(x, it)
        if torch.is_autocast_enabled():
            out = out.to(torch.bfloat16)
        return x + self.out(out)


class GLABlockV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = GLAAttnV2()
        self.ffn = SwiGLUFFN()

    def forward(self, x, it=0):
        return self.ffn(self.attn(x, it))


class TiedGLABlockV2(nn.Module):
    """z += α·block(z, it=j) — 每次迭代一个时标档位."""

    def __init__(self, kmax, alpha_init=0.1):
        super().__init__()
        self.kmax = kmax
        self.attn = GLABlockV2()
        self.alpha_param = nn.Parameter(torch.tensor(math.log(alpha_init / (1 - alpha_init))))

    def alpha(self):
        return torch.clamp(torch.sigmoid(self.alpha_param), 0.01, 0.9)

    def forward(self, z0):
        alpha = self.alpha()
        z = z0
        for j in range(self.kmax):
            z = z + alpha * self.attn(z, it=j)
        return z


class RubikFFNBlockV2Wrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = RubikRateAttn()
        self.ffn = SwiGLUFFN()

    def forward(self, x):
        return self.ffn(self.attn(x))


class GLAHybrid2LM(nn.Module):
    """[pre:GLA][A:GLA分层×10][RUBIK-rate×1][C:GLA分层×10][post:GLA] — 零softmax, 全局视野."""

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.pre = GLABlockV2()
        self.loop_a = TiedGLABlockV2(KS[0])
        self.rubik_bridge = RubikFFNBlockV2Wrapper()
        self.loop_c = TiedGLABlockV2(KS[2])
        self.post = GLABlockV2()
        self.ln_f = RMSNorm(D)
        self.head = nn.Linear(D, V, bias=False)
        nn.init.normal_(self.emb.weight, std=0.02)
        nn.init.normal_(self.head.weight, std=0.02)
        for blk in (self.pre, self.post, self.loop_a.attn, self.loop_c.attn):
            for mod in (blk.attn.wq, blk.attn.wk, blk.attn.wv, blk.ffn.gate, blk.ffn.up, blk.ffn.down):
                nn.init.normal_(mod.weight, std=0.02)

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
    m = GLAHybrid2LM().to(DEV)
    log("v27: %.2fM, 速率门多尺度+时标分层循环 (修复v26视野死因: λmean=0.4→retention 1e-14), %d 步 "
        "(对照 v24=2.4241, v26=2.5506)" % (sum(p.numel() for p in m.parameters()) / 1e6, PT_STEPS))
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
    log("v27 完成: 含pad=%.4f 真实=%.4f  [v24=2.4241, v26=2.5506]" % (a, mk))
    json.dump(dict(all=a, masked=mk,
                   alphas=[round(m.loop_a.alpha().item(), 4), round(m.loop_c.alpha().item(), 4)]),
              open(OUT, "w"), indent=2)


if __name__ == "__main__":
    main()
