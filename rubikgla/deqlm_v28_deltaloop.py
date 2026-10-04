#!/usr/bin/env python3
"""v28: 全 Delta 规则循环 — 目标超过 softmax 冠军 v24=2.4241, 零 softmax 零 rotation.
结构: [pre:Δ][A:Δ循环×10][Δ桥][C:Δ循环×10][post:Δ] — 23 站全是 Gated-Delta.
核心机制:
  1. Delta 规则: S = λ(S − η k kᵀ S) + k vᵀ — 按 key 智能覆写 (精确检索的线性回答)
  2. kv 空间 GQA 扫描 (2 kv头, 组内5头共享状态): compile 后 4ms/站 (H=10 全空间要 24ms)
  3. v27 速率门多尺度 (软饱和, 256步保留率 0.6) + 时标分层循环 rate×ρ^j
  4. v24 配件: 短因果卷积 v + 零初始化 out
参数 23.3M (FFN 4.6D 补齐, 对照 v24=23.19M).
对照: v24=2.4241 (softmax), v26=2.5506 (GLA视野死), v27=~2.50 中止."""
import sys, os, time, math, json, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deqlm_v19_tuned23m import RMSNorm, masked_suffix_eval
from deqlm_v5_30m import get_batch, PT_CACHE
from scan import affine_scan

DEV = "cuda"
V, D, H = 23005, 320, 10
KV_H = 2
DH = D // H
SEQ = 256
PT_STEPS = 39695
WARMUP = 1000
KS = [10, 10, 10]
FFN_H = int(4.6 * D)          # 1472 — 补参数至 v24 同级 (4D 时仅 22.4M)


def multiscale_inv_softplus(kv=KV_H):
    r = 10.0 ** torch.linspace(math.log10(0.002), math.log10(0.5), DH)
    r = r.repeat(kv)                                     # (KV*dh,)
    return torch.log(torch.expm1(r))


def delta_scan_kv(k, v, rate, eta, q):
    """k,v,rate,eta: (B,L,KV,dh); q: (B,L,H,dh). kv空间扫描 + 组内读出 y=S@q."""
    B, L, KV, dh = k.shape
    H_ = q.shape[2]
    kn = F.normalize(k.float(), dim=-1)
    lam = torch.exp(-rate.float())
    I = torch.eye(dh, device=k.device, dtype=torch.float32).view(1, 1, 1, dh, dh)
    A = lam.unsqueeze(-1) * (I - eta.float().unsqueeze(-1) * (kn.unsqueeze(-1) * kn.unsqueeze(-2)))
    W = kn.unsqueeze(-1) * v.float().unsqueeze(-2)
    _, S = affine_scan(A.to(torch.bfloat16), W.to(torch.bfloat16))       # (B,L,KV,dh,dh)
    qg = q.float().view(B, L, KV, H_ // KV, dh)
    y = torch.einsum("blgpq,blgmq->blgmp", S.float(), qg).reshape(B, L, H_ * dh)
    return y


DELTA_FN = torch.compile(delta_scan_kv)

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "deqlm28.log")
OUT = os.path.join(HERE, "deqlm28_results.json")
CKPT = os.path.join(HERE, "deqlm28_pt_full.pth")


def log(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


class SwiGLUFFN46(nn.Module):
    """SwiGLU FFN @ 4.6D (参数补齐用, 其余同 v19 SwiGLUFFN)."""

    def __init__(self):
        super().__init__()
        self.ln = RMSNorm(D)
        self.gate = nn.Linear(D, FFN_H, bias=False)
        self.up = nn.Linear(D, FFN_H, bias=False)
        self.down = nn.Linear(FFN_H, D, bias=False)

    def forward(self, x):
        h = self.ln(x)
        return x + self.down(F.silu(self.gate(h)) * self.up(h))


class DeltaAttn(nn.Module):
    """Gated-Delta 站: RMSNorm + GQA(kv空间扫描) + 短卷积v + 速率门(多尺度, 时标分层) + η覆写门."""

    def __init__(self):
        super().__init__()
        self.KV, self.dh = KV_H, DH
        self.ln = RMSNorm(D)
        self.wq = nn.Linear(D, D, bias=False)
        self.wk = nn.Linear(D, KV_H * self.dh, bias=False)
        self.wv = nn.Linear(D, KV_H * self.dh, bias=False)
        self.out = nn.Linear(D, D, bias=False)
        nn.init.zeros_(self.out.weight)
        self.w_rate = nn.Linear(D, KV_H * self.dh, bias=True)
        nn.init.zeros_(self.w_rate.weight)
        with torch.no_grad():
            self.w_rate.bias.copy_(multiscale_inv_softplus())
        self.w_eta = nn.Linear(D, KV_H * self.dh, bias=True)
        nn.init.zeros_(self.w_eta.weight)                 # η=sigmoid(0)=0.5 起步
        nn.init.zeros_(self.w_eta.bias)
        self.dwconv = nn.Conv1d(KV_H * self.dh, KV_H * self.dh, 4, groups=KV_H * self.dh, bias=False)
        nn.init.zeros_(self.dwconv.weight)
        self.rho_p = nn.Parameter(torch.full((KV_H, self.dh), math.log(0.8 / 0.2)))

    def _scan_core(self, x, it):
        B, L, _ = x.shape
        h = self.ln(x)
        q = self.wq(h).view(B, L, H, self.dh)
        k = self.wk(h).view(B, L, self.KV, self.dh)
        v = self.wv(h).view(B, L, self.KV, self.dh)
        v_flat = v.reshape(B, L, -1).transpose(1, 2)
        v_conv = F.conv1d(F.pad(v_flat, (3, 0)), self.dwconv.weight, groups=KV_H * self.dh)
        v = (v_flat + v_conv).transpose(1, 2).reshape(B, L, self.KV, self.dh)
        rate = F.softplus(self.w_rate(h)).view(B, L, self.KV, self.dh)
        if it > 0:
            rate = rate * (torch.sigmoid(self.rho_p) ** it)
        eta = torch.sigmoid(self.w_eta(h)).view(B, L, self.KV, self.dh)
        return DELTA_FN(k, v, rate, eta, q)

    def forward(self, x, it=0):
        if self.training:
            out = checkpoint(self._scan_core, x, it, use_reentrant=False)
        else:
            with torch.no_grad():
                out = self._scan_core(x, it)
        if torch.is_autocast_enabled():
            out = out.to(torch.bfloat16)
        return x + self.out(out)


class DeltaBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = DeltaAttn()
        self.ffn = SwiGLUFFN46()

    def forward(self, x, it=0):
        return self.ffn(self.attn(x, it))


class TiedDeltaBlock(nn.Module):
    def __init__(self, kmax, alpha_init=0.1):
        super().__init__()
        self.kmax = kmax
        self.attn = DeltaBlock()
        self.alpha_param = nn.Parameter(torch.tensor(math.log(alpha_init / (1 - alpha_init))))

    def alpha(self):
        return torch.clamp(torch.sigmoid(self.alpha_param), 0.01, 0.9)

    def forward(self, z0):
        alpha = self.alpha()
        z = z0
        for j in range(self.kmax):
            z = z + alpha * self.attn(z, it=j)
        return z


class DeltaLoopLM(nn.Module):
    """[pre:Δ][A:Δ×10][Δ桥][C:Δ×10][post:Δ] — 23 站纯 Gated-Delta."""

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.pre = DeltaBlock()
        self.loop_a = TiedDeltaBlock(KS[0])
        self.bridge = DeltaBlock()
        self.loop_c = TiedDeltaBlock(KS[2])
        self.post = DeltaBlock()
        self.ln_f = RMSNorm(D)
        self.head = nn.Linear(D, V, bias=False)
        nn.init.normal_(self.emb.weight, std=0.02)
        nn.init.normal_(self.head.weight, std=0.02)
        for blk in (self.pre, self.post, self.loop_a.attn, self.loop_c.attn, self.bridge):
            for mod in (blk.attn.wq, blk.attn.wk, blk.attn.wv,
                        blk.ffn.gate, blk.ffn.up, blk.ffn.down):
                nn.init.normal_(mod.weight, std=0.02)

    def forward(self, ids):
        h = self.emb(ids)
        h = self.pre(h)
        h = self.loop_a(h)
        h = self.bridge(h)
        h = self.loop_c(h)
        h = self.post(h)
        return self.head(self.ln_f(h))


def main():
    torch.manual_seed(0)
    random.seed(0)
    seqs = torch.load(PT_CACHE, map_location="cpu", weights_only=True)
    train, val = seqs[:int(0.95 * len(seqs))], seqs[:5000]
    m = DeltaLoopLM().to(DEV)
    log("v28: %.2fM, 全Gated-Delta循环 (kv空间4ms扫描, 速率门多尺度+η覆写+时标分层), %d 步 "
        "(对照 v24=2.4241@425ms softmax, v26=2.5506)"
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
    log("v28 完成: 含pad=%.4f 真实=%.4f  [v24=2.4241, v26=2.5506]" % (a, mk))
    json.dump(dict(all=a, masked=mk,
                   alphas=[round(m.loop_a.alpha().item(), 4), round(m.loop_c.alpha().item(), 4)]),
              open(OUT, "w"), indent=2)


if __name__ == "__main__":
    main()
