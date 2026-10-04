#!/usr/bin/env python3
"""v29: Delta 循环 × Rubik 桥 — 补 v28 最可疑失血点 (v28 把 v24 的 rubik 桥换成了 delta 桥).
结构: [pre:Δ][A:Δ循环×10][RUBIK桥][C:Δ循环×10][post:Δ] — 22 个 Gated-Delta 站 + v24 原版桥.
与 v28 唯一差异: 桥 = RubikFFNBlockV2 (v24 夺冠组件, 数据依赖 λ + 旋转 G 短卷积, 零 softmax),
FFN 退回 4D 配平参数 (桥多出的参数由 4.6D→4D 扣回).
对照: v24=2.4241 (softmax+同款桥), v28=2.5148 (纯 delta 桥), 目标 <2.4241."""
import sys, os, time, math, json, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deqlm_v19_tuned23m import RMSNorm, SwiGLUFFN, masked_suffix_eval
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
R_LR = 4
FFN_H = 4 * D                # 4D — v24 同款 (rubik 桥回填后扣回 4.6D)


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
LOG = os.path.join(HERE, "deqlm29.log")
OUT = os.path.join(HERE, "deqlm29_results.json")
CKPT = os.path.join(HERE, "deqlm29_pt_full.pth")


def log(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


class RubikLinearAttnV2(nn.Module):
    """v24 原版桥: 数据依赖 lam_t + 旋转 G + 短卷积 v + bf16 扫描 (零 softmax)."""

    def __init__(self, r=R_LR):
        super().__init__()
        self.r = r
        self.ln_in = RMSNorm(D)
        self.q = nn.Linear(D, D, bias=False)
        self.k = nn.Linear(D, D, bias=False)
        self.v = nn.Linear(D, D, bias=False)
        self.uv = nn.Linear(D, H * 2 * r * (D // H), bias=False)
        nn.init.normal_(self.uv.weight, 0.0, 0.01)
        self.w_lam = nn.Linear(D, H * (D // H), bias=True)
        nn.init.zeros_(self.w_lam.weight)
        nn.init.constant_(self.w_lam.bias, -1.0)
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
        lam = torch.sigmoid(self.w_lam(h)).view(B, L, H, dh)
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


class RubikFFNBlockV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = RubikLinearAttnV2()
        self.ffn = SwiGLUFFN()

    def forward(self, x):
        return self.ffn(self.attn(x))


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
        self.ffn = SwiGLUFFN()

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
    """[pre:Δ][A:Δ×10][RUBIK桥][C:Δ×10][post:Δ] — 22 个 Delta 站 + v24 原版桥."""

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.pre = DeltaBlock()
        self.loop_a = TiedDeltaBlock(KS[0])
        self.bridge = RubikFFNBlockV2()
        self.loop_c = TiedDeltaBlock(KS[2])
        self.post = DeltaBlock()
        self.ln_f = RMSNorm(D)
        self.head = nn.Linear(D, V, bias=False)
        nn.init.normal_(self.emb.weight, std=0.02)
        nn.init.normal_(self.head.weight, std=0.02)
        for blk in (self.pre, self.post, self.loop_a.attn, self.loop_c.attn):
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
    log("v29: %.2fM, Delta循环×22 + Rubik桥(v24原版), %d 步 "
        "(对照 v24=2.4241 同款桥, v28=2.5148 delta桥)"
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
    log("v29 完成: 含pad=%.4f 真实=%.4f  [v24=2.4241, v28=2.5148]" % (a, mk))
    json.dump(dict(all=a, masked=mk,
                   alphas=[round(m.loop_a.alpha().item(), 4), round(m.loop_c.alpha().item(), 4)]),
              open(OUT, "w"), indent=2)


if __name__ == "__main__":
    main()
