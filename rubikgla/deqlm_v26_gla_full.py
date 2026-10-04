#!/usr/bin/env python3
"""v26: softmax 全部换成 GLA — 全递归架构 (KV cache = 0).
结构: [pre:GLA][A:GLA×10][RUBIK-v2×1][C:GLA×10][post:GLA] — v24 骨架不动, 仅换算子.
GLA 块: RMSNorm + GQA(kv=2) + 数据依赖λ + 短DWConv + 对角GLA扫描 (无RoPE, 位置由衰减编码).
扫描: gla_fast (行解耦融合, torch.compile 后 4ms/次 vs 稠密HS 46ms).
对照: v24=2.4241 (22softmax+1rubik, 425ms), v20=2.4284 (23ms/步softmax), 目标: NLL 全面逼近且全模型O(1)状态."""
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
from gla_fast import gla_fast

DEV = "cuda"
V, D, H = 23005, 320, 10
KV_H = 2
SEQ = 256
PT_STEPS = 39695
WARMUP = 1000
KS = [10, 10, 10]

GLA_FN = torch.compile(gla_fast)

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "deqlm26.log")
OUT = os.path.join(HERE, "deqlm26_results.json")
CKPT = os.path.join(HERE, "deqlm26_pt_full.pth")


def log(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


class GLAAttn(nn.Module):
    """GLA 站: 与 v19 GQAAttn 同骨架, softmax -> 对角GLA; 带 v24 配件 (数据依赖λ, 短卷积)."""

    def __init__(self):
        super().__init__()
        self.H, self.KV, self.dh = H, KV_H, D // H
        self.ln = RMSNorm(D)
        self.wq = nn.Linear(D, D, bias=False)
        self.wk = nn.Linear(D, KV_H * self.dh, bias=False)
        self.wv = nn.Linear(D, KV_H * self.dh, bias=False)
        self.out = nn.Linear(D, D, bias=False)
        nn.init.zeros_(self.out.weight)                          # 残差式零初始化
        self.w_lam = nn.Linear(D, H * self.dh, bias=True)        # 数据依赖 λ
        nn.init.zeros_(self.w_lam.weight)
        nn.init.constant_(self.w_lam.bias, -1.0)                 # λ≈0.27 起点 (同v24)
        self.dwconv = nn.Conv1d(D, D, 4, groups=D, bias=False)   # 短因果卷积 (v24配件)
        nn.init.zeros_(self.dwconv.weight)

    def _scan_core(self, x):
        B, L, _ = x.shape
        h = self.ln(x)
        q = self.wq(h).view(B, L, self.H, self.dh)
        k = self.wk(h).view(B, L, self.KV, self.dh).repeat_interleave(self.H // self.KV, dim=2)
        v = self.wv(h).view(B, L, self.KV, self.dh).repeat_interleave(self.H // self.KV, dim=2)
        v_flat = v.reshape(B, L, D).transpose(1, 2)
        v_conv = F.conv1d(F.pad(v_flat, (3, 0)), self.dwconv.weight, groups=D)
        v = (v_flat + v_conv).transpose(1, 2).reshape(B, L, H, self.dh)
        lam = torch.sigmoid(self.w_lam(h)).view(B, L, H, self.dh)
        o = GLA_FN(k, v, lam, q) / math.sqrt(self.dh)
        return o.reshape(B, L, D)

    def forward(self, x):
        if self.training:
            out = checkpoint(self._scan_core, x, use_reentrant=False)
        else:
            with torch.no_grad():
                out = self._scan_core(x)
        if torch.is_autocast_enabled():
            out = out.to(torch.bfloat16)
        return x + self.out(out)


class GLABlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = GLAAttn()
        self.ffn = SwiGLUFFN()

    def forward(self, x):
        return self.ffn(self.attn(x))


class TiedGLABlock(nn.Module):
    """同 v24 TiedBlock (z += α·block(z)), 但块为 GLABlock."""

    def __init__(self, kmax, alpha_init=0.1):
        super().__init__()
        self.kmax = kmax
        self.attn = GLABlock()
        self.alpha_param = nn.Parameter(torch.tensor(math.log(alpha_init / (1 - alpha_init))))

    def alpha(self):
        return torch.clamp(torch.sigmoid(self.alpha_param), 0.01, 0.9)

    def forward(self, z0):
        alpha = self.alpha()
        z = z0
        for _ in range(self.kmax):
            z = z + alpha * self.attn(z)
        return z


class RubikFFNBlockV2Wrapper(nn.Module):
    """v24 桥原样 (RubikLinearAttnV2 + SwiGLUFFN)."""

    def __init__(self):
        super().__init__()
        self.attn = RubikLinearAttnV2()
        self.ffn = SwiGLUFFN()

    def forward(self, x):
        return self.ffn(self.attn(x))


class GLAHybrid2LM(nn.Module):
    """[pre:GLA][A:GLA×10][RUBIK-v2×1][C:GLA×10][post:GLA] — 零softmax站."""

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.pre = GLABlock()
        self.loop_a = TiedGLABlock(KS[0])
        self.rubik_bridge = RubikFFNBlockV2Wrapper()
        self.loop_c = TiedGLABlock(KS[2])
        self.post = GLABlock()
        self.ln_f = RMSNorm(D)
        self.head = nn.Linear(D, V, bias=False)
        nn.init.normal_(self.emb.weight, std=0.02)
        nn.init.normal_(self.head.weight, std=0.02)
        # 标准件 normal(0.02); GLAAttn 的 out/w_lam/dwconv 零初始化在构造时已设, 不被覆盖
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
    log("v26: %.2fM, 22softmax全换GLA+rubik桥 (零KV cache), %d 步 (对照 v24=2.4241@425ms, v20=2.4284)"
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
    log("v26 完成: 含pad=%.4f 真实=%.4f  [v24=2.4241, v20=2.4284]" % (a, mk))
    json.dump(dict(all=a, masked=mk,
                   alphas=[round(m.loop_a.alpha().item(), 4), round(m.loop_c.alpha().item(), 4)]),
              open(OUT, "w"), indent=2)


if __name__ == "__main__":
    main()
