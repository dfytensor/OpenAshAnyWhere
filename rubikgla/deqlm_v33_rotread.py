#!/usr/bin/env python3
"""v33: 旋转检索头 — v24 血统反哺零 softmax 线 (站内级状态旋转).
依据: delta 线检索缺口 (~9pp repeat) 的唯一未试机制 = 数据依赖旋转 (v24 rubik 的独门武器).
设计 (单变量 over v31):
  1. 每站状态侧旋转: S ← G_t @ S_{t-1}, 再走 delta 擦写 (改变键几何, 降相似键干扰)
  2. G_t = exp(θ_t·[u wᵀ − w uᵀ]): u,w 单位化, θ ≤ π/8 (有界正交, 范数保持, 无新爆炸通道)
  3. G 在 torch 侧快 Taylor exp (6 项) 构建, autograd 自动反传; Triton kernel 只多一个 G 输入 + dG
  4. 每站一对 (u,w,θ) 全组共享 (+0.10M 参数 → 23.21M, 配平内)
其余全同 v31 (双槽/MA-norm s=1.0/v30 四项修复/槽求和/断点/NaN 绊线).
对照: v24=2.4241, v31=2.4531 (零 softmax 纪录), v32=2.4563."""
import sys, os, time, math, json, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deqlm_v19_tuned23m import masked_suffix_eval
from deqlm_v5_30m import get_batch, PT_CACHE
from delta_triton import delta_scan_triton

DEV = "cuda"
V, D, H = 23005, 320, 10
KV_H = 5
SLOTS = 2
DH = D // H
SEQ = 256
PT_STEPS = 39695
WARMUP = 1000
KS = [10, 10, 10]
FFN_H = 1104               # 配平: wk/wv/rate/eta 升到 D 后由 4D 扣回


def multiscale_inv_softplus(ch=KV_H * SLOTS):
    r = 10.0 ** torch.linspace(math.log10(0.0005), math.log10(0.5), DH)
    r = r.repeat(ch)                                     # (ch*dh,)
    return torch.log(torch.expm1(r))


class MARMSNorm(nn.Module):
    """MA-RMSNorm: y = (x/RMS)·γ·s + α·log(RMS)·W  (ma_rms_frsm3.7 血统).
    v31b: s=2.0→1.0 — s=2 使 lr 峰值区有效步长翻倍, @2000 后 loss NaN (两次复现);
    保留幅值注入路径 (log(RMS)·W), 去掉放大器."""

    def __init__(self, dim, s=1.0, alpha=0.1):
        super().__init__()
        self.s, self.alpha = s, alpha
        self.gamma = nn.Parameter(torch.ones(dim))
        self.mag_w = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        ms = x.float().pow(2).mean(-1, keepdim=True) + 1e-6
        y = x.float() * torch.rsqrt(ms) * self.gamma * self.s \
            + self.alpha * (ms + 1e-6).sqrt().log() * self.mag_w
        return y.to(x.dtype)


def delta_scan_kv(k, v, rate, eta, q, grot=None):
    """Triton 融合扫描 (v33 带可选状态旋转 G_t). 返回 y (B,L,H*dh). 槽求和在外层."""
    y = delta_scan_triton(k, v, rate, eta, q, KV_H, SLOTS, H // KV_H, grot)   # (B,L,G,KVG,dh)
    B, L = k.shape[0], k.shape[1]
    y = y.view(B, L, KV_H, SLOTS, H // KV_H, DH).sum(3)
    return y.reshape(B, L, H * DH)


DELTA_FN = delta_scan_kv

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "deqlm33.log")
OUT = os.path.join(HERE, "deqlm33_results.json")
CKPT = os.path.join(HERE, "deqlm33_pt_full.pth")
RESUME = os.path.join(HERE, "deqlm33_resume.pth")


def log(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


class SwiGLUFFN(nn.Module):
    """SwiGLU FFN @ 1104 (参数配平)."""

    def __init__(self):
        super().__init__()
        self.ln = MARMSNorm(D)
        self.gate = nn.Linear(D, FFN_H, bias=False)
        self.up = nn.Linear(D, FFN_H, bias=False)
        self.down = nn.Linear(FFN_H, D, bias=False)

    def forward(self, x):
        h = self.ln(x)
        return x + self.down(F.silu(self.gate(h)) * self.up(h))


class DeltaAttn(nn.Module):
    """双槽 Gated-Delta 站: MA-RMSNorm + GQA(槽=虚拟kv头) + 短卷积v + 速率门 + η覆写门."""

    def __init__(self):
        super().__init__()
        self.KV, self.S, self.dh = KV_H, SLOTS, DH
        self.ln = MARMSNorm(D)
        self.wq = nn.Linear(D, D, bias=False)
        self.wk = nn.Linear(D, KV_H * SLOTS * self.dh, bias=False)
        self.wv = nn.Linear(D, KV_H * SLOTS * self.dh, bias=False)
        self.out = nn.Linear(D, D, bias=False)
        nn.init.zeros_(self.out.weight)
        self.w_rate = nn.Linear(D, KV_H * SLOTS * self.dh, bias=True)
        nn.init.zeros_(self.w_rate.weight)
        with torch.no_grad():
            self.w_rate.bias.copy_(multiscale_inv_softplus())
        self.w_eta = nn.Linear(D, KV_H * SLOTS * self.dh, bias=True)
        nn.init.zeros_(self.w_eta.weight)
        nn.init.constant_(self.w_eta.bias, -2.0)           # η=0.12 起步
        self.dwconv = nn.Conv1d(KV_H * SLOTS * self.dh, KV_H * SLOTS * self.dh, 4,
                                groups=KV_H * SLOTS * self.dh, bias=False)
        nn.init.zeros_(self.dwconv.weight)
        self.rho_p = nn.Parameter(torch.full((KV_H, SLOTS, self.dh), math.log(0.8 / 0.2)))
        self.w_rot = nn.Linear(D, 2 * self.dh + 1, bias=True)
        nn.init.normal_(self.w_rot.weight, std=0.01)
        with torch.no_grad():
            self.w_rot.bias.zero_()
            self.w_rot.bias[-1] = -1.5                     # θ = (π/8)σ(-1.5) ≈ 0.072 近恒等起步

    def _build_grot(self, h):
        B, L, _ = h.shape
        uw = self.w_rot(h.float())                         # (B,L,2dh+1)
        u = F.normalize(uw[..., :self.dh], dim=-1, eps=1e-5)
        wv = F.normalize(uw[..., self.dh:2 * self.dh], dim=-1, eps=1e-5)
        th = (math.pi / 8) * torch.sigmoid(uw[..., 2 * self.dh:])          # (B,L,1) ≤0.3927
        M = u.unsqueeze(-1) * wv.unsqueeze(-2) - wv.unsqueeze(-1) * u.unsqueeze(-2)
        GM = th.unsqueeze(-1) * M                                          # 斜对称, ‖GM‖≤0.786
        eye = torch.eye(self.dh, device=h.device, dtype=torch.float32).view(1, 1, self.dh, self.dh)
        P = GM
        E = eye + P
        for i in range(2, 7):                                              # exp Taylor 6 项, 误差 <4e-5
            P = P @ GM / i
            E = E + P
        return E

    def _scan_core(self, x, it):
        B, L, _ = x.shape
        G = self.KV * self.S
        h = self.ln(x)
        grot = self._build_grot(h)                         # (B,L,dh,dh) fp32
        q = self.wq(h).view(B, L, H, self.dh)
        k = self.wk(h).view(B, L, G, self.dh)
        v = self.wv(h).view(B, L, G, self.dh)
        v_flat = v.reshape(B, L, -1).transpose(1, 2)
        v_conv = F.conv1d(F.pad(v_flat, (3, 0)), self.dwconv.weight, groups=G * self.dh)
        v = (v_flat + v_conv).transpose(1, 2).reshape(B, L, G, self.dh)
        rate = F.softplus(self.w_rate(h)).view(B, L, G, self.dh)
        if it > 0:
            rate = rate * (torch.sigmoid(self.rho_p) ** it).transpose(1, 2).reshape(1, 1, G, self.dh)
        eta = torch.sigmoid(self.w_eta(h)).view(B, L, G, self.dh)
        return DELTA_FN(k, v, rate, eta, q, grot)

    def forward(self, x, it=0):
        if self.training:
            out = checkpoint(self._scan_core, x, it, use_reentrant=False)
        else:
            with torch.no_grad():
                out = self._scan_core(x, it)
        if torch.is_autocast_enabled():
            out = out.to(torch.bfloat16)
        else:
            out = out.to(x.dtype)          # eval 无 autocast: 对齐权重 dtype (fp32)
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
    """[pre:Δ][A:Δ×10][Δ桥][C:Δ×10][post:Δ] — 23 站双槽 Gated-Delta + MA-RMSNorm."""

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.pre = DeltaBlock()
        self.loop_a = TiedDeltaBlock(KS[0])
        self.bridge = DeltaBlock()
        self.loop_c = TiedDeltaBlock(KS[2])
        self.post = DeltaBlock()
        self.ln_f = MARMSNorm(D)
        self.head = nn.Linear(D, V, bias=False)
        nn.init.normal_(self.emb.weight, std=0.02)
        nn.init.normal_(self.head.weight, std=0.02)
        for blk in (self.pre, self.post, self.loop_a.attn, self.loop_c.attn, self.bridge):
            for mod in (blk.attn.wq, blk.ffn.gate, blk.ffn.up, blk.ffn.down):
                nn.init.normal_(mod.weight, std=0.02)
            # wk/wv: 0.01 — MA-norm s=2 使 k 能量×4, η·‖k‖² 贴稳定边界(2)致慢通道爆炸;
            # 半初始化拉回 v30 安全域 (η‖k‖²≈0.49)
            for mod in (blk.attn.wk, blk.attn.wv):
                nn.init.normal_(mod.weight, std=0.01)

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
    log("v33: %.2fM, 旋转检索头 (G_t=exp(θ[uwᵀ−wuᵀ]) 站内状态旋转 + v31 全部), %d 步 "
        "(对照 v24=2.4241, v31=2.4531)"
        % (sum(p.numel() for p in m.parameters()) / 1e6, PT_STEPS))
    opt = optim.AdamW(m.parameters(), lr=4e-4, weight_decay=0.01, betas=(0.9, 0.95))

    start_step, rng = 0, random.Random(42)
    if os.path.exists(RESUME):
        ck = torch.load(RESUME, map_location="cpu")
        m.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        rng.setstate(ck["rng"])
        start_step = ck["st"]
        log("v33 续跑: 从 step %d 恢复 (批次流与原 seed42 流严格衔接)" % start_step)

    def lr_at(st):
        if st < WARMUP:
            return 4e-4 * st / WARMUP
        p_ = (st - WARMUP) / max(1, PT_STEPS - WARMUP)
        return 4e-5 + 0.5 * (4e-4 - 4e-5) * (1 + math.cos(math.pi * p_))

    t0 = time.time()
    B_MICRO = 16
    nan_run = 0
    try:
        for st in range(start_step, PT_STEPS):
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
                nan_run += 1
                if nan_run == 1:
                    log("⚠ v33 NaN 首现 @step %d (l1=%s l2=%s)" % (st, l1.item(), l2.item()))
                if nan_run > 200:
                    log("v33 NaN 失控 @step %d (连续 %d 步) — 存断点退出" % (st, nan_run))
                    torch.save(dict(model=m.state_dict(), opt=opt.state_dict(),
                                    rng=rng.getstate(), st=st), RESUME)
                    raise RuntimeError("NaN runaway at step %d" % st)
                continue
            nan_run = 0
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            if st % 2000 == 0:
                alphas = "%.3f/%.3f" % (m.loop_a.alpha().item(), m.loop_c.alpha().item())
                log("%d/%d loss=%.4f a=[%s] (%.0fms/步, ETA %.1f时)"
                    % (st, PT_STEPS, (0.5 * (l1 + l2)).item(), alphas,
                       (time.time() - t0) / (st - start_step + 1) * 1000,
                       (PT_STEPS - st) * (time.time() - t0) / (st - start_step + 1) / 3600))
                if st > 0:
                    torch.save(dict(model=m.state_dict(), opt=opt.state_dict(),
                                    rng=rng.getstate(), st=st + 1), RESUME)
    except BaseException as e:
        import traceback
        log("v33 异常 @step %s: %r\n%s" % (st if 'st' in dir() else "?", e, traceback.format_exc()))
        torch.save(dict(model=m.state_dict(), opt=opt.state_dict(),
                        rng=rng.getstate(), st=st + 1), RESUME)
        raise
    torch.save(m.state_dict(), CKPT)
    if os.path.exists(RESUME):
        os.remove(RESUME)
    a, mk = masked_suffix_eval(m, val)
    log("v33 完成: 含pad=%.4f 真实=%.4f  [v24=2.4241, v31=2.4531]" % (a, mk))
    json.dump(dict(all=a, masked=mk,
                   alphas=[round(m.loop_a.alpha().item(), 4), round(m.loop_c.alpha().item(), 4)]),
              open(OUT, "w"), indent=2)


if __name__ == "__main__":
    main()
