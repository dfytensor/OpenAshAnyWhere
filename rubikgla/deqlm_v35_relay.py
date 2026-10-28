#!/usr/bin/env python3
"""v31: v30 + 双槽 Delta + MA-RMSNorm — 零 softmax 第三发.
依据 v30 终评三发现捆绑:
  1. 多槽: 槽=虚拟 kv 头 (KV5×S2=10 组独立 k/v/rate/η, 组内 q 共享, 槽输出求和)
     — 针对残余 copy 缺口 -8pp (写入干扰): 槽分散覆写容量, 槽间内容/策略特化
  2. MA-RMSNorm 全局替换 (双边同换, ma_rms_frsm3.7 血统 -4.9% loss):
     y = (x/RMS)·γ·s + α·log(RMS)·W  (s=2.0, α=0.1, W 零初始化)
  3. v30 全部修复保留 (kv5/η0.12 起步/非归一化 k/速率地板 0.0005)
FFN 退到 1104 配平: ~23.1M (v24=23.19M).
对照: v24=2.4241, v28=2.5148, v30=2.4949 (零 softmax 纪录)."""
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


def delta_scan_kv(k, v, rate, eta, q, s0=None):
    """Triton 融合扫描 (2026-10-11): 寄存器常驻递归 + 手写 VJP, 3.8ms vs chunked-inductor 20ms (5.3x).
    k,v,rate,eta: (B,L,KV*S,dh); q: (B,L,H,dh). 返回 y (B,L,H*dh). 槽求和在外层.
    数值对照: y/5路梯度 vs fp32 参考 rel≈1.4e-3 (test_delta_triton.py 全绿)."""
    y = delta_scan_triton(k, v, rate, eta, q, KV_H, SLOTS, H // KV_H, None, s0)   # (B,L,G,KVG,dh)
    B, L = k.shape[0], k.shape[1]
    y = y.view(B, L, KV_H, SLOTS, H // KV_H, DH).sum(3)
    return y.reshape(B, L, H * DH)


DELTA_FN = delta_scan_kv

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "deqlm31s.log")
OUT = os.path.join(HERE, "deqlm31s_results.json")
CKPT = os.path.join(HERE, "deqlm31s_pt_full.pth")
RESUME = os.path.join(HERE, "deqlm31s_resume.pth")


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
        self.w_gate = nn.Linear(D, KV_H * SLOTS * self.dh, bias=True)   # v36 写入门
        nn.init.zeros_(self.w_gate.weight)
        nn.init.constant_(self.w_gate.bias, 3.0)            # sigmoid≈0.95 近直通起步
        self.use_gate = False                               # 由上层打开 (v2 checkpoint 兼容)
        self.use_sel = False                                # RWKV-6 式低秩选择性衰减
        self.sel_A = nn.Linear(D, 8, bias=False)            # 秩8 双线性: 重要token rate→0
        self.sel_B = nn.Linear(8, KV_H * SLOTS * self.dh, bias=False)
        self.sel_scale = nn.Parameter(torch.zeros(KV_H * SLOTS * self.dh))
        nn.init.zeros_(self.sel_A.weight)
        nn.init.zeros_(self.sel_B.weight)                   # 零初始化 = 恒等 (sel=0.5, scale=0)
        self.capture = False
        self.capture_bounds = None
        self._last_state = None
        self._last_states_b = None

    def _scan_core(self, x, itmult, s0=None):
        if self.capture:
            import delta_triton as _dt
            B, L, _ = x.shape
            G = self.KV * self.S
            hh = self.ln(x)
            qq = self.wq(hh).view(B, L, H, self.dh)
            kk = self.wk(hh).view(B, L, G, self.dh)
            vv = self.wv(hh).view(B, L, G, self.dh)
            vf = vv.reshape(B, L, -1).transpose(1, 2)
            import torch.nn.functional as _F
            vc = _F.conv1d(_F.pad(vf, (3, 0)), self.dwconv.weight, groups=G * self.dh)
            vv = (vf + vc).transpose(1, 2).reshape(B, L, G, self.dh)
            rr = _F.softplus(self.w_rate(hh)).view(B, L, G, self.dh) * itmult
            ee = torch.sigmoid(self.w_eta(hh)).view(B, L, G, self.dh)
            if self.use_gate:
                gg = torch.sigmoid(self.w_gate(hh)).view(B, L, G, self.dh)
                kk = kk * gg
                vv = vv * gg
            if self.use_sel:
                sel = torch.sigmoid(self.sel_B(self.sel_A(hh))).view(B, L, G, self.dh)
                rr = rr * torch.exp(self.sel_scale.view(1, 1, G, self.dh) * (sel - 0.5))
            bnd = getattr(self, "capture_bounds", None)
            if bnd is not None:
                yv, sb = _dt.delta_scan_states(kk, vv, rr, ee, qq, KV_H, SLOTS, H // KV_H, s0=s0, boundary_idx=bnd)
                self._last_states_b = sb                  # (B,n_b,G,DH,DH)
                self._last_state = sb[:, -1]
            else:
                yv, self._last_state = _dt.delta_scan_states(kk, vv, rr, ee, qq, KV_H, SLOTS, H // KV_H, s0=s0)
            return yv.view(B, L, KV_H, SLOTS, H // KV_H, DH).sum(3).reshape(B, L, H * DH)
        B, L, _ = x.shape
        G = self.KV * self.S
        h = self.ln(x)
        q = self.wq(h).view(B, L, H, self.dh)
        k = self.wk(h).view(B, L, G, self.dh)
        v = self.wv(h).view(B, L, G, self.dh)
        v_flat = v.reshape(B, L, -1).transpose(1, 2)
        v_conv = F.conv1d(F.pad(v_flat, (3, 0)), self.dwconv.weight, groups=G * self.dh)
        v = (v_flat + v_conv).transpose(1, 2).reshape(B, L, G, self.dh)
        rate = F.softplus(self.w_rate(h)).view(B, L, G, self.dh)
        rate = rate * itmult                                   # v31s: 恒乘法 (it=0 时 mult≡1, 精确等价)
        eta = torch.sigmoid(self.w_eta(h)).view(B, L, G, self.dh)
        if self.use_gate:
            g = torch.sigmoid(self.w_gate(h)).view(B, L, G, self.dh)
            k = k * g
            v = v * g
        if self.use_sel:
            sel = torch.sigmoid(self.sel_B(self.sel_A(h))).view(B, L, G, self.dh)
            rate = rate * torch.exp(self.sel_scale.view(1, 1, G, self.dh) * (sel - 0.5))
        return DELTA_FN(k, v, rate, eta, q, s0)

    def _apply_raw(self, x, itmult, s0=None):
        """编译区: 前向核心 + 残差/输出投影 (triton 扫描为叶子)."""
        o = self._scan_core(x, itmult, s0)
        return x + self.out(o.to(x.dtype))

    def forward(self, x, itmult, s0=None):
        if self.training:
            out = checkpoint(self._scan_core, x, itmult, s0, use_reentrant=False)
        else:
            with torch.no_grad():
                out = self._scan_core(x, itmult, s0)
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
        self._attn_impl = None
        self._ffn_impl = None

    def _mult(self, it):
        G = self.attn.KV * self.attn.S
        return (torch.sigmoid(self.attn.rho_p) ** it).transpose(1, 2).reshape(1, 1, G, self.attn.dh)

    def forward(self, x, it=0, s0=None):
        mult = self._mult(it)
        if self.training:
            if self._attn_impl is None:
                self._attn_impl = torch.compile(self.attn._apply_raw)
                self._ffn_impl = torch.compile(self.ffn.forward)
            o = torch.utils.checkpoint.checkpoint(self._attn_impl, x, mult, s0, use_reentrant=False)
            return self._ffn_impl(o)
        with torch.no_grad():
            return self.ffn(self.attn._apply_raw(x, mult, s0))


class TiedDeltaBlock(nn.Module):
    def __init__(self, kmax, alpha_init=0.1):
        super().__init__()
        self.kmax = kmax
        self.attn = DeltaBlock()
        self.alpha_param = nn.Parameter(torch.tensor(math.log(alpha_init / (1 - alpha_init))))

    def alpha(self):
        return torch.clamp(torch.sigmoid(self.alpha_param), 0.01, 0.9)

    def forward(self, z0, s0=None):
        alpha = self.alpha()
        z = z0
        for j in range(self.kmax):
            z = z + alpha * self.attn(z, it=j, s0=(s0 if j == 0 else None))
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

    def forward(self, ids, s0_list=None):
        s0s = s0_list if s0_list is not None else [None] * 5
        h = self.emb(ids)
        h = self.pre(h, s0=s0s[0])
        h = self.loop_a(h, s0=s0s[1])
        h = self.bridge(h, s0=s0s[2])
        h = self.loop_c(h, s0=s0s[3])
        h = self.post(h, s0=s0s[4])
        return self.head(self.ln_f(h))

    def capture_states(self, ids, boundary_idx=None, s0_list=None):
        """无梯度前向, 返回 5 站末状态 [(B,G,DH,DH)] (接力接口).
        boundary_idx 给定时额外返回边界状态列表 [(B,n_b,G,DH,DH)].
        s0_list: 5 站初始状态 (链式接力 capture 用, 支持分段低成本链)."""
        stations = (self.pre, self.loop_a.attn, self.bridge, self.loop_c.attn, self.post)
        was = [blk.attn.capture for blk in stations]
        for blk in stations:
            blk.attn.capture = True
            blk.attn._last_state = None
            blk.attn.capture_bounds = boundary_idx
            blk.attn._last_states_b = None
        self.eval()
        with torch.no_grad():
            self.forward(ids, s0_list=s0_list)
        ends = [blk.attn._last_state for blk in stations]
        for blk, w in zip(stations, was):
            blk.attn.capture = w
            blk.attn.capture_bounds = None
        if boundary_idx is None:
            return ends
        return ends, [blk.attn._last_states_b for blk in stations]


def main():
    torch.manual_seed(0)
    random.seed(0)
    seqs = torch.load(PT_CACHE, map_location="cpu", weights_only=True)
    train, val = seqs[:int(0.95 * len(seqs))], seqs[:5000]
    m = DeltaLoopLM().to(DEV)
    log("v31s: %.2fM, v31+整站编译+fused优化器 (S2/KV5/η0.12/非归一化k/地板5e-4), %d 步 "
        "(对照 v24=2.4241, v30=2.4949)"
        % (sum(p.numel() for p in m.parameters()) / 1e6, PT_STEPS))
    opt = optim.AdamW(m.parameters(), lr=4e-4, weight_decay=0.01, betas=(0.9, 0.95), fused=True)

    start_step, rng = 0, random.Random(42)
    if os.path.exists(RESUME):
        ck = torch.load(RESUME, map_location="cpu")
        m.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        rng.setstate(ck["rng"])
        start_step = ck["st"]
        log("v31s 续跑: 从 step %d 恢复 (批次流与原 seed42 流严格衔接)" % start_step)

    def lr_at(st):
        if st < WARMUP:
            return 4e-4 * st / WARMUP
        p_ = (st - WARMUP) / max(1, PT_STEPS - WARMUP)
        return 4e-5 + 0.5 * (4e-4 - 4e-5) * (1 + math.cos(math.pi * p_))

    t0 = time.time()
    B_MICRO = 16
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
                continue
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
        log("v31s 异常 @step %s: %r\n%s" % (st if 'st' in dir() else "?", e, traceback.format_exc()))
        torch.save(dict(model=m.state_dict(), opt=opt.state_dict(),
                        rng=rng.getstate(), st=st + 1), RESUME)
        raise
    torch.save(m.state_dict(), CKPT)
    if os.path.exists(RESUME):
        os.remove(RESUME)
    a, mk = masked_suffix_eval(m, val)
    log("v31s 完成: 含pad=%.4f 真实=%.4f  [v24=2.4241, v31=2.4531]" % (a, mk))
    json.dump(dict(all=a, masked=mk,
                   alphas=[round(m.loop_a.alpha().item(), 4), round(m.loop_c.alpha().item(), 4)]),
              open(OUT, "w"), indent=2)


if __name__ == "__main__":
    main()
