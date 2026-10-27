#!/usr/bin/env python3
"""delta_triton: v31 delta 扫描的 Triton 融合 kernel (前向寄存器常驻 + 手写反向 VJP).
语义 (与 chunked_affine_scan 路径数值等价, 全 fp32 状态):
  S_t = λ ⊙ (S_{t-1} − η ⊙ k ⊙ (kᵀS_{t-1})) + k vᵀ      λ=exp(−rate)
  y_{t,m} = q_{t,m}ᵀ S_t   (m ∈ 该 group 的 KVG 个 q 头; 槽求和在外层 torch.sum)
并行: 程序 (b, g, vh) — 状态按 v 维 VSPLIT=2 切分; k/rate/eta 全量; dv 无冲突, dk/dλ/dη/dq atomic.
前向不物化完整 S (仅存 bf16 状态轨迹供反向), 反向为精确 VJP.
"""
import os
import torch
import torch.nn as nn
import triton
import triton.language as tl

VSPLIT = int(os.environ.get("DELTA_VSPLIT", "1"))   # 1=单程序全v(默认, 无原子竞争, 直写, 快28%); 2=旧对照
STAGES = int(os.environ.get("DELTA_STAGES", "5"))   # tl.range 流水线级数 (预取)


@triton.jit
def delta_fwd_kernel(
    K, V, RATE, ETA, Q, GROT, S0, OUT, SSTORE, SOUT,
    L: tl.constexpr, G: tl.constexpr, KV: tl.constexpr, H: tl.constexpr,
    DH: tl.constexpr, VH: tl.constexpr, VSPLIT: tl.constexpr, KVG: tl.constexpr, STAGES: tl.constexpr, HAS_ROT: tl.constexpr,
    HAS_S0: tl.constexpr, RETURN_STATE: tl.constexpr,
):
    pid = tl.program_id(0)
    ns = G * VSPLIT
    b = pid // ns
    r = pid % ns
    g = r // VSPLIT
    vh = r % VSPLIT

    offs_i = tl.arange(0, DH)
    offs_js = tl.arange(0, VH) + vh * VH          # v 维全局 (读 v)
    offs_jl = tl.arange(0, VH)                    # v 维局部 (v 切分内)

    S = tl.zeros((DH, VH), dtype=tl.float32)
    if HAS_S0:
        s0_base = S0 + (b * G + g) * DH * DH
        S = tl.load(s0_base + offs_i[:, None] * DH + offs_jl[None, :]).to(tl.float32)

    k_base = K + (b * L * G + g) * DH
    v_base = V + (b * L * G + g) * DH
    r_base = RATE + (b * L * G + g) * DH
    e_base = ETA + (b * L * G + g) * DH
    q_base = Q + (b * L * H + (g // (G // KV)) * KVG) * DH
    o_base = OUT + (b * L * G + g) * KVG * DH
    s_base = SSTORE + (b * L * G + g) * DH * DH
    g_base = GROT + b * L * DH * DH               # G 只按 (b,t) 索引 (站内全体共享)

    for t in tl.range(0, L, num_stages=STAGES):
        kt = tl.load(k_base + t * G * DH + offs_i).to(tl.float32)
        vt = tl.load(v_base + t * G * DH + offs_js).to(tl.float32)
        lam = tl.exp(-tl.load(r_base + t * G * DH + offs_i).to(tl.float32))
        et = tl.load(e_base + t * G * DH + offs_i).to(tl.float32)
        if HAS_ROT:
            Gt = tl.load(g_base + t * DH * DH + offs_i[:, None] * DH + offs_i[None, :]).to(tl.float32)
            R = tl.dot(Gt, S, input_precision="ieee")
        else:
            R = S
        rr = tl.sum(kt[:, None] * R, axis=0)                       # (VH,)
        S = lam[:, None] * (R - et[:, None] * kt[:, None] * rr[None, :]) + kt[:, None] * vt[None, :]
        tl.store(s_base + t * G * DH * DH + offs_i[:, None] * DH + offs_js[None, :],
                 S.to(SSTORE.dtype.element_ty))
        for m in tl.static_range(KVG):
            qm = tl.load(q_base + t * H * DH + m * DH + offs_i).to(tl.float32)
            ym = tl.sum(qm[:, None] * S, axis=0)
            tl.store(o_base + t * G * KVG * DH + m * DH + offs_js,
                     ym.to(OUT.dtype.element_ty))


@triton.jit
def delta_bwd_kernel(
    K, V, RATE, ETA, Q, GROT, DY, SSTORE, DK, DV, DRATE, DETA, DQ, DGROT, DSL, DS0,
    L: tl.constexpr, G: tl.constexpr, KV: tl.constexpr, H: tl.constexpr,
    DH: tl.constexpr, VH: tl.constexpr, VSPLIT: tl.constexpr, KVG: tl.constexpr, STAGES: tl.constexpr, HAS_ROT: tl.constexpr, HAS_DSL: tl.constexpr, HAS_DS0: tl.constexpr, HAS_S0: tl.constexpr,
):
    pid = tl.program_id(0)
    ns = G * VSPLIT
    b = pid // ns
    r = pid % ns
    g = r // VSPLIT
    vh = r % VSPLIT

    offs_i = tl.arange(0, DH)
    offs_js = tl.arange(0, VH) + vh * VH
    offs_jl = tl.arange(0, VH)

    k_base = K + (b * L * G + g) * DH
    v_base = V + (b * L * G + g) * DH
    r_base = RATE + (b * L * G + g) * DH
    e_base = ETA + (b * L * G + g) * DH
    q_base = Q + (b * L * H + (g // (G // KV)) * KVG) * DH
    dy_base = DY + (b * L * G + g) * KVG * DH
    s_base = SSTORE + (b * L * G + g) * DH * DH
    dk_base = DK + (b * L * G + g) * DH
    dv_base = DV + (b * L * G + g) * DH
    dr_base = DRATE + (b * L * G + g) * DH
    de_base = DETA + (b * L * G + g) * DH
    dq_base = DQ + (b * L * H + (g // (G // KV)) * KVG) * DH
    g_base = GROT + b * L * DH * DH
    dg_base = DGROT + b * L * DH * DH

    dS = tl.zeros((DH, VH), dtype=tl.float32)

    for _i in tl.range(0, L, num_stages=STAGES):
        t = (L - 1) - _i
        kt = tl.load(k_base + t * G * DH + offs_i).to(tl.float32)
        vt = tl.load(v_base + t * G * DH + offs_js).to(tl.float32)
        lam = tl.exp(-tl.load(r_base + t * G * DH + offs_i).to(tl.float32))
        et = tl.load(e_base + t * G * DH + offs_i).to(tl.float32)
        if HAS_ROT:
            Gt = tl.load(g_base + t * DH * DH + offs_i[:, None] * DH + offs_i[None, :]).to(tl.float32)

        Sp = tl.load(s_base + t * G * DH * DH + offs_i[:, None] * DH + offs_js[None, :]).to(tl.float32)
        if t > 0:
            Sprev = tl.load(s_base + (t - 1) * G * DH * DH
                            + offs_i[:, None] * DH + offs_js[None, :]).to(tl.float32)
        else:
            Sprev = tl.zeros((DH, VH), dtype=tl.float32)

        # 读出路径 (S' = Sp): dq (atomic, 跨 vh+槽) + dS 注入 (局部)
        for m in tl.static_range(KVG):
            qm = tl.load(q_base + t * H * DH + m * DH + offs_i).to(tl.float32)
            dym = tl.load(dy_base + t * G * KVG * DH + m * DH + offs_js).to(tl.float32)
            tl.atomic_add(dq_base + t * H * DH + m * DH + offs_i,
                          tl.sum(dym[None, :] * Sp, axis=1))
            dS += qm[:, None] * dym[None, :]

        # 步 VJP (R = G S_{t-1} 为旋转后状态)
        if HAS_ROT:
            R = tl.dot(Gt, Sprev, input_precision="ieee")
        else:
            R = Sprev
        rr = tl.sum(kt[:, None] * R, axis=0)                        # (VH,)
        t1 = R - et[:, None] * kt[:, None] * rr[None, :]
        gu = dS
        dv = tl.sum(gu * kt[:, None], axis=0)                       # (VH,) 局部, 无冲突
        dk_a = tl.sum(gu * vt[None, :], axis=1)                     # (DH,)
        gt1 = lam[:, None] * dS
        dlam = tl.sum(dS * t1, axis=1)                              # (DH,)
        dr = -tl.sum(gt1 * (et[:, None] * kt[:, None]), axis=0)     # (VH,) 局部
        gk = tl.sum(gt1 * rr[None, :], axis=1)                      # (DH,) Σ_j gt1_ij r_j
        det = -kt * gk
        dk_b = -et * gk
        dk_c = tl.sum(dr[None, :] * R, axis=1)                      # Σ_j dr_j R_ij
        drate = -lam * dlam
        tl.store(dv_base + t * G * DH + offs_js, dv)
        if VSPLIT == 1:
            # VSPLIT=1: (b,t,g) 所有权唯一, 直写 (省原子开销); DQ 仍原子(跨槽共享 q 头)
            tl.store(dk_base + t * G * DH + offs_i, dk_a + dk_b + dk_c)
            tl.store(dr_base + t * G * DH + offs_i, drate)
            tl.store(de_base + t * G * DH + offs_i, det)
        else:
            tl.atomic_add(dk_base + t * G * DH + offs_i, dk_a + dk_b + dk_c)
            tl.atomic_add(dr_base + t * G * DH + offs_i, drate)
            tl.atomic_add(de_base + t * G * DH + offs_i, det)

        dR = gt1 + kt[:, None] * dr[None, :]
        if HAS_ROT:
            dGt = tl.dot(dR, tl.trans(Sprev), input_precision="ieee")
            tl.atomic_add(dg_base + t * DH * DH + offs_i[:, None] * DH + offs_i[None, :], dGt)
            dS = tl.dot(tl.trans(Gt), dR, input_precision="ieee")
        else:
            dS = dR
        if HAS_DS0:
            ds0_base = DS0 + (b * G + g) * DH * DH
            tl.atomic_add(ds0_base + offs_i[:, None] * DH + offs_jl[None, :], dS)


class DeltaScanTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, k, v, rate, eta, q, grot, s0, kv, slots, kvg, return_state=False):
        B, L, G, DH = k.shape
        H = q.shape[2]
        VH = DH // VSPLIT
        k = k.contiguous(); v = v.contiguous(); rate = rate.contiguous()
        eta = eta.contiguous(); q = q.contiguous()
        ctx.has_rot = grot is not None
        if ctx.has_rot:
            grot = grot.contiguous()
        else:
            grot = k
        ctx.has_s0 = s0 is not None
        if ctx.has_s0:
            s0 = s0.contiguous()
        else:
            s0 = k
        HAS_ROT_ = 1 if ctx.has_rot else 0
        HAS_S0_ = 1 if ctx.has_s0 else 0
        out = torch.empty(B, L, G, kvg, DH, device=k.device, dtype=torch.bfloat16)
        sstore = torch.empty(B, L, G, DH, DH, device=k.device, dtype=torch.float32)
        sout = (torch.empty(B, G, DH, DH, device=k.device, dtype=torch.bfloat16) if return_state
                else torch.empty(1, device=k.device, dtype=torch.bfloat16))
        grid = (B * G * VSPLIT,)
        delta_fwd_kernel[grid](k, v, rate, eta, q, grot, s0, out, sstore, sout,
                               L=L, G=G, KV=kv, H=H, DH=DH, VH=VH, VSPLIT=VSPLIT, KVG=kvg, STAGES=STAGES, HAS_ROT=HAS_ROT_, HAS_S0=HAS_S0_, RETURN_STATE=return_state)
        ctx.save_for_backward(k, v, rate, eta, q, grot, sstore, sout)
        ctx.kv, ctx.kvg = kv, kvg
        ctx.return_state = return_state
        if return_state:
            return out, sout
        return out

    @staticmethod
    def backward(ctx, dy, ds_last=None):
        k, v, rate, eta, q, grot, sstore, sout = ctx.saved_tensors
        B, L, G, DH = k.shape
        H = q.shape[2]
        VH = DH // VSPLIT
        dy = dy.contiguous()
        dk = torch.zeros((B, L, G, DH), device=k.device, dtype=torch.float32)
        dv = torch.zeros_like(dk)
        dr = torch.zeros_like(dk)
        de = torch.zeros_like(dk)
        dq = torch.zeros((B, L, H, DH), device=k.device, dtype=torch.float32)
        dg = torch.zeros((B, L, DH, DH), device=k.device, dtype=torch.float32)
        dsl_t = ds_last.contiguous().float() if ds_last is not None else dv
        HAS_DSL_ = 1 if ds_last is not None else 0
        HAS_S0_ = 1 if ctx.has_s0 else 0
        HAS_DS0_ = HAS_S0_
        HAS_ROT_ = 1 if ctx.has_rot else 0
        grid = (B * G * VSPLIT,)
        delta_bwd_kernel[grid](k, v, rate, eta, q, grot, dy, sstore, dk, dv, dr, de, dq, dg, dsl_t, dv,
                               L=L, G=G, KV=ctx.kv, H=H, DH=DH, VH=VH, VSPLIT=VSPLIT, KVG=ctx.kvg, STAGES=STAGES, HAS_ROT=HAS_ROT_, HAS_S0=HAS_S0_, HAS_DSL=HAS_DSL_, HAS_DS0=HAS_DS0_)
        dsl_out = ds_last.to(k.dtype) if ds_last is not None else None
        return (dk.to(k.dtype), dv.to(v.dtype), dr.to(rate.dtype), de.to(eta.dtype),
                dq.to(q.dtype), (dg.to(grot.dtype) if ctx.has_rot else None),
                dsl_out, None, None, None, (dsl_out if ctx.return_state else None))


def delta_scan_triton(k, v, rate, eta, q, kv, slots, kvg, grot=None, s0=None, return_state=False):
    """k,v,rate,eta: (B,L,KV*S,dh); q: (B,L,H,dh); s0: (B,G,dh,dh) 可选初始状态.
    返回 y (B,L,KV*S,KVG,dh) 或 (y, sout) 当 return_state=True."""
    return DeltaScanTriton.apply(k, v, rate, eta, q, grot, s0, kv, slots, kvg, return_state)


def delta_scan_states(k, v, rate, eta, q, kv, slots, kvg, s0=None):
    """无梯度: 返回 (y, 每站末状态 (B,G,dh,dh) fp32) — 接力接口用."""
    B, L, G, DH = k.shape
    H = q.shape[2]
    VH = DH // VSPLIT
    k = k.contiguous(); v = v.contiguous(); rate = rate.contiguous()
    eta = eta.contiguous(); q = q.contiguous()
    if s0 is None:
        s0 = torch.zeros(B, G, DH, DH, device=k.device, dtype=torch.float32)
    else:
        s0 = s0.contiguous().float()
    out = torch.empty(B, L, G, kvg, DH, device=k.device, dtype=torch.bfloat16)
    sstore = torch.empty(B, L, G, DH, DH, device=k.device, dtype=torch.float32)
    dummy = torch.empty(1, device=k.device, dtype=torch.bfloat16)
    grid = (B * G * VSPLIT,)
    delta_fwd_kernel[grid](k, v, rate, eta, q, k, s0, out, sstore, dummy,
                           L=L, G=G, KV=kv, H=H, DH=DH, VH=VH, VSPLIT=VSPLIT, KVG=kvg, STAGES=STAGES, HAS_ROT=0, HAS_S0=1, RETURN_STATE=0)
    return out, sstore[:, -1].clone()
