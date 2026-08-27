"""ConvLinear Triton 前向 + Triton 反向: 训练全链 kernel 化.

前向: T = Kw·x (3-tap) + b -> relu -> @w_out     (两级 tl.dot)
反向: dT = dy·w_out·gate
      dx = dT @ Kw^T (转置 3-tap)
      dKw = dT^T·x  -> 反解 dK, dw_in (小矩阵, torch)
      dw_out = dy·relu(T)^T ; db = ΣdT
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _row_kernel_dot(XP, KW, WOUT, BIAS, Y,
                    SP: tl.constexpr, H: tl.constexpr, W: tl.constexpr, K: tl.constexpr,
                    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr, BLOCK_K: tl.constexpr,
                    ACT: tl.constexpr):
    pid_row = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_k = tl.arange(0, BLOCK_K)
    offs_w = tl.arange(0, BLOCK_W)
    mh = offs_h < H
    mk = offs_k < K
    mw = offs_w < W
    base = pid_row * SP
    a_ptrs = XP + base + offs_h[:, None] + offs_k[None, :]
    A = tl.load(a_ptrs, mask=mh[:, None] & mk[None, :], other=0.0).to(tl.float32)
    kW = tl.load(KW + offs_k[:, None] * W + offs_w[None, :],
                 mask=mk[:, None] & mw[None, :], other=0.0)
    T = tl.dot(A, kW, input_precision="tf32")                                   # [BLOCK_H, BLOCK_W] fp32
    bias = tl.load(BIAS + offs_w, mask=mw, other=0.)
    T = T + bias[None, :]
    if ACT:
        T = tl.maximum(T, 0.0)
    w2 = tl.load(WOUT + offs_w, mask=mw, other=0.)
    out = tl.dot(T, w2[:, None], input_precision="tf32")                        # [BLOCK_H, 1]
    tl.store(Y + pid_row * H + offs_h, tl.ravel(out).to(Y.dtype.element_ty), mask=mh)


@triton.jit
def _row_kernel_bwd(XP, DY, KW, WOUT, BIAS,
                    DX_S, DKW_P, DW_P, DB_P,
                    SP: tl.constexpr, H: tl.constexpr, W: tl.constexpr, K: tl.constexpr,
                    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr, BLOCK_K: tl.constexpr,
                    ACT: tl.constexpr):
    """反向: 每 program 一整行, 循环 h-block, 无全局原子.

    dX 平铺存 [rows,K,H] (槽唯一); 权重梯度在寄存器内跨 h-block 累加,
    行尾写本行 partial [rows,...], torch.sum(0) 归约.
    """
    pid_row = tl.program_id(0)
    offs_k = tl.arange(0, BLOCK_K)
    offs_w = tl.arange(0, BLOCK_W)
    mk = offs_k < K
    mw = offs_w < W
    base = pid_row * SP
    dkw_acc = tl.zeros((BLOCK_K, BLOCK_W), dtype=tl.float32)
    dw_acc = tl.zeros((BLOCK_W,), dtype=tl.float32)
    db_acc = tl.zeros((BLOCK_W,), dtype=tl.float32)

    for h0 in tl.range(0, H, BLOCK_H):
        offs_h = h0 + tl.arange(0, BLOCK_H)
        mh = offs_h < H
        a_ptrs = XP + base + offs_h[:, None] + offs_k[None, :]
        A = tl.load(a_ptrs, mask=mh[:, None] & mk[None, :], other=0.0).to(tl.float32)
        kW = tl.load(KW + offs_k[:, None] * W + offs_w[None, :],
                     mask=mk[:, None] & mw[None, :], other=0.0)
        T = tl.dot(A, kW, input_precision="tf32")
        bias = tl.load(BIAS + offs_w, mask=mw, other=0.)
        T = T + bias[None, :]
        R = tl.maximum(T, 0.0)
        gate = tl.where(T > 0, 1.0, 0.0)
        if not ACT:
            R = T
            gate = tl.where(T > -1e30, 1.0, 0.0)

        dy = tl.load(DY + pid_row * H + offs_h, mask=mh, other=0.)   # [BH]
        w2 = tl.load(WOUT + offs_w, mask=mw, other=0.)                # [BW]
        dT = dy[:, None] * w2[None, :] * gate                         # [BH, BW]

        # dx[h+k] = Σ_w dT[h,w]·Kw[k,w]  -> 平铺存 [rows,K,H] 槽 (h,k) 唯一
        dX_tile = tl.dot(dT, tl.trans(kW), input_precision="tf32")    # [BH, BW]@[BW, BK]
        tl.store(DX_S + pid_row * K * H + offs_k[None, :] * H + offs_h[:, None],
                 dX_tile, mask=mh[:, None] & mk[None, :])

        dkw_acc += tl.dot(tl.trans(A), dT, input_precision="tf32x3")    # [BK, BW]
        dw_acc += tl.sum(dy[:, None] * R, axis=0)                     # [BW]
        db_acc += tl.sum(dT, axis=0)

    tl.store(DKW_P + pid_row * BLOCK_K * W + offs_k[:, None] * W + offs_w[None, :],
             dkw_acc, mask=mk[:, None] & mw[None, :])
    tl.store(DW_P + pid_row * W + offs_w, dw_acc, mask=mw)
    tl.store(DB_P + pid_row * W + offs_w, db_acc, mask=mw)


@triton.jit
def _row_kernel_scatter(DX_S, DXP,
                        H: tl.constexpr, K: tl.constexpr, SP: tl.constexpr,
                        BLOCK_P: tl.constexpr):
    """dxp[r, p] = Σ_k dx_s[r, k, p-k]  (转置卷积散射, 每 program 一块 p, 无原子)."""
    pid_row = tl.program_id(0)
    pid_p = tl.program_id(1)
    offs_p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    mp = offs_p < SP
    acc = tl.zeros((BLOCK_P,), dtype=tl.float32)
    for kk in tl.static_range(K):
        src = offs_p - kk
        ms = mp & (src >= 0)
        v = tl.load(DX_S + pid_row * K * H + kk * H + src, mask=ms, other=0.0)
        acc += v
    tl.store(DXP + pid_row * SP + offs_p, acc, mask=mp)


class _ConvLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, Kw, w_out, bias, act):
        # x: [b,s,h] 保持原 dtype (autocast 下 bf16 激活, 免 fp32 拷贝; 权重转 fp32 主副本)
        Kw = Kw.float()
        w_out = w_out.float()
        bias = bias.float()
        b, s, h = x.shape
        k = Kw.shape[0]
        p = k // 2
        w = w_out.shape[0]
        dev = x.device
        xp = torch.nn.functional.pad(x, (p, p), mode="replicate").contiguous()
        sp = h + 2 * p
        y = torch.zeros(b * s * h, device=dev, dtype=x.dtype)
        BH, BW = 128, 64
        BK = max(triton.next_power_of_2(k), 16)
        grid = (b * s, triton.cdiv(h, BH))
        nw = triton.cdiv(w, BW)
        for ci in range(nw):
            w0 = ci * BW
            wc = min(BW, w - w0)
            yc = torch.empty(b * s * h, device=dev, dtype=x.dtype)
            _row_kernel_dot[grid](xp, Kw[:, w0:w0 + wc].contiguous(),
                                  w_out[w0:w0 + wc], bias[w0:w0 + wc], yc,
                                  SP=sp, H=h, W=wc, K=k,
                                  BLOCK_H=BH, BLOCK_W=BW, BLOCK_K=BK,
                                  ACT=act)
            y += yc
        ctx.save_for_backward(xp, Kw, w_out, bias)
        ctx.sp, ctx.h, ctx.w, ctx.k, ctx.act = sp, h, w, k, act
        return y.view(b, s, h)

    @staticmethod
    def backward(ctx, dy):
        xp, Kw, w_out, bias = ctx.saved_tensors
        dy = dy.float()
        sp, h, w, k, act = ctx.sp, ctx.h, ctx.w, ctx.k, ctx.act
        b_s = dy.shape[0] * dy.shape[1]
        p = k // 2
        dy_flat = dy.contiguous().reshape(-1).float()
        dev = dy.device
        BH, BW = 128, 64
        BK = max(triton.next_power_of_2(k), 16)
        grid = (b_s,)
        nw = triton.cdiv(w, BW)
        dx_s = torch.zeros(b_s, k, h, device=dev, dtype=torch.float32)
        dKw = torch.zeros(k, w, device=dev, dtype=torch.float32)
        dw_out = torch.zeros(w, device=dev, dtype=torch.float32)
        db = torch.zeros(w, device=dev, dtype=torch.float32)
        for ci in range(nw):
            w0 = ci * BW
            wc = min(BW, w - w0)
            dx_sc = torch.empty(b_s, k, h, device=dev, dtype=torch.float32)
            dkw_p = torch.empty(b_s, BK, wc, device=dev, dtype=torch.float32)
            dw_p = torch.empty(b_s, wc, device=dev, dtype=torch.float32)
            db_p = torch.empty(b_s, wc, device=dev, dtype=torch.float32)
            _row_kernel_bwd[grid](xp, dy_flat, Kw[:, w0:w0 + wc].contiguous(),
                                  w_out[w0:w0 + wc], bias[w0:w0 + wc],
                                  dx_sc, dkw_p, dw_p, db_p,
                                  SP=sp, H=h, W=wc, K=k,
                                  BLOCK_H=BH, BLOCK_W=BW, BLOCK_K=BK,
                                  ACT=act, num_warps=8, num_stages=2)
            dx_s += dx_sc
            dKw[:, w0:w0 + wc] = dkw_p.sum(0)[:k]
            dw_out[w0:w0 + wc] = dw_p.sum(0)
            db[w0:w0 + wc] = db_p.sum(0)
        dxp = torch.empty(b_s, sp, device=dev, dtype=torch.float32)
        NP = 256
        _row_kernel_scatter[(b_s, triton.cdiv(sp, NP))](dx_s, dxp,
                                                        H=h, K=k, SP=sp, BLOCK_P=NP)
        dx = dxp[:, p:p + h]
        dx[:, 0] += dxp[:, :p].sum(1)
        dx[:, h - 1] += dxp[:, sp - p:].sum(1)
        dx = dx.view(dy.shape[0], dy.shape[1], h).to(dy.dtype)
        return dx, dKw.to(Kw.dtype), dw_out.to(w_out.dtype), db.to(bias.dtype), None
