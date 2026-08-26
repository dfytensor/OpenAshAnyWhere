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
                    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr, BLOCK_K: tl.constexpr):
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
    A = tl.load(a_ptrs, mask=mh[:, None] & mk[None, :], other=0.0)
    kW = tl.load(KW + offs_k[:, None] * W + offs_w[None, :],
                 mask=mk[:, None] & mw[None, :], other=0.0)
    T = tl.dot(A, kW)
    bias = tl.load(BIAS + offs_w, mask=mw, other=0.)
    T = tl.maximum(T + bias[None, :], 0.0)
    w2 = tl.load(WOUT + offs_w, mask=mw, other=0.)
    out = tl.dot(T, w2[:, None])
    tl.store(Y + pid_row * H + offs_h, tl.ravel(out), mask=mh)


@triton.jit
def _row_kernel_bwd(XP, DY, KW, WOUT, BIAS,
                    DX, DKW, DWOUT, DB, RB,  # RB=relu(T) 缓存重算
                    SP: tl.constexpr, H: tl.constexpr, W: tl.constexpr, K: tl.constexpr,
                    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr, BLOCK_K: tl.constexpr):
    """反向: 每 program 一行 tile. 原子累加 DKW/DWOUT/DB."""
    pid_row = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_k = tl.arange(0, BLOCK_K)
    offs_w = tl.arange(0, BLOCK_W)
    mh = offs_h < H
    mk = offs_k < K
    mw = offs_w < W
    base = pid_row * SP

    # 重算 A 与 T (前向值, 免存中间)
    a_ptrs = XP + base + offs_h[:, None] + offs_k[None, :]
    A = tl.load(a_ptrs, mask=mh[:, None] & mk[None, :], other=0.0)
    kW = tl.load(KW + offs_k[:, None] * W + offs_w[None, :],
                 mask=mk[:, None] & mw[None, :], other=0.0)
    T = tl.dot(A, kW)
    bias = tl.load(BIAS + offs_w, mask=mw, other=0.)
    T = T + bias[None, :]
    R = tl.maximum(T, 0.0)
    gate = tl.where(T > 0, 1.0, 0.0)

    dy = tl.load(DY + pid_row * H + offs_h, mask=mh, other=0.)   # [BH]
    w2 = tl.load(WOUT + offs_w, mask=mw, other=0.)                # [BW]
    dT = dy[:, None] * w2[None, :] * gate                         # [BH, BW]

    # dx[h+k] += Σ_w dT[h,w]·Kw[k,w]  -> 转置 3-tap (原子累加: 相邻 tile 的 h+k 重叠)
    dX_tile = tl.dot(dT, tl.trans(kW))                            # [BH, BW]@[BW, BK]
    dx_ptrs = DX + base + offs_h[:, None] + offs_k[None, :]
    tl.atomic_add(dx_ptrs, dX_tile, mask=mh[:, None] & mk[None, :])

    # dKw[k,w] += Σ_rows Σ_h dT[h,w]·A[h,k]
    dKw_tile = tl.dot(tl.trans(A), dT)                            # [BK, BW]
    tl.atomic_add(DKW + offs_k[:, None] * W + offs_w[None, :],
                  dKw_tile, mask=mk[:, None] & mw[None, :])
    # dw_out[w] += Σ dy·R
    dw_tile = tl.sum(dy[:, None] * R, axis=0)                     # [BW]
    tl.atomic_add(DWOUT + offs_w, dw_tile, mask=mw)
    # db[w] += Σ dT
    db_tile = tl.sum(dT, axis=0)
    tl.atomic_add(DB + offs_w, db_tile, mask=mw)


class _ConvLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, Kw, w_out, bias):
        # x: [b,s,h]; 统一 fp32 (kernel 用 fp32 dot, autocast 兼容)
        x = x.float()
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
        y = torch.empty(b * s * h, device=dev, dtype=x.dtype)
        BH, BW = 128, 64
        BK = max(triton.next_power_of_2(k), 16)
        grid = (b * s, triton.cdiv(h, BH))
        _row_kernel_dot[grid](xp, Kw, w_out, bias, y,
                              SP=sp, H=h, W=w, K=k, BLOCK_H=BH, BLOCK_W=BW, BLOCK_K=BK)
        ctx.save_for_backward(xp, Kw, w_out, bias)
        ctx.sp, ctx.h, ctx.w, ctx.k = sp, h, w, k
        return y.view(b, s, h)

    @staticmethod
    def backward(ctx, dy):
        xp, Kw, w_out, bias = ctx.saved_tensors
        dy = dy.float()
        sp, h, w, k = ctx.sp, ctx.h, ctx.w, ctx.k
        b_s = dy.shape[0] * dy.shape[1]
        p = k // 2
        dy_flat = dy.contiguous().reshape(-1).float()
        dev = dy.device
        dKw = torch.zeros(k, w, device=dev, dtype=torch.float32)
        dw_out = torch.zeros(w, device=dev, dtype=torch.float32)
        db = torch.zeros(w, device=dev, dtype=torch.float32)
        BH, BW = 128, 64
        BK = max(triton.next_power_of_2(k), 16)
        grid = (b_s, triton.cdiv(h, BH))
        dx_acc = torch.zeros(b_s, sp, device=dev, dtype=torch.float32)
        _row_kernel_bwd[grid](xp, dy_flat, Kw, w_out, bias,
                              dx_acc, dKw, dw_out, db, None,
                              SP=sp, H=h, W=w, K=k, BLOCK_H=BH, BLOCK_W=BW, BLOCK_K=BK)
        dx = dx_acc[:, p:p + h]                     # 去 replicate-pad
        dx = dx.view(dy.shape[0], dy.shape[1], h).to(dy.dtype)
        return dx, dKw.to(Kw.dtype), dw_out.to(w_out.dtype), db.to(bias.dtype)
