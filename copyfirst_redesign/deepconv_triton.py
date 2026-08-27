"""DeepConv2DLinear 的 Triton 训练链: w=64 网格多层 2D 卷积 (宽+深).

数学: E[h,w] = x[h]·win[w] (rank-1 展开, h 维 replicate pad, w 维 zero pad)
  T1[c,h,w] = relu( Σ_{u,v} x[h+u]·win[w+v]·K1[c,u,v] + b1[c] )
  out[h]    = Σ_{c,u,v,w'} wo[w']·T1[c,h+u,w'+v]·K2[c,u,v]

预组合 (torch, autograd 跟踪, 每调用重算):
  W1[u, cW+w] = Σ_v K1[c,u,v]·win_pad[w+v]
  W2[u, cW+w] = Σ_v K2[c,u,v]·wo_pad[w+v]
则: T1 = relu(A @ W1 + b1); out = Σ_u T1[h+u]·W2[u]
bwd: dT1g[h] = gate ⊙ Σ_u dy[h-u+p]·W2[u] (dy 位移加载, 不物化 dT1)
  dA = dT1g @ W1^T; dW1 = A^T @ dT1g; dW2[u] = Σ_h dy[h-u+p]·R[h]
"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from conv_linear_triton_train import _row_kernel_scatter


def _shift0(t, v):
    """shift0(t, v)[i] = t[i+v] (越界零填充)."""
    if v == 0:
        return t
    if v > 0:
        return F.pad(t, (0, v))[v:v + t.numel()]
    return F.pad(t, (-v, 0))[:t.numel()]


@triton.jit
def _row_kernel_d2s1(XP, W1, B1, T1S,
                     SP: tl.constexpr, H: tl.constexpr, W1N: tl.constexpr, K: tl.constexpr,
                     BLOCK_H: tl.constexpr, BLOCK_W1: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_row = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_k = tl.arange(0, BLOCK_K)
    offs_w1 = tl.arange(0, BLOCK_W1)
    mh = offs_h < H
    mk = offs_k < K
    mw1 = offs_w1 < W1N
    base = pid_row * SP
    a_ptrs = XP + base + offs_h[:, None] + offs_k[None, :]
    A = tl.load(a_ptrs, mask=mh[:, None] & mk[None, :], other=0.0).to(tl.float32)
    w1 = tl.load(W1 + offs_k[:, None] * W1N + offs_w1[None, :],
                 mask=mk[:, None] & mw1[None, :], other=0.0)
    T1 = tl.dot(A, w1, input_precision="tf32")
    b1 = tl.load(B1 + offs_w1, mask=mw1, other=0.)
    T1 = tl.maximum(T1 + b1[None, :], 0.0)
    tl.store(T1S + pid_row * H * W1N + offs_h[:, None] * W1N + offs_w1[None, :],
             T1.to(T1S.dtype.element_ty), mask=mh[:, None] & mw1[None, :])


@triton.jit
def _row_kernel_d2s2(T1S, W2, Y,
                     H: tl.constexpr, W1N: tl.constexpr, K: tl.constexpr, P: tl.constexpr,
                     BLOCK_H: tl.constexpr, BLOCK_W1: tl.constexpr):
    pid_row = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_w1 = tl.arange(0, BLOCK_W1)
    mh = offs_h < H
    mw1 = offs_w1 < W1N
    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for u in tl.static_range(K):
        hu = offs_h + u - P
        mhu = mh & (hu >= 0) & (hu < H)
        t1i = tl.load(T1S + pid_row * H * W1N + hu[:, None] * W1N + offs_w1[None, :],
                      mask=mhu[:, None] & mw1[None, :], other=0.0).to(tl.float32)
        w2u = tl.load(W2 + u * W1N + offs_w1, mask=mw1, other=0.)
        acc += tl.ravel(tl.dot(t1i, w2u[:, None], input_precision="tf32"))
    tl.store(Y + pid_row * H + offs_h, acc.to(Y.dtype.element_ty), mask=mh)


@triton.jit
def _row_kernel_d2b(XP, DY, W1, B1, W2,
                    DX_S, DW1_P, DW2_P, DB_P,
                    SP: tl.constexpr, H: tl.constexpr, W1N: tl.constexpr, K: tl.constexpr,
                    P: tl.constexpr,
                    BLOCK_H: tl.constexpr, BLOCK_W1: tl.constexpr, BLOCK_K: tl.constexpr):
    """反向: 每 program 一整行循环 h-block. dT1g 内联, dA->dx_s 平铺, dW1/dW2/db 行 partial."""
    pid_row = tl.program_id(0)
    offs_k = tl.arange(0, BLOCK_K)
    offs_w1 = tl.arange(0, BLOCK_W1)
    mk = offs_k < K
    mw1 = offs_w1 < W1N
    base = pid_row * SP
    dw1_acc = tl.zeros((BLOCK_K, BLOCK_W1), dtype=tl.float32)
    dw2_0 = tl.zeros((BLOCK_W1,), dtype=tl.float32)
    dw2_1 = tl.zeros((BLOCK_W1,), dtype=tl.float32)
    dw2_2 = tl.zeros((BLOCK_W1,), dtype=tl.float32)
    db_acc = tl.zeros((BLOCK_W1,), dtype=tl.float32)
    w2u0 = tl.load(W2 + 0 * W1N + offs_w1, mask=mw1, other=0.)
    w2u1 = tl.load(W2 + 1 * W1N + offs_w1, mask=mw1, other=0.)
    w2u2 = tl.load(W2 + 2 * W1N + offs_w1, mask=mw1, other=0.)

    for h0 in tl.range(0, H, BLOCK_H):
        offs_h = h0 + tl.arange(0, BLOCK_H)
        mh = offs_h < H
        a_ptrs = XP + base + offs_h[:, None] + offs_k[None, :]
        A = tl.load(a_ptrs, mask=mh[:, None] & mk[None, :], other=0.0).to(tl.float32)
        w1 = tl.load(W1 + offs_k[:, None] * W1N + offs_w1[None, :],
                     mask=mk[:, None] & mw1[None, :], other=0.0)
        T1 = tl.dot(A, w1, input_precision="tf32")
        b1 = tl.load(B1 + offs_w1, mask=mw1, other=0.)
        T1 = T1 + b1[None, :]
        R = tl.maximum(T1, 0.0)
        gate = tl.where(T1 > 0, 1.0, 0.0)

        dy0 = tl.load(DY + pid_row * H + offs_h + 0 - P, mask=mh & (offs_h - P >= 0) & (offs_h - P < H), other=0.)
        dy1 = tl.load(DY + pid_row * H + offs_h + 1 - P, mask=mh & (offs_h + 1 - P >= 0) & (offs_h + 1 - P < H), other=0.)
        dy2 = tl.load(DY + pid_row * H + offs_h + 2 - P, mask=mh & (offs_h + 2 - P >= 0) & (offs_h + 2 - P < H), other=0.)

        dT1g = gate * (dy0[:, None] * w2u0[None, :]
                       + dy1[:, None] * w2u1[None, :]
                       + dy2[:, None] * w2u2[None, :])

        dA = tl.dot(dT1g, tl.trans(w1), input_precision="tf32")   # [BH, BK]
        tl.store(DX_S + pid_row * K * H + offs_k[None, :] * H + offs_h[:, None],
                 dA, mask=mh[:, None] & mk[None, :])

        dw1_acc += tl.dot(tl.trans(A), dT1g, input_precision="tf32x3")
        dw2_0 += tl.sum(dy0[:, None] * R, axis=0)
        dw2_1 += tl.sum(dy1[:, None] * R, axis=0)
        dw2_2 += tl.sum(dy2[:, None] * R, axis=0)
        db_acc += tl.sum(dT1g, axis=0)

    tl.store(DW1_P + pid_row * BLOCK_K * W1N + offs_k[:, None] * W1N + offs_w1[None, :],
             dw1_acc, mask=mk[:, None] & mw1[None, :])
    tl.store(DW2_P + (pid_row * K + 0) * W1N + offs_w1, dw2_0, mask=mw1)
    tl.store(DW2_P + (pid_row * K + 1) * W1N + offs_w1, dw2_1, mask=mw1)
    tl.store(DW2_P + (pid_row * K + 2) * W1N + offs_w1, dw2_2, mask=mw1)
    tl.store(DB_P + pid_row * W1N + offs_w1, db_acc, mask=mw1)


class _ConvD2Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, win, wo, k1, k2, bias1):
        c, _, kk, _ = k1.shape
        w = win.shape[0]
        p = kk // 2
        dev = x.device
        win_pad = F.pad(win, (p, p))
        wo_pad = F.pad(wo, (p, p))
        w1 = torch.zeros(kk, c, w, device=dev, dtype=torch.float32)
        w2 = torch.zeros(kk, c, w, device=dev, dtype=torch.float32)
        for kk_ in range(kk):
            for j in range(kk):
                v = j - p
                w1[kk_] += k1[:, 0, kk_, j][:, None] * _shift0(win, v)[None, :]
                w2[kk_] += k2[0, :, kk_, j][:, None] * _shift0(wo, -v)[None, :]
        W1 = w1.reshape(kk, c * w)
        W2 = w2.reshape(kk, c * w)
        b1v = bias1[:, None].expand(c, w).reshape(-1).contiguous()

        b, s, h = x.shape
        xp = F.pad(x, (p, p), mode="replicate").contiguous()
        sp = h + 2 * p
        w1n = c * w
        rows = b * s
        y = torch.empty(rows * h, device=dev, dtype=x.dtype)
        t1s = torch.empty(rows * h * w1n, device=dev, dtype=torch.bfloat16)
        BH = 128 if w1n <= 128 else 64
        BW1, BK = w1n, max(triton.next_power_of_2(kk), 16)
        _row_kernel_d2s1[(rows, triton.cdiv(h, BH))](xp, W1, b1v, t1s,
                                                     SP=sp, H=h, W1N=w1n, K=kk,
                                                     BLOCK_H=BH, BLOCK_W1=BW1, BLOCK_K=BK)
        _row_kernel_d2s2[(rows, triton.cdiv(h, BH))](t1s, W2, y,
                                                     H=h, W1N=w1n, K=kk, P=p,
                                                     BLOCK_H=BH, BLOCK_W1=BW1)
        ctx.save_for_backward(xp, win, wo, k1, k2, W1, b1v, W2)
        ctx.sp, ctx.h, ctx.w1n, ctx.k, ctx.p, ctx.c = sp, h, w1n, kk, p, c
        return y.view(b, s, h)

    @staticmethod
    def backward(ctx, dy):
        xp, win, wo, k1, k2, W1, b1v, W2 = ctx.saved_tensors
        sp, h, w1n, k, p, c = ctx.sp, ctx.h, ctx.w1n, ctx.k, ctx.p, ctx.c
        w = w1n // c
        dy = dy.float()
        b, s = dy.shape[0], dy.shape[1]
        rows = b * s
        dev = dy.device
        dy_flat = dy.contiguous().reshape(-1)
        BH = 128 if w1n <= 128 else 64
        BW1, BK = w1n, max(triton.next_power_of_2(k), 16)
        dx_s = torch.empty(rows, k, h, device=dev, dtype=torch.float32)
        dw1_p = torch.empty(rows, BK, w1n, device=dev, dtype=torch.float32)
        dw2_p = torch.empty(rows * k, w1n, device=dev, dtype=torch.float32)
        db_p = torch.empty(rows, w1n, device=dev, dtype=torch.float32)
        _row_kernel_d2b[(rows,)](xp, dy_flat, W1, b1v, W2,
                                 dx_s, dw1_p, dw2_p, db_p,
                                 SP=sp, H=h, W1N=w1n, K=k, P=p,
                                 BLOCK_H=BH, BLOCK_W1=BW1, BLOCK_K=BK,
                                 num_warps=8, num_stages=1)
        dW1 = dw1_p.sum(0)[:k]
        dW2 = dw2_p.view(rows, k, w1n).sum(0)
        db1v_g = db_p.sum(0)
        dxp = torch.empty(rows, sp, device=dev, dtype=torch.float32)
        NP = 256
        _row_kernel_scatter[(rows, triton.cdiv(sp, NP))](dx_s, dxp,
                                                         H=h, K=k, SP=sp, BLOCK_P=NP)
        dx = dxp[:, p:p + h]
        dx[:, 0] += dxp[:, 0]
        dx[:, h - 1] += dxp[:, sp - 1]
        dx = dx.view(b, s, h).to(dy.dtype)
        dW1v = dW1.view(k, c, w1n // c)
        dW2v = dW2.view(k, c, w1n // c)
        dwin = torch.zeros(w, device=dev, dtype=torch.float32)
        dwo = torch.zeros(w, device=dev, dtype=torch.float32)
        dk1 = torch.zeros(c, k, k, device=dev, dtype=torch.float32)
        dk2 = torch.zeros(c, k, k, device=dev, dtype=torch.float32)
        for kk_ in range(k):
            for j in range(k):
                v = j - p
                dk1[:, kk_, j] += (dW1v[kk_] * _shift0(win, v)[None, :]).sum(-1)
                dk2[:, kk_, j] += (dW2v[kk_] * _shift0(wo, -v)[None, :]).sum(-1)
                dwin += _shift0((dW1v[kk_] * k1[:, 0, kk_, j][:, None]).sum(0), -v)
                dwo += _shift0((dW2v[kk_] * k2[0, :, kk_, j][:, None]).sum(0), v)
        db1 = db1v_g.view(c, w1n // c).sum(-1)
        return dx, dwin, dwo, dk1[:, None], dk2[None], db1


class DeepConv2DLinearT(torch.nn.Module):
    """Triton 版: w=64 网格 1->C->1 两层 2D 卷积 (k=3)."""

    def __init__(self, w=64, c=2, k=3):
        super().__init__()
        self.win = torch.nn.Parameter(torch.empty(w))
        self.wo = torch.nn.Parameter(torch.empty(w))
        self.k1 = torch.nn.Parameter(torch.empty(c, 1, k, k))
        self.k2 = torch.nn.Parameter(torch.empty(1, c, k, k))
        self.bias1 = torch.nn.Parameter(torch.zeros(c))
        torch.nn.init.normal_(self.win, 0.0, 0.02)
        torch.nn.init.normal_(self.wo, 0.0, 0.02)
        torch.nn.init.normal_(self.k1, 0.0, 0.02)
        torch.nn.init.normal_(self.k2, 0.0, 0.02)

    def forward(self, x):
        return _ConvD2Fn.apply(x, self.win, self.wo, self.k1, self.k2, self.bias1)


class FeedForwardD2T(torch.nn.Module):
    def __init__(self, hidden_size, w=64, c=2, k=3):
        super().__init__()
        self.ffn1 = DeepConv2DLinearT(w, c, k)
        self.gate = DeepConv2DLinearT(w, c, k)
        self.ffn2 = DeepConv2DLinearT(w, c, k)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.ffn2(self.ffn1(x) * self.relu(self.gate(x)))


def apply_d2t(m, c=2, k=3, w=64):
    dev = next(m.parameters()).device
    for layer in m.decoder_layers:
        layer.ffn = FeedForwardD2T(layer.ffn.ffn1.in_features, w, c, k).to(dev)
    return m
