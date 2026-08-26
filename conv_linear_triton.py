"""ConvLinear 的 Triton 融合 kernel: 整条链 (Kw 加权3-tap + ReLU + 收缩) 单 kernel.

关键: w 维完全不物化 —— 每个 program 处理 [BLOCK_SH] 个 (s,h) 位置,
w 维 (96) 在寄存器/共享内存中循环, 输出直接是收缩后的标量.

数学: out[b,s,h] = Σ_w w_out[w] · relu( Σ_dh Kw[dh,w] · x[b,s,h+dh] + bias )
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _convlinear_kernel(XP, KW, WOUT, BIAS, Y,
                       N, H: tl.constexpr, W: tl.constexpr, K: tl.constexpr,
                       BLOCK_N: tl.constexpr, BLOCK_W: tl.constexpr):
    """XP: [b, s, H+2p] (replicate padded); 每 program 处理 BLOCK_N 个 (b*s, h)."""
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_w = tl.arange(0, BLOCK_W)
    mn = offs_n < N
    mw = offs_w < W

    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for w0 in range(0, W, BLOCK_W):
        wv = w0 + offs_w
        wmask = wv < W
        # Σ_dh Kw[dh,w] · x[n, h+dh]  (h = n % H, dh 偏移在 XP 上)
        # 注: 每 n 的 h 不同 -> 直接对 dh 循环, 每次 gather x[n + dh*strideH]
        s_dh = tl.zeros([BLOCK_N, BLOCK_W], dtype=tl.float32)
        for dh in range(K):
            kw = tl.load(KW + dh * W + wv, mask=wmask, other=0.)
            xv = tl.load(XP + offs_n + dh, mask=mn, other=0.)
            s_dh += kw[None, :] * xv[:, None]
        bias = tl.load(BIAS, mask=wmask, other=0.)
        s_dh += bias[None, :]
        s_dh = tl.maximum(s_dh, 0.0)
        wout = tl.load(WOUT + wv, mask=wmask, other=0.)
        acc += tl.sum(s_dh * wout[None, :], axis=1)
    tl.store(Y + offs_n, acc, mask=mn)


def convlinear_triton(x, w_in, w_out, conv_weight, conv_bias=None):
    b, s, h = x.shape
    k = conv_weight.shape[-1]
    p = k // 2
    w = w_out.shape[0]
    # Kw[k,w] = K @ shifted w_in (circular)
    K = conv_weight[0, 0].float()
    win = w_in.float().reshape(-1)
    idx = ((torch.arange(w, device=x.device).view(1, -1)
            + torch.arange(k, device=x.device).view(-1, 1) - p) % w)
    Kw = (K @ win[idx]).contiguous()
    xp = torch.nn.functional.pad(x.float(), (p, p), mode="replicate").contiguous()
    # 展平 [b*s, h] -> 输出按行主序; 注意 dh 偏移跨行 (h+dh 越行) 需按整体展平处理:
    # 为正确性: 对每行独立 —— 直接用 3D 展平 [b, s, h+2p], 输出 [b, s, h]
    y = torch.empty(b * s * h, device=x.device, dtype=torch.float32)
    # 手动按 (b*s, h) 索引: XP 展平后 x[b, s, hh] = XP_flat[(b*s)*(h+2p) + hh]
    # kernel 需知道行距 -> 简化: 逐 s 行调用代价高; 改为 kernel 内计算行号
    # 这里用简单方案: 每 program 处理一行 (b*s) 的 BLOCK_N 个 h
    return _launch(xp, Kw, w_out, conv_bias, y, b, s, h, w, k, p)


def _launch(xp, Kw, w_out, conv_bias, y, b, s, h, w, k, p):
    raise NotImplementedError  # 由下方的行式 kernel 替代


@triton.jit
def _convlinear_row_kernel(XP, KW, WOUT, BIAS, Y,
                           SP: tl.constexpr, H: tl.constexpr,
                           W: tl.constexpr, K: tl.constexpr, BLOCK_H: tl.constexpr):
    """每 program 处理一行 (b*s) 的 BLOCK_H 个 h. XP: [b*s, H+2p]."""
    pid_row = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_w = tl.arange(0, W)
    mh = offs_h < H

    base = pid_row * SP + offs_h          # SP = H + 2p
    acc = tl.zeros([BLOCK_H], dtype=tl.float32)
    wout = tl.load(WOUT + offs_w)
    bias = tl.load(BIAS + offs_w) if BIAS is not None else tl.zeros([W], dtype=tl.float32)
    for dh in range(K):
        kw = tl.load(KW + dh * W + offs_w)
        xv = tl.load(XP + base + dh, mask=mh, other=0.)
        # 此处仅一个 dh 的贡献; 需先累计所有 dh 再过 relu —— 改为两段
        pass
    return acc


def convlinear_triton_v2(x, w_in, w_out, conv_weight, conv_bias=None):
    """行式 Triton: 两阶段 kernel (先 Σ_dh 存中间? 不物化 w -> 直接在寄存器全 W)."""
    b, s, h = x.shape
    k = conv_weight.shape[-1]
    p = k // 2
    w = w_out.shape[0]
    dev = x.device
    K = conv_weight[0, 0].float()
    win = w_in.float().reshape(-1)
    idx = ((torch.arange(w, device=dev).view(1, -1)
            + torch.arange(k, device=dev).view(-1, 1) - p) % w)
    Kw = (K @ win[idx]).contiguous()                     # [k, w]
    wout_c = w_out.float().reshape(-1).contiguous()
    bias_c = (conv_bias.float().reshape(-1).contiguous()
              if conv_bias is not None else torch.zeros(1, device=dev))
    xp = torch.nn.functional.pad(x.float(), (p, p), mode="replicate").contiguous()
    sp = h + 2 * p
    y = torch.empty(b * s * h, device=dev, dtype=torch.float32)
    BLOCK_H = 64
    grid = (b * s, triton.cdiv(h, BLOCK_H))
    _row_kernel[grid](xp, Kw, wout_c, bias_c, y,
                      rows=b * s, SP=sp, H=h, W=w, K=k,
                      BLOCK_H=BLOCK_H, BLOCK_W=triton.next_power_of_2(w))
    return y.view(b, s, h).to(x.dtype)


@triton.jit
def _row_kernel(XP, KW, WOUT, BIAS, Y, rows,
                SP: tl.constexpr, H: tl.constexpr, W: tl.constexpr, K: tl.constexpr,
                BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr):
    pid_row = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_w = tl.arange(0, BLOCK_W)
    mh = offs_h < H
    mw = offs_w < W

    base = pid_row * SP
    # Σ_dh Kw[dh,w]·x[h+dh] : [BLOCK_H, BLOCK_W]
    sdh = tl.zeros([BLOCK_H, BLOCK_W], dtype=tl.float32)
    for dh in range(K):
        kw = tl.load(KW + dh * W + offs_w, mask=mw, other=0.)
        xv = tl.load(XP + base + offs_h + dh, mask=mh, other=0.)
        sdh += kw[None, :] * xv[:, None]
    bias = tl.load(BIAS + offs_w, mask=mw, other=0.)
    sdh += bias[None, :]
    sdh = tl.maximum(sdh, 0.0)
    wout = tl.load(WOUT + offs_w, mask=mw, other=0.)
    out = tl.sum(sdh * wout[None, :], axis=1)
    tl.store(Y + pid_row * H + offs_h, out, mask=mh)
