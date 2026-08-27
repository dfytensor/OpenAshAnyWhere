"""ConvLinear 重排融合: 数学等价变换, 消除 [b,s,h,w] 中间张量物化.

原版: full[b,s,h,w] = x[b,s,h]·w_in[w]
      conv3x3 over (h,w) -> act -> @w_out -> [b,s,h]
变换: Σ_dw K[dh,dw]·w_in[w+dw-p] = Kw[dh,w]  (预计算 [k,w])
      out[b,s,h] = Σ_w w_out[w] · act( Σ_dh Kw[dh,w] · x[b,s,h+dh] )
=> 只需 h 维 3-tap 位移加权 + 逐 w 缩放激活 + 收缩, 无大中间物化 (k 次 [b,s,h,w] 累加).
边界: replicate-pad (原版 zero-pad, 差异仅在 h 边界 1-2 元素).
"""
import torch


def make_Kw(w_in, conv_weight):
    """预计算 Kw[k, w] = Σ_dw K[dh,dw]·w_in[(w+dw-p) mod w] (w 维 circular)."""
    K = conv_weight[0, 0].float()                     # [k, k]
    k = K.shape[0]
    p = k // 2
    w = w_in.shape[-1]
    dev = w_in.device
    win = w_in.float().reshape(-1)                     # [w] (原 [1,w])
    # shift 矩阵: idx[dw, w] = (w + dw - p) mod w
    dw = torch.arange(k, device=dev).view(-1, 1) + torch.zeros(1, w, device=dev, dtype=torch.long)
    wv = torch.zeros(k, 1, device=dev, dtype=torch.long) + torch.arange(w, device=dev).view(1, -1)
    idx = ((wv + dw - p) % w).reshape(-1)
    gathered = win[idx].reshape(k, w)
    Kw = K @ gathered                                 # [k,k]@[k,w] = [k,w]
    return Kw


def convlinear_fused(x, w_in, w_out, conv_weight, conv_bias=None, act="relu"):
    """x: [b,s,h] -> [b,s,h]. 与 ConvLinear 数学等价 (h 边界 replicate vs zero).
    输入权重自动搬到 x 的设备 (nn.Parameter 可能留在 cpu)."""
    dev = x.device
    w_in = w_in.to(dev, non_blocking=False)
    w_out = w_out.to(dev, non_blocking=False)
    conv_weight = conv_weight.to(dev, non_blocking=False)
    if conv_bias is not None:
        conv_bias = conv_bias.to(dev, non_blocking=False)
    b, s, h = x.shape
    Kw = make_Kw(w_in, conv_weight)                   # [k, w]
    k = Kw.shape[0]
    p = k // 2
    w = w_out.shape[0]
    xf = x.float()
    xp = torch.nn.functional.pad(xf, (p, p), mode="replicate")    # [b,s,h+2p]
    # terms[b,s,h,w] = Σ_dh Kw[dh,w] · xp[b,s,h+dh]  —— 用 k 次广播乘累加
    terms = torch.zeros(b, s, h, w, device=x.device, dtype=torch.float32)
    for dh in range(k):
        terms += Kw[dh].view(1, 1, 1, w) * xp[:, :, dh:dh + h].unsqueeze(-1)
    if conv_bias is not None:
        terms = terms + conv_bias.float()
    y = torch.relu(terms) if act == "relu" else terms
    out = (y * w_out.float().view(1, 1, 1, w)).sum(-1)
    return out.to(x.dtype)
