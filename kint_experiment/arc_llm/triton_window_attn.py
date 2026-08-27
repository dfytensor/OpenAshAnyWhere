"""Loop 7: Triton 融合滑动窗口因果注意力 kernel (O(T·W)).
每 program 处理一个 (batch*head, query 块), 一次加载整个 key 窗口(WKV), 直接 softmax(无需 FlashAttn 在线循环).
先严格验正确性(vs PyTorch band-mask 参考), 再测速(vs torch 融合 sdpa).
"""
import torch, triton, triton.language as tl, math, time


@triton.jit
def _wa_fwd(Q, K, V, Out, sm_scale, W: tl.constexpr,
            stride_z, stride_t, T,
            BQ: tl.constexpr, WKV: tl.constexpr, HD: tl.constexpr):
    z = tl.program_id(0)      # batch*head
    qb = tl.program_id(1)     # query 块索引
    i = qb * BQ
    offs_d = tl.arange(0, HD)
    offs_q = i + tl.arange(0, BQ)
    q_ptrs = Q + z*stride_z + offs_q[:, None]*stride_t + offs_d[None, :]
    q = tl.load(q_ptrs, mask=offs_q[:, None] < T, other=0.0)        # (BQ, HD)
    # key 窗口 [j0, j0+WKV)
    j0 = tl.maximum(0, i + BQ - WKV)
    offs_kv = j0 + tl.arange(0, WKV)
    kvmask = offs_kv < T
    kv_ptrs_base = z*stride_z + offs_kv[:, None]*stride_t + offs_d[None, :]
    k = tl.load(K + kv_ptrs_base, mask=kvmask[:, None], other=0.0)  # (WKV, HD)
    v = tl.load(V + kv_ptrs_base, mask=kvmask[:, None], other=0.0)
    s = tl.dot(q, tl.trans(k)) * sm_scale                            # (BQ, WKV)
    qp = offs_q[:, None]; kp = offs_kv[None, :]
    valid = (kp <= qp) & ((qp - kp) <= W) & (offs_q[:, None] < T) & kvmask[None, :]
    s = tl.where(valid, s, float('-inf'))
    # 数值稳定 softmax
    m = tl.max(s, axis=1)                                            # (BQ,)
    e = tl.exp(s - m[:, None])
    e = e / tl.sum(e, axis=1)[:, None]
    o = tl.dot(e.to(v.dtype), v)                                     # (BQ, HD)
    o_ptrs = Out + z*stride_z + offs_q[:, None]*stride_t + offs_d[None, :]
    tl.store(o_ptrs, o, mask=(offs_q[:, None] < T))


def window_attn(q, k, v, W, BQ=64, WKV=256):
    """q,k,v: (B,H,T,HD). 返回 (B,H,T,HD). 滑动窗口因果注意力, O(T·W)."""
    B, H, T, HD = q.shape
    assert HD == 32, 'kernel 写死 HD=32(d=256,h=8)'
    Z = B * H
    q2 = q.reshape(Z, T, HD).contiguous(); k2 = k.reshape(Z, T, HD).contiguous(); v2 = v.reshape(Z, T, HD).contiguous()
    out = torch.empty_like(q2)
    grid = (Z, triton.cdiv(T, BQ))
    _wa_fwd[grid](q2, k2, v2, out, 1.0/math.sqrt(HD), W,
                  q2.stride(0), q2.stride(1), T, BQ=BQ, WKV=WKV, HD=HD)
    return out.reshape(B, H, T, HD)


def ref_window_attn(q, k, v, W):
    """PyTorch 参考: 全 T×T band-mask 窗口因果注意力."""
    B, H, T, HD = q.shape
    s = (q @ k.transpose(-1, -2)) / math.sqrt(HD)                    # (B,H,T,T)
    idx = torch.arange(T, device=q.device)
    rel = idx[None, :] - idx[:, None]
    mask = (rel > 0) | (rel < -W)
    s = s.masked_fill(mask, float('-inf'))
    return torch.softmax(s, -1) @ v


if __name__ == '__main__':
    DEV = 'cuda'
    # === 1. 正确性 ===
    torch.manual_seed(0)
    B, H, T, HD, W = 2, 8, 200, 32, 64
    q = torch.randn(B, H, T, HD, device=DEV) * 0.5
    k = torch.randn(B, H, T, HD, device=DEV) * 0.5
    v = torch.randn(B, H, T, HD, device=DEV) * 0.5
    o_tri = window_attn(q, k, v, W, WKV=128)
    o_ref = ref_window_attn(q, k, v, W)
    err = (o_tri - o_ref).abs().max().item()
    print(f'correctness: max|triton-ref| = {err:.2e}  ({"PASS(tf32)" if err < 2e-3 else "FAIL"})')
    # 边界: 末尾 query 块不足 BQ
    B2, T2 = 2, 70
    q2 = torch.randn(B2, 8, T2, 32, device=DEV)*0.5; k2=torch.randn_like(q2); v2=torch.randn_like(q2)
    err2=(window_attn(q2,k2,v2,64,WKV=128)-ref_window_attn(q2,k2,v2,64)).abs().max().item()
    print(f'boundary(T=70): max err = {err2:.2e}  ({"PASS(tf32)" if err2<2e-3 else "FAIL"})')
    # === 2. 速度: Triton 窗口 O(T·W) vs torch 融合 sdpa O(T²) ===
    print('\nspeed (infer fwd, bs=4, h=8, hd=32, W=128):')
    for T in [1024, 2048, 4096, 8192]:
        q=torch.randn(4,8,T,32,device=DEV)*0.5; k=torch.randn_like(q); v=torch.randn_like(q)
        # warmup triton
        for _ in range(3): window_attn(q,k,v,128)
        torch.cuda.synchronize(); t0=time.time()
        for _ in range(10): window_attn(q,k,v,128)
        torch.cuda.synchronize(); dt_tri=(time.time()-t0)/10
        # torch sdpa (full softmax O(T²))
        qh=q.transpose(1,2); kh=k.transpose(1,2); vh=v.transpose(1,2)  # (B,T,H,HD)
        for _ in range(3): torch.nn.functional.scaled_dot_product_attention(qh,kh,vh,is_causal=True)
        torch.cuda.synchronize(); t0=time.time()
        for _ in range(10): torch.nn.functional.scaled_dot_product_attention(qh,kh,vh,is_causal=True)
        torch.cuda.synchronize(); dt_sdpa=(time.time()-t0)/10
        print(f'  T={T:>5}: triton_window={dt_tri*1000:>6.1f}ms  sdpa_full={dt_sdpa*1000:>6.1f}ms  speedup={dt_sdpa/dt_tri:.2f}x')
