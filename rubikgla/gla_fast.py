#!/usr/bin/env python3
"""快速对角 GLA 扫描 (全向量化, 无 Python 循环).
S_t = Σ_{i<=t} exp(cl_t - cl_i) ⊙_row k_i v_i^T,  cl = cumsum(log λ)
λ<=1 ⇒ cl 单调降 ⇒ exp(cl_t-cl_i)<=1 对 t>=i 恒成立 — 数值安全, 无需分块."""
import sys, time
import torch

sys.path.insert(0, r"F:\夸克\rubikgla")
from scan import affine_scan

DEV = "cuda"
B, L, H, dh = 16, 256, 10, 32


def gla_fast(k, v, lam, q):
    """k,v,q: (B,L,H,dh); lam: (B,L,H,dh) in (0,1). 返回 o: (B,L,H,dh).
    对角结构下状态行解耦: o[t,h,c] = Σ_i (Σ_j q·M·k)·v[i,h,c] — 先融合 j, matmul 无 dh^2."""
    a = torch.log(lam.clamp(min=1e-6))                 # (B,L,H,dh)
    cl = torch.cumsum(a, dim=1)                         # (B,L,H,dh)
    diff = cl.unsqueeze(2) - cl.unsqueeze(1)            # (B,t,i,H,dh)
    # 有效位 (t>=i) 时 cl_t<=cl_i ⇒ diff<=0; 无效位 clamp 后 exp=1, 再乘掩码归零 — 永不溢出
    M = torch.exp(diff.clamp(max=0.0))
    L_ = k.shape[1]
    mask = torch.tril(torch.ones(L_, L_, device=k.device, dtype=torch.bool))
    M = M * mask[None, :, :, None, None]
    # u[t,i,h] = Σ_j q[t,h,j]·M[t,i,h,j]·k[i,h,j]  -> (B,T,I,H)
    u = (M * q.unsqueeze(2) * k.unsqueeze(1)).sum(-1)
    # o[t,h,c] = Σ_i u[t,i,h]·v[i,h,c]
    o = torch.einsum("btih,bihc->bthc", u, v)
    return o


def gla_ref(k, v, lam, q):
    """参考: 逐 token 递归."""
    B, L, H, dh = k.shape
    S = torch.zeros(B, H, dh, dh, device=k.device, dtype=k.dtype)
    outs = []
    for t in range(L):
        S = lam[:, t].unsqueeze(-1) * S + k[:, t].unsqueeze(-1) * v[:, t].unsqueeze(-2)
        outs.append(torch.einsum("bhj,bhjc->bhc", q[:, t], S))
    return torch.stack(outs, 1)


if __name__ == "__main__":
    torch.manual_seed(0)
    k = torch.randn(B, L, H, dh, device=DEV, dtype=torch.bfloat16) * 0.5
    v = torch.randn(B, L, H, dh, device=DEV, dtype=torch.bfloat16) * 0.5
    q = torch.randn(B, L, H, dh, device=DEV, dtype=torch.bfloat16) * 0.5
    lam = torch.sigmoid(torch.randn(B, L, H, dh, device=DEV, dtype=torch.bfloat16) * 1.0 - 1.0)

    # 正确性 (小尺寸 fp32)
    k32, v32, q32, lam32 = (t.float()[:2, :64] for t in (k, v, q, lam))
    o_fast = gla_fast(k32, v32, lam32, q32)
    o_ref = gla_ref(k32, v32, lam32, q32)
    print("正确性 max diff: %.2e" % (o_fast - o_ref).abs().max().item())

    # 速度: fwd+bwd @ B16 L256 — fp32
    kk = k.float().requires_grad_(True)
    vv = v.float().requires_grad_(True)
    qq = q.float().requires_grad_(True)
    ll = lam.float().requires_grad_(True)
    o = gla_fast(kk, vv, ll, qq)
    o.pow(2).mean().backward()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(5):
        o = gla_fast(kk, vv, ll, qq)
        o.pow(2).mean().backward()
    torch.cuda.synchronize()
    print("gla_fast fwd+bwd fp32: %.0f ms (B16 L256)" % ((time.time() - t0) / 5 * 1000))

    # bf16 口径 (训练实际走这条路)
    kb = k.detach().requires_grad_(True)
    vb = v.detach().requires_grad_(True)
    qb = q.detach().requires_grad_(True)
    lb = lam.detach().requires_grad_(True)
    ob = gla_fast(kb, vb, lb, qb)
    ob.pow(2).mean().backward()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(5):
        ob = gla_fast(kb, vb, lb, qb)
        ob.pow(2).mean().backward()
    torch.cuda.synchronize()
    print("gla_fast fwd+bwd bf16: %.0f ms (B16 L256)" % ((time.time() - t0) / 5 * 1000))

    # bf16 + 梯度检查点口径 (重算成本)
    from torch.utils.checkpoint import checkpoint
    kk = k.detach().requires_grad_(True)
    vv = v.detach().requires_grad_(True)
    qq = q.detach().requires_grad_(True)
    ll = lam.detach().requires_grad_(True)
    oc = checkpoint(gla_fast, kk, vv, ll, qq, use_reentrant=False)
    oc.pow(2).mean().backward()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(5):
        oc = checkpoint(gla_fast, kk, vv, ll, qq, use_reentrant=False)
        oc.pow(2).mean().backward()
    torch.cuda.synchronize()
    print("gla_fast ckpt bf16:    %.0f ms (B16 L256)" % ((time.time() - t0) / 5 * 1000))

    # 对照: 稠密 HS 扫描 bf16
    eye = torch.eye(dh, device=DEV).view(1, 1, 1, dh, dh)
    Ag = (lam.float().unsqueeze(-1) * eye).to(torch.bfloat16).requires_grad_(True)
    W_ = (k.float().unsqueeze(-1) * v.float().unsqueeze(-2)).to(torch.bfloat16).requires_grad_(True)
    S = affine_scan(Ag, W_)[1]
    oo = torch.einsum("nlhp,nlhpq->nlhq", q.float().to(torch.bfloat16), S)
    oo.float().pow(2).mean().backward()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(5):
        S = affine_scan(Ag, W_)[1]
        oo = torch.einsum("nlhp,nlhpq->nlhq", q.float().to(torch.bfloat16), S)
        oo.float().pow(2).mean().backward()
    torch.cuda.synchronize()
    print("dense-HS 对照:   %.0f ms" % ((time.time() - t0) / 5 * 1000))
