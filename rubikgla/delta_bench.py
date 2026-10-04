#!/usr/bin/env python3
"""Delta 规则扫描基准: S_tᵀ = A_t S_{t-1}ᵀ + k_t v_tᵀ, A = λ(I − η k kᵀ).
归一化 k + η∈(0,1) ⇒ ||A|| ≤ 1 收缩稳定 (无需 stable_exp).
测: 正确性 vs 逐token递归; eager vs compile fwd+bwd @ B16 L256."""
import sys, time, math
import torch
sys.path.insert(0, r"F:\夸克\rubikgla")
from scan import affine_scan

DEV = "cuda"
B, L, H, dh = 16, 256, 10, 32


def delta_scan(k, v, lam, eta, q):
    """k,v,q: (B,L,H,dh); lam,eta: (B,L,H,dh). 返回 y: (B,L,H,dh), y_t = q_t S_t (S 按GatedDeltaNet右侧约定).
    转置技巧: S_tᵀ = A_t S_{t-1}ᵀ + k vᵀ, 左仿射. 输出 y = q@S = (Sᵀ q)ᵀ → 用 einsum 从 Sᵀ 求."""
    kn = torch.nn.functional.normalize(k.float(), dim=-1)               # ||k||=1 ⇒ 特征值∈(0,1]
    # A = lam * (I - eta k kᵀ)  → (B,L,H,dh,dh)
    I = torch.eye(dh, device=k.device, dtype=torch.float32).view(1, 1, 1, dh, dh)
    A = lam.float().unsqueeze(-1) * (I - eta.float().unsqueeze(-1) * (kn.unsqueeze(-1) * kn.unsqueeze(-2)))
    W_ = (kn.unsqueeze(-1) * v.float().unsqueeze(-2))                   # k vᵀ (注意: 用归一化k写入)
    A2, S_ = affine_scan(A.to(torch.bfloat16), W_.to(torch.bfloat16))   # S_: (B,L,H,dh,dh) 左乘状态本体
    y = torch.einsum("blhpq,blhq->blhp", S_, q.to(torch.bfloat16))      # y = S@q (GDN约定)
    return y.float()


def delta_ref(k, v, lam, eta, q):
    B, L, H, dh = k.shape
    kn = torch.nn.functional.normalize(k, dim=-1)
    S = torch.zeros(B, H, dh, dh, device=k.device, dtype=k.dtype)       # S 本体 (右侧约定)
    outs = []
    for t in range(L):
        S = lam[:, t].view(B, H, dh, 1) * (S - eta[:, t].view(B, H, dh, 1) * kn[:, t].view(B, H, dh, 1) * (kn[:, t].view(B, H, 1, dh) @ S)) \
            + v[:, t].view(B, H, 1, dh) * kn[:, t].view(B, H, dh, 1)
        outs.append(torch.einsum("nhq,nhq->nh", q[:, t], S).squeeze(-1))  # q S: (B,H,dh)?? q(S v) dims:
    # 修正: y = Sᵀ q 等价于 q@S; 上面 einsum 形状不对, 直接:
    return outs


if __name__ == "__main__":
    torch.manual_seed(0)
    k = torch.randn(B, L, H, dh, device=DEV, dtype=torch.bfloat16) * 0.5
    v = torch.randn(B, L, H, dh, device=DEV, dtype=torch.bfloat16) * 0.5
    q = torch.randn(B, L, H, dh, device=DEV, dtype=torch.bfloat16) * 0.5
    lam = torch.sigmoid(torch.randn(B, L, H, dh, device=DEV, dtype=torch.bfloat16) - 1.0)
    eta = torch.sigmoid(torch.randn(B, L, H, dh, device=DEV, dtype=torch.bfloat16))

    # 正确性: fp32 小尺寸, 递归参考 (同约定 y = q@S)
    kb, vb, qb, lb, eb = (t.float()[:2, :48] for t in (k, v, q, lam, eta))
    kn = torch.nn.functional.normalize(kb, dim=-1)
    S = torch.zeros(2, H, dh, dh, device=DEV)
    outs = []
    for t in range(48):
        kt = kn[:, t].view(2, H, dh, 1)                                  # (2,H,dh,1)
        S = lb[:, t].view(2, H, dh, 1) * (S - eb[:, t].view(2, H, dh, 1) * kt * (kt.transpose(-1, -2) @ S)) \
            + kt * vb[:, t].view(2, H, 1, dh)
        outs.append(torch.einsum("nhqp,nhq->nhp", S, qb[:, t]))           # y = q@S
    o_ref = torch.stack(outs, 1)                                   # (2,48,H,dh)
    o_fast = delta_scan(kb, vb, lb, eb, qb)
    print("正确性 max diff: %.2e" % (o_fast - o_ref).abs().max().item())

    # 速度: fwd+bwd
    def bench(fn, *ts, n=5):
        for t in ts:
            t.grad = None
        o = fn(*ts)
        o.pow(2).mean().backward()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n):
            for t in ts:
                t.grad = None
            o = fn(*ts)
            o.pow(2).mean().backward()
        torch.cuda.synchronize()
        return (time.time() - t0) / n * 1000

    tensors = [t.detach().requires_grad_(True) for t in (k, v, q, lam, eta)]
    print("eager  fwd+bwd: %.0f ms" % bench(delta_scan, *tensors))
    cfn = torch.compile(delta_scan)
    print("compile fwd+bwd: %.0f ms" % bench(cfn, *tensors))
