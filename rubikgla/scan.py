# 仿射前缀扫描: S_t = A_t S_{t-1} + B_t 的 log(L) 趟并行化
# A_t = diag(lam) @ G_t (旋转x衰减, 谱范数<=1, 乘积有界 => 数值稳定)
# 复合律: (A2,B2)∘(A1,B1) = (A2@A1, A2@B1+B2) — 可结合 => Hillis-Steele
import torch
import torch.nn.functional as F


def affine_scan(A, Bw):
    """A,Bw: (B,L,H,dh,dh) -> 包含式前缀复合 (A*, B*), fp32, 全批量."""
    B, L, H = A.shape[:3]
    d = 1
    while d < L:
        A_t, B_t = A[:, d:], Bw[:, d:]
        A_s, B_s = A[:, :-d], Bw[:, :-d]
        A_new = A_t @ A_s
        B_new = A_t @ B_s + B_t
        A = torch.cat([A[:, :d], A_new], 1)
        Bw = torch.cat([Bw[:, :d], B_new], 1)
        d *= 2
    return A, Bw


def rubik_scan_forward(G, lam, W, q):
    """G: (B,L,H,dh,dh) 旋转; lam: (1,H,dh,1) or None; W: (B,L,H,dh,dh) 写入;
    q: (B,L,H,dh). 返回 o (B,L,H,dh). 语义与逐步循环一致:
    S_t = lam*(G_t S_{t-1}) + W_t ;  o_t = q_t S_t."""
    A = G if lam is None else lam * G
    Bw = W
    A_star, B_star = affine_scan(A, Bw)
    # S_t = A*_t @ S_0 + B*_t, S_0 = 0  =>  S_t = B*_t
    o = torch.einsum("nlhp,nlhpq->nlhq", q, B_star)
    return o


def rubik_loop_forward(G, lam, W, q):
    """逐步参考实现 (与 models.RubikLayer 语义一致)."""
    B, L, H, dh, _ = G.shape
    state = G.new_zeros(B, H, dh, dh)
    outs = []
    for t in range(L):
        state = G[:, t] @ state
        if lam is not None:
            state = lam * state
        state = state + W[:, t]
        outs.append((q[:, t].unsqueeze(-2) @ state).squeeze(-2))
    return torch.stack(outs, 1)


def check(B=4, L=128, H=4, dh=16, seed=0):
    torch.manual_seed(seed)
    M = torch.randn(B, L, H, dh, dh) * 0.1
    G = torch.matrix_exp(M - M.transpose(-1, -2))          # SO(dh)
    lam = torch.sigmoid(torch.randn(1, H, dh, 1) * 0.5)
    W = torch.randn(B, L, H, dh, dh) * 0.5
    q = torch.randn(B, L, H, dh) * 0.5
    o_scan = rubik_scan_forward(G, lam, W, q)
    o_loop = rubik_loop_forward(G, lam, W, q)
    err = (o_scan - o_loop).abs().max().item()
    print("scan vs loop max diff: %.2e  (o norm %.2f)" % (err, o_loop.norm().item()))
    return err


def bench(B=64, L=128, H=8, dh=32, iters=10):
    torch.manual_seed(0)
    M = torch.randn(B, L, H, dh, dh) * 0.1
    G = torch.matrix_exp(M - M.transpose(-1, -2))
    lam = torch.sigmoid(torch.randn(1, H, dh, 1))
    W = torch.randn(B, L, H, dh, dh) * 0.5
    q = torch.randn(B, L, H, dh) * 0.5
    for tag, fn in [("loop", rubik_loop_forward), ("scan", rubik_scan_forward)]:
        G_ = G.detach().clone().requires_grad_(True)
        W_ = W.detach().clone().requires_grad_(True)
        o = fn(G_, lam, W_, q)
        loss = o.float().pow(2).mean()
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(iters):
            o = fn(G_, lam, W_, q)
            o.float().pow(2).mean().backward()
        torch.cuda.synchronize()
        print("%s B=%d L=%d: %.0f ms/fwd+bwd" % (tag, B, L, (time.time() - t0) / iters * 1000))


import time
if __name__ == "__main__":
    err = check()
    assert err < 1e-3, "FAIL"
    print("PASS")
    if torch.cuda.is_available():
        bench(B=64, L=256, H=8, dh=32, iters=3)
        bench(B=64, L=512, H=8, dh=32, iters=3)
        bench(B=128, L=512, H=4, dh=16, iters=3)
