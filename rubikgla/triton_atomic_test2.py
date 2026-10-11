import torch, triton, triton.language as tl


@triton.jit
def k_c(OUT, N: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, N)
    tl.atomic_add(OUT + offs, tl.full((N,), 1.0, tl.float32))


@triton.jit
def k_l(OUT, VALS, N: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, N)
    val = tl.load(VALS + pid * N + offs).to(tl.float32)
    tl.atomic_add(OUT + offs, val)


@triton.jit
def k_l2(OUT, VALS, N: tl.constexpr):
    """仿 kernel: 值来自求和 而非常数."""
    pid = tl.program_id(0)
    offs = tl.arange(0, N)
    vals2 = tl.load(VALS + pid * N * 2 + tl.arange(0, 2 * N)).to(tl.float32)
    mat = vals2.reshape(2, N)          # 模拟 (2, N)
    val = tl.sum(mat * 2.0, axis=0)    # 求和回归 (N,)
    tl.atomic_add(OUT + offs, val)


out = torch.zeros(8, device="cuda", dtype=torch.float32)
k_c[(2,)](out, N=8)
print("常量值 2CTA:", out[0].item(), "(expect 2.0)")

vals = torch.zeros(2, 8)
vals[0] = 1.0
vals[1] = 2.0
out = torch.zeros(8, device="cuda", dtype=torch.float32)
k_l[(2,)](out, vals.cuda(), N=8)
print("加载值 2CTA:", out[0].item(), "(expect 3.0)")

out = torch.zeros(8, device="cuda", dtype=torch.float32)
k_l2[(2,)](out, vals.cuda(), N=8)
print("求和值 2CTA:", out[0].item(), "(expect (1+2)*2=6.0)")
