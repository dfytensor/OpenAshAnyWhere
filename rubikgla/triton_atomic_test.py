import torch, triton, triton.language as tl


@triton.jit
def k_add(OUT, N: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, N)
    tl.atomic_add(OUT + offs, tl.full((N,), 1.0, tl.float32) + pid * 0.0)


out = torch.zeros(32, device="cuda", dtype=torch.float32)
k_add[(2,)](out, N=32)
print("cross-CTA atomic: got", out[0].item(), "(expect 2.0)")
out2 = torch.zeros(32, device="cuda", dtype=torch.float32)
k_add[(1,)](out2, N=32)
print("single-CTA atomic: got", out2[0].item(), "(expect 1.0)")

# 带循环 + 多元素的复现 (贴近 bwd kernel 用法)
@triton.jit
def k_add2(OUT, L: tl.constexpr, N: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, N)
    for t in range(L - 1, -1, -1):
        tl.atomic_add(OUT + t * N + offs, tl.full((N,), 0.5, tl.float32) * (1 + pid))


ALT = torch.zeros(4, 32, device="cuda", dtype=torch.float32)
k_add2[(4,)](ALT, L=4, N=32)
print("4 CTA x L4 loop: got", ALT[0, 0].item(), "(expect 0.5*(1+0+1+2+3)=3.5)")
