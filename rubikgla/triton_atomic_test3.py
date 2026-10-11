import torch, triton, triton.language as tl


@triton.jit
def k_dq(SS, DY, DQ, L: tl.constexpr, N: tl.constexpr, VH: tl.constexpr):
    pid = tl.program_id(0)          # vh = pid
    offs_i = tl.arange(0, N)
    offs_jl = tl.arange(0, VH)
    offs_js = offs_jl + pid * VH
    for t in range(L - 1, -1, -1):
        Sp = tl.load(SS + pid * L * N * VH + t * N * VH
                     + offs_i[:, None] * VH + offs_jl[None, :]).to(tl.float32)
        dym = tl.load(DY + t * 2 * VH + offs_js).to(tl.float32)
        tl.atomic_add(DQ + t * N + offs_i, tl.sum(dym[None, :] * Sp, axis=1))


torch.manual_seed(0)
L, N, VH = 4, 8, 4
SS = torch.randn(L, N, VH, device="cuda").repeat(2, 1, 1, 1).view(2, L, N, VH)
DY = torch.randn(L, 2 * VH, device="cuda")
DQ = torch.zeros(L, N, device="cuda")
k_dq[(2,)](SS, DY, DQ, L=L, N=N, VH=VH)

ref = torch.zeros(L, N, device="cuda")
for t in range(L):
    ref[t] = DY[t, 0:4] @ SS[0, t].T + DY[t, 4:8] @ SS[1, t].T
print("DQ vs ref max diff:", (DQ - ref).abs().max().item())
print("DQ[0,:4]:", [round(x, 3) for x in DQ[0, :4].tolist()])
print("ref[0,:4]:", [round(x, 3) for x in ref[0, :4].tolist()])
