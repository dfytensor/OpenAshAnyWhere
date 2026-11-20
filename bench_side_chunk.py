import sys
sys.path.insert(0, r'/mnt/f/OpenASH2605')
import time
import torch
from fla.ops.delta_rule import chunk_delta_rule
from fla.ops.gated_delta_rule import chunk_gated_delta_rule

DEV = 'cuda'
B, S, H, D = 32, 383, 4, 96
x = torch.randn(B, S, 768, device=DEV, dtype=torch.bfloat16, requires_grad=True)

dm_k = torch.nn.Linear(768, H * D, bias=False).to(DEV).to(torch.bfloat16)
dm_q = torch.nn.Linear(768, H * D, bias=False).to(DEV).to(torch.bfloat16)
dm_v = torch.nn.Linear(768, H * D, bias=False).to(DEV).to(torch.bfloat16)
dm_beta = torch.nn.Parameter(torch.zeros(H, device=DEV))
dm_beta_w = torch.nn.Parameter(torch.zeros(H, 768, device=DEV))


def timeit(fn, n=20):
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / n * 1000


def mk_bf16():
    d_k = dm_k(x).view(B, S, H, D).to(torch.bfloat16)
    d_k = d_k / (d_k.norm(dim=-1, keepdim=True) + 1e-6)
    d_q = dm_q(x).view(B, S, H, D).to(torch.bfloat16)
    d_v = dm_v(x).view(B, S, H, D).to(torch.bfloat16)
    db = torch.sigmoid(torch.einsum('bsd,hd->bsh', x.float(), dm_beta_w.float())).to(torch.bfloat16)
    return d_k, d_q, d_v, db


def mk_fp32():
    d_k = dm_k(x).view(B, S, H, D).float()
    d_k = d_k / (d_k.norm(dim=-1, keepdim=True) + 1e-6)
    d_q = dm_q(x).view(B, S, H, D).float()
    d_v = dm_v(x).view(B, S, H, D).float()
    db = torch.sigmoid(torch.einsum('bsd,hd->bsh', x.float(), dm_beta_w.float())).float()
    return d_k, d_q, d_v, db


for r in range(3):
    dk, dq, dv, db = mk_bf16()
    dk.requires_grad_(True)
    torch.cuda.synchronize()
    t0 = time.time()
    o, _ = chunk_delta_rule(q=dq, k=dk, v=dv, beta=db, initial_state=None,
                            output_final_state=False)
    o.sum().backward()
    torch.cuda.synchronize()
    print('delta bf16 fwd+bwd run%d: %.0f ms' % (r, (time.time() - t0) * 1000))

for r in range(3):
    fk, fq, fv, fb = mk_fp32()
    fk.requires_grad_(True)
    torch.cuda.synchronize()
    t0 = time.time()
    o, _ = chunk_gated_delta_rule(q=fq, k=fk, v=fv,
                                  g=torch.zeros(B, S, H, device=DEV), beta=fb,
                                  initial_state=None, output_final_state=False,
                                  use_qk_l2norm_in_kernel=False)
    o.sum().backward()
    torch.cuda.synchronize()
    print('gated fp32 fwd+bwd run%d: %.0f ms' % (r, (time.time() - t0) * 1000))
