"""delta_chunk.py 正确性验证：前向/初态/终态/反向 全对照 FLA chunk_gated_delta_rule。"""
import sys
sys.path.insert(0, r'F:\OpenASH2605')
import torch
from delta_chunk import chunked_delta_rule
from fla.ops.gated_delta_rule import chunk_gated_delta_rule

torch.manual_seed(7)
DEV = 'cuda'
FAIL = 0

for (B, T, H, K, V) in [(2, 64, 2, 16, 16), (2, 128, 2, 32, 32), (3, 96, 4, 48, 48), (2, 384, 8, 48, 48)]:
    q = torch.randn(B, T, H, K, device=DEV, dtype=torch.float32) * 0.3
    kk = torch.randn(B, T, H, K, device=DEV, dtype=torch.float32)
    k = kk / (kk.norm(dim=-1, keepdim=True) + 1e-6)
    v = k.clone()
    g = -torch.rand(B, T, H, device=DEV, dtype=torch.float32) * 0.2   # log 域衰减 ≤ 0
    beta = torch.sigmoid(torch.randn(B, T, H, device=DEV, dtype=torch.float32))

    q1 = q.clone().requires_grad_(True); k1 = k.clone().requires_grad_(True)
    v1 = v.clone().requires_grad_(True); g1 = g.clone().requires_grad_(True)
    b1 = beta.clone().requires_grad_(True)
    o_ref, s_ref = chunk_gated_delta_rule(q=q1, k=k1, v=v1, g=g1, beta=b1,
                                          output_final_state=True, use_qk_l2norm_in_kernel=False)
    q2 = q.clone().requires_grad_(True); k2 = k.clone().requires_grad_(True)
    v2 = v.clone().requires_grad_(True); g2 = g.clone().requires_grad_(True)
    b2 = beta.clone().requires_grad_(True)
    o_ours, s_ours = chunked_delta_rule(q2, k2, v2, g2, b2, output_final_state=True)

    eo = (o_ours - o_ref).abs().max().item()
    es = (s_ours - s_ref).abs().max().item()
    tag = 'PASS' if (eo < 1e-3 and es < 1e-3) else 'FAIL'
    FAIL += (tag == 'FAIL')
    print(f'fwd B{B} T{T} H{H} K{K}: o_err={eo:.2e} s_err={es:.2e} {tag}')

    if tag == 'PASS':
        grad = torch.randn_like(o_ref)
        (o_ref * grad).sum().backward()
        (o_ours * grad).sum().backward()
        errs = []
        for nm, a, b in [('dq', q1.grad, q2.grad), ('dk', k1.grad, k2.grad),
                         ('dv', v1.grad, v2.grad), ('dg', g1.grad, g2.grad), ('dbeta', b1.grad, b2.grad)]:
            e = (a - b).abs().max().item() if a is not None and b is not None else float('nan')
            errs.append(f'{nm}={e:.1e}')
        ok = all((x.split('=')[1] != 'nan') and float(x.split('=')[1]) < 5e-3 for x in errs)
        FAIL += (not ok)
        print(f'  bwd: {" ".join(errs)} {"PASS" if ok else "FAIL"}')

# 初态
B, T, H, K, V = 2, 128, 2, 32, 32
q = torch.randn(B, T, H, K, device=DEV) * 0.3
kk = torch.randn(B, T, H, K, device=DEV)
k = kk / (kk.norm(dim=-1, keepdim=True) + 1e-6)
v = k.clone()
g = -torch.rand(B, T, H, device=DEV) * 0.2
beta = torch.sigmoid(torch.randn(B, T, H, device=DEV))
S0 = torch.randn(B, H, K, V, device=DEV) * 0.1
o_ref, s_ref = chunk_gated_delta_rule(q=q, k=k, v=v, g=g, beta=beta, initial_state=S0.clone(),
                                      output_final_state=True, use_qk_l2norm_in_kernel=False)
o_ours, s_ours = chunked_delta_rule(q, k, v, g, beta, initial_state=S0, output_final_state=True)
eo = (o_ours - o_ref).abs().max().item()
es = (s_ours - s_ref).abs().max().item()
tag = 'PASS' if (eo < 1e-3 and es < 1e-3) else 'FAIL'
FAIL += (tag == 'FAIL')
print(f'initial_state: o_err={eo:.2e} s_err={es:.2e} {tag}')

print(f'\n{"ALL PASS" if FAIL == 0 else f"{FAIL} FAILURES"}')
sys.exit(1 if FAIL else 0)
