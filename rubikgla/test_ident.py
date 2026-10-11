import sys, torch
sys.path.insert(0, r"F:\夸克\rubikgla")
from delta_triton import delta_scan_triton

DEV = "cuda"
torch.manual_seed(0)
B, L, KV, SLOTS, KVG, DH = 2, 64, 5, 2, 2, 32
G, H = KV * SLOTS, KV * KVG
dt = torch.bfloat16

k = (torch.randn(B, L, G, DH, device=DEV) * 0.3).to(dt).requires_grad_(True)
v = (torch.randn(B, L, G, DH, device=DEV) * 0.3).to(dt).requires_grad_(True)
rate = (torch.rand(B, L, G, DH, device=DEV) * 0.3 + 1e-3).to(dt).requires_grad_(True)
eta = (torch.rand(B, L, G, DH, device=DEV) * 0.3 + 0.05).to(dt).requires_grad_(True)
q = (torch.randn(B, L, H, DH, device=DEV) * 0.5).to(dt).requires_grad_(True)


def ref_loop(k, v, rate, eta, q, Gmat=None):
    kk = k.detach().float().requires_grad_(True)
    vv = v.detach().float().requires_grad_(True)
    rr_ = rate.detach().float().requires_grad_(True)
    ee = eta.detach().float().requires_grad_(True)
    qq = q.detach().float().requires_grad_(True)
    gg = None
    if Gmat is not None:
        gg = Gmat.detach().float().requires_grad_(True)
    S = torch.zeros(B, G, DH, DH, device=DEV)
    out = torch.zeros(B, L, G, KVG, DH, device=DEV)
    for t in range(L):
        R = S if gg is None else torch.einsum("bij,bgjk->bgik", gg[:, t], S)
        rrx = torch.einsum("bgi,bgij->bgj", kk[:, t], R)
        S = torch.exp(-rr_[:, t])[..., None] * (R - ee[:, t][..., None] * kk[:, t][..., :, None] * rrx[..., None, :]) \
            + kk[:, t][..., :, None] * vv[:, t][..., None, :]
        for g in range(G):
            kvg_i = g // SLOTS
            for m in range(KVG):
                out[:, t, g, m] = torch.einsum("bi,bij->bj", qq[:, t, kvg_i * KVG + m], S[:, g])
    return out, (kk, vv, rr_, ee, qq, gg)


I = torch.eye(DH, device=DEV).view(1, 1, DH, DH).repeat(B, L, 1, 1)
gtr = I.clone().requires_grad_(True)
y_tri = delta_scan_triton(k, v, rate, eta, q, KV, SLOTS, KVG, gtr)
y_ref, refs = ref_loop(k, v, rate, eta, q, I)
dy = torch.randn_like(y_tri)
print("fwd rel:", ((y_tri.float() - y_ref).abs() / (y_ref.abs() + 1e-3)).mean().item())
y_tri.backward(dy)
y_ref.backward(dy.float())
for name, a, b in [("dk", k.grad, refs[0].grad), ("dv", v.grad, refs[1].grad), ("dq", q.grad, refs[4].grad), ("dG", gtr.grad, refs[5].grad)]:
    print("%-4s rel=%.2e" % (name, (a.float() - b).abs().mean().item() / (b.abs().mean().item() + 1e-9)))

# ===== 非恒等旋转 =====
_u = torch.randn(B, L, DH, device=DEV)
_w = torch.randn(B, L, DH, device=DEV)
_u = _u / _u.norm(dim=-1, keepdim=True)
_w = _w / _w.norm(dim=-1, keepdim=True)
_th = torch.rand(B, L, 1, device=DEV) * 0.3 + 0.05
_Mm = _u.unsqueeze(-1) * _w.unsqueeze(-2) - _w.unsqueeze(-1) * _u.unsqueeze(-2)
_GM = _th.unsqueeze(-1) * _Mm
_E = torch.eye(DH, device=DEV).view(1, 1, DH, DH)
_P = _GM.clone()
for i in range(2, 7):
    _E = _E + _P
    _P = _P @ _GM / i
print("E 最大绝对值:", _E.abs().max().item(), " 正交偏差:", (_E @ _E.transpose(-1, -2) - I[0:0+_E.shape[0]]).abs().max().item() if False else (_E.transpose(-1,-2) @ _E - torch.eye(DH,device=DEV)).abs().max().item())

gtr2 = _E.clone().requires_grad_(True)
k2 = k.detach().clone().requires_grad_(True); v2 = v.detach().clone().requires_grad_(True)
r2 = rate.detach().clone().requires_grad_(True); e2 = eta.detach().clone().requires_grad_(True); q2 = q.detach().clone().requires_grad_(True)
y_tri2 = delta_scan_triton(k2, v2, r2, e2, q2, KV, SLOTS, KVG, gtr2)
y_ref2, refs2 = ref_loop(k2, v2, r2, e2, q2, _E)
dy2 = torch.randn_like(y_tri2)
print("rot fwd rel:", ((y_tri2.float() - y_ref2).abs() / (y_ref2.abs() + 1e-3)).mean().item())
y_tri2.backward(dy2)
y_ref2.backward(dy2.float())
for name, a, b in [("dk", k2.grad, refs2[0].grad), ("dv", v2.grad, refs2[1].grad),
                   ("dr", r2.grad, refs2[2].grad), ("de", e2.grad, refs2[3].grad),
                   ("dq", q2.grad, refs2[4].grad), ("dG", gtr2.grad, refs2[5].grad)]:
    print("rot %-4s |a|=%.3e |ref|=%.3e  rel=%.2e" % (name, a.float().abs().mean().item(), b.abs().mean().item(),
          (a.float() - b).abs().mean().item() / (b.abs().mean().item() + 1e-9)))
