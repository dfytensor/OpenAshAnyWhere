import sys, torch
sys.path.insert(0, r"F:\夸克\rubikgla")
import delta_triton as DT

DEV = "cuda"
torch.manual_seed(0)
B, L, KV, SLOTS, KVG, DH = 1, 1, 1, 1, 1, 32
G = KV * SLOTS
H = KV * KVG
dt = torch.bfloat16


def ref(k, v, rate, eta, q):
    k = k.detach().float().requires_grad_(True)
    v = v.detach().float().requires_grad_(True)
    rate = rate.detach().float().requires_grad_(True)
    eta = eta.detach().float().requires_grad_(True)
    q = q.detach().float().requires_grad_(True)
    S = torch.zeros(B, G, DH, DH, device=k.device)
    out = torch.zeros(B, L, G, KVG, DH, device=k.device)
    for t in range(L):
        kt, vt = k[:, t], v[:, t]
        lam = torch.exp(-rate[:, t])
        rr = torch.einsum("bgi,bgij->bgj", kt, S)
        S = lam[..., None] * (S - eta[:, t][..., None] * kt[..., :, None] * rr[..., None, :]) \
            + kt[..., :, None] * vt[..., None, :]
        out[:, t, 0, 0] = torch.einsum("bi,bij->bj", q[:, t, 0], S[:, 0])
    return out, (k, v, rate, eta, q)


k = (torch.randn(B, L, G, DH, device=DEV) * 0.3).to(dt)
v = (torch.randn(B, L, G, DH, device=DEV) * 0.3).to(dt)
r = (torch.rand(B, L, G, DH, device=DEV) * 0.3 + 1e-3).to(dt)
e = (torch.rand(B, L, G, DH, device=DEV) * 0.3 + 0.05).to(dt)
q = (torch.randn(B, L, H, DH, device=DEV) * 0.5).to(dt)
dy = torch.randn(B, L, G, KVG, DH, device=DEV)

y_ref, refs = ref(k, v, r, e, q)
y_ref.backward(dy)
for name, b in [("dk", refs[0].grad), ("dv", refs[1].grad), ("dr", refs[2].grad),
                ("de", refs[3].grad), ("dq", refs[4].grad)]:
    a = None
    for vs in (1, 2):
        DT.VSPLIT = vs
        for x in (k, v, r, e, q):
            x.grad = None
        kk = k.clone().requires_grad_(True); vv = v.clone().requires_grad_(True)
        rr_ = r.clone().requires_grad_(True); ee = e.clone().requires_grad_(True)
        qq = q.clone().requires_grad_(True)
        y = DT.delta_scan_triton(kk, vv, rr_, ee, qq, KV, SLOTS, KVG)
        y.backward(dy)
        g = {"dk": kk.grad, "dv": vv.grad, "dr": rr_.grad, "de": ee.grad, "dq": qq.grad}[name]
        if vs == 1:
            g1 = g.float()
        else:
            g2 = g.float()
    scale = b.abs().mean().item() + 1e-9
    print("%-3s ref=%.4e | vs1 rel=%.2e  vs2 rel=%.2e" % (
        name, scale,
        (g1 - b).abs().mean().item() / scale,
        (g2 - b).abs().mean().item() / scale))
print("ref dk[:6]:", b[:6].tolist() if False else "")
