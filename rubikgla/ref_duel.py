import sys, torch
sys.path.insert(0, r"F:\夸克\rubikgla")
import importlib
tdt = importlib.import_module("test_delta_triton") if False else None

DEV = "cuda"
torch.manual_seed(0)
B, L, KV, SLOTS, KVG, DH = 2, 64, 5, 2, 2, 32
G, H = KV * SLOTS, KV * KVG
dt = torch.bfloat16

k = (torch.randn(B, L, G, DH, device=DEV) * 0.3).to(dt)
v = (torch.randn(B, L, G, DH, device=DEV) * 0.3).to(dt)
rate = (torch.rand(B, L, G, DH, device=DEV) * 0.3 + 1e-3).to(dt)
eta = (torch.rand(B, L, G, DH, device=DEV) * 0.3 + 0.05).to(dt)
q = (torch.randn(B, L, H, DH, device=DEV) * 0.5).to(dt)
_u = torch.randn(B, L, DH, device=DEV); _u = _u / _u.norm(dim=-1, keepdim=True)
_w = torch.randn(B, L, DH, device=DEV); _w = _w / _w.norm(dim=-1, keepdim=True)
_th = torch.rand(B, L, 1, device=DEV) * 0.3 + 0.05
_Mm = _u.unsqueeze(-1) * _w.unsqueeze(-2) - _w.unsqueeze(-1) * _u.unsqueeze(-2)
_GM = _th.unsqueeze(-1) * _Mm
_E = torch.eye(DH, device=DEV).view(1, 1, DH, DH); _P = _GM.clone()
for i in range(2, 7):
    _E = _E + _P
    _P = _P @ _GM / i


def ref_A(k, v, rate, eta, q, grot):          # 套件版 (参数式)
    k = k.detach().float(); v = v.detach().float(); rate = rate.detach().float()
    eta = eta.detach().float(); q = q.detach().float()
    k.requires_grad_(True); v.requires_grad_(True); rate.requires_grad_(True)
    eta.requires_grad_(True); q.requires_grad_(True)
    grot = grot.detach().float().requires_grad_(True)
    S = torch.zeros(B, G, DH, DH, device=DEV, dtype=torch.float32)
    out = torch.zeros(B, L, G, KVG, DH, device=DEV, dtype=torch.float32)
    for t in range(L):
        kt, vt = k[:, t], v[:, t]
        lam = torch.exp(-rate[:, t])
        R = torch.einsum("bij,bgjk->bgik", grot[:, t], S)
        rr = torch.einsum("bgi,bgij->bgj", kt, R)
        S = lam[..., None] * (R - eta[:, t][..., None] * kt[..., :, None] * rr[..., None, :]) \
            + kt[..., :, None] * vt[..., None, :]
        for g in range(G):
            kvg_i = g // SLOTS
            for m in range(KVG):
                h = kvg_i * KVG + m
                out[:, t, g, m] = torch.einsum("bi,bij->bj", q[:, t, h], S[:, g])
    return out, (k, v, rate, eta, q, grot)


def ref_B(k, v, rate, eta, q, grot):          # minimal 版
    kk = k.detach().float().requires_grad_(True)
    vv = v.detach().float().requires_grad_(True)
    rr_ = rate.detach().float().requires_grad_(True)
    ee = eta.detach().float().requires_grad_(True)
    qq = q.detach().float().requires_grad_(True)
    gg = grot.detach().float().requires_grad_(True)
    S = torch.zeros(B, G, DH, DH, device=DEV)
    out = torch.zeros(B, L, G, KVG, DH, device=DEV)
    for t in range(L):
        R = torch.einsum("bij,bgjk->bgik", gg[:, t], S)
        rrx = torch.einsum("bgi,bgij->bgj", kk[:, t], R)
        S = torch.exp(-rr_[:, t])[..., None] * (R - ee[:, t][..., None] * kk[:, t][..., :, None] * rrx[..., None, :]) \
            + kk[:, t][..., :, None] * vv[:, t][..., None, :]
        for g in range(G):
            kvg_i = g // SLOTS
            for m in range(KVG):
                out[:, t, g, m] = torch.einsum("bi,bij->bj", qq[:, t, kvg_i * KVG + m], S[:, g])
    return out, (kk, vv, rr_, ee, qq, gg)


yA, refsA = ref_A(k, v, rate, eta, q, _E)
yB, refsB = ref_B(k, v, rate, eta, q, _E)
print("yA vs yB max diff:", (yA - yB).abs().max().item())
dy = torch.randn_like(yA)
yA.backward(dy)
yB.backward(dy)
print("dk A=%.4e  B=%.4e" % (refsA[0].grad.abs().mean().item(), refsB[0].grad.abs().mean().item()))
