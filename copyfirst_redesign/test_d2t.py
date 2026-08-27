import sys, torch
sys.path.insert(0, r"F:\OpenASH2605\copyfirst_redesign")
import importlib
import deepconv_triton as M
importlib.reload(M)
F = M._ConvD2Fn


def shift0(t, v):
    if v == 0:
        return t
    if v > 0:
        return torch.nn.functional.pad(t, (0, v))[v:v + t.numel()]
    return torch.nn.functional.pad(t, (-v, 0))[:t.numel()]


def ref(x, win, wo, k1, k2, bias1):
    """torch 参考: T1 = relu(Σ_{u,v} xp[h+u]·win[w+v]·K1[c,u,v] + b1); out = Σ_{c,u,v,w'} wo[w']·T1[c,h+u,w'+v]·K2[c,u,v]"""
    c = k1.shape[0]
    k = k1.shape[2]
    p = k // 2
    w = win.shape[0]
    b, s, h = x.shape
    xp = torch.nn.functional.pad(x, (p, p), mode='replicate')
    T1 = torch.zeros(b, s, c, h, w, device=x.device, dtype=x.dtype)
    for i in range(k):
        for j in range(k):
            v = j - p
            T1 += k1[:, 0, i, j].view(1, 1, c, 1, 1) * (
                xp[:, :, None, i:i + h, None].float() *
                shift0(win, v)[None, None, None, None, :].float())
    T1 = torch.relu(T1 + bias1.view(1, 1, c, 1, 1))
    T1wp = torch.nn.functional.pad(T1, (p, p, p, p, 0, 0, 0, 0, 0, 0))
    out = torch.zeros(b, s, h, device=x.device, dtype=x.dtype)
    for i in range(k):
        for j in range(k):
            out += torch.einsum('bschw,cw->bsh', T1wp[:, :, :, i:i + h, j:j + w],
                                k2[0, :, i, j][:, None] * wo[None, :])
    return out


torch.manual_seed(0)
for b, s, h, c, dt in [(2, 64, 96, 2, torch.float32), (2, 64, 96, 2, torch.bfloat16),
                       (8, 128, 192, 2, torch.float32), (8, 128, 192, 4, torch.float32)]:
    x = (torch.randn(b, s, h, device='cuda') * 0.3).to(dt).requires_grad_(True)
    win = (torch.randn(64, device='cuda') * 0.2).requires_grad_(True)
    wo = (torch.randn(64, device='cuda') * 0.2).requires_grad_(True)
    k1 = (torch.randn(c, 1, 3, 3, device='cuda') * 0.2).requires_grad_(True)
    k2 = (torch.randn(1, c, 3, 3, device='cuda') * 0.2).requires_grad_(True)
    bias1 = (torch.randn(c, device='cuda') * 0.1).requires_grad_(True)
    y = F.apply(x, win, wo, k1, k2, bias1)
    xr = x.detach().float()
    yr = ref(xr, win.detach(), wo.detach(), k1.detach(), k2.detach(), bias1.detach())
    ferr = (y.float() - yr).abs().max().item()
    gs = torch.autograd.grad(y.sum(), [x, win, wo, k1, k2, bias1])
    x2 = xr.clone().requires_grad_(True)
    win2 = win.detach().clone().requires_grad_(True)
    wo2 = wo.detach().clone().requires_grad_(True)
    k12 = k1.detach().clone().requires_grad_(True)
    k22 = k2.detach().clone().requires_grad_(True)
    b12 = bias1.detach().clone().requires_grad_(True)
    ref(x2, win2, wo2, k12, k22, b12).sum().backward()
    errs = [(gs[0].float() - x2.grad).abs().max().item(), (gs[1] - win2.grad).abs().max().item(),
            (gs[2] - wo2.grad).abs().max().item(), (gs[3] - k12.grad).abs().max().item(),
            (gs[4] - k22.grad).abs().max().item(), (gs[5] - b12.grad).abs().max().item()]
    print('b=%d s=%d h=%d c=%d %s: fwd=%.2e  gx=%.2e gwin=%.2e gwo=%.2e gk1=%.2e gk2=%.2e gb=%.2e' %
          (b, s, h, c, dt, ferr, *errs))
