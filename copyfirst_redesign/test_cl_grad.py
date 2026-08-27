import sys, torch
sys.path.insert(0, r"F:\OpenASH2605\copyfirst_redesign")
import importlib
import conv_linear_triton_train as M
importlib.reload(M)
F = M._ConvLinearFn

torch.manual_seed(0)


def ref(x, Kw, w, bias, act):
    xp = torch.nn.functional.pad(x, (1, 1), mode='replicate')
    A = xp.unfold(2, 3, 1)
    T = torch.einsum('bshk,kw->bshw', A, Kw) + bias
    T = torch.relu(T) if act else T
    return torch.einsum('bshw,w->bsh', T, w)


def check(b, s, h, act, in_dtype):
    x = (torch.randn(b, s, h, device='cuda') * 0.3).to(in_dtype).requires_grad_(True)
    Kw = (torch.randn(3, 64, device='cuda') * 0.2).requires_grad_(True)
    w_out = (torch.randn(64, device='cuda') * 0.2).requires_grad_(True)
    bias = (torch.randn(64, device='cuda') * 0.1).requires_grad_(True)
    y2 = F.apply(x, Kw, w_out, bias, act)
    xr = x.detach().float()
    err = (y2.float() - ref(xr, Kw.detach(), w_out.detach(), bias.detach(), act)).abs().max().item()
    gx, gK, gw, gb = torch.autograd.grad(y2.sum(), [x, Kw, w_out, bias])
    x2 = xr.clone().requires_grad_(True); K2 = Kw.detach().clone().requires_grad_(True)
    w2 = w_out.detach().clone().requires_grad_(True); b2 = bias.detach().clone().requires_grad_(True)
    ref(x2, K2, w2, b2, act).sum().backward()
    print('b=%d s=%d h=%d act=%s x=%s  fwd_err=%.2e  gx_err=%.2e gK_err=%.2e gw_err=%.2e gb_err=%.2e' % (
        b, s, h, act, in_dtype, err,
        (gx.float() - x2.grad).abs().max().item(), (gK - K2.grad).abs().max().item(),
        (gw - w2.grad).abs().max().item(), (gb - b2.grad).abs().max().item()))


for dt in [torch.float32, torch.bfloat16]:
    check(2, 64, 96, True, dt)
    check(8, 128, 192, False, dt)
    check(32, 256, 640, True, dt)
