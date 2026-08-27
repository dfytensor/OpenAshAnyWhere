import sys, torch
sys.path.insert(0, r"F:\OpenASH2605\copyfirst_redesign")
from conv_linear_triton_train import _ConvLinearFn as F

torch.manual_seed(3)
b, s, h = 1, 4, 32
x = (torch.randn(b, s, h, device='cuda') * 0.3).requires_grad_(True)
Kw = (torch.randn(3, 8, device='cuda') * 0.2).requires_grad_(True)
w_out = (torch.randn(8, device='cuda') * 0.2).requires_grad_(True)
bias = (torch.randn(8, device='cuda') * 0.1).requires_grad_(True)

y2 = F.apply(x, Kw, w_out, bias, True)
gx, gK, gw, gb = torch.autograd.grad(y2.sum(), [x, Kw, w_out, bias])

x2 = x.detach().clone().requires_grad_(True); K2 = Kw.detach().clone().requires_grad_(True)
w2 = w_out.detach().clone().requires_grad_(True); b2 = bias.detach().clone().requires_grad_(True)
xp = torch.nn.functional.pad(x2, (1, 1), mode='replicate')
A = xp.unfold(2, 3, 1)
T = torch.einsum('bshk,kw->bshw', A, K2) + b2
R = torch.relu(T)
ref = torch.einsum('bshw,w->bsh', R, w2)
ref.sum().backward()

print('gx kernel row0 [0:12]:', gx[0, 0, :12].tolist())
print('gx ref    row0 [0:12]:', x2.grad[0, 0, :12].tolist())
diff = (gx - x2.grad).abs()
print('gx max err:', diff.max().item(), ' at', torch.nonzero(diff == diff.max()).tolist())
print('gK kernel:', gK.flatten().tolist())
print('gK ref   :', K2.grad.flatten().tolist())
print('gw kernel:', gw.tolist())
print('gw ref   :', w2.grad.tolist())
print('gb kernel:', gb.tolist())
print('gb ref   :', b2.grad.tolist())
