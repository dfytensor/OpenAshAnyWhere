import sys, time, importlib
sys.path.insert(0, r"F:\OpenASH2605\copyfirst_redesign")
sys.path.insert(0, r"F:\OpenASH2605")
import torch
import torch.nn.functional as F
import conv_linear_triton_train as M

torch.manual_seed(0)
Kw = (torch.randn(9, 64, device='cuda') * 0.2).requires_grad_(True)
w_out = (torch.randn(64, device='cuda') * 0.2).requires_grad_(True)
bias = (torch.randn(64, device='cuda') * 0.1).requires_grad_(True)
B, S = 32, 256
x = (torch.randn(B, S, 640, device='cuda') * 0.3).requires_grad_(True)


def bench(tag, n=5):
    y = M._ConvLinearFn.apply(x, Kw, w_out, bias, True)
    loss = y.sum()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        loss.backward(retain_graph=True)
    torch.cuda.synchronize()
    print('%s: %.2f ms/bwd' % (tag, (time.perf_counter() - t0) / n * 1000), flush=True)


bench('v1 ieee+loop+dxatomic')
