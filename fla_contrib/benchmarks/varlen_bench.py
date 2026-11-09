import sys, time
sys.path.insert(0, 'fla_contrib')
import torch
from ops.decay_max_scan.chunk import chunk_decay_max_scan

def bench(fn, iters=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters

B, T, D, H = 8, 2048, 96, 8
x = torch.randn(B, T, D, H, device='cuda')
v = torch.randn(D, device='cuda') * 0.05
b = torch.randn(D, device='cuda') * 0.05
cu = torch.tensor([0, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096], device='cuda')[:5].cpu()
xv = x.view(4 * B if False else B, T, D, H)
# varlen: treat T=2048 as 4 packed sequences of 512 within one batch row
cu = torch.tensor([0, 512, 1024, 1536, 2048], device='cuda')
def dense():
    chunk_decay_max_scan(x=x, decay_v=v, decay_b=b)
def varlen():
    chunk_decay_max_scan(x=x[:1, :2048], decay_v=v, decay_b=b, cu_seqlens=cu)
td = bench(dense)
tv = bench(varlen)
print(f'dense  B8 T2048: {td*1e3:7.2f} ms | varlen packed 4x512 T2048: {tv*1e3:7.2f} ms (torch path)')
