import sys, random, torch
import torch.nn.functional as F
sys.path.insert(0, r"F:\夸克\rubikgla")
import deqlm_v33_rotread as M

torch.manual_seed(0)
random.seed(0)
seqs = torch.load(M.PT_CACHE, map_location="cpu", weights_only=True)
train = seqs[: int(0.95 * len(seqs))]
m = M.DeltaLoopLM().cuda()
print("params: %.3fM (v31=23.11M, v24=23.19M)" % (sum(p.numel() for p in m.parameters()) / 1e6))

# G 构建 sanity
h = torch.randn(2, 16, M.D, device="cuda")
Gm = m.pre.attn._build_grot(h)
orth = (Gm.transpose(-1, -2) @ Gm - torch.eye(M.DH, device="cuda")).abs().max().item()
ident = (Gm - torch.eye(M.DH, device="cuda")).abs().max().item()
print("G: 正交偏差=%.2e  初始近恒等 max|G−I|=%.4f" % (orth, ident))

rng = random.Random(42)
x, y = M.get_batch(train, bs=16, device="cuda", rng=rng)
m.train()
with torch.autocast("cuda", dtype=torch.bfloat16):
    lo = m(x)
l = F.cross_entropy(lo.float().reshape(-1, M.V), y.reshape(-1), ignore_index=0)
print("step0 loss = %.4f (v31 参照 10.0853)" % l.item())
l.backward()
gn = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
print("grad=%.2f finite=%s" % (gn.item(), torch.isfinite(gn).item()))
wr = m.pre.attn.w_rot
print("w_rot grad mean=%.3e (非零=反传到)" % wr.weight.grad.abs().mean().item())

import time
m.train()
t0 = time.time()
for i in range(8):
    x, y = M.get_batch(train, bs=16, device="cuda", rng=rng)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lo = m(x)
    l = F.cross_entropy(lo.float().reshape(-1, M.V), y.reshape(-1), ignore_index=0)
    l.backward()
    m.zero_grad(set_to_none=True)
torch.cuda.synchronize()
print("纯 fwd+bwd: %.0f ms/微批" % ((time.time() - t0) / 8 * 1000))
print("peak mem: %.1f GB" % (torch.cuda.max_memory_allocated() / 1e9))
print("SMOKE OK")
