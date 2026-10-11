import sys, random, torch
import torch.nn.functional as F
sys.path.insert(0, r"F:\夸克\rubikgla")
import deqlm_v32_slotgate as M

torch.manual_seed(0)
random.seed(0)
seqs = torch.load(M.PT_CACHE, map_location="cpu", weights_only=True)
train = seqs[: int(0.95 * len(seqs))]
m = M.DeltaLoopLM().cuda()
rng = random.Random(42)
x, y = M.get_batch(train, bs=16, device="cuda", rng=rng)
m.train()
with torch.autocast("cuda", dtype=torch.bfloat16):
    lo = m(x)
l = F.cross_entropy(lo.float().reshape(-1, M.V), y.reshape(-1), ignore_index=0)
v = l.item()
print("step0 loss = %.4f" % v)
print("v31 参照     = 10.0853")
print("逐位一致:", abs(v - 10.0853) < 1e-9)
