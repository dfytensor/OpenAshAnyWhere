import sys, random, torch
import torch.nn.functional as F
sys.path.insert(0, r"F:\夸克\rubikgla")
import deqlm_v33_rotread as M

torch.manual_seed(0)
m = M.DeltaLoopLM().cuda()
seqs = torch.load(M.PT_CACHE, map_location="cpu", weights_only=True)
train = seqs[: int(0.95 * len(seqs))]
rng = random.Random(42)
x, y = M.get_batch(train, bs=16, device="cuda", rng=rng)
m.train()
with torch.autocast("cuda", dtype=torch.bfloat16):
    lo = m(x)
l = F.cross_entropy(lo.float().reshape(-1, M.V), y.reshape(-1), ignore_index=0)
l.backward()

mods = [("pre.attn.wq", m.pre.attn.wq), ("pre.attn.w_rot", m.pre.attn.w_rot),
        ("pre.attn.wv", m.pre.attn.wv), ("loop_a.attn.w_rot", m.loop_a.attn.attn.w_rot)]
for nm, mod in mods:
    g = mod.weight.grad
    print("%-20s grad %s" % (nm, "None" if g is None else "%.3e" % g.abs().mean().item()))

# 单独测: _build_grot 的梯度
h = torch.randn(2, 32, M.D, device="cuda", requires_grad=True)
att = m.pre.attn
Gm = att._build_grot(h)
loss = Gm.float().pow(2).sum()
loss.backward()
print("build_grot: h.grad=%.3e  w_rot.weight.grad=%.3e" % (
    h.grad.abs().mean().item(),
    att.w_rot.weight.grad.abs().mean().item() if att.w_rot.weight.grad is not None else -1))
