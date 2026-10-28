#!/usr/bin/env python3
"""决定性探针: capture_states@4096(boundary) 是否污染权重/状态, 使后续前向破损."""
import sys, os, random
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import deqlm_v35_relay as M
from deqlm_v5_30m import PT_CACHE

DEV = "cuda"
torch.manual_seed(0)
HERE = os.path.dirname(os.path.abspath(__file__))

net = M.DeltaLoopLM()
net.load_state_dict(torch.load(os.path.join(HERE, "deqlmv31r_pt_full.pth"),
                               map_location="cpu", weights_only=True))
net = net.to(DEV)
net.train()

seqs = torch.load(PT_CACHE, map_location="cpu", weights_only=True)
train = seqs[:int(0.95 * len(seqs))]
rng = random.Random(42)

def block(bs, ln):
    out = []
    for _ in range(bs):
        buf, tot = [], 0
        while tot < ln:
            s = train[rng.randrange(len(train))]
            buf.append(s)
            tot += len(s)
        out.append(torch.cat(buf)[:ln])
    return torch.stack(out).to(DEV)

def tloss(x):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lo = net(x)
    return torch.nn.functional.cross_entropy(lo.float()[:, :-1].reshape(-1, lo.shape[-1]),
                                             x[:, 1:].reshape(-1), ignore_index=0).item()

x4096 = block(2, 4096)
w_before = net.pre.attn.wq.weight.detach().clone()

print("0) 训练模式前向 @4096 (干净)      loss %.4f" % tloss(x4096), flush=True)

with torch.no_grad():
    net.capture_states(x4096)                      # 无边界 (老路径)
print("1) capture@4096 无boundary 后     loss %.4f" % tloss(x4096), flush=True)

with torch.no_grad():
    net.capture_states(x4096, boundary_idx=[127, 255, 511, 1023, 2047, 4095])
print("2) capture@4096 有boundary 后     loss %.4f" % tloss(x4096), flush=True)

w_after = net.pre.attn.wq.weight.detach()
print("权重是否变化:", not torch.equal(w_before, w_after), flush=True)
print("net.training =", net.training, flush=True)
