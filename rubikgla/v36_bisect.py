#!/usr/bin/env python3
"""二分定位: 训练模式前向在哪个 (L, checkpoint) 组合下数值破坏."""
import sys, os, random
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import deqlm_v35_relay as M
from deqlm_v5_30m import PT_CACHE

DEV = "cuda"
torch.manual_seed(0)

net = M.DeltaLoopLM()
net.load_state_dict(torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            "deqlmv31r_pt_full.pth"), map_location="cpu", weights_only=True))
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

def train_loss(L, use_ckpt):
    for blk in (net.pre, net.loop_a.attn, net.bridge, net.loop_c.attn, net.post):
        if use_ckpt:
            blk._attn_impl = None          # 恢复 compile 路径 (惰性)
            def fwd(x, mult, s0, _a=blk.attn):
                return _a._apply_raw(x, mult, s0)
            blk._attn_impl = fwd           # 无 compile 的原始函数 (保留 checkpoint)
            blk._ffn_impl = blk.ffn.forward
        else:
            blk._attn_impl = blk.attn._apply_raw
            blk._ffn_impl = blk.ffn.forward
    x = block(2, L)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lo = net(x)
    return torch.nn.functional.cross_entropy(lo.float()[:, :-1].reshape(-1, lo.shape[-1]),
                                             x[:, 1:].reshape(-1), ignore_index=0).item()

for L in (256, 1024, 4096):
    for ck in (True, False):
        loss = train_loss(L, ck)
        print("L=%4d checkpoint=%-5s  train_loss %.4f" % (L, ck, loss), flush=True)
