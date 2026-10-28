#!/usr/bin/env python3
"""16k 异常定位: 同一 4096/16384 块上, 不同窗口的零state NLL + 数据统计."""
import sys, os, random
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import deqlm_v35_relay as M
from deqlm_v5_30m import PT_CACHE

DEV = "cuda"
torch.manual_seed(0)
rng = random.Random(7)
seqs = torch.load(PT_CACHE, map_location="cpu", weights_only=True)
VAL = [s for s in seqs[int(0.95 * len(seqs)):][:2000] if len(s) >= 256]
print("VAL: %d 条, 长度 min/max = %d/%d" % (len(VAL), min(len(s) for s in VAL), max(len(s) for s in VAL)))

net = M.DeltaLoopLM()
net.load_state_dict(torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            "deqlmv31r_pt_full.pth"), map_location="cpu", weights_only=True),
                   strict=False)
net = net.to(DEV).eval()

def block(n, ln):
    out = []
    for _ in range(n):
        buf, tot = [], 0
        while tot < ln:
            s = VAL[rng.randrange(len(VAL))]
            buf.append(s)
            tot += len(s)
        out.append(torch.cat(buf)[:ln])
    return torch.stack(out).to(DEV)

def win_nll(ids, start, ln):
    x = ids[:, start:start + ln]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lo = net(x)
    return F.cross_entropy(lo.float()[:, :-1].reshape(-1, lo.shape[-1]),
                           x[:, 1:].reshape(-1), ignore_index=0).item()

for L in (4096, 16384):
    ids = block(2, L)
    pad_suffix = (ids[:, L - 2048:] == 0).float().mean().item()
    print("L=%5d | 后缀pad占比 %.3f | 首窗(0) %.4f | 中窗(%d) %.4f | 尾窗(%d) %.4f"
          % (L, pad_suffix,
             win_nll(ids, 0, 2048),
             L // 2 - 1024, win_nll(ids, L // 2 - 1024, 2048),
             L - 2048, win_nll(ids, L - 2048, 2048)), flush=True)
print("PROBE DONE", flush=True)
