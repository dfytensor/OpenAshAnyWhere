#!/usr/bin/env python3
"""门/选择性行为分析: 平均 g、sel 分布 — 稀疏化是否发生 (RWKV-6 选择性假设检验).
用法: python v37_behavior.py --ckpt v36ms_e4.pth
"""
import sys, os, argparse, random
os.environ.setdefault("DELTA_STAGES", "1")
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import deqlm_v35_relay as M
from deqlm_v5_30m import PT_CACHE

DEV = "cuda"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "v36ms_e4.pth"))
    a = ap.parse_args()
    torch.manual_seed(0)
    seqs = torch.load(PT_CACHE, map_location="cpu", weights_only=True)
    val = [s for s in seqs[int(0.95 * len(seqs)):][:2000] if len(s) >= 256]
    rng_b = random.Random(5)

    def blk4096():
        buf, tot = [], 0
        while tot < 4096:
            s = val[rng_b.randrange(len(val))]
            buf.append(s)
            tot += len(s)
        return torch.cat(buf)[:4096]

    ids = torch.stack([blk4096() for _ in range(4)]).to(DEV)

    net = M.DeltaLoopLM()
    sd = torch.load(a.ckpt, map_location="cpu", weights_only=True)
    missing, unexpected = net.load_state_dict(sd, strict=False)
    print("加载: missing %d | unexpected %d" % (len(missing), len(unexpected)), flush=True)
    for blk in (net.pre, net.loop_a.attn, net.bridge, net.loop_c.attn, net.post):
        blk.attn.use_gate = True
        blk.attn.use_sel = True
    net = net.to(DEV).eval()

    gate_caps, sel_caps = [], []
    attns = [("pre", net.pre.attn), ("loop_a", net.loop_a.attn.attn), ("bridge", net.bridge.attn),
             ("loop_c", net.loop_c.attn.attn), ("post", net.post.attn)]
    hooks = []
    for name, attn in attns:
        def mk(attn):
            def hook(mod, inp, out):
                with torch.no_grad():
                    hh = attn.ln(inp[0])               # w_gate 的输入 = MARMSNorm 后的 h
                    # F.linear 直呼权重: 不经模块调用, 避免重入本 hook
                    gate_caps.append(torch.sigmoid(F.linear(hh, attn.w_gate.weight, attn.w_gate.bias)).detach().flatten())
                    sel_caps.append(torch.sigmoid(attn.sel_B(attn.sel_A(hh))).detach().flatten())
            return hook
        hooks.append(attn.w_gate.register_forward_hook(mk(attn)))

    with torch.no_grad():
        net(ids)

    names = ["pre", "loop_a", "bridge", "loop_c", "post"]
    print("=" * 64)
    print("门 g = sigmoid(w_gate):  稀疏化判定 — 平均 g ↓ / g<0.5 占比 ↑ = 选择性写入发生")
    print("sel = sigmoid(sel_B(sel_A)): sel>0.7 = 保持通道 (rate↓), sel<0.3 = 快遗忘通道")
    print("=" * 64)
    for name, g in zip(names, gate_caps):
        gm, gstd = g.mean().item(), g.std().item()
        f05 = (g < 0.5).float().mean().item()
        f02 = (g < 0.2).float().mean().item()
        print("  [%-7s] g 平均 %.3f ± %.3f | g<0.5: %.1f%% | g<0.2: %.1f%%"
              % (name, gm, gstd, f05 * 100, f02 * 100))
    for name, s in zip(names, sel_caps):
        sm = s.mean().item()
        f07 = (s > 0.7).float().mean().item()
        f03 = (s < 0.3).float().mean().item()
        print("  [%-7s] sel 平均 %.3f | sel>0.7 (保保持): %.1f%% | sel<0.3 (快遗忘): %.1f%%"
              % (name, sm, f07 * 100, f03 * 100))
    # sel_scale 幅度: 选择性的实际作用力
    for name, attn in attns:
        sc = attn.sel_scale.detach()
        print("  [%-7s] sel_scale |mean|=%.3f max=%.3f (0=无选择性)" % (name, sc.abs().mean().item(), sc.abs().max().item()))
    print("BEHAVIOR DONE", flush=True)


if __name__ == "__main__":
    main()
