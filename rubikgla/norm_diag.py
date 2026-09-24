#!/usr/bin/env python3
"""范数爆炸源诊断: 逐层 hook 追踪激活范数 + 状态范数 + 写入范数, 找增长源."""
import sys, os, torch, random
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_ced30 import CED30, V, DEV, P, Q
from lowrank import RubikLowRankFast

LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "norm_diag.log")
SFT = r"F:\rowcol_llm\sft_cached_256.pt"
PT_CKPT = r"F:\OpenASH2605\copyfirst_redesign\cedlr30_pt_step16000.pth"

def log(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

def main():
    random.seed(0)
    torch.manual_seed(0)
    cache = torch.load(SFT, map_location="cpu", weights_only=True)
    G, T = cache["grids"], cache["tgts"]
    NS = G.shape[0]
    m = CED30().to(DEV)
    m.load_state_dict(torch.load(PT_CKPT, map_location=DEV, weights_only=True))
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=0.01)

    # 每层输出范数 hook
    layernorms = {}
    def mk(name):
        def hook(mod, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            layernorms[name] = o.float().pow(2).mean().sqrt().item()
        return hook
    hs = []
    for i, l in enumerate(m.enc):
        hs.append(l.register_forward_hook(mk(f"enc{i}.{type(l).__name__[:6]}")))
    for i, l in enumerate(m.dec):
        hs.append(l.register_forward_hook(mk(f"dec{i}.{type(l).__name__[:6]}")))

    # rubik 状态范数追踪 (编码器第 1 层 + 解码器第 1 层)
    trace = []
    orig_scan = []
    import lowrank
    from scan import affine_scan as real_scan
    def scan_wrap(A, W):
        r0 = A[0, 0].norm().item(); r1 = A[0, int(A.shape[1] * 0.5)].norm().item()
        r2 = A[:, -1].norm(dim=(-2,-1)).max().item()
        trace.append((r0, r1, A.abs().max().item()))
        return real_scan(A, W)

    # patch: 找到 rubik attn 的 scan 调用点 (bench_ced30.RubikAttn30 直接调 affine_scan)
    import bench_ced30
    bench_ced30.affine_scan = scan_wrap

    for st in range(2000):
        idx = torch.randint(0, NS, (32,))
        seq = (G[idx] - 34).clamp(min=0).to(DEV).reshape(32, 256)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = m(seq)
        logits = logits.float()
        ys = seq[:, P + 1:P + Q]
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, V), ys.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0).item()
        opt.step()
        if st % 50 == 0 or (trace and trace[-1][2] > 50):
            lns = " ".join("%s:%.1f" % (k, v) for k, v in layernorms.items())
            log("st%d loss=%.3f gn=%.2e | %s | scanA_max=%.1f" %
                (st, loss.item(), gn, lns, trace[-1][2] if trace else -1))
        if trace and trace[-1][2] > 500:
            log("ST%d scan A 范数爆炸 %.1f — 停止" % (st, trace[-1][2]))
            break
    log("END")

if __name__ == "__main__":
    main()
