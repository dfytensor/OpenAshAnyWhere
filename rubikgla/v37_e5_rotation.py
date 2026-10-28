#!/usr/bin/env python3
"""E5: 整模轮换端侧 demo (V37-E5).
两个模型实例交替常驻 (模拟边缘设备显存只容一个), state (20KB 级) 交接,
验证: (1) 轮换后 NLL 与连续前向逐位一致 (2) 每次轮换的搬运耗时.
"""
import sys, os, time, random
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import deqlm_v35_relay as M
from deqlm_v5_30m import PT_CACHE

DEV = "cuda"
CPU = "cpu"
torch.manual_seed(0)

seqs = torch.load(PT_CACHE, map_location="cpu", weights_only=True)
val = [s for s in seqs[int(0.95 * len(seqs)):][:2000] if len(s) >= 256]
rng0 = random.Random(3)


def build_ids(n, ln):
    out = []
    for _ in range(n):
        buf, tot = [], 0
        while tot < ln:
            s = val[rng0.randrange(len(val))]
            buf.append(s)
            tot += len(s)
        out.append(torch.cat(buf)[:ln])
    return torch.stack(out).to(DEV)


ids = build_ids(2, 8192)
SEG = 2048
N_SEG = ids.shape[1] // SEG


def build():
    net = M.DeltaLoopLM()
    net.load_state_dict(torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                "deqlmv31r_pt_full.pth"),
                                   map_location="cpu", weights_only=True), strict=False)
    return net


log = lambda m: print(m, flush=True)

# ---- 参考: 单实例连续前向 (最后一段 logits) ----
net_ref = build().to(DEV).eval()
with torch.no_grad():
    cur = None
    for i in range(N_SEG):
        seg = ids[:, i * SEG:(i + 1) * SEG]
        cur = net_ref.capture_states(seg, s0_list=cur)
        if i == N_SEG - 1:
            lo_ref = net_ref(seg, s0_list=cur)
tgt = ids[:, -2047:]

def suf(lo):
    lo = lo[:, :2047].float()
    return F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tgt.reshape(-1), ignore_index=0).item()

log("参考 连续8段链 NLL = %.4f" % suf(lo_ref))

# ---- E5: 整模轮换 (A/B 交替常驻, 非常驻者移出 GPU) ----
A, B = build().to(DEV).eval(), build().to(DEV).eval()
cur_states = None
swap_times = []
t_rot0 = time.time()
with torch.no_grad():
    for i in range(N_SEG):
        seg = ids[:, i * SEG:(i + 1) * SEG]
        resident = A if i % 2 == 0 else B
        offload = B if i % 2 == 0 else A
        offload.to(CPU)                                   # 非常驻 → 移出 GPU (模拟显存释放)
        resident.to(DEV)                                  # 常驻 → 载入 GPU
        torch.cuda.synchronize()
        t_swap = time.time()
        cur_states = resident.capture_states(seg, s0_list=cur_states)
        if i == N_SEG - 1:
            lo_rot = resident(seg, s0_list=cur_states)
        torch.cuda.synchronize()
        if i > 0:
            swap_times.append(time.time() - t_swap)
        log("  段 %d: 常驻=%s (另一实例已移出GPU) state=5×%.1fKB"
            % (i, "A" if i % 2 == 0 else "B", sum(s.numel() * 4 for s in cur_states) / 1024))
rot_total = time.time() - t_rot0

nll_rot = suf(lo_rot)
swap_ms = sum(swap_times) / len(swap_times) * 1e3
log("E5 轮换 NLL = %.4f (参考 %.4f, Δ %.2e)" % (nll_rot, suf(lo_ref), abs(nll_rot - suf(lo_ref))))
log("每次轮换 (capture+收尾) 平均 %.0f ms × %d 次 | 8段总耗时 %.1f s" % (swap_ms, len(swap_times), rot_total))
log("state 交接体积: 5 站 × (10,32,32) fp32 ≈ %.1f KB" % (5 * 10 * 32 * 32 * 4 / 1024))
log("E5 DONE", flush=True)
