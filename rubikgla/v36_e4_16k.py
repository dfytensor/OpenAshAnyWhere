#!/usr/bin/env python3
"""E4: 16k 长度内训练 (长度外推定律的修复实验).
L=16384, CHUNK=128 (K=128), B=1. 分段 capture (每段 2048, s0 链) 避开 TDR.
从 deqlmv31r_pt_full.pth 热启动; ramp 0→1 (300 步); 课程式 (前 1/4 冻结主干).
终评: 16k 后缀 [14337,16384) 三方 — 零state / relay链(8×2048) / E4-v36-MS.
"""
import sys, os, time, random
os.environ.setdefault("DELTA_STAGES", "1")
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import deqlm_v35_relay as M
from deqlm_v5_30m import PT_CACHE
from v36_multistate import DeltaLoopLM_MS, build_ms_from_v31, log

DEV = "cuda"
LONG = 16384
CHUNK = 128
STEPS = 3000
LR = 3e-5
BS = 1
HERE = os.path.dirname(os.path.abspath(__file__))
STATIONS = 5


def capture_segmented(net, ids, chunk, max_seg=2048):
    L = ids.shape[1]
    bounds_all = list(range(chunk - 1, L, chunk))
    banks = [[] for _ in range(STATIONS)]
    cur = None
    i = 0
    while i < L:
        j = min(i + max_seg, L)
        seg = ids[:, i:j]
        bnd = [b - i for b in bounds_all if i <= b < j]
        ends, bs = net.capture_states(seg, boundary_idx=bnd, s0_list=cur)
        for k in range(STATIONS):
            banks[k].append(bs[k])
        cur = ends
        i = j
    return [torch.cat(b, dim=1) for b in banks]


def build_blocks(seqs, n, ln, rng):
    out = []
    while len(out) < n:
        buf, tot = [], 0
        while tot < ln:
            s = seqs[rng.randrange(len(seqs))]
            buf.append(s)
            tot += len(s)
        out.append(torch.cat(buf)[:ln])
    return out


def main():
    torch.manual_seed(0)
    rng_data = random.Random(13)
    seqs = torch.load(PT_CACHE, map_location="cpu", weights_only=True)
    train = seqs[:int(0.95 * len(seqs))]
    val = [s for s in seqs[int(0.95 * len(seqs)):][:2000] if len(s) >= 256]
    log("E4 16k 长度内训练: train %d / val %d" % (len(train), len(val)))

    net = build_ms_from_v31(enable_gate=True, enable_sel=True)
    model = DeltaLoopLM_MS(net).to(DEV)
    model.train()

    RESUME = os.path.join(HERE, "v36ms_e4.pth")
    start = 0
    if os.path.exists(RESUME):
        model.load_state_dict(torch.load(RESUME, map_location="cpu", weights_only=True))
        start = 600
        log("E4 续跑: 从 %d 步 checkpoint 恢复 → 目标 %d 步" % (start, STEPS))
    freeze_until = 1500
    if start < freeze_until:
        for p in model.net.parameters():
            p.requires_grad = False
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=LR * 5 if start < freeze_until else LR,
                            weight_decay=0.01, betas=(0.9, 0.95), fused=True)

    ids_eval = torch.stack(build_blocks(val, 4, LONG, rng_data)).to(DEV)
    tgt = ids_eval[:, LONG - 2047:LONG]

    def suffix(logits, start):
        lo = logits[:, start:start + 2047].float()
        return F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tgt.reshape(-1), ignore_index=0).item()

    # 参考锚点 (预训练 v31, 训练前测一次)
    with torch.no_grad():
        lo_zero_pre = net(ids_eval[:, LONG - 2048:])
        s0p = net.capture_states(ids_eval[:, :2048])
        cur, lo_r = ids_eval[:, :2048], None
        for seg_i in range(1, 8):
            s0p = net.capture_states(cur)
            nxt = ids_eval[:, seg_i * 2048:(seg_i + 1) * 2048]
            lo_r = net(nxt, s0_list=s0p)
            cur = torch.cat([cur, nxt], dim=1)
    log("参考 [预训练v31]: 零state %.4f | relay链 %.4f" % (suffix(lo_zero_pre, 0), suffix(lo_r, 0)))

    # ---- finetune (课程式) ----
    opt = torch.optim.AdamW(model.ms.parameters(), lr=LR * 5, weight_decay=0.01,
                            betas=(0.9, 0.95), fused=True)
    trunk = list(model.net.parameters())
    for p in trunk:
        p.requires_grad = False
    rng = random.Random(42)
    t0 = time.time()
    freeze_until = 1500
    unfrozen = start >= freeze_until
    for stp in range(start, STEPS):
        x = torch.stack(build_blocks(train, BS, LONG, rng)).to(DEV)
        opt.zero_grad(set_to_none=True)
        ramp = min(1.0, (stp + 1) / 300)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lo = model(x, chunk=CHUNK, ramp=ramp)
        loss = F.cross_entropy(lo.float()[:, :-1].reshape(-1, V := 23005),
                               x[:, 1:].reshape(-1), ignore_index=0)
        if not torch.isfinite(loss) or not loss.requires_grad:
            opt.zero_grad(set_to_none=True)
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        if (not unfrozen) and stp >= freeze_until:
            for p in trunk:
                p.requires_grad = True
            opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01,
                                    betas=(0.9, 0.95), fused=True)
            unfrozen = True
            log("  [课程] 主干解冻", flush=True)
        if stp % 250 == 0 or stp == STEPS - 1:
            model.eval()
            try:
                torch.cuda.synchronize()
                with torch.no_grad():
                    lo_e = model(ids_eval, chunk=CHUNK, ramp=1.0)
                se = suffix(lo_e, LONG - 2048)
                model.train()
                log("step %d/%d loss %.4f 16k后缀 %.4f (%.0fs)"
                    % (stp, STEPS, loss.item(), se, time.time() - t0), flush=True)
            except Exception as e:
                model.train()
                log("  [探针失败跳过] %r" % (e,), flush=True)
        if stp in (999, 1499, 1999, 2499, 2999) or stp == STEPS - 1:
            torch.save(model.state_dict(), os.path.join(HERE, "v36ms_e4.pth"))
            log("  [存档] v36ms_e4.pth @%d" % stp, flush=True)

    # ---- 16k 三方终评 (E4 模型自身主干做 ramp0 对照) ----
    model.eval()
    with torch.no_grad():
        lo_ms = model(ids_eval, chunk=CHUNK, ramp=1.0)
        lo_r0 = model(ids_eval, chunk=CHUNK, ramp=0.0)
    log("=== 16k 终评 (E4) ===")
    log("参考: 预训练v31 零state %.4f | relay链 %.4f" % (suffix(lo_zero_pre, 0), suffix(lo_r, 0)))
    log("E4:   ramp0(自身主干) %.4f | ** v36-MS %.4f **"
        % (suffix(lo_r0, LONG - 2048), suffix(lo_ms, LONG - 2048)))
    torch.save(model.state_dict(), os.path.join(HERE, "v36ms_e4.pth"))
    log("checkpoint: v36ms_e4.pth")
    log("E4 DONE", flush=True)


if __name__ == "__main__":
    main()
