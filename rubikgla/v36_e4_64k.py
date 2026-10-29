#!/usr/bin/env python3
"""E4-64k: 64k 长度内训练 — 分段截断 BPTT (8×8192, s0 链携带历史, 每段独立反传).
绕开 L=65536 autograd 内核 (driver 异常); 状态跨段连续, 梯度段内截断 (标准 truncated-BPTT).
sel_scale → 0.5 激活选择性. 从 v36ms_e4.pth (16k/3000步) 热启动.
终评: 64k 后缀 [63489,65536) 三方 — 零state / relay链(32×2048廉价接力) / E4-64k v36-MS.
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
LONG = 65536
SEG = 8192
N_SEG = LONG // SEG
CHUNK = 128
STEPS = 1000
LR = 2e-5
HERE = os.path.dirname(os.path.abspath(__file__))
RESUME = os.path.join(HERE, "v36ms_e4.pth")


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
    rng_data = random.Random(23)
    seqs = torch.load(PT_CACHE, map_location="cpu", weights_only=True)
    train = seqs[:int(0.95 * len(seqs))]
    val = [s for s in seqs[int(0.95 * len(seqs)):][:2000] if len(s) >= 256]

    net = build_ms_from_v31(enable_gate=True, enable_sel=True)
    model = DeltaLoopLM_MS(net).to(DEV)
    RESUME64 = os.path.join(HERE, "v36ms_e464.pth")
    start = 300
    if os.path.exists(RESUME64):
        model.load_state_dict(torch.load(RESUME64, map_location="cpu", weights_only=True), strict=False)
        log("E4-64k 续跑: 从 %d 步 checkpoint 恢复 → 目标 %d 步 (sel_scale 保持训练值)" % (start, STEPS))
    else:
        model.load_state_dict(torch.load(RESUME, map_location="cpu", weights_only=True), strict=False)
        with torch.no_grad():                               # 激活选择性
            for blk in (net.pre, net.loop_a.attn, net.bridge, net.loop_c.attn, net.post):
                blk.attn.sel_scale.fill_(0.5)
        log("E4-64k: 从 16k/3000步 ckpt 热启动; sel_scale → 0.5")
    model.train()
    for blk in (model.net.pre, model.net.loop_a.attn, model.net.bridge,
                model.net.loop_c.attn, model.net.post):     # 长上下文禁用 compile
        blk._attn_impl = blk.attn._apply_raw
        blk._ffn_impl = blk.ffn.forward
    log("E4-64k: 热启动 16k/3000步 ckpt; sel_scale→0.5; 分段截断BPTT (8×8192); compile 关")

    ids_eval = torch.stack(build_blocks(val, 1, LONG, rng_data)).to(DEV)
    tgt = ids_eval[:, LONG - 2047:LONG]

    def suffix(logits, start=0):
        lo = logits[:, start:start + 2047].float()
        return F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tgt.reshape(-1), ignore_index=0).item()

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=LR, weight_decay=0.01, betas=(0.9, 0.95), fused=True)
    rng = random.Random(99)
    t0 = time.time()
    for stp in range(start, STEPS):
        x = torch.stack(build_blocks(train, 1, LONG, rng)).to(DEV)
        opt.zero_grad(set_to_none=True)
        ramp = min(1.0, (stp + 1) / 150)
        losses = []
        s0_list = None
        for s in range(N_SEG):
            seg = x[:, s * SEG:(s + 1) * SEG]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lo = model(seg, chunk=CHUNK, ramp=ramp, s0_list=s0_list)
            loss_seg = F.cross_entropy(lo.float()[:, :-1].reshape(-1, 23005),
                                       seg[:, 1:].reshape(-1), ignore_index=0)
            if torch.isfinite(loss_seg):
                (loss_seg / N_SEG).backward()
                losses.append(loss_seg.item())
            # 下一段的 s0: 无梯度 capture (状态连续, 梯度段内截断 = truncated BPTT)
            with torch.no_grad():
                s0_list = [s_.detach() for s_ in net.capture_states(seg, s0_list=s0_list)]
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if torch.isfinite(gn):
            opt.step()
        if stp % 50 == 0 or stp == STEPS - 1:
            model.eval()
            try:
                torch.cuda.synchronize()
                with torch.no_grad():
                    lo_e = model(ids_eval, chunk=CHUNK, ramp=1.0)
                se = suffix(lo_e, LONG - 2048)
                model.train()
                log("step %d/%d loss %.4f 64k后缀 %.4f (%.0fs)"
                    % (stp, STEPS, sum(losses) / max(1, len(losses)), se, time.time() - t0), flush=True)
            except Exception as e:
                model.train()
                log("  [探针失败跳过] %r" % (e,), flush=True)
        if stp in (499, 699, 999) or stp == STEPS - 1:
            torch.save(model.state_dict(), os.path.join(HERE, "v36ms_e464.pth"))
            log("  [存档] v36ms_e464.pth @%d" % stp, flush=True)

    model.eval()
    with torch.no_grad():
        lo_zero = net(ids_eval[:, LONG - 2048:])
        cur_states = None
        lo_last = None
        for i in range(N_SEG):
            seg = ids_eval[:, i * SEG:(i + 1) * SEG]
            cur_states = net.capture_states(seg, s0_list=cur_states)
            if i == N_SEG - 1:
                lo_last = net(seg, s0_list=cur_states)
        lo_ms = model(ids_eval, chunk=CHUNK, ramp=1.0)
    log("=== 64k 终评 (E4-64k) ===")
    log("零state %.4f | relay-s0链(32×2048) %.4f | v36-MS %.4f"
        % (suffix(lo_zero, 0), suffix(lo_last, 0), suffix(lo_ms, LONG - 2048)))
    torch.save(model.state_dict(), os.path.join(HERE, "v36ms_e464.pth"))
    log("E4-64k DONE", flush=True)


if __name__ == "__main__":
    main()
