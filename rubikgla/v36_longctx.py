#!/usr/bin/env python3
"""v36 长上下文决定性实验: L=4096, CHUNK=128, K=32.
后缀 [2049,4096) NLL 三方: relay-s0(2048链) vs v36-MS bank vs 零state.
数据: 256-token 序列拼接成 4096 块. 从 v31 checkpoint 热启动 (同 v36_multistate v2).
"""
import sys, os, time, random
os.environ.setdefault("DELTA_STAGES", "1")   # 修: capture@4096 偶发异步故障, 关内核流水线
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import deqlm_v35_relay as M
from deqlm_v5_30m import PT_CACHE
from v36_multistate import DeltaLoopLM_MS

DEV = "cuda"
V = M.V
LONG = 4096
CHUNK = 128
STEPS = 2000
LR = 3e-5
BS = 2
HERE = os.path.dirname(os.path.abspath(__file__))


def log(msg, **kw):
    print(msg, flush=True)


def build_blocks(seqs, n, ln, rng):
    out = []
    while len(out) < n:
        buf = []
        tot = 0
        while tot < ln:
            s = seqs[rng.randrange(len(seqs))]
            buf.append(s)
            tot += len(s)
        out.append(torch.cat(buf)[:ln])
    return out


def main():
    torch.manual_seed(0)
    rng_data = random.Random(7)
    seqs = torch.load(PT_CACHE, map_location="cpu", weights_only=True)
    train = seqs[:int(0.95 * len(seqs))]
    val = [s for s in seqs[int(0.95 * len(seqs)):][:2000] if len(s) >= 256]
    log("长上下文: train %d / val %d 序列" % (len(train), len(val)))

    net = M.DeltaLoopLM()
    net.load_state_dict(torch.load(os.path.join(HERE, "deqlmv31r_pt_full.pth"),
                                   map_location="cpu", weights_only=True))
    net = net.to(DEV).eval()

    model = DeltaLoopLM_MS(net).to(DEV)
    model.train()
    # 诊断: compile@L=4096 数值异常 (train前向 loss 12 vs eval 2.34) — 长上下文禁用 compile
    for blk in (model.net.pre, model.net.loop_a.attn, model.net.bridge,
                model.net.loop_c.attn, model.net.post):
        blk._attn_impl = blk.attn._apply_raw
        blk._ffn_impl = blk.ffn.forward

    ids_eval = torch.stack(build_blocks(val[:len(val) // 8], 12, LONG, rng_data)).to(DEV)
    tgt = ids_eval[:, 2049:LONG]

    def suffix(logits, start=0):
        lo = logits[:, start:start + 2047].float()
        return F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tgt.reshape(-1), ignore_index=0).item()

    with torch.no_grad():
        lo_zero = net(ids_eval[:, 2048:])
        s0 = net.capture_states(ids_eval[:, :2048])
        lo_relay = net(ids_eval[:, 2048:], s0_list=s0)
        lo_ms = model(ids_eval, chunk=CHUNK, ramp=1.0)
    log("4096后缀 NLL [finetune前]: 零state %.4f | relay-s0(2048) %.4f | v36-MS %.4f"
        % (suffix(lo_zero, 0), suffix(lo_relay, 0), suffix(lo_ms, 2048)))

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01, betas=(0.9, 0.95), fused=True)
    rng = random.Random(42)
    t0 = time.time()
    for stp in range(STEPS):
        x = torch.stack(build_blocks(train, BS, LONG, rng)).to(DEV)
        opt.zero_grad(set_to_none=True)
        ramp = min(1.0, stp / 300)                              # bank 读出 0→1 暖启动
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lo = model(x, chunk=CHUNK, ramp=ramp)
        loss = F.cross_entropy(lo.float()[:, :-1].reshape(-1, V), x[:, 1:].reshape(-1), ignore_index=0)
        if not torch.isfinite(loss):
            opt.zero_grad(set_to_none=True)
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if stp % 100 == 0 or stp == STEPS - 1:
            model.eval()
            with torch.no_grad():
                lo_e = model(ids_eval, chunk=CHUNK, ramp=1.0)
            se = suffix(lo_e, 2048)
            model.train()
            log("step %d/%d loss %.4f 4096后缀 %.4f (零 %.4f / relay %.4f) (%.0fs)"
                % (stp, STEPS, loss.item(), se, suffix(lo_zero, 0), suffix(lo_relay, 0), time.time() - t0), flush=True)
        if stp in (999, 1499, STEPS - 1):                   # 周期存档: 崩溃不丢模型
            torch.save(model.state_dict(), os.path.join(HERE, "v36ms_long.pth"))
            log("  [存档] v36ms_long.pth @step %d" % stp, flush=True)

    model.eval()
    try:
        with torch.no_grad():
            lo_ms = model(ids_eval, chunk=CHUNK, ramp=1.0)
        log("=== 4096 终评 === 零state %.4f | relay-s0(2048) %.4f | v36-MS %.4f"
            % (suffix(lo_zero, 0), suffix(lo_relay, 0), suffix(lo_ms, 2048)))
    except Exception as e:
        log("终评异常 (模型已存档): %r" % (e,))
    torch.save(model.state_dict(), os.path.join(HERE, "v36ms_long.pth"))
    log("checkpoint: v36ms_long.pth")


if __name__ == "__main__":
    main()
