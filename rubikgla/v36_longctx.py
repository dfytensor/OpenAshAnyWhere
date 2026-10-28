#!/usr/bin/env python3
"""v36 v3 4096 训练 (2000步) + 4096/16384 三方终评. crash-hardened: 存档先于探针, 探针带重试."""
import sys, os, time, random
os.environ.setdefault("DELTA_STAGES", "1")
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import deqlm_v35_relay as M
from deqlm_v5_30m import PT_CACHE
from v36_multistate import DeltaLoopLM_MS, build_ms_from_v31, finetune, log

DEV = "cuda"
LONG = 4096
CHUNK = 128
STEPS = 2000
LR = 3e-5
BS = 2
HERE = os.path.dirname(os.path.abspath(__file__))


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
    rng_data = random.Random(7)
    seqs = torch.load(PT_CACHE, map_location="cpu", weights_only=True)
    train = seqs[:int(0.95 * len(seqs))]
    val = [s for s in seqs[int(0.95 * len(seqs)):][:2000] if len(s) >= 256]
    log("v36-MS v3 长跑: train %d / val %d" % (len(train), len(val)))

    net = build_ms_from_v31()
    model = DeltaLoopLM_MS(net).to(DEV)
    model.train()

    ids_eval = torch.stack(build_blocks(val, 12, LONG, rng_data)).to(DEV)
    tgt = ids_eval[:, 2049:LONG]

    def suffix(logits, start=0):
        lo = logits[:, start:start + 2047].float()
        return F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tgt.reshape(-1), ignore_index=0).item()

    ckpt_path = os.path.join(HERE, "v36ms_v3.pth")
    with torch.no_grad():
        lo_zero = net(ids_eval[:, 2048:])
        s0 = net.capture_states(ids_eval[:, :2048])
        lo_relay = net(ids_eval[:, 2048:], s0_list=s0)
        lo_ms = model(ids_eval, chunk=CHUNK, ramp=1.0)
    log("4096后缀 NLL [finetune前]: 零state %.4f | relay-s0 %.4f | v36-MS %.4f"
        % (suffix(lo_zero, 0), suffix(lo_relay, 0), suffix(lo_ms, 2048)))

    finetune(model, train, STEPS, BS, LR, CHUNK, save_path=ckpt_path)
    torch.save(model.state_dict(), ckpt_path)

    model.eval()
    try:
        with torch.no_grad():
            lo_ms = model(ids_eval, chunk=CHUNK, ramp=1.0)
        log("=== 4096 终评 === 零state %.4f | relay-s0 %.4f | v36-MS %.4f"
            % (suffix(lo_zero, 0), suffix(lo_relay, 0), suffix(lo_ms, 2048)))
    except Exception as e:
        log("4096 终评失败 (checkpoint 已存): %r" % (e,))

    # ---- 16384 三方 (relay 8×2048 链 vs v36 全长 K=128) ----
    try:
        LONG2, SUFF = 16384, 2048
        ids16k = torch.stack(build_blocks(val, 4, LONG2, rng_data)).to(DEV)
        tgt16 = ids16k[:, LONG2 - SUFF + 1:LONG2]

        def suffix16(logits, start):
            lo = logits[:, start:start + SUFF - 1].float()
            return F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tgt16.reshape(-1), ignore_index=0).item()

        with torch.no_grad():
            lo_z = net(ids16k[:, LONG2 - SUFF:])
            cur = ids16k[:, :2048]
            for seg in range(1, LONG2 // 2048):
                s0c = net.capture_states(cur)
                nxt = ids16k[:, seg * 2048:(seg + 1) * 2048]
                lo_relay16 = net(nxt, s0_list=s0c)
                cur = torch.cat([cur, nxt], dim=1)
            lo_ms16 = model(ids16k, chunk=CHUNK, ramp=1.0)
        log("=== 16384 终评 === 零state %.4f | relay-s0链 %.4f | v36-MS %.4f"
            % (suffix16(lo_z, 0), suffix16(lo_relay16, 0), suffix16(lo_ms16, LONG2 - SUFF)))
    except Exception as e:
        log("16384 终评失败: %r" % (e,))
    log("DONE")


if __name__ == "__main__":
    main()
