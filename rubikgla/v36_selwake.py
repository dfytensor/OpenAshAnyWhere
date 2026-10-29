#!/usr/bin/env python3
"""选择性唤醒实验: 检索密集混合数据 + sel_scale 非零初始化.
数据: 16k 序列, 前 8k 埋 16 对 (key→value), 后 8k 出 16 个查询 (key→须答 value), 余填充.
判据: ① 查询位 value 召回 acc/NLL ② 训后 sel_scale 生长? (对照: 纯pretrain数据 0.5→0 衰减)
      ③ sel 分布偏移 ④ 训后行为分析: python v37_behavior.py --ckpt v36ms_selwake.pth
"""
import sys, os, time, random
os.environ.setdefault("DELTA_STAGES", "1")
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deqlm_v5_30m import PT_CACHE
from v36_multistate import DeltaLoopLM_MS, build_ms_from_v31, log

DEV = "cuda"
LONG = 16384
CHUNK = 128
STEPS = 3000
LR = 1e-4
N_PAIR, N_Q = 16, 16
N_KEY = 64
HERE = os.path.dirname(os.path.abspath(__file__))
RESUME = os.path.join(HERE, "v36ms_e464.pth")
FILL0, KEY0, VAL0 = 100, 20001, 20065          # filler 100..20000, key 64 个, value 64 个
VOCAB = 23005


def make_sample(rng):
    """单条 16k 检索样本 → (ids (1,L), tgt (1,L), qmask (1,L))."""
    g = torch.Generator().manual_seed(rng.randint(0, 2**31 - 1))
    ids = torch.randint(FILL0, FILL0 + 19900, (1, LONG))
    tgt = torch.zeros(1, LONG, dtype=torch.long)
    qmask = torch.zeros(1, LONG)
    keys = KEY0 + torch.randperm(N_KEY, generator=g)
    vals = VAL0 + torch.randperm(N_KEY, generator=g)
    p = 1 + torch.randperm(8190, generator=g)[:2 * N_PAIR]
    for i in range(N_PAIR):
        p1, p2 = sorted([int(p[2 * i]), int(p[2 * i + 1])])
        ids[0, p1] = int(keys[i])
        ids[0, p2] = int(vals[i])
        tgt[0, p1] = int(vals[i])                          # key→value 绑定
    q = 8192 + 1 + torch.randperm(LONG - 8194, generator=g)[:N_Q]
    for i in range(N_Q):
        t = int(q[i])
        ids[0, t] = int(keys[i % N_KEY])                   # 查询: 只给 key
        tgt[0, t] = int(vals[i % N_KEY])                   # 须远程召回 value
        qmask[0, t] = 1.0
    return ids.to(DEV), tgt.to(DEV), qmask.to(DEV)


def main():
    torch.manual_seed(0)
    rng_data = random.Random(41)

    net = build_ms_from_v31(enable_gate=True, enable_sel=True)
    model = DeltaLoopLM_MS(net).to(DEV)
    model.load_state_dict(torch.load(RESUME, map_location="cpu", weights_only=True), strict=False)
    with torch.no_grad():                                  # 选择性唤醒: 双侧非零 (单侧=互锁死锁)
        for blk in (net.pre.attn, net.loop_a.attn.attn, net.bridge.attn,
                    net.loop_c.attn.attn, net.post.attn):
            blk.sel_scale.fill_(0.5)                       # 恒等性暂时破坏, 换取梯度通路
            blk.sel_A.weight.normal_(0, 0.01)
            blk.sel_B.weight.normal_(0, 0.01)
    model.train()
    log("selwake: sel_scale→0.5 起步 (检索密集数据)")

    evals = [make_sample(random.Random(1000 + i)) for i in range(4)]

    def recall_eval():
        model.eval()
        accs, nlls, n = [], [], 0
        with torch.no_grad():
            for ids, tgt, qm in evals:
                qm2 = qm.bool()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lo = model(ids, chunk=CHUNK, ramp=1.0)
                lo_q = lo[qm2].float()
                nlls.append(F.cross_entropy(lo_q, tgt[qm2], reduction="sum").item())
                accs.append((lo_q.argmax(-1) == tgt[qm2]).float().sum().item())
                n += int(qm2.sum().item())
        model.train()
        return sum(accs) / n, sum(nlls) / n

    acc0, qnll0 = recall_eval()
    log("初始 召回acc %.3f | 查询NLL %.4f (随机=1/64=%.3f... 查询只在64个key内, 随机≈%.3f)"
        % (acc0, qnll0, 1 / 64, 1 / 64), flush=True)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=LR, weight_decay=0.01, betas=(0.9, 0.95), fused=True)
    rng = random.Random(55)
    t0 = time.time()
    for stp in range(STEPS):
        ids, tgt, qm = make_sample(rng)
        opt.zero_grad(set_to_none=True)
        ramp = min(1.0, (stp + 1) / 150)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lo = model(ids, chunk=CHUNK, ramp=ramp)
        qm2 = qm.bool()
        loss = F.cross_entropy(lo[qm2].float(), tgt[qm2])
        if not torch.isfinite(loss) or not loss.requires_grad:
            opt.zero_grad(set_to_none=True)
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if stp % 100 == 0 or stp == STEPS - 1:
            acc, qnll = recall_eval()
            sc = [blk.sel_scale.detach().abs().mean().item()
                  for blk in (net.pre.attn, net.loop_a.attn.attn, net.bridge.attn,
                              net.loop_c.attn.attn, net.post.attn)]
            model.train()
            log("step %d/%d loss %.4f 召回acc %.3f 查询NLL %.4f | sel_scale %s (%.0fs)"
                % (stp, STEPS, loss.item(), acc, qnll,
                   "/".join("%.2f" % s for s in sc), time.time() - t0), flush=True)

    model.eval()
    acc, qnll = recall_eval()
    log("=== selwake 终评 === 召回acc %.3f | 查询NLL %.4f" % (acc, qnll))
    torch.save(model.state_dict(), os.path.join(HERE, "v36ms_selwake.pth"))
    log("checkpoint: v36ms_selwake.pth")
    log("训后行为分析: python v37_behavior.py --ckpt v36ms_selwake.pth")
    log("SELWAKE DONE", flush=True)


if __name__ == "__main__":
    main()
