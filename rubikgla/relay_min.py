#!/usr/bin/env python3
"""最小实验: 两模型接力处理长上下文.
A 训 [0,128); B 训 [128,256) 以 A 末状态起始; 对照 B0(无状态) 与 Full(单模型训全 256).
指标: 位置 [129,256) 的 NLL (同 127 目标 token).
"""
import sys, os, random, time, math, json
import torch
import torch.nn.functional as F

HERE = r"F:\夸克\rubikgla"
sys.path.insert(0, HERE)
from deqlm_v5_30m import PT_CACHE
import deqlm_v35_relay as M

DEV = "cuda"
STEPS = 20000
BS = 32
LR = 4e-4


def log(msg, path=None):
    print(msg, flush=True)
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


def get_halfA(data, bs, rng, sl=128):
    xs, ys = [], []
    for _ in range(bs):
        s = data[rng.randrange(len(data))][:sl]
        xs.append(s)
        ys.append(torch.cat([s[1:], torch.zeros(1, dtype=s.dtype)]))
    x = F.pad(torch.stack(xs).to(DEV), (0, 0))
    y = F.pad(torch.stack(ys).to(DEV), (0, 0))
    return x, y


def get_halfB(data, bs, rng):
    xs, ys = [], []
    for _ in range(bs):
        s = data[rng.randrange(len(data))][128:256]
        xs.append(s[:-1])
        ys.append(s[1:])
    x = F.pad(torch.stack(xs).to(DEV), (0, 0))
    y = F.pad(torch.stack(ys).to(DEV), (0, 0))
    return x, y


def train_lm(m, data, steps, bs, rng_seed, lr=4e-4, half=None, A=None, logp=None):
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95), fused=True)
    rng = random.Random(rng_seed)
    m.train()
    t0 = time.time()
    nan_run = 0
    for st in range(steps):
        if half == "A":
            x, y = get_halfA(data, bs, rng)
        elif half == "B":
            x, y = get_halfB(data, bs, rng)
        else:
            x, y = get_batch_pub(data, bs, rng)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lo = m(x, s0_list=[s.to(torch.bfloat16) for s in A_states(m, A, x)] if A is not None else None)
        l = F.cross_entropy(lo.float().reshape(-1, lo.shape[-1]), y.reshape(-1), ignore_index=0)
        if not torch.isfinite(l):
            nan_run += 1
            if nan_run > 200:
                raise RuntimeError("NaN runaway")
            continue
        nan_run = 0
        l.backward()
        gn = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        if torch.isfinite(gn):
            opt.step()
        if st % 4000 == 0:
            log("  %d/%d loss=%.4f (%.0fms)" % (st, steps, l.item(), (time.time() - t0) / (st + 1) * 1000), logp)
    return m


def get_batch_pub(data, bs, rng, sl=256):
    xs = []
    for _ in range(bs):
        s = data[rng.randrange(len(data))][:sl]
        xs.append(F.pad(s, (0, sl - s.numel())))
    x = torch.stack(xs).to(DEV)
    y = x.roll(-1, dims=1)
    y[:, -1] = 0
    return x, y


def A_states(mA, mB, xB):
    """A 冻结推断: xB 是 B 的输入 (token 128..), A 的输入应为 token 0..127 = xB 的前 128 减一偏移...
    实际: B 的输入 t[128:255]; A 的输入 = 完整序列前 128 = t[0:128]. 由调用方直接给 A 输入更清晰.
    这里简化: relay 阶段直接传 A 输入."""
    raise NotImplementedError


def second_half_nll(mB, mA, val, bs=32, with_state=True, full_model=None):
    mB.eval()
    tot, tok = 0.0, 0
    rng = random.Random(777)
    n = min(len(val), 1000)
    for i in range(0, n, bs):
        chunk = val[i:i + bs]
        xs, ys = [], []
        for s in chunk:
            t = s[128:256]
            xs.append(t[:-1])
            ys.append(t[1:])
        L = min(len(x) for x in xs)
        x = torch.stack([x[:L] for x in xs]).to(DEV)
        y = torch.stack([y[:L] for y in ys]).to(DEV)
        with torch.no_grad():
            if full_model is not None:
                xf = torch.stack([s[:255] for s in chunk]).to(DEV)
                lo = full_model(xf)
                lg = lo.float()[:, 128:255, :]
            elif with_state:
                st = mA.capture_states(torch.stack([s[:128] for s in chunk]).to(DEV))
                st = [s.to(torch.bfloat16) for s in st]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lo = mB(x, s0_list=st)
                lg = lo.float()
            else:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lo = mB(x)
                lg = lo.float()
        l = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), y.reshape(-1), reduction="sum")
        tot += l.item()
        tok += y.numel()
    mB.train()
    return tot / tok


# ============ 主流程 ============
logp = os.path.join(HERE, "relay_min.log")
log("== 接力最小实验 ==", logp)
torch.manual_seed(0)
random.seed(0)
seqs = torch.load(PT_CACHE, map_location="cpu", weights_only=True)
seqs = [s for s in seqs if len(s) >= 257]
tr = seqs[: int(0.95 * len(seqs))]
va = seqs[int(0.95 * len(seqs)):][:1000]
print("len>=257 序列数:", len(seqs), flush=True)

# Phase A
print("== Phase A: 训练 [0,128) ==", flush=True)
A = M.DeltaLoopLM().to(DEV)
train_lm(A, tr, STEPS, BS, 42, half="A", logp=logp)
A.eval()
for p in A.parameters():
    p.requires_grad_(False)
torch.save(A.state_dict(), os.path.join(HERE, "relay_A.pth"))

# Phase B (带状态)
print("== Phase B: 训练 [128,256) + A 状态 ==", flush=True)
Bm = M.DeltaLoopLM().to(DEV)


def A_states_fix(mA, xA):
    return [s.to(torch.bfloat16) for s in mA.capture_states(xA)]


def train_B(mB, use_state, steps=STEPS):
    opt = torch.optim.AdamW(mB.parameters(), lr=LR, weight_decay=0.01, betas=(0.9, 0.95), fused=True)
    rng = random.Random(123)
    mB.train()
    t0 = time.time()
    for st in range(steps):
        idsA = torch.stack([tr[rng.randrange(len(tr))][:128] for _ in range(BS)]).to(DEV)
        x, y = get_halfB(tr, BS, rng)
        st_states = [s.to(torch.bfloat16) for s in A.capture_states(idsA)] if use_state else None
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lo = mB(x, s0_list=st_states)
        l = F.cross_entropy(lo.float().reshape(-1, lo.shape[-1]), y.reshape(-1), ignore_index=0)
        if not torch.isfinite(l):
            opt.zero_grad(set_to_none=True)
            continue
        l.backward()
        gn = torch.nn.utils.clip_grad_norm_(mB.parameters(), 1.0)
        if torch.isfinite(gn):
            opt.step()
        if st % 4000 == 0:
            log("  %d/%d loss=%.4f (%.0fms)" % (st, steps, l.item(), (time.time() - t0) / (st + 1) * 1000), logp)
    return mB


train_B(Bm, True)
torch.save(Bm.state_dict(), os.path.join(HERE, "relay_B.pth"))

# Phase B0 (无状态对照)
print("== Phase B0: 无状态对照 ==", flush=True)
B0 = M.DeltaLoopLM().to(DEV)
train_B(B0, False)
torch.save(B0.state_dict(), os.path.join(HERE, "relay_B0.pth"))

# Phase Full (单模型参考)
print("== Phase Full: 单模型 [0,256) ==", flush=True)
Fm = M.DeltaLoopLM().to(DEV)
train_lm(Fm, tr, STEPS, BS, 42, half=None, logp=logp)
torch.save(Fm.state_dict(), os.path.join(HERE, "relay_Full.pth"))

# ============ 评测 ============
print("== 评测: [129,256) NLL ==", flush=True)
n_state = second_half_nll(Bm, A, va, with_state=True)
n_none = second_half_nll(B0, None, va, with_state=False)
n_full = second_half_nll(None, None, va, full_model=Fm)
res = dict(B_with_state=n_state, B0_no_state=n_none, Full_ref=n_full)
print(json.dumps(res, indent=2))
json.dump(res, open(os.path.join(HERE, "relay_min_results.json"), "w"), indent=2)
log("== 完成 ==", logp)
