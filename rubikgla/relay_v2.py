#!/usr/bin/env python3
"""正确的接力最小实验: A 和 B 各自按自己的段长度从头训练.
A: [0,128) 窗口 LM (20k 步)
B: [128,256) 窗口 LM, 以 A 末状态 S0 起始 (20k 步)
对照: Full 单模型 [0,256) (20k 步)
评测: 三者在 [129,256) 的 NLL
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
torch.manual_seed(0); random.seed(0)
seqs = torch.load(PT_CACHE, map_location="cpu", weights_only=True)
seqs = [s for s in seqs if len(s) >= 257]
tr = seqs[: int(0.95 * len(seqs))]
va = seqs[int(0.95 * len(seqs)):][:500]
print(f"train={len(tr)} val={len(va)}", flush=True)

logf = open(os.path.join(HERE, "relay_v2.log"), "a", encoding="utf-8")
def log(msg):
    print(msg, flush=True)
    logf.write(msg + "\n"); logf.flush()

def get_halfA(data, bs, rng):
    xs, ys = [], []
    for _ in range(bs):
        s = data[rng.randrange(len(data))][:128]
        xs.append(s); ys.append(torch.cat([s[1:], torch.zeros(1, dtype=s.dtype)]))
    return (torch.stack(xs).to(DEV), F.pad(torch.stack(ys).to(DEV), (0,0)))

def get_halfB(data, bs, rng):
    xs, ys = [], []
    for _ in range(bs):
        s = data[rng.randrange(len(data))][128:256]
        xs.append(s[:-1]); ys.append(s[1:])
    return torch.stack(xs).to(DEV), torch.stack(ys).to(DEV)

def get_full(data, bs, rng):
    xs = []
    for _ in range(bs):
        xs.append(data[rng.randrange(len(data))][:256])
    x = torch.stack(xs).to(DEV)
    y = x.roll(-1, dims=1); y[:, -1] = 0
    return x, y


def train_model(m, data_fn, steps, lr0=4e-4, warmup=2000, tag=""):
    opt = torch.optim.AdamW(m.parameters(), lr=lr0, weight_decay=0.01, betas=(0.9,0.95), fused=True)
    rng = random.Random(42)
    m.train()
    nan_run = 0
    t0 = time.time()
    for st in range(steps):
        lr = lr0 * min(1.0, st / warmup) * (0.5 * (1 + math.cos(math.pi * st / steps)) + 0.1 * (1 - 0.5 * (1 + math.cos(math.pi * st / steps))))
        for g in opt.param_groups: g["lr"] = lr
        x, y = data_fn(st)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lo = m(x)
        l = F.cross_entropy(lo.float().reshape(-1, lo.shape[-1]), y.reshape(-1), ignore_index=0)
        if not torch.isfinite(l):
            nan_run += 1
            if nan_run > 200: raise RuntimeError("NaN runaway @%d" % st)
            opt.zero_grad(set_to_none=True); continue
        nan_run = 0
        l.backward()
        gn = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        if torch.isfinite(gn): opt.step()
        if st % 5000 == 0:
            log("  %s %d/%d loss=%.4f (%.0fms)" % (tag, st, steps, l.item(), (time.time()-t0)/(st+1)*1000))
    torch.cuda.synchronize()
    log("  %s done: %.0fms/步" % (tag, (time.time()-t0)/steps*1000))


@torch.no_grad()
def eval_halfB(mB, s0_provider, val, bs=32):
    """评测 [129,256) NLL."""
    mB.eval()
    tot, tok = 0.0, 0
    for i in range(0, min(len(val), 200), bs):
        chunk = val[i:i+bs]
        idsA = torch.stack([s[:128] for s in chunk]).to(DEV)
        xs, ys = [], []
        for s in chunk:
            t = s[128:256]; xs.append(t[:-1]); ys.append(t[1:])
        L = min(len(x) for x in xs)
        x = torch.stack([x[:L] for x in xs]).to(DEV)
        y = torch.stack([y[:L] for y in ys]).to(DEV)
        if s0_provider:
            s0_list = s0_provider(idsA)
            s0_list = [s.to(torch.bfloat16) for s in s0_list]
        else:
            s0_list = None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lo = mB(x, s0_list=s0_list)
        l = F.cross_entropy(lo.float().reshape(-1, lo.shape[-1]), y.reshape(-1), reduction="sum")
        tot += l.item(); tok += y.numel()
    mB.train()
    return tot / tok


# ============ Phase A: 训练 [0,128) ============
log("== Phase A: [0,128) LM ==")
A = M.DeltaLoopLM().to(DEV)
A_train = 20000

# 简单包装: get_halfA 返回 (x, y)
def data_A(st):
    return get_halfA(tr, BS, rng_A)
rng_A = random.Random(42)

log("  A training %d steps..." % A_train)
train_model(A, data_A, A_train, tag="A")
A.eval()
for p in A.parameters(): p.requires_grad_(False)
torch.save(A.state_dict(), os.path.join(HERE, "relay2_A.pth"))
log("  A done, saved")


# ============ S0 provider ============
def s0_from_A(idsA):
    """A 的逐站末状态."""
    with torch.no_grad():
        return A.capture_states(idsA)


# ============ Phase B: 训练 [128,256) 以 A 状态起始 ============
log("== Phase B: [128,256) 以 A S0 起始 ==")
B = M.DeltaLoopLM().to(DEV)
rng_B = random.Random(43)

def data_B(st):
    idsA, xB, yB = [], [], []
    for _ in range(BS):
        s = tr[rng_B.randrange(len(tr))]
        idsA.append(s[:128])
        t = s[128:256]
        xB.append(t[:-1]); yB.append(t[1:])
    return (torch.stack(idsA).to(DEV), torch.stack(xB).to(DEV), torch.stack(yB).to(DEV))

# 自定义训练: A 前向提取状态 → B 消费
def train_relay_B(steps=20000):
    opt = torch.optim.AdamW(B.parameters(), lr=4e-4, weight_decay=0.01, betas=(0.9,0.95), fused=True)
    B.train()
    t0 = time.time()
    nan_run = 0
    for st in range(steps):
        lr = 4e-4 * min(1.0, st / 2000) * (0.5 * (1 + math.cos(math.pi * st / steps)) + 0.1 * (1 - 0.5 * (1 + math.cos(math.pi * st / steps))))
        for g in opt.param_groups: g["lr"] = lr
        idsA, xB, yB = data_B(st)
        opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            s0_list = A.capture_states(idsA) if False else None
        # A 冻结, 不需要梯度; B 前向以 s0 起始
        with torch.autocast("cuda", dtype=torch.bfloat16):
            # A 前向 (无梯度): 提取状态
            A.eval()
            with torch.no_grad():
                s0_list = A.capture_states(idsA)
            B.train()
            lo = B(xB, s0_list=s0_list)
        l = F.cross_entropy(lo.float().reshape(-1, lo.shape[-1]), yB.reshape(-1), ignore_index=0)
        if not torch.isfinite(l):
            nan_run += 1
            if nan_run > 200: raise RuntimeError("NaN runaway")
            opt.zero_grad(set_to_none=True); continue
        nan_run = 0
        l.backward()
        gn = torch.nn.utils.clip_grad_norm_(B.parameters(), 1.0)
        if torch.isfinite(gn): opt.step()
        if st % 5000 == 0:
            log("  B %d/%d loss=%.4f (%.0fms)" % (st, l.item(), (time.time()-t0)/(st+1)*1000))
    log("  B relay-train done")

log("  B training 20k steps (with A states)...")
train_relay_B(20000)
torch.save(B.state_dict(), os.path.join(HERE, "relay2_B.pth"))

# ============ Phase Full: 单模型 [0,256) ============
log("== Phase Full: [0,256) LM ==")
Full = M.DeltaLoopLM().to(DEV)
rng_F = random.Random(44)
def data_F(st):
    return get_full(tr, BS, rng_F)
log("  Full training 20k steps...")
train_model(Full, data_F, 20000, tag="Full")
torch.save(Full.state_dict(), os.path.join(HERE, "relay2_Full.pth"))

# ============ 评测 ============
log("== 评测 [129,256) NLL ==")
A.eval(); B.eval(); Full.eval()

# B+state
n_relay = eval_halfB(B, lambda idsA: A.capture_states(idsA), va)
# Full
n_full = 0.0; ftok = 0
with torch.no_grad():
    for i in range(0, min(len(va), 200), 32):
        chunk = va[i:i+32]
        ids = torch.stack([s[:256] for s in chunk]).to(DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lo = Full(ids)
        lf = F.cross_entropy(lo.float()[:, 128:255].reshape(-1, lo.shape[-1]),
                             ids[:, 129:256].reshape(-1), reduction="sum")
        n_full += lf.item(); ftok += 16 * 127
n_full /= ftok

# B0 (无状态)
B0 = M.DeltaLoopLM().to(DEV)
B0.load_state_dict(B.state_dict())  # 同权重, 仅不给 s0
n_b0 = eval_halfB(B0, None, va)

res = dict(relay_B_with_state=n_relay, Full_ref=n_full, B0_no_state=n_b0)
log(json.dumps(res, indent=2))
json.dump(res, open(os.path.join(HERE, "relay2_results.json"), "w"), indent=2)
log("== 完成 ==")
