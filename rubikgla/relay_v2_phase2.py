#!/usr/bin/env python3
"""从 Phase A checkpoint 继续跑 Phase B/B0/Full + 评测."""
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
print("train:", len(tr), "val:", len(va), flush=True)

logf = open(os.path.join(HERE, "relay_v2.log"), "a", encoding="utf-8")
def log(msg):
    print(msg, flush=True); logf.write(msg + "\n"); logf.flush()


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


# 加载 A
A = M.DeltaLoopLM().to(DEV)
A.load_state_dict(torch.load(os.path.join(HERE, "relay2_A.pth"), map_location="cpu"))
A.eval()
for p in A.parameters(): p.requires_grad_(False)
log("A loaded (frozen)", )


def s0_from_A(idsA):
    with torch.no_grad():
        return [s.to(torch.bfloat16) for s in A.capture_states(idsA)]


# Phase B: 训练 B 以 A 状态起始
log("== Phase B: [128,256) + A S0 ==")
B = M.DeltaLoopLM().to(DEV)
opt = torch.optim.AdamW(B.parameters(), lr=4e-4, weight_decay=0.01, betas=(0.9,0.95), fused=True)
rng = random.Random(43)
B.train()
t0 = time.time()
for st in range(STEPS):
    lr = 4e-4 * min(1.0, st / 2000) * (0.5 * (1 + math.cos(math.pi * st / STEPS)) + 0.1 * (1 - 0.5 * (1 + math.cos(math.pi * st / STEPS))))
    for g in opt.param_groups: g["lr"] = lr
    idsA = torch.stack([tr[rng.randrange(len(tr))][:128] for _ in range(BS)]).to(DEV)
    x, y = get_halfB(tr, BS, rng)
    opt.zero_grad(set_to_none=True)
    with torch.no_grad():
        s0_list = [s.to(torch.bfloat16) for s in A.capture_states(idsA)]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lo = B(x, s0_list=s0_list)
    l = F.cross_entropy(lo.float().reshape(-1, lo.shape[-1]), y.reshape(-1), ignore_index=0)
    if not torch.isfinite(l):
        opt.zero_grad(set_to_none=True); continue
    l.backward()
    gn = torch.nn.utils.clip_grad_norm_(B.parameters(), 1.0)
    if torch.isfinite(gn): opt.step()
    if st % 5000 == 0:
        log("  B %d/%d loss=%.4f (%.0fms)" % (st, STEPS, l.item(), (time.time()-t0)/(st+1)*1000))
log("  B relay-train done")
torch.save(B.state_dict(), os.path.join(HERE, "relay2_B.pth"))

# Phase B0: 无状态对照
log("== Phase B0: 无状态 ==")
B0 = M.DeltaLoopLM().to(DEV)
opt = torch.optim.AdamW(B0.parameters(), lr=4e-4, weight_decay=0.01, betas=(0.9,0.95), fused=True)
rng = random.Random(45)
B0.train()
t0 = time.time()
for st in range(STEPS):
    lr = 4e-4 * min(1.0, st / 2000) * (0.5 * (1 + math.cos(math.pi * st / STEPS)) + 0.1 * (1 - 0.5 * (1 + math.cos(math.pi * st / STEPS))))
    for g in opt.param_groups: g["lr"] = lr
    x, y = get_halfB(tr, BS, rng)
    opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lo = B0(x)
    l = F.cross_entropy(lo.float().reshape(-1, lo.shape[-1]), y.reshape(-1), ignore_index=0)
    if not torch.isfinite(l):
        opt.zero_grad(set_to_none=True); continue
    l.backward()
    gn = torch.nn.utils.clip_grad_norm_(B0.parameters(), 1.0)
    if torch.isfinite(gn): opt.step()
    if st % 5000 == 0:
        log("  B0 %d/%d loss=%.4f (%.0fms)" % (st, STEPS, l.item(), (time.time()-t0)/(st+1)*1000))
log("  B0 done")
torch.save(B0.state_dict(), os.path.join(HERE, "relay2_B0.pth"))

# Phase Full: 单模型 [0,256)
log("== Phase Full: [0,256) LM ==")
Full = M.DeltaLoopLM().to(DEV)
opt = torch.optim.AdamW(Full.parameters(), lr=4e-4, weight_decay=0.01, betas=(0.9,0.95), fused=True)
rng = random.Random(44)
Full.train()
t0 = time.time()
for st in range(STEPS):
    lr = 4e-4 * min(1.0, st / 2000) * (0.5 * (1 + math.cos(math.pi * st / STEPS)) + 0.1 * (1 - 0.5 * (1 + math.cos(math.pi * st / STEPS))))
    for g in opt.param_groups: g["lr"] = lr
    x, y = get_full(tr, BS, rng)
    opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lo = Full(x)
    l = F.cross_entropy(lo.float().reshape(-1, lo.shape[-1]), y.reshape(-1), ignore_index=0)
    if not torch.isfinite(l):
        opt.zero_grad(set_to_none=True); continue
    l.backward()
    gn = torch.nn.utils.clip_grad_norm_(Full.parameters(), 1.0)
    if torch.isfinite(gn): opt.step()
    if st % 5000 == 0:
        log("  Full %d/%d loss=%.4f (%.0fms)" % (st, STEPS, l.item(), (time.time()-t0)/(st+1)*1000))
log("  Full done")
torch.save(Full.state_dict(), os.path.join(HERE, "relay2_Full.pth"))

# ============ 评测 ============
log("== 评测 [129,256) NLL ==")
A.eval(); B.eval(); B0.eval(); Full.eval()

tot_relay, tot_b0, tot_full, tok = 0.0, 0.0, 0.0, 0
for i in range(0, min(len(va), 200), 32):
    chunk = va[i:i+32]
    idsA = torch.stack([s[:128] for s in chunk]).to(DEV)
    xs, ys = [], []
    for s in chunk:
        t = s[128:256]; xs.append(t[:-1]); ys.append(t[1:])
    L = min(len(x) for x in xs)
    x = torch.stack([x[:L] for x in xs]).to(DEV)
    y = torch.stack([y[:L] for y in ys]).to(DEV)
    with torch.no_grad():
        s0_list = [s.to(torch.bfloat16) for s in A.capture_states(idsA)]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lo_r = B(x, s0_list=s0_list)
            lo_b0 = B0(x)
        ids_full = torch.cat([idsA, x], dim=1)  # (B, 255)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lo_f = Full(ids_full[:, :255])
        # Full NLL: logits[:, 127:255] vs targets[:, 128:256]
        lr_ = F.cross_entropy(lo_r.float().reshape(-1, lo_r.shape[-1]), y.reshape(-1), reduction="sum")
        lb0 = F.cross_entropy(lo_b0.float().reshape(-1, lo_b0.shape[-1]), y.reshape(-1), reduction="sum")
        lf = F.cross_entropy(lo_f.float().reshape(-1, lo_f.shape[-1]), y.reshape(-1), reduction="sum")
        tot_relay += lr_.item(); tot_b0 += lb0.item(); tot_full += lf.item()
        tok += y.numel()

res = dict(
    relay_B_with_state=tot_relay / tok,
    B0_no_state=tot_b0 / tok,
    Full_ref=tot_full / tok,
    state_gain=tot_b0 / tok - tot_relay / tok,
    residual=tot_relay / tok - tot_full / tok,
)
log(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
json.dump(res, open(os.path.join(HERE, "relay2_results.json"), "w"), indent=2)
log("== 完成 ==")
