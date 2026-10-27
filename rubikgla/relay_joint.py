#!/usr/bin/env python3
"""简化联合接力: A 编码 [0,128) → 隐状态投影 → B 解码 [128,256). 端到端梯度."""
import sys, os, random, time, math, json
import torch
import torch.nn as nn
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

A = M.DeltaLoopLM().to(DEV)
B = M.DeltaLoopLM().to(DEV)
# 接口投影: A 的 [0,128) 末隐状态 → B 的 5 站 S0
# 简化: 学一个线性映射 hidden(320) → 每站 S0 (5×10×32×32 = 51200)
adapter = nn.Sequential(nn.Linear(320, 320), nn.GELU(), nn.Linear(320, 5 * 10 * 32 * 32 // 8)).to(DEV)  # 低秩分解
# 实际用: Linear(320, 6400) → reshape(5,10,32,32) 的低秩近似
adapter = nn.Sequential(nn.Linear(320, 320), nn.GELU(), nn.Linear(320, 320)).to(DEV)  # 简化: 同维映射

params = list(A.parameters()) + list(B.parameters()) + list(adapter.parameters())
opt = torch.optim.AdamW(params, lr=4e-4, weight_decay=0.01, betas=(0.9, 0.95), fused=True)
rng = random.Random(42)
A.train(); B.train(); adapter.train()

logf = open(os.path.join(HERE, "relay_joint.log"), "a", encoding="utf-8")
def log(msg):
    print(msg, flush=True)
    logf.write(msg + "\n"); logf.flush()

log("== 联合接力训练 (隐状态投影版) ==")
log("params: A=%.2fM B=%.2fM adapter=%.3fM" % (
    sum(p.numel() for p in A.parameters()) / 1e6,
    sum(p.numel() for p in B.parameters()) / 1e6,
    sum(p.numel() for p in adapter.parameters()) / 1e6))

t0 = time.time()
nan_run = 0
for st in range(STEPS):
    lr = 4e-4 * min(1.0, st / 2000)
    for g in opt.param_groups: g["lr"] = lr
    idsA, xB, yB = [], [], []
    for _ in range(BS):
        s = tr[rng.randrange(len(tr))]
        idsA.append(s[:128])
        t = s[128:256]
        xB.append(t[:-1]); yB.append(t[1:])
    idsA = torch.stack(idsA).to(DEV)
    xB = torch.stack(xB).to(DEV)
    yB = torch.stack(yB).to(DEV)
    opt.zero_grad(set_to_none=True)
    # A 前向: 取 [0,128) 的最终隐状态 (pre+A_loop+bridge+loop_c+post 之后)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h_A = A.emb(idsA)
        h_A = A.pre(h_A)
        h_A = A.loop_a(h_A)
        h_A = A.bridge(h_A)
        h_A = A.loop_c(h_A)
        h_A = A.post(h_A)
        h_last = h_A[:, -1, :]                                      # (B,320) 末 token 隐状态
        # 投影到 B 的状态空间: 简化为 B 的 emb 空间偏移
        h_proj = adapter(h_last.float()).to(torch.bfloat16)         # (B,320)
        # B 前向: 每站 S0 = h_proj 广播到 (B,G,DH,DH) 的外积近似
        # 简化: 直接加到 B 的嵌入上 (状态→初始上下文)
        h_B = B.emb(xB) + 0.1 * h_proj.unsqueeze(1)                       # (B,127,320) 广播
        # 通过 B 的站
        h_B = B.pre(h_B)
        h_B = B.loop_a(h_B)
        h_B = B.bridge(h_B)
        h_B = B.loop_c(h_B)
        h_B = B.post(h_B)
        logits = B.head(B.ln_f(h_B))
    loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), yB.reshape(-1), ignore_index=0)
    if not torch.isfinite(loss):
        nan_run += 1
        if nan_run > 200: raise RuntimeError("NaN runaway @%d" % st)
        opt.zero_grad(set_to_none=True); continue
    nan_run = 0
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
    if torch.isfinite(gn): opt.step()
    if st % 2000 == 0:
        log("%d/%d loss=%.4f gn=%.2f lr=%.2e (%.0fms)" % (st, STEPS, loss.item(), gn.item(), lr, (time.time()-t0)/(st+1)*1000))

# 评测
log("== 评测 ==")
A.eval(); B.eval(); adapter.eval()

@torch.no_grad()
def eval_relay(val, bs=16):
    tot, tok = 0.0, 0
    for i in range(0, min(len(val), 500), bs):
        chunk = val[i:i+bs]
        idsA = torch.stack([s[:128] for s in chunk]).to(DEV)
        xs, ys = [], []
        for s in chunk:
            t = s[128:256]; xs.append(t[:-1]); ys.append(t[1:])
        L = min(len(x) for x in xs)
        x = torch.stack([x[:L] for x in xs]).to(DEV)
        y = torch.stack([y[:L] for y in ys]).to(DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h_A = A.emb(idsA); h_A = A.pre(h_A); h_A = A.loop_a(h_A)
            h_A = A.bridge(h_A); h_A = A.loop_c(h_A); h_A = A.post(h_A)
            h_proj = adapter(h_A[:, -1, :].float()).to(torch.bfloat16)
            h_B = B.emb(x) + 0.1 * h_proj.unsqueeze(1)
            h_B = B.pre(h_B); h_B = B.loop_a(h_B); h_B = B.bridge(h_B)
            h_B = B.loop_c(h_B); h_B = B.post(h_B)
            logits = B.head(B.ln_f(h_B))
        l = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="sum")
        tot += l.item(); tok += y.numel()
    return tot / tok

n_relay = eval_relay(va)
log("接力 NLL [129,256) = %.4f" % n_relay)
json.dump({"relay_joint_NLL": n_relay}, open(os.path.join(HERE, "relay_joint_results.json"), "w"), indent=2)
log("== 完成 ==")
logf.close()
