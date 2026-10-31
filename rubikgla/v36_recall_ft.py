#!/usr/bin/env python3
"""核心兑现实验 (最小可靠版): 真实 23M v36-MS (v36ms_e464.pth 热启动) × 检索密集 finetune.
数据: 16k = 8 条真实文本拼接; 前 8192 注入 8 对 (稀有key→稀有value) 事实;
      后 8192 注入 8 个查询 (key 在 q 位, ids[q+1]=value → 标准 next-token CE 自然覆盖).
损失: 标准 next-token CE (全位置), 检索信号由数据注入.
判据: ① 查询位召回 acc/NLL (前后对比) ② sel_coef 幅度 ③ 16k 后缀 NLL (质量守卫).
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
STEPS = 500
LR_TRUNK = 5e-5
LR_MS = 5e-4
VOCAB = 23005
KEY0, VAL0 = 22000, 22100                    # 稀有 token 池 (64 key + 64 value)
N_FACT, N_Q = 8, 8
HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "v36ms_e464.pth")


def make_sample(val, rng):
    """16k 检索样本 → (ids (1,L), facts [(k,v,q)], queries [(k,v,q)])."""
    parts, tot = [], 0
    while tot < LONG:
        s = val[rng.randrange(len(val))]
        parts.append(s)
        tot += len(s)
    ids = torch.cat(parts)[:LONG]
    g = torch.Generator().manual_seed(rng.randint(0, 2**31 - 1))
    keys = (KEY0 + torch.randperm(64, generator=g)).tolist()
    vals = (VAL0 + torch.randperm(64, generator=g)).tolist()
    facts, queries = [], []
    pp = sorted(random.sample(range(4, 8180), N_FACT))
    for i in range(N_FACT):
        p = pp[i]
        ids[p] = keys[i]
        ids[p + 1] = vals[i]
        facts.append((keys[i], vals[i], p))
    qq = sorted(random.sample(range(8192, LONG - 4), N_Q))
    for i in range(N_Q):
        q = qq[i]
        ids[q] = facts[i % N_FACT][0]
        queries.append((facts[i % N_FACT][0], facts[i % N_FACT][1], q))
    return ids.unsqueeze(0).to(DEV), facts, queries


def recall_eval(model, samples):
    """查询位: logits[q] 应预测 v. 返回 (acc, nll)."""
    was = model.training
    model.eval()
    hits, nll_sum, n = 0, 0.0, 0
    with torch.no_grad():
        for ids, facts, queries in samples:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lo = model(ids, chunk=CHUNK, ramp=1.0)
            for (k, v, q) in queries:
                lo_q = lo[0, q].float()
                nll_sum += F.cross_entropy(lo_q.unsqueeze(0),
                                           torch.tensor([v], device=DEV)).item()
                hits += int(lo_q.argmax().item() == v)
                n += 1
    if was:
        model.train()
    return hits / max(n, 1), nll_sum / max(n, 1)


def main():
    torch.manual_seed(0)
    rng_data = random.Random(55)
    seqs = torch.load(PT_CACHE, map_location="cpu", weights_only=True)
    val = [s for s in seqs[int(0.95 * len(seqs)):][:2000] if len(s) >= 256]
    log("val 文本 %d 条" % len(val))

    net = build_ms_from_v31(enable_gate=True, enable_sel=True)
    model = DeltaLoopLM_MS(net).to(DEV)
    model.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=True), strict=False)
    sel_attns = (net.pre.attn, net.loop_a.attn.attn, net.bridge.attn,
                 net.loop_c.attn.attn, net.post.attn)
    with torch.no_grad():                               # 选择性双侧非零 + 增益 0.5
        for attn in sel_attns:
            attn.sel_scale.fill_(0.5)
            attn.sel_A.weight.normal_(0, 0.01)
            attn.sel_B.weight.normal_(0, 0.01)
    model.train()
    log("recall_ft: sel 双侧非零 + sel_scale 0.5 (从 v36ms_e464.pth 热启动)")

    eval_samples = [make_sample(val, random.Random(500 + i)) for i in range(3)]
    acc0, nll0 = recall_eval(model, eval_samples)
    log("== finetune 前 == 查询召回 acc %.4f | 查询 NLL %.4f" % (acc0, nll0))

    ms_params = list(model.ms.parameters())
    sel_params = []
    for attn in sel_attns:
        sel_params += [attn.sel_scale, attn.sel_A.weight, attn.sel_B.weight]
    sel_params = list(set(sel_params))
    trunk_params = [p for p in model.parameters()
                    if all(p is not sp for sp in sel_params) and all(p is not mp for mp in ms_params)]
    opt = torch.optim.AdamW([
        {"params": trunk_params, "lr": LR_TRUNK},
        {"params": ms_params + sel_params, "lr": LR_MS},
    ], weight_decay=0.01, betas=(0.9, 0.95), fused=True)
    log("参数组: trunk %.1fM @ %.0e | ms+sel %.2fM @ %.0e"
        % (sum(p.numel() for p in trunk_params) / 1e6, LR_TRUNK,
           (sum(p.numel() for p in ms_params) + sum(p.numel() for p in sel_params)) / 1e6, LR_MS))

    rng = random.Random(55)
    t0 = time.time()
    for stp in range(STEPS):
        ids, facts, queries = make_sample(val, rng)
        opt.zero_grad(set_to_none=True)
        ramp = min(1.0, (stp + 1) / 150)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lo = model(ids, chunk=CHUNK, ramp=ramp)
        loss = F.cross_entropy(lo.float()[:, :-1].reshape(-1, VOCAB),
                               ids[:, 1:].reshape(-1), ignore_index=0)
        if not torch.isfinite(loss):
            opt.zero_grad(set_to_none=True)
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if stp % 50 == 0 or stp == STEPS - 1:
            acc, qnll = recall_eval(model, eval_samples)
            selc = model.ms[0].sel_coef.detach().abs().mean().item()
            log("step %d/%d loss %.4f | 召回acc %.4f 查询NLL %.4f | sel_coef %.4f (%.0fs)"
                % (stp, STEPS, loss.item(), acc, qnll, selc, time.time() - t0), flush=True)

    torch.save(model.state_dict(), os.path.join(HERE, "v36ms_recall_ft.pth"))
    acc1, nll1 = recall_eval(model, eval_samples)
    with torch.no_grad():
        ids_e, _, _ = eval_samples[0]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lo_full = model(ids_e, chunk=CHUNK, ramp=1.0)
    suf = F.cross_entropy(lo_full[0, 8192:LONG - 1].float().reshape(-1, VOCAB),
                          ids_e[0, 8193:LONG].reshape(-1), ignore_index=0).item()
    log("=== 核心兑现终评 ===")
    log("查询召回: %.4f → %.4f (%+.4f) | 查询 NLL: %.4f → %.4f" % (acc0, acc1, acc1 - acc0, nll0, nll1))
    log("16k 后缀 NLL (LM 质量守卫): %.4f" % suf)
    log("RECALL-FT DONE", flush=True)


if __name__ == "__main__":
    main()
