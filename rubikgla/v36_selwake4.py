#!/usr/bin/env python3
"""selwake-v4: aux 选择性监督 — 直接给 sel 一个 O(1) 梯度信号 (打破零初始化死锁).
数据: 同 selwake (16k, 前 8k 埋 64 对 key→value, 后 8k 查询).
损失: 主 CE (查询位) + 0.1 × aux (pair 位 sel→1: 高 sel = 慢衰减 = 保持绑定).
诊断: 首步后打印 sel_scale/sel_A/sel_B 的梯度范数 (vs emb — 死锁的直接实测).
判据: ① 召回 acc/NLL 显著优于 v3 (4.59) ② sel_scale 生长/保持 0.5 ③ sel 分布偏移.
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
AUX_W = 0.1
N_PAIR, N_Q = 16, 16
N_KEY = 64
HERE = os.path.dirname(os.path.abspath(__file__))
RESUME = os.path.join(HERE, "v36ms_e464.pth")
FILL0, KEY0, VAL0 = 100, 20001, 20065


def make_sample(rng):
    """16k 检索样本 → (ids, tgt, qmask, pmask). tgt>0 计损失; qmask 查询位; pmask pair 位."""
    g = torch.Generator().manual_seed(rng.randint(0, 2**31 - 1))
    ids = torch.randint(FILL0, FILL0 + 19900, (1, LONG))
    tgt = torch.zeros(1, LONG, dtype=torch.long)
    qmask = torch.zeros(1, LONG)
    pmask = torch.zeros(1, LONG)
    keys = KEY0 + torch.randperm(N_KEY, generator=g)
    vals = VAL0 + torch.randperm(N_KEY, generator=g)
    p = 1 + torch.randperm(8190, generator=g)[:2 * N_PAIR]
    for i in range(N_PAIR):
        p1, p2 = sorted([int(p[2 * i]), int(p[2 * i + 1])])
        ids[0, p1] = int(keys[i])
        ids[0, p2] = int(vals[i])
        tgt[0, p1] = int(vals[i])                       # key→value 绑定
        pmask[0, p1] = 1.0                              # pair 位置 (选择性监督点)
    q = 8192 + 1 + torch.randperm(LONG - 8194, generator=g)[:N_Q]
    for i in range(N_Q):
        t = int(q[i])
        ids[0, t] = int(keys[i % N_KEY])
        tgt[0, t] = int(vals[i % N_KEY])
        qmask[0, t] = 1.0
    return ids.to(DEV), tgt.to(DEV), qmask.to(DEV), pmask.to(DEV)


class SelHook:
    """捕获 5 站 sel_B 输出 (pre-sigmoid logit → sigmoid = sel)."""

    def __init__(self, net):
        self.acts = []
        self.handles = []
        for blk in (net.pre.attn, net.loop_a.attn.attn, net.bridge.attn,
                    net.loop_c.attn.attn, net.post.attn):
            self.handles.append(blk.sel_B.register_forward_hook(
                lambda mod, inp, out: self.acts.append(out)))

    def clear(self):
        self.acts = []

    def remove(self):
        for h in self.handles:
            h.remove()


def main():
    torch.manual_seed(0)
    rng_data = random.Random(41)

    net = build_ms_from_v31(enable_gate=True, enable_sel=True)
    model = DeltaLoopLM_MS(net).to(DEV)
    model.load_state_dict(torch.load(RESUME, map_location="cpu", weights_only=True), strict=False)
    with torch.no_grad():                               # 激活选择性: 双侧非零
        for blk in (net.pre.attn, net.loop_a.attn.attn, net.bridge.attn,
                    net.loop_c.attn.attn, net.post.attn):
            blk.sel_scale.fill_(0.5)
            blk.sel_A.weight.normal_(0, 0.01)
            blk.sel_B.weight.normal_(0, 0.01)
    model.train()
    log("selwake-v4: aux 监督 + 双侧非零 + sel_scale 0.5")
    # 禁用 compile (hook + checkpoint + dynamo 组合会反复重编译/元数据校验崩)
    for blk in (model.net.pre, model.net.loop_a.attn, model.net.bridge,
                model.net.loop_c.attn, model.net.post):
        blk._attn_impl = blk.attn._apply_raw
        blk._ffn_impl = blk.ffn.forward

    hook = SelHook(net)

    evals = [make_sample(random.Random(1000 + i)) for i in range(4)]

    def recall_eval():
        model.eval()
        accs, nlls, n = [], [], 0
        hook.clear()
        with torch.no_grad():
            for ids, tgt, qm, pm in evals:
                hook.clear()
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
    log("初始 召回acc %.3f | 查询NLL %.4f" % (acc0, qnll0))

    # ---- 梯度范数诊断 (首步) ----
    ids, tgt, qm, pm = make_sample(random.Random(1))
    hook.clear()
    opt0 = torch.optim.AdamW(model.parameters(), lr=LR)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lo = model(ids, chunk=CHUNK, ramp=1.0)
    qm2 = qm.bool()
    loss = F.cross_entropy(lo[qm2].float(), tgt[qm2])
    loss.backward()
    g_sel_scale = sum(blk.sel_scale.grad.abs().mean().item()
                      for blk in (net.pre.attn, net.loop_a.attn.attn, net.bridge.attn,
                                  net.loop_c.attn.attn, net.post.attn)) / 5
    g_selA = sum(blk.sel_A.weight.grad.abs().mean().item()
                 for blk in (net.pre.attn, net.loop_a.attn.attn, net.bridge.attn,
                             net.loop_c.attn.attn, net.post.attn)) / 5
    g_emb = net.emb.weight.grad.abs().mean().item()
    log("[梯度诊断] sel_scale %.3e | sel_A %.3e | emb %.3e — 死锁判定: %s"
        % (g_sel_scale, g_selA, g_emb,
           "活 (非零)" if g_sel_scale > 0 else "死 (精确零)"), flush=True)
    del opt0

    # ---- 训练 ----
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=LR, weight_decay=0.01, betas=(0.9, 0.95), fused=True)
    rng = random.Random(55)
    t0 = time.time()
    for stp in range(STEPS):
        ids, tgt, qm, pm = make_sample(rng)
        opt.zero_grad(set_to_none=True)
        hook.clear()
        ramp = min(1.0, (stp + 1) / 150)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lo = model(ids, chunk=CHUNK, ramp=ramp)
        qm2 = qm.bool()
        loss = F.cross_entropy(lo[qm2].float(), tgt[qm2])
        # aux: pair 位置 sel → 1 (慢衰减保持绑定)
        if hook.acts:
            mpos = pm[0].bool()
            aux_terms = []
            for act in hook.acts:
                if act.shape[1] != LONG:                 # capture 分段的 2048 长条目跳过
                    continue
                s = torch.sigmoid(act).mean(dim=-1)[0]   # (L,) 通道均值
                if mpos.sum() > 0:
                    aux_terms.append(((s[mpos] - 1.0) ** 2).mean())
            if aux_terms:
                loss = loss + AUX_W * sum(aux_terms) / len(aux_terms)
        if not torch.isfinite(loss) or not loss.requires_grad:
            opt.zero_grad(set_to_none=True)
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        hook.clear()
        if stp % 50 == 0 or stp == STEPS - 1:
            model.eval()
            acc, qnll = recall_eval()
            model.train()
            sc = [blk.sel_scale.detach().abs().mean().item()
                  for blk in (net.pre.attn, net.loop_a.attn.attn, net.bridge.attn,
                              net.loop_c.attn.attn, net.post.attn)]
            log("step %d/%d loss %.4f 召回acc %.3f 查询NLL %.4f | sel_scale %s (%.0fs)"
                % (stp, STEPS, loss.item(), acc, qnll,
                   "/".join("%.2f" % s for s in sc), time.time() - t0), flush=True)

    model.eval()
    acc, qnll = recall_eval()
    log("=== selwake-v4 终评 === 召回acc %.3f | 查询NLL %.4f (v3 对照: 0.031/4.59)" % (acc, qnll))
    torch.save(model.state_dict(), os.path.join(HERE, "v36ms_selwake4.pth"))
    log("checkpoint: v36ms_selwake4.pth")
    log("训后行为: python v37_behavior.py --ckpt v36ms_selwake4.pth")
    log("SELWAKE-V4 DONE", flush=True)


if __name__ == "__main__":
    main()
