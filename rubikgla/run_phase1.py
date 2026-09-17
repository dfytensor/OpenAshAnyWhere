#!/usr/bin/env python3
"""Rubik-GLA 阶段一核心验证: 4 任务 x 4 模型, 同预算 (2000 步) 同种子.
可断点续跑 (结果逐项落盘 rubikgla_results.json).
"""
import sys, os, time, json
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import LM
import tasks

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "rubikgla_results.json")
DEV = "cuda"
STEPS = 2000
BS = 32
EVAL_N = 256
LR = 3e-4
KINDS = ["rubik", "gla", "gru", "tf"]

TASKS = {
    "bracket": dict(n=64),
    "copy": dict(n=48, reverse=False),
    "reverse": dict(n=48, reverse=True),
    "fsm": dict(n=64),
}


def make_batch(task, batch, seed):
    if task == "bracket":
        return tasks.gen_bracket(batch, n=64, seed=seed)
    if task == "copy":
        return tasks.gen_copy(batch, n=48, seed=seed, reverse=False)
    if task == "reverse":
        return tasks.gen_copy(batch, n=48, seed=seed, reverse=True)
    if task == "fsm":
        return tasks.gen_fsm(batch, n=64, fsm=make_batch.fsm, seed=seed)
    raise KeyError(task)


def evaluate(m, task, V, seed=99_000):
    m.eval()
    correct = total = 0
    with torch.no_grad():
        for i in range(4):
            d, _ = make_batch(task, EVAL_N // 4, seed=seed + i)
            x = d["input"].to(DEV); y = d["target"].to(DEV); mk = d["mask"].to(DEV)
            logits = m(x)
            pred = logits.argmax(-1)
            hit = ((pred == y) & (mk == 1)).sum().item()
            correct += hit; total += mk.sum().item()
    m.train()
    return correct / max(total, 1)


def run(task, kind, seed=0):
    torch.manual_seed(seed)
    d0, V = make_batch(task, BS, seed=1)
    m = LM(V, d=128, kind=kind, layers=4, H=4).to(DEV)
    n_par = sum(p.numel() for p in m.parameters())
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=0.01)
    best = 0.0
    t0 = time.time()
    for st in range(STEPS):
        d, _ = make_batch(task, BS, seed=10_000 + st)
        x = d["input"].to(DEV); y = d["target"].to(DEV); mk = d["mask"].to(DEV)
        logits = m(x)
        loss = (F.cross_entropy(logits.reshape(-1, V), y.reshape(-1), reduction="none")
                .reshape(y.shape) * mk).sum() / mk.sum().clamp(min=1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if (st + 1) % 250 == 0:
            acc = evaluate(m, task, V)
            best = max(best, acc)
            print("  %-8s %-6s %4d loss=%.4f acc=%.4f best=%.4f (%.0fs)" %
                  (task, kind, st + 1, loss.item(), acc, best, time.time() - t0), flush=True)
    return dict(best_acc=round(best, 4), params=n_par, wall_s=round(time.time() - t0))


def main():
    torch.manual_seed(0)
    import tasks as T
    make_batch.fsm = T.make_fsm(42)
    data = {}
    if os.path.exists(RES):
        with open(RES, encoding="utf-8") as f:
            data = json.load(f)
    for task in TASKS:
        for kind in KINDS:
            key = f"{task}|{kind}"
            if key in data:
                continue
            print("=== %s ===" % key, flush=True)
            try:
                data[key] = run(task, kind)
            except Exception as e:
                data[key] = dict(error="%s: %s" % (type(e).__name__, e))
                import traceback; traceback.print_exc()
            with open(RES, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=1)

    print("\n================ 阶段一结果 (best acc) ================")
    hdr = "%-10s" + "%9s" * len(KINDS)
    print(hdr % ("task", *KINDS))
    for task in TASKS:
        row = []
        for kind in KINDS:
            r = data.get(f"{task}|{kind}", {})
            row.append(r.get("best_acc", "ERR"))
        print(("%-10s" + "%9s" * len(KINDS)) % (task, *row))
    print("params:", {k: data.get(f"copy|{k}", {}).get("params") for k in KINDS})


if __name__ == "__main__":
    main()
