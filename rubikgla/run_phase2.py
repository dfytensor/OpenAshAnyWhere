#!/usr/bin/env python3
"""阶段二: B=128 吃满显存 + 3 种子稳健性 (fsm 全模型 / reverse 双模型) + 长度外推.
- fsm: seeds {1,2} × {rubik,gla,gru,tf}, 训练 n=64, 评测 n=64/128/256
- reverse: seeds {1,2} × {rubik,gla}, 训练 n=48, 评测 n=48/96/192
增量落盘, 可断点续跑.
"""
import sys, os, time, json
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import LM
import tasks

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "phase2_results.json")
DEV = "cuda"
STEPS, BS, LR = 2000, 128, 3e-4
FSM = None


def batch(task, batch, seed, n):
    if task == "fsm":
        return tasks.gen_fsm(batch, n=n, fsm=FSM, seed=seed)
    return tasks.gen_copy(batch, n=n, seed=seed, reverse=True)


def evaluate(m, task, V, seed, n):
    m.eval()
    correct = total = 0
    with torch.no_grad():
        for i in range(2):
            d, _ = batch(task, 128, seed=seed + i, n=n)
            x = d["input"].to(DEV); y = d["target"].to(DEV); mk = d["mask"].to(DEV)
            pred = m(x).argmax(-1)
            correct += ((pred == y) & (mk == 1)).sum().item()
            total += mk.sum().item()
    m.train()
    return correct / max(total, 1)


def run(task, kind, seed, n_train, eval_lens):
    torch.manual_seed(seed)
    d0, V = batch(task, BS, seed=1, n=n_train)
    m = LM(V, d=128, kind=kind, layers=4, H=4).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=0.01)
    best = 0.0
    t0 = time.time()
    for st in range(STEPS):
        d, _ = batch(task, BS, seed=10_000 + st, n=n_train)
        x = d["input"].to(DEV); y = d["target"].to(DEV); mk = d["mask"].to(DEV)
        logits = m(x)
        loss = (F.cross_entropy(logits.reshape(-1, V), y.reshape(-1), reduction="none")
                .reshape(y.shape) * mk).sum() / mk.sum().clamp(min=1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if (st + 1) % 250 == 0:
            a = evaluate(m, task, V, seed=99000, n=n_train)
            best = max(best, a)
            mem = torch.cuda.max_memory_allocated() / 2**30
            print("  %-6s %-6s s%d %4d loss=%.4f acc=%.4f best=%.4f mem=%.1fGB (%.0fs)" %
                  (task, kind, seed, st + 1, loss.item(), a, best, mem, time.time() - t0), flush=True)
    res = {f"acc@{n}": round(evaluate(m, task, V, seed=99000 + 7 * j, n=n), 4)
           for j, n in enumerate(eval_lens)}
    res.update(best_train=round(best, 4), wall_s=round(time.time() - t0),
               mem_gb=round(torch.cuda.max_memory_allocated() / 2**30, 1))
    del m, opt
    torch.cuda.empty_cache()
    return res


def main():
    torch.manual_seed(0)
    import tasks as T
    global FSM
    FSM = T.make_fsm(42)
    data = {}
    if os.path.exists(RES):
        with open(RES, encoding="utf-8") as f:
            data = json.load(f)

    grid = []
    for seed in (1, 2):
        for kind in ("rubik", "gla", "gru", "tf"):
            grid.append(("fsm", kind, seed, 64, [64, 128, 256]))
        for kind in ("rubik", "gla"):
            grid.append(("reverse", kind, seed, 48, [48, 96, 192]))

    for task, kind, seed, n_train, eval_lens in grid:
        key = f"{task}|{kind}|s{seed}"
        if key in data:
            continue
        print("===", key, "===", flush=True)
        try:
            data[key] = run(task, kind, seed, n_train, eval_lens)
        except Exception as e:
            import traceback; traceback.print_exc()
            data[key] = dict(error="%s: %s" % (type(e).__name__, e))
        with open(RES, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1)

    print("\n================ 阶段二: 原生/2x/4x 长度 ================")
    print("%-22s %8s %8s %8s" % ("run", "@n", "@2n", "@4n"))
    for task, kind, seed, n_train, eval_lens in grid:
        key = f"{task}|{kind}|s{seed}"
        r = data.get(key, {})
        if "acc@" in str(r):
            vals = [r.get(f"acc@{n}", "-") for n in eval_lens]
            print("%-22s %8s %8s %8s" % (key, *vals))
    print("done", flush=True)


if __name__ == "__main__":
    main()
