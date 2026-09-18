#!/usr/bin/env python3
"""H4 低秩消融: rubik-fullrank vs rubik-lowrank(r=2) vs gla.
任务: fsm / reverse / bracket; B=128, 2000 步, seed 1. 测 best acc + wall_s."""
import sys, os, time, json
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import LM, RubikLayer, GLALayer
from lowrank import RubikLowRankLayer, RubikLowRankFast
import tasks

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "lowrank_results.json")
DEV = "cuda"
STEPS, BS, LR = 2000, 128, 3e-4
FSM = tasks.make_fsm(42)


def batch(task, b, seed, n):
    if task == "fsm":
        return tasks.gen_fsm(b, n=n, fsm=FSM, seed=seed)
    if task == "reverse":
        return tasks.gen_copy(b, n=n, seed=seed, reverse=True)
    return tasks.gen_bracket(b, n=n, seed=seed)


def make(task, kind):
    V = {"fsm": 11, "reverse": 12, "bracket": 8}[task]
    m = LM(V, d=128, kind="gla", layers=4, H=4).to(DEV) if kind == "gla" else None
    if m is None:
        if kind == "rubik":
            cls = lambda d, H: RubikLayer(d, H, decay=True)
        elif kind == "rubiklrf":
            cls = lambda d, H: RubikLowRankFast(d, H, r=2, decay=True)
        else:
            cls = lambda d, H: RubikLowRankLayer(d, H, r=2, decay=True)
        class LM2(nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = nn.Embedding(V, 128)
                self.stack = nn.ModuleList([cls(128, 4) for _ in range(4)])
                self.ln = nn.LayerNorm(128)
                self.head = nn.Linear(128, V)
                nn.init.zeros_(self.head.weight)
                nn.init.zeros_(self.head.bias)
            def forward(self, ids):
                x = self.emb(ids)
                for layer in self.stack:
                    x, _ = layer(x)
                return self.head(self.ln(x))
        m = LM2().to(DEV)
    return m


def evaluate(m, task, seed, n):
    m.eval()
    c = t = 0
    with torch.no_grad():
        for i in range(2):
            d, _ = batch(task, 128, seed + i, n)
            x = d["input"].to(DEV); y = d["target"].to(DEV); mk = d["mask"].to(DEV)
            pred = m(x).argmax(-1)
            c += ((pred == y) & (mk == 1)).sum().item(); t += mk.sum().item()
    m.train()
    return c / max(t, 1)


def run(task, kind):
    torch.manual_seed(1)
    m = make(task, kind)
    n_par = sum(p.numel() for p in m.parameters())
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=0.01)
    n = {"fsm": 64, "reverse": 48, "bracket": 64}[task]
    best = 0.0
    t0 = time.time()
    for st in range(STEPS):
        d, _ = batch(task, BS, seed=10_000 + st, n=n)
        x = d["input"].to(DEV); y = d["target"].to(DEV); mk = d["mask"].to(DEV)
        logits = m(x)
        loss = (F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1),
                                reduction="none").reshape(y.shape) * mk).sum() / mk.sum().clamp(min=1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if (st + 1) % 250 == 0:
            best = max(best, evaluate(m, task, seed=99000, n=n))
    wall = time.time() - t0
    acc = evaluate(m, task, seed=99000, n=n)
    return dict(best=round(best, 4), final=round(acc, 4), params=n_par,
                wall_s=round(wall), ms_per_step=round(wall / STEPS * 1000, 1))


def main():
    data = {}
    if os.path.exists(RES):
        with open(RES, encoding="utf-8") as f:
            data = json.load(f)
    for task in ("fsm", "reverse", "bracket"):
        for kind in ("rubik", "rubiklr", "rubiklrf", "gla"):
            key = f"{task}|{kind}"
            if key in data:
                continue
            print("===", key, "===", flush=True)
            try:
                data[key] = run(task, kind)
            except Exception as e:
                import traceback; traceback.print_exc()
                data[key] = dict(error="%s: %s" % (type(e).__name__, e))
            with open(RES, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=1)
    print("\n%-14s %8s %8s %10s %8s" % ("run", "best", "final", "wall_s", "ms/step"))
    for k, r in data.items():
        if "error" in r:
            print("%-14s %s" % (k, r["error"]))
        else:
            print("%-14s %8.4f %8.4f %10d %8.1f" %
                  (k, r["best"], r["final"], r["wall_s"], r["ms_per_step"]))


if __name__ == "__main__":
    main()
