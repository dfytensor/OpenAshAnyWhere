#!/usr/bin/env python3
"""fsm n=256 训练 A/B: rubik-loop vs rubik-scan, 同种子同数据, 比精度+墙钟."""
import sys, os, time, json
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import RubikLayer
from lowrank import RubikScanLayer
import tasks

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "scan_ab_results.json")
DEV = "cuda"
STEPS, BS, LR, N = 1000, 64, 3e-4, 256
FSM = tasks.make_fsm(42)


def make(kind):
    class LM2(nn.Module):
        def __init__(self):
            super().__init__()
            cls = RubikScanLayer if kind == "scan" else RubikLayer
            self.emb = nn.Embedding(11, 128)
            self.stack = nn.ModuleList([cls(128, 4, decay=True) for _ in range(4)])
            self.ln = nn.LayerNorm(128)
            self.head = nn.Linear(128, 11)
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)
        def forward(self, ids):
            x = self.emb(ids)
            for layer in self.stack:
                x, _ = layer(x)
            return self.head(self.ln(x))
    return LM2().to(DEV)


def evaluate(m, seed, n):
    m.eval()
    c = t = 0
    with torch.no_grad():
        for i in range(2):
            d, _ = tasks.gen_fsm(128, n=n, fsm=FSM, seed=seed + i)
            x = d["input"].to(DEV); y = d["target"].to(DEV); mk = d["mask"].to(DEV)
            pred = m(x).argmax(-1)
            c += ((pred == y) & (mk == 1)).sum().item(); t += mk.sum().item()
    m.train()
    return c / max(t, 1)


def run(kind, steps=STEPS):
    torch.manual_seed(0)
    m = make(kind)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=0.01)
    t0 = time.time()
    for st in range(steps):
        d, _ = tasks.gen_fsm(BS, n=N, fsm=FSM, seed=30_000 + st)
        x = d["input"].to(DEV); y = d["target"].to(DEV); mk = d["mask"].to(DEV)
        logits = m(x)
        loss = (F.cross_entropy(logits.reshape(-1, 11), y.reshape(-1), reduction="none")
                .reshape(y.shape) * mk).sum() / mk.sum().clamp(min=1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if (st + 1) % 250 == 0:
            print("  %-5s %4d loss=%.4f acc=%.4f (%.0fs, %.0fms/step)" %
                  (kind, st + 1, loss.item(), evaluate(m, 99000, N),
                   time.time() - t0, (time.time() - t0) / (st + 1) * 1000), flush=True)
    return dict(final=round(evaluate(m, 99000, N), 4),
                wall_s=round(time.time() - t0),
                ms_per_step=round((time.time() - t0) / steps * 1000, 1))


def main():
    data = {}
    if os.path.exists(RES):
        with open(RES, encoding="utf-8") as f:
            data = json.load(f)
    for kind in ("scan", "loop"):
        if kind in data:
            continue
        print("===", kind, "fsm n=256 ===", flush=True)
        try:
            data[kind] = run(kind)
        except Exception as e:
            import traceback; traceback.print_exc()
            data[kind] = dict(error=str(e))
        with open(RES, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1)
    print("\n汇总:", json.dumps(data, indent=1))


if __name__ == "__main__":
    main()
