"""XGRU vs GRU(1L/2L) vs LSTM vs MetaGRU: CUDA Graph 鍔犻€熺増, 缁熶竴蹇崗璁?

鍗忚: bs=512, 姝ユ暟 = 鎱㈠崗璁?4 (鎬绘牱鏈噺鐩稿悓), 鍏ㄦā鍨嬪悓鍗忚, 2 绉嶅瓙.
"""
import sys, os, json, time
sys.path.insert(0, r"F:\OpenASH2605\metaru")
import torch
import torch.nn as nn
import torch.nn.functional as F
from metaru import SeqModel, LMModel
from hybrid import MetaGRUCell
from xgru import XGRUCell
from tasks import adding_batch, CharLM
from hybrid_compare import chaos_batch

DEV = "cuda"
B = 512
RESULTS = {}
RJ = r"F:\OpenASH2605\metaru\xgru_results_fast.json"
if os.path.exists(RJ):
    RESULTS = json.load(open(RJ))


class XModel(nn.Module):
    def __init__(self, kind, m, d, p, last=True):
        super().__init__()
        self.kind, self.d, self.last = kind, d, last
        if kind == "xgru":
            self.cell = XGRUCell(m, d)
        elif kind == "metagru":
            self.cell = MetaGRUCell(m, d)
        else:
            self.core = SeqModel(kind, m, d, p, last)
        self.head = nn.Linear(d, p)

    def forward(self, x):
        if self.kind in ("xgru", "metagru"):
            hs = self.cell(x)
        else:
            hs = self.core(x)
        if self.last:
            return self.head(hs[:, -1])
        return self.head(hs)


class XLM(nn.Module):
    def __init__(self, kind, vocab, d):
        super().__init__()
        self.kind = kind
        self.emb = nn.Embedding(vocab, d)
        if kind == "xgru":
            self.cell = XGRUCell(d, d)
        elif kind == "metagru":
            self.cell = MetaGRUCell(d, d)
        else:
            self.core = SeqModel(kind, d, d, vocab, last=False)
        self.head = nn.Linear(d, vocab)

    def forward(self, ids):
        if self.kind in ("xgru", "metagru"):
            hs = self.cell(self.emb(ids))
        else:
            hs = self.core(self.emb(ids))
        return self.head(hs)


def build(kind, m, d, p, last=True):
    if kind in ("xgru", "metagru"):
        return XModel(kind, m, d, p, last)
    return SeqModel(kind, m, d, p, last)


def build_lm(kind, vocab, d):
    if kind in ("xgru", "metagru"):
        return XLM(kind, vocab, d)
    return LMModel(kind, vocab, d)


def run(tag, model_fn, steps, batch_fn, loss_fn, eval_fn, mode="min", seeds=(0, 1),
        use_graph=False):
    if tag in RESULTS:
        print("  %s 宸插畬鎴?(%.5f), 璺宠繃" % (tag, RESULTS[tag]["mean"]), flush=True)
        return
    best_all = []
    t_all = time.time()
    for seed in seeds:
        torch.manual_seed(seed)
        model = model_fn().to(DEV)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        gm = None
        if use_graph:
            x0, y0 = batch_fn()
            gx = x0.clone()
            gm = torch.cuda.make_graphed_callables(model, (gx,))
        best = None
        for st in range(steps):
            model.train()
            x, y = batch_fn()
            if use_graph:
                gx.copy_(x)
                loss = loss_fn(gm(gx), y)
            else:
                loss = loss_fn(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if st % 62 == 0 or st == steps - 1:
                model.eval()
                with torch.no_grad():
                    ev = eval_fn(model)
                if best is None or (ev < best[0] if mode == "min" else ev > best[0]):
                    best = (ev, st)
        best_all.append(best[0])
        del model, gm, opt
        torch.cuda.empty_cache()
    mean = sum(best_all) / len(best_all)
    RESULTS[tag] = {"mean": mean, "seeds": best_all}
    json.dump(RESULTS, open(RJ, "w"))
    fmt = ("%.5f" if mode == "min" else "%.2f%%")
    print("  %s: %s seeds=%s (%.0fs)" % (tag, fmt % mean,
          [fmt % v for v in best_all], time.time() - t_all), flush=True)


def main():
    torch.manual_seed(0)
    kinds = ["gru", "gru2", "lstm", "metagru", "xgru"]

    print("=== adding T=200 (bs512, 625 姝? ===", flush=True)
    for kind in kinds:
        def batch_fn():
            return adding_batch(B, 200, DEV)

        def eval_fn(m):
            x, y = adding_batch(1024, 200, DEV)
            return F.mse_loss(m(x).squeeze(-1), y).item()
        run("add200_" + kind, lambda k=kind: build(k, 2, 128, 1), 625, batch_fn,
            lambda o, y: F.mse_loss(o.squeeze(-1), y), eval_fn,
            use_graph=kind in ("metagru", "xgru"))

    print("=== adding T=400 (bs512, 750 姝? ===", flush=True)
    for kind in kinds:
        def batch_fn():
            return adding_batch(B, 400, DEV)

        def eval_fn(m):
            x, y = adding_batch(1024, 400, DEV)
            return F.mse_loss(m(x).squeeze(-1), y).item()
        run("add400_" + kind, lambda k=kind: build(k, 2, 128, 1), 750, batch_fn,
            lambda o, y: F.mse_loss(o.squeeze(-1), y), eval_fn,
            use_graph=kind in ("metagru", "xgru"))

    print("=== 娣锋矊鍒ゅ埆 T=64 (bs512, 375 姝? ===", flush=True)
    for kind in kinds:
        def batch_fn():
            return chaos_batch(B, 64, DEV)

        def eval_fn(m):
            x, y = chaos_batch(1024, 64, DEV)
            return (m(x).argmax(-1) == y).float().mean().item()
        run("chaos_" + kind, lambda k=kind: build(k, 1, 128, 2), 375, batch_fn,
            lambda o, y: F.cross_entropy(o, y), eval_fn, mode="max", seeds=(0,),
            use_graph=kind in ("metagru", "xgru"))

    print("=== char-LM ctx=128 (bs256, 625 姝? ===", flush=True)
    lm = CharLM(ctx=128, device=DEV)
    vx, vy = lm.batch(512)
    for kind in kinds:
        def batch_fn():
            return lm.batch(256)

        def eval_fn(m):
            o = m(vx)
            return F.cross_entropy(o.reshape(-1, o.shape[-1]), vy.reshape(-1)).item()
        run("lm_" + kind, lambda k=kind: build_lm(kind, lm.vocab, 256), 625, batch_fn,
            lambda o, y: F.cross_entropy(o.reshape(-1, o.shape[-1]), y.reshape(-1)),
            eval_fn, use_graph=kind in ("metagru", "xgru"))

    print("\n===== 姹囨€?=====")
    for k in sorted(RESULTS.keys()):
        print("  %-16s %.5f" % (k, RESULTS[k]["mean"]))


if __name__ == "__main__":
    main()
