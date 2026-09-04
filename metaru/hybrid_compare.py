"""结合实验: GRU vs MetaGRU-reset (对方设计) vs MetaRU-v2 (我方修正).

任务: ① adding T=200  ② 混沌/周期判别 (对方的核心优势任务)  ③ char-LM
每任务 2 种子取均值.
"""
import sys, time
sys.path.insert(0, r"F:\OpenASH2605\metaru")
import torch
import torch.nn as nn
import torch.nn.functional as F
from metaru import LMModel
from metaru import SeqModel
from hybrid import MetaGRUCell, MetaRU2Cell
from tasks import adding_batch, CharLM

DEV = "cuda"
RESULTS = {}


class HybridModel(nn.Module):
    """统一封装: MetaGRU / MetaRU2 cell + 读出头."""

    def __init__(self, cell_kind, m, d, p, last=True):
        super().__init__()
        self.cell_kind, self.d, self.last = cell_kind, d, last
        self.cell = MetaGRUCell(m, d) if cell_kind == "metagru" else MetaRU2Cell(m, d)
        self.head = nn.Linear(d, p)

    def forward(self, x):
        hs = self.cell(x)
        if self.last:
            return self.head(hs[:, -1])
        return self.head(hs)


class HybridLM(nn.Module):
    def __init__(self, cell_kind, vocab, d):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.cell = MetaGRUCell(d, d) if cell_kind == "metagru" else MetaRU2Cell(d, d)
        self.head = nn.Linear(d, vocab)

    def forward(self, ids):
        hs = self.cell(self.emb(ids))
        return self.head(hs)


def chaos_batch(b, T, device):
    """logistic 轨道判别: 周期区(r<3.57)=0, 混沌区(r>3.57)=1. 向量化 GPU 生成."""
    half = torch.rand(b, device=device) < 0.5
    r = torch.where(half, 2.9 + 0.5 * torch.rand(b, device=device),
                    3.7 + 0.3 * torch.rand(b, device=device))
    ys = (r > 3.57).long()
    x = torch.rand(b, device=device)
    for _ in range(150):
        x = r * x * (1 - x)
    out = []
    for _ in range(T):
        x = r * x * (1 - x)
        out.append(x)
    return torch.stack(out, 1).unsqueeze(-1), ys


def run(tag, model_fn, steps, batch_fn, loss_fn, eval_fn, eval_every=500, mode="min", seeds=(0, 1)):
    best_all, t_all = [], 0.0
    for seed in seeds:
        torch.manual_seed(seed)
        model = model_fn().to(DEV)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        best, t0 = None, time.time()
        for st in range(steps):
            model.train()
            x, y = batch_fn()
            loss = loss_fn(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if st % eval_every == 0 or st == steps - 1:
                model.eval()
                with torch.no_grad():
                    ev = eval_fn(model)
                if best is None or (ev < best[0] if mode == "min" else ev > best[0]):
                    best = (ev, st)
        best_all.append(best[0])
        t_all += time.time() - t0
        del model
        torch.cuda.empty_cache()
    mean = sum(best_all) / len(best_all)
    RESULTS[tag] = mean
    print("  %s: best=%s seeds=%s (%.0fs)" %
          (tag, "%.5f" % mean if mode == "min" else "%.2f%%" % (mean * 100),
           ["%.5f" % v if mode == "min" else "%.1f%%" % (v * 100) for v in best_all],
           t_all), flush=True)


def main():
    torch.manual_seed(0)

    print("=== 任务1: adding T=200 (MSE) ===", flush=True)
    for kind in ["gru", "metagru", "metaru2"]:
        def batch_fn():
            return adding_batch(128, 200, DEV)

        def eval_fn(m):
            x, y = adding_batch(1024, 200, DEV)
            return F.mse_loss(m(x).squeeze(-1), y).item()
        fn = (lambda k=kind: SeqModel("gru", 2, 128, 1)) if kind == "gru" \
            else (lambda k=kind: HybridModel(k, 2, 128, 1))
        run("add_" + kind, fn, 2500, batch_fn,
            lambda o, y: F.mse_loss(o.squeeze(-1), y), eval_fn)

    print("=== 任务2: 混沌/周期判别 T=64 (ACC) ===", flush=True)
    for kind in ["gru", "metagru", "metaru2"]:
        def batch_fn():
            return chaos_batch(128, 64, DEV)

        def eval_fn(m):
            x, y = chaos_batch(512, 64, DEV)
            return (m(x).argmax(-1) == y).float().mean().item()
        fn = (lambda k=kind: SeqModel("gru", 1, 128, 2)) if kind == "gru" \
            else (lambda k=kind: HybridModel(k, 1, 128, 2))
        run("chaos_" + kind, fn, 2000, batch_fn,
            lambda o, y: F.cross_entropy(o, y), eval_fn, mode="max", eval_every=250)

    print("=== 任务3: char-LM ctx=128 ===", flush=True)
    lm = CharLM(ctx=128, device=DEV)
    vx, vy = lm.batch(256)
    for kind in ["gru", "metagru", "metaru2"]:
        def batch_fn():
            return lm.batch(64)

        def eval_fn(m):
            o = m(vx)
            return F.cross_entropy(o.reshape(-1, o.shape[-1]), vy.reshape(-1)).item()
        fn = (lambda k=kind: LMModel("gru", lm.vocab, 256)) if kind == "gru" \
            else (lambda k=kind: HybridLM(k, lm.vocab, 256))
        run("lm_" + kind, fn, 2500, batch_fn,
            lambda o, y: F.cross_entropy(o.reshape(-1, o.shape[-1]), y.reshape(-1)),
            eval_fn)

    print("\n===== 结合实验汇总 (2 种子均值) =====")
    for k, v in RESULTS.items():
        print("  %-16s %.5f" % (k, v))


if __name__ == "__main__":
    main()
