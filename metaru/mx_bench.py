"""MX-GRU 取长补短测试: ①流内召回外推 (MetaGRU 主场) ②LM+生成多样性 (XGRU 主场)."""
import sys, os, json, time
sys.path.insert(0, r"F:\OpenASH2605\metaru")
import torch
import torch.nn as nn
import torch.nn.functional as F
from mxgru import MXGRUCell
from needle_trained import make_batch, eval_at
from bench_xgru import build_lm
from tasks import CharLM

DEV = "cuda"
KINDS = ["metagru", "xgru", "mxgru"]
OUT = {}
RJ = r"F:\OpenASH2605\metaru\mx_results.json"
if os.path.exists(RJ):
    OUT = json.load(open(RJ))


class MXRecall(nn.Module):
    def __init__(self, kind, d=128):
        super().__init__()
        self.kind = kind
        if kind == "mxgru":
            self.cell = MXGRUCell(2, d)
        elif kind == "metagru":
            from hybrid import MetaGRUCell
            self.cell = MetaGRUCell(2, d)
        elif kind == "xgru":
            from xgru import XGRUCell
            self.cell = XGRUCell(2, d)
        self.head = nn.Linear(d, 1)

    def forward(self, x):
        hs = self.cell(x)
        return self.head(hs[:, -1]).squeeze(-1)


def train_recall(kind, steps=5000):
    ckpt = r"F:\OpenASH2605\metaru\recall_mx_%s.pth" % kind
    if os.path.exists(ckpt):
        return ckpt
    torch.manual_seed(0)
    m = MXRecall(kind).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01)
    x0, y0 = make_batch(256, 512)
    gm = torch.cuda.make_graphed_callables(m, (x0,))
    t0 = time.time()
    for st in range(steps):
        x, y = make_batch(256, 512)
        x0.copy_(x)
        loss = F.mse_loss(gm(x0), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if st % 1000 == 0:
            print("  %s %d loss=%.5f (%.0fs)" % (kind, st, loss.item(), time.time() - t0),
                  flush=True)
    torch.save(m.state_dict(), ckpt)
    print("  == %s 训完 (%.0fs) ==" % (kind, time.time() - t0), flush=True)
    return ckpt


def train_lm_fast(kind, lm, steps=800):
    ckpt = r"F:\OpenASH2605\metaru\lm_mx_%s.pth" % kind
    if os.path.exists(ckpt):
        return ckpt
    torch.manual_seed(0)
    m = build_lm(kind, lm.vocab, 256).to(DEV) if kind != "mxgru" else None
    if kind == "mxgru":
        class MXLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = nn.Embedding(lm.vocab, 256)
                self.cell = MXGRUCell(256, 256)
                self.head = nn.Linear(256, lm.vocab)

            def forward(self, ids):
                return self.head(self.cell(self.emb(ids)))
        m = MXLM().to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01)
    x0, y0 = lm.batch(256)
    gm = torch.cuda.make_graphed_callables(m, (x0,))
    t0 = time.time()
    for st in range(steps):
        x, y = lm.batch(256)
        x0.copy_(x)
        loss = F.cross_entropy(gm(x0).reshape(-1, lm.vocab), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if st % 200 == 0:
            print("  %s-lm %d loss=%.4f (%.0fs)" % (kind, st, loss.item(), time.time() - t0),
                  flush=True)
    torch.save(m.state_dict(), ckpt)
    print("  == %s-lm 训完 (%.0fs) ==" % (kind, time.time() - t0), flush=True)
    return ckpt


@torch.no_grad()
def gen_distinct(model, lm, n=400, prompt="第一"):
    ids = torch.tensor([[lm.stoi.get(c, 0) for c in prompt]], device=DEV)
    cur = ids
    for _ in range(n):
        logits = model(cur)
        nxt = logits[0, -1].argmax().view(1, 1)
        cur = torch.cat([cur, nxt], 1)
    g = cur[0, 1:].tolist()
    d2 = len(set(zip(g[:-1], g[1:]))) / max(len(g) - 1, 1)
    run = best = 1
    for a, b2 in zip(g[:-1], g[1:]):
        run = run + 1 if a == b2 else 1
        best = max(best, run)
    return d2, best


def main():
    torch.manual_seed(0)
    lm = CharLM(ctx=128, device=DEV)

    print("=== ① 流内召回训练 (L=512, 5000 步) ===", flush=True)
    for kind in KINDS:
        train_recall(kind)

    print("=== ① 召回外推 (MSE) ===", flush=True)
    for L in [512, 1024, 2048, 4096]:
        row = {}
        for kind in KINDS:
            me = MXRecall(kind).to(DEV)
            me.load_state_dict(torch.load(
                r"F:\OpenASH2605\metaru\recall_mx_%s.pth" % kind,
                map_location="cpu", weights_only=True))
            row[kind] = eval_at(me, L)
        OUT["recall_%d" % L] = row
        json.dump(OUT, open(RJ, "w"))
        print("  L=%4d: %s" % (L, "  ".join("%s=%.4f" % (k, v) for k, v in row.items())),
              flush=True)

    print("=== ② LM 训练 (800 步) ===", flush=True)
    for kind in KINDS:
        train_lm_fast(kind, lm)

    print("=== ② LM 评估 + 生成多样性 ===", flush=True)
    vx, vy = lm.batch(512)
    for kind in KINDS:
        m = build_lm(kind, lm.vocab, 256).to(DEV) if kind != "mxgru" else None
        if kind == "mxgru":
            class MXLM(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.emb = nn.Embedding(lm.vocab, 256)
                    self.cell = MXGRUCell(256, 256)
                    self.head = nn.Linear(256, lm.vocab)

                def forward(self, ids):
                    return self.head(self.cell(self.emb(ids)))
            m = MXLM().to(DEV)
        m.load_state_dict(torch.load(r"F:\OpenASH2605\metaru\lm_mx_%s.pth" % kind,
                                     map_location="cpu", weights_only=True))
        m.eval()
        with torch.no_grad():
            ce = F.cross_entropy(m(vx).reshape(-1, lm.vocab), vy.reshape(-1)).item()
        d2, run = gen_distinct(m, lm)
        OUT["lm_%s" % kind] = {"ce": ce, "distinct2": d2, "max_run": run}
        json.dump(OUT, open(RJ, "w"))
        print("  %s: CE=%.4f distinct2=%.3f max_run=%d" % (kind, ce, d2, run), flush=True)

    print("\n完成")


if __name__ == "__main__":
    main()
