"""R 门控 diag 变体测试: mx-rup (diag 随 R 升强) vs mx-rdn (diag 随 R 降强) vs 基线.

测试: ①召回外推 (MetaGRU 主场) ②LM CE + 生成多样性.
"""
import sys, os, json, time
sys.path.insert(0, r"F:\OpenASH2605\metaru")
import torch
import torch.nn as nn
import torch.nn.functional as F
from mxgru import MXGRUCell
from needle_trained import make_batch, eval_at
from tasks import CharLM
from mx_bench import MXRecall, gen_distinct

DEV = "cuda"
MODES = ["mx-rup", "mx-rdn"]
OUT = {}
RJ = r"F:\OpenASH2605\metaru\mx_results.json"
if os.path.exists(RJ):
    OUT = json.load(open(RJ))


def make_recall(mode, d=128):
    class R(nn.Module):
        def __init__(self):
            super().__init__()
            self.cell = MXGRUCell(2, d, mode=mode.split("-")[1])
            self.head = nn.Linear(d, 1)

        def forward(self, x):
            return self.head(self.cell(x)[:, -1]).squeeze(-1)
    return R()


def train_recall(mode, steps=4000):
    ckpt = r"F:\OpenASH2605\metaru\recall_mx_%s.pth" % mode
    if os.path.exists(ckpt):
        return ckpt
    torch.manual_seed(0)
    m = make_recall(mode).to(DEV)
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
            print("  %s %d loss=%.5f (%.0fs)" % (mode, st, loss.item(), time.time() - t0),
                  flush=True)
    torch.save(m.state_dict(), ckpt)
    print("  == %s 训完 (%.0fs) ==" % (mode, time.time() - t0), flush=True)
    return ckpt


def make_lm(mode, vocab, d=256):
    class MXLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(vocab, d)
            self.cell = MXGRUCell(d, d, mode=mode.split("-")[1])
            self.head = nn.Linear(d, vocab)

        def forward(self, ids):
            return self.head(self.cell(self.emb(ids)))
    return MXLM()


def main():
    torch.manual_seed(0)
    lm = CharLM(ctx=128, device=DEV)

    print("=== ① 召回训练 (L=512, 4000 步) ===", flush=True)
    for mode in MODES:
        train_recall(mode)

    print("=== ① 召回外推 (MSE) ===", flush=True)
    for L in [512, 1024, 2048, 4096]:
        row = dict(OUT.get("recall_%d" % L, {}))
        for mode in MODES:
            me = make_recall(mode).to(DEV)
            me.load_state_dict(torch.load(r"F:\OpenASH2605\metaru\recall_mx_%s.pth" % mode,
                                          map_location="cpu", weights_only=True))
            row[mode] = eval_at(me, L)
        OUT["recall_%d" % L] = row
        json.dump(OUT, open(RJ, "w"))
        print("  L=%4d: %s" % (L, "  ".join("%s=%.4f" % (k, v)
                                            for k, v in row.items())), flush=True)

    print("=== ② LM (500 步) ===", flush=True)
    vx, vy = lm.batch(512)
    for mode in MODES:
        ckpt = r"F:\OpenASH2605\metaru\lm_mx_%s.pth" % mode
        if not os.path.exists(ckpt):
            torch.manual_seed(0)
            m = make_lm(mode, lm.vocab).to(DEV)
            opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01)
            x0, y0 = lm.batch(256)
            gm = torch.cuda.make_graphed_callables(m, (x0,))
            t0 = time.time()
            for st in range(500):
                x, y = lm.batch(256)
                x0.copy_(x)
                loss = F.cross_entropy(gm(x0).reshape(-1, lm.vocab), y.reshape(-1))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step()
                if st % 100 == 0:
                    print("  %s-lm %d loss=%.4f (%.0fs)" % (mode, st, loss.item(),
                                                            time.time() - t0), flush=True)
            torch.save(m.state_dict(), ckpt)
        m = make_lm(mode, lm.vocab).to(DEV)
        m.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        m.eval()
        with torch.no_grad():
            ce = F.cross_entropy(m(vx).reshape(-1, lm.vocab), vy.reshape(-1)).item()
        d2, run = gen_distinct(m, lm)
        OUT["lm_%s" % mode] = {"ce": ce, "distinct2": d2, "max_run": run}
        json.dump(OUT, open(RJ, "w"))
        print("  %s: CE=%.4f distinct2=%.3f" % (mode, ce, d2), flush=True)

    print("\n完成")


if __name__ == "__main__":
    main()
