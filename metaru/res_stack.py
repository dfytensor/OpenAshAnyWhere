"""GPT 式残差深堆: LN(h + Cell(h)) block, 纯 MetaGRU / 纯 GRU 各 2 层.

检验: MetaGRU 在 GPT 式深架构 (残差+LN) 下能否保留长程记忆 + 堆叠收益.
"""
import sys, os, json, time
sys.path.insert(0, r"F:\OpenASH2605\metaru")
import torch
import torch.nn as nn
import torch.nn.functional as F
from hybrid import MetaGRUCell
from needle_trained import make_batch, eval_at
from tasks import CharLM

DEV = "cuda"
OUT = {}
RJ = r"F:\OpenASH2605\metaru\mx_results.json"
if os.path.exists(RJ):
    OUT = json.load(open(RJ))


class ResBlock(nn.Module):
    """GPT 式 block: LN(x + Cell(x)). Cell 用同一 hidden 维的循环单元."""

    def __init__(self, kind, d):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        if kind == "metagru":
            self.cell = MetaGRUCell(d, d)
        else:
            self.cell = nn.GRUCell(d, d)

    def forward(self, x):
        # x: [b,T,d] -> 逐时间步调用 GRUCell/MetaGRUCell
        if isinstance(self.cell, MetaGRUCell):
            return self.ln(x + self.cell(x))
        b, T, d = x.shape
        h = torch.zeros(b, d, device=x.device, dtype=x.dtype)
        hs = []
        for t in range(T):
            h = self.cell(x[:, t], h)
            hs.append(h)
        return self.ln(x + torch.stack(hs, 1))


class ResStack(nn.Module):
    def __init__(self, kind, m, d, nlayers):
        super().__init__()
        self.inp = nn.Linear(m, d)
        self.blocks = nn.ModuleList([ResBlock(kind, d) for _ in range(nlayers)])

    def forward(self, u):
        x = self.inp(u)
        for blk in self.blocks:
            x = blk(x)
        return x


class ResRecall(nn.Module):
    def __init__(self, kind, d=128, nlayers=2):
        super().__init__()
        self.core = ResStack(kind, 2, d, nlayers)
        self.head = nn.Linear(d, 1)

    def forward(self, x):
        return self.head(self.core(x)[:, -1]).squeeze(-1)


def make_reslm(kind, vocab, d=256, nlayers=2):
    class RLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(vocab, d)
            self.core = ResStack(kind, d, d, nlayers)
            self.head = nn.Linear(d, vocab)

        def forward(self, ids):
            return self.head(self.core(self.emb(ids)))
    return RLM()


def train(kind, steps=3000, tag="recall"):
    ckpt = r"F:\OpenASH2605\metaru\res_%s_%s.pth" % (tag, kind)
    done = r"F:\OpenASH2605\metaru\res_%s_%s.done" % (tag, kind)
    if os.path.exists(done):
        return ckpt
    torch.manual_seed(0)
    if tag == "recall":
        m = ResRecall(kind).to(DEV)
    else:
        m = make_reslm(kind, CharLM(ctx=128, device=DEV).vocab).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01)
    start = 0
    if os.path.exists(ckpt):
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        m.load_state_dict(sd["model"])
        opt.load_state_dict(sd["opt"])
        start = sd["step"]
        print("  从 step %d 恢复" % start, flush=True)
    lm = CharLM(ctx=128, device=DEV) if tag == "lm" else None
    if tag == "recall":
        x0, y0 = make_batch(256, 512)
        batch_fn = lambda: make_batch(256, 512)
    else:
        x0, y0 = lm.batch(256)
        batch_fn = lambda: lm.batch(256)
    gm = torch.cuda.make_graphed_callables(m, (x0,))
    t0 = time.time()
    for st in range(start, steps):
        x, y = batch_fn()
        x0.copy_(x)
        if tag == "recall":
            loss = F.mse_loss(gm(x0), y)
        else:
            loss = F.cross_entropy(gm(x0).reshape(-1, lm.vocab), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if st % 500 == 0:
            print("  %s-%s %d loss=%.5f (%.0fs)" % (kind, tag, st, loss.item(),
                                                    time.time() - t0), flush=True)
        if (st + 1) % 1000 == 0:
            torch.save({"model": m.state_dict(), "opt": opt.state_dict(), "step": st + 1}, ckpt)
            print("  ckpt @%d 已存" % (st + 1), flush=True)
    torch.save({"model": m.state_dict(), "opt": opt.state_dict(), "step": steps}, ckpt)
    open(done, "w").write("1")
    print("  == %s-%s 训完 (%.0fs) ==" % (kind, tag, time.time() - t0), flush=True)
    return ckpt


def main():
    torch.manual_seed(0)
    print("=== GPT 式残差 2 层: 召回训练 (L=512, 3000 步) ===", flush=True)
    for kind in ["metagru", "gru"]:
        train(kind, 3000, "recall")

    print("=== 召回外推 ===", flush=True)
    for L in [512, 1024, 2048, 4096]:
        row = dict(OUT.get("recall_%d" % L, {}))
        for kind in ["metagru", "gru"]:
            me = ResRecall(kind).to(DEV)
            sd = torch.load(r"F:\OpenASH2605\metaru\res_recall_%s.pth" % kind,
                            map_location="cpu", weights_only=False)
            me.load_state_dict(sd["model"])
            row["res2_%s" % kind] = eval_at(me, L)
        OUT["recall_%d" % L] = row
        json.dump(OUT, open(RJ, "w"))
        print("  L=%4d: %s" % (L, "  ".join("%s=%.4f" % (k, v)
                                            for k, v in row.items())), flush=True)


if __name__ == "__main__":
    main()
