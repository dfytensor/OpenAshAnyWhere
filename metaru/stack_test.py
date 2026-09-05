"""堆叠路线: XGRU + MetaGRU 双层堆叠 (层间分工, 不融合单细胞).

x-m: 底层 XGRU (表征/多样性) -> 顶层 MetaGRU (记忆)
m-x: 底层 MetaGRU (记忆) -> 顶层 XGRU (表征)
测试: 召回外推 + LM + 生成多样性.
"""
import sys, os, json, time
sys.path.insert(0, r"F:\OpenASH2605\metaru")
import torch
import torch.nn as nn
import torch.nn.functional as F
from mxgru import MXGRUCell
from needle_trained import make_batch, eval_at
from tasks import CharLM
from xgru import XGRUCell
from hybrid import MetaGRUCell

DEV = "cuda"
STACKS = ["x-m", "m-x"]
OUT = {}
RJ = r"F:\OpenASH2605\metaru\mx_results.json"
if os.path.exists(RJ):
    OUT = json.load(open(RJ))


KIND = {"x": "xgru", "m": "metagru"}


def make_cell(kind, m, d):
    if kind == "xgru":
        return XGRUCell(m, d)
    if kind == "metagru":
        return MetaGRUCell(m, d)
    if kind == "mxgru":
        return MXGRUCell(m, d)
    raise ValueError(kind)


class StackCell(nn.Module):
    """两层堆叠: 输入 m 维 -> c1 -> (d 维) -> c2 -> 输出 d 维序列."""

    def __init__(self, kind1, kind2, m, d):
        super().__init__()
        self.d = d
        self.c1 = make_cell(kind1, m, d)
        self.c2 = make_cell(kind2, d, d)

    def forward(self, u_seq):
        h1 = self.c1(u_seq)
        return self.c2(h1)


class BridgeStackCell(nn.Module):
    """瓶颈压缩桥堆叠: c1 -> tanh(Linear(d, bd)) 压缩到 bd 维 -> c2."""

    def __init__(self, kind1, kind2, m, d, bd=8):
        super().__init__()
        self.d = d
        self.c1 = make_cell(kind1, m, d)
        self.bridge = nn.Linear(d, bd)
        self.c2 = make_cell(kind2, bd, d)

    def forward(self, u_seq):
        h1 = self.c1(u_seq)
        u2 = torch.tanh(self.bridge(h1))
        return self.c2(u2)


class StackRecall(nn.Module):
    def __init__(self, order, d=128, bd=8):
        super().__init__()
        if order.startswith("b"):
            k1, k2 = KIND[order.split("-")[1][0]], KIND[order.split("-")[1][1]]
            # order 形如 b-xm / b-mx
            self.cell = BridgeStackCell(k1, k2, 2, d, bd)
        else:
            k1, k2 = KIND[order.split("-")[0]], KIND[order.split("-")[1]]
            self.cell = StackCell(k1, k2, 2, d)
        self.head = nn.Linear(d, 1)

    def forward(self, x):
        return self.head(self.cell(x)[:, -1]).squeeze(-1)


def make_lm(order, vocab, d=256, bd=8):
    if order.startswith("b"):
        k1, k2 = KIND[order.split("-")[1][0]], KIND[order.split("-")[1][1]]
    else:
        k1, k2 = KIND[order.split("-")[0]], KIND[order.split("-")[1]]

    class SLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(vocab, d)
            if order.startswith("b"):
                self.cell = BridgeStackCell(k1, k2, d, d, bd)
            else:
                self.cell = StackCell(k1, k2, d, d)
            self.head = nn.Linear(d, vocab)

        def forward(self, ids):
            return self.head(self.cell(self.emb(ids)))
    return SLM()


def train_recall(order, steps=3000):
    ckpt = r"F:\OpenASH2605\metaru\recall_stack_%s.pth" % order
    done = r"F:\OpenASH2605\metaru\recall_stack_%s.done" % order
    if os.path.exists(done):
        return ckpt
    torch.manual_seed(0)
    m = StackRecall(order).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01)
    start = 0
    if os.path.exists(ckpt):
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        m.load_state_dict(sd["model"])
        opt.load_state_dict(sd["opt"])
        start = sd["step"]
        print("  从 step %d 恢复" % start, flush=True)
    x0, y0 = make_batch(256, 512)
    gm = torch.cuda.make_graphed_callables(m, (x0,))
    t0 = time.time()
    for st in range(start, steps):
        x, y = make_batch(256, 512)
        x0.copy_(x)
        loss = F.mse_loss(gm(x0), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if st % 500 == 0:
            print("  %s %d loss=%.5f (%.0fs)" % (order, st, loss.item(), time.time() - t0),
                  flush=True)
        if (st + 1) % 1000 == 0:
            torch.save({"model": m.state_dict(), "opt": opt.state_dict(), "step": st + 1}, ckpt)
            print("  ckpt @%d 已存" % (st + 1), flush=True)
    torch.save({"model": m.state_dict(), "opt": opt.state_dict(), "step": steps}, ckpt)
    open(done, "w").write("1")
    print("  == %s 召回训完 (%.0fs) ==" % (order, time.time() - t0), flush=True)
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


def main(order):
    torch.manual_seed(0)
    lm = CharLM(ctx=128, device=DEV)

    print("=== 堆叠 %s: 召回训练 (L=512, 3000 步) ===" % order, flush=True)
    train_recall(order)

    print("=== 召回外推 (MSE) ===", flush=True)
    for L in [512, 1024, 2048, 4096]:
        row = dict(OUT.get("recall_%d" % L, {}))
        me = StackRecall(order).to(DEV)
        me.load_state_dict(torch.load(r"F:\OpenASH2605\metaru\recall_stack_%s.pth" % order,
                                      map_location="cpu", weights_only=True))
        row["stack_" + order] = eval_at(me, L)
        OUT["recall_%d" % L] = row
        json.dump(OUT, open(RJ, "w"))
        print("  L=%4d: %s" % (L, "  ".join("%s=%.4f" % (k, v)
                                            for k, v in row.items())), flush=True)

    print("=== LM 训练 (500 步) + 评估 ===", flush=True)
    ckpt = r"F:\OpenASH2605\metaru\lm_stack_%s.pth" % order
    if not os.path.exists(ckpt):
        torch.manual_seed(0)
        m = make_lm(order, lm.vocab).to(DEV)
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
                print("  %s-lm %d loss=%.4f (%.0fs)" % (order, st, loss.item(),
                                                        time.time() - t0), flush=True)
        torch.save(m.state_dict(), ckpt)
    m = make_lm(order, lm.vocab).to(DEV)
    m.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    m.eval()
    vx, vy = lm.batch(512)
    with torch.no_grad():
        ce = F.cross_entropy(m(vx).reshape(-1, lm.vocab), vy.reshape(-1)).item()
    d2, run = gen_distinct(m, lm)
    OUT["lm_stack_%s" % order] = {"ce": ce, "distinct2": d2, "max_run": run}
    json.dump(OUT, open(RJ, "w"))
    print("  %s: CE=%.4f distinct2=%.3f" % (order, ce, d2), flush=True)
    print("\n完成 %s" % order)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "x-m")
