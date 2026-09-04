"""标记标量召回: 流 70% 处埋一个带标记的值, 流尾问该值 (MSE). 训 512 / 测 1024/2048."""
import sys, os, json, time
sys.path.insert(0, r"F:\OpenASH2605\metaru")
import torch
import torch.nn as nn
import torch.nn.functional as F
from tasks import CharLM
from bench_xgru import build_lm

DEV = "cuda"
KINDS = ["gru", "gru2", "lstm", "metagru", "xgru"]
OUT = {}
RJ = r"F:\OpenASH2605\metaru\xgru_infer.json"
if os.path.exists(RJ):
    OUT = json.load(open(RJ))


class RecallModel(nn.Module):
    """噪声+标记双通道流 -> cell -> 末位置线性读出标量."""

    def __init__(self, kind, d=128):
        super().__init__()
        self.kind = kind
        if kind == "metagru":
            from hybrid import MetaGRUCell
            self.cell = MetaGRUCell(2, d)
        elif kind == "xgru":
            from xgru import XGRUCell
            self.cell = XGRUCell(2, d)
        elif kind == "gru":
            self.cell = nn.GRU(2, d, batch_first=True)
        elif kind == "gru2":
            self.cell = nn.GRU(2, d, batch_first=True, num_layers=2)
        elif kind == "lstm":
            self.cell = nn.LSTM(2, d, batch_first=True)
        self.head = nn.Linear(d, 1)

    def forward(self, x):
        if self.kind in ("metagru", "xgru"):
            hs = self.cell(x)
        else:
            hs, _ = self.cell(x)
        return self.head(hs[:, -1]).squeeze(-1)


def make_batch(b, L, device=DEV):
    vals = torch.rand(b, L, device=device) - 0.5
    marks = torch.zeros(b, L, device=device)
    pos = int(L * 0.7)
    marks[:, pos] = 1.0
    y = vals[:, pos]
    x = torch.stack([vals, marks], -1)
    return x, y


@torch.no_grad()
def eval_at(model, L, n=4):
    tot = 0.0
    for _ in range(n):
        x, y = make_batch(256, L)
        tot += F.mse_loss(model(x), y).item()
    return tot / n


def main():
    torch.manual_seed(0)
    models = {}
    print("=== 训练 (L=512, 5000 步, MSE, metagru/xgru 用 CUDA graph) ===", flush=True)
    for kind in KINDS:
        torch.manual_seed(0)
        m = RecallModel(kind).to(DEV)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01)
        t0 = time.time()
        use_graph = kind in ("metagru", "xgru")
        x0, y0 = make_batch(256, 512)
        gm = torch.cuda.make_graphed_callables(m, (x0,)) if use_graph else None
        for st in range(5000):
            x, y = make_batch(256, 512)
            if use_graph:
                x0.copy_(x)
                loss = F.mse_loss(gm(x0), y)
            else:
                loss = F.mse_loss(m(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            if st % 1000 == 0:
                print("  %s %d loss=%.5f (%.0fs)" % (kind, st, loss.item(), time.time() - t0),
                      flush=True)
        torch.save(m.state_dict(), r"F:\OpenASH2605\metaru\recall0_%s.pth" % kind)
        models[kind] = kind          # 存 kind, eval 时用新实例加载
        print("  == %s 训完 (%.0fs) ==" % (kind, time.time() - t0), flush=True)

    print("\n=== 召回泛化 (MSE, 训练 L=512, 测更长) ===", flush=True)
    for L in [512, 1024, 2048, 4096]:
        row = {}
        for kind in KINDS:
            me = RecallModel(kind).to(DEV)
            me.load_state_dict(torch.load(
                r"F:\OpenASH2605\metaru\recall0_%s.pth" % kind, map_location="cpu",
                weights_only=True))
            row[kind] = eval_at(me, L)
        OUT["recall_%d" % L] = row
        json.dump(OUT, open(RJ, "w"))
        print("  L=%4d: %s" % (L, "  ".join("%s=%.4f" % (k, v) for k, v in row.items())),
              flush=True)

    print("\n完成")


if __name__ == "__main__":
    main()
