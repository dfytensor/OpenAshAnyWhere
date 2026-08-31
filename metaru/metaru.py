"""Meta-RU (元可塑性循环单元): 混沌映射推演出的门控 RNN + 可调变体.

    r_t = sigmoid(Wr h + Ur u + 2R - 1)          # 快门控 (R=慢速内参经 logit 注入)
    g_t = r_t * h * (1 - h)                      # 二次非线性繁衍项
    a_t = W h + U u                              # 耦合与注入
    h_t  = (1 - r_t) h + r_t g_t + a_t  [clamp 0,1]
    R    <- R + eta * (rho - mean(h)) * R        # 慢速内稳态 (无梯度, 任务级)

变体开关:
    gated_a=True : 注入项门控 h_t = (1-r)h + r(g + a)  (修复未门控注入)
    clamp_h=False: 不 clamp (观察饱和/发散)
    eta          : 内稳态步长 (0 = 关闭)
"""
import torch
import torch.nn as nn


class MetaRUCell(nn.Module):
    def __init__(self, m, d, r_init=0.875, eta=0.05, rho=0.5,
                 clamp_h=True, gated_a=False):
        super().__init__()
        self.d = d
        self.Wr = nn.Linear(d, d)
        self.Ur = nn.Linear(m, d)
        self.W = nn.Linear(d, d, bias=False)
        self.U = nn.Linear(m, d)
        self.register_buffer("R", torch.full((d,), float(r_init)))
        self.eta, self.rho = eta, rho
        self.clamp_h, self.gated_a = clamp_h, gated_a

    def forward(self, u_seq, h=None):
        b, T, _ = u_seq.shape
        if h is None:
            h = torch.zeros(b, self.d, device=u_seq.device, dtype=u_seq.dtype)
        ur = self.Ur(u_seq)                      # [b,T,d] 输入投影预计算
        uu = self.U(u_seq)
        hs = []
        for t in range(T):
            r = torch.sigmoid(self.Wr(h) + ur[:, t] + 2.0 * self.R - 1.0)
            g = r * h * (1 - h)
            if self.gated_a:
                h = (1 - r) * h + r * (g + uu[:, t] + self.W(h))
            else:
                h = (1 - r) * h + r * g + self.W(h) + uu[:, t]
            if self.clamp_h:
                h = h.clamp(0, 1)
            if self.eta > 0:
                with torch.no_grad():
                    self.R.add_(self.eta * (self.rho - h.float().mean(0)) * self.R)
                    self.R.clamp_(0.05, 1.0)
            hs.append(h)
        return torch.stack(hs, 1)


class SeqModel(nn.Module):
    """统一封装: cell(MetaRU/GRU/LSTM) + 输出头. last=True 用末状态, 否则逐步输出."""

    def __init__(self, kind, m, d, p, last=True, **kw):
        super().__init__()
        self.kind, self.d, self.last = kind, d, last
        if kind == "metaru":
            self.cell = MetaRUCell(m, d, **kw)
        elif kind == "gru":
            self.cell = nn.GRU(m, d, batch_first=True, num_layers=kw.get("layers", 1))
        elif kind == "lstm":
            self.cell = nn.LSTM(m, d, batch_first=True, num_layers=kw.get("layers", 1))
        else:
            raise ValueError(kind)
        self.head = nn.Linear(d, p)

    def forward(self, x):
        if self.kind == "metaru":
            hs = self.cell(x)
        else:
            hs, _ = self.cell(x)
        if self.last:
            return self.head(hs[:, -1])
        return self.head(hs)


class LMModel(nn.Module):
    """词元输入版: embedding -> cell -> 逐步 logits."""

    def __init__(self, kind, vocab, d, **kw):
        super().__init__()
        self.kind = kind
        self.emb = nn.Embedding(vocab, d)
        if kind == "metaru":
            self.cell = MetaRUCell(d, d, **kw)
        elif kind == "gru":
            self.cell = nn.GRU(d, d, batch_first=True, num_layers=kw.get("layers", 1))
        elif kind == "lstm":
            self.cell = nn.LSTM(d, d, batch_first=True, num_layers=kw.get("layers", 1))
        self.head = nn.Linear(d, vocab)

    def forward(self, ids):
        x = self.emb(ids)
        if self.kind == "metaru":
            hs = self.cell(x)
        else:
            hs, _ = self.cell(x)
        return self.head(hs)
