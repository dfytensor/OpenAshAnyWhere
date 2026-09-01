"""Meta-GRU 混合体 (来自 metaru_vs_gru 报告的设计) + 修正版 Meta-RU (v2).

MetaGRU: 繁殖项 R·h(1-h) 只进重置门预激活 (策略层), 候选态纯 tanh (内容层):
    r_t = sigmoid(Wr h + Ur u + b + scale·R·h(1-h))
    z_t = sigmoid(Wz h + Uz u + b)
    c_t = tanh(Wc(r h) + Uc u + b)
    h_t = z h + (1-z) c
    R   <- clamp(R + eta R (rho - h), 0.1, 4)

MetaRU-v2 (我方修正): g 用门控 r 缩放 (非 R≈3.5), R 走 logit 偏置, eta=0:
    h_t = (1-r) h + r (g + a),  g = r h (1-h)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MetaGRUCell(nn.Module):
    def __init__(self, m, d, eta=0.02, rho=0.5, r_init=3.5,
                 r_min=0.1, r_max=4.0, scale=1.0, mode="reset"):
        super().__init__()
        self.d = d
        self.mode = mode
        self.scale = scale
        self.eta, self.rho = eta, rho
        self.r_min, self.r_max = r_min, r_max
        self.Wr = nn.Linear(d, d)
        self.Ur = nn.Linear(m, d)
        self.Wz = nn.Linear(d, d)
        self.Uz = nn.Linear(m, d)
        self.Wc = nn.Linear(d, d)
        self.Uc = nn.Linear(m, d)
        self.register_buffer("R", torch.full((d,), float(r_init)))

    def reset(self, b, device):
        self.R = torch.full((b, self.d), 3.5, device=device)

    def forward(self, u_seq, h=None):
        b, T, _ = u_seq.shape
        if h is None:
            h = torch.zeros(b, self.d, device=u_seq.device, dtype=u_seq.dtype)
        self.reset(b, u_seq.device)
        ur, uz, uc = self.Ur(u_seq), self.Uz(u_seq), self.Uc(u_seq)
        hs = []
        for t in range(T):
            repro = self.scale * self.R * h * (1 - h)
            if self.mode == "reset":
                r = torch.sigmoid(self.Wr(h) + ur[:, t] + repro)
            else:
                r = torch.sigmoid(self.Wr(h) + ur[:, t])
            z = torch.sigmoid(self.Wz(h) + uz[:, t])
            pre = self.Wc(r * h) + uc[:, t] + (repro if self.mode == "pre" else 0.0)
            c = torch.tanh(pre)
            h = z * h + (1 - z) * c
            with torch.no_grad():
                self.R += self.eta * self.R * (self.rho - h)
                self.R.clamp_(self.r_min, self.r_max)
            hs.append(h)
        return torch.stack(hs, 1)


class MetaRU2Cell(nn.Module):
    """修正版 Meta-RU (v2): g 用门控 r 缩放, R 走 logit 偏置, eta=0."""

    def __init__(self, m, d, r_init=0.875, clamp_h=True, eta=0.0, rho=0.5):
        super().__init__()
        self.d = d
        self.Wr = nn.Linear(d, d)
        self.Ur = nn.Linear(m, d)
        self.W = nn.Linear(d, d, bias=False)
        self.U = nn.Linear(m, d)
        self.register_buffer("R", torch.full((d,), float(r_init)))
        self.clamp_h = clamp_h

    def forward(self, u_seq, h=None):
        b, T, _ = u_seq.shape
        if h is None:
            h = torch.zeros(b, self.d, device=u_seq.device, dtype=u_seq.dtype)
        ur, uu = self.Ur(u_seq), self.U(u_seq)
        hs = []
        for t in range(T):
            r = torch.sigmoid(self.Wr(h) + ur[:, t] + 2.0 * self.R - 1.0)
            g = r * h * (1 - h)
            h = (1 - r) * h + r * (g + self.W(h) + uu[:, t])
            if self.clamp_h:
                h = h.clamp(0, 1)
            hs.append(h)
        return torch.stack(hs, 1)
