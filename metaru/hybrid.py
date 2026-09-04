"""Meta-GRU 混合体 (繁殖项进重置门) + 修正版 Meta-RU (v2) — 融合投影加速版.

融合: z/r 门共用一次 h 投影 + u 侧投影全序列预计算 (每步仅 2 次 GEMM).
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
        self.Wzr = nn.Linear(d, 2 * d)          # h -> (z_pre, r_pre)
        self.Uzr = nn.Linear(m, 2 * d)
        self.Wc = nn.Linear(d, d)
        self.Uc = nn.Linear(m, d)
        self.register_buffer("R", torch.full((1, d), float(r_init)))

    def reset(self, b, device):
        self.R.fill_(3.5)

    def forward(self, u_seq, h=None):
        b, T, _ = u_seq.shape
        if h is None:
            h = torch.zeros(b, self.d, device=u_seq.device, dtype=u_seq.dtype)
        self.reset(b, u_seq.device)
        uzr, uc = self.Uzr(u_seq), self.Uc(u_seq)      # 全序列预计算
        hs = []
        for t in range(T):
            z_pre, r_pre = (self.Wzr(h) + uzr[:, t]).chunk(2, -1)
            repro = self.scale * self.R * h * (1 - h)
            if self.mode == "reset":
                r_pre = r_pre + repro
            r = torch.sigmoid(r_pre)
            z = torch.sigmoid(z_pre)
            pre = self.Wc(r * h) + uc[:, t] + (repro if self.mode == "pre" else 0.0)
            h = z * h + (1 - z) * torch.tanh(pre)
            with torch.no_grad():
                self.R += self.eta * self.R * (self.rho - h.float().mean(0, keepdim=True))
                self.R.clamp_(self.r_min, self.r_max)
            hs.append(h)
        return torch.stack(hs, 1)


class MetaRU2Cell(nn.Module):
    """修正版 Meta-RU (v2): g 用门控 r 缩放, R 走 logit 偏置. 融合投影."""

    def __init__(self, m, d, r_init=0.875, clamp_h=True, eta=0.0, rho=0.5):
        super().__init__()
        self.d = d
        self.clamp_h = clamp_h
        self.Wh = nn.Linear(d, 2 * d)           # h -> (W_r h + b_r, W_a h)
        self.Wa_bias = nn.Parameter(torch.zeros(d))
        self.Uu = nn.Linear(m, 2 * d)           # u -> (U_r u, U_a u)

    def forward(self, u_seq, h=None):
        b, T, _ = u_seq.shape
        if h is None:
            h = torch.zeros(b, self.d, device=u_seq.device, dtype=u_seq.dtype)
        pru = self.Uu(u_seq)                    # 全序列预计算
        hs = []
        for t in range(T):
            r_pre, a = (self.Wh(h) + pru[:, t]).chunk(2, -1)
            r = torch.sigmoid(r_pre + 2.0 * self.R - 1.0)
            g = r * h * (1 - h)
            h = (1 - r) * h + r * (g + self.Wa_bias + a)
            if self.clamp_h:
                h = h.clamp(0, 1)
            hs.append(h)
        return torch.stack(hs, 1)
