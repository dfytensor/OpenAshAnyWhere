"""MXGRU: MetaGRU + XGRU 取长补短.

    z_pre, r_pre = Wzr(h) + Uzr(u)
    r_pre += scale_r * R * h * (1-h)          # MetaGRU 天线 (进重置门)
    z = sigmoid(z_pre);  r = sigmoid(r_pre)
    c = tanh(Wc(r*h) + Uc(u) + scale_d * diag * h)   # XGRU 对角直达
    h = z*h + (1-z)*c
    R += eta*R*(rho - mean_batch(h)); clamp    # 内稳态

z 偏置初始 +1.  scale_r=0 时退化为 XGRU, scale_d=0 时退化为 MetaGRU(+bias init).
"""
import torch
import torch.nn as nn


class MXGRUCell(nn.Module):
    def __init__(self, m, d, eta=0.02, rho=0.5, r_min=0.1, r_max=4.0,
                 scale_r=1.0, scale_d=1.0, z_bias=1.0, mode="always"):
        super().__init__()
        self.d = d
        self.eta, self.rho = eta, rho
        self.r_min, self.r_max = r_min, r_max
        self.scale_r, self.scale_d = scale_r, scale_d
        self.mode = mode          # always / rup (diag 随 R 升强) / rdn (随 R 降强)
        self.Wzr = nn.Linear(d, 2 * d)
        self.Uzr = nn.Linear(m, 2 * d)
        self.Wc = nn.Linear(d, d)
        self.Uc = nn.Linear(m, d)
        self.diag = nn.Parameter(torch.zeros(d))
        self.register_buffer("R", torch.full((1, d), 3.5))
        with torch.no_grad():
            self.Wzr.bias[:d].fill_(z_bias)
            self.Wzr.bias[d:].zero_()
            self.Wc.bias.zero_()

    def reset(self):
        self.R.fill_(3.5)

    def diag_gain(self):
        """R -> diag 门控系数 [0,1]. rup: R 高时直通开; rdn: R 低时直通开."""
        if self.mode == "always":
            return torch.ones_like(self.R)
        g = (self.R - self.r_min) / (self.r_max - self.r_min)   # [0,1]
        if self.mode == "rdn":
            g = 1.0 - g
        return g.clamp(0, 1)

    def forward(self, u_seq, h=None):
        b, T, _ = u_seq.shape
        if h is None:
            h = torch.zeros(b, self.d, device=u_seq.device, dtype=u_seq.dtype)
        self.reset()
        uzr, uc = self.Uzr(u_seq), self.Uc(u_seq)
        hs = []
        for t in range(T):
            z_pre, r_pre = (self.Wzr(h) + uzr[:, t]).chunk(2, -1)
            repro = self.scale_r * self.R * h * (1 - h)
            r = torch.sigmoid(r_pre + repro)
            z = torch.sigmoid(z_pre)
            diag_inj = self.scale_d * self.diag * h * self.diag_gain()
            c = torch.tanh(self.Wc(r * h) + uc[:, t] + diag_inj)
            h = z * h + (1 - z) * c
            with torch.no_grad():
                self.R += self.eta * self.R * (self.rho - h.float().mean(0, keepdim=True))
                self.R.clamp_(self.r_min, self.r_max)
            hs.append(h)
        return torch.stack(hs, 1)
