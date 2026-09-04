"""XGRU: 升级版 GRU 单元.

在标准 GRU 上加两个经过验证的改动:
  1. 候选态内的对角循环直达路径: c = tanh(Wc(r*h) + Uc u + diag*h)
     -> 每个单元一条绕过重置门的长程梯度通道 (IndRNN 思想), 帮助跨"会/不会"相变
  2. 更新门偏置初始化 +1 -> 初始 z≈0.73, 偏保守更新, 长序列记忆保持更久 (forget-bias 技巧)

diag 初始化 0: 关闭时严格退化为 GRU(+bias init), 便于消融.
"""
import torch
import torch.nn as nn


class XGRUCell(nn.Module):
    def __init__(self, m, d):
        super().__init__()
        self.d = d
        self.Wzr = nn.Linear(d, 2 * d)          # h -> (z_pre, r_pre)
        self.Uzr = nn.Linear(m, 2 * d)
        self.Wc = nn.Linear(d, d)
        self.Uc = nn.Linear(m, d)
        self.diag = nn.Parameter(torch.zeros(d))
        with torch.no_grad():
            self.Wzr.bias[:d].fill_(1.0)         # 更新门偏置 +1
            self.Wzr.bias[d:].zero_()
            self.Wc.bias.zero_()

    def forward(self, u_seq, h=None):
        b, T, _ = u_seq.shape
        if h is None:
            h = torch.zeros(b, self.d, device=u_seq.device, dtype=u_seq.dtype)
        uzr, uc = self.Uzr(u_seq), self.Uc(u_seq)      # 全序列预计算
        hs = []
        for t in range(T):
            z_pre, r_pre = (self.Wzr(h) + uzr[:, t]).chunk(2, -1)
            z = torch.sigmoid(z_pre)
            r = torch.sigmoid(r_pre)
            c = torch.tanh(self.Wc(r * h) + uc[:, t] + self.diag * h)
            h = z * h + (1 - z) * c
            hs.append(h)
        return torch.stack(hs, 1)
