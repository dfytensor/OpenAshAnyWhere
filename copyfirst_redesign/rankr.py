"""RankRLinear: 低秩通道混合层 (去卷积版).

out = relu?((x @ W_in) @ W_mid) @ W_out + b   (r 路特征投影 + 交叉混合 + 收缩)

参数: 2·h·r + r² + h  (Linear(h,h): h² + h)
"""
import torch
import torch.nn as nn


class RankRLinear(nn.Module):
    def __init__(self, h, r, act=False):
        super().__init__()
        self.h, self.r, self.act = h, r, act
        self.w_in = nn.Parameter(torch.empty(h, r))
        self.w_mid = nn.Parameter(torch.empty(r, r))
        self.w_out = nn.Parameter(torch.empty(r, h))
        self.bias = nn.Parameter(torch.zeros(h))
        nn.init.normal_(self.w_in, 0.0, 0.02)
        nn.init.orthogonal_(self.w_mid)
        nn.init.normal_(self.w_out, 0.0, 0.02)

    def forward(self, x):
        # x: [b,s,h] -> [bs,h]
        b, s, h = x.shape
        z = x.reshape(-1, h) @ self.w_in
        if self.act:
            z = torch.relu(z)
        z = z @ self.w_mid
        out = z @ self.w_out + self.bias
        return out.view(b, s, h)


class FeedForwardR(nn.Module):
    """OpenASH FeedForward 语义: ffn1(x) * relu(gate(x)) -> ffn2, 三个投影全换 RankRLinear."""

    def __init__(self, hidden_size, r):
        super().__init__()
        self.ffn1 = RankRLinear(hidden_size, r, act=False)
        self.gate = RankRLinear(hidden_size, r, act=True)
        self.ffn2 = RankRLinear(hidden_size, r, act=False)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x1 = self.ffn1(x)
        x2 = self.relu(self.gate(x))
        xx = x1 * x2
        return self.ffn2(xx)


def apply_rankr(m, r):
    dev = next(m.parameters()).device
    for layer in m.decoder_layers:
        layer.ffn = FeedForwardR(layer.ffn.ffn1.in_features, r).to(dev)
    return m
