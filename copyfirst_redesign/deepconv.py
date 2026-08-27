"""DeepConv2DLinear: [bs,1,h,w] 网格上的多层 2D 卷积 (rank-1 展开 -> conv2d 堆 -> 收缩).

x [b,s,h] -> E = x * w_in  [b,s,h,w] (rank-1 展开, w=64 网格)
-> conv2d(1 -> c1) -> relu -> conv2d(c1 -> c2) -> ... -> conv2d(cn -> 1)
-> out = Σ_w w_out[w] * T[h,w]  [b,s,h]
参数: w + Σ c_i*c_{i+1}*k² + w
"""
import torch
import torch.nn as nn


class DeepConv2DLinear(nn.Module):
    def __init__(self, h, w=64, channels=(2,), k=3):
        super().__init__()
        self.w_in = nn.Parameter(torch.empty(w))
        self.w_out = nn.Parameter(torch.empty(w))
        nn.init.normal_(self.w_in, 0.0, 0.02)
        nn.init.normal_(self.w_out, 0.0, 0.02)
        cs = [1] + list(channels) + [1]
        layers = []
        for i in range(len(cs) - 1):
            conv = nn.Conv2d(cs[i], cs[i + 1], k, padding=k // 2)
            nn.init.normal_(conv.weight, 0.0, 0.02)
            nn.init.zeros_(conv.bias)
            layers.append(conv)
            if i < len(cs) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        b, s, h = x.shape
        e = (x[..., None] * self.w_in).reshape(b * s, 1, h, self.w_in.numel())
        t = self.net(e)                                   # [bs, 1, h, w]
        t = t.view(b, s, h, self.w_out.numel())
        return (t * self.w_out).sum(-1)                   # [b,s,h]


class FeedForwardD2(nn.Module):
    """OpenASH FeedForward 语义: ffn1(x) * relu(gate(x)) -> ffn2, 三投影全换 DeepConv2DLinear."""

    def __init__(self, hidden_size, w=64, channels=(2,), k=3):
        super().__init__()
        self.ffn1 = DeepConv2DLinear(hidden_size, w, channels, k)
        self.gate = DeepConv2DLinear(hidden_size, w, channels, k)
        self.ffn2 = DeepConv2DLinear(hidden_size, w, channels, k)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.ffn2(self.ffn1(x) * self.relu(self.gate(x)))


def apply_d2(m, channels, k=3, w=16):
    dev = next(m.parameters()).device
    for layer in m.decoder_layers:
        layer.ffn = FeedForwardD2(layer.ffn.ffn1.in_features, w, channels, k).to(dev)
    return m
