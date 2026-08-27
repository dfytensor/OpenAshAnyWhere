"""ConvLinear: 用 rank-1 扩展 + (h,w) 空间卷积 + 收缩 替代标准 Linear(h,h).

数据流 (按定义):
  [b,s,h] -> @w(1,w) -> [b,s,h,w] -> reshape [b*s,1,h,w]
          -> Conv2d(1->1,k) -> 激活 -> [b,s,h,w] -> @w(w,1) -> [b,s,h]
参数: 2w + k^2  (vs 标准 h^2)  — 权重共享的通道混合器.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLinear(nn.Module):
    def __init__(self, h, w=None, k=3, act=None, bias=True):
        super().__init__()
        w = w or h
        self.h, self.w = h, w
        self.w_in = nn.Parameter(torch.empty(1, w))     # 扩展向量
        self.w_out = nn.Parameter(torch.empty(w, 1))    # 收缩向量
        self.conv = nn.Conv2d(1, 1, k, padding=k // 2, bias=bias)
        self.act = act if act is not None else nn.ReLU()
        nn.init.normal_(self.w_in, 0.0, 0.02)
        nn.init.normal_(self.w_out, 0.0, 0.02)

    def forward(self, x):
        b, s, h = x.shape
        xw = x.unsqueeze(-1) * self.w_in                # [b,s,h,1]@[1,w] = [b,s,h,w]
        img = xw.reshape(b * s, 1, h, self.w)
        img = self.conv(img)
        img = self.act(img)
        out = img.reshape(b, s, h, self.w) @ self.w_out  # [b,s,h,w]@[w,1]
        return out.squeeze(-1)                          # [b,s,h]
