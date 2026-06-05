"""
OpenASH Enhanced with WDLM-inspired mechanisms
借鉴 WDLM 的三个改进:
  1. WaveState: cummax 加入可学习指数衰减 (模拟 exp(-α|distance|))
  2. PhaseGating: 复数辐角分解 + sin/cos 相位门控替代简单 ReLU
  3. InterferenceCombine: 多分支干涉式组合替代 gen_model
"""

import torch
from torch import nn
import math


# ============================================================
# 改进 1: 带衰减的 Cummax State (借鉴 exp(-α|distance|))
# ============================================================
class MaxStateWithDecay(nn.Module):
    """
    原始 cummax: state = max(prev_state, new_values) -- 无衰减, 信息永存
    WDLM 启发: decay = exp(-α * |distance|), 距离远的信息衰减
    实现: state = max(decay * prev_state, new_values)
    """
    def __init__(self, dim_size, heads, model_flag="train"):
        super().__init__()
        self.heads = heads
        self.d_head = dim_size // heads
        self.model_flag = model_flag
        assert dim_size % heads == 0

        self.combined = nn.Linear(dim_size, 4 * dim_size, bias=False)

        # WDLM-inspired: per-head learnable decay
        self.state_decay = nn.Parameter(torch.ones(heads) * 0.95)  # 接近1 = 长记忆
        self.state_bias = nn.Parameter(torch.zeros(heads))

        self.alpha1 = nn.Parameter(torch.tensor(0.5))
        self.alpha2 = nn.Parameter(torch.tensor(0.5))
        self.alpha3 = nn.Parameter(torch.tensor(0.5))
        self.head_linear = nn.Linear(heads * 5, heads, bias=False)

    def forward(self, x, state=None):
        b, s, d = x.shape
        combined = self.combined(x).view(b, s, 4, self.heads, -1)
        out, out1, out2, out3 = combined.unbind(2)

        out = out.permute(0, 3, 1, 2)
        out1 = out1.permute(0, 3, 1, 2)
        out2 = out2.permute(0, 3, 1, 2)
        out3 = out3.permute(0, 3, 1, 2)

        # --- WDLM-inspired: 带衰减的 cummax ---
        decay = torch.sigmoid(self.state_decay).view(1, self.heads, 1, 1)

        if state is None:
            # 首次用全部 cummax
            out4, _ = torch.cummax(out2, dim=2)
            state = out4[:, :, -1:]
        else:
            # WDLM: decay * prev_state → 越远的过去越衰减
            decayed_state = decay * state + self.state_bias.view(1, self.heads, 1, 1)
            out4, _ = torch.cummax(torch.cat([decayed_state, out2], dim=2), dim=2)
            if self.model_flag == "train":
                out4 = out4[:, :, 1:]
            else:
                out4 = out4[:, :, -1:]
            state = out4[:, :, -1:]

        out_l = self.combine_branches(out, out1, out2, out3, out4)
        return out_l.transpose(1, 2).contiguous().view(b, s, d), state

    def combine_branches(self, a, b, c, d, e):
        """原始 gen_model (保持兼容)"""
        combined = torch.cat([a, b, c, d, e], dim=-1)
        combined = self.head_linear(combined) * e
        term1 = a * b
        term2 = self.alpha1 * b + self.alpha2 * d
        term3 = a * (self.alpha3 * e + d)
        term4 = b * (c + e)
        return term1 + term2 + term3 + term4 + c * e + combined


# ============================================================
# 改进 2: 复数相位门控 FeedForward (借鉴 sin/cos + gate)
# ============================================================
class PhaseGatingFFN(nn.Module):
    """
    WDLM 借鉴: sin/cos 周期非线性替代 ReLU 门控
    核心: sin(gate) + cos(gate) 在 [-√2, √2] 区间震荡, 提供有界非线性
    """
    def __init__(self, hidden_size, expansion_factor=2):
        super().__init__()
        h = hidden_size
        self.value_proj = nn.Linear(h, h * expansion_factor)
        self.gate_proj = nn.Linear(h, h * expansion_factor)
        self.out_proj = nn.Linear(h * expansion_factor, h)

    def forward(self, x):
        v = self.value_proj(x)  # 主值通路
        g = self.gate_proj(x)   # 相位门控
        # WDLM 风格: sin+cos 替代 ReLU, 提供 -1.4~1.4 有界非线性
        modulated = v * (torch.sin(g) + torch.cos(g)) * 0.5
        return self.out_proj(modulated)


# ============================================================
# 改进 3: 干涉式组合 (借鉴 WaveInterference 的多波叠加)
# ============================================================
class InterferenceCombine(nn.Module):
    """
    替代 gen_model 的 4-branch 组合
    借鉴 WDLM: 多个分支视为不同"波", 用干涉权重叠加
    比原始 term1+term2+term3+term4 更有学习空间
    """
    def __init__(self, heads, d_head):
        super().__init__()
        self.heads = heads
        self.d_head = d_head
        hd = heads * d_head

        # 5个分支 (a,b,c,d,e) 各有自己的干涉权重
        self.branch_weights = nn.Parameter(torch.randn(5, heads, d_head) * 0.01)

        # 分支间交叉耦合权重 (稀疏)
        self.cross_coupling = nn.Linear(heads * 5, heads * 5, bias=False)

        # 输出投影
        self.out_proj = nn.Linear(heads * 5, heads, bias=False)

    def forward(self, branches):
        """
        branches: list of 5 tensors [b, d_head, s, heads]
        返回: [b, d_head, s, heads]
        """
        # 加权各分支
        weighted = []
        for i, br in enumerate(branches):
            w = self.branch_weights[i].view(1, self.heads, 1, self.d_head).permute(0, 2, 1, 3)
            # br: [b, d_head, s, heads] + w: [1, 1, heads, d_head]
            weighted.append(br * w.permute(0, 1, 3, 2))

        # 拼接所有分支: [b, d_head, s, heads*5]
        concat = torch.cat(weighted, dim=-1)

        # 交叉耦合
        tmp = concat.permute(0, 2, 1, 3).contiguous()
        tmp = tmp.view(-1, self.heads * 5)
        coupled = self.cross_coupling(tmp)
        coupled = coupled.view(concat.shape[0], concat.shape[2], concat.shape[1], self.heads * 5)
        coupled = coupled.permute(0, 2, 1, 3).contiguous()

        # 输出: 压缩 5*heads → heads
        out = self.out_proj(coupled)
        return out


# ============================================================
# 完整 Enhanced DecoderLayer (集成所有改进)
# ============================================================
class EnhancedDecoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, model_flag="train"):
        super().__init__()
        self.self_attention = MaxStateWithDecay(hidden_size, num_heads, model_flag)
        self.ffn = PhaseGatingFFN(hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, state=None):
        x1, state = self.self_attention(x, state)
        x = self.layer_norm(self.alpha * self.ffn(x1) + (1 - self.alpha) * x)
        return x, state


# ============================================================
# Enhanced OpenASH (集成所有改进)
# ============================================================
class OpenASHEnhanced(nn.Module):
    def __init__(self, voc_size, hidden_size, num_heads, num_layers, model_flag="train"):
        super().__init__()
        self.em = nn.Embedding(voc_size, hidden_size, padding_idx=0)
        self.decoder_layers = nn.ModuleList([
            EnhancedDecoderLayer(hidden_size, num_heads, model_flag)
            for _ in range(num_layers)
        ])
        self.head_score = nn.Linear(hidden_size, voc_size, bias=False)

    def forward(self, x, state=None):
        x = self.em(x)
        if state is None:
            state = [None] * len(self.decoder_layers)
        for i, layer in enumerate(self.decoder_layers):
            x1, state[i] = layer(x, state[i])
            x = x1 + x
        return self.head_score(x), state
