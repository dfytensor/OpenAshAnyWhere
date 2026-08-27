"""OpenASH_Stable: 基座 cummax 长序列稳定性修复 — 逐位置状态范数 cap.

out4 是 cummax 状态 [b, heads, s, d_head]; 每个位置的 state 若范数超 R 则缩放到 R.
- 可微 (scale 是状态函数) / 保序 (标量缩放不改变最大次序) / 训练推理一致
"""
import torch
import torch.nn as nn
from open_ash import MaxStateSuper, FeedForward


class MaxStateSuperStable(MaxStateSuper):
    def __init__(self, dim_size, heads, model_flag="train", R=10.0):
        super().__init__(dim_size, heads, model_flag)
        self.R = nn.Parameter(torch.tensor(R))

    def forward(self, x, state=None):
        b, s, d = x.shape
        combined = self.combined(x).view(b, s, 4, self.heads, -1)
        out, out1, out2, out3 = combined.unbind(2)
        out = out.permute(0, 3, 1, 2)
        out1 = out1.permute(0, 3, 1, 2)
        out2 = out2.permute(0, 3, 1, 2)
        out3 = out3.permute(0, 3, 1, 2)

        if state is None:
            out4, _ = torch.cummax(out2, dim=2)
            state = out4[:, :, -1:]
        else:
            out4, _ = torch.cummax(torch.cat([state, out2], dim=2), dim=2)
            if self.model_flag == "train":
                out4 = out4[:, :, 1:]
            else:
                out4 = out4[:, :, -1:]
            state = out4[:, :, -1:]

        # === 稳定性: 逐位置状态范数 cap ===
        n = out4.norm(dim=-1, keepdim=True)
        scale = torch.clamp(self.R / (n + 1e-8), max=1.0)
        out4 = out4 * scale

        cat = torch.cat([out, out1, out2, out3, out4], dim=-1)
        combined_g = self.head_linear(cat) * out4
        term1 = out * out1
        term2 = self.alpha1 * out1 + self.alpha2 * out3
        term3 = out * (self.alpha3 * out4 + out3)
        term4 = out1 * (out2 + out4)
        result = term1 + term2 + term3 + term4 + out2 * out4 + combined_g

        out_l = result.transpose(1, 2).contiguous().view(b, s, d)
        return out_l, state


class DecoderLayerStable(nn.Module):
    def __init__(self, hidden_size, num_heads, model_flag="train", R=10.0):
        super().__init__()
        self.self_attention_linear = MaxStateSuperStable(hidden_size, num_heads, model_flag, R)
        self.ffn = FeedForward(hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, state=None):
        x1, state = self.self_attention_linear(x, state)
        x = self.layer_norm(self.alpha * self.ffn(x1) + (1 - self.alpha) * x)
        return x, state


class OpenASHStable(nn.Module):
    def __init__(self, voc_size, hidden_size, num_heads, num_layers, model_flag="train", R=10.0):
        super().__init__()
        self.em = nn.Embedding(voc_size, hidden_size, padding_idx=0)
        self.decoder_layers = nn.ModuleList(
            [DecoderLayerStable(hidden_size, num_heads, model_flag, R) for _ in range(num_layers)])
        self.head_score = nn.Linear(hidden_size, voc_size, bias=False)

    def forward(self, x, state=None):
        x = self.em(x)
        if state is None:
            state = [None] * len(self.decoder_layers)
        i = 0
        for ii, layer in enumerate(self.decoder_layers):
            x1, state[i] = layer(x, state[i])
            x = x1 + x
            i += 1
        return self.head_score(x), state
