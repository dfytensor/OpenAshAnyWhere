"""
WDLM-Real: 用 Linear 模拟所有复数运算, 彻底消除 [H,2] 表示
  - 无 sin/cos 显式构造
  - 无分离的实部/虚部路径
  - 全部实数 [B,S,H] 操作, 与 OpenASH 一致
  - Linear 隐式学习"复数"变换
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RealWaveEncoder(nn.Module):
    """1 个 Embedding, 直接输出 (no extra projection)"""
    def __init__(self, vocab_size, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)

    def forward(self, token_ids):
        return self.embedding(token_ids)


class RealEvolution(nn.Module):
    """sin/cos 门控演化 (借鉴 OpenASH_V2 PhaseGateFFN)"""
    def __init__(self, hidden_dim):
        super().__init__()
        H = hidden_dim
        self.evo_kernel = nn.Linear(H, H, bias=False)
        self.evo_gate = nn.Linear(H, H, bias=False)
        self.dt = nn.Parameter(torch.tensor(0.1))

    def forward(self, psi):
        h_psi = self.evo_kernel(psi)
        g = self.evo_gate(psi)
        # PhaseGate: sin+cos 替代 Tanh/ReLU 非线性
        nonlinear = h_psi * (torch.sin(g) + torch.cos(g)) * 0.5
        return psi + self.dt * nonlinear


class MultiBranchCummax(nn.Module):
    """5 分支 cummax, 含 state 增量模式"""
    def __init__(self, hidden_dim):
        super().__init__()
        H = hidden_dim
        self.combined = nn.Linear(H, H * 4, bias=False)
        self.alpha1 = nn.Parameter(torch.tensor(0.5))
        self.alpha2 = nn.Parameter(torch.tensor(0.5))
        self.alpha3 = nn.Parameter(torch.tensor(0.5))
        self.out_proj = nn.Linear(H * 5, H, bias=False)

    def forward(self, x, state=None):
        B, S, H = x.shape
        br = self.combined(x).view(B, S, 4, H)
        a, b, c, d = br[:, :, 0], br[:, :, 1], br[:, :, 2], br[:, :, 3]

        if state is None:
            e, _ = torch.cummax(c, dim=1)
            state = e[:, -1:, :]
        else:
            e, _ = torch.cummax(torch.cat([state, c], dim=1), dim=1)
            e = e[:, 1:, :]
            state = e[:, -1:, :]

        t1 = a * b
        t2 = self.alpha1 * b + self.alpha2 * d
        t3 = a * (self.alpha3 * e + d)
        t4 = b * (c + e)
        t5 = c * e
        return self.out_proj(torch.cat([t1, t2, t3, t4, t5], dim=-1)), state


class RealWaveBlock(nn.Module):
    """1 次演化 + 1 次 cummax 交互 + state 支持"""
    def __init__(self, hidden_dim, evo_steps=3):
        super().__init__()
        H = hidden_dim
        self.evolution = RealEvolution(H)
        self.mix = MultiBranchCummax(H)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.norm = nn.LayerNorm(H)
        self.evo_steps = evo_steps

    def forward(self, x, state=None):
        residual = x
        for _ in range(self.evo_steps):
            x = self.evolution(x)
        x, state = self.mix(x, state)
        return self.norm(self.alpha * x + (1 - self.alpha) * residual), state


class WaveDynamicsLM_Real(nn.Module):
    """完全实数 WDLM: 无复数, 无 sin/cos, 全 Linear 操作"""
    def __init__(self, vocab_size, hidden_dim=128, num_layers=4, evo_steps=3):
        super().__init__()
        self.encoder = RealWaveEncoder(vocab_size, hidden_dim)
        self.blocks = nn.ModuleList([
            RealWaveBlock(hidden_dim, evo_steps) for _ in range(num_layers)
        ])
        self.head_score = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, input_ids, state=None):
        x = self.encoder(input_ids)
        if state is None:
            state = [None] * len(self.blocks)
        for i, block in enumerate(self.blocks):
            x1, state[i] = block(x, state[i])
            x = x1 + x
        return self.head_score(x), x, state
