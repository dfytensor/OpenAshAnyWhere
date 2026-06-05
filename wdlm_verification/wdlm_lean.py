"""
WDLM-Lean: 借鉴 OpenASH 设计哲学的精简版 WDLM
  - 单 Embedding + sin/cos 分拆 (省 50% 编码参数)
  - 单层 1 次演化 (省 6x 计算)
  - Linear(H, V) 直接输出 (省 80% 输出参数)
  - cummax 风格特征交互 (替代 O(S^2) WaveAttention)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantumStateEncodingLean(nn.Module):
    """单 Embedding, 自动拆分为振幅/相位"""
    def __init__(self, vocab_size, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.frequencies = nn.Parameter(torch.linspace(1.0, 10.0, hidden_dim))

    def forward(self, token_ids):
        B, S = token_ids.shape
        H = self.frequencies.shape[0]
        device = token_ids.device

        x = self.embedding(token_ids)  # [B, S, H]
        amplitude = x.abs()            # 直接振幅
        phase = x                      # 原始值当相位

        positions = torch.arange(H, device=device).float()
        arg = self.frequencies.view(1, 1, H) * positions.view(1, 1, H) + phase
        real = amplitude * torch.cos(arg)
        imag = amplitude * torch.sin(arg)
        return torch.stack([real, imag], dim=-1)  # [B, S, H, 2]


class LeanSchrodingerEvolution(nn.Module):
    """OpenASH 风格: 单步演化, 对角分解, 无冗余"""
    def __init__(self, hidden_dim):
        super().__init__()
        H = hidden_dim
        self.H_base = nn.Parameter(torch.randn(H, H) * 0.01)
        self.dt = nn.Parameter(torch.tensor(0.1))
        self.nonlinear_mlp = nn.Sequential(
            nn.Linear(H, H * 2), nn.Tanh(), nn.Linear(H * 2, H)
        )

    def forward(self, psi):
        B, S, H, _ = psi.shape

        # 对角势 V_nl
        psi_mag_sq = psi[..., 0]**2 + psi[..., 1]**2  # [B, S, H]
        V_nl = self.nonlinear_mlp(psi_mag_sq.view(-1, H)).view(B, S, H)

        # H@psi = H_base@psi + V_nl*psi
        psi_r = psi[..., 0].view(B * S, H)
        psi_i = psi[..., 1].view(B * S, H)
        V_f = V_nl.view(B * S, H)

        Hpsi_r = torch.mm(psi_r, self.H_base.T) + V_f * psi_r
        Hpsi_i = torch.mm(psi_i, self.H_base.T) + V_f * psi_i

        new_r = psi_r + self.dt * Hpsi_i
        new_i = psi_i - self.dt * Hpsi_r

        psi = torch.stack([new_r.view(B, S, H), new_i.view(B, S, H)], dim=-1)
        inv = torch.rsqrt(psi[..., 0]**2 + psi[..., 1]**2 + 1e-8).unsqueeze(-1)
        return psi * inv


class CummaxFeatureMix(nn.Module):
    """
    借鉴 OpenASH gen_model 的多分支交叉交互
    原版: 5 分支 (a,b,c,d,e), 4 个乘加 term + head_linear
    移植到 WDLM: 4 分支 + cummax(e), 5 个乘加 term, 直接输出投影
    """
    def __init__(self, hidden_dim):
        super().__init__()
        H = hidden_dim
        # 合并投影 → 4 分支 × 复数 (同 OpenASH 的 combined 设计)
        self.combined = nn.Linear(H * 2, H * 8, bias=False)
        # 可学习交互系数
        self.alpha1 = nn.Parameter(torch.tensor(0.5))
        self.alpha2 = nn.Parameter(torch.tensor(0.5))
        self.alpha3 = nn.Parameter(torch.tensor(0.5))
        # 输出投影: 5 分支汇总
        self.out_proj = nn.Linear(H * 10, H * 2, bias=False)

    def forward(self, psi):
        B, S, H, _ = psi.shape

        # 投影到 4 分支: [B, S, H*8] → [B, S, 4, H, 2]
        flat = torch.cat([psi[..., 0], psi[..., 1]], dim=-1)
        b_tmp = self.combined(flat).view(B, S, 4, H, 2)
        a = b_tmp[:, :, 0]
        b = b_tmp[:, :, 1]
        c = b_tmp[:, :, 2]
        d = b_tmp[:, :, 3]

        # 对 c 的实部做 cummax → e (同 OpenASH out4)
        c_r = c[..., 0]
        e_r, _ = torch.cummax(c_r, dim=1)
        e = torch.stack([e_r, c[..., 1]], dim=-1)

        # term1 = a * b
        t1 = torch.stack([a[..., 0] * b[..., 0], a[..., 1] * b[..., 1]], dim=-1)
        # term2 = alpha1*b + alpha2*d
        t2 = torch.stack([
            self.alpha1 * b[..., 0] + self.alpha2 * d[..., 0],
            self.alpha1 * b[..., 1] + self.alpha2 * d[..., 1],
        ], dim=-1)
        # term3 = a * (alpha3*e + d)
        t3 = torch.stack([
            a[..., 0] * (self.alpha3 * e[..., 0] + d[..., 0]),
            a[..., 1] * (self.alpha3 * e[..., 1] + d[..., 1]),
        ], dim=-1)
        # term4 = b * (c + e)
        t4 = torch.stack([
            b[..., 0] * (c[..., 0] + e[..., 0]),
            b[..., 1] * (c[..., 1] + e[..., 1]),
        ], dim=-1)
        # term5 = c * e
        t5 = torch.stack([c[..., 0] * e[..., 0], c[..., 1] * e[..., 1]], dim=-1)

        # 汇总: concat 5 分支 → Linear 投影
        cat = torch.cat([t1, t2, t3, t4, t5], dim=-1)  # [B,S,H,10]
        result = self.out_proj(cat.view(B, S, H * 10)).view(B, S, H, 2)

        # 归一化
        inv = torch.rsqrt(result[..., 0]**2 + result[..., 1]**2 + 1e-8).unsqueeze(-1)
        return result * inv


class LeanWaveBlock(nn.Module):
    """OpenASH 风格: 1 次演化 + 1 次 cummax 交互"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.evolution = LeanSchrodingerEvolution(hidden_dim)
        self.mix = CummaxFeatureMix(hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2)
        )

    def forward(self, psi):
        residual = psi
        psi = self.evolution(psi)
        psi = self.mix(psi)

        # 门控残差
        mag = torch.sqrt(psi[..., 0]**2 + psi[..., 1]**2 + 1e-8)
        res_mag = torch.sqrt(residual[..., 0]**2 + residual[..., 1]**2 + 1e-8)
        gv = self.gate(torch.cat([mag, res_mag], dim=-1))
        gr, gi = torch.sigmoid(gv).chunk(2, dim=-1)

        out_r = gr * psi[..., 0] + (1 - gr) * residual[..., 0]
        out_i = gi * psi[..., 1] + (1 - gi) * residual[..., 1]
        return torch.stack([out_r, out_i], dim=-1)


class WaveDynamicsLM_Lean(nn.Module):
    """精简 WDLM: 参数少, 速度快, 保留波动力学校心"""
    def __init__(self, vocab_size, hidden_dim=128, num_layers=4):
        super().__init__()
        self.wave_encoder = QuantumStateEncodingLean(vocab_size, hidden_dim)
        self.wave_blocks = nn.ModuleList([
            LeanWaveBlock(hidden_dim) for _ in range(num_layers)
        ])
        # OpenASH 风格直出: Linear(H, V)
        self.head_score = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, input_ids):
        psi = self.wave_encoder(input_ids)

        for block in self.wave_blocks:
            psi = block(psi)

        # 振幅作为 logits 输入 (丢弃相位)
        mag = torch.sqrt(psi[..., 0]**2 + psi[..., 1]**2 + 1e-8)
        logits = self.head_score(mag)
        return logits, psi
