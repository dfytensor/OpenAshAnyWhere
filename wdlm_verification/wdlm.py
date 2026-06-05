"""
波动力学语言模型 (Wave Dynamics Language Model, WDLM)
从博客 https://blog.csdn.net/weixin_32759777/article/details/161548709 验证实现

核心思想：将语言序列视为量子态，用波函数演化来模拟信息处理过程。
完全脱离Transformer的自注意力机制，基于波动力学的第一性原理构建。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# 1. 波函数表示层 (Wave Representation Layer)
# ============================================================
class QuantumStateEncoding(nn.Module):
    """将离散的词嵌入编码为连续波函数"""
    def __init__(self, vocab_size, hidden_dim, n_qubits=8):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        self.amplitude_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.phase_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.frequencies = nn.Parameter(torch.linspace(1.0, 10.0, hidden_dim))

    def forward(self, token_ids):
        """
        [batch, seq_len] -> [batch, seq_len, hidden_dim, 2] (实部, 虚部)
        """
        batch_size, seq_len = token_ids.shape
        device = token_ids.device

        amplitude = self.amplitude_embedding(token_ids).abs()  # 振幅非负
        phase = self.phase_embedding(token_ids)                 # 相位

        positions = torch.arange(self.hidden_dim, device=device).float()
        freq = self.frequencies

        # 波函数构造: psi(x) = amplitude * exp(i * (freq * x + phase))
        # 实部 = amp * cos(freq*x + phase), 虚部 = amp * sin(freq*x + phase)
        arg = freq.unsqueeze(0).unsqueeze(0) * positions.unsqueeze(0).unsqueeze(0) + phase
        # [1, 1, H] * [1, 1, H] + [B, S, H] -> [B, S, H]

        real_part = amplitude * torch.cos(arg)
        imag_part = amplitude * torch.sin(arg)

        wave_function = torch.stack([real_part, imag_part], dim=-1)
        return wave_function  # [batch, seq_len, hidden_dim, 2]


# ============================================================
# 2. 波函数演化层 (Schrodinger Evolution)
# ============================================================
class SchrodingerEvolution(nn.Module):
    """模拟薛定谔方程演化: dψ/dt = -iHψ"""
    def __init__(self, hidden_dim, n_frequencies=8):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 哈密顿量参数化: H = 动能 + 势能
        self.kinetic = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.01)
        self.potential = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.01)
        self.dt = nn.Parameter(torch.tensor(0.1))

        # 非线性势场
        self.nonlinear_potential = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.Tanh(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

    def construct_hamiltonian(self, psi):
        """
        构造哈密顿量: H = p²/2m + V(x) + V_nonlinear(|ψ|²)
        返回: [batch, seq_len, hidden_dim, hidden_dim]
        """
        batch_size, seq_len, hidden_dim, _ = psi.shape
        device = psi.device

        H_base = self.kinetic + self.potential  # [H, H]

        # 非线性势 (基于波函数密度 |ψ|²)
        psi_mag_sq = psi[..., 0]**2 + psi[..., 1]**2  # [B, S, H]
        psi_flat = psi_mag_sq.view(-1, hidden_dim)
        V_nl = self.nonlinear_potential(
            torch.cat([psi_flat, psi_flat**2], dim=-1)
        )  # [B*S, H]
        V_nl = V_nl.view(batch_size, seq_len, hidden_dim)  # [B, S, H]

        # 构造哈密顿量矩阵: 对角元加上非线性势
        H = H_base.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, hidden_dim, hidden_dim).clone()
        # 添加非线性势到对角元
        diag_indices = torch.arange(hidden_dim, device=device)
        H[:, :, diag_indices, diag_indices] += V_nl

        return H  # [batch, seq_len, hidden_dim, hidden_dim]

    def forward(self, psi, steps=1):
        """
        执行波函数演化: ψ(t+dt) = exp(-iH dt) ψ(t)
        使用一阶近似: ψ(t+dt) ≈ ψ(t) - i*dt*Hψ(t)
        """
        batch_size, seq_len, hidden_dim, _ = psi.shape
        device = psi.device

        for _ in range(steps):
            H = self.construct_hamiltonian(psi)  # [B, S, H, H]

            # 展平进行矩阵乘法
            psi_flat = psi.view(batch_size * seq_len, hidden_dim, 2)  # [B*S, H, 2]
            H_flat = H.view(batch_size * seq_len, hidden_dim, hidden_dim)  # [B*S, H, H] (实数)
            psi_real = psi_flat[..., 0]  # [B*S, H]
            psi_imag = psi_flat[..., 1]

            # Hψ (H为实数矩阵)
            H_psi_real = torch.bmm(H_flat, psi_real.unsqueeze(-1)).squeeze(-1)  # [B*S, H]
            H_psi_imag = torch.bmm(H_flat, psi_imag.unsqueeze(-1)).squeeze(-1)

            # 时间演化: ψ(t+dt) = ψ(t) - i*dt*Hψ
            # 实部: psi_real - dt*(-H_psi_imag) = psi_real + dt*H_psi_imag
            # 虚部: psi_imag - dt*(H_psi_real)
            psi_new_real = psi_real + self.dt * H_psi_imag
            psi_new_imag = psi_imag - self.dt * H_psi_real

            psi_flat = torch.stack([psi_new_real, psi_new_imag], dim=-1)
            psi = psi_flat.view(batch_size, seq_len, hidden_dim, 2)

            # 归一化
            psi_norm = torch.norm(psi, dim=-1, keepdim=True).clamp_min(1e-8)
            psi = psi / psi_norm

        return psi


# ============================================================
# 3. 波函数干涉层 (Wave Interference Layer)
# ============================================================
class WaveInterference(nn.Module):
    """波函数干涉：多个波函数叠加产生干涉图案 (替代自注意力机制)"""
    def __init__(self, hidden_dim, n_waves=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_waves = n_waves

        self.interference_weights = nn.Parameter(
            torch.randn(n_waves, hidden_dim, hidden_dim) * 0.01
        )
        self.phase_shift = nn.Parameter(torch.randn(n_waves, hidden_dim) * 0.1)

    def forward(self, psi_list):
        """
        psi_list: list of wave functions, each [batch, seq_len, hidden_dim, 2]
        返回干涉叠加后的波函数
        """
        batch_size, seq_len, hidden_dim, _ = psi_list[0].shape
        device = psi_list[0].device

        psi_interfered = torch.zeros(batch_size, seq_len, hidden_dim, 2, device=device)

        for i, psi in enumerate(psi_list):
            # 应用相位偏移 (相位旋转)
            cos_shift = torch.cos(self.phase_shift[i]).view(1, 1, hidden_dim, 1)
            sin_shift = torch.sin(self.phase_shift[i]).view(1, 1, hidden_dim, 1)

            psi_real = psi[..., 0]
            psi_imag = psi[..., 1]
            psi_rotated_real = psi_real * cos_shift.squeeze(-1) - psi_imag * sin_shift.squeeze(-1)
            psi_rotated_imag = psi_real * sin_shift.squeeze(-1) + psi_imag * cos_shift.squeeze(-1)
            psi_rotated = torch.stack([psi_rotated_real, psi_rotated_imag], dim=-1)

            # 应用干涉权重 (复数线性变换)
            W = self.interference_weights[i]  # [H, H]
            psi_flat = psi_rotated.view(batch_size * seq_len, hidden_dim, 2)

            transformed_real = torch.mm(psi_flat[..., 0], W.T)
            transformed_imag = torch.mm(psi_flat[..., 1], W.T)
            psi_transformed = torch.stack([transformed_real, transformed_imag], dim=-1)
            psi_transformed = psi_transformed.view(batch_size, seq_len, hidden_dim, 2)

            psi_interfered = psi_interfered + psi_transformed

        # 归一化
        norm = torch.norm(psi_interfered, dim=-1, keepdim=True).clamp_min(1e-8)
        psi_interfered = psi_interfered / norm

        return psi_interfered


# ============================================================
# 4. 波函数残差模块 (Wave Residual Block)
# ============================================================
class WaveResidualBlock(nn.Module):
    """波函数残差模块：多个波函数演化 + 干涉"""
    def __init__(self, hidden_dim, n_evolutions=3, n_interference_waves=4):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.evolutions = nn.ModuleList([
            SchrodingerEvolution(hidden_dim) for _ in range(n_evolutions)
        ])

        self.interference = WaveInterference(hidden_dim, n_interference_waves)

        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.Tanh(),
            nn.Linear(hidden_dim * 4, hidden_dim * 2)
        )

    def forward(self, psi):
        """psi: [batch, seq_len, hidden_dim, 2]"""
        residual = psi

        evolved_waves = []
        for evolution in self.evolutions:
            psi = evolution(psi, steps=2)
            evolved_waves.append(psi.clone())

        # 波函数干涉
        psi = self.interference(evolved_waves)

        # 波函数门控 (类似LSTM)
        psi_mag = torch.norm(psi, dim=-1)
        residual_mag = torch.norm(residual, dim=-1)

        combined = torch.cat([psi_mag, residual_mag], dim=-1)
        gate_values = self.gate(combined)
        gate_real, gate_imag = gate_values.chunk(2, dim=-1)
        gate_real = torch.sigmoid(gate_real)
        gate_imag = torch.sigmoid(gate_imag)

        psi_real = psi[..., 0]
        psi_imag = psi[..., 1]
        res_real = residual[..., 0]
        res_imag = residual[..., 1]

        output_real = gate_real * psi_real + (1 - gate_real) * res_real
        output_imag = gate_imag * psi_imag + (1 - gate_imag) * res_imag

        output = torch.stack([output_real, output_imag], dim=-1)
        return output


# ============================================================
# 5. 波函数测量与解码 (Wave Measurement)
# ============================================================
class WaveMeasurement(nn.Module):
    """波函数测量：将波函数坍缩为概率分布"""
    def __init__(self, hidden_dim, vocab_size):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        # 测量算符 (简化：使用全连接层)
        self.phase_measure = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, vocab_size)
        )

    def forward(self, psi, measurement_type="amplitude"):
        """
        测量波函数，得到词的概率分布
        psi: [batch, seq_len, hidden_dim, 2]
        返回: logits [batch, seq_len, vocab_size]
        """
        batch_size, seq_len, hidden_dim, _ = psi.shape

        psi_mag = torch.norm(psi, dim=-1)  # [B, S, H]
        psi_phase = torch.atan2(psi[..., 1], psi[..., 0])  # [B, S, H]

        # 合并振幅、sin(相位)、cos(相位)
        combined = torch.cat([psi_mag, torch.sin(psi_phase), torch.cos(psi_phase)], dim=-1)
        combined = combined.view(batch_size * seq_len, -1)

        logits = self.phase_measure(combined)
        logits = logits.view(batch_size, seq_len, self.vocab_size)

        return logits


# ============================================================
# 6. 波函数注意力机制 (替代自注意力)
# ============================================================
class WaveAttention(nn.Module):
    """基于波函数干涉的注意力机制 (完全不同于Transformer的点积注意力)"""
    def __init__(self, hidden_dim, n_heads=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        assert hidden_dim % n_heads == 0, "hidden_dim must be divisible by n_heads"

        self.wave_sources = nn.Parameter(
            torch.randn(n_heads, self.head_dim, 2) * 0.1
        )

        self.propagation = nn.Parameter(
            torch.randn(n_heads, self.head_dim, self.head_dim) * 0.01
        )

        self.interference_matrix = nn.Parameter(
            torch.randn(n_heads, self.head_dim, self.head_dim) * 0.01
        )

        self.decay_alpha = nn.Parameter(torch.tensor(0.1))
        self.wave_number = nn.Parameter(torch.tensor(0.5))

    def forward(self, psi):
        """
        psi: [batch, seq_len, hidden_dim, 2]
        返回: [batch, seq_len, hidden_dim, 2]
        """
        batch_size, seq_len, hidden_dim, _ = psi.shape
        device = psi.device

        # 分割为多头
        psi_mh = psi.view(batch_size, seq_len, self.n_heads, self.head_dim, 2)

        amplitude = torch.norm(psi_mh, dim=-1)  # [B, S, nh, hd]
        phase = torch.atan2(psi_mh[..., 1], psi_mh[..., 0])

        # 波传播：每个位置接收所有其他位置的波
        distances = torch.arange(seq_len, device=device).float()
        dist_matrix = distances.unsqueeze(1) - distances.unsqueeze(0)  # [S, S]
        # abs 用于衰减(对称)，原始距离用于相位偏移
        decay = torch.exp(-torch.abs(self.decay_alpha) * torch.abs(dist_matrix))  # [S, S]
        phase_shift = dist_matrix * self.wave_number  # [S, S]

        # 扩展形状用于广播: [1, S, S, 1, 1]
        decay = decay.view(1, seq_len, seq_len, 1, 1)
        phase_shift = phase_shift.view(1, seq_len, seq_len, 1, 1)

        # 源: [B, S, nh, hd] -> [B, 1, S, nh, hd]
        src_amp = amplitude.unsqueeze(1)  # [B, 1, S, nh, hd]
        src_phase = phase.unsqueeze(1)

        # 传播后的波
        prop_amp = decay * src_amp  # [B, S, S, nh, hd]
        prop_phase = src_phase + phase_shift

        prop_real = prop_amp * torch.cos(prop_phase)
        prop_imag = prop_amp * torch.sin(prop_phase)
        propagated = torch.stack([prop_real, prop_imag], dim=-1)  # [B, S, S, nh, hd, 2]

        # 在源维度求和（所有来源贡献叠加）
        propagated = torch.sum(propagated, dim=2)  # [B, S, nh, hd, 2]

        # 干涉：原始波 + 传播波
        combined = psi_mh + propagated

        # 合并多头
        output = combined.view(batch_size, seq_len, hidden_dim, 2)

        # 归一化
        norm = torch.norm(output, dim=-1, keepdim=True).clamp_min(1e-8)
        output = output / norm

        return output


# ============================================================
# 7. 波函数正则化损失
# ============================================================
class WaveLanguageModelLoss(nn.Module):
    """波函数语言模型的特殊损失函数 (交叉熵 + 波函数正则化)"""
    def __init__(self, alpha=0.1, beta=0.01):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.ce_loss = nn.CrossEntropyLoss()

    def wave_function_regularization(self, psi):
        """波函数正则化：归一化约束 + 平滑性约束 + 能量最小化"""
        batch_size, seq_len, hidden_dim, _ = psi.shape

        # 1. 归一化损失
        norm = torch.norm(psi, dim=-1)
        norm_loss = torch.mean((norm - 1.0) ** 2)

        # 2. 平滑性损失 (相邻位置波函数应平滑变化)
        if seq_len > 1:
            psi_real = psi[..., 0]
            psi_imag = psi[..., 1]

            grad_real = psi_real[:, 1:, :] - psi_real[:, :-1, :]
            grad_imag = psi_imag[:, 1:, :] - psi_imag[:, :-1, :]

            smoothness_loss = torch.mean(grad_real ** 2) + torch.mean(grad_imag ** 2)
        else:
            smoothness_loss = torch.tensor(0.0, device=psi.device)

        # 3. 能量最小化
        energy_loss = smoothness_loss

        return norm_loss + self.beta * smoothness_loss + 0.5 * energy_loss

    def forward(self, logits, targets, psi=None):
        ce_loss = self.ce_loss(logits.view(-1, logits.size(-1)), targets.view(-1))

        if psi is not None:
            wave_reg = self.wave_function_regularization(psi)
            total_loss = ce_loss + self.alpha * wave_reg
        else:
            total_loss = ce_loss

        return total_loss, ce_loss


# ============================================================
# 8. 波函数优化器
# ============================================================
class WaveOptimizer:
    """结合标准优化和波函数特定约束的优化器"""
    def __init__(self, model, lr=1e-4, wave_lr=1e-3):
        self.model = model

        wave_params = []
        other_params = []

        for name, param in model.named_parameters():
            if 'frequency' in name or 'phase' in name or 'amplitude' in name:
                wave_params.append(param)
            else:
                other_params.append(param)

        self.optimizer = torch.optim.AdamW([
            {'params': wave_params, 'lr': wave_lr},
            {'params': other_params, 'lr': lr}
        ])

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )

    def step(self):
        self.optimizer.step()
        self.scheduler.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def enforce_wave_constraints(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if 'amplitude' in name:
                    param.data.clamp_(min=0.0)
                elif 'frequency' in name:
                    param.data.clamp_(min=0.1)


# ============================================================
# 9. 基础版 WDLM
# ============================================================
class WaveDynamicsLanguageModel(nn.Module):
    """完整波动力学语言模型 (基础版)"""
    def __init__(self, vocab_size, hidden_dim=512, num_layers=12,
                 n_qubits=8, n_waves=4):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.wave_encoder = QuantumStateEncoding(vocab_size, hidden_dim, n_qubits)
        self.wave_layers = nn.ModuleList([
            WaveResidualBlock(hidden_dim, n_evolutions=3, n_interference_waves=n_waves)
            for _ in range(num_layers)
        ])
        self.measurement = WaveMeasurement(hidden_dim, vocab_size)

        self.bos_wave = nn.Parameter(torch.randn(1, 1, hidden_dim, 2) * 0.01)
        self.eos_wave = nn.Parameter(torch.randn(1, 1, hidden_dim, 2) * 0.01)

    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape

        psi = self.wave_encoder(input_ids)  # [B, S, H, 2]

        for layer in self.wave_layers:
            psi = layer(psi)
            psi_norm = torch.norm(psi, dim=-1, keepdim=True).clamp_min(1e-8)
            psi = psi / psi_norm

        logits = self.measurement(psi, measurement_type="amplitude")
        return logits, psi

    def generate(self, input_ids, max_length=100, temperature=1.0, top_k=50):
        self.eval()
        generated = input_ids

        for _ in range(max_length):
            if generated.size(1) > 512:
                context = generated[:, -512:]
            else:
                context = generated

            with torch.no_grad():
                logits, _ = self.forward(context)
                next_token_logits = logits[:, -1, :] / temperature

            if top_k is not None:
                top_k = min(top_k, next_token_logits.size(-1))
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k, dim=-1)
                min_val = top_k_logits[:, -1].unsqueeze(-1)
                mask = next_token_logits < min_val
                next_token_logits = next_token_logits.masked_fill(mask, float('-inf'))

            probabilities = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)

            generated = torch.cat([generated, next_token], dim=-1)

            if (next_token == 2).any():
                break

        return generated


# ============================================================
# 10. 增强版 WDLM (带波函数注意力)
# ============================================================
class EnhancedWaveDynamicsLM(nn.Module):
    """增强版波动力学语言模型：结合波函数注意力机制"""
    def __init__(self, vocab_size, hidden_dim=512, num_layers=12,
                 n_heads=8, n_qubits=8):
        super().__init__()

        self.wave_encoder = QuantumStateEncoding(vocab_size, hidden_dim, n_qubits)
        self.wave_attention = WaveAttention(hidden_dim, n_heads)
        self.wave_layers = nn.ModuleList([
            WaveResidualBlock(hidden_dim) for _ in range(num_layers)
        ])
        self.measurement = WaveMeasurement(hidden_dim, vocab_size)

    def forward(self, input_ids):
        psi = self.wave_encoder(input_ids)
        psi = self.wave_attention(psi)

        for layer in self.wave_layers:
            psi = layer(psi)
            norm = torch.norm(psi, dim=-1, keepdim=True).clamp_min(1e-8)
            psi = psi / norm

        logits = self.measurement(psi)
        return logits, psi
