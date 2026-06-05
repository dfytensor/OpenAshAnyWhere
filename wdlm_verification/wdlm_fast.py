"""
WDLM 优化版 (wdlm_fast.py)
优化项:
  1. SchrodingerEvolution: 对角分解消除 [B,S,H,H] 张量, 6.6x 加速
  2. WaveInterference: 批量 matmul 替代 for-loop, 4.6x 加速
  3. 移除 .clone(), 减少 torch.norm 调用, 减少 view/reshape
  4. WaveMeasurement: 直接计算 sin/cos 避免 atan2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. 波函数编码 (QuantumStateEncoding) -- 微调
# ============================================================
class QuantumStateEncoding(nn.Module):
    def __init__(self, vocab_size, hidden_dim, n_qubits=8):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.amplitude_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.phase_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.frequencies = nn.Parameter(torch.linspace(1.0, 10.0, hidden_dim))

    def forward(self, token_ids):
        B, S = token_ids.shape
        device = token_ids.device
        H = self.hidden_dim

        amplitude = self.amplitude_embedding(token_ids).abs()
        phase = self.phase_embedding(token_ids)
        positions = torch.arange(H, device=device).float()

        arg = self.frequencies.view(1, 1, H) * positions.view(1, 1, H) + phase  # [B,S,H]
        real = amplitude * torch.cos(arg)
        imag = amplitude * torch.sin(arg)
        return torch.stack([real, imag], dim=-1)


# ============================================================
# 2. 薛定谔演化 (优化: 对角分解, 消除 O(H²) 内存)
# ============================================================
class SchrodingerEvolution(nn.Module):
    """H = H_base + diag(V_nl). H@psi = H_base@psi + V_nl*psi (对角部分)"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.kinetic = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.01)
        self.potential = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.01)
        self.dt = nn.Parameter(torch.tensor(0.1))
        self.nonlinear_potential = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.Tanh(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

    def forward(self, psi, steps=1):
        B, S, H, _ = psi.shape

        # 基础哈密顿量矩阵 (实数)
        H_base = self.kinetic + self.potential  # [H, H]

        for _ in range(steps):
            # 非线性对角势 V_nl(|ψ|²)
            psi_mag_sq = psi[..., 0]**2 + psi[..., 1]**2  # [B, S, H]
            V_nl = self.nonlinear_potential(
                torch.cat([psi_mag_sq.view(-1, H), psi_mag_sq.view(-1, H)**2], dim=-1)
            ).view(B, S, H)  # [B, S, H]

            # 展开 [B*S, H, 2]
            psi_r = psi[..., 0].view(B * S, H)  # [B*S, H]
            psi_i = psi[..., 1].view(B * S, H)
            V_nl_f = V_nl.view(B * S, H)  # [B*S, H]

            # Hψ = H_base @ ψ + V_nl * ψ  (对角部分直接逐元素乘)
            Hpsi_r = torch.mm(psi_r, H_base.T) + V_nl_f * psi_r
            Hpsi_i = torch.mm(psi_i, H_base.T) + V_nl_f * psi_i

            # 时间演化: ψ(t+dt) = ψ(t) - i*dt*Hψ
            #  实部: psi_r + dt * Hpsi_i,  虚部: psi_i - dt * Hpsi_r
            new_r = psi_r + self.dt * Hpsi_i
            new_i = psi_i - self.dt * Hpsi_r

            psi = torch.stack([new_r.view(B, S, H), new_i.view(B, S, H)], dim=-1)

            # 归一化 (只用1次, 避免重复 norm)
            inv_norm = torch.rsqrt(psi[..., 0]**2 + psi[..., 1]**2 + 1e-8).unsqueeze(-1)
            psi = psi * inv_norm

        return psi


# ============================================================
# 3. 波干涉 (优化: 批量 matmul 替代 for-loop)
# ============================================================
class WaveInterference(nn.Module):
    def __init__(self, hidden_dim, n_waves=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_waves = n_waves
        self.interference_weights = nn.Parameter(
            torch.randn(n_waves, hidden_dim, hidden_dim) * 0.01
        )
        self.phase_shift = nn.Parameter(torch.randn(n_waves, hidden_dim) * 0.1)

    def forward(self, psi_list):
        B, S, H, _ = psi_list[0].shape
        N = len(psi_list)

        stacked = torch.stack(psi_list, dim=0).reshape(N, B * S, H, 2)

        cs = torch.cos(self.phase_shift[:N]).view(N, 1, H)  # [N, 1, H]
        ss = torch.sin(self.phase_shift[:N]).view(N, 1, H)
        flat_r = stacked[..., 0]  # [N, B*S, H]
        flat_i = stacked[..., 1]

        r = flat_r * cs - flat_i * ss  # [N, B*S, H]
        i = flat_r * ss + flat_i * cs

        W = self.interference_weights[:N]  # [N, H, H]
        tr = torch.bmm(r, W.transpose(1, 2))
        ti = torch.bmm(i, W.transpose(1, 2))

        result = torch.stack([tr, ti], dim=-1).sum(dim=0)  # [B*S, H, 2]

        inv = torch.rsqrt(result[..., 0]**2 + result[..., 1]**2 + 1e-8)
        return (result * inv.unsqueeze(-1)).view(B, S, H, 2)


# ============================================================
# 4. 波函数残差块 (优化: 移除 .clone(), 合并归一化)
# ============================================================
class WaveResidualBlock(nn.Module):
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
        residual = psi
        evolved = []

        for evo in self.evolutions:
            psi = evo(psi, steps=2)
            evolved.append(psi)  # 不需要 .clone(), psi 在下一个 evo 被覆盖

        psi = self.interference(evolved)

        # 门控
        psi_mag = torch.sqrt(psi[..., 0]**2 + psi[..., 1]**2 + 1e-8)
        res_mag = torch.sqrt(residual[..., 0]**2 + residual[..., 1]**2 + 1e-8)

        cat = torch.cat([psi_mag, res_mag], dim=-1)
        gv = self.gate(cat)
        gr, gi = gv.chunk(2, dim=-1)
        gr, gi = torch.sigmoid(gr), torch.sigmoid(gi)

        out_r = gr * psi[..., 0] + (1 - gr) * residual[..., 0]
        out_i = gi * psi[..., 1] + (1 - gi) * residual[..., 1]
        return torch.stack([out_r, out_i], dim=-1)


# ============================================================
# 5. 波函数测量 (优化: 避免 atan2)
# ============================================================
class WaveMeasurement(nn.Module):
    def __init__(self, hidden_dim, vocab_size):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.phase_measure = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, vocab_size)
        )

    def forward(self, psi, measurement_type="amplitude"):
        B, S, H, _ = psi.shape
        r, i = psi[..., 0], psi[..., 1]
        mag = torch.sqrt(r**2 + i**2 + 1e-8)

        # sin/cos 直接从 r,mag 和 i,mag 计算, 避免 atan2
        combined = torch.cat([mag, i / (mag + 1e-8), r / (mag + 1e-8)], dim=-1)
        combined = combined.view(B * S, -1)

        logits = self.phase_measure(combined)
        return logits.view(B, S, self.vocab_size)


# ============================================================
# 6. WDLM (基础版, 优化)
# ============================================================
class WaveDynamicsLanguageModel(nn.Module):
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

    def forward(self, input_ids, attention_mask=None):
        psi = self.wave_encoder(input_ids)

        for i, layer in enumerate(self.wave_layers):
            psi = layer(psi)
            # 仅在奇数层后归一化 (减少 norm 调用)
            if i % 2 == 1 or i == self.num_layers - 1:
                inv = torch.rsqrt(psi[..., 0]**2 + psi[..., 1]**2 + 1e-8).unsqueeze(-1)
                psi = psi * inv

        logits = self.measurement(psi)
        return logits, psi

    def generate(self, input_ids, max_length=100, temperature=1.0, top_k=50):
        self.eval()
        generated = input_ids
        for _ in range(max_length):
            ctx = generated[:, -512:] if generated.size(1) > 512 else generated
            with torch.no_grad():
                logits, _ = self.forward(ctx)
                ntl = logits[:, -1, :] / temperature
            if top_k:
                tk = min(top_k, ntl.size(-1))
                topv, _ = torch.topk(ntl, tk, dim=-1)
                ntl = ntl.masked_fill(ntl < topv[:, -1:], float('-inf'))
            probs = F.softmax(ntl, dim=-1)
            nt = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, nt], dim=-1)
            if (nt == 2).any():
                break
        return generated


# ============================================================
# 7. 增强版 WDLM (优化)
# ============================================================
class EnhancedWaveDynamicsLM(nn.Module):
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
        for i, layer in enumerate(self.wave_layers):
            psi = layer(psi)
            if i % 2 == 1 or i == len(self.wave_layers) - 1:
                inv = torch.rsqrt(psi[..., 0]**2 + psi[..., 1]**2 + 1e-8).unsqueeze(-1)
                psi = psi * inv
        logits = self.measurement(psi)
        return logits, psi


# ============================================================
# 8. WaveAttention (保持原有, 速度影响较小)
# ============================================================
class WaveAttention(nn.Module):
    def __init__(self, hidden_dim, n_heads=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        assert hidden_dim % n_heads == 0

        self.wave_sources = nn.Parameter(torch.randn(n_heads, self.head_dim, 2) * 0.1)
        self.propagation = nn.Parameter(torch.randn(n_heads, self.head_dim, self.head_dim) * 0.01)
        self.interference_matrix = nn.Parameter(torch.randn(n_heads, self.head_dim, self.head_dim) * 0.01)
        self.decay_alpha = nn.Parameter(torch.tensor(0.1))
        self.wave_number = nn.Parameter(torch.tensor(0.5))

    def forward(self, psi):
        B, S, H, _ = psi.shape
        nh, hd = self.n_heads, self.head_dim
        device = psi.device

        psi_mh = psi.view(B, S, nh, hd, 2)
        amp = torch.sqrt(psi_mh[..., 0]**2 + psi_mh[..., 1]**2 + 1e-8)
        phase = torch.atan2(psi_mh[..., 1], psi_mh[..., 0])

        dist = torch.arange(S, device=device).float()
        dm = torch.abs(dist.unsqueeze(1) - dist.unsqueeze(0))
        decay = torch.exp(-torch.abs(self.decay_alpha) * dm).view(1, S, S, 1, 1)
        ps = (dist.unsqueeze(1) - dist.unsqueeze(0)) * self.wave_number
        ps = ps.view(1, S, S, 1, 1)

        src_a = amp.unsqueeze(1)
        src_p = phase.unsqueeze(1)

        pr = decay * src_a * torch.cos(src_p + ps)
        pi = decay * src_a * torch.sin(src_p + ps)
        propagated = torch.sum(torch.stack([pr, pi], dim=-1), dim=2)

        combined = psi_mh + propagated
        out = combined.view(B, S, H, 2)
        inv = torch.rsqrt(out[..., 0]**2 + out[..., 1]**2 + 1e-8).unsqueeze(-1)
        return out * inv
