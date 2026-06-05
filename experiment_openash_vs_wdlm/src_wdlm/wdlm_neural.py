# ============================================================
# WDLM-Turbo (Neural Wave Edition)
# No Complex Numbers. Pure Real-Valued Neural Simulation.
# ============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ============================================================
# 1. Quantum State Encoding (Amplitude + Phase)
# ============================================================
class QuantumStateEncoding(nn.Module):
    """Single embedding → Linear projection (no sin/cos)"""
    def __init__(self, vocab_size, hidden_dim):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, hidden_dim)

    def forward(self, token_ids):
        return self.emb(token_ids)  # [B,S,H] — pure linear, no trig


class NeuralWaveStep(nn.Module):
    """Linear-predicted rotation (no sin/cos) + amplitude gate"""
    def __init__(self, hidden_dim):
        super().__init__()
        H = hidden_dim
        self.proj = nn.Linear(H, H * 3, bias=False)  # H→3H: d_amp, gate, 4 rotation params

    def forward(self, psi):
        # psi is [B,S,H] (pure real now)
        h = self.proj(psi)  # [B,S,H*3]
        d_amp, gate, rot_a = h.chunk(3, dim=-1)
        # rot_b = rot_a  # simplified 2x2 rotation: [a, a] → rotation-like transform
        psi_new = psi * rot_a + gate * d_amp  # linear rotation + gated amplitude
        return psi_new + psi  # residual


class WaveInterference(nn.Module):
    """Pure Linear feature mixing (no trig, no [H,2] stack)"""
    def __init__(self, hidden_dim):
        super().__init__()
        H = hidden_dim
        self.proj1 = nn.Linear(H, H, bias=False)
        self.proj2 = nn.Linear(H, H, bias=False)

    def forward(self, psi):
        a = self.proj1(psi)
        b = self.proj2(psi)
        return a * b


class GenModelMix(nn.Module):
    """5-branch cummax + gen_model multiplicative interaction (from OpenASH MaxStateSuper)"""
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

        # cummax with state
        if state is None:
            e, _ = torch.cummax(c, dim=1)
            state = e[:, -1:, :]
        else:
            e, _ = torch.cummax(torch.cat([state, c], dim=1), dim=1)
            e = e[:, 1:, :]
            state = e[:, -1:, :]

        # 5-branch gen_model
        t1 = a * b
        t2 = self.alpha1 * b + self.alpha2 * d
        t3 = a * (self.alpha3 * e + d)
        t4 = b * (c + e)
        t5 = c * e
        return self.out_proj(torch.cat([t1, t2, t3, t4, t5], dim=-1)), state


# ============================================================
# 4. Residual Block
# ============================================================
class WaveResidualBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.step = NeuralWaveStep(hidden_dim)
        self.inter = WaveInterference(hidden_dim)
        self.gen = GenModelMix(hidden_dim)    # 5-branch multiplicative mixing
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, psi, state=None):
        residual = psi
        psi = self.step(psi)
        psi = self.inter(psi)
        psi, state = self.gen(psi, state)     # gen_model with cummax state
        return self.norm(self.alpha * psi + (1 - self.alpha) * residual), state


# ============================================================
# 5. FFT-Based Wave Attention
# ============================================================
class WaveAttentionFFT(nn.Module):
    def __init__(self, hidden_dim, n_heads=8):
        super().__init__()
        assert hidden_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, psi):
        """
        psi: [B, S, H, 2]  (real, imag)
        """
        B, S, H, _ = psi.shape

        # → 复数张量 [B, S, H]
        psi_c = torch.view_as_complex(psi)

        # → [B, n_heads, S, head_dim]
        psi_c = psi_c.view(B, S, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # ★ 对序列维度做 FFT（这才是正经复数 FFT）
        psi_f = torch.fft.fft(psi_c, dim=2)

        # 可学习频域缩放
        psi_f = psi_f * self.scale

        # IFFT 回来
        psi_c = torch.fft.ifft(psi_f, dim=2)

        # 回到 [B, S, H, 2]
        psi_c = psi_c.permute(0, 2, 1, 3).contiguous().view(B, S, H)
        return torch.view_as_real(psi_c)

# ============================================================
# 6. Measurement Head
# ============================================================
class WaveMeasurement(nn.Module):
    def __init__(self, hidden_dim, vocab_size):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, x):
        return self.proj(x)  # x is [B,S,H]


class WaveDynamicsLanguageModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim=512, num_layers=12):
        super().__init__()
        self.encoder = QuantumStateEncoding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList([
            WaveResidualBlock(hidden_dim) for _ in range(num_layers)
        ])
        self.head = WaveMeasurement(hidden_dim, vocab_size)

    def forward(self, input_ids, state=None):
        x = self.encoder(input_ids)
        if state is None:
            state = [None] * len(self.layers)
        for i, layer in enumerate(self.layers):
            x, state[i] = layer(x, state[i])
        logits = self.head(x)
        return logits, state


# ============================================================
# 8. Generation
# ============================================================
@torch.no_grad()
def generate(model, input_ids, max_new=50, temp=1.0, top_k=50):
    model.eval()
    for _ in range(max_new):
        ctx = input_ids[:, -512:]
        logits, _ = model(ctx)
        logits = logits[:, -1] / temp

        if top_k:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits = logits.masked_fill(logits < v[:, [-1]], float('-inf'))

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1)
        input_ids = torch.cat([input_ids, next_token], dim=1)

    return input_ids


# ============================================================
# Test
# ============================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = WaveDynamicsLanguageModel(
        vocab_size=32000,
        hidden_dim=256,
        num_layers=6
    ).to(device)

    x = torch.randint(0, 32000, (2, 128)).to(device)

    logits, _ = model(x)
    print("Logits:", logits.shape)

    gen = generate(model, x, max_new=10)
    print("Generated:", gen.shape)