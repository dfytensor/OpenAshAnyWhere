"""
FRSMASH — F-layer 线性递推骨干 + 1 慢尺度记忆

设计思路:
  FRSM V6a 实验证实 content-gated 等变体 PPL 随长度增长 (+1127%)
  F-layer (线性递推 h_t = A·h + B) 是有界系统, PPL 仅 +47%
  gen_model (5-branch multiplicative interaction) 提供强表达力

  FRSMASH = F-layer 线性递推 (强 LM, 有界状态) + gen_model + 慢尺度 (强记忆)
  → 目标: LM loss 接近 OpenASH, PPL 稳定, 记忆接近 HybridFRSM
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# 1. F-layer 线性递推 + gen_model
# ============================================================
class MaxStateSuper(nn.Module):
    """F-layer 线性递推 + gen_model (5-branch)"""
    def __init__(self, dim_size, heads, model_flag="train"):
        super().__init__()
        self.heads = heads
        self.d_head = dim_size // heads
        self.model_flag = model_flag
        self.combined = nn.Linear(dim_size, 4 * dim_size, bias=False)
        self.alpha1 = nn.Parameter(torch.tensor(0.5))
        self.alpha2 = nn.Parameter(torch.tensor(0.5))
        self.alpha3 = nn.Parameter(torch.tensor(0.5))
        self.head_linear = nn.Linear(heads * 5, heads, bias=False)
        # F-layer: 线性递推 (有界)
        self.fast_proj = nn.Linear(dim_size, 4 * dim_size, bias=False)

    @staticmethod
    def _parallel_scan(A, B, h_prev=None):
        """h_t = A_t * h_{{t-1}} + B_t 的并行前缀和"""
        A_s = A.clamp(min=1e-4, max=1.0)
        Acp = torch.cumprod(A_s, dim=1)
        csB = torch.cumsum(B / A_s, dim=1)
        if h_prev is None:
            return Acp * csB
        return Acp * (h_prev.unsqueeze(1) + csB)

    def forward(self, x, state=None):
        """
        x: (B, T, D)
        state: (B, D) or None — F-layer 递推状态
        返回: (B, T, D), (B, D) — 输出和新状态
        """
        b, s, d = x.shape
        combined = self.combined(x).view(b, s, 4, self.heads, -1)
        out, out1, out2, out3 = combined.unbind(2)
        out = out.permute(0, 3, 1, 2)
        out1 = out1.permute(0, 3, 1, 2)
        out2 = out2.permute(0, 3, 1, 2)
        out3 = out3.permute(0, 3, 1, 2)

        # F-layer: 线性递推 → out4 (有界状态)
        fg = self.fast_proj(x).reshape(b, s, 4, d)
        af = torch.sigmoid(fg[..., 0, :])   # 写入门
        ff = torch.sigmoid(fg[..., 1, :])   # forget 门
        i_f = torch.sigmoid(fg[..., 2, :])  # input 门
        cf = torch.tanh(fg[..., 3, :])      # candidate
        A = af * ff + (1 - af)               # 递推系数 ∈ (0, 1]
        B_coeff = af * i_f * cf             # 递推偏置

        H = self._parallel_scan(A, B_coeff, state)
        out4 = H.reshape(b, s, self.heads, self.d_head).permute(0, 3, 1, 2)
        new_state = H[:, -1, :]  # (B, D)

        # gen_model: 5-branch multiplicative interaction
        cat = torch.cat([out, out1, out2, out3, out4], dim=-1)
        combined_g = self.head_linear(cat) * out4
        term1 = out * out1
        term2 = self.alpha1 * out1 + self.alpha2 * out3
        term3 = out * (self.alpha3 * out4 + out3)
        term4 = out1 * (out2 + out4)
        result = term1 + term2 + term3 + term4 + out2 * out4 + combined_g

        out_l = result.transpose(1, 2).contiguous().view(b, s, d)
        return out_l, new_state


class FeedForward(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.ffn1 = nn.Linear(hidden_size, hidden_size)
        self.ffn2 = nn.Linear(hidden_size, hidden_size)
        self.gate = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.ffn2(self.ffn1(x) * self.relu(self.gate(x)))


class ASHDecoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, model_flag="train"):
        super().__init__()
        self.attn = MaxStateSuper(hidden_size, num_heads, model_flag)
        self.ffn = FeedForward(hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, state=None, return_attn_state=False):
        x1, attn_state = self.attn(x, state)
        x = self.norm(self.alpha * self.ffn(x1) + (1 - self.alpha) * x)
        if return_attn_state:
            return x, attn_state
        return x, None


# ============================================================
# 2. 慢尺度记忆 (从 HybridFRSM 移植)
# ============================================================
class SlowMemoryCell(nn.Module):
    """
    内容门控慢记忆 — 选择性写入

    h_new = α * candidate + (1-α) * h_prev
    α = sigmoid(MLP([h_prev; inp]))  ← 内容决定写入强度
    """
    def __init__(self, d_model):
        super().__init__()
        d = d_model
        # 三门
        self.W_forget = nn.Linear(d * 2, d)
        self.W_input  = nn.Linear(d * 2, d)
        self.W_cand   = nn.Linear(d * 2, d)
        nn.init.constant_(self.W_forget.bias, 1.0)
        nn.init.constant_(self.W_input.bias, -2.0)
        # 内容门控
        dh = max(d // 4, 1)
        self.gate = nn.Sequential(
            nn.Linear(d * 2, dh), nn.GELU(),
            nn.Linear(dh, 1), nn.Sigmoid()
        )

    def forward(self, x_t, h_prev):
        c = torch.cat([h_prev, x_t], dim=-1)
        f = torch.sigmoid(self.W_forget(c))
        i = torch.sigmoid(self.W_input(c))
        cand = f * h_prev + i * torch.tanh(self.W_cand(c))
        alpha = self.gate(c).squeeze(-1).unsqueeze(-1)
        return alpha * cand + (1 - alpha) * h_prev


# ============================================================
# 3. FRSMASH — 融合模型
# ============================================================
class FRSMASH(nn.Module):
    """
    FRSMASH = OpenASH backbone + 1 SlowMemory

    架构:
      1. 共享 embedding
      2. OpenASH 多层骨干 (cummax + gen_model + FFN) → 强 LM 特征
      3. 慢尺度记忆 (内容门控, 每 K 步更新) → 选择性长期记忆
      4. 门控融合: per-token 决定依赖 LM 还是记忆

    参数:
        voc_size:     词表大小
        hidden_size:  隐藏维度
        num_heads:    注意力头数
        num_layers:   OpenASH 层数
        K:            慢尺度更新周期 (默认 8)
    """
    def __init__(self, voc_size, hidden_size, num_heads, num_layers, K=8):
        super().__init__()
        self.D = hidden_size
        self.K = K

        # 共享 embedding
        self.em = nn.Embedding(voc_size, hidden_size, padding_idx=0)

        # OpenASH 骨干
        self.ash_layers = nn.ModuleList([
            ASHDecoderLayer(hidden_size, num_heads, "train")
            for _ in range(num_layers)
        ])
        self.ash_norm = nn.LayerNorm(hidden_size)

        # 慢尺度记忆
        self.mem_input_proj = nn.Linear(hidden_size, hidden_size)
        self.slow_cell = SlowMemoryCell(hidden_size)

        # 融合
        self.mem_proj = nn.Linear(hidden_size, hidden_size)
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )
        self.fusion_norm = nn.LayerNorm(hidden_size)

        # 输出
        self.head = nn.Linear(hidden_size, voc_size, bias=False)

    def forward(self, x, return_state=False):
        """
        x: (B, T) token ids
        return_state: 若 True, 返回 (logits, ash_states, h_slow)
        返回: (B, T, voc_size) logits, 或 (logits, ash_states, h_slow)
        """
        B, T = x.shape
        D = self.D

        x_emb = self.em(x)  # (B, T, D)
        model_dtype = self.head.weight.dtype
        x_emb = x_emb.to(dtype=model_dtype)

        # ========== 1. OpenASH 骨干 ==========
        h = x_emb
        ash_states = [] if return_state else None
        for layer in self.ash_layers:
            if return_state:
                layer.attn.model_flag = "infer"
                h1, s = layer(h, state=None, return_attn_state=True)
                ash_states.append(s)
            else:
                h1, _ = layer(h)
            h = h1 + h  # 残差
        x_ash = self.ash_norm(h)  # (B, T, D)

        # ========== 2. 慢尺度记忆 (分段常数) ==========
        inp_seq = self.mem_input_proj(x_emb)  # (B, T, D)
        h_slow = torch.zeros(B, D, device=x.device, dtype=model_dtype)
        H_slow = torch.zeros(B, T, D, device=x.device, dtype=model_dtype)
        prev = 0
        for t in range(0, T, self.K):
            h_slow = self.slow_cell(inp_seq[:, t], h_slow)
            H_slow[:, prev:t+1] = h_slow.unsqueeze(1)
            prev = t + 1
        if prev < T:
            H_slow[:, prev:] = h_slow.unsqueeze(1)
        x_mem = self.mem_proj(H_slow)  # (B, T, D)

        # ========== 3. 门控融合 ==========
        cat = torch.cat([x_ash, x_mem], dim=-1)  # (B, T, 2D)
        gate = self.fusion_gate(cat)  # (B, T, 1)
        fused = self.fusion_norm(
            gate * x_ash + (1 - gate) * x_mem + x_emb
        )

        logits = self.head(fused)
        if return_state:
            return logits, ash_states, h_slow
        return logits

    @torch.no_grad()
    def generate_step(self, token_id, ash_states, h_slow):
        """
        推理单步 O(1)

        token_id: (B, 1)
        ash_states: list of state (每层一个)
        h_slow: (B, D) 慢尺度状态
        返回: logits, new_ash_states, new_h_slow
        """
        B = token_id.size(0)
        x = self.em(token_id).to(dtype=self.head.weight.dtype)  # (B, 1, D)

        # OpenASH 逐层 (用 state 模式)
        h = x
        new_states = []
        for i, layer in enumerate(self.ash_layers):
            layer.attn.model_flag = "infer"
            h1, s = layer.attn(h, ash_states[i])
            h1 = layer.norm(layer.alpha * layer.ffn(h1) + (1 - layer.alpha) * h)
            h = h1 + h
            new_states.append(s)
        x_ash = self.ash_norm(h[:, 0])  # (B, D)

        # 慢尺度
        inp = self.mem_input_proj(x[:, 0])
        h_slow_new = self.slow_cell(inp, h_slow)
        x_mem = self.mem_proj(h_slow_new)  # (B, D)

        # 融合
        cat = torch.cat([x_ash, x_mem], dim=-1)
        gate = self.fusion_gate(cat)
        fused = self.fusion_norm(gate * x_ash + (1 - gate) * x_mem + x[:, 0])
        logits = self.head(fused)

        return logits, new_states, h_slow_new


# ============================================================
# 测试
# ============================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    VOCAB = 23005
    H = 256
    HEADS = 8
    LAYERS = 4

    model = FRSMASH(VOCAB, H, HEADS, LAYERS, K=8).to(device)
    n = sum(p.numel() for p in model.parameters())
    print(f"FRSMASH: {n:,} params")
    print(f"  OpenASH: {LAYERS}L × {HEADS}h × {H}d")
    print(f"  SlowMemory: 1 scale, K=8")

    x = torch.randint(0, VOCAB, (4, 384), device=device)
    logits = model(x)
    print(f"Train output: {logits.shape}")  # (4, 384, 23005)

    # 推理
    token = torch.tensor([[42]], device=device)
    ash_states = [None] * LAYERS
    h_slow = torch.zeros(1, H, device=device)
    for step in range(5):
        logits, ash_states, h_slow = model.generate_step(token, ash_states, h_slow)
        token = logits.argmax(dim=-1, keepdim=True)
        print(f"  Step {step+1}: token={token.item()}")
