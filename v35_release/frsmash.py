"""
FRSMASH — OpenASH 骨干 + 1 慢尺度记忆

设计思路:
  OpenASH 的 cummax + gen_model 是优秀的 LM 先验 (SFT loss=3.60, 同规模最优)
  但 cummax 单调递增, 无法选择性遗忘 → CopyFirst 失败 (best_loss=1.6)

  HybridFRSM 的慢尺度内容门控可以完美选择性记忆 (CopyFirst@65K=100%)
  但快尺度是简单线性递推, LM 表达力弱

  FRSMASH = OpenASH 骨干 (强 LM) + 1 慢尺度 (强记忆)
  → 目标: LM loss 接近 OpenASH, CopyFirst 接近 HybridFRSM
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# 1. OpenASH 组件 (从 open_ash.py 移植)
# ============================================================
class MaxStateSuper(nn.Module):
    """OpenASH 核心: 多头 cummax + gen_model"""
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

        # gen_model: 5-branch multiplicative interaction
        cat = torch.cat([out, out1, out2, out3, out4], dim=-1)
        combined_g = self.head_linear(cat) * out4
        term1 = out * out1
        term2 = self.alpha1 * out1 + self.alpha2 * out3
        term3 = out * (self.alpha3 * out4 + out3)
        term4 = out1 * (out2 + out4)
        result = term1 + term2 + term3 + term4 + out2 * out4 + combined_g

        out_l = result.transpose(1, 2).contiguous().view(b, s, d)
        return out_l, state


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

    def forward(self, x, state=None):
        x1, state = self.attn(x, state)
        x = self.norm(self.alpha * self.ffn(x1) + (1 - self.alpha) * x)
        return x, state


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

    def forward(self, x):
        """
        x: (B, T) token ids
        返回: (B, T, voc_size) logits
        """
        B, T = x.shape
        D = self.D

        x_emb = self.em(x)  # (B, T, D)

        # ========== 1. OpenASH 骨干 ==========
        h = x_emb
        for layer in self.ash_layers:
            h1, _ = layer(h)  # 训练时不需要传 state
            h = h1 + h  # 残差
        x_ash = self.ash_norm(h)  # (B, T, D)

        # ========== 2. 慢尺度记忆 (分段常数) ==========
        inp_seq = self.mem_input_proj(x_emb)  # (B, T, D)
        h_slow = torch.zeros(B, D, device=x.device)
        H_slow = torch.zeros(B, T, D, device=x.device)
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

        return self.head(fused)

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
        x = self.em(token_id)  # (B, 1, D)

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
