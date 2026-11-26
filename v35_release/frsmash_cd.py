"""
FRSMASH-CD — OpenASH 骨干 + 慢尺度记忆 + Cap/Decay 状态稳定

相对 FRSMASH 的改进:
  1. SlowMemoryCell 输出后加 state cap + decay
  2. 推理时每步对慢状态做 norm capping，防止长期累积爆炸
  3. 训练时 cap/decay 默认关闭 (cap=None)，推理时可通过参数开启
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# 1. OpenASH 组件
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
# 2. 慢尺度记忆 + Cap/Decay
# ============================================================
class SlowMemoryCellCD(nn.Module):
    """
    内容门控慢记忆 + 可选 Cap/Decay 状态稳定

    h_new = α * candidate + (1-α) * h_prev
    → norm cap (if > cap) → decay
    """
    def __init__(self, d_model):
        super().__init__()
        d = d_model
        self.W_forget = nn.Linear(d * 2, d)
        self.W_input  = nn.Linear(d * 2, d)
        self.W_cand   = nn.Linear(d * 2, d)
        nn.init.constant_(self.W_forget.bias, 1.0)
        nn.init.constant_(self.W_input.bias, -2.0)

        dh = max(d // 4, 1)
        self.gate = nn.Sequential(
            nn.Linear(d * 2, dh), nn.GELU(),
            nn.Linear(dh, 1), nn.Sigmoid()
        )

    def forward(self, x_t, h_prev, cap=None, decay=None):
        c = torch.cat([h_prev, x_t], dim=-1)
        f = torch.sigmoid(self.W_forget(c))
        i = torch.sigmoid(self.W_input(c))
        cand = f * h_prev + i * torch.tanh(self.W_cand(c))
        alpha = self.gate(c).squeeze(-1).unsqueeze(-1)
        h_new = alpha * cand + (1 - alpha) * h_prev

        # === Cap/Decay ===
        if cap is not None:
            sn = h_new.norm(dim=-1, keepdim=True)
            scale = (cap / sn).clamp(max=1.0)
            h_new = h_new * scale
        if decay is not None:
            h_new = h_new * decay

        return h_new


# ============================================================
# 3. FRSMASH-CD — 融合模型 + Cap/Decay
# ============================================================
class FRSMASH_CD(nn.Module):
    """
    FRSMASH-CD = OpenASH backbone + 1 SlowMemory + Cap/Decay

    相对 FRSMASH:
      - state_cap:  状态 norm 上限 (默认 150.0, None=关闭)
      - state_decay: 状态衰减系数 (默认 0.97, None=关闭)
      - 训练时建议 cap=None 或很大值，推理时开启

    参数:
        voc_size:     词表大小
        hidden_size:  隐藏维度
        num_heads:    注意力头数
        num_layers:   OpenASH 层数
        K:            慢尺度更新周期 (默认 8)
        state_cap:    norm 上限 (默认 None, 不限制)
        state_decay:  衰减系数 (默认 None, 不衰减)
    """
    def __init__(self, voc_size, hidden_size, num_heads, num_layers, K=8,
                 state_cap=None, state_decay=None):
        super().__init__()
        self.D = hidden_size
        self.K = K
        self.state_cap = state_cap
        self.state_decay = state_decay

        self.em = nn.Embedding(voc_size, hidden_size, padding_idx=0)

        # OpenASH 骨干
        self.ash_layers = nn.ModuleList([
            ASHDecoderLayer(hidden_size, num_heads, "train")
            for _ in range(num_layers)
        ])
        self.ash_norm = nn.LayerNorm(hidden_size)

        # 慢尺度记忆 (带 Cap/Decay)
        self.mem_input_proj = nn.Linear(hidden_size, hidden_size)
        self.slow_cell = SlowMemoryCellCD(hidden_size)

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
        B, T = x.shape
        D = self.D
        cap = self.state_cap
        decay = self.state_decay

        x_emb = self.em(x)

        # 1. OpenASH 骨干
        h = x_emb
        for layer in self.ash_layers:
            h1, _ = layer(h)
            h = h1 + h
        x_ash = self.ash_norm(h)

        # 2. 慢尺度记忆 (分段常数)
        inp_seq = self.mem_input_proj(x_emb)
        h_slow = torch.zeros(B, D, device=x.device)
        H_slow = torch.zeros(B, T, D, device=x.device)
        prev = 0
        for t in range(0, T, self.K):
            h_slow = self.slow_cell(inp_seq[:, t], h_slow, cap=cap, decay=decay)
            H_slow[:, prev:t+1] = h_slow.unsqueeze(1)
            prev = t + 1
        if prev < T:
            H_slow[:, prev:] = h_slow.unsqueeze(1)
        x_mem = self.mem_proj(H_slow)

        # 3. 门控融合
        cat = torch.cat([x_ash, x_mem], dim=-1)
        gate = self.fusion_gate(cat)
        fused = self.fusion_norm(
            gate * x_ash + (1 - gate) * x_mem + x_emb
        )

        return self.head(fused)

    @torch.no_grad()
    def generate_step(self, token_id, ash_states, h_slow):
        B = token_id.size(0)
        cap = self.state_cap
        decay = self.state_decay

        x = self.em(token_id)

        # OpenASH 逐层
        h = x
        new_states = []
        for i, layer in enumerate(self.ash_layers):
            layer.attn.model_flag = "infer"
            h1, s = layer.attn(h, ash_states[i])
            h1 = layer.norm(layer.alpha * layer.ffn(h1) + (1 - layer.alpha) * h)
            h = h1 + h
            new_states.append(s)
        x_ash = self.ash_norm(h[:, 0])

        # 慢尺度 + Cap/Decay
        inp = self.mem_input_proj(x[:, 0])
        h_slow_new = self.slow_cell(inp, h_slow, cap=cap, decay=decay)
        x_mem = self.mem_proj(h_slow_new)

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

    print("FRSMASH-CD variants:")
    print("=" * 60)

    for label, cap, decay in [
        ("无 cap/decay",       None,  None),
        ("cap=150, decay=0.97", 150.0, 0.97),
        ("cap=300, decay=0.99", 300.0, 0.99),
        ("cap=50,  decay=0.95", 50.0,  0.95),
    ]:
        model = FRSMASH_CD(VOCAB, H, HEADS, LAYERS, K=8,
                           state_cap=cap, state_decay=decay).to(device)
        n = sum(p.numel() for p in model.parameters())
        print(f"\n  {label}")
        print(f"  Params: {n:,} | cap={cap} | decay={decay}")

        # 训练前向
        x = torch.randint(0, VOCAB, (4, 384), device=device)
        logits = model(x)
        print(f"  Forward: {logits.shape}")

        # 推理
        token = torch.tensor([[42]], device=device)
        ash_states = [None] * LAYERS
        h_slow = torch.zeros(1, H, device=device)
        for step in range(5):
            logits, ash_states, h_slow = model.generate_step(
                token, ash_states, h_slow)
            token = logits.argmax(dim=-1, keepdim=True)
            sn = h_slow.norm().item()
        print(f"  Gen 5 steps done | slow state norm={sn:.4f}")
