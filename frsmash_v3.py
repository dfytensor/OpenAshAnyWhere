"""
FRSMASH v3 — F-layer 工作记忆 + 软截断 cummax 永久记忆 + 慢尺度选择性记忆

设计思路 (取长补短):
  v1 (cummax): 完美长程记忆 (Copy gap=+0.89@4K) 但状态单调增长 (norm 20+)
  v2 (F-layer): 状态有界稳定 (norm 2-3) 但快速遗忘 (Copy gap=-0.81@4K)

  v3 = v2 的工作记忆 (F-layer, 有界) + v1 的永久记忆 (cummax, 软截断) + 自适应融合

  关键创新:
    1. 双路状态: F-layer (有界) + 软截断 cummax (永久)
    2. 软截断: scale * tanh(x / scale) — 保留 max 排序信息但有界
    3. 自适应融合: 可学习 alpha 控制两路权重
    4. gen_model 用融合后的 out4 — 兼具记忆保持和 LM 能力

  目标: LM loss ≈ v2 (有界) + 长程 Copy ≈ v1 (不遗忘)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# 1. 双路状态机: F-layer + 软截断 cummax
# ============================================================
class DualStateSuper(nn.Module):
    """
    双路状态:
      Path A — F-layer 线性递推 (有界, 可并行, 强 LM)
      Path B — 软截断 cummax (永久记忆, 有界, 保留 max 排序)

    out4 = α · cummax_clamped + (1-α) · flayer
    α = sigmoid(learnable) — 训练时自动学习最优混合比
    """

    def __init__(self, dim_size, heads, model_flag="train"):
        super().__init__()
        self.heads = heads
        self.d_head = dim_size // heads
        self.model_flag = model_flag

        # 共享投影: out, out1, out2, out3
        self.combined = nn.Linear(dim_size, 4 * dim_size, bias=False)

        # gen_model (5-branch multiplicative interaction)
        self.alpha1 = nn.Parameter(torch.tensor(0.5))
        self.alpha2 = nn.Parameter(torch.tensor(0.5))
        self.alpha3 = nn.Parameter(torch.tensor(0.5))
        self.head_linear = nn.Linear(heads * 5, heads, bias=False)

        # === Path A: F-layer ===
        self.fast_proj = nn.Linear(dim_size, 4 * dim_size, bias=False)

        # === Path B: 软截断 cummax ===
        self.cm_scale = nn.Parameter(torch.tensor(3.0))  # 软截断尺度 (可学习)

        # === 自适应融合 ===
        self.fuse_logit = nn.Parameter(torch.tensor(0.0))  # sigmoid(0)=0.5

    @staticmethod
    def _parallel_scan(A, B, h_prev=None):
        """h_t = A_t * h_{t-1} + B_t 的并行前缀和"""
        A_s = A.clamp(min=1e-4, max=1.0)
        Acp = torch.cumprod(A_s, dim=1)
        csB = torch.cumsum(B / A_s, dim=1)
        if h_prev is None:
            return Acp * csB
        return Acp * (h_prev.unsqueeze(1) + csB)

    def forward(self, x, states=None):
        """
        x: (B, T, D)
        states: (state_f, state_c) or None
            state_f: (B, D) — F-layer 状态
            state_c: (B, heads, 1, d_head) — cummax 状态
        返回: (B, T, D), (new_state_f, new_state_c)
        """
        b, s, d = x.shape

        # 共享投影
        combined = self.combined(x).view(b, s, 4, self.heads, -1)
        out, out1, out2, out3 = combined.unbind(2)
        out = out.permute(0, 3, 1, 2)
        out1 = out1.permute(0, 3, 1, 2)
        out2 = out2.permute(0, 3, 1, 2)
        out3 = out3.permute(0, 3, 1, 2)

        state_f = states[0] if states is not None else None
        state_c = states[1] if states is not None else None

        # ===== Path A: F-layer (有界工作记忆) =====
        fg = self.fast_proj(x).reshape(b, s, 4, d)
        af = torch.sigmoid(fg[..., 0, :])
        ff = torch.sigmoid(fg[..., 1, :])
        i_f = torch.sigmoid(fg[..., 2, :])
        cf = torch.tanh(fg[..., 3, :])
        A = af * ff + (1 - af)
        B_coeff = af * i_f * cf
        H_f = self._parallel_scan(A, B_coeff, state_f)
        out4_f = H_f.reshape(b, s, self.heads, self.d_head).permute(0, 3, 1, 2)
        new_state_f = H_f[:, -1, :]

        # ===== Path B: 软截断 cummax (永久记忆) =====
        scale = F.softplus(self.cm_scale) + 0.5
        if state_c is None:
            out4_c, _ = torch.cummax(out2, dim=2)
        else:
            out4_c, _ = torch.cummax(torch.cat([state_c, out2], dim=2), dim=2)
            if self.model_flag == "train":
                out4_c = out4_c[:, :, 1:]
            else:
                out4_c = out4_c[:, :, -1:]
        new_state_c = out4_c[:, :, -1:]

        # 软截断: scale * tanh(x / scale)
        # 保留 max 的排序信息, 但值域有界 [-scale, +scale]
        out4_c = scale * torch.tanh(out4_c / scale)

        # ===== 自适应融合 =====
        alpha = torch.sigmoid(self.fuse_logit)
        out4 = alpha * out4_c + (1 - alpha) * out4_f

        # ===== gen_model (5-branch multiplicative interaction) =====
        cat = torch.cat([out, out1, out2, out3, out4], dim=-1)
        combined_g = self.head_linear(cat) * out4
        term1 = out * out1
        term2 = self.alpha1 * out1 + self.alpha2 * out3
        term3 = out * (self.alpha3 * out4 + out3)
        term4 = out1 * (out2 + out4)
        result = term1 + term2 + term3 + term4 + out2 * out4 + combined_g

        out_l = result.transpose(1, 2).contiguous().view(b, s, d)
        return out_l, (new_state_f, new_state_c)


# ============================================================
# 2. FFN + Decoder Layer
# ============================================================
class FeedForward(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.ffn1 = nn.Linear(hidden_size, hidden_size)
        self.ffn2 = nn.Linear(hidden_size, hidden_size)
        self.gate = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.ffn2(self.ffn1(x) * self.relu(self.gate(x)))


class DualDecoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, model_flag="train"):
        super().__init__()
        self.attn = DualStateSuper(hidden_size, num_heads, model_flag)
        self.ffn = FeedForward(hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, states=None, return_attn_state=False):
        x1, attn_states = self.attn(x, states)
        x = self.norm(self.alpha * self.ffn(x1) + (1 - self.alpha) * x)
        if return_attn_state:
            return x, attn_states
        return x, None


# ============================================================
# 3. 慢尺度记忆 (内容门控, 同 v1/v2)
# ============================================================
class SlowMemoryCell(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        d = d_model
        self.W_forget = nn.Linear(d * 2, d)
        self.W_input = nn.Linear(d * 2, d)
        self.W_cand = nn.Linear(d * 2, d)
        nn.init.constant_(self.W_forget.bias, 1.0)
        nn.init.constant_(self.W_input.bias, -2.0)
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
# 4. FRSMASH v3 — 三路融合
# ============================================================
class FRSMASH(nn.Module):
    """
    FRSMASH v3 = F-layer (工作记忆) + 软截断 cummax (永久记忆) + SlowMemory (选择性记忆)

    三路互补:
      - F-layer:   有界, 可并行, 强 LM 特征
      - cummax:    永久保持 max 信号, 软截断后有界
      - SlowMemory: 内容门控选择性记忆

    参数:
        voc_size:    词表大小
        hidden_size: 隐藏维度
        num_heads:   注意力头数
        num_layers:  层数
        K:           慢尺度更新周期
    """

    def __init__(self, voc_size, hidden_size, num_heads, num_layers, K=8):
        super().__init__()
        self.D = hidden_size
        self.K = K
        self.num_layers = num_layers

        self.em = nn.Embedding(voc_size, hidden_size, padding_idx=0)

        # 双路骨干
        self.ash_layers = nn.ModuleList([
            DualDecoderLayer(hidden_size, num_heads, "train")
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
        B, T = x.shape
        D = self.D

        x_emb = self.em(x)

        # 1. 双路骨干
        h = x_emb
        ash_states = [] if return_state else None
        for layer in self.ash_layers:
            if return_state:
                layer.attn.model_flag = "infer"
                h1, s = layer(h, states=None, return_attn_state=True)
                ash_states.append(s)
            else:
                h1, _ = layer(h)
            h = h1 + h
        x_ash = self.ash_norm(h)

        # 2. 慢尺度记忆
        inp_seq = self.mem_input_proj(x_emb)
        h_slow = torch.zeros(B, D, device=x.device)
        H_slow = torch.zeros(B, T, D, device=x.device)
        prev = 0
        for t in range(0, T, self.K):
            h_slow = self.slow_cell(inp_seq[:, t], h_slow)
            H_slow[:, prev:t + 1] = h_slow.unsqueeze(1)
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

        logits = self.head(fused)
        if return_state:
            return logits, ash_states, h_slow
        return logits

    @torch.no_grad()
    def generate_step(self, token_id, ash_states, h_slow):
        """
        推理单步 O(1)

        ash_states[i] = (state_f_i, state_c_i)
        """
        B = token_id.size(0)
        x = self.em(token_id)

        # 双路骨干 (逐层, 用 state)
        h = x
        new_states = []
        for i, layer in enumerate(self.ash_layers):
            layer.attn.model_flag = "infer"
            h1, (sf, sc) = layer.attn(h, ash_states[i] if ash_states[i] is not None else None)
            h1 = layer.norm(layer.alpha * layer.ffn(h1) + (1 - layer.alpha) * h)
            h = h1 + h
            new_states.append((sf, sc))
        x_ash = self.ash_norm(h[:, 0])

        # 慢尺度
        inp = self.mem_input_proj(x[:, 0])
        h_slow_new = self.slow_cell(inp, h_slow)
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

    model = FRSMASH(VOCAB, H, HEADS, LAYERS, K=8).to(device)
    n = sum(p.numel() for p in model.parameters())
    print(f"FRSMASH v3 (DualState): {n:,} params")
    print(f"  Backbone: {LAYERS}L x {HEADS}h x {H}d")
    print(f"  Path A: F-layer (working memory)")
    print(f"  Path B: soft-clamped cummax (permanent memory)")
    print(f"  Path C: SlowMemory (selective, K=8)")

    # 前向
    x = torch.randint(0, VOCAB, (4, 384), device=device)
    logits = model(x)
    print(f"\nForward: {logits.shape}")

    # return_state
    logits, ash_states, h_slow = model(
        torch.randint(0, VOCAB, (1, 64), device=device), return_state=True)
    print(f"return_state: logits={logits.shape}, "
          f"ash_states={len(ash_states)} layers, "
          f"h_slow={h_slow.shape}")
    print(f"  state_f norm: {ash_states[0][0].norm():.4f}")
    print(f"  state_c norm: {ash_states[0][1].norm():.4f}")

    # 推理
    token = torch.tensor([[42]], device=device)
    ash_s = [None] * LAYERS
    h_s = torch.zeros(1, H, device=device)
    for step in range(5):
        logits, ash_s, h_s = model.generate_step(token, ash_s, h_s)
        token = logits.argmax(dim=-1, keepdim=True)
    print(f"\nGen 5 steps OK | h_slow norm={h_s.norm():.4f}")

    # 检查融合 alpha
    print(f"\nPer-layer fusion alpha (cummax vs flayer):")
    for i, layer in enumerate(model.ash_layers):
        a = torch.sigmoid(layer.attn.fuse_logit).item()
        s = F.softplus(layer.attn.cm_scale).item() + 0.5
        print(f"  Layer {i}: alpha={a:.3f} (cummax={a*100:.0f}%, flayer={100-a*100:.0f}%) | cm_scale={s:.2f}")
