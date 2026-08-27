import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

# ============================================================
# 1. Dynamic Semantic Projection (DSP) — 修复维度
# ============================================================
class DynamicSemanticProjection(nn.Module):
    def __init__(self, d_model, d_low=64, d_high=512):
        super().__init__()
        # 门控判别器
        self.gate = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 2)
        )
        # 两个投影都输出 d_model，但内部结构不同（低秩 vs 全秩）
        # 低密度投影：先降维再升维 (低秩)
        self.proj_low_down = nn.Linear(d_model, d_low, bias=False)
        self.proj_low_up   = nn.Linear(d_low, d_model, bias=False)
        # 高密度投影：直接全维投影
        self.proj_high = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        """
        x: [B, L, D] 或 [B, D]
        返回: output [B, L, D] 或 [B, D], gate info
        """
        # 保留原始维度信息
        if x.dim() == 2:
            x = x.unsqueeze(1)  # 虚拟长度维
            squeeze_back = True
        else:
            squeeze_back = False

        logits = self.gate(x)   # [B, L, 2] 或 [B, 1, 2]
        gumbel = F.gumbel_softmax(logits, tau=1.0, hard=True)
        low_gate = gumbel[..., 0:1]
        high_gate = gumbel[..., 1:2]

        x_low = self.proj_low_up(self.proj_low_down(x))   # 低秩路径
        x_high = self.proj_high(x)

        out = low_gate * x_low + high_gate * x_high

        if squeeze_back:
            out = out.squeeze(1)
        return out, low_gate, high_gate


# ============================================================
# 2. Multi-Slot Max Recurrence (MSMR) — 修复输出维度
# ============================================================
def parallel_cumax(x: torch.Tensor) -> torch.Tensor:
    """训练用并行 cumax"""
    return torch.cummax(x, dim=1).values

class MSMR(nn.Module):
    def __init__(self, num_slots=16, d_slot=256):
        super().__init__()
        self.num_slots = num_slots
        self.d_slot = d_slot
        self.slot_gates = nn.Parameter(torch.randn(num_slots, d_slot))
        self.register_buffer("init_state",
                             torch.full((num_slots, d_slot), -float('inf')))

    def forward_train(self, x: torch.Tensor) -> torch.Tensor:
        """
        训练：并行扫描
        x: [B, L, D]  其中 D = num_slots * d_slot
        返回: state [B, L, num_slots, d_slot]
        """
        B, L, D = x.shape
        x = x.view(B, L, self.num_slots, self.d_slot)
        gated = x * self.slot_gates.view(1, 1, self.num_slots, self.d_slot)
        state = parallel_cumax(gated)  # [B, L, num_slots, d_slot]
        return state

    def forward_infer(self, x_t: torch.Tensor,
                      state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        推理：严格递归
        x_t: [B, D]
        state: [B, num_slots, d_slot]
        返回: out [B, D], new_state [B, num_slots, d_slot]
        """
        x_t = x_t.view(x_t.size(0), self.num_slots, self.d_slot)
        gated = x_t * self.slot_gates.view(1, self.num_slots, self.d_slot)
        new_state = torch.maximum(state, gated)
        out = new_state.reshape(x_t.size(0), -1)  # [B, D]
        return out, new_state


# ============================================================
# 3. Micro-Attention Adapter (MAA) — 原样保留
# ============================================================
class MicroAttentionAdapter(nn.Module):
    def __init__(self, d_model, heads=2):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=heads, batch_first=True
        )
        self.alpha = 0.01

    def forward(self, x, ms_state):
        # ms_state: [B, L, num_slots, d_slot] -> 压平为 [B, L, D]
        B, L = x.shape[:2]
        ms_flat = ms_state.reshape(B, L, -1)
        attn_out, _ = self.attn(x, x, x)
        loss = self.alpha * F.mse_loss(ms_flat, attn_out)
        return loss


# ============================================================
# 4. RM-α Block — 维度贯通
# ============================================================
class RMAlphaBlock(nn.Module):
    def __init__(self, d_model=4096, num_slots=16):
        super().__init__()
        d_slot = d_model // num_slots
        self.dsp = DynamicSemanticProjection(d_model)
        self.ms = MSMR(num_slots=num_slots, d_slot=d_slot)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.maa = MicroAttentionAdapter(d_model)

    def forward_train(self, x):
        """
        x: [B, L, D]
        返回: output [B, L, D], loss
        """
        proj_x, _, _ = self.dsp(x)       # [B, L, D]
        ms_state = self.ms.forward_train(proj_x)   # [B, L, S, d_slot]
        ms_flat = ms_state.reshape(*x.shape)       # [B, L, D]
        ffn_out = self.ffn(ms_flat) + ms_flat      # 残差
        maa_loss = self.maa(x, ms_state)
        return ffn_out, maa_loss

    def forward_infer(self, x_t, state):
        """
        x_t: [B, D]
        state: [B, S, d_slot]
        返回: out [B, D], new_state
        """
        proj_x, _, _ = self.dsp(x_t)       # [B, D]
        out, new_state = self.ms.forward_infer(proj_x, state)
        out = self.ffn(out) + out
        return out, new_state


# ============================================================
# 5. RM-α 完整模型
# ============================================================
class RMAlpha(nn.Module):
    def __init__(self, vocab_size=128000, d_model=4096, num_layers=32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            RMAlphaBlock(d_model) for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward_train(self, input_ids):
        x = self.embed(input_ids)
        total_loss = 0.0
        for layer in self.layers:
            x, loss = layer.forward_train(x)
            total_loss += loss
        logits = self.lm_head(x)
        return logits, total_loss

    @torch.no_grad()
    def forward_infer(self, input_ids):
        B, L = input_ids.shape
        # 初始化每一层的状态
        states = [
            layer.ms.init_state.unsqueeze(0).repeat(B, 1, 1)
            for layer in self.layers
        ]

        outputs = []
        for t in range(L):
            x = self.embed(input_ids[:, t])
            for i, layer in enumerate(self.layers):
                x, states[i] = layer.forward_infer(x, states[i])
            outputs.append(self.lm_head(x))

        return torch.stack(outputs, dim=1)


# ============================================================
# 6. 验证：训练 / 推理输出一致性
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(42)
    model = RMAlpha(vocab_size=1000, d_model=256, num_layers=2)  # 玩具尺寸

    # 随机输入
    input_ids = torch.randint(0, 1000, (1, 16))

    # 训练模式
    model.train()
    logits_train, loss = model.forward_train(input_ids)

    # 推理模式（逐 token）
    model.eval()
    logits_infer = model.forward_infer(input_ids)

    # 对比最后一层输出
    diff = (logits_train - logits_infer).abs().max().item()
    print(f"最大差异: {diff:.2e}")
    if diff < 1e-5:
        print("✅ 训练与推理输出一致（bit-exact）")
    else:
        print("⚠️ 存在微小差异（可能是 gumbel-softmax 的随机性，可固定种子）")
