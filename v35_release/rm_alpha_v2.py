"""
RM-Alpha v2 — 吸收 v3 的 gen_model + 软截断 cummax, 保持 RM-Alpha 的全并行速度

改进点 (相对 RM-Alpha v1):
  1. + gen_model (5-branch multiplicative interaction)  ← 来自 v3, 零速度成本
  2. + 软截断 cummax (scale * tanh(x/scale))             ← 来自 v3, 零速度成本
  3. + 残差连接 (h = h1 + h)                              ← 来自 v3
  4. - 丢弃 SlowMemoryCell                                ← v3 的速度瓶颈
  5. - 丢弃 MAA aux loss                                  ← 影响微小
  6. - 丢弃 Gumbel-softmax (改为 deterministic routing)   ← 消除训练噪声

目标: LM loss ≈ v3 + 速度 ≈ RM-Alpha v1
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


# ============================================================
# 1. 软截断多 slot cummax (来自 v3 的软截断 + RM-Alpha 的多 slot)
# ============================================================
class SoftMSMR(nn.Module):
    """
    Multi-Slot Max Recurrence + 软截断

    改进:
      - cummax 后加 scale * tanh(x / scale), 有界化
      - scale 可学习
    """
    def __init__(self, num_slots=8, d_slot=32):
        super().__init__()
        self.num_slots = num_slots
        self.d_slot = d_slot
        self.slot_gates = nn.Parameter(torch.randn(num_slots, d_slot))
        self.cm_scale = nn.Parameter(torch.tensor(3.0))  # 软截断尺度 (可学习)
        self.register_buffer("init_state",
                             torch.full((num_slots, d_slot), -float('inf')))

    def forward(self, x):
        """
        x: [B, L, D]  D = num_slots * d_slot
        返回: state [B, L, D]
        """
        B, L, D = x.shape
        x = x.view(B, L, self.num_slots, self.d_slot)
        gated = x * self.slot_gates.view(1, 1, self.num_slots, self.d_slot)
        state = torch.cummax(gated, dim=2).values  # [B, L, num_slots, d_slot]

        # 软截断: 有界化, 保留排序
        scale = F.softplus(self.cm_scale) + 0.5
        state = scale * torch.tanh(state / scale)

        return state.reshape(B, L, D)

    @torch.no_grad()
    def step(self, x_t, state):
        """推理单步"""
        B = x_t.size(0)
        x_t = x_t.view(B, self.num_slots, self.d_slot)
        gated = x_t * self.slot_gates.view(1, self.num_slots, self.d_slot)
        new_state = torch.maximum(state, gated)
        scale = F.softplus(self.cm_scale) + 0.5
        new_state = scale * torch.tanh(new_state / scale)
        return new_state.reshape(B, -1), new_state


# ============================================================
# 2. gen_model (来自 v3/FRSMASH, 5-branch 乘法交互)
# ============================================================
class GenModel(nn.Module):
    """
    5-branch multiplicative interaction — v3 的核心表达力来源

    输入 x → 投影出 out/out1/out2/out3
    状态 state → out4 (来自 cummax)
    5 路非线性交互 → 丰富特征

    全并行, 零串行开销
    """
    def __init__(self, d_model, heads=8):
        super().__init__()
        self.heads = heads
        self.d_head = d_model // heads
        # 投影到 4 路 (out, out1, out2, out3)
        self.combined = nn.Linear(d_model, 4 * d_model, bias=False)
        # gen_model 可学习参数
        self.alpha1 = nn.Parameter(torch.tensor(0.5))
        self.alpha2 = nn.Parameter(torch.tensor(0.5))
        self.alpha3 = nn.Parameter(torch.tensor(0.5))
        self.head_linear = nn.Linear(heads * 5, heads, bias=False)

    def forward(self, x, state):
        """
        x: [B, L, D]
        state: [B, L, D]  (来自 cummax)
        返回: [B, L, D]
        """
        B, L, D = x.shape
        b, s, d = B, L, D
        h, dh = self.heads, self.d_head

        combined = self.combined(x).view(b, s, 4, h, -1)
        out, out1, out2, out3 = combined.unbind(2)
        out = out.permute(0, 3, 1, 2)
        out1 = out1.permute(0, 3, 1, 2)
        out2 = out2.permute(0, 3, 1, 2)
        out3 = out3.permute(0, 3, 1, 2)
        out4 = state.view(b, s, h, dh).permute(0, 3, 1, 2)  # cummax 状态

        # 5-branch multiplicative interaction
        cat = torch.cat([out, out1, out2, out3, out4], dim=-1)
        combined_g = self.head_linear(cat) * out4
        term1 = out * out1
        term2 = self.alpha1 * out1 + self.alpha2 * out3
        term3 = out * (self.alpha3 * out4 + out3)
        term4 = out1 * (out2 + out4)
        result = term1 + term2 + term3 + term4 + out2 * out4 + combined_g

        return result.transpose(1, 2).contiguous().view(b, s, d)


# ============================================================
# 3. RM-Alpha v2 Block
# ============================================================
class RMA2Block(nn.Module):
    """
    RM-Alpha v2 Block = SoftMSMR + GenModel + FFN + 残差

    vs v1: 用 gen_model 替代简单 FFN, 加软截断, 加残差
    vs v3: 无 SlowMemory (保持全并行速度)
    """
    def __init__(self, d_model, num_slots=8, heads=8):
        super().__init__()
        d_slot = d_model // num_slots
        self.ms = SoftMSMR(num_slots=num_slots, d_slot=d_slot)
        self.gen = GenModel(d_model, heads=heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm = nn.LayerNorm(d_model)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        """x: [B, L, D] → [B, L, D]"""
        state = self.ms(x)            # 软截断 cummax
        gen_out = self.gen(x, state)  # gen_model 5-branch
        out = self.norm(self.alpha * self.ffn(gen_out) + (1 - self.alpha) * gen_out)
        return out + x  # 残差


# ============================================================
# 4. RM-Alpha v2 完整模型
# ============================================================
class RMAlpha2(nn.Module):
    """
    RM-Alpha v2

    = SoftMSMR (软截断多 slot cummax)
    + GenModel (5-branch 乘法交互, 来自 v3)
    + FFN + 残差
    — 无 SlowMemory (保持全并行速度)
    """
    def __init__(self, vocab_size, d_model=256, num_layers=4, num_slots=8, heads=8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.layers = nn.ModuleList([
            RMA2Block(d_model, num_slots, heads) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        """x: [B, T] token ids → [B, T, vocab] logits"""
        h = self.embed(x)
        for layer in self.layers:
            h = layer(h)
        return self.head(self.norm(h))


# ============================================================
# 测试
# ============================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    VOCAB = 23005
    H = 256

    model = RMAlpha2(VOCAB, d_model=H, num_layers=4, num_slots=8, heads=8).to(device)
    n = sum(p.numel() for p in model.parameters())
    print(f"RM-Alpha v2: {n:,} params")
    print(f"  {4}L x {8}slots x {8}heads x {H}d")

    x = torch.randint(0, VOCAB, (4, 384), device=device)
    logits = model(x)
    print(f"Forward: {logits.shape}")

    # 速度测试
    import time
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    t0 = time.time()
    for _ in range(100):
        x = torch.randint(0, VOCAB, (64, 256), device=device)
        loss = F.cross_entropy(model(x)[:,:-1].reshape(-1,VOCAB), x[:,1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    torch.cuda.synchronize()
    print(f"Speed: {100/(time.time()-t0):.1f} step/s")
