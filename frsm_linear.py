"""
HybridFRSM — 快慢尺度分离的分形递归状态机

实验验证最优配置:
  - 3快+1慢: CF@65K=100%, best=0.00026, 58s (推荐默认)
  - 1快+1慢: 最少参数(231K), CF@65K=100%, best=0.00024
  - 0快+4慢: 最优收敛(0.00021), 但最慢(132s) = V6a

关键发现:
  - 快尺度(线性递推)负责即时预测, 无门控开销
  - 慢尺度(内容门控)负责选择性记忆, 只需1个即可达100%
  - 快慢分离比纯门控(V6a)快4.7×, 参数少24%
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SlowScaleCell(nn.Module):
    """
    慢尺度状态更新单元 — 完整内容门控

    更新流程:
      1. forget/input/candidate 三门 → 候选值
      2. 内容门控 MLP → 写入强度 α ∈ [0,1]
      3. h_new = α * candidate + (1-α) * h_prev

    初始化策略:
      forget bias = 1.0  (默认记住)
      input bias  = -2.0 (默认不写)
      → 模型需主动学习何时写入

    参数:
        num_slow:  慢尺度数量
        d_model:   模型维度
    """
    def __init__(self, num_slow, d_model):
        super().__init__()
        self.num_slow = num_slow
        self.d_model = d_model

        # 三门参数: (num_slow, d_out, d_in) = (S, D, 2D)
        self.W_forget = nn.Parameter(torch.empty(num_slow, d_model, 2 * d_model))
        self.b_forget = nn.Parameter(torch.empty(num_slow, d_model))
        self.W_input  = nn.Parameter(torch.empty(num_slow, d_model, 2 * d_model))
        self.b_input  = nn.Parameter(torch.empty(num_slow, d_model))
        self.W_cand   = nn.Parameter(torch.empty(num_slow, d_model, 2 * d_model))
        self.b_cand   = nn.Parameter(torch.empty(num_slow, d_model))

        # 内容门控 MLP: 2D → D/4 → 1
        d_hidden = max(d_model // 4, 1)
        self.gate_W1 = nn.Parameter(torch.empty(num_slow, d_hidden, 2 * d_model))
        self.gate_b1 = nn.Parameter(torch.empty(num_slow, d_hidden))
        self.gate_W2 = nn.Parameter(torch.empty(num_slow, 1, d_hidden))
        self.gate_b2 = nn.Parameter(torch.empty(num_slow, 1))

        self._init_weights()

    def _init_weights(self):
        for p in [self.W_forget, self.W_input, self.W_cand,
                  self.gate_W1, self.gate_W2]:
            for s in range(self.num_slow):
                nn.init.kaiming_uniform_(p[s], a=math.sqrt(5))
        for p in [self.b_forget, self.b_input, self.b_cand,
                  self.gate_b1, self.gate_b2]:
            nn.init.zeros_(p)
        nn.init.constant_(self.b_forget, 1.0)
        nn.init.constant_(self.b_input, -2.0)

    def forward(self, x_t, h_prev):
        """
        单步更新 (所有慢尺度并行 via einsum)

        x_t:    (B, d_model)          当前输入
        h_prev: (B, num_slow, d_model) 上一时刻状态
        返回:   (B, num_slow, d_model) 新状态
        """
        S = self.num_slow

        # 拼接状态与输入
        x_exp = x_t.unsqueeze(1).expand(-1, S, -1)
        gate_in = torch.cat([h_prev, x_exp], dim=-1)   # (B, S, 2D)

        # 三门: gate_in(B,S,2D) × W(S,D,2D) → (B,S,D)
        f = torch.sigmoid(
            torch.einsum('bnj,nij->bni', gate_in, self.W_forget) + self.b_forget
        )
        i = torch.sigmoid(
            torch.einsum('bnj,nij->bni', gate_in, self.W_input) + self.b_input
        )
        cand = torch.tanh(
            torch.einsum('bnj,nij->bni', gate_in, self.W_cand) + self.b_cand
        )
        candidate = f * h_prev + i * cand

        # 内容门控: gate_in(B,S,2D) × W1(S,H,2D) → (B,S,H), 再 × W2(S,1,H) → (B,S,1)
        h1 = F.gelu(
            torch.einsum('bnj,nij->bni', gate_in, self.gate_W1) + self.gate_b1
        )
        alpha = torch.sigmoid(
            torch.einsum('bni,noi->bno', h1, self.gate_W2) + self.gate_b2
        )

        return alpha * candidate + (1 - alpha) * h_prev


class HybridFRSM(nn.Module):
    """
    混合 FRSM — 快尺度(线性并行) + 慢尺度(内容门控)

    快尺度:
      - 纯线性递推 h_t = A_t * h_{t-1} + B_t
      - A, B 由输入决定 (不依赖状态), 可用 parallel scan 实现 O(log T) 训练
      - 负责即时预测和局部语法

    慢尺度:
      - 完整内容门控, 每 K 步更新一次
      - 负责选择性长期记忆

    实验结果 (CopyFirst, H=128, 2500 steps):
      3F+1S: 396K params, best=0.00026, CF@65K=100%, 58s
      1F+1S: 231K params, best=0.00024, CF@65K=100%, 59s
      0F+4S: 518K params, best=0.00021, CF@65K=100%, 132s (= V6a)

    参数:
        d_model:          模型维度
        num_fast:         快尺度数量 (默认 3)
        num_slow:         慢尺度数量 (默认 1)
        slow_update_freq: 慢尺度更新周期 K (默认 8)
    """
    def __init__(self, d_model=256, num_fast=3, num_slow=1, slow_update_freq=8):
        super().__init__()
        self.d_model = d_model
        self.num_fast = num_fast
        self.num_slow = num_slow
        self.slow_update_freq = slow_update_freq

        # 快尺度: 一次投影计算所有快尺度的 (alpha, forget, input, candidate)
        self.fast_proj = nn.Linear(d_model, num_fast * 4 * d_model)

        # 慢尺度: 完整内容门控
        self.slow_cell = SlowScaleCell(num_slow, d_model)

        # 融合层
        total_scales = num_fast + num_slow
        self.fusion = nn.Linear(total_scales * d_model, d_model)
        self.fusion_norm = nn.LayerNorm(d_model)

        # 输出投影 (保持维度, 可接 LM head)
        self.output_proj = nn.Linear(d_model, d_model)

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.fast_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.fast_proj.bias)
        nn.init.kaiming_uniform_(self.fusion.weight, a=math.sqrt(5))
        nn.init.zeros_(self.fusion.bias)
        nn.init.kaiming_uniform_(self.output_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.output_proj.bias)

    def _parallel_scan(self, A, B, h_prev):
        """
        并行扫描: 解析求解 h_t = A_t * h_{t-1} + B_t

        闭式解:
          h_t = (∏_{j=0}^{t} A_j) * h_0 + Σ_{j=0}^{t} (∏_{i=j+1}^{t} A_i) * B_j

        用 cumprod + cumsum 实现 O(T) 并行 (可用 Blelloch scan 优化为 O(log T))

        A:      (B, T, NF, D)  递推系数 ∈ [0, 1]
        B:      (B, T, NF, D)  递推偏置
        h_prev: (B, NF, D)     初始状态
        返回:   (B, T, NF, D)  每步状态
        """
        # 数值稳定: A ∈ (0, 1]
        A_safe = A.clamp(min=1e-4, max=1.0)

        # 前缀积: P[t] = ∏_{j=0}^{t} A[j]
        A_cumprod = torch.cumprod(A_safe, dim=1)

        # 前缀和: S[t] = Σ_{j=0}^{t} B[j] / P[j]
        B_div_A = B / A_safe
        cumsum_B = torch.cumsum(B_div_A, dim=1)

        # h_t = P[t] * (h_0 + S[t])
        if h_prev is None:
            H = A_cumprod * cumsum_B
        else:
            H = A_cumprod * (h_prev.unsqueeze(1) + cumsum_B)
        return H

    def forward(self, x, h_prev=None, return_state=False):
        """
        训练模式: 全序列前向

        快尺度用 parallel scan (完全并行, 无时间步循环)
        慢尺度用分段常数近似 (每 K 步一次完整更新)

        x:      (B, T, d_model)  输入特征
        h_prev:  可选 (B, NF+NS, D) 初始状态
        返回:    (B, T, d_model) 输出特征
        """
        B, T, D = x.shape
        NF, NS, K = self.num_fast, self.num_slow, self.slow_update_freq

        # ========== 1. 快尺度: parallel scan ==========
        fast_gates = self.fast_proj(x)                     # (B, T, NF*4*D)
        fast_gates = fast_gates.reshape(B, T, NF, 4, D)

        alpha_f = torch.sigmoid(fast_gates[..., 0, :])     # 写入门
        f_f     = torch.sigmoid(fast_gates[..., 1, :])     # forget
        i_f     = torch.sigmoid(fast_gates[..., 2, :])     # input
        cand_f  = torch.tanh(fast_gates[..., 3, :])        # candidate

        # 线性递推系数: h = A*h + B
        A = alpha_f * f_f + (1 - alpha_f)                  # (B, T, NF, D) ∈ [0, 1]
        B_coeff = alpha_f * i_f * cand_f                   # (B, T, NF, D)

        # 初始快状态
        if h_prev is None:
            h_fast_start = None  # parallel_scan 内部用 0
        else:
            h_fast_start = h_prev[:, :NF, :]

        H_fast = self._parallel_scan(A, B_coeff, h_fast_start)  # (B, T, NF, D)

        # ========== 2. 慢尺度: 分段常数 (每 K 步更新) ==========
        if h_prev is None:
            h_slow = torch.zeros(B, NS, D, device=x.device, dtype=x.dtype)
        else:
            h_slow = h_prev[:, NF:, :]

        H_slow = torch.zeros(B, T, NS, D, device=x.device, dtype=x.dtype)

        prev_t = 0
        for t in range(0, T, K):
            h_slow = self.slow_cell(x[:, t, :], h_slow)
            H_slow[:, prev_t:t+1] = h_slow.unsqueeze(1)
            prev_t = t + 1
        if prev_t < T:
            H_slow[:, prev_t:] = h_slow.unsqueeze(1)

        # ========== 3. 融合输出 ==========
        H_all = torch.cat([H_fast, H_slow], dim=2)         # (B, T, (NF+NS), D)
        H_flat = H_all.reshape(B, T, -1)
        fused = self.fusion_norm(self.fusion(H_flat))
        output = self.output_proj(fused)

        if return_state:
            final_fast = H_fast[:, -1, :, :]
            final_slow = H_slow[:, -1, :, :]
            final_state = torch.cat([final_fast, final_slow], dim=1)
            return output, final_state
        return output

    @torch.no_grad()
    def generate_step(self, x_t, h_prev):
        """
        推理模式: 单步 O(1)

        x_t:    (B, 1, d_model) 或 (B, d_model)  当前输入
        h_prev: (B, NF+NS, d_model)               上一时刻完整状态
        返回:   output_t (B, d_model), next_h (B, NF+NS, d_model)
        """
        if x_t.dim() == 3:
            x_t = x_t.squeeze(1)
        B, D = x_t.shape
        NF, NS = self.num_fast, self.num_slow

        # 分离快慢状态
        h_fast = h_prev[:, :NF, :]
        h_slow = h_prev[:, NF:, :]

        # 快尺度: 线性递推
        fg = self.fast_proj(x_t).reshape(B, NF, 4, D)
        alpha_f = torch.sigmoid(fg[..., 0, :])
        f_f     = torch.sigmoid(fg[..., 1, :])
        i_f     = torch.sigmoid(fg[..., 2, :])
        cand_f  = torch.tanh(fg[..., 3, :])
        h_fast_next = (alpha_f * f_f + (1 - alpha_f)) * h_fast + alpha_f * i_f * cand_f

        # 慢尺度: 完整内容门控
        h_slow_next = self.slow_cell(x_t, h_slow)

        # 合并状态
        next_h = torch.cat([h_fast_next, h_slow_next], dim=1)

        # 融合输出
        h_flat = next_h.reshape(B, -1)
        fused = self.fusion_norm(self.fusion(h_flat))
        output_t = self.output_proj(fused)
        return output_t, next_h


class HybridFRSM_LM(nn.Module):
    """
    HybridFRSM 语言模型封装

    在 HybridFRSM 基础上增加:
      - Token embedding
      - Input projection
      - LM output head (→ vocab_size)

    参数:
        vocab_size:       词表大小
        d_model:          模型维度 (默认 256)
        num_fast:         快尺度数量 (默认 3)
        num_slow:         慢尺度数量 (默认 1)
        slow_update_freq: 慢尺度更新周期 (默认 8)
    """
    def __init__(self, vocab_size, d_model=256, num_fast=3, num_slow=1,
                 slow_update_freq=8):
        super().__init__()
        self.d_model = d_model
        self.num_fast = num_fast
        self.num_slow = num_slow

        self.embed = nn.Embedding(vocab_size, d_model)
        self.input_proj = nn.Linear(d_model, d_model)

        self.frsm = HybridFRSM(
            d_model=d_model,
            num_fast=num_fast,
            num_slow=num_slow,
            slow_update_freq=slow_update_freq,
        )

        self.lm_head = nn.Linear(d_model, vocab_size)

        nn.init.normal_(self.embed.weight, mean=0, std=0.02)
        nn.init.kaiming_uniform_(self.input_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.input_proj.bias)
        nn.init.kaiming_uniform_(self.lm_head.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lm_head.bias)

    def forward(self, token_ids, h_prev=None):
        """
        token_ids: (B, T)  token id 序列
        返回:      (B, T, vocab_size) logits
        """
        x = self.input_proj(self.embed(token_ids))   # (B, T, D)
        out = self.frsm(x, h_prev=h_prev)            # (B, T, D)
        return self.lm_head(out)                      # (B, T, vocab)

    @torch.no_grad()
    def generate_step(self, token_id, h_fast, h_slow):
        """
        推理单步生成

        token_id: (B, 1)  当前 token
        h_fast:   (B, NF, D)  快尺度状态
        h_slow:   (B, NS, D)  慢尺度状态
        返回:     logits (B, vocab), h_fast_new, h_slow_new
        """
        B = token_id.size(0)
        x = self.input_proj(self.embed(token_id).squeeze(1))  # (B, D)

        # 快尺度
        fg = self.frsm.fast_proj(x).reshape(B, self.num_fast, 4, self.d_model)
        alpha = torch.sigmoid(fg[..., 0, :])
        f_f   = torch.sigmoid(fg[..., 1, :])
        i_f   = torch.sigmoid(fg[..., 2, :])
        c_f   = torch.tanh(fg[..., 3, :])
        h_fast_new = (alpha * f_f + (1 - alpha)) * h_fast + alpha * i_f * c_f

        # 慢尺度
        h_slow_new = self.frsm.slow_cell(x, h_slow)

        # 融合
        H = torch.cat([h_fast_new, h_slow_new], dim=1).reshape(B, -1)
        fused = self.frsm.fusion_norm(self.frsm.fusion(H))
        logits = self.lm_head(fused)
        return logits, h_fast_new, h_slow_new


# ============================================================
# 测试与演示
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === 1. 特征提取器模式 ===
    print("=== HybridFRSM (Feature Extractor) ===")
    B, T, D = 4, 256, 256
    model = HybridFRSM(d_model=D, num_fast=3, num_slow=1, slow_update_freq=8).to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    x = torch.randn(B, T, D, device=device)
    out = model(x)
    print(f"Train output: {out.shape} (expect {B},{T},{D})")

    h = torch.zeros(B, 4, D, device=device)
    out_step, h = model.generate_step(x[:, :1, :], h)
    print(f"Inference step: {out_step.shape} (expect {B},{D})")

    # === 2. 语言模型模式 ===
    print("\n=== HybridFRSM_LM (Language Model) ===")
    VOCAB = 23005
    lm = HybridFRSM_LM(vocab_size=VOCAB, d_model=D, num_fast=3, num_slow=1).to(device)
    print(f"LM Params: {sum(p.numel() for p in lm.parameters()):,}")

    token_ids = torch.randint(0, VOCAB, (B, T), device=device)
    logits = lm(token_ids)
    print(f"LM output: {logits.shape} (expect {B},{T},{VOCAB})")

    # 推理生成
    h_fast = torch.zeros(1, 3, D, device=device)
    h_slow = torch.zeros(1, 1, D, device=device)
    token = torch.tensor([[42]], device=device)
    for step in range(5):
        logits, h_fast, h_slow = lm.generate_step(token, h_fast, h_slow)
        token = logits.argmax(dim=-1, keepdim=True)
        print(f"  Step {step+1}: token={token.item()}")

    # === 3. 不同配置参数量 ===
    print("\n=== Config Comparison ===")
    for nf, ns in [(3,1), (2,2), (1,1), (0,4), (4,0)]:
        m = HybridFRSM_LM(VOCAB, D, nf, ns)
        n = sum(p.numel() for p in m.parameters())
        tag = f"{nf}F+{ns}S"
        slow_only = ns > 0
        print(f"  {tag:>6}: {n:>10,} params  {'with content gate' if slow_only else 'no memory'}")
