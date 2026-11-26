"""
FRSM V6a Fast — 训练尺度并行 + 推理 State 模式
核心优化：将 num_scales 个独立小 Linear 合并为可训练的批量参数，
        使用 einsum 一次处理所有尺度，消除 Python 循环。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FRSM_V6_Fast(nn.Module):
    """
    Content‑Gated Multi‑Scale State Machine（快速版）

    参数:
        vocab_size:  词表大小
        d_model:     模型维度（默认 256）
        num_scales:  并行尺度数（默认 4）
        hard_infer:  推理时是否用硬阈值（True 为硬阈值 0.5，False 为软 sigmoid）
    """
    def __init__(self, vocab_size, d_model=256, num_scales=4, hard_infer=True):
        super().__init__()
        self.d_model = d_model
        self.num_scales = num_scales
        self.hard_infer = hard_infer

        # Embedding & input projection
        self.embed = nn.Embedding(vocab_size, d_model)
        self.input_proj = nn.Linear(d_model, d_model)

        # ---- 合并多尺度参数：形状为 (num_scales, d_out, d_in) ----
        # RawBlock 的三个门：forget, input, cand
        self.W_forget = nn.Parameter(torch.empty(num_scales, d_model, 2 * d_model))
        self.b_forget = nn.Parameter(torch.empty(num_scales, d_model))
        self.W_input  = nn.Parameter(torch.empty(num_scales, d_model, 2 * d_model))
        self.b_input  = nn.Parameter(torch.empty(num_scales, d_model))
        self.W_cand   = nn.Parameter(torch.empty(num_scales, d_model, 2 * d_model))
        self.b_cand   = nn.Parameter(torch.empty(num_scales, d_model))

        # 内容门控网络：两层（2*d_model -> d_model//4 -> 1）
        d_hidden = d_model // 4
        self.gate_W1 = nn.Parameter(torch.empty(num_scales, d_hidden, 2 * d_model))
        self.gate_b1 = nn.Parameter(torch.empty(num_scales, d_hidden))
        self.gate_W2 = nn.Parameter(torch.empty(num_scales, 1, d_hidden))
        self.gate_b2 = nn.Parameter(torch.empty(num_scales, 1))

        # Fusion + 输出
        self.fusion = nn.Linear(d_model * num_scales, d_model)
        self.fusion_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        # Stacked scale params: init each scale slice individually to match per-Linear scale
        def _init_scale_stack(param):
            for s in range(self.num_scales):
                nn.init.kaiming_uniform_(param[s], a=math.sqrt(5))
        for p in [self.W_forget, self.W_input, self.W_cand]:
            _init_scale_stack(p)
        for p in [self.gate_W1]:
            _init_scale_stack(p)
        for p in [self.gate_W2]:
            _init_scale_stack(p)
        # Biases
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.zeros_(param)
        nn.init.zeros_(self.b_cand)
        nn.init.zeros_(self.gate_b1)
        nn.init.zeros_(self.gate_b2)
        nn.init.constant_(self.b_forget, 1.0)
        nn.init.constant_(self.b_input, -2.0)
        # Fusion & output
        nn.init.kaiming_uniform_(self.fusion.weight, a=math.sqrt(5))
        nn.init.zeros_(self.fusion.bias)
        nn.init.kaiming_uniform_(self.output_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.output_proj.bias)
        # Embedding
        nn.init.normal_(self.embed.weight, mean=0, std=0.02)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.kaiming_uniform_(self.input_proj.weight, a=math.sqrt(5))

    def _raw_block(self, h, inp, gate_in):
        """
        批量计算所有尺度的 RawBlock 输出候选值。
        h:       (B, num_scales, d_model)
        inp:     (B, d_model)
        gate_in: (B, num_scales, 2*d_model)  拼接后的门控输入
        返回:
            candidate: (B, num_scales, d_model)
        """
        # 三个门 (gate_in: B,ns,2d → W: ns,d,2d → out: B,ns,d)
        f = torch.sigmoid(
            torch.einsum('bnj,nij->bni', gate_in, self.W_forget) + self.b_forget
        )
        i = torch.sigmoid(
            torch.einsum('bnj,nij->bni', gate_in, self.W_input) + self.b_input
        )
        cand = torch.tanh(
            torch.einsum('bnj,nij->bni', gate_in, self.W_cand) + self.b_cand
        )
        return f * h + i * cand

    def _content_gate(self, gate_in):
        """
        批量计算内容门控的写入强度。
        gate_in: (B, num_scales, 2*d_model)
        返回:
            update_strength: (B, num_scales, 1)  sigmoid 值
        """
        h1 = F.gelu(
            torch.einsum('bnj,nij->bni', gate_in, self.gate_W1) + self.gate_b1
        )
        return torch.sigmoid(
            torch.einsum('bni,noi->bno', h1, self.gate_W2) + self.gate_b2
        )

    def forward(self, x, h_prev=None, return_state=False):
        """
        训练模式：全序列前向（时间步仍串行，尺度完全并行）
        x: (B, T)  token ids
        h_prev:   可选，形状 (B, num_scales, d_model) 的初始状态
        return_state: 是否返回最终状态（用于衔接推理）
        """
        B, T = x.shape

        if h_prev is None:
            H = torch.zeros(B, self.num_scales, self.d_model, device=x.device)
        else:
            H = h_prev  # 期望 (B, num_scales, d_model)

        # 嵌入 + 投影，一次性得到所有时间步的输入向量
        x_emb = self.embed(x)                # (B, T, d_model)
        inp_seq = self.input_proj(x_emb)     # (B, T, d_model)

        logits = torch.zeros(B, T, self.output_proj.out_features, device=x.device, dtype=x_emb.dtype)

        for t in range(T):
            inp = inp_seq[:, t, :]            # (B, d_model)

            # 构造 gate_in: 拼接当前状态 H 与输入 inp
            inp_exp = inp.unsqueeze(1).expand(-1, self.num_scales, -1)
            gate_in = torch.cat([H, inp_exp], dim=-1)   # (B, ns, 2*d)

            # 1. 所有尺度的候选值
            candidate = self._raw_block(H, inp, gate_in)

            # 2. 内容门控写入强度
            update_strength = self._content_gate(gate_in)  # (B, ns, 1)

            # 3. 软更新
            H = update_strength * candidate + (1 - update_strength) * H

            # 4. 融合所有尺度状态并投影到词表
            fused = self.fusion_norm(self.fusion(H.reshape(B, -1)))
            logits[:, t, :] = self.output_proj(fused)

        if return_state:
            return logits, H
        return logits

    @torch.no_grad()
    def generate_step(self, token, h_prev):
        """
        推理模式：单步前向 O(1)
        token: (B, 1)  当前 token id
        h_prev: (B, num_scales, d_model)  上一时刻的状态
        返回:
            logits: (B, vocab_size)
            next_h: (B, num_scales, d_model)
        """
        B = token.size(0)
        x_emb = self.embed(token).squeeze(1)         # (B, d_model)
        inp = self.input_proj(x_emb)                 # (B, d_model)

        # 拼接
        inp_exp = inp.unsqueeze(1).expand(-1, self.num_scales, -1)
        gate_in = torch.cat([h_prev, inp_exp], dim=-1)

        # 候选值
        candidate = self._raw_block(h_prev, inp, gate_in)

        # 内容门控
        update_strength = self._content_gate(gate_in)  # (B, ns, 1)

        # 硬 / 软选择
        if self.hard_infer:
            update = (update_strength > 0.5).float()
            next_h = update * candidate + (1 - update) * h_prev
        else:
            next_h = update_strength * candidate + (1 - update_strength) * h_prev

        # 融合 + 输出
        fused = self.fusion_norm(self.fusion(next_h.reshape(B, -1)))
        logits = self.output_proj(fused)
        return logits, next_h


# ============================================================
# 简单使用示例（训练 + 推理）
# ============================================================
if __name__ == "__main__":
    VOCAB = 23005
    B, T = 4, 384

    model = FRSM_V6_Fast(vocab_size=VOCAB, d_model=256, num_scales=4)
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

    # ---- 训练 ----
    x = torch.randint(0, VOCAB, (B, T))
    logits, final_h = model(x, return_state=True)
    print(f"训练输出 shape: {logits.shape}")   # (4, 384, 23005)
    print(f"最终状态 shape: {final_h.shape}")  # (4, 4, 256)

    # ---- 推理（续写） ----
    token = torch.tensor([[42]])  # 起始 token
    h = None
    for step in range(10):
        if h is None:
            # 第一步可以用 forward 取最后一个 logits，或直接用 generate_step（状态初始化为零）
            # 这里直接用 generate_step 并传入零状态
            h = torch.zeros(1, model.num_scales, model.d_model)
        logits, h = model.generate_step(token, h)
        token = logits.argmax(dim=-1, keepdim=True)
        print(f"Step {step+1}: token id = {token.item()}")