"""
FRSM V6a MoE — Sparse Mixture-of-Experts Content-Gated Multi-Scale State Machine
===============================================================================
真稀疏条件计算版本（True Sparse MoE） + 共享专家（DeepSeek-MoE 风格）：

  * n_experts 个路由专家（参数沿专家维度堆叠），Router 为每个 token 路由到 top-k
  * **昂贵的矩阵乘（状态转移 / 门控 / 融合）只对选中的 k 个路由专家计算**
    —— 通过 gather 选中专家的参数与状态实现条件计算
  * 未被选中的路由专家状态保持冻结
  * n_shared 个共享专家：对所有 token 始终激活、不经过路由，捕获通用知识
    —— 用高效堆叠 einsum 计算（无需 gather，内存友好）
  * 输出 = 共享专家输出 + 路由专家加权输出（相加）
  * 状态更新用可微的 one-hot scatter（仅加法，开销可忽略）
  * 负载均衡辅助损失 (Switch Transformer 风格，仅作用于路由专家)
  * 训练时可选 noisy gating

收益：相比"计算全部专家"的伪稀疏，前向 FLOPs 降低约 (1 - k/E) 的路由专家部分；
      共享专家保证每个 token 都有稳定的通用表征通路。

接口：
    forward(x, h_prev=None, return_state=False) -> logits[, (H_routed, H_shared)]
    generate_step(token, h_prev) -> (logits, (H_routed, H_shared))
辅助损失存于 self.aux_loss。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FRSM_V6_MoE(nn.Module):
    """
    Content‑Gated Multi-Scale State Machine + Sparse MoE（真稀疏）+ 共享专家

    参数:
        vocab_size:     词表大小
        d_model:        模型维度（默认 256）
        num_scales:     每个专家内部的并行尺度数（默认 4）
        n_experts:      路由专家数量（默认 16）
        n_activated:    每个 token 激活的路由专家数 top-k（默认 4）
        n_shared:       共享专家数量，始终激活、不路由（默认 1，0 表示不用）
        hard_infer:     推理时是否使用硬阈值写入门
        router_noise:   训练时 router 噪声标准差（0 关闭）
        aux_loss_weight:负载均衡损失权重（仅记录，实际加权由训练脚本控制）
        use_checkpoint: 训练时每步梯度检查点（避免 gather 副本沿 T 步累积 OOM）
        chunk_size:     chunk 并行大小。0=自动设为 sqrt(T)；C 个 token 共享同一路由 + 状态起始点，
                       matmul 维度扩大 C 倍，tensor core 利用率大幅提升。
    """

    def __init__(self, vocab_size, d_model=256, num_scales=4,
                 n_experts=16, n_activated=4, n_shared=1, hard_infer=True,
                 router_noise=1.0, aux_loss_weight=0.01,
                 use_checkpoint=True, chunk_size=0):
        super().__init__()
        assert 1 <= n_activated <= n_experts, "n_activated must be in [1, n_experts]"
        assert n_shared >= 0, "n_shared must be >= 0"
        self.d_model = d_model
        self.num_scales = num_scales
        self.n_experts = n_experts
        self.n_activated = n_activated
        self.n_shared = n_shared
        self.hard_infer = hard_infer
        self.router_noise = router_noise
        self.aux_loss_weight = aux_loss_weight
        # 训练时对每个时间步启用梯度检查点：避免 gather 参数副本沿 T 步累积导致 OOM
        # （代价：前向重算一次。推理不受影响）
        self.use_checkpoint = use_checkpoint
        # chunk 并行：C 个 token 共享同一路由 + H 起始点，每步 matmul 维度扩大 C 倍
        self.chunk_size = chunk_size  # 0 = auto: floor(sqrt(T)) 在 forward 中决定

        # 占位：每次 forward 后更新
        self.aux_loss = torch.tensor(0.0)

        E, S, D = n_experts, num_scales, d_model
        d_hidden = D // 4

        # ---- 共享 embedding & 输入投影 ----
        self.embed = nn.Embedding(vocab_size, D)
        self.input_proj = nn.Linear(D, D)

        # ---- 路由专家参数：形状 (n_experts, num_scales, d, 2d) ----
        self.W_forget = nn.Parameter(torch.empty(E, S, D, 2 * D))
        self.b_forget = nn.Parameter(torch.empty(E, S, D))
        self.W_input = nn.Parameter(torch.empty(E, S, D, 2 * D))
        self.b_input = nn.Parameter(torch.empty(E, S, D))
        self.W_cand = nn.Parameter(torch.empty(E, S, D, 2 * D))
        self.b_cand = nn.Parameter(torch.empty(E, S, D))

        # 内容门控网络（两层）：(n_experts, num_scales, ...)
        self.gate_W1 = nn.Parameter(torch.empty(E, S, d_hidden, 2 * D))
        self.gate_b1 = nn.Parameter(torch.empty(E, S, d_hidden))
        self.gate_W2 = nn.Parameter(torch.empty(E, S, 1, d_hidden))
        self.gate_b2 = nn.Parameter(torch.empty(E, S, 1))

        # ---- 每个路由专家独立的融合投影：(n_experts, num_scales*d, d) ----
        self.fusion_W = nn.Parameter(torch.empty(E, S * D, D))
        self.fusion_b = nn.Parameter(torch.empty(E, D))

        # ---- 共享专家（始终激活，不路由）：(n_shared, num_scales, ...) ----
        if n_shared > 0:
            NS = n_shared
            self.W_forget_sh = nn.Parameter(torch.empty(NS, S, D, 2 * D))
            self.b_forget_sh = nn.Parameter(torch.empty(NS, S, D))
            self.W_input_sh = nn.Parameter(torch.empty(NS, S, D, 2 * D))
            self.b_input_sh = nn.Parameter(torch.empty(NS, S, D))
            self.W_cand_sh = nn.Parameter(torch.empty(NS, S, D, 2 * D))
            self.b_cand_sh = nn.Parameter(torch.empty(NS, S, D))
            self.gate_W1_sh = nn.Parameter(torch.empty(NS, S, d_hidden, 2 * D))
            self.gate_b1_sh = nn.Parameter(torch.empty(NS, S, d_hidden))
            self.gate_W2_sh = nn.Parameter(torch.empty(NS, S, 1, d_hidden))
            self.gate_b2_sh = nn.Parameter(torch.empty(NS, S, 1))
            self.fusion_W_sh = nn.Parameter(torch.empty(NS, S * D, D))
            self.fusion_b_sh = nn.Parameter(torch.empty(NS, D))

        # ---- Router（门控网络，仅作用于路由专家）----
        self.router = nn.Linear(D, E)

        # ---- 输出 ----
        self.output_norm = nn.LayerNorm(D)
        self.output_proj = nn.Linear(D, vocab_size)

        self._init_weights()

    # ------------------------------------------------------------------ init
    def _init_stack(self, param, count):
        """对 (count, S, ...) 堆叠参数逐 (专家, 尺度) 切片做 kaiming 初始化"""
        for e in range(count):
            for s in range(self.num_scales):
                nn.init.kaiming_uniform_(param[e, s], a=math.sqrt(5))

    def _init_weights(self):
        # 路由专家状态机权重
        for p in [self.W_forget, self.W_input, self.W_cand,
                  self.gate_W1, self.gate_W2]:
            self._init_stack(p, self.n_experts)
        # 路由专家融合权重（按专家切片）
        for e in range(self.n_experts):
            nn.init.kaiming_uniform_(self.fusion_W[e], a=math.sqrt(5))

        # 共享专家权重
        if self.n_shared > 0:
            for p in [self.W_forget_sh, self.W_input_sh, self.W_cand_sh,
                      self.gate_W1_sh, self.gate_W2_sh]:
                self._init_stack(p, self.n_shared)
            for e in range(self.n_shared):
                nn.init.kaiming_uniform_(self.fusion_W_sh[e], a=math.sqrt(5))

        # 子模块 bias -> 0（router / input_proj / output_proj）
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.zeros_(param)

        # 自定义 bias（路由专家）
        nn.init.zeros_(self.b_cand)
        nn.init.zeros_(self.gate_b1)
        nn.init.zeros_(self.gate_b2)
        nn.init.zeros_(self.fusion_b)
        nn.init.constant_(self.b_forget, 1.0)   # 初始偏向"保留"
        nn.init.constant_(self.b_input, -2.0)   # 初始偏向"不写入"

        # 自定义 bias（共享专家）
        if self.n_shared > 0:
            nn.init.zeros_(self.b_cand_sh)
            nn.init.zeros_(self.gate_b1_sh)
            nn.init.zeros_(self.gate_b2_sh)
            nn.init.zeros_(self.fusion_b_sh)
            nn.init.constant_(self.b_forget_sh, 1.0)
            nn.init.constant_(self.b_input_sh, -2.0)

        # Router: 小方差初始化，初始路由接近均匀
        nn.init.normal_(self.router.weight, mean=0.0, std=0.02)

        # Embedding
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
        # 输入 / 输出投影
        nn.init.kaiming_uniform_(self.input_proj.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.output_proj.weight, a=math.sqrt(5))

    # ------------------------------------------------------------- 路由
    def _route(self, inp):
        """
        Router：返回选中专家索引、权重、mask、概率
        inp: (B, D)
        返回:
            top_idx: (B, k)     选中的专家索引
            top_w:   (B, k)     归一化后的专家权重
            mask:    (B, E)     选中为 1，其余 0
            probs:   (B, E)     原始 softmax 概率（用于 aux loss）
        """
        logits = self.router(inp)                       # (B, E)
        if self.training and self.router_noise > 0:
            logits = logits + torch.randn_like(logits) * self.router_noise
        probs = F.softmax(logits, dim=-1)               # (B, E)

        k = self.n_activated
        topk_vals, top_idx = probs.topk(k, dim=-1)      # (B, k)
        top_w = topk_vals / (topk_vals.sum(dim=-1, keepdim=True) + 1e-9)

        mask = torch.zeros_like(probs)
        mask.scatter_(1, top_idx, 1.0)
        return top_idx, top_w, mask, probs

    # --------------------------------------------- 专家条件计算（真稀疏）
    def _expert_transition(self, H_sel, inp, top_idx):
        """
        只对选中的 k 个专家计算状态转移（矩阵乘仅作用于 k 个专家）。
        H_sel:   (B, k, S, D)   选中专家的当前状态
        inp:     (B, D)
        top_idx: (B, k)
        返回:
            candidate: (B, k, S, D)   f*H + i*cand
            strength:  (B, k, S, 1)   内容门控写入强度
        """
        B, k, S, D = H_sel.shape

        # gather 选中专家的参数：param[top_idx] -> (B, k, ...)
        W_f = self.W_forget[top_idx]    # (B,k,S,D,2D)
        W_i = self.W_input[top_idx]
        W_c = self.W_cand[top_idx]
        b_f = self.b_forget[top_idx]    # (B,k,S,D)
        b_i = self.b_input[top_idx]
        b_c = self.b_cand[top_idx]
        gW1 = self.gate_W1[top_idx]     # (B,k,S,dh,2D)
        gW2 = self.gate_W2[top_idx]     # (B,k,S,1,dh)
        gb1 = self.gate_b1[top_idx]     # (B,k,S,dh)
        gb2 = self.gate_b2[top_idx]     # (B,k,S,1)

        inp_exp = inp.unsqueeze(1).unsqueeze(2).expand(B, k, S, D)   # (B,k,S,D)
        gate_in = torch.cat([H_sel, inp_exp], dim=-1)                # (B,k,S,2D)

        f = torch.sigmoid(torch.einsum('bksj,bksij->bksi', gate_in, W_f) + b_f)
        i = torch.sigmoid(torch.einsum('bksj,bksij->bksi', gate_in, W_i) + b_i)
        cand = torch.tanh(torch.einsum('bksj,bksij->bksi', gate_in, W_c) + b_c)
        candidate = f * H_sel + i * cand                             # (B,k,S,D)

        h1 = F.gelu(torch.einsum('bksj,bksij->bksi', gate_in, gW1) + gb1)
        strength = torch.sigmoid(torch.einsum('bksi,bksoi->bkso', h1, gW2) + gb2)
        return candidate, strength

    def _expert_fusion(self, state, top_idx):
        """
        只对选中的 k 个专家做融合投影。
        state:   (B, k, S, D)
        top_idx: (B, k)
        返回:    (B, k, D)
        """
        B, k = state.shape[:2]
        S, D = self.num_scales, self.d_model
        fW = self.fusion_W[top_idx]            # (B,k,S*D,D)
        fb = self.fusion_b[top_idx]            # (B,k,D)
        H_flat = state.reshape(B, k, S * D)
        return torch.einsum('bkp,bkpi->bki', H_flat, fW) + fb

    @staticmethod
    def _scatter_state(candidate_state, top_idx, E):
        """
        将选中专家的候选状态可微地散射回 (E, B, S, D) 全专家布局。
        candidate_state: (B, k, S, D)
        top_idx:         (B, k)
        返回: (E, B, S, D)，非选中位置为 0。
        """
        one_hot = F.one_hot(top_idx, E).to(candidate_state.dtype)          # (B,k,E)
        scattered = (candidate_state.unsqueeze(2)
                     * one_hot.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)     # (B,E,S,D)
        return scattered.permute(1, 0, 2, 3)                               # (E,B,S,D)

    # --------------------------------------------------------- 共享专家计算
    def _shared_forward(self, H_shared, inp):
        """
        共享专家：始终全量激活（不路由），用高效堆叠 einsum 计算。
        H_shared: (NS, B, S, D)   共享专家当前状态
        inp:      (B, D)
        返回:
            new_H_shared: (NS, B, S, D)
            shared_combined: (B, D)  所有共享专家融合输出之和
        """
        NS = self.n_shared
        B = inp.size(0)
        S, D = self.num_scales, self.d_model

        inp_exp = inp.unsqueeze(0).unsqueeze(2).expand(NS, B, S, D)        # (NS,B,S,D)
        gate_in = torch.cat([H_shared, inp_exp], dim=-1)                   # (NS,B,S,2D)

        # bias (NS,S,D) -> unsqueeze(1) -> (NS,1,S,D) 以广播到 (NS,B,S,D)
        f = torch.sigmoid(torch.einsum('ebsj,esij->ebsi', gate_in, self.W_forget_sh)
                          + self.b_forget_sh.unsqueeze(1))
        i = torch.sigmoid(torch.einsum('ebsj,esij->ebsi', gate_in, self.W_input_sh)
                          + self.b_input_sh.unsqueeze(1))
        cand = torch.tanh(torch.einsum('ebsj,esij->ebsi', gate_in, self.W_cand_sh)
                          + self.b_cand_sh.unsqueeze(1))
        candidate = f * H_shared + i * cand                                # (NS,B,S,D)

        h1 = F.gelu(torch.einsum('ebsj,esij->ebsi', gate_in, self.gate_W1_sh)
                    + self.gate_b1_sh.unsqueeze(1))
        strength = torch.sigmoid(torch.einsum('ebsi,esoi->ebso', h1, self.gate_W2_sh)
                                 + self.gate_b2_sh.unsqueeze(1))
        new_H_shared = strength * candidate + (1 - strength) * H_shared    # (NS,B,S,D)

        H_flat = new_H_shared.reshape(NS, B, S * D)
        fused = torch.einsum('ebk,eki->ebi', H_flat, self.fusion_W_sh) + self.fusion_b_sh.unsqueeze(1)
        shared_combined = fused.sum(dim=0)                                 # (B,D)
        return new_H_shared, shared_combined

    # ----------------------------------------------- 单时间步计算体（路由+共享）
    def _step(self, H, H_shared, inp_t):
        """
        一个时间步：路由专家（稀疏 k 个）+ 共享专家（全量）。可作为梯度检查点单元。
        H:        (E, B, S, D)   路由专家状态
        H_shared: (NS, B, S, D)  共享专家状态（NS=0 时为 None）
        inp_t:    (B, D)
        返回:
            new_H:        (E, B, S, D)
            new_H_shared: (NS, B, S, D) 或 None
            combined:     (B, D)   路由+共享相加后的特征
            probs:        (B, E)
            mask:         (B, E)
        """
        E, k = self.n_experts, self.n_activated
        B = inp_t.size(0)

        # ---- 路由专家（稀疏）----
        top_idx, top_w, mask, probs = self._route(inp_t)
        idx_b = torch.arange(B, device=inp_t.device).unsqueeze(1).expand(B, k)
        H_sel = H[top_idx, idx_b]                                          # (B,k,S,D)
        candidate, strength = self._expert_transition(H_sel, inp_t, top_idx)
        candidate_state = strength * candidate + (1 - strength) * H_sel    # (B,k,S,D)

        scattered = self._scatter_state(candidate_state, top_idx, E)       # (E,B,S,D)
        mask_e = mask.t().unsqueeze(-1).unsqueeze(-1)                      # (E,B,1,1)
        new_H = mask_e * scattered + (1 - mask_e) * H

        fused = self._expert_fusion(candidate_state, top_idx)              # (B,k,D)
        routed_combined = (top_w.unsqueeze(-1) * fused).sum(dim=1)         # (B,D)

        # ---- 共享专家（全量）----
        if self.n_shared > 0:
            new_H_shared, shared_combined = self._shared_forward(H_shared, inp_t)
            combined = routed_combined + shared_combined
        else:
            new_H_shared = None
            combined = routed_combined
        return new_H, new_H_shared, combined, probs, mask

    # --------------------------------------------- Chunk 并行（路由共享 + H 固定）
    def _chunk_step(self, H, H_shared, inp_chunk):
        """
        C 个 token 共享同一路由 + 同一起始 H，matmul 维度扩大 C 倍。
        内部将 C 展平到 batch 维度 (B*C)，复用 _expert_transition / _fusion / _shared。
        inp_chunk: (B, C, D)   — C 个连续 token
        H:         (E, B, S, D) — chunk 起始的路由专家状态
        返回:
            new_H:        (E, B, S, D)  — chunk 内最后一个 token 更新后的状态
            new_H_shared: (NS, B, S, D) 或 None
            combined:     (B, C, D)      — 所有 C 个 token 的输出特征
            probs:        (B, E)         — 负载均衡用（来自 chunk 的第一个 token）
            mask:         (B, E)
        """
        B, C, D = inp_chunk.shape
        E = self.n_experts; k = self.n_activated; S = self.num_scales; NS = self.n_shared

        # 用 chunk 的第一个 token 做路由
        first_inp = inp_chunk[:, 0, :]
        top_idx, top_w, mask, probs = self._route(first_inp)

        # 展平 C→batch 维度：B*C 个 token 共享 top_idx/top_w/mask
        inp_flat = inp_chunk.reshape(B * C, D)
        top_idx_flat = top_idx.unsqueeze(1).expand(B, C, k).reshape(B * C, k)
        top_w_flat = top_w.unsqueeze(1).expand(B, C, k).reshape(B * C, k)
        mask_flat = mask.unsqueeze(1).expand(B, C, E).reshape(B * C, E)

        # 平铺状态：每个 token 都从 chunk 起始 H 出发
        H_flat = H.unsqueeze(2).expand(E, B, C, S, D).reshape(E, B * C, S, D)
        H_shared_flat = (H_shared.unsqueeze(2).expand(NS, B, C, S, D).reshape(NS, B * C, S, D)
                         if NS > 0 else None)

        # ---- 路由专家（稀疏）----
        idx_b = torch.arange(B * C, device=inp_chunk.device).unsqueeze(1).expand(B * C, k)
        H_sel = H_flat[top_idx_flat, idx_b]
        candidate, strength = self._expert_transition(H_sel, inp_flat, top_idx_flat)
        candidate_state = strength * candidate + (1 - strength) * H_sel

        scattered = self._scatter_state(candidate_state, top_idx_flat, E)
        mask_e = mask_flat.t().unsqueeze(-1).unsqueeze(-1)
        new_H_flat = mask_e * scattered + (1 - mask_e) * H_flat

        fused = self._expert_fusion(candidate_state, top_idx_flat)
        routed_combined = (top_w_flat.unsqueeze(-1) * fused).sum(dim=1)

        # ---- 共享专家（全量）----
        if NS > 0:
            new_H_shared_flat, shared_combined = self._shared_forward(H_shared_flat, inp_flat)
            combined_flat = routed_combined + shared_combined
        else:
            new_H_shared_flat = None
            combined_flat = routed_combined

        # 恢复 (B, C, D) 布局
        combined = combined_flat.reshape(B, C, D)

        # 只保留 chunk 内最后一个 token 的状态传递给下一 chunk
        last_idx = torch.arange(B, device=inp_chunk.device) * C + (C - 1)
        new_H = new_H_flat[:, last_idx, :, :]
        new_H_shared = (new_H_shared_flat[:, last_idx, :, :] if NS > 0 else None)
        return new_H, new_H_shared, combined, probs, mask

    # ----------------------------------------------------------------- 前向
    def forward(self, x, h_prev=None, return_state=False):
        """
        训练模式：全序列前向（时间步串行；路由专家仅 k 个稀疏计算 + 共享专家全量）
        训练时默认启用梯度检查点（use_checkpoint），使 gather 真稀疏训练不会 OOM。
        x: (B, T) token ids
        h_prev: 可选，(H_routed, H_shared) 初始状态元组
        return_state: 是否返回最终状态 (H_routed, H_shared)
        """
        B, T = x.shape
        E, S, D = self.n_experts, self.num_scales, self.d_model
        NS = self.n_shared

        x_emb = self.embed(x)                 # (B, T, D)
        inp_seq = self.input_proj(x_emb)      # (B, T, D)  浮点 dtype（autocast 下为 bf16）

        # 状态张量 dtype 跟随 inp_seq（浮点），而非输入 token 的 int64
        if h_prev is None:
            H = torch.zeros(E, B, S, D, device=x.device, dtype=inp_seq.dtype)
            H_shared = (torch.zeros(NS, B, S, D, device=x.device, dtype=inp_seq.dtype)
                        if NS > 0 else None)
        else:
            H, H_shared = h_prev

        out_features = self.output_proj.out_features
        logits = torch.zeros(B, T, out_features, device=x.device, dtype=inp_seq.dtype)

        # 负载均衡损失用 float32 累加，避免 bf16 长序列累加丢精度
        aux_accum = torch.zeros((), device=x.device, dtype=torch.float32)

        # 自动决定 chunk 大小：0 = sqrt(T)
        C = self.chunk_size if self.chunk_size > 0 else max(1, int(math.sqrt(T)))

        for t_start in range(0, T, C):
            t_end = min(t_start + C, T)
            inp_chunk = inp_seq[:, t_start:t_end, :]                # (B, C_actual, D)

            if self.training and self.use_checkpoint:
                new_H, new_H_shared, combined_chunk, probs, mask = \
                    torch.utils.checkpoint.checkpoint(
                        self._chunk_step, H, H_shared, inp_chunk, use_reentrant=False
                    )
            else:
                new_H, new_H_shared, combined_chunk, probs, mask = \
                    self._chunk_step(H, H_shared, inp_chunk)
            H = new_H
            H_shared = new_H_shared
            logits[:, t_start:t_end, :] = self.output_proj(self.output_norm(combined_chunk))

            # 负载均衡辅助损失（每 chunk 一次路由，float32）
            tokens_per_expert = mask.float().mean(dim=0)        # (E,)
            probs_per_expert = probs.float().mean(dim=0)        # (E,)
            aux_accum = aux_accum + E * torch.sum(tokens_per_expert * probs_per_expert)

        self.aux_loss = aux_accum / max(1, (T + C - 1) // C)  # 按 chunk 数量归一化

        if return_state:
            return logits, (H, H_shared)
        return logits

    # --------------------------------------------------------------- 推理
    @torch.no_grad()
    def generate_step(self, token, h_prev):
        """
        推理模式：单步前向 O(1)，路由专家仅 k 个 + 共享专家全量（无检查点、无梯度）
        token:  (B, 1)
        h_prev: (H_routed, H_shared)
        返回:   logits (B, vocab), (new_H_routed, new_H_shared)
        """
        E, k = self.n_experts, self.n_activated
        B = token.size(0)

        H, H_shared = h_prev

        x_emb = self.embed(token).squeeze(1)      # (B, D)
        inp = self.input_proj(x_emb)              # (B, D)

        # ---- 路由专家（稀疏）----
        top_idx, top_w, mask, probs = self._route(inp)
        idx_b = torch.arange(B, device=token.device).unsqueeze(1).expand(B, k)
        H_sel = H[top_idx, idx_b]            # (B,k,S,D)

        candidate, strength = self._expert_transition(H_sel, inp, top_idx)
        upd = (strength > 0.5).float() if self.hard_infer else strength
        candidate_state = upd * candidate + (1 - upd) * H_sel

        scattered = self._scatter_state(candidate_state, top_idx, E)
        mask_e = mask.t().unsqueeze(-1).unsqueeze(-1)
        new_H = mask_e * scattered + (1 - mask_e) * H

        fused = self._expert_fusion(candidate_state, top_idx)
        routed_combined = (top_w.unsqueeze(-1) * fused).sum(dim=1)

        # ---- 共享专家（全量）----
        if self.n_shared > 0:
            new_H_shared, shared_combined = self._shared_forward(H_shared, inp)
            combined = routed_combined + shared_combined
        else:
            new_H_shared = None
            combined = routed_combined

        logits = self.output_proj(self.output_norm(combined))
        return logits, (new_H, new_H_shared)


# ============================================================
# 简单使用示例（训练 + 推理）
# ============================================================
if __name__ == "__main__":
    VOCAB = 23005
    B, T = 4, 64

    model = FRSM_V6_MoE(vocab_size=VOCAB, d_model=256, num_scales=4,
                        n_experts=16, n_activated=4, n_shared=1)
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

    # ---- 训练 ----
    x = torch.randint(0, VOCAB, (B, T))
    logits, (H_routed, H_shared) = model(x, return_state=True)
    print(f"训练输出 shape: {logits.shape}")        # (4, 64, 23005)
    print(f"路由专家状态 shape: {H_routed.shape}")   # (16, 4, 4, 256)
    print(f"共享专家状态 shape: {H_shared.shape}")   # (1, 4, 4, 256)
    print(f"辅助损失 (负载均衡): {model.aux_loss.item():.4f}")

    # ---- 推理（续写） ----
    token = torch.tensor([[42]])
    h_r = torch.zeros(model.n_experts, 1, model.num_scales, model.d_model)
    h_s = torch.zeros(model.n_shared, 1, model.num_scales, model.d_model)
    for step in range(10):
        logits_step, (h_r, h_s) = model.generate_step(token, (h_r, h_s))
        token = logits_step.argmax(dim=-1, keepdim=True)
        print(f"Step {step + 1}: token id = {token.item()}")
