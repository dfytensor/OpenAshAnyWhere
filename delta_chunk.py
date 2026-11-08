# OpenASH v3.x — 自研分块 delta-rule（纯 torch，无 FLA 依赖）
#
# 数学结构逐行锚定 FLA chunk_delta_h.py / wy_fast.py / chunk_fwd.py（2026-11-08）：
#   状态递推（fwd_h kernel 核心三条语句）：
#       S_start = S                               # chunk 起始状态存档（供 inter 输出与残差）
#       v_res = u − w @ S_start                   # 相对 chunk 起始状态的残差（v_new）
#       S ← S_start·exp(g_last) + kᵀ @ (v_res·exp(g_last − g_t))
#   WY 变换（recompute_w_u kernel）：
#       u = A @ (β·v)，w = A @ (β·exp(g)·k)
#   WY 逆矩阵 A（intra，DeltaNet App.B）：A = (I − tril(diag(β)·K̃·(diag(β)·K̃)ᵀ, −1))⁻¹
#       其中 (K̃K̃ᵀ)_{s,t} = (k_s·k_t)·exp(g_s − g_t)，s > t 严格下三角；单位下三角 → solve_triangular
#   输出读出：o_t = exp(g_t)·q_t·S_start + Σ_{s≤t} exp(g_t−g_s)·β_s·(q_t·k_s)·v_res_s
#
# 与手工 Triton 反向的两次失败（REPORT_v35.md §9）对照：纯 torch + autograd 无需
# 手工 λ 递推——NT=T/BT 个 chunk 的顺序步（本仓库 T=384、BT=64 → NT=6）autograd
# 图极浅，反向自动可行。速度低于 FLA 融合 kernel（预计 2-4×），但可训练、可自持
# （去 flash-linear-attention 训练依赖）、O(1) 推理语义一致。
import torch
import torch.nn.functional as F


def chunked_delta_rule(q, k, v, g, beta, initial_state=None, output_final_state=False, bt=64):
    """分块 gated delta rule（纯 torch，可微，autograd 反向）。

    布局与 FLA chunk_gated_delta_rule 一致:
        q, k: (B, T, H, K)  v: (B, T, H, V)  fp32
        g:    (B, T, H) 逐 token log 域衰减（自然底）
        beta: (B, T, H) 写入强度 ∈ (0,1)
        initial_state: (B, H, K, V) 或 None
    返回:
        o: (B, T, H, V)；S_final: (B, H, K, V)（output_final_state 时）
    内部 batch 约定：位置维 C=bt 提前，H 作 batch → 全部 matmul 无 einsum 歧义。
    """
    B, T, H, K = k.shape
    V = v.shape[-1]
    NT = (T + bt - 1) // bt
    Tp = NT * bt
    if Tp != T:  # 右侧零填充（g 填 0 → 衰减因子 1；读出截回 T）
        q = F.pad(q, (0, 0, 0, 0, 0, Tp - T))
        k = F.pad(k, (0, 0, 0, 0, 0, Tp - T))
        v = F.pad(v, (0, 0, 0, 0, 0, Tp - T))
        g = F.pad(g, (0, 0, 0, Tp - T))
        beta = F.pad(beta, (0, 0, 0, Tp - T))

    gc = g.view(B, NT, bt, H).cumsum(dim=2).permute(0, 1, 3, 2)   # (B, NT, H, C) 含自身
    g_last = gc[..., -1]                                          # (B, NT, H)

    qh = q.view(B, NT, bt, H, K).permute(0, 1, 3, 2, 4)           # (B, NT, H, C, K)
    kh = k.view(B, NT, bt, H, K).permute(0, 1, 3, 2, 4)
    vh = v.view(B, NT, bt, H, V).permute(0, 1, 3, 2, 4)
    bh = beta.view(B, NT, bt, H).permute(0, 1, 3, 2)              # (B, NT, H, C)

    dev, flt = q.device, torch.float32
    eye = torch.eye(bt, dtype=flt, device=dev)
    tril_s = torch.tril(torch.ones(bt, bt, dtype=torch.bool, device=dev), diagonal=-1)
    tril_le = torch.tril(eye.bool() if False else torch.ones(bt, bt, dtype=torch.bool, device=dev))

    S = initial_state.float().clone() if initial_state is not None \
        else q.new_zeros(B, H, K, V)
    o_chunks = []
    for j in range(NT):
        S_start = S
        gci = gc[:, j]                                            # (B, H, C)
        e_t = torch.exp(gci)                                      # chunk 头对齐
        d_t = torch.exp(g_last[:, j].unsqueeze(-1) - gci)         # chunk 尾对齐
        gd = torch.exp(gci.unsqueeze(-1) - gci.unsqueeze(-2))     # (B,H,C,C) exp(g_s − g_t)

        kbh = kh[:, j] * bh[:, j].unsqueeze(-1)                   # β·k（行 β；列 β 在 w/u 中）
        # L[s,t] = β_s·(k_s·k_t)·e^{g_s−g_t}（锚定 chunk_scaled_dot_kkt_fwd：β 仅乘行）
        M = (kbh @ kh[:, j].transpose(-1, -2)) * gd               # (βk)(k)ᵀ 衰减
        # WY 逆 = (I + L)⁻¹（锚定 FLA solve_tril：正号——chunk 内 delta 修正是加法复合）
        Nm = torch.where(tril_s, M, torch.zeros_like(M)) + eye
        A = torch.linalg.solve_triangular(Nm, eye.expand(Nm.shape),
                                          upper=False, unitriangular=True)

        w = A @ (kbh * e_t.unsqueeze(-1))                         # (B,H,C,K)
        u = A @ (vh[:, j] * bh[:, j].unsqueeze(-1))               # (B,H,C,V)

        v_res = u - w @ S_start                                   # 相对起始状态的残差
        v_al = v_res * d_t.unsqueeze(-1)
        S = S_start * torch.exp(g_last[:, j]).unsqueeze(-1).unsqueeze(-1) \
            + kh[:, j].transpose(-1, -2) @ v_al

        inter = (qh[:, j] * e_t.unsqueeze(-1)) @ S_start          # (B,H,C,V)
        Mqk = (qh[:, j] @ kh[:, j].transpose(-1, -2)) * gd
        Mqk = torch.where(tril_le, Mqk, torch.zeros_like(Mqk))
        intra = Mqk @ v_res   # β 已在 u=A@(βv) 中吸收，读出直接作用于残差
        # 全局缩放（锚定 chunk_fwd_o：scale 默认 K^−0.5，inter/intra 同乘）
        o_chunks.append(((inter + intra) * (K ** -0.5)).permute(0, 2, 1, 3))  # → (B, C, H, V)

    o = torch.cat(o_chunks, dim=1).reshape(B, Tp, H, V)[:, :T]
    return o, (S if output_final_state else None)
