"""OpenASH v3 的 Triton 融合算子：cummax 扫描 + gen_model 多项式混合。

训练加速 / 显存优化的三个来源：
1. cummax 前向按 stride 直接读取（避免 PyTorch 对非连续 view 的强制 contiguous 拷贝），
   backward 只保存 int32 argmax（PyTorch 保存输出 + int64 索引）。
2. e = max(prefix_max(c), state) 的数学改写消除了 cat([state, c]) 的整段拷贝
   （train/infer 两种模式统一为同一公式，与 cat 版逐位等价）。
3. gen_model 的 ~10 个 eager 逐元素 kernel + cat 融合为单个 kernel，
   backward 同样融合，用重算代替保存中间张量。

所有 kernel 对任意 stride 的输入都正确；CPU / 无 Triton 时回退到等价 eager 实现。
"""
import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False

F_softplus = torch.nn.functional.softplus

TRITON_OK = _HAS_TRITON and torch.cuda.is_available()


# ---------------------------------------------------------------------------
# 分块并行 cummax / 衰减最大值扫描（v3.1）
#
# v3.1 归约算子:  e_j = max(γ_j ⊙ e_{j-1}, x_j)   γ_j = exp(-softplus(x_j·v+b)) ∈ (0,1)
#   - 状态可下降 → 根治纯 cummax "单调不减、长程饱和" 的风险（选择性遗忘）
#   - 块级 carry = (γ^CHUNK 乘积 Γ, 衰减加权块内最大 m) 二元组，结合律封闭：
#     compose((Γ1,m1),(Γ2,m2)) = (Γ1Γ2, max(m2, Γ2·m1))
#   - USE_DECAY=False 时 γ≡1，退化为纯 cummax（与旧语义逐位一致）
#   - owner 链（idx）仍全局单调不减 → 逆序段冲洗/winner 注入的反向基建原样复用
#
# 前向 F1: 各块独立逆序扫描 → 块 (Γ, m, owner)（只读 c，不写 e/idx）
# 前向 F2: carry 组合 + 以 carry 为种子的单遍扫描（写 e/owner）
# 反向 B0: 每块流出梯度 = Σ g[j]·(owner[j] < base)
# 反向 B1: 块级 winner 链（含 state）+ 后缀和 + state 梯度
# 反向 B2: 段内逆序冲洗 + winner 注入（无原子操作）
# 反向 B4: γ 梯度（仅 USE_DECAY）：g_γ[j] = g[j]·e_{j-1}，e_{j-1}=e_j/γ_j，
#          经 dγ/dpre = -γ·σ(pre) 链回 c 与 v/b 参数
# ---------------------------------------------------------------------------
if _HAS_TRITON:
    @triton.jit
    def _softplus(pre):
        return tl.where(pre > 0, pre + tl.log(1 + tl.exp(-pre)), tl.log(1 + tl.exp(pre)))

    @triton.jit
    def _cm_local_kernel(c_ptr, v_ptr, b_ptr, cm_ptr, cg_ptr, ca_ptr, cs_ptr,
                         S, C, DH, H, CHUNK,
                         sc0, sc1, sc2, sc3,
                         USE_DECAY: tl.constexpr,
                         BLOCK_DH: tl.constexpr):
        pid = tl.program_id(0)
        ck = tl.program_id(1)
        b = pid // H
        h = pid % H
        offs = tl.arange(0, BLOCK_DH)
        lmask = offs < DH
        pv = tl.load(v_ptr + offs * H + h, mask=lmask, other=0.0).to(tl.float32)
        pb = tl.load(b_ptr + offs * H + h, mask=lmask, other=0.0).to(tl.float32)
        P = tl.full([BLOCK_DH], 1.0, tl.float32)     # 后缀 γ 乘积
        M = tl.full([BLOCK_DH], float('-inf'), tl.float32)
        own = tl.full([BLOCK_DH], -1, tl.int32)
        cs = tl.full([BLOCK_DH], 1.0, tl.float32)    # owner→块末 的 γ 后缀积（B2 注入用）
        base = ck * CHUNK
        for jj in range(0, CHUNK):
            j = base + CHUNK - 1 - jj                # 逆序 → suffix product
            msk = lmask & (j < S)
            x = tl.load(c_ptr + b * sc0 + j * sc2 + h * sc3 + offs * sc1,
                        mask=msk, other=float('-inf')).to(tl.float32)
            x_s = tl.where(msk, x, 0.0)
            if USE_DECAY:
                pre = x_s * pv + pb
                g = tl.where(msk, tl.exp(-_softplus(pre)), 1.0)
            else:
                g = tl.full([BLOCK_DH], 1.0, tl.float32)
            cand = P * x                             # Π_{k>j} γ_k · x_j（不含自身 γ！）
            upd = msk & (cand >= M)                  # 降序扫描 + >= → tie 归最早位置（串行语义）
            own = tl.where(upd, j, own)
            cs = tl.where(upd, P, cs)
            M = tl.maximum(M, cand)
            P = P * g
        m_off = (b * DH + offs) * (C * H) + ck * H + h
        tl.store(cm_ptr + m_off, M, mask=lmask)
        tl.store(cg_ptr + m_off, P, mask=lmask)
        tl.store(ca_ptr + m_off, own, mask=lmask)
        tl.store(cs_ptr + m_off, cs, mask=lmask)

    @triton.jit
    def _cm_fixup_kernel(c_ptr, v_ptr, b_ptr, cm_ptr, cg_ptr, ca_ptr,
                         e_ptr, idx_ptr, st_ptr,
                         S, C, DH, H, CHUNK,
                         sc0, sc1, sc2, sc3,
                         se0, se1, se2, se3,
                         sst0, sst1, sst2, sst3,
                         HAS_INIT: tl.constexpr,
                         USE_DECAY: tl.constexpr,
                         BLOCK_DH: tl.constexpr):
        pid = tl.program_id(0)
        ck = tl.program_id(1)
        b = pid // H
        h = pid % H
        offs = tl.arange(0, BLOCK_DH)
        lmask = offs < DH
        pv = tl.load(v_ptr + offs * H + h, mask=lmask, other=0.0).to(tl.float32)
        pb = tl.load(b_ptr + offs * H + h, mask=lmask, other=0.0).to(tl.float32)
        run = tl.full([BLOCK_DH], float('-inf'), tl.float32)
        cur = tl.full([BLOCK_DH], -1, tl.int32)
        if HAS_INIT:
            run = tl.load(st_ptr + b * sst0 + 0 * sst2 + h * sst3 + offs * sst1,
                          mask=lmask, other=float('-inf')).to(tl.float32)
        for cc in range(0, ck):                      # carry 组合（C 很小）
            m_off = (b * DH + offs) * (C * H) + cc * H + h
            Gm = tl.load(cg_ptr + m_off, mask=lmask, other=1.0).to(tl.float32)
            M = tl.load(cm_ptr + m_off, mask=lmask, other=float('-inf')).to(tl.float32)
            p = tl.load(ca_ptr + m_off, mask=lmask, other=-1)
            win = M > Gm * run
            cur = tl.where(win, p, cur)
            run = tl.maximum(Gm * run, M)
        base = ck * CHUNK
        for j in range(0, CHUNK):
            pos = base + j
            msk = lmask & (pos < S)
            x = tl.load(c_ptr + b * sc0 + pos * sc2 + h * sc3 + offs * sc1,
                        mask=msk, other=float('-inf')).to(tl.float32)
            x_s = tl.where(msk, x, 0.0)
            if USE_DECAY:
                pre = x_s * pv + pb
                g = tl.exp(-_softplus(pre))
            else:
                g = 1.0
            cand = g * run
            win = x > cand                           # tie → carry/init 胜（首次达峰语义）
            cur = tl.where(msk & win, pos, cur)
            run = tl.maximum(cand, tl.where(msk, x, float('-inf')))
            tl.store(e_ptr + b * se0 + pos * se2 + h * se3 + offs * se1,
                     run.to(e_ptr.dtype.element_ty), mask=msk)
            tl.store(idx_ptr + (b * DH + offs) * (S * H) + pos * H + h, cur, mask=msk)

    # ---------------------------------------------------------------------------
    # 反向：精确串行 BPTT（单 kernel，降序递归）。
    #   G_j = ge_j + γ_{j+1}·G_{j+1}·[carry@{j+1}]
    #   dL/dx_j = G_j·[owner_j == j]（输入分支胜出）
    #   dL/dγ_j = G_j·e_{j−1}·[carry@j] → dL/dpre_j = −G_j·e_j·σ_j·[carry@j]
    #   dL/db_j = dL/dpre_j;  dL/dv_j = dL/dpre_j·x_j（workspace 行求和）
    #   state 梯度 = γ_0·G_0·[carry@0]（carry@0: owner_0 != 0）
    # G 的 γ 链自动包含全部跨块衰减 ✓ 无 scatter/段机制
    # ---------------------------------------------------------------------------
    @triton.jit
    def _cm_bwd_kernel(ge_ptr, e_ptr, c_ptr, v_ptr, b_ptr, idx_ptr, st_ptr,
                       gc_ptr, gs_ptr, ws_gv_ptr, ws_gb_ptr,
                       S, DH, H,
                       sg0, sg1, sg2, sg3,
                       se0, se1, se2, se3,
                       sc0, sc1, sc2, sc3,
                       sst0, sst1, sst2, sst3,
                       HAS_INIT: tl.constexpr,
                       USE_DECAY: tl.constexpr,
                       BLOCK_DH: tl.constexpr):
        pid = tl.program_id(0)
        b = pid // H
        h = pid % H
        offs = tl.arange(0, BLOCK_DH)
        lmask = offs < DH
        pv = tl.load(v_ptr + offs * H + h, mask=lmask, other=0.0).to(tl.float32)
        pb = tl.load(b_ptr + offs * H + h, mask=lmask, other=0.0).to(tl.float32)
        G = tl.zeros([BLOCK_DH], tl.float32)
        Gcarry = tl.zeros([BLOCK_DH], tl.float32)
        gv_acc = tl.zeros([BLOCK_DH], tl.float32)
        gb_acc = tl.zeros([BLOCK_DH], tl.float32)
        for jj in range(0, S):
            j = S - 1 - jj
            msk = lmask & (j >= 0)
            g_j = tl.load(ge_ptr + b * sg0 + j * sg2 + h * sg3 + offs * sg1,
                          mask=msk, other=0.0).to(tl.float32)
            G = g_j + Gcarry                           # G_j = ge_j + γ_{j+1}·G_{j+1}·[carry@{j+1}]
            i = tl.load(idx_ptr + (b * DH + offs) * (S * H) + j * H + h, mask=msk, other=0)
            carry_j = i != j
            # x 路径：输入分支胜出（owner == j）
            tl.store(gc_ptr + b * DH * S * H + offs * (S * H) + j * H + h,
                     G, mask=msk & (i == j))
            if USE_DECAY:
                x = tl.load(c_ptr + b * sc0 + j * sc2 + h * sc3 + offs * sc1,
                            mask=msk, other=0.0).to(tl.float32)
                pre = x * pv + pb
                sig = tl.sigmoid(pre)
                gmm = tl.exp(-_softplus(pre))
                e_jm1 = tl.load(e_ptr + b * se0 + (j - 1) * se2 + h * se3 + offs * se1,
                                mask=msk & (j >= 1), other=0.0).to(tl.float32)
                if HAS_INIT:
                    st_v = tl.load(st_ptr + b * sst0 + 0 * sst2 + h * sst3 + offs * sst1,
                                   mask=msk & (j == 0), other=0.0).to(tl.float32)
                    e_jm1 = tl.where(j == 0, st_v, e_jm1)
                ggamma = G * e_jm1                     # dL/dγ_j
                dLdpre_j = ggamma * (-gmm * sig)       # dL/dpre_j = dL/dγ·(−γσ)
                xwin = tl.where(i == j, 1.0, 0.0)
                # x 路径（owner == j）+ γ→pre→c 路径（dLdpre·v）合成一次存储
                tl.store(gc_ptr + b * DH * S * H + offs * (S * H) + j * H + h,
                         G * xwin + dLdpre_j * pv, mask=msk)
                gv_acc += tl.where(msk & carry_j, dLdpre_j * x, 0.0)
                gb_acc += tl.where(msk & carry_j, dLdpre_j, 0.0)
            # 递归: grad into e_{j−1} += γ_j·G_j·[carry@j]
            if USE_DECAY:
                x = tl.load(c_ptr + b * sc0 + j * sc2 + h * sc3 + offs * sc1,
                            mask=msk, other=0.0).to(tl.float32)
                pre = x * pv + pb
                Gcarry = tl.where(msk & carry_j, G * tl.exp(-_softplus(pre)), 0.0)
                G = Gcarry
            else:
                Gcarry = tl.where(msk & carry_j, G, 0.0)
                G = Gcarry
        if HAS_INIT:
            tl.store(gs_ptr + b * DH * H + offs * H + h, G, mask=lmask)
        ws_off = pid * (DH * H) + offs * H + h
        tl.store(ws_gv_ptr + ws_off, gv_acc, mask=lmask)
        tl.store(ws_gb_ptr + ws_off, gb_acc, mask=lmask)
    class _CumMaxFn(torch.autograd.Function):
        """e_j = max(γ_j·e_{j-1}, x_j)（衰减最大值；v=None 时 γ≡1 即纯 cummax）。

        反向（torch 向量化，log 域衰减权重 + scatter，精确）:
          owner_j = idx_j（e_j 的归属位置，全局单调不减）
          x 路径:  dL/dx_p = Σ_{k: owner_k == p} exp(cw_k − cw_p)·ge_k   (+ state 项)
          γ 链:    G_j = ge_j + routed_j;  dL/dγ_j = G_j·e_{j−1}·[owner_j != j]
                   dL/dpre_j = dL/dγ_j·(−γ_j·σ_j) → c/v/b 梯度
          其中 cw = cumsum(log γ)，e_{j−1} = state (j=0)。"""

        @staticmethod
        def forward(ctx, c, state, v, b):
            B, DH, S, H = c.shape
            CHUNK = 128
            C = triton.cdiv(S, CHUNK)
            BDH = triton.next_power_of_2(DH)
            use_decay = v is not None
            e = torch.empty(c.shape, dtype=c.dtype, device=c.device)
            idx = torch.empty((B, DH, S, H), dtype=torch.int32, device=c.device)
            cm = torch.empty((B, DH, C, H), dtype=torch.float32, device=c.device)
            cg = torch.empty((B, DH, C, H), dtype=torch.float32, device=c.device)
            ca = torch.empty((B, DH, C, H), dtype=torch.int32, device=c.device)
            cs = torch.empty((B, DH, C, H), dtype=torch.float32, device=c.device)
            grid = (B * H, C)
            if state is not None:
                s_strides = state.stride()
            else:
                s_strides = (0, 0, 0, 0)
            _cm_local_kernel[grid](c, v if use_decay else c, b if use_decay else c,
                                   cm, cg, ca, cs, S, C, DH, H, CHUNK,
                                   *c.stride(), USE_DECAY=use_decay,
                                   BLOCK_DH=BDH, num_warps=4)
            _cm_fixup_kernel[grid](c, v if use_decay else c, b if use_decay else c,
                                   cm, cg, ca, e, idx, state if state is not None else c,
                                   S, C, DH, H, CHUNK,
                                   *c.stride(), *e.stride(), *s_strides,
                                   HAS_INIT=state is not None, USE_DECAY=use_decay,
                                   BLOCK_DH=BDH, num_warps=4)
            ctx.save_for_backward(c, e, idx, state)
            globals()['_last_idx'] = idx.detach().clone()  # DEBUG
            ctx.v = v
            ctx.b = b
            ctx.has_init = state is not None
            ctx.use_decay = use_decay
            return e

        @staticmethod
        def backward(ctx, ge):
            c, e, idx, state = ctx.saved_tensors
            v, b = ctx.v, ctx.b
            B, DH, S, H = c.shape
            has_init = ctx.has_init
            use_decay = ctx.use_decay

            # 反向：serial 前向重计算 + autograd（与 oracle 完全相同的操作序列 → 按构造精确）
            c_in = c.float().detach().requires_grad_(True)
            v_in = v.float().detach().requires_grad_(True) if use_decay else None
            b_in = b.float().detach().requires_grad_(True) if use_decay else None
            s_in = state.float().detach().requires_grad_(True) if has_init else None

            with torch.enable_grad():
                e_re = decay_serial_ref(c_in, s_in if has_init else None,
                                        v_in if use_decay else None,
                                        b_in if use_decay else None)
                # 纯 cummax 模式（use_decay=False）没有 v/b 依赖（γ≡1 常数），
                # 不把 None 传给 autograd.grad；gv/gb 直接返回 None
                inputs_list = [c_in]
                if use_decay:
                    inputs_list += [v_in, b_in]
                if has_init:
                    inputs_list += [s_in]
                all_grads = torch.autograd.grad(
                    (e_re * ge.float()).sum(),
                    inputs_list,
                    allow_unused=True)

            gc = all_grads[0].to(ge.dtype) if all_grads[0] is not None else torch.zeros_like(ge)
            gv = all_grads[1].to(ge.dtype).reshape(-1) if (use_decay and all_grads[1] is not None) else None
            gb = all_grads[2].to(ge.dtype).reshape(-1) if (use_decay and all_grads[2] is not None) else None
            if has_init:
                s_grad = all_grads[3] if use_decay else all_grads[1]
                gs = s_grad.to(ge.dtype).reshape(B, DH, 1, H) if s_grad is not None \
                    else torch.zeros((B, DH, 1, H), dtype=ge.dtype, device=ge.device)
            else:
                gs = None
            return gc, (gs if has_init else None), gv, gb



    # ---------------------------------------------------------------------------
    # 训练反向 kernel（备用）：单 Triton BPTT（连续衰减，无 owner/scatter 机制）。
    # 公式已经 torch 复算逐项验证正确；但该 kernel 曾在本环境表现出地址相关的
    # 非确定性输出（miscompilation 嫌疑），当前训练路径改用 DecayScanFn 的
    # 纯 torch 仿射前缀扫描反向。保留此 kernel 供后续排查/启用。
    # 前向保存反向所需的全部系数（fp32，torch 精确计算）:
    #   rp_j = run_{j-1}（j=0 时为 state；无 init 时置 -1e30 哨兵）
    #   mx_j = [x_j > γ_j·rp_j]（x 分支胜出；tie → carry 胜）
    #   ρ_j  = (1−mx_j)·γ_j        （递推传播因子，含 mask）
    #   κ_j  = (1−mx_j)·γ_j·σ_j    （γ→pre 链系数，含 mask）
    # 反向递推（每个 (b,h) program 降序扫描）:
    #   R_j    = ge_j + ρ_{j+1}·R_{j+1}
    #   dpre_j = −R_j·rp_j·κ_j
    #   dL/dx_j = mx_j·R_j + dpre_j·v
    #   gv += dpre_j·x_j ; gb += dpre_j
    #   state 梯度 = ρ_0·R_0
    # ---------------------------------------------------------------------------
    # ---------------------------------------------------------------------------
    # 训练反向 kernel（v2，无跨步累加器）：单 Triton BPTT。
    # 公式经 torch 复算逐项验证。v1 kernel 的非确定性定位结论：
    #   循环内逐步 store（gc）跨 launch 恒定正确；仅"循环外对跨步累加寄存器
    #   （gva/gba）的一次性 store"非确定 —— 疑似本环境 Triton 对循环携带累加
    #   寄存器 + 循环出口 store 的 codegen 问题。
    # v2 重构：去掉全部累加器与循环出口 store，逐位置 store：
    #   gc_j  = mx_j·R_j + dpre_j·v      （输入分支 + γ→pre→x 链）
    #   gdx_j = dpre_j·x_j               （gv = Σ_j gdx，torch 归约）
    #   gd_j  = dpre_j                   （gb = Σ_j gd，torch 归约）
    #   r0    = R_0（j==0 时 store；gs = ρ_0·R_0 在 torch 侧合成）
    # ---------------------------------------------------------------------------
    @triton.jit
    def _decay_bwd_kernel(ge_ptr, rp_ptr, mx_ptr, rho_ptr, kap_ptr, kapx_ptr, v_ptr, b_ptr,
                          gc_ptr, gd_ptr, gdx_ptr, r0_ptr,
                          S, DH, H,
                          sg0, sg1, sg2, sg3,
                          BLOCK_DH: tl.constexpr):
        pid = tl.program_id(0)
        b = pid // H
        h = pid % H
        offs = tl.arange(0, BLOCK_DH)
        lmask = offs < DH
        pv = tl.load(v_ptr + offs * H + h, mask=lmask, other=0.0).to(tl.float32)
        pb = tl.load(b_ptr + offs * H + h, mask=lmask, other=0.0).to(tl.float32)
        base = b * sg0 + h * sg3 + offs * sg1
        R = tl.zeros([BLOCK_DH], tl.float32)
        rho_c = tl.zeros([BLOCK_DH], tl.float32)   # ρ_{j+1}（上一步载入）
        for jj in range(0, S):
            j = S - 1 - jj
            off_j = base + j * sg2
            Rj = tl.load(ge_ptr + off_j, mask=lmask, other=0.0).to(tl.float32) \
                + rho_c * R                          # R_j = ge_j + ρ_{j+1}·R_{j+1}
            mx = tl.load(mx_ptr + off_j, mask=lmask, other=0.0).to(tl.float32)
            kp = tl.load(kap_ptr + off_j, mask=lmask, other=0.0).to(tl.float32)
            kpx = tl.load(kapx_ptr + off_j, mask=lmask, other=0.0).to(tl.float32)
            rp = tl.load(rp_ptr + off_j, mask=lmask, other=0.0).to(tl.float32)
            dpre = -Rj * rp * kp                     # 首步无 init: rp=-1e30(有限), κ=0 → 0
            tl.store(gc_ptr + off_j, mx * Rj + dpre * pv, mask=lmask)
            tl.store(gdx_ptr + off_j, -Rj * rp * kpx, mask=lmask)
            tl.store(gd_ptr + off_j, dpre, mask=lmask)
            if j == 0:
                tl.store(r0_ptr + b * (DH * H) + offs * H + h, Rj, mask=lmask)
            rho_c = tl.load(rho_ptr + off_j, mask=lmask, other=0.0).to(tl.float32)
            R = Rj

    # ---------------------------------------------------------------------------
    # 训练反向（纯 torch，无 autograd 图、无 Triton）：
    #   R_j = ge_j + ρ_{j+1}·R_{j+1} 为线性递推 → 反转后是仿射前缀扫描
    #   u_m = a_m + b_m·u_{m-1}，a_m = ge_{S-1-m}，b_m = ρ_{S-m}（m≥1，b_0=0）
    #   用 Hillis–Steele doubling（log2 S 步）向量化，随后逐元素闭式求梯度。
    #   注：曾用单 Triton BPTT kernel 实现同公式（公式经 torch 复算验证正确），
    #   但该 kernel 在本环境表现出地址相关的非确定性输出（miscompilation 嫌疑），
    #   故改回纯 torch 实现。
    # ---------------------------------------------------------------------------
    class DecayScanFn(torch.autograd.Function):
        """训练路径（v3.1 decay）：前向 torch 向量化分块扫描（无递归、建图开销 0），
        反向单 Triton BPTT kernel（`_decay_bwd_kernel`）。

        反向所需的分支 mask 与系数全部由前向以 torch fp32 精确预算并保存 →
        kernel 内无 mask 重算、无 owner idx 依赖，与 autograd oracle 分支一致。"""

        @staticmethod
        def forward(ctx, c, state, v, b):
            B, DH, S, H = c.shape
            use_decay = v is not None
            if TRITON_OK and c.is_cuda:
                # ---- Triton 前向（v3.0 的 F1/F2 kernel，USE_DECAY 分支），输出 fp32 e ----
                CHUNK = 128
                C = triton.cdiv(S, CHUNK)
                BDH = triton.next_power_of_2(DH)
                cc = c.contiguous()
                e = torch.empty(cc.shape, dtype=torch.float32, device=cc.device)
                idx = torch.empty((B, DH, S, H), dtype=torch.int32, device=cc.device)
                cm = torch.empty((B, DH, C, H), dtype=torch.float32, device=cc.device)
                cg = torch.empty((B, DH, C, H), dtype=torch.float32, device=cc.device)
                ca = torch.empty((B, DH, C, H), dtype=torch.int32, device=cc.device)
                cs = torch.empty((B, DH, C, H), dtype=torch.float32, device=cc.device)
                grid = (B * H, C)
                s_strides = state.stride() if state is not None else (0, 0, 0, 0)
                _cm_local_kernel[grid](cc, v if use_decay else cc, b if use_decay else cc,
                                       cm, cg, ca, cs, S, C, DH, H, CHUNK,
                                       *cc.stride(), USE_DECAY=use_decay,
                                       BLOCK_DH=BDH, num_warps=4)
                _cm_fixup_kernel[grid](cc, v if use_decay else cc, b if use_decay else cc,
                                       cm, cg, ca, e, idx, state if state is not None else cc,
                                       S, C, DH, H, CHUNK,
                                       *cc.stride(), *e.stride(), *s_strides,
                                       HAS_INIT=state is not None, USE_DECAY=use_decay,
                                       BLOCK_DH=BDH, num_warps=4)
                c = cc
            else:
                # ---- torch 回退：分块 cumprod/cummax（chunk≤128 保证 pref 不触 clamp）----
                cf = c.float()
                vv = v.float().view(1, DH, 1, H)
                bb = b.float().view(1, DH, 1, H)
                pre = cf * vv + bb
                g = torch.exp(-torch.nn.functional.softplus(pre))    # γ_j
                run = torch.full((B, DH, 1, H), float('-inf'), device=c.device, dtype=torch.float32) \
                    if state is None else state.float()
                e = torch.empty((B, DH, S, H), device=c.device, dtype=torch.float32)
                chunk = 128
                for ck in range((S + chunk - 1) // chunk):
                    lo, hi = ck * chunk, min((ck + 1) * chunk, S)
                    gc_ = g[:, :, lo:hi]
                    pref = torch.cumprod(gc_, dim=2).clamp_min(1e-30)
                    m = torch.cummax(torch.cat([run, cf[:, :, lo:hi] / pref], dim=2),
                                     dim=2).values[:, :, 1:]
                    ec = pref * torch.maximum(run, m)
                    e[:, :, lo:hi] = ec
                    run = ec[:, :, -1:]

            vv = v.float().view(1, DH, 1, H)
            bb = b.float().view(1, DH, 1, H)
            cf = c.float()
            pre = cf * vv + bb
            g = torch.exp(-torch.nn.functional.softplus(pre))    # γ_j
            sig = torch.sigmoid(pre)
            # ---- 反向系数（torch 精确预算）----
            rp0 = torch.full((B, DH, 1, H), -1e30, device=c.device, dtype=torch.float32) \
                if state is None else state.float()
            rp = torch.cat([rp0, e[:, :, :S - 1]], dim=2).contiguous()
            # xwin = [x_j > γ_j·rp_j]：x 分支胜出指示（tie → carry 胜，与推理 kernel 惯例一致）
            xwin = (cf > g * rp).float()
            if state is None:
                xwin[:, :, 0, :] = 1.0       # 无 init 首步 x 必胜（γ·(-1e30) 极小，显式置位防边界）
            mx = xwin.contiguous()
            rho = ((1.0 - xwin) * g).contiguous()
            kap = ((1.0 - xwin) * g * sig).contiguous()
            kapx = ((1.0 - xwin) * g * sig * cf).contiguous()   # κ 预乘 x：gv = Σ(-R·rp·κx)
            ctx.save_for_backward(c, e, state, v, b)
            ctx.has_init = state is not None
            return e.to(c.dtype)

        @staticmethod
        def backward(ctx, ge):
            """显存优化（v3.5）：不再保存 5 个 fp32 系数张量（rp/mx/rho/kap/kapx，
            12 层合计 ~1.8GB），反向由保存的 (c, e, state, v, b) 重算——
            多 ~10 个逐元素 kernel（~0.4ms/层），换 1.8GB 峰值显存。"""
            c, e, state, v, b = ctx.saved_tensors
            B, DH, S, H = c.shape
            gef = ge.float().contiguous()
            cf = c.float().contiguous()
            vv = v.float().view(1, DH, 1, H)
            bb = b.float().view(1, DH, 1, H)
            # ---- 重算反向系数（与 forward 完全相同的公式）----
            pre = cf * vv + bb
            g = torch.exp(-torch.nn.functional.softplus(pre))
            sig = torch.sigmoid(pre)
            rp0 = torch.full((B, DH, 1, H), -1e30, device=c.device, dtype=torch.float32) \
                if state is None else state.float()
            rp = torch.cat([rp0, e.float()[:, :, :S - 1]], dim=2)
            xwin = (cf > g * rp).float()
            if state is None:
                xwin[:, :, 0, :] = 1.0
            mx = xwin
            rho = (1.0 - xwin) * g
            kap = rho * sig
            kapx = kap * cf
            # ---- 单 Triton BPTT kernel：逐步 store（无跨步累加器，稳定性已验证）----
            gc = torch.empty(c.shape, dtype=torch.float32, device=c.device)
            gd = torch.empty(c.shape, dtype=torch.float32, device=c.device)
            gdx = torch.empty(c.shape, dtype=torch.float32, device=c.device)
            r0 = torch.empty(B * DH * H, dtype=torch.float32, device=c.device)
            BDH = triton.next_power_of_2(DH)
            _decay_bwd_kernel[(B * H,)](gef, rp.contiguous(), mx.contiguous(),
                                        rho.contiguous(), kap.contiguous(), kapx.contiguous(),
                                        v, b, gc, gd, gdx, r0, S, DH, H,
                                        DH * S * H, S * H, H, 1,
                                        BLOCK_DH=BDH, num_warps=4)
            gv = gdx.sum(dim=(0, 2)).reshape(-1) if ctx.needs_input_grad[2] else None
            gb = gd.sum(dim=(0, 2)).reshape(-1) if ctx.needs_input_grad[3] else None
            gc = gc.to(c.dtype) if ctx.needs_input_grad[0] else None
            gs = (rho[:, :, 0, :] * r0.view(B, DH, H)).unsqueeze(2).to(c.dtype) \
                if (ctx.has_init and ctx.needs_input_grad[1]) else None
            return gc, gs, gv, gb

    # ---------------------------------------------------------------------------
    # Kernel 2a: gen_model 前向融合
    # out = head_linear(cat[a,b,c,d,e]) * e + a*b + a1*b + a2*d
    #       + a*(a3*e + d) + b*(c+e) + c*e
    # tile 形状 [BLOCK_DH*BLOCK_S, BLOCK_H]（N 行位置 × padded 头维），
    # 头混合用 5 个 [N,K]@[K,O] 的 fp32 ieee dot 完成。
    # ---------------------------------------------------------------------------
    @triton.jit
    def _gen_fwd_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, w_ptr,
                        al1, al2, al3, o_ptr,
                        S, DH, H,
                        sa0, sa1, sa2, sa3,
                        sb0, sb1, sb2, sb3,
                        sc0, sc1, sc2, sc3,
                        sd0, sd1, sd2, sd3,
                        se0, se1, se2, se3,
                        so0, so1, so2, so3,
                        BLOCK_DH: tl.constexpr, BLOCK_S: tl.constexpr,
                        BLOCK_H: tl.constexpr):
        pid_b = tl.program_id(0)
        pid_d = tl.program_id(1)
        pid_s = tl.program_id(2)

        a1 = tl.load(al1).to(tl.float32)
        a2 = tl.load(al2).to(tl.float32)
        a3 = tl.load(al3).to(tl.float32)

        pp = pid_d * BLOCK_DH + tl.arange(0, BLOCK_DH * BLOCK_S) // BLOCK_S
        tt = pid_s * BLOCK_S + tl.arange(0, BLOCK_DH * BLOCK_S) % BLOCK_S
        offs_h = tl.arange(0, BLOCK_H)
        m2 = (pp[:, None] < DH) & (tt[:, None] < S) & (offs_h[None, :] < H)

        A = tl.load(a_ptr + pid_b * sa0 + pp[:, None] * sa1 + tt[:, None] * sa2 + offs_h[None, :] * sa3,
                    mask=m2, other=0.0).to(tl.float32)
        B = tl.load(b_ptr + pid_b * sb0 + pp[:, None] * sb1 + tt[:, None] * sb2 + offs_h[None, :] * sb3,
                    mask=m2, other=0.0).to(tl.float32)
        C = tl.load(c_ptr + pid_b * sc0 + pp[:, None] * sc1 + tt[:, None] * sc2 + offs_h[None, :] * sc3,
                    mask=m2, other=0.0).to(tl.float32)
        D = tl.load(d_ptr + pid_b * sd0 + pp[:, None] * sd1 + tt[:, None] * sd2 + offs_h[None, :] * sd3,
                    mask=m2, other=0.0).to(tl.float32)
        E = tl.load(e_ptr + pid_b * se0 + pp[:, None] * se1 + tt[:, None] * se2 + offs_h[None, :] * se3,
                    mask=m2, other=0.0).to(tl.float32)

        dot = tl.zeros((BLOCK_DH * BLOCK_S, BLOCK_H), tl.float32)
        for t in tl.static_range(5):
            if t == 0:
                X = A
            elif t == 1:
                X = B
            elif t == 2:
                X = C
            elif t == 3:
                X = D
            else:
                X = E
            # wt[o, k] = W[o, t*H + k]；头混合 dot[n, o] = Σ_k X[n, k] * W[o, tH+k] = X @ wt^T
            wt = tl.load(w_ptr + offs_h[:, None] * (5 * H) + (t * H + offs_h)[None, :],
                         mask=(offs_h[:, None] < H) & (offs_h[None, :] < H), other=0.0)
            dot += tl.dot(X, tl.trans(wt))

        out = dot * E + A * B + a1 * B + a2 * D + A * (a3 * E + D) + B * (C + E) + C * E
        tl.store(o_ptr + pid_b * so0 + pp[:, None] * so1 + tt[:, None] * so2 + offs_h[None, :] * so3,
                 out.to(o_ptr.dtype.element_ty), mask=m2)

    # ---------------------------------------------------------------------------
    # Kernel 2b: gen_model 反向融合（重算 dot，逐元素梯度 + gW/gAlpha 分块归约）
    # gW/gAlpha 不用原子累加（CTA 数大时同地址串行原子是瓶颈）：
    # 每个 CTA 把局部结果写入 workspace 自己的槽位，wrapper 一次性 sum 归约
    # ---------------------------------------------------------------------------
    @triton.jit
    def _gen_bwd_kernel(g_ptr, a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, w_ptr,
                        al1, al2, al3,
                        ga_ptr, gb_ptr, gc_ptr, gd_ptr, ge_ptr,
                        ws_gw_ptr, ws_al_ptr,
                        S, DH, H, NUM_PD, NUM_PS,
                        sg0, sg1, sg2, sg3,
                        sa0, sa1, sa2, sa3,
                        sb0, sb1, sb2, sb3,
                        sc0, sc1, sc2, sc3,
                        sd0, sd1, sd2, sd3,
                        se0, se1, se2, se3,
                        BLOCK_DH: tl.constexpr, BLOCK_S: tl.constexpr,
                        BLOCK_H: tl.constexpr):
        pid_b = tl.program_id(0)
        pid_d = tl.program_id(1)
        pid_s = tl.program_id(2)
        pid = (pid_b * NUM_PD + pid_d) * NUM_PS + pid_s

        a1 = tl.load(al1).to(tl.float32)
        a2 = tl.load(al2).to(tl.float32)
        a3 = tl.load(al3).to(tl.float32)

        pp = pid_d * BLOCK_DH + tl.arange(0, BLOCK_DH * BLOCK_S) // BLOCK_S
        tt = pid_s * BLOCK_S + tl.arange(0, BLOCK_DH * BLOCK_S) % BLOCK_S
        offs_h = tl.arange(0, BLOCK_H)
        m2 = (pp[:, None] < DH) & (tt[:, None] < S) & (offs_h[None, :] < H)
        mw = (offs_h[:, None] < H) & (offs_h[None, :] < H)

        G = tl.load(g_ptr + pid_b * sg0 + pp[:, None] * sg1 + tt[:, None] * sg2 + offs_h[None, :] * sg3,
                    mask=m2, other=0.0).to(tl.float32)
        A = tl.load(a_ptr + pid_b * sa0 + pp[:, None] * sa1 + tt[:, None] * sa2 + offs_h[None, :] * sa3,
                    mask=m2, other=0.0).to(tl.float32)
        B = tl.load(b_ptr + pid_b * sb0 + pp[:, None] * sb1 + tt[:, None] * sb2 + offs_h[None, :] * sb3,
                    mask=m2, other=0.0).to(tl.float32)
        C = tl.load(c_ptr + pid_b * sc0 + pp[:, None] * sc1 + tt[:, None] * sc2 + offs_h[None, :] * sc3,
                    mask=m2, other=0.0).to(tl.float32)
        D = tl.load(d_ptr + pid_b * sd0 + pp[:, None] * sd1 + tt[:, None] * sd2 + offs_h[None, :] * sd3,
                    mask=m2, other=0.0).to(tl.float32)
        E = tl.load(e_ptr + pid_b * se0 + pp[:, None] * se1 + tt[:, None] * se2 + offs_h[None, :] * se3,
                    mask=m2, other=0.0).to(tl.float32)

        headval = tl.zeros((BLOCK_DH * BLOCK_S, BLOCK_H), tl.float32)
        gcata = tl.zeros((BLOCK_DH * BLOCK_S, BLOCK_H), tl.float32)
        gcatb = tl.zeros((BLOCK_DH * BLOCK_S, BLOCK_H), tl.float32)
        gcatc = tl.zeros((BLOCK_DH * BLOCK_S, BLOCK_H), tl.float32)
        gcatd = tl.zeros((BLOCK_DH * BLOCK_S, BLOCK_H), tl.float32)
        gcate = tl.zeros((BLOCK_DH * BLOCK_S, BLOCK_H), tl.float32)
        # head 路径 = dot(cat, W^T) * e：对 cat/W 的梯度都带 E[n, o] 因子，必须在 dot 前缩放 G
        GE = G * E
        for t in tl.static_range(5):
            if t == 0:
                X = A
            elif t == 1:
                X = B
            elif t == 2:
                X = C
            elif t == 3:
                X = D
            else:
                X = E
            wt = tl.load(w_ptr + offs_h[:, None] * (5 * H) + (t * H + offs_h)[None, :],
                         mask=mw, other=0.0)
            headval += tl.dot(X, tl.trans(wt))
            gcat = tl.dot(GE, wt)  # gcat[n, k] = Σ_o G[n,o]·E[n,o]·W[o, tH+k]
            if t == 0:
                gcata = gcat
            elif t == 1:
                gcatb = gcat
            elif t == 2:
                gcatc = gcat
            elif t == 3:
                gcatd = gcat
            else:
                gcate = gcat
            gw_t = tl.dot(tl.trans(GE), X)
            # 写入本 CTA 的 workspace 槽位（padded 列布局 t*BH + k，归约后重映射）
            tl.store(ws_gw_ptr + pid * (BLOCK_H * 5 * BLOCK_H)
                     + offs_h[:, None] * (5 * BLOCK_H) + (t * BLOCK_H + offs_h)[None, :], gw_t)

        # 各流的多项式直接项互不对称，必须逐流写出（head 路径贡献 gcat_t 已含 E 因子）：
        ga = G * (B + a3 * E + D) + gcata
        gb = G * (A + a1 + C + E) + gcatb
        gc = G * (B + E) + gcatc
        gd = G * (A + a2) + gcatd
        # ge 额外项：G·headval（head_linear*e 乘子角色的梯度）+ a3*a + b + c（多项式项）+ gcate（cat 角色）
        ge = G * (headval + a3 * A + B + C) + gcate
        # ga..ge 为新分配的连续张量，按连续布局寻址（勿用输入流的 stride！）
        dst = ((pid_b * DH + pp[:, None]) * S + tt[:, None]) * H + offs_h[None, :]
        tl.store(ga_ptr + dst, ga.to(ga_ptr.dtype.element_ty), mask=m2)
        tl.store(gb_ptr + dst, gb.to(gb_ptr.dtype.element_ty), mask=m2)
        tl.store(gc_ptr + dst, gc.to(gc_ptr.dtype.element_ty), mask=m2)
        tl.store(gd_ptr + dst, gd.to(gd_ptr.dtype.element_ty), mask=m2)
        tl.store(ge_ptr + dst, ge.to(ge_ptr.dtype.element_ty), mask=m2)

        tl.store(ws_al_ptr + pid * 3 + 0, tl.sum(G * B))
        tl.store(ws_al_ptr + pid * 3 + 1, tl.sum(G * D))
        tl.store(ws_al_ptr + pid * 3 + 2, tl.sum(G * A * E))

# ---------------------------------------------------------------------------
# Autograd 封装
# ---------------------------------------------------------------------------
if _HAS_TRITON:
    class _GenModelFn(torch.autograd.Function):
        """gen_model 多项式混合的前向/反向融合。"""

        @staticmethod
        def forward(ctx, a, b, c, d, e, weight, al1, al2, al3):
            B, DH, S, H = a.shape
            out = torch.empty(a.shape, dtype=a.dtype, device=a.device)
            BD, BS, BH = 16, 16, max(16, triton.next_power_of_2(H))
            grid = (B, triton.cdiv(DH, BD), triton.cdiv(S, BS))
            _gen_fwd_kernel[grid](a, b, c, d, e, weight, al1, al2, al3, out,
                                  S, DH, H,
                                  *a.stride(), *b.stride(), *c.stride(),
                                  *d.stride(), *e.stride(), *out.stride(),
                                  BLOCK_DH=BD, BLOCK_S=BS, BLOCK_H=BH,
                                  num_warps=4)
            ctx.save_for_backward(a, b, c, d, e, weight, al1, al2, al3)
            ctx.dims = (B, DH, S, H)
            return out

        @staticmethod
        def backward(ctx, go):
            a, b, c, d, e, weight, al1, al2, al3 = ctx.saved_tensors
            B, DH, S, H = ctx.dims
            # bwd 活跃 tile 更多（G + 5 流 + dot 暂存），用更小 tile 与单级流水避免共享内存超限
            BD, BS, BH = 16, 8, max(16, triton.next_power_of_2(H))
            GD, GS = triton.cdiv(DH, BD), triton.cdiv(S, BS)
            ga, gb, gc, gd, ge = (torch.empty(x.shape, dtype=x.dtype, device=x.device)
                                  for x in (a, b, c, d, e))
            # gW / gAlpha 的 per-CTA workspace：纯 store 零竞争，最后 sum 归约
            P = B * GD * GS
            ws_gw = torch.empty(P, BH, 5 * BH, dtype=torch.float32, device=a.device)
            ws_al = torch.empty(P, 3, dtype=torch.float32, device=a.device)
            grid = (B, GD, GS)
            _gen_bwd_kernel[grid](go, a, b, c, d, e, weight, al1, al2, al3,
                                  ga, gb, gc, gd, ge, ws_gw, ws_al,
                                  S, DH, H, GD, GS,
                                  *go.stride(), *a.stride(), *b.stride(), *c.stride(),
                                  *d.stride(), *e.stride(),
                                  BLOCK_DH=BD, BLOCK_S=BS, BLOCK_H=BH,
                                  num_warps=4, num_stages=1)
            # [P, BH, 5BH] → 列重映射 (t*BH + k) → (t*H + k)
            gw = ws_gw.sum(0).view(BH, 5, BH)[:H, :, :H].reshape(H, 5 * H)
            al_grads = ws_al.sum(0)
            return ga, gb, gc, gd, ge, gw, al_grads[0], al_grads[1], al_grads[2]


# ---------------------------------------------------------------------------
# 公共接口（自动回退 eager）
# ---------------------------------------------------------------------------
def cummax_scan(c, state=None, v=None, b=None):
    """衰减最大值扫描：e_j = max(γ_j·e_{j-1}, x_j)；v=None 时 γ≡1（纯 cummax）。

    返回 (e)。γ_j = exp(-softplus(c_j·v_d + b_d))，v/b 为 dim 维参数。"""
    if TRITON_OK and c.is_cuda:
        return _CumMaxFn.apply(c, state, v, b)
    return decay_ref(c, state, v, b)


def gen_model_fused(a, b, c, d, e, weight, al1, al2, al3):
    """gen_model 多项式混合的融合实现；入参/出参布局不限（kernel 按 stride 读取）。"""
    if TRITON_OK and a.is_cuda:
        return _GenModelFn.apply(a, b, c, d, e, weight, al1, al2, al3)
    return gen_model_ref(a, b, c, d, e, weight, al1, al2, al3)


# ---------------------------------------------------------------------------
# eager 参考实现（CPU / 无 Triton 回退，也是正确性基准；可微，autograd 即参考梯度）
# ---------------------------------------------------------------------------
def cummax_ref(c, state=None):
    """纯 cummax（γ≡1 特例）。"""
    e = torch.cummax(c, dim=2).values
    if state is not None:
        e = torch.maximum(e, state)  # 广播 [.., 1, H]，等价于 cat 版切片
    return e


def decay_serial_ref(c, state=None, v=None, b=None):
    """衰减最大值扫描的精确串行 oracle（fp32，可微，无除法/无下溢，用于正确性测试）。

    注意: 分块 cumprod 版本在 γ→0 时会因下溢+除法产生 NaN/Inf，勿用于极端 γ。"""
    B, DH, S, H = c.shape
    cf = c.float()
    if v is not None:
        vv = v.float().view(1, DH, H)
        bb = b.float().view(1, DH, H)
    run = torch.full((B, DH, H), -1e30, device=c.device, dtype=torch.float32) \
        if state is None else state[:, :, 0].float()
    e = torch.empty((B, DH, S, H), device=c.device, dtype=torch.float32)
    for j in range(S):
        x = cf[:, :, j]
        if v is not None:
            g = torch.exp(-torch.nn.functional.softplus(x * vv + bb))
        else:
            g = torch.ones_like(x)
        run = torch.maximum(g * run, x)
        e[:, :, j] = run
    return e


def decay_ref(c, state=None, v=None, b=None, chunk=128):
    B, DH, S, H = c.shape
    cf = c.float()
    if v is not None:
        vv = v.float().view(1, DH, 1, H)
        bb = b.float().view(1, DH, 1, H)
        g = torch.exp(-torch.nn.functional.softplus(cf * vv + bb))
    else:
        g = torch.ones_like(cf)
    run = torch.full((B, DH, 1, H), float('-inf'), device=c.device, dtype=torch.float32) \
        if state is None else state.float()
    e = torch.empty((B, DH, S, H), device=c.device, dtype=torch.float32)
    C = (S + chunk - 1) // chunk
    for ck in range(C):
        lo, hi = ck * chunk, min((ck + 1) * chunk, S)
        gc_ = g[:, :, lo:hi]
        pref = torch.cumprod(gc_, dim=2).clamp_min(1e-30)
        m = torch.cummax(torch.cat([run, cf[:, :, lo:hi] / pref], dim=2), dim=2).values[:, :, 1:]
        ec = pref * torch.maximum(run, m)
        e[:, :, lo:hi] = ec
        run = ec[:, :, -1:]
    return e


def gen_model_ref(a, b, c, d, e, weight, al1, al2, al3):
    # 内部升 fp32 计算（与 kernel 的 fp32 精度对齐），结果回投到输入 dtype
    out_dtype = a.dtype
    a, b, c, d, e = (t.float() for t in (a, b, c, d, e))
    al1, al2, al3 = al1.float(), al2.float(), al3.float()
    cat = torch.cat([a, b, c, d, e], dim=-1)
    head = torch.nn.functional.linear(cat, weight.float()) * e
    return (a * b + al1 * b + al2 * d + a * (al3 * e + d) + b * (c + e) + c * e + head).to(out_dtype)


def kv_chunked_scan(q, k, v, log_a, chunk=128, s0=None):
    """衰减 KV 记忆通道（线性注意）：S_j = a_j·S_{j-1} + k_j⊗v_j，读出 r_j = q_j·S_j。

    q,k: (B,H,S,DH)；v: (B,H,S,DV)；log_a: (B,H,S) 每步对数衰减（≤0，fp32）。
    s0: (B,H,DH,DV) 初始状态（增量推理用；None = 零状态）。
    返回 (r, carry_new)：carry_new 即段尾状态，推理时作为下一步 s0 → O(1) 状态。
    分块实现（块内因果矩阵形式，块间串行 carry），纯 torch、可微、无除法；
    dtype 直通（训练 bf16 / 推理 fp32 均可），仅 log_a 累加与因果权重保持 fp32。
    """
    B, H, S, DH = q.shape
    DV = v.shape[-1]
    dt = q.dtype
    r = torch.empty(B, H, S, DV, device=q.device, dtype=dt)
    carry = s0.to(dt) if s0 is not None \
        else torch.zeros(B, H, DH, DV, device=q.device, dtype=dt)
    for ck in range((S + chunk - 1) // chunk):
        lo, hi = ck * chunk, min((ck + 1) * chunk, S)
        qc, kc, vc = q[:, :, lo:hi], k[:, :, lo:hi], v[:, :, lo:hi]
        l = log_a[:, :, lo:hi].float().cumsum(-1)            # 块内 log 前缀 (B,H,C) fp32
        Cm = (l[:, :, :, None] - l[:, :, None, :]).exp().tril().to(dt)   # 因果 ≤1
        intra = (Cm * (qc @ kc.transpose(-1, -2))) @ vc      # 块内检索 (B,H,C,DV)
        carry_r = (qc * l.exp()[..., None].to(dt)) @ carry   # 跨块 carry 读出
        r[:, :, lo:hi] = intra + carry_r
        # 块尾状态：exp(l_last)·carry + Σ_i exp(l_last - l_i)·k_i⊗v_i
        kw = kc * (l[:, :, -1:] - l).exp()[..., None].to(dt)
        carry = l[:, :, -1:].exp().unsqueeze(-1).to(dt) * carry + kw.transpose(-1, -2) @ vc
    return r, carry


def kv_scan_naive(q, k, v, log_a):
    """kv_chunked_scan 的逐步串行参考实现（正确性对照用）。"""
    B, H, S, DH = q.shape
    DV = v.shape[-1]
    Sc = torch.zeros(B, H, DH, DV, device=q.device, dtype=torch.float32)
    r = torch.empty(B, H, S, DV, device=q.device, dtype=torch.float32)
    for j in range(S):
        Sc = log_a[:, :, j].exp()[..., None, None] * Sc \
            + torch.einsum('bhd,bhv->bhdv', k[:, :, j].float(), v[:, :, j].float())
        r[:, :, j] = torch.einsum('bhd,bhdv->bhv', q[:, :, j].float(), Sc)
    return r


def kv_delta_scan_naive(q, k, v, log_a):
    """Delta-rule 写入的逐步串行实现：S_j = a_j·S_{j-1}(I − k̂_j k̂_jᵀ) + k̂_j v_jᵀ。

    k 应已 L2 归一化（调用方负责）；重复键精确覆盖旧值（消除求和干扰），
    是精确键值检索（归纳头）的理论最优写入规则。读出 r_j = q_j·S_j（含 j 自身写入）。
    逐步串行、纯 torch、可微——正确性 A/B 用；稳定后可做分块/UT 变换加速。
    """
    B, H, S, DH = q.shape
    DV = v.shape[-1]
    Sc = torch.zeros(B, H, DH, DV, device=q.device, dtype=torch.float32)
    r = torch.empty(B, H, S, DV, device=q.device, dtype=torch.float32)
    for j in range(S):
        a = log_a[:, :, j].exp()[..., None, None]              # (B,H,1,1)
        kj = k[:, :, j].float()                                # (B,H,DH) 已归一化
        vj = v[:, :, j].float()                                # (B,H,DV)
        u = torch.einsum('bhd,bhdv->bhv', kj, Sc)              # (B,H,DV) = kᵀ·S
        Sc = a * Sc \
            - a * kj[..., None] * u[..., None, :] \
            + kj[..., None] * vj[..., None, :]
        r[:, :, j] = torch.einsum('bhd,bhdv->bhv', q[:, :, j].float(), Sc)
    return r


def _delta_run(q_seg, k_seg, v_seg, la_seg, S0):
    """单段 delta 递推：从边界状态 S0 起，返回 (段内读出 r, 段尾状态)。"""
    B, H, C, DH = q_seg.shape
    DV = v_seg.shape[-1]
    Sc = S0
    r = torch.empty(B, H, C, DV, device=q_seg.device, dtype=torch.float32)
    for j in range(C):
        a = la_seg[:, :, j].exp()[..., None, None]
        kj = k_seg[:, :, j]
        vj = v_seg[:, :, j]
        u = torch.einsum('bhd,bhdv->bhv', kj, Sc)
        Sc = a * Sc - a * kj[..., None] * u[..., None, :] + kj[..., None] * vj[..., None, :]
        r[:, :, j] = torch.einsum('bhd,bhdv->bhv', q_seg[:, :, j], Sc)
    return r, Sc


def kv_delta_scan_tbptt(q, k, v, log_a, seg=64, s0=None):
    """delta-rule 分段 checkpoint 版：段边界状态 no-grad 前向 + 段内重算（TBPTT 语义，
    段间梯度截断），显存 O(段) 而非 O(全序列)。训练用；推理/正确性对照用 kv_delta_scan_naive。
    返回 (r, None)——delta 训练路径不携带跨调用状态。"""
    from torch.utils.checkpoint import checkpoint as cp
    B, H, S, DH = q.shape
    DV = v.shape[-1]
    r = torch.empty(B, H, S, DV, device=q.device, dtype=torch.float32)
    Sc = s0.float() if s0 is not None \
        else torch.zeros(B, H, DH, DV, device=q.device, dtype=torch.float32)
    for lo in range(0, S, seg):
        hi = min(lo + seg, S)
        qs, ks, vs, las = q[:, :, lo:hi], k[:, :, lo:hi], v[:, :, lo:hi], log_a[:, :, lo:hi]
        with torch.no_grad():
            _, Sc_end = _delta_run(qs.float(), ks.float(), vs.float(), las.float(), Sc)
        Sc_end = Sc_end.detach()
        rs, _ = cp(_delta_run, qs.float(), ks.float(), vs.float(), las.float(), Sc_end,
                   use_reentrant=False)
        r[:, :, lo:hi] = rs
        Sc = Sc_end
    return r, None


# ---------------------------------------------------------------------------
# v3.5 KV 记忆通道的 Triton 前向 + 重算反向：
#   前向 kernel：grid=(B*H,)，每 program 循环全部 chunk（块间串行 carry），
#   块内 tl.dot 完成检索矩阵乘；r 逐 chunk store、carry 存 per-chunk buffer
#   （循环内 store，遵循 v2 max-scan 反向 kernel 的稳定模式，规避本环境
#   "循环出口累加器 store" 的 codegen 非确定性）。
#   反向：torch.autograd.grad 对可微 torch 版（kv_chunked_scan）重算求梯度
#   （重算一次 ~3.5ms/层，远低于 autograd 直接反传的 ~10ms/层）。
# ---------------------------------------------------------------------------
@triton.jit
def _kv_scan_kernel(q_ptr, k_ptr, v_ptr, la_ptr, s0_ptr,
                    r_ptr, cb_ptr,
                    S, NC, DH, DV, H,
                    sq0, sq1, sq2, sq3,
                    sk0, sk1, sk2, sk3,
                    sv0, sv1, sv2, sv3,
                    sla0, sla1, sla2,
                    ss0, ss1, ss2, ss3,
                    sr0, sr1, sr2, sr3,
                    scb0, scb1, scb2, scb3, scb4,
                    HAS_S0: tl.constexpr,
                    CH: tl.constexpr,
                    BDH: tl.constexpr,
                    BDV: tl.constexpr):
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H
    offs_c = tl.arange(0, CH)
    offs_d = tl.arange(0, BDH)
    offs_v = tl.arange(0, BDV)
    dmask = offs_d < DH
    vmask = offs_v < DV
    tril = (offs_c[:, None] >= offs_c[None, :]).to(tl.float32)
    if HAS_S0:
        carry = tl.load(s0_ptr + b * ss0 + h * ss1 + offs_d[:, None] * ss2 + offs_v[None, :] * ss3,
                        mask=dmask[:, None] & vmask[None, :], other=0.0)
    else:
        carry = tl.zeros([BDH, BDV], tl.float32)
    for ck in range(0, NC):
        base = ck * CH
        pmask = (base + offs_c) < S
        m2 = pmask[:, None] & dmask[None, :]
        la_blk = tl.load(la_ptr + b * sla0 + h * sla1 + (base + offs_c) * sla2,
                         mask=pmask, other=0.0)
        l = tl.cumsum(la_blk, axis=0)                        # 块内 log 前缀 (CH,)
        qc = tl.load(q_ptr + b * sq0 + h * sq1 + (base + offs_c[:, None]) * sq2 + offs_d[None, :] * sq3,
                     mask=m2, other=0.0).to(q_ptr.dtype.element_ty)
        kc = tl.load(k_ptr + b * sk0 + h * sk1 + (base + offs_c[:, None]) * sk2 + offs_d[None, :] * sk3,
                     mask=m2, other=0.0).to(q_ptr.dtype.element_ty)
        vc = tl.load(v_ptr + b * sv0 + h * sv1 + (base + offs_c[:, None]) * sv2 + offs_v[None, :] * sv3,
                     mask=pmask[:, None] & vmask[None, :], other=0.0).to(q_ptr.dtype.element_ty)
        qk = tl.dot(qc, tl.trans(kc))                        # (CH,CH) fp32 累加
        dj = l[:, None] - l[None, :]
        Cm = tl.exp(tl.where(tril > 0, dj, float('-inf')))   # 因果，防非因果上溢
        T = (Cm * qk).to(vc.dtype)
        intra = tl.dot(T, vc)                                # (CH,BDV) fp32 累加
        carry_r = tl.dot((qc * l.exp()[:, None]).to(carry.dtype), carry.to(carry.dtype))
        r_blk = intra + carry_r.to(tl.float32)
        tl.store(r_ptr + b * sr0 + h * sr1 + (base + offs_c[:, None]) * sr2 + offs_v[None, :] * sr3,
                 r_blk, mask=pmask[:, None] & vmask[None, :])
        # 块尾状态更新 + per-chunk buffer store（稳定模式）
        w_tail = tl.exp(tl.min(l, axis=0) - l)               # (CH,) 各位置→块尾衰减
        kw = kc * w_tail[:, None]
        newc = tl.dot(tl.trans(kw).to(vc.dtype), vc)         # (BDH,BDV) 块内写入
        carry = tl.min(l, axis=0).exp() * carry.to(tl.float32) + newc.to(tl.float32)
        tl.store(cb_ptr + b * scb0 + h * scb1 + ck * scb2 + offs_d[:, None] * scb3 + offs_v[None, :] * scb4,
                 carry, mask=dmask[:, None] & vmask[None, :])


class KvScanFn(torch.autograd.Function):
    """KV 通道 Triton 前向 + torch 重算反向（显存友好、绕开手工梯度推导）。"""

    @staticmethod
    def forward(ctx, q, k, v, log_a, s0):
        B, H, S, DH = q.shape
        DV = v.shape[-1]
        CH, NC = 64, (S + 63) // 64
        BDH, BDV = triton.next_power_of_2(DH), triton.next_power_of_2(DV)
        r = torch.empty(B, H, S, DV, device=q.device, dtype=q.dtype)
        cb = torch.empty(B, H, NC, BDH, BDV, dtype=torch.float32, device=q.device)
        la = log_a.float().contiguous()
        dt = q.dtype
        q_c, k_c, v_c = q.contiguous(), k.contiguous().to(dt), v.contiguous().to(dt)
        has_s0 = s0 is not None
        s0_c = s0.contiguous() if has_s0 else k_c
        _kv_scan_kernel[(B * H,)](q_c, k_c, v_c, la, s0_c,
                                  r, cb,
                                  S, NC, DH, DV, H,
                                  *q_c.stride(), *k_c.stride(), *v_c.stride(),
                                  la.stride(0), la.stride(1), la.stride(2),
                                  *s0_c.stride(),
                                  *r.stride(),
                                  cb.stride(0), cb.stride(1), cb.stride(2),
                                  cb.stride(3), cb.stride(4),
                                  HAS_S0=has_s0, CH=CH, BDH=BDH, BDV=BDV,
                                  num_warps=4, num_stages=1)
        carry = cb[:, :, -1, :DH, :DV].to(dt).contiguous()
        ctx.save_for_backward(q, k, v, log_a, s0)
        return r, carry

    @staticmethod
    def backward(ctx, gr, _gcarry):
        q, k, v, log_a, s0 = ctx.saved_tensors
        with torch.enable_grad():
            q2 = q.detach().requires_grad_(True)
            k2 = k.detach().to(q.dtype).requires_grad_(True)
            v2 = v.detach().to(q.dtype).requires_grad_(True)
            r, _ = kv_chunked_scan(q2, k2, v2, log_a, chunk=128, s0=s0)
            gq, gk, gv = torch.autograd.grad(r, [q2, k2, v2], gr.detach().to(r.dtype))
        return gq.to(q.dtype), gk.to(k.dtype), gv.to(v.dtype), None, None




