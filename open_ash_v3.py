import time
import math

import torch
import torch.nn.functional as F
from torch import nn, optim

try:
    import triton_ops as tops
except ImportError:
    tops = None

try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    FLA_GDN_OK = True
except ImportError:
    chunk_gated_delta_rule = None
    FLA_GDN_OK = False


def _fused_available(x):
    return tops is not None and tops.TRITON_OK and x.is_cuda


class MaxStateSuper(torch.nn.Module):
    """v3 核心：cummax 的输入流 c 来自本层专属词嵌入，不再由 combined 投影产生。

    与 v1/v2 的差异：
    - combined 从 4 路投影减为 3 路 (a=out, b=out1, d=out3)，c 路由 state_em(ids) 查表给出
    - 每层独立 state_em → 每层学到自己的"状态码本"，层间不共享 cummax 输入语义
    - em_scale 可学习缩放：状态幅度可控
    - use_decay=True（v3.1）：归约算子为衰减最大值 e_j = max(γ_j·e_{j-1}, x_j)，
      γ_j = exp(-softplus(c·v+b)) 逐维可学习（输入相关）→ 状态可下降，
      根治纯 cummax 单调不减的长程饱和风险；USE_DECAY=False 时 γ≡1（纯 cummax）
    - use_fused=True 时走 Triton 融合 kernel（训练加速/省显存），否则 eager 回退
    """

    def __init__(self, dim_size, heads, voc_size, model_flag="train", use_fused=True,
                 state_rank=None, use_decay=True,
                 decay_init=(0.001, 0.15), decay_v_std=0.0,
                 kv_channel=False, kv_decay=0.98, kv_norm_keys=False, kv_beta=False,
                 kv_delta=False, kv_dim=None, kv_backend='sum', kv_beta_init=0.0, kv_beta_tok=False,
                 em_ref=None, wkey_em=None, state_split=False, kv_out_gdn=False):
        super(MaxStateSuper, self).__init__()
        self.heads = heads
        self.d_head = int(dim_size // heads)
        self.kv_dh = kv_dim if kv_dim else self.d_head   # KV 通道头维（瘦身消融用）
        self.kv_backend = kv_backend                     # 'sum'（v3.5s）| 'delta_fla'（v3.6）
        self.model_flag = model_flag
        self.use_fused = use_fused
        self.use_decay = use_decay
        # v4.2: state 三分化——MEM 头组零衰减永久记忆 + 多位读取（aux 读出混入）
        self.state_split = state_split
        if state_split:
            self.o_read = nn.Parameter(torch.randn(heads, self.kv_dh, self.kv_dh) * 0.02)
            self.o_mix = nn.Parameter(torch.full((heads,), 4.0))   # sigmoid≈0.98 主读出
        # v4.3: GDN 式输出——RMS 归一化 + per-channel 门（通道恒开）
        self.kv_out_gdn = kv_out_gdn
        if kv_out_gdn:
            self.kv_out_gate = nn.Parameter(torch.ones(dim_size))
        assert dim_size % heads == 0, "Dimension size must be divisible by head size."
        # v4.0: combined 投影由 em 变换得到（W = A @ em 切片，与输入嵌入共享梯度）
        self.em_ref = em_ref

        # v3.4: 衰减 KV 记忆通道（query 相关检索），与 max-scan 互补:
        #   S_j = a_j·S_{j-1} + c_j⊗c_j, 读出 r_j = q_j·S_j, k=v=c (状态嵌入自身)
        #   → 检索到相似历史 token 即取回其嵌入 (归纳头行为); gate 零初始化渐进启用
        self.kv_channel = kv_channel
        if kv_channel:
            self.q_proj = nn.Linear(dim_size, self.heads * self.kv_dh, bias=False)
            theta0 = math.log(math.expm1(-math.log(kv_decay)))
            self.kv_loga = nn.Parameter(torch.full((heads,), theta0))
            self.kv_gate = nn.Parameter(torch.tensor(0.0))
            self.kv_norm_keys = kv_norm_keys   # k̂ L2 归一化（检索锐化，DeltaNet 惯例）
            self.kv_delta = kv_delta           # delta-rule 写入（重复键精确覆盖）
            self.kv_beta = nn.Parameter(torch.ones(heads)) if kv_beta else None
            # v3.6/v3.9: delta 写入（写入强度 β 可学习；FLA kernel 与自研分块共用）
            if self.kv_backend in ('delta_fla', 'delta_chunk'):
                self.kv_beta_fla = nn.Parameter(torch.full((heads,), kv_beta_init))
                # v3.8: GDN 式 per-token β = sigmoid(w_h·k̂_t + b_h)
                # W 零初始化 → 起点等价常数 β=sigmoid(2.0)≈0.88，训练中学习 token 依赖的写入强度
                self.kv_beta_tok = kv_beta_tok
                if kv_beta_tok:
                    self.kv_beta_w = nn.Parameter(torch.zeros(heads, self.kv_dh))
                    self.kv_beta_b = nn.Parameter(torch.full((heads,), 2.0))

        # v1/v2 为 4 路投影；v3 中 c 路改由 state_em 查表提供，故只需 3 路
        if em_ref is not None:
            self.combined = EmLinear(3 * dim_size, em_ref, 0, dim_size)
        else:
            self.combined = nn.Linear(dim_size, 3 * dim_size, bias=False)

        # 衰减参数（v3.1）：γ = exp(-softplus(c·v + b))，逐维、输入相关
        # 多尺度初始化：初始 γ 逐维 log 均匀散布；v3.2 扩展范围引入慢通道
        # （γ≈0.9999 → 半衰期 ~6931 tok，远程逐字记忆载体）
        if use_decay:
            s_lo, s_hi = decay_init
            bias = torch.log(torch.expm1(torch.linspace(s_lo, s_hi, dim_size)))
            self.decay_v = nn.Parameter(torch.randn(dim_size) * decay_v_std)
            self.decay_b = nn.Parameter(bias)
        else:
            self.decay_v = None
            self.decay_b = None

        # 每层专属嵌入：查表输出直接作为 torch.cummax(c, dim=2) 的输入
        # state_rank: 低秩分解 E(voc, r) @ P(r, dim)。稠密嵌入梯度 [voc, dim] 在 12 层
        # 85M 配置下约占 846MB，低秩后降至 ~72MB。输出方差保持 0.02：
        # std(E)=0.02, std(P)=r^-0.5 → std(c)=sqrt(r·0.02²·1/r)=0.02
        self.state_proj = None
        if state_rank:
            self.state_em = nn.Embedding(voc_size, state_rank)
            self.state_proj = nn.Linear(state_rank, dim_size, bias=False)
            nn.init.normal_(self.state_em.weight, mean=0.0, std=0.02)
            nn.init.normal_(self.state_proj.weight, mean=0.0, std=state_rank ** -0.5)
        else:
            # 用 0.02 方差初始化：c 是单调 max 累积器，初值过大会提前饱和
            self.state_em = nn.Embedding(voc_size, dim_size)
            nn.init.normal_(self.state_em.weight, mean=0.0, std=0.02)

        # 状态幅度可学习缩放
        self.em_scale = nn.Parameter(torch.tensor(1.0))

        self.alpha1 = torch.nn.Parameter(torch.tensor(0.5))
        self.alpha2 = torch.nn.Parameter(torch.tensor(0.5))
        self.alpha3 = torch.nn.Parameter(torch.tensor(0.5))

        self.head_linear = torch.nn.Linear(heads * 5, heads, bias=False)

        # v4.1: token-id 查表驱动的 rank-1 权重调制通路（state_em 模式搬到值通路）
        #   combined_out_t += iw_gate · a_t · <w2, x_t>，a_t = Pa(wkey_em[ids_t])
        #   门零初始化 → 起点严格等价 v37；通路是否有用由训练决定
        self.wkey_em = wkey_em
        if wkey_em is not None:
            self.iw_Pa = nn.Linear(wkey_em.weight.shape[1], dim_size, bias=False)
            self.iw_w2 = nn.Parameter(torch.randn(dim_size) * 0.02)  # w2≠0：通路有信号；安全性由 gate=0 保证
            self.iw_gate = nn.Parameter(torch.tensor(0.0))

    def _kv_out(self, o_p):
        """KV 通道输出进入残差：默认 scalar kv_gate（零初始化渐进启用）；
        v4.3 GDN 式：RMS 归一化 + per-channel 门（通道恒开）。"""
        if self.kv_out_gdn:
            rms = o_p.float().pow(2).mean(dim=(2, 3), keepdim=True).add_(1e-6).rsqrt().to(o_p.dtype)
            return self.kv_out_gate.view(1, -1, 1, 1).to(o_p.dtype) * o_p * rms
        return self.kv_gate * o_p

    def forward(self, x, ids, state=None):
        b, s, d = x.shape
        # 布局 [b, s, 3, d_head, heads] → 各流 permute 为 [b, d_head, s, heads]（heads 连续，利于合并访问）
        combined = self.combined(x).view(b, s, 3, self.d_head, self.heads)
        if self.wkey_em is not None:
            # v4.1: token-id 查表 rank-1 权重调制（门零初始化，渐进启用）
            a_t = self.iw_Pa(self.wkey_em(ids))                       # (B,S,D)
            mod = (self.iw_gate * a_t * (x @ self.iw_w2).unsqueeze(-1))  # (B,S,D)
            combined = combined + mod.view(b, s, 1, self.d_head, self.heads)
        out, out1, out3 = combined.unbind(2)

        out = out.permute(0, 2, 1, 3)
        out1 = out1.permute(0, 2, 1, 3)
        out3 = out3.permute(0, 2, 1, 3)

        # v3 核心：c = 本层嵌入查表结果，直接作为 cummax 的输入流
        c_raw = self.state_em(ids)
        if self.state_proj is not None:
            c_raw = self.state_proj(c_raw)
        c = c_raw.view(b, s, self.d_head, self.heads).permute(0, 2, 1, 3)
        c = c * self.em_scale

        # e = 衰减最大值扫描（v3.1）
        #   训练: tops.DecayScanFn — 前向 torch 分块扫描，反向纯 torch 仿射前缀扫描
        #         （无 autograd 图；Triton 不可用时回退 decay_ref = autograd 精确反向）
        #   推理: tops._CumMaxFn — Triton 前向，携带状态 O(1) 解码
        dv = self.decay_v if self.use_decay else None
        db = self.decay_b if self.use_decay else None
        # state 载体（v3.5）：tuple (e_state, kv_S) —— max-scan 的 O(1) 向量状态
        # + KV 通道的 O(1) 矩阵状态；兼容旧的纯 e_state 张量传入
        if state is None:
            e_st, s_kv = None, None
        elif isinstance(state, tuple):
            e_st, s_kv = state
        else:
            e_st, s_kv = state, None
        qq = None
        if self.kv_channel:
            bx, sx = x.shape[0], x.shape[1]
            qq = self.q_proj(x).view(bx, sx, self.heads, self.kv_dh).permute(0, 2, 1, 3)
            log_a = (-torch.nn.functional.softplus(self.kv_loga)).view(1, -1, 1).expand(bx, self.heads, sx)
        infer_mode = (not torch.is_grad_enabled()) and _fused_available(c) and self.use_fused
        if infer_mode:
            e = tops._CumMaxFn.apply(c, e_st, dv, db)
            out_l = tops.gen_model_fused(out, out1, c, out3, e,
                                         self.head_linear.weight,
                                         self.alpha1, self.alpha2, self.alpha3)
        else:
            if self.use_decay and getattr(tops, 'TRITON_OK', False):
                e = tops.DecayScanFn.apply(c, e_st, dv, db)
            else:
                e = tops.decay_ref(c, e_st, dv, db)
            if self.use_fused and getattr(tops, 'TRITON_OK', False):
                # v3.0 融合训练路径：gen_model Triton 前向 + eager autograd 反向
                out_l = tops.gen_model_fused(out, out1, c, out3, e,
                                             self.head_linear.weight,
                                             self.alpha1, self.alpha2, self.alpha3)
            else:
                out_l = self.gen_model(out, out1, c, out3, e)
        s_kv_new = None
        if qq is not None:
            cc_perm = c.permute(0, 3, 2, 1)
            if self.kv_dh < self.d_head:
                cc_perm = cc_perm[..., :self.kv_dh]     # 瘦身：KV 通道用 c 流前 kv_dh 个 lane
            kk = cc_perm
            if self.kv_norm_keys:
                kk = kk / (kk.norm(dim=-1, keepdim=True) + 1e-6)
            if self.kv_beta is not None:
                kk = kk * self.kv_beta.view(1, -1, 1, 1).to(kk.dtype)
            if self.kv_backend in ('delta_fla', 'delta_chunk') and FLA_GDN_OK \
                    and not (s_kv is not None and sx == 1):
                r = None
                # ---- v3.6/v3.9：delta-rule 写入（FLA kernel 或自研分块 torch 版）----
                # 布局：q/k/v (B,S,H,D)、g (B,S,H) log 域、beta (B,S,H)
                qq_f = qq.permute(0, 2, 1, 3).float()        # (B,S,H,kv_dh)
                kk_f = kk.permute(0, 2, 1, 3).float()        # (B,S,H,kv_dh) 归一化键
                vv_f = kk_f.float()                          # delta 写入 v = k
                g_f = log_a.permute(0, 2, 1).float()         # (B,S,H) log 域衰减
                if getattr(self, 'kv_beta_tok', False):
                    # v3.8: per-token β（fp32 einsum，kk_f 已归一化）
                    beta_f = torch.sigmoid(
                        torch.einsum('bshd,hd->bsh', kk_f, self.kv_beta_w.float())
                        + self.kv_beta_b.view(1, 1, -1).float())
                else:
                    beta_f = torch.sigmoid(self.kv_beta_fla).view(1, 1, -1)\
                        .expand(bx, sx, self.heads).to(qq_f.dtype)
                if self.state_split:
                    # v4.2: MEM 头组（0:2）零衰减 + β=1 → 永久键值存储（永不遗忘）
                    g_f = torch.cat([torch.zeros_like(g_f[..., 0:2]), g_f[..., 2:]], dim=-1)
                    beta_f = torch.cat([torch.ones_like(beta_f[..., 0:2]), beta_f[..., 2:]], dim=-1)
                if self.kv_backend == 'delta_chunk':
                    # v3.9: 自研分块实现（纯 torch，autograd 反向；test_delta_chunk.py 全 PASS）
                    from delta_chunk import chunked_delta_rule
                    o, s_kv_new = chunked_delta_rule(
                        qq_f, kk_f, vv_f, g_f, beta_f,
                        initial_state=s_kv, output_final_state=True)
                else:
                    o, s_kv_new = chunk_gated_delta_rule(
                        q=qq_f, k=kk_f, v=vv_f, g=g_f, beta=beta_f,
                        initial_state=s_kv, output_final_state=True,
                        use_qk_l2norm_in_kernel=False)
                # o (B,S,H,kv_dh) → gen_model 布局 (B,d_head,S,H)
                if self.state_split:
                    # v4.2: 多位读取——aux 读出（同记忆的不同读取视角），per-head 混合
                    o_aux = torch.einsum('bshd,hde->bshe', o, self.o_read.to(o.dtype))
                    mix = torch.sigmoid(self.o_mix).view(1, 1, -1, 1).to(o.dtype)
                    o = mix * o + (1 - mix) * o_aux
                o_p = o.permute(0, 3, 1, 2)                     # (B,kv_dh,S,H)
                if self.kv_dh < self.d_head:
                    o_p = F.pad(o_p, (0, 0, 0, 0, 0, self.d_head - self.kv_dh))
                out_l = out_l + self._kv_out(o_p)
            elif s_kv is not None and sx == 1 and self.kv_backend in ('delta_fla', 'delta_chunk'):
                # 增量单步（O(1) 解码，delta 写入，纯 torch fp32 精确）：
                #   S ← a·S + β·k̂⊗(k̂ − β·(k̂S))；r = q·S·K^−0.5
                #   （与训练时 delta 覆盖语义一致；v3.9 修复：旧版误用 sum 写入）
                a_step = torch.exp(log_a[..., 0])                       # (B,H)
                # kk (B,H,S,D) → kd (B,H,D)；s_kv (B,H,K,V)（FLA 布局）
                kd = kk.to(qq.dtype).squeeze(2)
                if getattr(self, 'kv_beta_tok', False):
                    bd = torch.sigmoid(torch.einsum('bhd,hd->bh', kd.float(),
                                                    self.kv_beta_w.float())
                                       + self.kv_beta_b.view(1, -1).float()).to(qq.dtype)
                else:
                    bd = torch.sigmoid(self.kv_beta_fla).view(1, -1).to(qq.dtype)  # (B,H)
                sv = torch.einsum('bhd,bhdv->bhv', kd, s_kv.to(qq.dtype))
                if self.state_split:
                    # v4.2: MEM 头组（0:2）在解码侧同样强制零衰减 + β=1（与训练一致）
                    a_step = torch.cat([torch.ones_like(a_step[..., 0:2]), a_step[..., 2:]], dim=-1)
                    bd = torch.cat([torch.ones_like(bd[..., 0:2]), bd[..., 2:]], dim=-1)
                # 锚定 FLA 单 token 语义：S ← a·S + k⊗(β·k − β·a·(kS))（衰减在读出前应用）
                corr = kd - a_step.unsqueeze(-1) * sv                  # (B,H,D)
                s_new = a_step[..., None, None] * s_kv.to(qq.dtype) \
                    + (kd * bd.unsqueeze(-1)).unsqueeze(-1) * corr.unsqueeze(-2)  # (B,H,D,V)
                r = torch.einsum('bhd,bhdv->bhv', qq.to(qq.dtype).squeeze(2), s_new) \
                    * float(self.kv_dh) ** -0.5                         # 锚定 chunk_fwd_o scale
                if self.state_split:
                    r_aux = torch.einsum('bhd,hde->bhe', r, self.o_read.to(r.dtype))
                    mix = torch.sigmoid(self.o_mix).view(1, -1, 1).to(r.dtype)
                    r = mix * r + (1 - mix) * r_aux
                s_kv_new = s_new
            elif s_kv is not None and sx == 1:
                # 增量单步（O(1) 解码，sum 写入）：S ← a·S + k̂⊗k̂；r = q·S
                a_step = torch.exp(log_a[..., 0])                       # (B,H)
                s_new = a_step[..., None, None] * s_kv \
                    + torch.einsum('bhd,bhv->bhdv',
                                   kk.to(qq.dtype).squeeze(2), kk.to(qq.dtype).squeeze(2))
                r = torch.einsum('bhd,bhdv->bhv', qq.to(qq.dtype).squeeze(2), s_new)
                s_kv_new = s_new
            elif self.kv_delta:
                r, s_kv_new = tops.kv_delta_scan_tbptt(qq, kk, cc_perm, log_a, s0=s_kv)
            elif getattr(tops, 'TRITON_OK', False):
                r, s_kv_new = tops.KvScanFn.apply(qq, kk, cc_perm, log_a, s_kv)
            else:
                r, s_kv_new = tops.kv_chunked_scan(qq, kk, cc_perm, log_a, chunk=128, s0=s_kv)
            if r is not None and r.dim() == 4:
                rp = r.permute(0, 3, 2, 1)                       # (B, kv_dh, S, H)
                if self.kv_dh < self.d_head:
                    rp = F.pad(rp, (0, 0, 0, 0, 0, self.d_head - self.kv_dh))
                out_l = out_l + self._kv_out(rp)
            elif r is not None:
                rp = r.permute(0, 2, 1)                          # (B, kv_dh, H)
                if self.kv_dh < self.d_head:
                    rp = F.pad(rp, (0, 0, 0, self.d_head - self.kv_dh))
                out_l = out_l + self._kv_out(rp.unsqueeze(2))

        out = out_l.transpose(1, 2).contiguous().view(b, s, d)

        if self.kv_channel:
            return out, (e[:, :, -1:], s_kv_new)
        return out, e[:, :, -1:]

    def gen_model(self, a, b, c, d, e):

        combined = torch.cat([a, b, c, d, e], dim=-1)
        combined = self.head_linear(combined) * e
        term1 = a * b
        term2 = self.alpha1 * b + self.alpha2 * d
        term3 = a * (self.alpha3 * e + d)
        term4 = b * (c + e)
        return term1 + term2 + term3 + term4 + c * e + combined


class EmLinear(torch.nn.Module):
    """Weight generated live from the embedding bank: W = A @ em.weight[lo:hi].
    The em reference is held in a plain list (not a submodule) so parameters are
    not duplicated; gradients flow into the shared embedding."""

    def __init__(self, out_features, em, lo, hi, out_shape=None):
        super().__init__()
        R = out_shape[0] if out_shape else (hi - lo)
        self.A = torch.nn.Parameter(torch.randn(out_features, R) / R ** 0.5)
        self._em_ref = [em]
        self.lo, self.hi = lo, hi
        self.out_shape = out_shape

    def forward(self, x):
        w = self._em_ref[0].weight[self.lo:self.hi]
        if self.out_shape is not None:
            w = w.reshape(self.out_shape)
        return torch.nn.functional.linear(x, self.A @ w)


class EmFeedForward(torch.nn.Module):
    """Gated FFN with all three matrices generated from embedding-derived bases."""

    def __init__(self, hidden_size, ffn_hidden, em, act="silu"):
        super().__init__()
        R2 = hidden_size // 2
        self.ffn1 = EmLinear(ffn_hidden, em, 0, hidden_size)
        self.gate = EmLinear(ffn_hidden, em, 0, hidden_size)
        self.ffn2 = EmLinear(hidden_size, em, hidden_size,
                             hidden_size + R2 * ffn_hidden // hidden_size,
                             out_shape=(R2, ffn_hidden))
        self.act = {"relu": torch.nn.ReLU(), "silu": torch.nn.SiLU()}[act]

    def forward(self, x):
        x1 = self.ffn1(x)
        x2 = self.act(self.gate(x))
        return self.ffn2(x1 * x2)


class MoEFeedForward(torch.nn.Module):
    """Token-gated MoE-FFN: N gated-FFN experts, top-k routing (dense-compute,
    masked-accumulate formulation — correct, simple; token-sorted dispatch is a
    perf follow-up). Router zero-bias init -> near-uniform at start."""

    def __init__(self, hidden_size, act="silu", num_experts=4, topk=2, expert_hidden=None):
        super().__init__()
        eh = expert_hidden or hidden_size
        self.experts = torch.nn.ModuleList([FeedForward(hidden_size, act, ffn_hidden=eh) for _ in range(num_experts)])
        self.router = torch.nn.Linear(hidden_size, num_experts)
        self.topk = topk

    def forward(self, x):
        w = torch.softmax(self.router(x.float()), dim=-1)              # (B,S,N)
        topv, topi = w.topk(self.topk, dim=-1)
        topv = topv / (topv.sum(-1, keepdim=True) + 1e-9)
        y = torch.zeros_like(x)
        for e, expert in enumerate(self.experts):
            msk = (topi == e).any(-1, keepdim=True).to(x.dtype)        # (B,S,1) selected tokens
            wt = (topi == e).to(x.dtype) * topv                        # (B,S,k) this expert's weight
            wt = wt.sum(-1, keepdim=True)                              # (B,S,1)
            y = y + wt * expert(x * msk) * msk
        return y


class FeedForward(torch.nn.Module):
    def __init__(self, hidden_size, act="relu", ffn_hidden=None):
        super(FeedForward, self).__init__()
        fh = ffn_hidden or hidden_size
        self.ffn1 = torch.nn.Linear(hidden_size, fh)
        self.ffn2 = torch.nn.Linear(fh, hidden_size)
        self.gate = torch.nn.Linear(hidden_size, fh)

        self.act = {"relu": torch.nn.ReLU(), "silu": torch.nn.SiLU()}[act]

    def forward(self, x):
        x1 = self.ffn1(x)
        x2 = self.act(self.gate(x))
        xx = x1 * x2
        x = self.ffn2(xx)
        return x


class RMSNorm(torch.nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype)


class DecoderLayer(torch.nn.Module):
    def __init__(self, hidden_size, num_heads, voc_size, model_flag="train", use_fused=True,
                 state_rank=None, use_decay=True, pre_norm=False, ffn_act="relu",
                 decay_init=(0.001, 0.15), decay_v_std=0.0, kv_channel=False, kv_decay=0.98,
                 kv_norm_keys=False, kv_beta=False, kv_delta=False, kv_dim=None,
                 kv_backend='sum', kv_beta_init=0.0, short_conv=False, conv_kernel=4, kv_beta_tok=False,
                 em_ref=None, ffn_hidden=None, wkey_em=None, moe_ffn=False,
                 state_split=False, kv_out_gdn=False):
        super(DecoderLayer, self).__init__()
        self.self_attention_linear = MaxStateSuper(hidden_size, num_heads, voc_size, model_flag,
                                                   use_fused, state_rank, use_decay,
                                                   decay_init, decay_v_std,
                                                   kv_channel, kv_decay,
                                                   kv_norm_keys, kv_beta, kv_delta, kv_dim,
                                                   kv_backend, kv_beta_init, kv_beta_tok,
                                                   em_ref=em_ref, wkey_em=wkey_em,
                                                   state_split=state_split, kv_out_gdn=kv_out_gdn)
        if em_ref is not None:
            # v4.0: FFN 由 em 变换得到（fused CE/解码等其余路径不变）
            fh = ffn_hidden or hidden_size
            self.ffn = EmFeedForward(hidden_size, fh, em_ref, ffn_act)
        elif moe_ffn:
            self.ffn = MoEFeedForward(hidden_size, ffn_act, num_experts=4, topk=2,
                                      expert_hidden=hidden_size // 2)
        else:
            self.ffn = FeedForward(hidden_size, ffn_act)
        self.layer_norm = torch.nn.LayerNorm(hidden_size)
        self.pre_norm = pre_norm
        if pre_norm:
            self.norm1 = RMSNorm(hidden_size)
            self.norm2 = RMSNorm(hidden_size)

        # v3.7: depthwise causal short conv（GDN/DeltaNet/Mamba 标配）
        # 给线性注意力补上窗口内精确 n-gram 匹配能力
        self.short_conv = None
        if short_conv:
            self.short_conv = nn.Conv1d(hidden_size, hidden_size, conv_kernel,
                                        padding=conv_kernel - 1, groups=hidden_size,
                                        bias=False)
            nn.init.normal_(self.short_conv.weight, std=0.02)
            self.conv_gate = nn.Parameter(torch.zeros(hidden_size))  # per-ch gate

        self.alpha = torch.nn.Parameter(torch.tensor(0.5))

    def forward(self, x, ids, state=None):
        if self.pre_norm:
            xn = self.norm1(x)
            if self.short_conv is not None:
                # v3.9: 因果 conv 增量解码支持——状态携带最近 k−1 个输入的滚动缓冲
                # （无缓冲的全序列路径不变；sx==1 时用缓冲重建左上下文）
                conv_buf = None
                sub_state = state
                if isinstance(state, tuple) and len(state) == 3:
                    conv_buf = state[2]
                    sub_state = (state[0], state[1])
                xn_c = xn.transpose(1, 2)                          # (B, D, S)
                kc = self.short_conv.kernel_size[0]
                if conv_buf is not None and xn.shape[1] == 1:
                    cat = torch.cat([conv_buf, xn_c], dim=2)       # (B, D, buf+1)
                    # 最后一个输出位 = 真实 token 的因果卷积（tap 对齐：最新 token 用 w3）
                    xc = self.short_conv(cat)[:, :, cat.shape[2] - 1][:, :, None]
                    new_buf = cat[:, :, -(kc - 1):]
                else:
                    xc = self.short_conv(xn_c)[:, :, :xn.shape[1]]
                    new_buf = xn_c[:, :, -(kc - 1):]
                xn = xn + torch.nn.functional.silu(xc.transpose(1, 2)) * (1 + self.conv_gate)
            else:
                new_buf = None
                sub_state = state
            x1, sub_state = self.self_attention_linear(xn, ids, sub_state)
            if new_buf is not None:
                # 重打包：state = (e, S, conv_buf)（kv_channel）或 (e, conv_buf)
                if isinstance(sub_state, tuple) and len(sub_state) == 2:
                    sub_state = (sub_state[0], sub_state[1], new_buf)
                else:
                    sub_state = (state, None, new_buf)
            x = x + self.ffn(self.norm2(x1))
            state = sub_state
        else:
            x1, state = self.self_attention_linear(x, ids, state)
            x = self.layer_norm(self.alpha * self.ffn(x1) + (1 - self.alpha) * x)
        return x, state


class FusedCrossEntropy(torch.autograd.Function):
    """线性层 + CrossEntropy 融合：不物化 [N, V] logits（B=8,S=512,V=23004 fp32 约 377MB，
    反向链路上还有同量级的 softmax 中间量）。

    前向分块计算 logits → logsumexp + 目标 logit；反向重算 logits 分块，
    手工推导梯度：gx = (softmax - onehot) @ W，gW = (softmax - onehot)^T @ x。
    GEMM FLOPs 与普通路径相同，仅反向多一次 logits 重算，换来 O(chunk·V) 的峰值显存。
    """

    @staticmethod
    def forward(ctx, x, weight, targets, chunk_size):
        N = x.shape[0]
        lse = torch.empty(N, dtype=torch.float32, device=x.device)
        loss = x.new_zeros((), dtype=torch.float32)
        with torch.no_grad():
            for s in range(0, N, chunk_size):
                e = min(s + chunk_size, N)
                logits = (x[s:e] @ weight.t()).float()
                lse[s:e] = torch.logsumexp(logits, dim=-1)
                tgt = logits.gather(1, targets[s:e, None]).squeeze(1).float()
                loss = loss + (lse[s:e] - tgt).sum()
        ctx.save_for_backward(x, weight, targets, lse)
        ctx.chunk_size = chunk_size
        return loss / N

    @staticmethod
    def backward(ctx, g):
        x, weight, targets, lse = ctx.saved_tensors
        N, d = x.shape
        chunk = ctx.chunk_size
        wf = weight.float()
        gx = torch.empty_like(x)
        gW = torch.zeros(weight.shape[0], d, dtype=torch.float32, device=x.device)
        scale = float(g) / N   # forward 返回 mean，这里补 1/N
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            logits = (x[s:e] @ weight.t()).float()
            p = torch.softmax(logits, dim=-1)
            neg_one = p.new_full((e - s, 1), -1.0)
            p.scatter_add_(1, targets[s:e, None], neg_one)   # p - onehot(target)
            gx[s:e] = ((p @ wf) * scale).to(x.dtype)
            gW += (p.t() @ x[s:e].float()) * scale
        return gx, gW.to(weight.dtype), None, None


class OpenASH_V3(torch.nn.Module):
    def __init__(self, voc_size, hidden_size, num_heads, num_layers, max_seq_len=8192, model_flag="train",
                 use_fused=True, state_rank=64, ce_chunk=256, use_decay=True,
                 pre_norm=False, ffn_act="relu", decay_init=(0.001, 0.15), decay_v_std=0.0,
                 kv_channel=False, kv_decay=0.98, kv_norm_keys=False, kv_beta=False, kv_delta=False,
                 kv_dim=None, kv_backend='sum', kv_beta_init=0.0, short_conv=False, kv_beta_tok=False,
                 em_weights=False, ffn_hidden=None, id_weight=False, wkey_rank=64,
                 moe_ffn=False, state_split=False, kv_out_gdn=False, no_pos=False):
        super(OpenASH_V3, self).__init__()
        self.max_seq_len = max_seq_len
        self.ce_chunk = ce_chunk

        self.em = torch.nn.Embedding(voc_size, hidden_size, padding_idx=0)
        # v4.1: token-id 查表权重调制通路（state_em 模式；共享一张表服务所有层）
        self.wkey_em = nn.Embedding(voc_size, wkey_rank) if id_weight else None
        nn.init.normal_(self.wkey_em.weight, mean=0.0, std=0.02) if self.wkey_em is not None else None

        # v4.0: em 作为权重库——每层大权重 = A @ em 切片（与输入嵌入共享梯度），LM 头与 em 绑定
        em_ref = self.em if em_weights else None

        # v3: 可学习位置编码（GPT-2 风格），补上 v1/v2 完全缺失的位置信息
        self.pos_em = torch.nn.Embedding(max_seq_len, hidden_size)
        nn.init.normal_(self.pos_em.weight, mean=0.0, std=0.02)

        self.decoder_layers = torch.nn.ModuleList(
            [DecoderLayer(hidden_size, num_heads, voc_size, model_flag, use_fused, state_rank, use_decay,
                          pre_norm, ffn_act, decay_init, decay_v_std, kv_channel, kv_decay,
                          kv_norm_keys, kv_beta, kv_delta, kv_dim, kv_backend, kv_beta_init,
                          short_conv, kv_beta_tok=kv_beta_tok,
                          em_ref=em_ref, ffn_hidden=ffn_hidden, wkey_em=self.wkey_em,
                          moe_ffn=moe_ffn, state_split=state_split)
             for _ in range(num_layers)])
        self.final_norm = RMSNorm(hidden_size) if pre_norm else None
        self.em_weights_tied = em_weights
        self.no_pos = no_pos
        if em_weights:
            self.head_score = None
            self._head_weight = self.em.weight
        else:
            self.head_score = nn.Linear(hidden_size, voc_size, bias=False)

    def _hidden(self, x, state=None, pos_offset=0):
        """前向到最后一层隐藏态（不含 LM head）。"""
        b, s = x.shape
        ids = x  # 保留原始 token id，供各层 state_em 查表

        pos = torch.arange(pos_offset, pos_offset + s, device=x.device).clamp(max=self.max_seq_len - 1)
        x = self.em(x) + (0.0 if self.no_pos else self.pos_em(pos))

        if state is None:
            state = [None] * len(self.decoder_layers)
        for i, decoder_layer in enumerate(self.decoder_layers):
            x1, state[i] = decoder_layer(x, ids, state[i])
            x = x1 + x
        if self.final_norm is not None:
            x = self.final_norm(x)
        return x, state

    def forward(self, x, state=None, pos_offset=0):
        x, state = self._hidden(x, state, pos_offset)
        if self.em_weights_tied:
            return x @ self._head_weight.t(), state
        return self.head_score(x), state

    def forward_loss(self, ids, targets, state=None, pos_offset=0, fused_ce=True):
        """训练用：hidden → fused CE，跳过 [N, V] logits 物化。targets 为下一 token id。"""
        x, state = self._hidden(ids, state, pos_offset)
        w = self._head_weight if self.em_weights_tied else self.head_score.weight
        if fused_ce:
            loss = FusedCrossEntropy.apply(x.reshape(-1, x.shape[-1]), w,
                                           targets.reshape(-1), self.ce_chunk)
        else:
            logits = x @ w.t()
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(), targets.reshape(-1))
        return loss, state


if __name__ == '__main__':

    # 定义超参数
    voc_size = 12506
    num_layers = 8
    hidden_size = 2 ** 6 * num_layers
    num_heads = num_layers
    max_seq_len = 1024
    learning_rate = 0.001
    batch_size = 32
    num_epochs = 1000

    # 初始化模型
    model = OpenASH_V3(voc_size=voc_size, hidden_size=hidden_size, num_heads=num_heads,
                       num_layers=num_layers, max_seq_len=max_seq_len)
    params = 0
    for i in model.parameters():
        if i.shape != torch.Size([]):
            params += i.numel()
    print(params)
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=3)  # 忽略填充标记的损失计算
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练循环
    start_time = time.time()
    for epoch in range(num_epochs):
        data = torch.randint(low=0, high=voc_size, size=(batch_size, 50))  # 输入序列长度为50
        input_tensor = data[:, :-1]
        target_tensor = data[:, 1:]

        # 前向传播
        output, _ = model(input_tensor)

        # 将输出reshape以适应 CrossEntropyLoss 的输入要求
        output = output.reshape(-1, voc_size)
        target_tensor = target_tensor.reshape(-1)

        # 计算损失
        loss = criterion(output, target_tensor)
        optimizer.zero_grad()  # 清除梯度

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}--')

    print("Training complete.{}".format(time.time() - start_time))
