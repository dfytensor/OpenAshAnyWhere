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
                 kv_delta=False, kv_dim=None, kv_backend='sum', kv_beta_init=0.0):
        super(MaxStateSuper, self).__init__()
        self.heads = heads
        self.d_head = int(dim_size // heads)
        self.kv_dh = kv_dim if kv_dim else self.d_head   # KV 通道头维（瘦身消融用）
        self.kv_backend = kv_backend                     # 'sum'（v3.5s）| 'delta_fla'（v3.6）
        self.model_flag = model_flag
        self.use_fused = use_fused
        self.use_decay = use_decay
        assert dim_size % heads == 0, "Dimension size must be divisible by head size."

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
            # v3.6: delta 写入走 FLA chunk_gated_delta_rule（写入强度 β 可学习）
            if self.kv_backend == 'delta_fla':
                self.kv_beta_fla = nn.Parameter(torch.full((heads,), kv_beta_init))

        # v1/v2 为 4 路投影；v3 中 c 路改由 state_em 查表提供，故只需 3 路
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

    def forward(self, x, ids, state=None):
        b, s, d = x.shape
        # 布局 [b, s, 3, d_head, heads] → 各流 permute 为 [b, d_head, s, heads]（heads 连续，利于合并访问）
        combined = self.combined(x).view(b, s, 3, self.d_head, self.heads)
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
            if self.kv_backend == 'delta_fla' and FLA_GDN_OK:
                r = None
                # ---- v3.6：delta-rule 写入（FLA chunk_gated_delta_rule，含 backward）----
                # FLA 布局：q/k/v (B,S,H,D)、g (B,S,H) log 域、beta (B,S,H)
                qq_f = qq.permute(0, 2, 1, 3).float()        # (B,S,H,kv_dh)
                kk_f = kk.permute(0, 2, 1, 3).float()        # (B,S,H,kv_dh) 归一化键
                vv_f = kk_f.float()                          # delta 写入 v = k
                g_f = log_a.permute(0, 2, 1).float()         # (B,S,H) log 域衰减
                beta_f = torch.sigmoid(self.kv_beta_fla).view(1, 1, -1)\
                    .expand(bx, sx, self.heads).to(qq_f.dtype)
                o, s_kv_new = chunk_gated_delta_rule(
                    q=qq_f, k=kk_f, v=vv_f, g=g_f, beta=beta_f,
                    initial_state=s_kv, output_final_state=True,
                    use_qk_l2norm_in_kernel=False)
                # o (B,S,H,kv_dh) → gen_model 布局 (B,d_head,S,H)
                o_p = o.permute(0, 3, 1, 2)                     # (B,kv_dh,S,H)
                if self.kv_dh < self.d_head:
                    o_p = F.pad(o_p, (0, 0, 0, 0, 0, self.d_head - self.kv_dh))
                out_l = out_l + self.kv_gate * o_p.to(out_l.dtype)
            elif s_kv is not None and sx == 1:
                # 增量单步（O(1) 解码）：S ← a·S + (β·k̂)⊗(β·k̂)；r = q·S——纯 torch 小 op
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
                out_l = out_l + self.kv_gate * rp.to(out_l.dtype)
            elif r is not None:
                rp = r.permute(0, 2, 1)                          # (B, kv_dh, H)
                if self.kv_dh < self.d_head:
                    rp = F.pad(rp, (0, 0, 0, self.d_head - self.kv_dh))
                out_l = out_l + self.kv_gate * rp.unsqueeze(2).to(out_l.dtype)

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


class FeedForward(torch.nn.Module):
    def __init__(self, hidden_size, act="relu"):
        super(FeedForward, self).__init__()
        self.ffn1 = torch.nn.Linear(hidden_size, hidden_size)
        self.ffn2 = torch.nn.Linear(hidden_size, hidden_size)
        self.gate = torch.nn.Linear(hidden_size, hidden_size)

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
                 kv_backend='sum', kv_beta_init=0.0, short_conv=False, conv_kernel=4):
        super(DecoderLayer, self).__init__()
        self.self_attention_linear = MaxStateSuper(hidden_size, num_heads, voc_size, model_flag,
                                                   use_fused, state_rank, use_decay,
                                                   decay_init, decay_v_std,
                                                   kv_channel, kv_decay,
                                                   kv_norm_keys, kv_beta, kv_delta, kv_dim,
                                                   kv_backend, kv_beta_init)
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
                # 因果 depthwise conv：左 pad，截尾
                xc = self.short_conv(xn.transpose(1, 2))[:, :, :xn.shape[1]].transpose(1, 2)
                xn = xn + torch.nn.functional.silu(xc) * (1 + self.conv_gate)
            x1, state = self.self_attention_linear(xn, ids, state)
            x = x + self.ffn(self.norm2(x1))
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
                 kv_dim=None, kv_backend='sum', kv_beta_init=0.0, short_conv=False):
        super(OpenASH_V3, self).__init__()
        self.max_seq_len = max_seq_len
        self.ce_chunk = ce_chunk

        self.em = torch.nn.Embedding(voc_size, hidden_size, padding_idx=0)

        # v3: 可学习位置编码（GPT-2 风格），补上 v1/v2 完全缺失的位置信息
        self.pos_em = torch.nn.Embedding(max_seq_len, hidden_size)
        nn.init.normal_(self.pos_em.weight, mean=0.0, std=0.02)

        self.decoder_layers = torch.nn.ModuleList(
            [DecoderLayer(hidden_size, num_heads, voc_size, model_flag, use_fused, state_rank, use_decay,
                          pre_norm, ffn_act, decay_init, decay_v_std, kv_channel, kv_decay,
                          kv_norm_keys, kv_beta, kv_delta, kv_dim, kv_backend, kv_beta_init,
                          short_conv)
             for _ in range(num_layers)])
        self.final_norm = RMSNorm(hidden_size) if pre_norm else None
        self.head_score = nn.Linear(hidden_size, voc_size, bias=False)

    def _hidden(self, x, state=None, pos_offset=0):
        """前向到最后一层隐藏态（不含 LM head）。"""
        b, s = x.shape
        ids = x  # 保留原始 token id，供各层 state_em 查表

        pos = torch.arange(pos_offset, pos_offset + s, device=x.device).clamp(max=self.max_seq_len - 1)
        x = self.em(x) + self.pos_em(pos)

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
        x_out = self.head_score(x)
        return x_out, state

    def forward_loss(self, ids, targets, state=None, pos_offset=0, fused_ce=True):
        """训练用：hidden → fused CE，跳过 [N, V] logits 物化。targets 为下一 token id。"""
        x, state = self._hidden(ids, state, pos_offset)
        if fused_ce:
            loss = FusedCrossEntropy.apply(x.reshape(-1, x.shape[-1]), self.head_score.weight,
                                           targets.reshape(-1), self.ce_chunk)
        else:
            logits = self.head_score(x)
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
