# -*- coding: utf-8 -*-
"""v6: 把「距离轴参数共享」移植成注意力偏置.
四种位置方案 (其余结构完全相同):
  rope          RoPE 旋转位置编码 (标准基线, 无加性偏置)
  alibi         ALiBi 固定线性斜率 (单调衰减, 现行方案)
  ballistic     可学习逐头高斯混合距离核 bias_h(d)=s_h·log(Σ_j w_hj·N(d;R_hj,σ_hj))
                — ALiBi 的泛化: 允许非单调 (近场峰 + 远场锚点), 直接对应 v5 的弹道原型
  ballistic_ng  同上, 但 R/σ 用 v5 n-gram 弹道拟合结果初始化 (近场簇 + k≈15 远场锚)
"""
import math, json, os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
CTX = 256


# ---------------- 偏置模块 ----------------

class RoPE(nn.Module):
    def __init__(self, dh, ctx=CTX, base=10000.0):
        super().__init__()
        self.dh, self.base = dh, base

    def rot(self, x):                                      # x: (B,H,L,dh)
        x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, q, k, dgrid=None):                   # 任意 L: 角度公式化生成 (标准外推基线)
        L = q.shape[2]
        inv = 1.0 / (self.base ** (torch.arange(0, self.dh, 2, device=q.device, dtype=q.dtype) / self.dh))
        t = torch.arange(L, device=q.device, dtype=q.dtype)
        freqs = torch.outer(t, inv)
        cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)[None, None]
        sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)[None, None]
        return q * cos + self.rot(q) * sin, k * cos + self.rot(k) * sin


class BallisticBias(nn.Module):
    """逐头 J 原型高斯混合距离核 (log 空间加性偏置). ALiBi 的弹道泛化.
    v6.1 修正 (来自 v5 包络教训 + v6 首轮诊断):
      - 可学习均匀地板 floor_h: bias = s_h·log(Σ_j w N_j + floor)  —— 核是偏差, 地板是背景
      - 分化头初始化 (diverse): 各头锚定不同距离带, 强制头间分工 (ALiBi 的斜率多样性)
    v6.3 新增 absolute 模式: 偏置按 key 绝对位置索引 (而非相对距离)——
      学习核在绝对参数化下反而更优 (可同时学 key 位置先验/attention sink), 见报告 8.5 节
    """
    NG_INIT = dict(R=[1.0, 2.0, 3.0, 15.0], sig=[1.0, 0.5, 0.7, 1.5])   # v5 n-gram 拟合的布局
    DIVERSE = {0: ([1, 2, 4, 8], [1.0, 1.0, 1.5, 2.0]),
               1: ([2, 5, 12, 30], [1.0, 1.5, 2.0, 3.0]),
               2: ([4, 12, 32, 80], [1.5, 2.0, 3.0, 4.0]),
               3: ([1, 8, 32, 128], [1.0, 2.0, 3.0, 5.0])}

    def __init__(self, heads, J=4, ctx=CTX, init='random', learn_floor=True, fit=None,
                 absolute=False):
        super().__init__()
        self.J, self.heads, self.absolute = J, heads, absolute
        if init == 'ngram':
            R0 = torch.tensor(self.NG_INIT['R'][:J], dtype=torch.float32).repeat(heads, 1)
            s0 = torch.tensor(self.NG_INIT['sig'][:J], dtype=torch.float32).repeat(heads, 1)
        elif init == 'fitted':
            # n-gram 弹道拟合 (autoballistic) 直接铺开到各头: 头 h 的带距倍数 m_h
            mult = [1.0, 2.5, 6.0, 15.0][:heads]
            Rf = torch.tensor(fit['R'][:J], dtype=torch.float32)
            sf = torch.tensor(fit['sig'][:J], dtype=torch.float32)
            R0 = torch.tensor([[max(r, 0.5) * m for r in Rf] for m in mult])
            s0 = torch.tensor([[s * math.sqrt(m) for s in sf] for m in mult])
        elif init == 'diverse':
            R0 = torch.tensor([self.DIVERSE[h % 4][0][:J] for h in range(heads)], dtype=torch.float32)
            s0 = torch.tensor([self.DIVERSE[h % 4][1][:J] for h in range(heads)], dtype=torch.float32)
        else:
            R0 = torch.linspace(1.0, ctx / 8.0, J).repeat(heads, 1)
            s0 = torch.full((heads, J), 2.0)
        self.r_raw = nn.Parameter(torch.zeros(heads, J))
        with torch.no_grad():                               # init: sigmoid 反解
            self.r_raw.copy_(torch.logit((R0.clamp(0.6, ctx - 1) - 0.5) / (ctx - 1.5)))
        self.s_raw = nn.Parameter(torch.log(torch.nn.functional.softplus(s0) - 0.25 + 1e-4))
        self.w_raw = nn.Parameter(torch.zeros(heads, J))
        self.scale = nn.Parameter(torch.ones(heads))
        floor0 = (fit or {}).get('floor', 0.01) if init == 'fitted' else (0.01 if learn_floor else 1e-9)
        self.log_floor = nn.Parameter(torch.full((heads,), math.log(floor0)))
        d = torch.arange(1, ctx + 1, dtype=torch.float32)
        self.register_buffer('d', d, persistent=False)

    def R_sigma_w(self):
        R = 0.5 + (CTX - 1.5) * torch.sigmoid(self.r_raw)
        sig = torch.nn.functional.softplus(self.s_raw) + 0.25
        w = torch.softmax(self.w_raw, dim=-1)
        return R, sig, w

    def bias(self, L=None):
        R, sig, w = self.R_sigma_w()
        n = L or self.d.shape[0]
        dd = torch.arange(1, n + 1, device=R.device, dtype=R.dtype)
        g = torch.exp(-(dd[None, None, :] - R[:, :, None]) ** 2 / (2 * sig[:, :, None] ** 2))
        mix = (g * w[:, :, None]).sum(1)                    # (H,n)
        floor = torch.exp(self.log_floor.clamp(-12.0, -2.0))
        return self.scale[:, None] * torch.log(mix + floor[:, None])   # (H,n) 长度可任意延伸

    def forward(self, q, k, dgrid=None):
        return q, k                                         # 偏置走 attn 的 mask 通道


class ALiBiBias(nn.Module):
    def __init__(self, heads, ctx=CTX):
        super().__init__()
        ex = torch.arange(1, heads + 1).float() * (-8.0 / heads)
        self.register_buffer('slopes', 2.0 ** ex, persistent=False)
        self.register_buffer('d', torch.arange(1, ctx + 1, dtype=torch.float32), persistent=False)

    def bias(self, L=None):
        n = L or self.d.shape[0]
        dd = torch.arange(1, n + 1, device=self.slopes.device, dtype=self.slopes.dtype)
        return -self.slopes[:, None] * dd[None, :]          # (H,n) 单调线性 (log 空间)


# ---------------- Transformer ----------------

class Block(nn.Module):
    def __init__(self, d, h, pos, bias):
        super().__init__()
        self.h, self.pos, self.bias = h, pos, bias
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.dh = d // h

    def attn(self, x, tbias=None):
        B, L, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=2)
        q = q.view(B, L, self.h, self.dh).transpose(1, 2)
        k = k.view(B, L, self.h, self.dh).transpose(1, 2)
        v = v.view(B, L, self.h, self.dh).transpose(1, 2)
        if isinstance(self.pos, RoPE):
            q, k = self.pos(q, k)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)      # (B,H,L,L)
        if tbias is not None:                                      # 内容条件化: 逐 token 对
            att = att + tbias
        elif self.bias is not None and hasattr(self.bias, 'bias'):
            b = self.bias.bias(L)                                  # (H,L)
            if getattr(self.bias, 'absolute', False):
                att = att + b[None, :, None, :]                    # 绝对参数化: 加在 key 位置
            else:
                pos = torch.arange(L, device=x.device)
                dmat = (pos[None, :] - pos[:, None]).clamp(min=0)  # dmat[i,j] = i−j (0↔距离1)
                att = att + b[:, dmat][None]                       # 相对距离参数化
        mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1)
        att = att.masked_fill(mask, float('-inf'))
        att = att.softmax(dim=-1)
        y = (att @ v).transpose(1, 2).reshape(B, L, D)
        return x + self.proj(y)

    def forward(self, x, tbias=None):
        x = self.attn(x, tbias)
        return x + self.mlp(self.ln2(x))


class ContentBallisticBias(nn.Module):
    """v6.3 内容条件化核: 每个键 token 学一个原型选择分布 w[tok,h,:],
    bias[b,h,i,j] = s_h·log( Σ_c w[tok_j,h,c]·N(d_ij; R_hc, σ_hc) + floor_h )
    —— 核仍只看相对距离, 但「用哪个原型」由内容决定 (标点近场 / 实词远场)"""

    def __init__(self, V, heads, J=4, ctx=CTX):
        super().__init__()
        self.core = BallisticBias(heads, J, ctx, init='diverse', learn_floor=True)
        self.Wtok = nn.Parameter(torch.zeros(V, heads, J))          # 零初始化 = 均匀混合

    def R_sigma_w(self):
        return self.core.R_sigma_w()

    def bias_for(self, idx):
        R, sig, w = self.core.R_sigma_w()
        L = idx.shape[1]
        d = torch.arange(1, L + 1, device=idx.device, dtype=torch.float32)
        Kd = torch.exp(-(d[None, None, :] - R[:, :, None]) ** 2 / (2 * sig[:, :, None] ** 2))
        pos = torch.arange(L, device=idx.device)
        dmat = (pos[None, :] - pos[:, None]).clamp(min=0)
        Kd = Kd[:, :, dmat]                                         # (H,J,L,L) 按距离取核值
        ws = torch.softmax(self.Wtok[idx], dim=-1)                  # (B,L,H,J)
        mix = torch.einsum('blhj,hjlm->bhlm', ws, Kd) + \
            torch.exp(self.core.log_floor.clamp(-12.0, -2.0))[None, :, None, None]
        return self.core.scale[None, :, None, None] * torch.log(mix)

    def forward(self, q, k, dgrid=None):
        return q, k


class AbsolutePrior(nn.Module):
    """逐 key 位置的可学习 logit 先验 (52 参数级). 刻意版的「bug 发现」:
    学习绝对位置先验 + RoPE —— 显式建模 attention sink 等位置频率效应"""

    def __init__(self, heads, ctx=CTX):
        super().__init__()
        self.ctx = ctx
        self.b = nn.Parameter(torch.zeros(heads, ctx))          # 零初始化: 初始等价无偏置

    def bias(self, L=None):
        n = L or self.ctx
        out = torch.zeros(self.b.shape[0], n, device=self.b.device, dtype=self.b.dtype)
        m = min(self.ctx, n)
        out[:, :m] = self.b[:, :m]                              # 未见过的位置保持中性 0
        return out

    def forward(self, q, k, dgrid=None):
        return q, k


class TinyLM(nn.Module):
    def __init__(self, V, d=256, n_layer=4, h=4, bias_kind='rope', J=4, ng_init=False, ctx=CTX,
                 bias_init=None):
        super().__init__()
        self.ctx = ctx
        self.bias = None
        if bias_kind == 'rope':
            self.pos, self.bias = RoPE(d // h), None
        elif bias_kind == 'alibi':
            self.pos, self.bias = None, ALiBiBias(h, ctx)
        elif bias_kind == 'ballistic':                      # v6 首轮: 硬地板 1e-9
            self.pos, self.bias = None, BallisticBias(h, J, ctx, init='ngram' if ng_init else 'random',
                                                      learn_floor=False)
        elif bias_kind == 'ballistic2':                     # v6.1: 可学习地板 (+可选 fitted 初始化)
            self.pos, self.bias = None, BallisticBias(
                h, J, ctx, init='fitted' if bias_init else 'diverse', learn_floor=True, fit=bias_init)
        elif bias_kind == 'rope_ballistic':                 # v6.1: RoPE + 加性弹道核
            self.pos = RoPE(d // h)
            self.bias = BallisticBias(h, J, ctx, init='diverse', learn_floor=True)
        elif bias_kind == 'ballistic3':                     # v6.3: 内容条件化距离核
            self.pos, self.bias = None, ContentBallisticBias(V, h, J, ctx)
        elif bias_kind == 'ballistic_abs':                  # v6.4: 学习核的绝对位置参数化
            self.pos, self.bias = None, BallisticBias(h, J, ctx, init='diverse',
                                                      learn_floor=True, absolute=True)
        elif bias_kind == 'rope_abs':                       # v6.4: RoPE + 可学习绝对位置先验
            self.pos = RoPE(d // h)
            self.bias = AbsolutePrior(h, ctx)
        else:
            raise ValueError(bias_kind)
        self.emb = nn.Embedding(V, d)
        self.blocks = nn.ModuleList([Block(d, h, self.pos, self.bias) for _ in range(n_layer)])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, V, bias=False)
        self.head.weight = self.emb.weight                          # tied
        # GPT 风格初始化 (否则 tied embedding 的 N(0,1) 会让 logits 爆炸)
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for b in self.blocks:                                       # 残差投影缩小
            nn.init.normal_(b.proj.weight, std=0.02 / math.sqrt(2 * n_layer))
            nn.init.normal_(b.mlp[2].weight, std=0.02 / math.sqrt(2 * n_layer))

    def forward(self, idx):
        x = self.emb(idx)
        tbias = None
        if isinstance(self.bias, ContentBallisticBias):
            tbias = self.bias.bias_for(idx)
        for b in self.blocks:
            x = b(x, tbias)
        return self.head(self.lnf(x))

    def ngram_bias_json(self):
        if not isinstance(self.bias, BallisticBias):
            return None
        R, sig, w = self.R_sigma_w_cpu()
        return dict(R=R.tolist(), sig=sig.tolist(), w=w.tolist(),
                    scale=self.bias.scale.detach().cpu().tolist(),
                    floor=torch.exp(self.bias.log_floor.clamp(-12, -2)).detach().cpu().tolist())

    def R_sigma_w_cpu(self):
        R, sig, w = self.bias.R_sigma_w()
        return R.detach().cpu(), sig.detach().cpu(), w.detach().cpu()
