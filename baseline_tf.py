"""同预算参考基线：RoPE + pre-norm RMSNorm + SwiGLU + FlashAttention 因果 Transformer。

参数量对齐 v3.1（~96.5M）：hidden=768, layers=11, ffn=2048, lm_head 与 embedding 绑定。
接口与 OpenASH_V3 对齐：forward(x)->(logits,state)、forward_loss(ids,targets,fused_ce)。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from open_ash_v3 import RMSNorm, FusedCrossEntropy


def rope_cache(seq_len, dim, device, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv)                       # (S, dim/2)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x, cos, sin):
    """x: (B, H, S, Dh)，Dh 偶数。前 Dh/2 与后 Dh/2 旋转对（GPT-NeoX 风格半分）。"""
    dh = x.shape[-1]
    x1, x2 = x[..., :dh // 2], x[..., dh // 2:]
    out = torch.empty_like(x)
    out[..., :dh // 2] = x1 * cos - x2 * sin
    out[..., dh // 2:] = x2 * cos + x1 * sin
    return out


class Attn(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.dh = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.o = nn.Linear(dim, dim, bias=False)

    def forward(self, x, cos, sin):
        B, S, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=-1)
        q = q.view(B, S, self.heads, self.dh).transpose(1, 2)
        k = k.view(B, S, self.heads, self.dh).transpose(1, 2)
        v = v.view(B, S, self.heads, self.dh).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(y.transpose(1, 2).reshape(B, S, D))


class Block(nn.Module):
    def __init__(self, dim, heads, ffn_hidden):
        super().__init__()
        self.n1 = RMSNorm(dim)
        self.attn = Attn(dim, heads)
        self.n2 = RMSNorm(dim)
        self.w1 = nn.Linear(dim, ffn_hidden, bias=False)
        self.w3 = nn.Linear(dim, ffn_hidden, bias=False)
        self.w2 = nn.Linear(ffn_hidden, dim, bias=False)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.n1(x), cos, sin)
        x = x + self.w2(F.silu(self.w1(self.n2(x))) * self.w3(self.n2(x)))
        return x


class TransformerLM(nn.Module):
    """现代 Transformer 基线（Llama 风格最小集）。"""

    def __init__(self, voc_size, hidden_size, num_heads, num_layers, max_seq_len=2048,
                 ffn_hidden=2048, ce_chunk=256):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.ce_chunk = ce_chunk
        self.em = nn.Embedding(voc_size, hidden_size)
        self.blocks = nn.ModuleList([Block(hidden_size, num_heads, ffn_hidden)
                                     for _ in range(num_layers)])
        self.nf = RMSNorm(hidden_size)
        self.head_score = nn.Linear(hidden_size, voc_size, bias=False)
        self.head_score.weight = self.em.weight          # 绑定（参数量对齐 v3）
        cos, sin = rope_cache(max_seq_len, hidden_size // num_heads, 'cpu')
        self.register_buffer('rope_cos', cos, persistent=False)
        self.register_buffer('rope_sin', sin, persistent=False)
        nn.init.normal_(self.em.weight, mean=0.0, std=0.02)

    def _hidden(self, x, state=None, pos_offset=0):
        B, S = x.shape
        cos = self.rope_cos[pos_offset:pos_offset + S].to(x.dtype)[None, None]
        sin = self.rope_sin[pos_offset:pos_offset + S].to(x.dtype)[None, None]
        h = self.em(x)
        for blk in self.blocks:
            h = blk(h, cos, sin)
        return self.nf(h), None

    def forward(self, x, state=None, pos_offset=0):
        h, state = self._hidden(x, state, pos_offset)
        return self.head_score(h), state

    def forward_loss(self, ids, targets, state=None, pos_offset=0, fused_ce=True):
        h, _ = self._hidden(ids, state, pos_offset)
        if fused_ce:
            loss = FusedCrossEntropy.apply(h.reshape(-1, h.shape[-1]), self.head_score.weight,
                                           targets.reshape(-1), self.ce_chunk)
        else:
            logits = self.head_score(h)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(),
                                   targets.reshape(-1))
        return loss, None


if __name__ == '__main__':
    m = TransformerLM(23004, 768, 8, 11, max_seq_len=2048)
    n = sum(p.numel() for p in m.parameters())
    print(f'TransformerLM params: {n:,}')
    ids = torch.randint(0, 23004, (2, 128))
    loss, _ = m.forward_loss(ids[:, :-1], ids[:, 1:])
    print('smoke loss:', loss.item())
