"""Standard backprop Transformer baseline (same depth/width as the ERB's *map*).

This is NOT trained by equilibrium propagation; it exists only so we can
compare the local EP rule against ordinary global backprop on the same data and
roughly the same parameter budget.  The baseline is a single-block residual
Transformer (attention + FFN) unrolled for a fixed number of "virtual layers"
to match the effective depth of the ERB relaxation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BaselineConfig:
    vocab_size: int = 64
    d_model: int = 64
    n_heads: int = 4
    d_ff: int = 128
    n_layers: int = 1     # one block keeps params comparable to the single ERB
    max_seq_len: int = 48
    device: str = "cuda"


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: BaselineConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d = cfg.d_model
        self.dh = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.o = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x):
        B, T, d = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.dh)
        q, k, v = qkv.unbind(dim=2)            # (B,T,H,dh) each
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]  # (B,H,T,dh)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)
        mask = torch.triu(
            torch.full((T, T), float("-inf"), device=x.device), diagonal=1
        )
        scores = scores + mask
        a = torch.softmax(scores, dim=-1)
        out = (a @ v).transpose(1, 2).reshape(B, T, d)
        return self.o(out)


class BaselineTransformer(nn.Module):
    def __init__(self, cfg: BaselineConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Parameter(torch.empty(cfg.vocab_size, cfg.d_model))
        nn.init.uniform_(self.tok_emb, -0.5 / math.sqrt(cfg.d_model), 0.5 / math.sqrt(cfg.d_model))
        self.register_buffer("pos", self._pos(cfg.max_seq_len, cfg.d_model), persistent=False)
        self.attns = nn.ModuleList([CausalSelfAttention(cfg) for _ in range(cfg.n_layers)])
        self.ffn1 = nn.Linear(cfg.d_model, cfg.d_ff)
        self.ffn2 = nn.Linear(cfg.d_ff, cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size)

    @staticmethod
    def _pos(seq_len, d):
        pos = torch.arange(seq_len).float().unsqueeze(1)
        inv = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe = torch.zeros(seq_len, d)
        pe[:, 0::2] = torch.sin(pos * inv)
        pe[:, 1::2] = torch.cos(pos * inv)
        return pe

    def forward(self, ids):
        T = ids.size(1)
        x = self.tok_emb[ids]
        x = x + self.pos[:T].to(x.device).to(x.dtype)
        h = x
        for attn in self.attns:
            h = h + attn(h)
        h = h + self.ffn2(F.relu(self.ffn1(h)))
        return self.head(h)

    @torch.no_grad()
    def generate(self, tokenizer, prompt: str, n_new: int = 60, temperature=0.8):
        device = self.tok_emb.device
        ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
        for _ in range(n_new):
            ctx = ids[:, -self.cfg.max_seq_len:]
            logits = self.forward(ctx)[0, -1] / max(temperature, 1e-5)
            probs = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            ids = torch.cat([ids, nxt.unsqueeze(0)], dim=1)
        return tokenizer.decode(ids[0])
