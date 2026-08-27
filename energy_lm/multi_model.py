"""Multi-block hierarchical-equilibrium EnergyLM.

Instead of a single Energy Recurrent Block we stack ``L`` independent
contractive blocks.  Each block ``l`` has its own state ``h_l`` and relaxes to
its own equilibrium given the previous block's equilibrium as input:

    h_0 = X
    h_l = h_{l-1} + g * ( Attn_l(h_l) + FFN_l(h_l) ),   l = 1..L
    logits = h_L @ W_out

Every block is individually contractive (its own synaptic-scaling homeostasis),
so we get genuine depth (``L`` equilibrium layers) without forcing any single
block against the contractivity boundary.  The implicit gradient is propagated
**through the chain of equilibria**: each layer contributes its own
``(I - J)^{-1}`` adjoint (GMRES), and the adjoint passed to the previous layer
is the VJP of the block w.r.t. its input.  No gradient ever flows through the
relaxation iterations.  Memory is ``O(L)`` equilibrium states, independent of
the number of relaxation steps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .acceleration import AndersonAccel
from .energy_model import EnergyLMConfig
from .ep_trainer import _spectral_norm


# ===========================================================================
# One contractive block (its own weights + forward + relaxation)
# ===========================================================================
class EnergyBlock(nn.Module):
    def __init__(self, cfg: EnergyLMConfig):
        super().__init__()
        self.cfg = cfg
        d, dh = cfg.d_model, cfg.d_head
        self.Wq = nn.Parameter(torch.empty(cfg.n_heads, d, dh))
        self.Wk = nn.Parameter(torch.empty(cfg.n_heads, d, dh))
        self.Wv = nn.Parameter(torch.empty(cfg.n_heads, d, dh))
        self.Wo = nn.Parameter(torch.empty(cfg.n_heads, dh, d))
        self.W1 = nn.Parameter(torch.empty(d, cfg.d_ff))
        self.b1 = nn.Parameter(torch.zeros(cfg.d_ff))
        self.W2 = nn.Parameter(torch.empty(cfg.d_ff, d))
        self.b2 = nn.Parameter(torch.zeros(d))
        self.reset_parameters()

    def reset_parameters(self):
        for p, fan in [(self.Wq, self.cfg.d_model), (self.Wk, self.cfg.d_model),
                       (self.Wv, self.cfg.d_model), (self.Wo, self.cfg.d_model),
                       (self.W1, self.cfg.d_model), (self.W2, self.cfg.d_ff)]:
            bound = self.cfg.init_scale / math.sqrt(max(fan, 1))
            nn.init.uniform_(p, -bound, bound)
        nn.init.zeros_(self.b1); nn.init.zeros_(self.b2)

    # ---- differentiable single-block forward (autograd ON) --------------
    def forward_diff(self, Z: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        T = Z.size(1)
        dh = self.cfg.d_head
        Q = torch.einsum("btd,hde->bhte", Z, self.Wq)
        K = torch.einsum("btd,hde->bhte", Z, self.Wk)
        V = torch.einsum("btd,hde->bhte", Z, self.Wv)
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(dh)
        causal = torch.triu(
            torch.full((T, T), float("-inf"), device=Z.device, dtype=Z.dtype), diagonal=1
        )
        scores = scores + causal
        A = torch.softmax(scores, dim=-1)
        attn = A @ V
        attn_out = torch.einsum("bhte,hed->btd", attn, self.Wo)
        h_act = F.relu(Z @ self.W1 + self.b1)
        ffn = h_act @ self.W2 + self.b2
        return X + self.cfg.res_gain * (attn_out + ffn)

    # ---- relaxation to this block's equilibrium (no grad) ---------------
    @torch.no_grad()
    def relax(self, X: torch.Tensor, steps: int, anderson: bool = True,
              anderson_m: int = 5, anderson_beta: float = 0.7) -> Dict:
        Z = X.clone()
        Z = 0.5 * X + 0.5 * Z
        aa = AndersonAccel(anderson_m, anderson_beta) if anderson else None
        residual = None
        diverged = False
        for _ in range(steps):
            target = self.forward_diff(Z, X)
            residual = (target - Z).detach()
            if not torch.isfinite(target).all() or Z.abs().max().item() > 60.0:
                Z = 0.5 * Z + 0.5 * X
                diverged = True
                if aa is not None:
                    aa.reset()
                continue
            if aa is not None:
                Z = aa.step(Z, target)
            else:
                Z = Z + self.cfg.dt * (target - Z)
        return {"Z": Z, "residual": residual.norm().item() / max(1, Z.numel()),
                "diverged": diverged}

    # ---- per-block contractivity homeostasis ---------------------------
    @torch.no_grad()
    def enforce_contractivity(self, budget: float):
        for Wa, Wb in [(self.W1, self.W2), (self.Wv, self.Wo)]:
            sa = _spectral_norm(Wa.data); sb = _spectral_norm(Wb.data)
            prod = sa * sb
            if prod > budget and prod > 0:
                scale = (budget / prod) ** 0.5
                Wa.data.mul_(scale); Wb.data.mul_(scale)
        for W in (self.Wq, self.Wk):
            s = _spectral_norm(W.data)
            cap = math.sqrt(budget)
            if s > cap and s > 0:
                W.data.mul_(cap / s)

    @property
    def params(self) -> List[torch.Tensor]:
        return [self.Wq, self.Wk, self.Wv, self.Wo, self.W1, self.b1, self.W2, self.b2]


# ===========================================================================
# Multi-block container
# ===========================================================================
class MultiBlockERB(nn.Module):
    def __init__(self, cfg: EnergyLMConfig, n_blocks: int = 3):
        super().__init__()
        self.cfg = cfg
        self.n_blocks = n_blocks
        d = cfg.d_model
        self.tok_emb = nn.Parameter(torch.empty(cfg.vocab_size, d))
        self.register_buffer("_pos_cache", self._build_pos(cfg.max_seq_len, d), persistent=False)
        self.blocks = nn.ModuleList([EnergyBlock(cfg) for _ in range(n_blocks)])
        self.b_out = nn.Parameter(torch.zeros(cfg.vocab_size))
        if not cfg.tie_embeddings:
            self.W_out = nn.Parameter(torch.empty(d, cfg.vocab_size))
        self.reset_parameters()

    @staticmethod
    def _build_pos(seq_len, d):
        pos = torch.arange(seq_len).float().unsqueeze(1)
        inv = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe = torch.zeros(seq_len, d)
        pe[:, 0::2] = torch.sin(pos * inv)
        pe[:, 1::2] = torch.cos(pos * inv)
        return pe

    def reset_parameters(self):
        bound = self.cfg.init_scale / math.sqrt(self.cfg.d_model)
        nn.init.uniform_(self.tok_emb, -bound, bound)
        nn.init.zeros_(self.b_out)
        if not self.cfg.tie_embeddings:
            nn.init.uniform_(self.W_out, -bound, bound)

    @property
    def output_weight(self) -> torch.Tensor:
        return self.tok_emb.t() if self.cfg.tie_embeddings else self.W_out

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        T = ids.size(1)
        return F.embedding(ids, self.tok_emb) + self._pos_cache[:T].to(self.tok_emb.device).to(self.tok_emb.dtype)

    # ---- relax all blocks; return list of equilibria [X, h_1, ..., h_L] -
    @torch.no_grad()
    def relax_all(self, X: torch.Tensor, steps: int, anderson: bool = True,
                  anderson_m: int = 5, anderson_beta: float = 0.7) -> Dict:
        states = [X]
        residuals = []
        diverged = False
        h = X
        for blk in self.blocks:
            out = blk.relax(h, steps=steps, anderson=anderson,
                            anderson_m=anderson_m, anderson_beta=anderson_beta)
            h = out["Z"]
            states.append(h)
            residuals.append(out["residual"])
            diverged = diverged or out["diverged"]
        return {"states": states, "residuals": residuals, "diverged": diverged}

    @torch.no_grad()
    def logits_from_state(self, Z: torch.Tensor) -> torch.Tensor:
        return Z @ self.output_weight + self.b_out
