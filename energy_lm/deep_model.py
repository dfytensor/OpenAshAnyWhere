"""Deep-map single-equilibrium EnergyLM.

Unlike the chained multi-block model (``multi_model.py``), here there is
**one** equilibrium ``Z* = X + g * U(Z*)`` but the recurrent map ``U`` is a
composition of ``L`` non-residual sub-layers,

    U(Z) = s_L ∘ s_{L-1} ∘ ... ∘ s_1 (Z),     s_l(z) = FFN_l(Attn_l(z))

so each fixed-point iteration applies a genuinely deep transform.  Because
there is only one fixed point, we relax once and solve **one** GMRES adjoint
on the full composed Jacobian (autograd handles the composition automatically).
This keeps the stability of the single-block DEQ while adding depth — avoiding
both the coupled-equilibrium stiffness and the compounding adjoint error that
sank the chained multi-block model.

Contractivity is enforced per sub-layer: the homeostasis keeps each layer's
Lipschitz estimate below ``(contractivity / g) ** (1/L)`` so that the product
``g * Π_l L(s_l)`` stays under the contractivity target and the (single)
Neumann/GMRES adjoint converges.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .acceleration import AndersonAccel
from .energy_model import EnergyLMConfig
from .ep_trainer import _spectral_norm


class MapLayer(nn.Module):
    """One non-residual sub-layer: ``s(z) = FFN(Attn(z))`` (causal attention)."""

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

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
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
        h_act = F.relu(attn_out @ self.W1 + self.b1)
        return h_act @ self.W2 + self.b2

    @property
    def params(self) -> List[torch.Tensor]:
        return [self.Wq, self.Wk, self.Wv, self.Wo, self.W1, self.b1, self.W2, self.b2]

    @torch.no_grad()
    def lip_estimate(self) -> float:
        # additive Lipschitz upper bound for a residual sub-block's increment
        # V(z) = ffn(attn(z)):  L(V) ~ sigma(Wv)*sigma(Wo) + sigma(W1)*sigma(W2)
        return (_spectral_norm(self.Wv.data) * _spectral_norm(self.Wo.data) +
                _spectral_norm(self.W1.data) * _spectral_norm(self.W2.data))

    @torch.no_grad()
    def scale_to(self, target_lip: float):
        cur = self.lip_estimate()
        if cur > target_lip and cur > 0:
            s = (target_lip / cur) ** 0.5   # split across the 4 matrices
            for W in (self.Wq, self.Wk, self.Wv, self.Wo, self.W1, self.W2):
                W.data.mul_(s)


class DeepERB(nn.Module):
    def __init__(self, cfg: EnergyLMConfig, n_map_layers: int = 2):
        super().__init__()
        self.cfg = cfg
        self.n_map_layers = n_map_layers
        d = cfg.d_model
        self.tok_emb = nn.Parameter(torch.empty(cfg.vocab_size, d))
        self.register_buffer("_pos_cache", self._build_pos(cfg.max_seq_len, d), persistent=False)
        self.layers = nn.ModuleList([MapLayer(cfg) for _ in range(n_map_layers)])
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

    def map_U(self, Z: torch.Tensor) -> torch.Tensor:
        """RESIDUAL composed transform: h = Z; h = h + V_l(h) for each layer.

        Residual structure lets information flow through (so depth is useful),
        unlike pure composition.  Each sub-block's increment V_l has an
        additive Lipschitz bound (attn + ffn in parallel), enforced by
        ``DeepDEQTrainer._enforce_contractivity``.
        """
        h = Z
        for layer in self.layers:
            h = h + layer(h)
        return h

    def forward_diff(self, Z: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """Recurrent map f(Z) = X + g * U(Z).  One equilibrium; autograd-ready."""
        return X + self.cfg.res_gain * self.map_U(Z)

    @torch.no_grad()
    def relax(self, X, steps, anderson=True, anderson_m=5, anderson_beta=0.7):
        Z = X.clone(); Z = 0.5 * X + 0.5 * Z
        aa = AndersonAccel(anderson_m, anderson_beta) if anderson else None
        residual = None; diverged = False
        for _ in range(steps):
            target = self.forward_diff(Z, X)
            residual = (target - Z).detach()
            if not torch.isfinite(target).all() or Z.abs().max().item() > 60.0:
                Z = 0.5 * Z + 0.5 * X; diverged = True
                if aa is not None: aa.reset()
                continue
            Z = aa.step(Z, target) if aa is not None else Z + self.cfg.dt * (target - Z)
        return {"Z": Z, "residual": residual.norm().item() / max(1, Z.numel()), "diverged": diverged}

    @torch.no_grad()
    def logits_from_state(self, Z):
        return Z @ self.output_weight + self.b_out

    @property
    def recurrent_params(self) -> List[torch.Tensor]:
        ps = []
        for layer in self.layers:
            ps.extend(layer.params)
        return ps
