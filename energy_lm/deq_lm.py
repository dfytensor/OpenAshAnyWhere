"""Standard DEQ-LM (Bai-Kolter-Koltun 2019 style) for the head-to-head test.

This is the canonical implicit-depth language model: a *proper* pre-norm
Transformer cell, weight-tied, whose hidden state is the **equilibrium**
``z* = f(z*, x)`` found by Anderson acceleration, and trained by ordinary
backpropagation through the implicit layer (IFT adjoint, implemented as a
``register_hook`` that solves the transpose fixed-point equation with
Anderson).  Crucially — and unlike every other model in this repo — it does
**not** enforce strict contraction; it relies on small weight init + LayerNorm
for *empirical* stability, exactly as the DEQ paper does.

Purpose: a controlled head-to-head on the *same* MiniMind data.  If this proper
DEQ-LM reaches a much lower CE than our strict-contraction EnergyLM/DEQ
(``0.80``), it empirically confirms the §2.0 claim that the 0.80 ceiling is an
artefact of strict contraction + a bare cell, not of the equilibrium formulation
itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from .acceleration import AndersonAccel


@dataclass
class DEQLMConfig:
    vocab_size: int = 4500
    d_model: int = 192
    n_heads: int = 6
    d_ff: int = 512
    max_seq_len: int = 128
    res_gain: float = 1.0        # the DEQ cell already contains its own residual scaling
    init_std: float = 0.02       # DEQ needs small init (per Bai 2019 / the tutorial)
    anderson_m: int = 5
    anderson_beta: float = 0.7
    fwd_iters: int = 30          # max Anderson iterations, forward
    bwd_iters: int = 30          # max Anderson iterations, adjoint
    tol: float = 1e-4            # convergence tolerance (relative residual)
    device: str = "cuda"


# ===========================================================================
# Pre-norm Transformer cell (the DEQ "layer" f)
# ===========================================================================
class TransformerCell(nn.Module):
    def __init__(self, cfg: DEQLMConfig):
        super().__init__()
        self.cfg = cfg
        d, dh = cfg.d_model, cfg.d_model // cfg.n_heads
        self.n_heads = cfg.n_heads
        self.dh = dh
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.Wq = nn.Linear(d, d, bias=False)
        self.Wk = nn.Linear(d, d, bias=False)
        self.Wv = nn.Linear(d, d, bias=False)
        self.Wo = nn.Linear(d, d, bias=False)
        self.W1 = nn.Linear(d, cfg.d_ff)
        self.W2 = nn.Linear(cfg.d_ff, d)
        self._reset()

    def _reset(self):
        for p in self.parameters():
            nn.init.normal_(p, std=self.cfg.init_std)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """NON-residual transformer cell: attn(LN) + ffn(LN).  The DEQ's
        'depth' comes from the equilibrium iteration, so the cell itself is a
        clean non-residual map; combined with input injection this gives a
        well-defined (non-degenerate) fixed point."""
        B, T, d = z.shape
        h = self.ln1(z)
        q = self.Wq(h).view(B, T, self.n_heads, self.dh).transpose(1, 2)
        k = self.Wk(h).view(B, T, self.n_heads, self.dh).transpose(1, 2)
        v = self.Wv(h).view(B, T, self.n_heads, self.dh).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)
        mask = torch.triu(torch.full((T, T), float("-inf"), device=z.device, dtype=z.dtype), 1)
        scores = scores + mask
        attn = torch.softmax(scores, dim=-1) @ v
        attn = attn.transpose(1, 2).reshape(B, T, d)
        a = self.Wo(attn)
        h = self.ln2(z)
        f = self.W2(F.gelu(self.W1(h)))
        return a + f


# ===========================================================================
# Anderson solvers (batched, type-II)
# ===========================================================================
def anderson_solve(f, x0, m=5, beta=0.7, lam=1e-4, max_iter=30, tol=1e-4):
    """Solve z = f(z) by batched type-II Anderson. Returns (z, n_iters)."""
    aa = AndersonAccel(m, beta)
    z = x0
    last_res = 1.0
    for k in range(max_iter):
        fz = f(z)
        res = (fz - z).flatten(1).norm(dim=1).mean().item() / (fz.flatten(1).norm(dim=1).mean().item() + 1e-8)
        last_res = res
        z_new = aa.step(z, fz)
        if res < tol:
            z = z_new
            break
        z = z_new
    return z, last_res


# ===========================================================================
# DEQ layer with IFT backward (register_hook, the canonical DEQ mechanism)
# ===========================================================================
class DEQLayer(nn.Module):
    def __init__(self, cell: TransformerCell, cfg: DEQLMConfig):
        super().__init__()
        self.cell = cell
        self.cfg = cfg
        self.fwd_res = 1.0
        self.bwd_res = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        # input-injected, non-degenerate fixed-point map:  z* = x + g*cell(z*)
        f = lambda z: x + cfg.res_gain * self.cell(z)

        with torch.no_grad():
            z, self.fwd_res = anderson_solve(
                f, x.clone(), m=cfg.anderson_m, beta=cfg.anderson_beta,
                max_iter=cfg.fwd_iters, tol=cfg.tol,
            )
        # re-engage autograd tape at the equilibrium
        z = f(z)

        if torch.is_grad_enabled():
            # IFT backward: solve (I - J^T) g = grad, i.e. g = J^T g + grad, via Anderson
            z0 = z.detach().requires_grad_(True)
            f0 = f(z0)

            def backward_hook(grad):
                g, self.bwd_res = anderson_solve(
                    lambda y: autograd.grad(f0, z0, y, retain_graph=True, create_graph=False)[0] + grad,
                    grad.clone(), m=cfg.anderson_m, beta=cfg.anderson_beta,
                    max_iter=cfg.bwd_iters, tol=cfg.tol,
                )
                return g

            z.register_hook(backward_hook)
        return z


# ===========================================================================
# Full DEQ-LM
# ===========================================================================
class DEQLM(nn.Module):
    def __init__(self, cfg: DEQLMConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.tok_emb = nn.Parameter(torch.empty(cfg.vocab_size, d))
        nn.init.normal_(self.tok_emb, std=cfg.init_std)
        self.register_buffer("pos", self._pos(cfg.max_seq_len, d), persistent=False)
        self.cell = TransformerCell(cfg)
        self.deq = DEQLayer(self.cell, cfg)
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb  # tie
        self.b_out = nn.Parameter(torch.zeros(cfg.vocab_size))

    @staticmethod
    def _pos(T, d):
        pos = torch.arange(T).float().unsqueeze(1)
        inv = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe = torch.zeros(T, d)
        pe[:, 0::2] = torch.sin(pos * inv); pe[:, 1::2] = torch.cos(pos * inv)
        return pe

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        T = ids.size(1)
        x = self.tok_emb[ids] + self.pos[:T].to(self.tok_emb.device).to(self.tok_emb.dtype)
        z = self.deq(x)
        z = self.ln_f(z)
        return self.head(z) + self.b_out

    @torch.no_grad()
    def generate(self, tok, prompt, n_new=80, temperature=0.5, top_k=12):
        dev = self.tok_emb.device
        ids = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=dev)
        for _ in range(n_new):
            ctx = ids[:, -self.cfg.max_seq_len:]
            logits = self.forward(ctx)[0, -1] / max(temperature, 1e-5)
            if top_k > 0:
                v, _ = torch.topk(logits, top_k); logits[logits < v[-1]] = float("-inf")
            ids = torch.cat([ids, torch.multinomial(torch.softmax(logits, -1), 1).unsqueeze(0)], 1)
        return tok.decode(ids[0])
