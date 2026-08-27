"""EnergyLM core: a Transformer reinterpreted as a continuous-time energy system.

This module implements the *Energy Recurrent Block* (ERB) from the design
document.  Instead of stacking discrete Transformer layers we define a single
recurrent map over a hidden state ``Z`` and relax it to a steady state with a
local, gradient-flow-on-the-state dynamics.  Learning uses **Equilibrium
Propagation (EP)**: weights are updated from the difference of local
pre-/post-synaptic correlations measured at the *free* steady state
(``beta = 0``) and the *clamped* steady state (``beta > 0``, nudged toward the
target).  No global backpropagation is used anywhere.

Concretely the recurrent map is a residual Transformer block

    f(Z) = X + Attention(Z) + FFN(Z)

with causal self-attention, and the relaxation

    Z <- Z + dt * ( f(Z) - Z )

converges to the fixed point ``Z* = f(Z*)`` (a deep-equilibrium-style fixed
point).  The associated Lyapunov / energy surrogate

    E(Z) = 0.5 * ||Z - f(Z)||_F^2

is non-negative and decreases along trajectories when ``dt`` is small enough.

The output head adds a cross-entropy cost ``C(Z, y)`` over next-token logits.
During the clamped phase the dynamics receive an extra local injection
``-beta * dC/dZ``; this is the only place the target touches the network and it
is a purely local signal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class EnergyLMConfig:
    vocab_size: int = 64
    d_model: int = 64          # state dimension d
    n_heads: int = 4           # multi-head attention
    d_ff: int = 128            # FFN hidden width
    max_seq_len: int = 48      # context length
    dt: float = 0.5            # relaxation step size
    free_steps: int = 24       # relaxation iterations, free phase
    clamped_steps: int = 24    # relaxation iterations, clamped phase
    act_avg: int = 4           # trailing steps to average correlations over
    beta: float = 1.0          # clamping / nudging strength
    init_scale: float = 0.5    # weight init (fan-in scaled internally)
    norm_eps: float = 1e-6     # RMSNorm epsilon
    res_gain: float = 0.5      # gain on attention+FFN residual (controls fixed-point richness vs stability)
    use_norm: bool = True      # RMSNorm (pre-norm) inside the recurrent block
    tie_embeddings: bool = True  # share input tok_emb and output head
    device: str = "cuda"
    dtype: torch.dtype = torch.float32

    @property
    def d_head(self) -> int:
        assert self.d_model % self.n_heads == 0, "d_model must divide n_heads"
        return self.d_model // self.n_heads


# ===========================================================================
# Energy Recurrent Block
# ===========================================================================
class EnergyRecurrentBlock(nn.Module):
    """The recurrent map ``f(Z) = X + Attention(Z) + FFN(Z)`` plus the local
    "energy landscape" that defines the relaxation dynamics.

    Every affine map keeps references to its input so that, at the steady
    state, we can read off the pre-/post-synaptic correlations needed by the EP
    update rule.
    """

    def __init__(self, cfg: EnergyLMConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model

        # --- token embedding + positional code -------------------------
        self.tok_emb = nn.Parameter(torch.empty(cfg.vocab_size, d))
        self.register_buffer("_pos_cache", self._build_pos(cfg.max_seq_len, d), persistent=False)

        # --- attention parameters (Q, K, V, O) -------------------------
        self.Wq = nn.Parameter(torch.empty(cfg.n_heads, d, cfg.d_head))
        self.Wk = nn.Parameter(torch.empty(cfg.n_heads, d, cfg.d_head))
        self.Wv = nn.Parameter(torch.empty(cfg.n_heads, d, cfg.d_head))
        self.Wo = nn.Parameter(torch.empty(cfg.n_heads, cfg.d_head, d))
        # --- FFN parameters -------------------------------------------
        self.W1 = nn.Parameter(torch.empty(d, cfg.d_ff))
        self.b1 = nn.Parameter(torch.zeros(cfg.d_ff))
        self.W2 = nn.Parameter(torch.empty(cfg.d_ff, d))
        self.b2 = nn.Parameter(torch.zeros(d))
        # --- RMSNorm gains (pre-norm) ---------------------------------
        self.ln1_g = nn.Parameter(torch.ones(d))
        self.ln2_g = nn.Parameter(torch.ones(d))
        # --- output head ----------------------------------------------
        self.b_out = nn.Parameter(torch.zeros(cfg.vocab_size))
        if not cfg.tie_embeddings:
            self.W_out = nn.Parameter(torch.empty(d, cfg.vocab_size))

        self.reset_parameters()

    # ------------------------------------------------------------------
    @staticmethod
    def _build_pos(seq_len: int, d: int) -> torch.Tensor:
        pos = torch.arange(seq_len).float().unsqueeze(1)
        inv_freq = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe = torch.zeros(seq_len, d)
        pe[:, 0::2] = torch.sin(pos * inv_freq)
        pe[:, 1::2] = torch.cos(pos * inv_freq)
        return pe

    def reset_parameters(self):
        params = [
            (self.tok_emb, self.cfg.d_model),
            (self.Wq, self.cfg.d_model),
            (self.Wk, self.cfg.d_model),
            (self.Wv, self.cfg.d_model),
            (self.Wo, self.cfg.d_model),
            (self.W1, self.cfg.d_model),
            (self.W2, self.cfg.d_ff),
        ]
        if not self.cfg.tie_embeddings:
            params.append((self.W_out, self.cfg.d_model))
        for p, fan in params:
            bound = self.cfg.init_scale / math.sqrt(max(fan, 1))
            nn.init.uniform_(p, -bound, bound)
        nn.init.zeros_(self.b1)
        nn.init.zeros_(self.b2)
        nn.init.zeros_(self.b_out)
        nn.init.ones_(self.ln1_g)
        nn.init.ones_(self.ln2_g)

    # ------------------------------------------------------------------
    def _rmsnorm(self, z: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        ms = z.pow(2).mean(-1, keepdim=True)
        return z * torch.rsqrt(ms + self.cfg.norm_eps) * gamma

    @property
    def output_weight(self) -> torch.Tensor:
        """The (d, V) readout matrix (tied to tok_emb when sharing)."""
        if self.cfg.tie_embeddings:
            return self.tok_emb.t()
        return self.W_out

    # ------------------------------------------------------------------
    @torch.no_grad()
    def embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Map token ids to the input "clamped voltage" ``X`` (B,T,d)."""
        T = token_ids.size(1)
        emb = F.embedding(token_ids, self.tok_emb)
        return emb + self._pos_cache[:T].to(emb.dtype).to(emb.device)

    # ------------------------------------------------------------------
    # The recurrent map f(Z) and the local activations it produces.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def forward_map(
        self, Z: torch.Tensor, X: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, d = Z.shape
        H, dh = self.cfg.n_heads, self.cfg.d_head

        # ---- attention (pre-norm) -------------------------------------
        Zr = self._rmsnorm(Z, self.ln1_g) if self.cfg.use_norm else Z
        Q = torch.einsum("btd,hde->bhte", Zr, self.Wq)
        K = torch.einsum("btd,hde->bhte", Zr, self.Wk)
        V = torch.einsum("btd,hde->bhte", Zr, self.Wv)

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(dh)  # (B,H,T,T)
        causal = torch.triu(
            torch.full((T, T), float("-inf"), device=Z.device, dtype=scores.dtype),
            diagonal=1,
        )
        scores = scores + causal
        A = torch.softmax(scores, dim=-1)
        attn = A @ V  # (B,H,T,dh)
        attn_out = torch.einsum("bhte,hed->btd", attn, self.Wo)  # (B,T,d)

        # ---- FFN (pre-norm) -------------------------------------------
        pre1 = self._rmsnorm(Z, self.ln2_g) if self.cfg.use_norm else Z
        pre_act = pre1 @ self.W1 + self.b1
        h_act = F.relu(pre_act)
        ffn = h_act @ self.W2 + self.b2

        g = self.cfg.res_gain
        target = X + g * (attn_out + ffn)

        acts = {
            "Wq": (Zr, Q),
            "Wk": (Zr, K),
            "Wv": (Zr, V),
            "Wo": (attn, attn_out),
            "W1": (pre1, pre_act),
            "W2": (h_act, ffn),
        }
        return target, acts

    # ------------------------------------------------------------------
    @torch.no_grad()
    def energy(self, Z: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        target, _ = self.forward_map(Z, X)
        return 0.5 * (Z - target).pow(2).sum(dim=(1, 2)).mean()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def step_free(
        self, Z: torch.Tensor, X: torch.Tensor, dt: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dt = self.cfg.dt if dt is None else dt
        target, _ = self.forward_map(Z, X)
        newZ = Z + dt * (target - Z)
        residual = (target - Z).detach()
        return newZ, residual

    # ------------------------------------------------------------------
    @torch.no_grad()
    def step_clamped(
        self,
        Z: torch.Tensor,
        X: torch.Tensor,
        targets: torch.Tensor,
        beta: float,
        dt: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dt = self.cfg.dt if dt is None else dt
        target, _ = self.forward_map(Z, X)
        logits = Z @ self.W_out + self.b_out
        probs = torch.softmax(logits, dim=-1)
        err = probs - F.one_hot(targets, self.cfg.vocab_size).float()
        gradC = err @ self.W_out.T  # local: error x output weights
        newZ = Z + dt * (target - Z) - dt * beta * gradC
        residual = (target - Z).detach()
        cost = F.cross_entropy(
            logits.reshape(-1, self.cfg.vocab_size), targets.reshape(-1)
        )
        return newZ, residual, cost

    # ------------------------------------------------------------------
    @torch.no_grad()
    def relax(
        self,
        X: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        beta: float = 0.0,
        steps: Optional[int] = None,
        collect_acts: bool = False,
        init: Optional[torch.Tensor] = None,
        anderson: bool = False,
        anderson_m: int = 5,
        anderson_beta: float = 0.8,
    ) -> Dict:
        clamped = beta > 0 and targets is not None
        steps = steps if steps is not None else (
            self.cfg.clamped_steps if clamped else self.cfg.free_steps
        )

        Z = X.clone() if init is None else init
        Z = 0.5 * X + 0.5 * Z  # light damping toward the clamped input

        residual = None
        cost = None
        collected: List[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = []
        avg = max(1, self.cfg.act_avg)
        dt = self.cfg.dt
        diverged = False

        from .acceleration import AndersonAccel
        aa = AndersonAccel(anderson_m, anderson_beta) if anderson else None

        for t in range(steps):
            if clamped:
                target, _, cost = self._forward_target_clamped(Z, X, targets, beta)
            else:
                target, _ = self.forward_map(Z, X)
            residual = (target - Z).detach()
            # divergence guard
            if not torch.isfinite(target).all() or Z.abs().max().item() > 50.0:
                dt = max(dt * 0.5, 1e-2)
                Z = 0.5 * Z + 0.5 * X
                diverged = True
                if aa is not None:
                    aa.reset()
                continue
            if aa is not None:
                Z = aa.step(Z, target)
            else:
                Z = Z + dt * (target - Z)
            if collect_acts and t >= steps - avg:
                _, acts = self.forward_map(Z, X)
                collected.append(acts)

        out: Dict = {"Z": Z, "residual": residual.norm().item() / max(1, Z.numel()),
                     "dt_used": dt, "diverged": diverged, "anderson": anderson}
        if cost is not None:
            out["cost"] = cost
        if collect_acts and collected:
            out["acts"] = self._average_acts(collected)
        return out

    # ------------------------------------------------------------------
    # Helper: compute the clamped target f(Z) and the clamped cost together.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _forward_target_clamped(self, Z, X, targets, beta):
        target, _ = self.forward_map(Z, X)
        logits = Z @ self.W_out + self.b_out
        probs = torch.softmax(logits, dim=-1)
        err = probs - F.one_hot(targets, self.cfg.vocab_size).float()
        gradC = err @ self.W_out.T
        # the clamped dynamics' "target" is the free target minus the local
        # output-error injection scaled by beta; the cost is reported for
        # monitoring only.
        clamped_target = target - beta * gradC
        cost = F.cross_entropy(
            logits.reshape(-1, self.cfg.vocab_size), targets.reshape(-1)
        )
        return clamped_target, None, cost.item()

    # ------------------------------------------------------------------
    @staticmethod
    def _average_acts(
        collected: List[Dict[str, Tuple[torch.Tensor, torch.Tensor]]]
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        keys = collected[0].keys()
        avg: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        for k in keys:
            pres = torch.stack([c[k][0] for c in collected]).mean(0)
            posts = torch.stack([c[k][1] for c in collected]).mean(0)
            avg[k] = (pres, posts)
        return avg

    # ------------------------------------------------------------------
    @torch.no_grad()
    def logits_from_state(self, Z: torch.Tensor) -> torch.Tensor:
        return Z @ self.output_weight + self.b_out

    # ------------------------------------------------------------------
    # Differentiable single-block forward (autograd ENABLED).  Used only by
    # the DEQ implicit-gradient trainer to obtain vector-Jacobian products at
    # the equilibrium.  This is a single block -- it does NOT backpropagate
    # through the relaxation iterations or any "depth".
    # ------------------------------------------------------------------
    def forward_diff(self, Z: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        T = Z.size(1)
        H, dh = self.cfg.n_heads, self.cfg.d_head
        Zr = self._rmsnorm(Z, self.ln1_g) if self.cfg.use_norm else Z
        Q = torch.einsum("btd,hde->bhte", Zr, self.Wq)
        K = torch.einsum("btd,hde->bhte", Zr, self.Wk)
        V = torch.einsum("btd,hde->bhte", Zr, self.Wv)
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(dh)
        causal = torch.triu(
            torch.full((T, T), float("-inf"), device=Z.device, dtype=Z.dtype),
            diagonal=1,
        )
        scores = scores + causal
        A = torch.softmax(scores, dim=-1)
        attn = A @ V
        attn_out = torch.einsum("bhte,hed->btd", attn, self.Wo)
        n2 = self._rmsnorm(Z, self.ln2_g) if self.cfg.use_norm else Z
        h_act = F.relu(n2 @ self.W1 + self.b1)
        ffn = h_act @ self.W2 + self.b2
        g = self.cfg.res_gain
        return X + g * (attn_out + ffn)
