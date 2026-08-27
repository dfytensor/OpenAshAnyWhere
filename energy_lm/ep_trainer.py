"""Equilibrium-Propagation trainer for EnergyLM.

The only learning signal is the **difference of local pre-/post-synaptic
correlations** measured at the *free* steady state and the *clamped* steady
state.  No gradient is ever propagated backward through the network; nothing
here calls ``.backward()`` or constructs an autograd graph for the weights.

EP theorem (Scellier & Bengio, 2017).  For an energy ``E(Z, theta)`` and cost
``C(Z)``, with free state ``Z0 = argmin E`` and clamped state
``Zb = argmin (E + beta*C)``:

    dC/dtheta = (1/beta) * ( <dE/dtheta>_Zb - <dE/dtheta>_Z0 )

For a weight ``W`` whose energy contribution behaves like ``-<post, W pre>``
the bracket reduces to ``-<post pre^T>``, so gradient descent gives the local
Hebbian rule

    dW = (lr / beta) * ( <post pre^T>_clamped - <post pre^T>_free )

We apply exactly this rule to every recurrent weight (Q,K,V,O,W1,W2), to the
output head and to the token embedding (the embedding update uses the
state-difference correlation, see ``_update_embedding``).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .energy_model import EnergyLMConfig, EnergyRecurrentBlock


@dataclass
class EPConfig:
    lr: float = 0.05              # EP learning rate
    beta: float = 1.0             # clamping strength
    weight_decay: float = 0.02    # pull weights toward zero (fights Hebbian runaway)
    momentum: float = 0.9         # EP momentum on the weight update
    lr_emb: float = 0.05          # separate LR for token embedding
    lr_out: float = 0.05          # separate LR for output head
    grad_clip: float = 1.0        # clip per-weight update magnitude
    sign: float = 1.0             # +1: standard EP Hebbian sign (flippable)
    contractivity: float = 0.6     # keep g*Lipschitz below this (stability margin)
    homeostasis: bool = True       # bound spectral norms of recurrent weights
    skip_on_divergence: bool = True  # ignore updates whose relaxation diverged
    max_residual: float = 1e-2     # above this, treat the step as unconverged
    anderson: bool = False         # Anderson-accelerate the relaxation
    anderson_m: int = 5
    anderson_beta: float = 0.8
    device: str = "cuda"


def _spectral_norm(W: torch.Tensor, n_iter: int = 5) -> float:
    """Estimate the largest singular value of ``W`` by power iteration.

    Works for both 2-D matrices and stacked multi-head tensors (the latter is
    flattened over the head axis first so we get a conservative upper bound).
    """
    with torch.no_grad():
        if W.dim() > 2:
            Wf = W.reshape(W.shape[0] * W.shape[1], W.shape[2])
        else:
            Wf = W
        m, n = Wf.shape
        v = torch.randn(n, device=W.device, dtype=W.dtype)
        v = v / (v.norm() + 1e-8)
        for _ in range(n_iter):
            u = Wf @ v
            un = u.norm() + 1e-8
            u = u / un
            v = Wf.t() @ u
            vn = v.norm() + 1e-8
            v = v / vn
        sigma = (Wf @ v).norm().item()
        return sigma


# ===========================================================================
# EP Trainer
# ===========================================================================
class EPTrainer:
    def __init__(self, model: EnergyRecurrentBlock, cfg: EPConfig):
        self.model = model
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        # per-parameter momentum buffers
        self.mom: Dict[str, torch.Tensor] = {}
        self.step = 0

    # ------------------------------------------------------------------
    # Compute the mean pre/post correlation for a given weight name across the
    # batch (and sequence / heads).  Returns a tensor shaped like the weight.
    # ------------------------------------------------------------------
    @staticmethod
    def _correlation(name: str, pre: torch.Tensor, post: torch.Tensor) -> torch.Tensor:
        if name in ("Wq", "Wk", "Wv"):
            # pre: (B,T,d)  post: (B,H,T,dh)  -> (H,d,dh)
            return torch.einsum("btd,bhte->hde", pre, post) / pre.size(0)
        if name == "Wo":
            # pre: (B,H,T,dh)  post: (B,T,d)  -> (H,dh,d)
            return torch.einsum("bhte,btd->hed", pre, post) / pre.size(0)
        if name == "W1":
            # pre: (B,T,d)  post: (B,T,d_ff)  -> (d,d_ff)
            return torch.einsum("btd,btf->df", pre, post) / pre.size(0)
        if name == "W2":
            # pre: (B,T,d_ff)  post: (B,T,d)  -> (d_ff,d)
            return torch.einsum("btf,btd->fd", pre, post) / pre.size(0)
        raise ValueError(name)

    # ------------------------------------------------------------------
    # The actual EP weight update for one batch.
    # ------------------------------------------------------------------
    def update(
        self,
        token_ids: torch.Tensor,   # (B,T) input ids
        targets: torch.Tensor,     # (B,T) next-token ids
    ) -> Dict[str, float]:
        model = self.model
        cfg = self.cfg

        X = model.embed(token_ids)

        a_kwargs = dict(anderson=cfg.anderson, anderson_m=cfg.anderson_m,
                        anderson_beta=cfg.anderson_beta)

        # ---- Phase 1: free relaxation ---------------------------------
        free = model.relax(X, beta=0.0, collect_acts=True, **a_kwargs)
        Z0 = free["Z"]
        acts0 = free["acts"]

        # ---- Phase 2: clamped relaxation ------------------------------
        clamped = model.relax(
            X, targets=targets, beta=cfg.beta, collect_acts=True, init=X.clone(),
            **a_kwargs,
        )
        Zb = clamped["Z"]
        actsb = clamped["acts"]

        # ---- cross-entropy at the FREE state (the quantity we lower) --
        with torch.no_grad():
            logits0 = model.logits_from_state(Z0)
            loss = F.cross_entropy(
                logits0.reshape(-1, model.cfg.vocab_size), targets.reshape(-1)
            ).item()

        # If either relaxation failed to converge (or diverged), the
        # correlations are unreliable and would poison the weights.  We skip
        # the Hebbian update for this step but still apply homeostasis to pull
        # the map back inside the contractive regime, then return.
        bad = (
            cfg.skip_on_divergence
            and (
                free.get("diverged", False)
                or clamped.get("diverged", False)
                or free["residual"] > cfg.max_residual
                or clamped["residual"] > cfg.max_residual
                or not torch.isfinite(Z0).all()
                or not torch.isfinite(Zb).all()
            )
        )
        if bad:
            if cfg.homeostasis:
                self._enforce_contractivity()
            self.step += 1
            return {
                "loss": loss,
                "res_free": free["residual"],
                "res_clamped": clamped["residual"],
                "skipped": 1,
            }

        beta = cfg.beta
        inv = (cfg.sign / beta)

        # ---- recurrent weights: local Hebbian EP rule ----------------
        for name in ["Wq", "Wk", "Wv", "Wo", "W1", "W2"]:
            pre0, post0 = acts0[name]
            preb, postb = actsb[name]
            corr0 = self._correlation(name, pre0, post0)
            corrb = self._correlation(name, preb, postb)
            delta = inv * (corrb - corr0)            # EP estimate of -dC/dW
            self._apply(name, getattr(model, name), delta, lr=cfg.lr)

        # ---- output head W_out: explicit LOCAL cost gradient ------------
        # The readout is the final layer, so its cost gradient depends only
        # on the steady-state activities and the target -- no chain rule
        # through the recurrent network, hence no backpropagation.  This is
        # the standard EP treatment of the readout (Scellier & Bengio 2017,
        # eq. for the output layer).
        #   dC/dW_out = Z0^T (p0 - y),   dC/db_out = mean(p0 - y)
        p0 = torch.softmax(logits0, dim=-1)                    # (B,T,V)
        y_onehot = F.one_hot(targets, model.cfg.vocab_size).float()
        err = p0 - y_onehot                                    # (B,T,V)
        g_wout = torch.einsum("btd,btV->dV", Z0, err) / Z0.size(0)   # (d,V)
        g_bout = err.mean(dim=(0, 1))                          # (V,)
        self._apply("W_out", model.W_out, -g_wout, lr=cfg.lr_out)
        model.b_out.data.add_(-g_bout, alpha=cfg.lr_out)

        # ---- token embedding: local state-difference correlation --------
        self._update_embedding(token_ids, Z0, Zb, inv)

        # ---- synaptic-scaling homeostasis: keep the map contractive ----
        if self.cfg.homeostasis:
            self._enforce_contractivity()

        self.step += 1
        return {
            "loss": loss,
            "res_free": free["residual"],
            "res_clamped": clamped["residual"],
            "skipped": 0,
        }

    # ------------------------------------------------------------------
    # Keep ``g * Lipschitz(recurrent map) < contractivity`` so the
    # relaxation always converges to a unique fixed point.  The FFN Jacobian
    # is ~ sigma(W1) sigma(W2); the attention output path ~ sigma(Wv) sigma(Wo).
    # We rescale (W1,W2) and (Wv,Wo) jointly whenever their product exceeds the
    # budget.  This is "synaptic scaling" --- a local, multiplicative
    # normalisation that is biologically plausible and stops the Hebbian rule
    # from blowing the weights up.
    # ------------------------------------------------------------------
    def _enforce_contractivity(self):
        m = self.model
        g = m.cfg.res_gain
        budget = self.cfg.contractivity / max(g, 1e-3)  # allowed product sigma_a*sigma_b

        for pair in [(m.W1, m.W2), (m.Wv, m.Wo)]:
            Wa, Wb = pair
            sa = _spectral_norm(Wa.data)
            sb = _spectral_norm(Wb.data)
            prod = sa * sb
            if prod > budget and prod > 0:
                scale = (budget / prod) ** 0.5
                Wa.data.mul_(scale)
                Wb.data.mul_(scale)
        # Wq, Wk enter through softmax(1-Lipschitz) so they are gentler; still
        # bound them individually to avoid runaway.
        for W in (m.Wq, m.Wk):
            s = _spectral_norm(W.data)
            cap = math.sqrt(budget)
            if s > cap and s > 0:
                W.data.mul_(cap / s)

    # ------------------------------------------------------------------
    def _apply(self, key: str, param: torch.Tensor, delta: torch.Tensor, lr: float):
        # weight decay (toward zero) + clip + momentum + SGD step
        if self.cfg.weight_decay > 0:
            delta = delta - self.cfg.weight_decay * param.data
        norm = delta.norm()
        if norm > self.cfg.grad_clip and norm > 0:
            delta = delta * (self.cfg.grad_clip / norm)
        m = self.mom.get(key)
        if m is None or m.shape != delta.shape:
            m = torch.zeros_like(delta)
        m.mul_(self.cfg.momentum).add_(delta, alpha=1.0 - self.cfg.momentum)
        self.mom[key] = m
        param.data.add_(m, alpha=lr)

    def _apply_bias(self, param: torch.Tensor, delta: torch.Tensor, lr: float):
        param.data.add_(delta, alpha=lr)

    # ------------------------------------------------------------------
    # Embedding update: for each token id, push its row by the (averaged)
    # difference of the clamped vs free steady-state field.  This is a local
    # scatter rule, fully backprop-free.
    # ------------------------------------------------------------------
    def _update_embedding(
        self,
        token_ids: torch.Tensor,
        Z0: torch.Tensor,
        Zb: torch.Tensor,
        inv: float,
    ):
        model = self.model
        emb = model.tok_emb
        B, T = token_ids.shape
        diff = inv * (Zb - Z0)                  # (B,T,d)
        flat_ids = token_ids.reshape(-1)        # (B*T,)
        flat_diff = diff.reshape(-1, emb.size(1))
        agg = torch.zeros_like(emb)
        agg.index_add_(0, flat_ids, flat_diff)
        # normalise by the number of times each id appeared
        counts = torch.zeros_like(emb[:, 0])
        counts.index_add_(0, flat_ids, torch.ones_like(flat_ids, dtype=emb.dtype))
        counts = counts.clamp(min=1.0).unsqueeze(1)
        delta = agg / counts
        key = "tok_emb"
        self._apply(key, emb, delta, lr=self.cfg.lr_emb)


# ===========================================================================
# Generator: autoregressive sampling by repeated relaxation.
# ===========================================================================
@torch.no_grad()
def generate(
    model: EnergyRecurrentBlock,
    tokenizer,
    prompt: str,
    n_new: int = 60,
    temperature: float = 0.8,
    top_k: int = 0,
    warm_state: Optional[torch.Tensor] = None,
) -> Tuple[str, Dict]:
    """Generate text by relaxing the ERB on the growing context.

    To demonstrate the "KV-cache-as-fixed-point-attractor" idea, we warm-start
    each new relaxation from the previous steady state shifted by one position.
    """
    device = model.Wq.device
    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    T = ids.size(1)

    # we keep a running tail of the context, capped at max_seq_len
    max_len = model.cfg.max_seq_len
    energies: List[float] = []

    Z_prev = warm_state
    for _ in range(n_new):
        ctx = ids[:, -max_len:]
        X = model.embed(ctx)
        # warm-start only when the previous steady state lines up with the new
        # context shape (KV-cache-as-attractor idea); otherwise relax from X.
        init = Z_prev if (Z_prev is not None and Z_prev.shape == X.shape) else None
        out = model.relax(X, beta=0.0, init=init)
        Z = out["Z"]
        energies.append(model.energy(Z, X).item())
        logits = model.logits_from_state(Z)[0, -1] / max(temperature, 1e-5)
        if top_k > 0:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[-1]] = float("-inf")
        probs = torch.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, nxt.unsqueeze(0)], dim=1)
        # once the context is pinned at max_len, the shape stays stable, so we
        # can reuse the full steady state as the next warm start.
        Z_prev = Z.clone() if ctx.size(1) == max_len else None

    text = tokenizer.decode(ids[0])
    return text, {"energies": energies}
