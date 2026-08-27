"""DEQ implicit-gradient trainer for EnergyLM.

This addresses the central weakness of plain Equilibrium Propagation: the
Hebbian *correlation difference* is only a rough estimate of the true cost
gradient.  Here we instead use the **exact implicit-function (DEQ) gradient**
of the cost through the equilibrium, with **no backpropagation through the
relaxation iterations**.

At the equilibrium ``Z* = f(Z*, X; theta)`` (found by Anderson-accelerated
relaxation, under no_grad), the implicit-function theorem gives

    dC/dtheta = (dC/dZ*) (I - J_f)^{-1} (df/dtheta)

where ``J_f = df/dZ`` evaluated at ``Z*``.  We never form or invert ``J_f``.
Instead we:

  1. compute the readout adjoint ``g = dC/dZ*`` (local, one layer);
  2. approximate ``(I - J_f^T)^{-1} g`` by a **Neumann series**
     ``v = sum_{i=0}^{K} (J_f^T)^i g`` -- each term is a single vector-Jacobian
     product of *one* block ``f`` (computed with autograd at the equilibrium,
     NOT through the iterations);
  3. read off ``dC/dtheta = adj^T (df/dtheta)`` as VJPs of the same single
     block.

The homeostasis that keeps ``J_f`` contractive (spectral radius < 1) is exactly
what guarantees the Neumann series converges.  Readout & embedding use their
exact local gradients.

Caveat: this relaxes the strict "zero autograd" rule -- we use single-block
vector-Jacobian products.  But there is still **no backward pass through the
iterations or any depth**: the gradient is computed entirely *at the
equilibrium*, which is the DEQ claim and the spirit of "no backprop".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List
import math

import torch
import torch.nn.functional as F

from .energy_model import EnergyRecurrentBlock
from .ep_trainer import EPTrainer, _spectral_norm


@dataclass
class DEQConfig:
    lr: float = 1e-3
    lr_out: float = 1e-3
    lr_emb: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 1e-4
    neumann_k: int = 5          # Neumann-series terms for (I-J^T)^{-1}
    adjoint: str = "neumann"    # "neumann" (truncated series) or "gmres" (Krylov)
    gmres_k: int = 8            # GMRES Krylov dimension (matvecs)
    contractivity: float = 0.6  # homeostasis target
    homeostasis: bool = True
    anderson: bool = True
    anderson_m: int = 5
    anderson_beta: float = 0.7
    free_steps: int = 20
    total_steps: int = 0          # for LR cosine schedule (0 = constant lr)
    warmup: int = 200             # warmup steps before cosine decay
    device: str = "cuda"


class DEQTrainer:
    """Adam-optimised DEQ implicit-gradient trainer."""

    def __init__(self, model: EnergyRecurrentBlock, cfg: DEQConfig):
        self.model = model
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        # Adam moment buffers keyed by param name
        self.m: Dict[str, torch.Tensor] = {}
        self.v: Dict[str, torch.Tensor] = {}
        self.step = 0
        self._params = {
            "Wq": model.Wq, "Wk": model.Wk, "Wv": model.Wv, "Wo": model.Wo,
            "W1": model.W1, "W2": model.W2,
            "b1": model.b1, "b2": model.b2, "b_out": model.b_out,
            "tok_emb": model.tok_emb,
        }
        if not model.cfg.tie_embeddings:
            self._params["W_out"] = model.W_out

    # ------------------------------------------------------------------
    def _adam(self, name: str, p: torch.Tensor, g: torch.Tensor, lr: float):
        if self.cfg.weight_decay > 0:
            g = g + self.cfg.weight_decay * p.data
        m = self.m.get(name)
        v = self.v.get(name)
        if m is None:
            m = torch.zeros_like(p); v = torch.zeros_like(p)
        m.mul_(self.cfg.beta1).add_(g, alpha=1 - self.cfg.beta1)
        v.mul_(self.cfg.beta2).add_(g * g, alpha=1 - self.cfg.beta2)
        self.m[name] = m; self.v[name] = v
        t = self.step + 1
        mhat = m / (1 - self.cfg.beta1 ** t)
        vhat = v / (1 - self.cfg.beta2 ** t)
        p.data.addcdiv_(mhat, vhat.sqrt().add_(self.cfg.eps), value=-lr)

    # ------------------------------------------------------------------
    def update(self, token_ids: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        model = self.model
        cfg = self.cfg
        V = model.cfg.vocab_size

        # ---- 1. embed (with grad for the embedding gradient) ----------
        X = model.tok_emb[token_ids] + model._pos_cache[:token_ids.size(1)].to(model.tok_emb.dtype)

        # ---- 2. relax to equilibrium Z* (no grad) ---------------------
        with torch.no_grad():
            out = model.relax(
                X.detach(), beta=0.0, steps=cfg.free_steps,
                anderson=cfg.anderson, anderson_m=cfg.anderson_m,
                anderson_beta=cfg.anderson_beta,
            )
            res_free = out["residual"]
            Z_star = out["Z"].detach()
        # skip if relaxation diverged
        if not torch.isfinite(Z_star).all() or Z_star.abs().max().item() > 80.0:
            if cfg.homeostasis:
                self._enforce_contractivity()
            self.step += 1
            return {"loss": float("nan"), "res_free": res_free, "skipped": 1}

        # ---- 3. differentiable block forward at Z* --------------------
        Z_star.requires_grad_(True)
        X.requires_grad_(True)
        target = model.forward_diff(Z_star, X)           # f(Z*, X; theta)

        # ---- 4. readout loss & adjoint --------------------------------
        logits = Z_star @ model.output_weight + model.b_out
        loss = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1))
        # g = dC/dZ*  (readout gradient, local)
        g, = torch.autograd.grad(loss, Z_star, retain_graph=True, create_graph=False)

        # ---- 5. solve (I - J^T) v = g  for the adjoint v -------------
        # Neumann:  v = sum_{i=0}^{K} (J^T)^i g
        # GMRES:    optimal Krylov approximation using the same VJP oracle.
        # A · w = w - J^T w.  GMRES is more accurate per matvec and stays
        # accurate closer to the contractivity boundary (||J|| -> 1), where the
        # Neumann geometric series becomes slow.
        g_d = g.detach()
        if cfg.adjoint == "gmres":
            def Av(w_flat):
                w = w_flat.view_as(Z_star)
                JTw, = torch.autograd.grad(
                    target, Z_star, grad_outputs=w, retain_graph=True, create_graph=False
                )
                return (w - JTw.detach()).reshape_as(w_flat)
            adj = self._gmres(Av, g_d.reshape(-1), cfg.gmres_k).view_as(Z_star)
        else:
            adj = g_d.clone()
            vk = g_d.clone()
            for _ in range(cfg.neumann_k):
                JTv, = torch.autograd.grad(
                    target, Z_star, grad_outputs=vk, retain_graph=True, create_graph=False
                )
                vk = JTv.detach()
                adj = adj + vk

        # ---- 6. weight gradients = VJP of f w.r.t. theta at adj -------
        recurrent = [model.Wq, model.Wk, model.Wv, model.Wo, model.W1, model.W2]
        gparams = torch.autograd.grad(
            target, recurrent + [X], grad_outputs=adj,
            retain_graph=True, create_graph=False,
        )
        g_emb = gparams[-1]                              # (B,T,d) adjoint to X

        # ---- 7. readout gradients (local) -----------------------------
        # dC/d(output_weight) where output_weight is (d,V):  einsum Z* err -> (d,V)
        p = torch.softmax(logits.detach(), dim=-1)
        err = p - F.one_hot(targets, V).float()
        g_OW = torch.einsum("btd,btV->dV", Z_star.detach(), err) / Z_star.size(0)
        g_bout = err.mean(dim=(0, 1))

        # ---- 8. embedding gradient via scatter of g_emb ---------------
        agg = torch.zeros_like(model.tok_emb)
        flat_ids = token_ids.reshape(-1)
        agg.index_add_(0, flat_ids, g_emb.reshape(-1, model.tok_emb.size(1)))
        counts = torch.zeros(model.tok_emb.size(0), device=model.tok_emb.device)
        counts.index_add_(0, flat_ids, torch.ones_like(flat_ids, dtype=model.tok_emb.dtype))
        counts = counts.clamp(min=1.0).unsqueeze(1)
        g_tokemb = agg / counts
        # when embeddings are tied, the readout also contributes to tok_emb
        if model.cfg.tie_embeddings:
            g_tokemb = g_tokemb + g_OW.t()               # (V,d) readout contribution

        # ---- 9. scheduled LR & Adam -----------------------------------
        lr = self._sched(cfg.lr)
        lr_out = self._sched(cfg.lr_out)
        lr_emb = self._sched(cfg.lr_emb)
        updates = [
            ("Wq", model.Wq, gparams[0], lr),
            ("Wk", model.Wk, gparams[1], lr),
            ("Wv", model.Wv, gparams[2], lr),
            ("Wo", model.Wo, gparams[3], lr),
            ("W1", model.W1, gparams[4], lr),
            ("W2", model.W2, gparams[5], lr),
            ("b1", model.b1, torch.zeros_like(model.b1), lr),
            ("b2", model.b2, torch.zeros_like(model.b2), lr),
            ("b_out", model.b_out, g_bout, lr_out),
            ("tok_emb", model.tok_emb, g_tokemb, lr_emb),
        ]
        if not model.cfg.tie_embeddings:
            updates.append(("W_out", model.W_out, g_OW, lr_out))
        for name, p, g, lrv in updates:
            self._adam(name, p, g, lrv)

        # ---- 10. homeostasis to keep the map contractive -------------
        if cfg.homeostasis:
            self._enforce_contractivity()

        self.step += 1
        return {"loss": loss.item(), "res_free": res_free, "skipped": 0}

    # ------------------------------------------------------------------
    # Right-preconditioning-free GMRES(k) for solving A x = b on flat
    # vectors, given a matvec oracle ``Av``.  Minimises ||b - A x|| over the
    # k-dimensional Krylov subspace K_k(A, b) via Arnoldi + a small
    # least-squares solve.  Operates per-batch-row by flattening batch into
    # the vector dimension (so all batch elements share one Krylov basis,
    # which is cheap and works well in practice).
    # ------------------------------------------------------------------
    @staticmethod
    def _gmres(Av, b: torch.Tensor, k: int, tol: float = 1e-4) -> torch.Tensor:
        beta = b.norm()
        if beta.item() < 1e-12:
            return torch.zeros_like(b)
        Q = [b / beta]
        H = torch.zeros(k + 1, k, device=b.device, dtype=b.dtype)
        j = 0
        for j in range(k):
            w = Av(Q[j])
            for i in range(j + 1):                # Arnoldi (modified Gram-Schmidt)
                H[i, j] = torch.dot(Q[i].reshape(-1), w.reshape(-1))
                w = w - H[i, j] * Q[i]
            hn = w.norm()
            H[j + 1, j] = hn
            if hn.item() < tol:
                break
            Q.append(w / hn)
        m = j + 1                                  # actual Krylov dimension used
        e1 = torch.zeros(m + 1, device=b.device, dtype=b.dtype)
        e1[0] = beta
        # least-squares: minimise || e1 - H[:m+1, :m] y ||
        y, _, _, _ = torch.linalg.lstsq(H[: m + 1, : m], e1.unsqueeze(-1))
        y = y.squeeze(-1)
        Qm = torch.stack(Q[:m])                    # (m, D)
        return (y.unsqueeze(1) * Qm).sum(0)

    # ------------------------------------------------------------------
    # Linear warmup then cosine decay to 10% of base lr.
    # ------------------------------------------------------------------
    def _sched(self, base_lr: float) -> float:
        cfg = self.cfg
        s = self.step
        if cfg.total_steps <= 0:
            return base_lr
        if s < cfg.warmup:
            return base_lr * (s + 1) / max(1, cfg.warmup)
        prog = (s - cfg.warmup) / max(1, cfg.total_steps - cfg.warmup)
        prog = min(max(prog, 0.0), 1.0)
        return base_lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * prog)))

    # ------------------------------------------------------------------
    def _enforce_contractivity(self):
        import math
        m = self.model
        g = m.cfg.res_gain
        budget = self.cfg.contractivity / max(g, 1e-3)
        for Wa, Wb in [(m.W1, m.W2), (m.Wv, m.Wo)]:
            sa = _spectral_norm(Wa.data); sb = _spectral_norm(Wb.data)
            prod = sa * sb
            if prod > budget and prod > 0:
                scale = (budget / prod) ** 0.5
                Wa.data.mul_(scale); Wb.data.mul_(scale)
        for W in (m.Wq, m.Wk):
            s = _spectral_norm(W.data)
            cap = math.sqrt(budget)
            if s > cap and s > 0:
                W.data.mul_(cap / s)
