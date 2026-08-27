"""Non-contractive (Newton-Krylov) DEQ trainer.

This escapes the strict-contraction effective-depth ceiling of §4.9.  The
implicit-gradient machinery only needs the fixed point to exist and
``1 ∉ eigenvalues(J_f)``; it does **not** need ``ρ(J_f) < 1``.  What needed
``ρ < 1`` was only the Picard/Anderson *forward* solver.

We therefore replace the forward solve with **Newton-Krylov**: each Newton step
solves ``(J_f − I) δ = −(f(Z) − Z)`` by GMRES, where the Jacobian-vector product
``J_f · w`` is obtained exactly via ``torch.func.jvp`` (forward-mode AD on a
single block).  Newton converges to the fixed point even when ``ρ(J_f) ≥ 1``.

The adjoint is unchanged in form — GMRES on ``(I − J_f^T) v = g`` via VJP — and
is a *linear solve* that is valid for any non-singular ``(I − J_f)`` regardless
of ``ρ``.  We drop the contraction homeostasis; a loose per-matrix spectral cap
keeps things finite without forcing ``ρ < 1``.  This lets the residual deep map
of §4.8 (which is only expressive for ``ρ ≳ 1``) actually learn.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.func import jvp

from .deep_model import DeepERB
from .deq_trainer import DEQTrainer
from .ep_trainer import _spectral_norm


@dataclass
class NewtonDEQConfig:
    lr: float = 1.5e-3
    lr_out: float = 4e-3
    lr_emb: float = 1.5e-3
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 1e-4
    gmres_k: int = 10           # Krylov dim for both Newton-forward and adjoint
    newton_steps: int = 4       # Newton iterations for the forward solve
    newton_alpha: float = 0.6   # damping for the Newton step
    matrix_cap: float = 3.0     # loose per-matrix spectral cap (NOT contraction)
    total_steps: int = 0
    warmup: int = 300
    device: str = "cuda"


class NewtonDEQTrainer:
    def __init__(self, model: DeepERB, cfg: NewtonDEQConfig):
        self.model = model
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.m: Dict[str, torch.Tensor] = {}
        self.v: Dict[str, torch.Tensor] = {}
        self.step = 0
        self._gmres = DEQTrainer._gmres

    def _adam(self, name, p, g, lr):
        if self.cfg.weight_decay > 0:
            g = g + self.cfg.weight_decay * p.data
        m = self.m.get(name); v = self.v.get(name)
        if m is None:
            m = torch.zeros_like(p); v = torch.zeros_like(p)
        m.mul_(self.cfg.beta1).add_(g, alpha=1 - self.cfg.beta1)
        v.mul_(self.cfg.beta2).add_(g * g, alpha=1 - self.cfg.beta2)
        self.m[name] = m; self.v[name] = v
        t = self.step + 1
        mhat = m / (1 - self.cfg.beta1 ** t)
        vhat = v / (1 - self.cfg.beta2 ** t)
        p.data.addcdiv_(mhat, vhat.sqrt().add_(self.cfg.eps), value=-lr)

    def _sched(self, base_lr):
        cfg = self.cfg; s = self.step
        if cfg.total_steps <= 0:
            return base_lr
        if s < cfg.warmup:
            return base_lr * (s + 1) / max(1, cfg.warmup)
        prog = (s - cfg.warmup) / max(1, cfg.total_steps - cfg.warmup)
        prog = min(max(prog, 0.0), 1.0)
        return base_lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * prog)))

    # ------------------------------------------------------------------
    # Newton-Krylov forward fixed-point solve (no contraction required).
    # Runs under no_grad: we only need the fixed point itself; the gradient
    # is computed separately at the equilibrium.  jvp is forward-mode AD and
    # works under no_grad without building a backward graph.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _solve_forward(self, X):
        model = self.model; cfg = self.cfg
        Xd = X.detach()
        f_only = lambda z: model.forward_diff(z, Xd)
        Z = Xd.clone()
        res = 0.0
        for _ in range(cfg.newton_steps):
            target = f_only(Z)
            gvec = target - Z
            res = gvec.detach().norm().item() / max(1, Z.numel())
            if res < 1e-4 or not torch.isfinite(gvec).all():
                break
            def Av(w, _Z=Z):
                ws = w.view_as(_Z)
                _, jv = jvp(f_only, (_Z,), (ws,))
                return (jv - ws).reshape(-1)
            try:
                delta = self._gmres(Av, (-gvec).reshape(-1), cfg.gmres_k).view_as(Z)
            except Exception:
                break
            if not torch.isfinite(delta).all():
                break
            Z = (Z + cfg.newton_alpha * delta).detach()
        return Z, res

    # ------------------------------------------------------------------
    def update(self, token_ids, targets):
        model = self.model; cfg = self.cfg
        V = model.cfg.vocab_size
        X = (model.tok_emb[token_ids]
             + model._pos_cache[:token_ids.size(1)].to(model.tok_emb.device).to(model.tok_emb.dtype))

        Z_star, res_free = self._solve_forward(X)
        if not torch.isfinite(Z_star).all() or Z_star.abs().max().item() > 80.0:
            self._cap_matrices()
            self.step += 1
            return {"loss": float("nan"), "res_free": res_free, "skipped": 1}
        Z_star = Z_star.detach()

        # ---- adjoint: GMRES on (I - J_f^T) v = g_readout ----
        Z_star.requires_grad_(True)
        target = model.forward_diff(Z_star, X.detach())
        logits = Z_star @ model.output_weight + model.b_out
        loss = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1))
        g, = torch.autograd.grad(loss, Z_star, retain_graph=True)

        def AvA(w):
            ws = w.view_as(Z_star)
            JTw, = torch.autograd.grad(target, Z_star, grad_outputs=ws, retain_graph=True)
            return (ws - JTw.detach()).reshape(-1)

        adj = self._gmres(AvA, g.detach().reshape(-1), cfg.gmres_k).view_as(Z_star)

        rec = model.recurrent_params
        gparams = torch.autograd.grad(target, rec, grad_outputs=adj, retain_graph=True)

        # embedding adjoint
        X2 = X.detach().requires_grad_(True)
        target2 = model.forward_diff(Z_star.detach(), X2)
        g_xin, = torch.autograd.grad(target2, X2, grad_outputs=adj.detach(), retain_graph=False)

        # readout grads
        p = torch.softmax(logits.detach(), dim=-1)
        err = p - F.one_hot(targets, V).float()
        g_OW = torch.einsum("btd,btV->dV", Z_star.detach(), err) / Z_star.size(0)
        g_bout = err.mean(dim=(0, 1))

        agg = torch.zeros_like(model.tok_emb)
        flat_ids = token_ids.reshape(-1)
        agg.index_add_(0, flat_ids, g_xin.reshape(-1, model.tok_emb.size(1)))
        counts = torch.zeros(model.tok_emb.size(0), device=model.tok_emb.device)
        counts.index_add_(0, flat_ids, torch.ones_like(flat_ids, dtype=model.tok_emb.dtype))
        counts = counts.clamp(min=1.0).unsqueeze(1)
        g_tokemb = agg / counts
        if model.cfg.tie_embeddings:
            g_tokemb = g_tokemb + g_OW.t()

        lr = self._sched(cfg.lr); lr_out = self._sched(cfg.lr_out); lr_emb = self._sched(cfg.lr_emb)
        names = []
        for li in range(len(model.layers)):
            names += [f"l{li}.Wq", f"l{li}.Wk", f"l{li}.Wv", f"l{li}.Wo",
                      f"l{li}.W1", f"l{li}.b1", f"l{li}.W2", f"l{li}.b2"]
        for name, p_, g_ in zip(names, rec, gparams):
            self._adam(name, p_, g_, lr)
        self._adam("b_out", model.b_out, g_bout, lr_out)
        self._adam("tok_emb", model.tok_emb, g_tokemb, lr_emb)
        if not model.cfg.tie_embeddings:
            self._adam("W_out", model.W_out, g_OW, lr_out)

        self._cap_matrices()
        self.step += 1
        return {"loss": loss.item(), "res_free": res_free, "skipped": 0}

    # ------------------------------------------------------------------
    # Loose per-matrix spectral cap: keeps weights finite but does NOT
    # force any product to be < 1, so ρ(J_f) may exceed 1.
    # ------------------------------------------------------------------
    def _cap_matrices(self):
        cap = self.cfg.matrix_cap
        for layer in self.model.layers:
            for W in (layer.Wq, layer.Wk, layer.Wv, layer.Wo, layer.W1, layer.W2):
                s = _spectral_norm(W.data)
                if s > cap and s > 0:
                    W.data.mul_(cap / s)
