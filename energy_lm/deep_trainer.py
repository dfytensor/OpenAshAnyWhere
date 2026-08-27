"""DEQ trainer for the deep-map single-equilibrium model.

Identical gradient machinery to the single-block ``DEQTrainer`` (one GMRES
adjoint on the full composed Jacobian), but the recurrent map is the deep
composition ``U`` and contractivity is enforced per sub-layer so that
``g * Π_l L(s_l) < contractivity``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn.functional as F

from .deep_model import DeepERB
from .deq_trainer import DEQTrainer


@dataclass
class DeepDEQConfig:
    lr: float = 1.5e-3
    lr_out: float = 4e-3
    lr_emb: float = 1.5e-3
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 1e-4
    gmres_k: int = 10
    contractivity: float = 0.80
    free_steps: int = 28
    anderson: bool = True
    anderson_m: int = 5
    anderson_beta: float = 0.7
    total_steps: int = 0
    warmup: int = 300
    device: str = "cuda"


class DeepDEQTrainer:
    def __init__(self, model: DeepERB, cfg: DeepDEQConfig):
        self.model = model
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.m: Dict[str, torch.Tensor] = {}
        self.v: Dict[str, torch.Tensor] = {}
        self.step = 0
        # reuse the proven GMRES from DEQTrainer (staticmethod)
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

    def update(self, token_ids, targets):
        model = self.model; cfg = self.cfg
        V = model.cfg.vocab_size
        X = model.tok_emb[token_ids] + model._pos_cache[:token_ids.size(1)].to(model.tok_emb.device).to(model.tok_emb.dtype)

        with torch.no_grad():
            out = model.relax(X.detach(), steps=cfg.free_steps, anderson=cfg.anderson,
                              anderson_m=cfg.anderson_m, anderson_beta=cfg.anderson_beta)
            res_free = out["residual"]
            if out["diverged"] or not torch.isfinite(out["Z"]).all() or out["Z"].abs().max().item() > 80.0:
                self._enforce_contractivity()
                self.step += 1
                return {"loss": float("nan"), "res_free": res_free, "skipped": 1}
            Z_star = out["Z"].detach()

        Z_star.requires_grad_(True)
        target = model.forward_diff(Z_star, X.detach())     # f(Z*, X)

        logits = Z_star @ model.output_weight + model.b_out
        loss = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1))
        g, = torch.autograd.grad(loss, Z_star, retain_graph=True)

        # one GMRES solve on the FULL composed Jacobian: (I - J^T) v = g
        def Av(w):
            ws = w.view_as(Z_star)
            JTw, = torch.autograd.grad(target, Z_star, grad_outputs=ws, retain_graph=True)
            return (ws - JTw.detach()).reshape(-1)

        adj = self._gmres(Av, g.detach().reshape(-1), cfg.gmres_k).view_as(Z_star)

        rec = model.recurrent_params
        gparams = torch.autograd.grad(target, rec, grad_outputs=adj, retain_graph=True)

        # readout grads
        p = torch.softmax(logits.detach(), dim=-1)
        err = p - F.one_hot(targets, V).float()
        g_OW = torch.einsum("btd,btV->dV", Z_star.detach(), err) / Z_star.size(0)
        g_bout = err.mean(dim=(0, 1))

        # embedding grad via input adjoint (recompute X with grad enabled)
        X2 = (model.tok_emb[token_ids]
              + model._pos_cache[:token_ids.size(1)].to(model.tok_emb.device).to(model.tok_emb.dtype)
              ).detach().requires_grad_(True)
        target2 = model.forward_diff(Z_star.detach(), X2)
        g_xin, = torch.autograd.grad(target2, X2, grad_outputs=adj.detach(), retain_graph=False)
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

        self._enforce_contractivity()
        self.step += 1
        return {"loss": loss.item(), "res_free": res_free, "skipped": 0}

    def _enforce_contractivity(self):
        # residual map: need g * prod_l (1 + lip_l) < contractivity.
        # give each layer an equal share: (1 + lip) <= (contractivity/g)^(1/L)
        g = self.model.cfg.res_gain
        ratio = self.cfg.contractivity / max(g, 1e-3)
        per_layer_lip = max(ratio ** (1.0 / len(self.model.layers)) - 1.0, 1e-3)
        for layer in self.model.layers:
            layer.scale_to(per_layer_lip)
