"""DEQ trainer for the multi-block hierarchical equilibrium model.

The implicit gradient is propagated **through the chain of equilibria**.  Each
block ``l`` has equilibrium ``h_{l+1} = f_l(h_{l+1}, h_l; theta_l)``.  Given the
readout adjoint ``g = dC/dh_L`` we walk the chain top-down:

    for l = L-1 ... 0:
        adj_l = (I - J_{l,Z}^T)^{-1} g         # GMRES on block l's VJP w.r.t. its state
        grad_theta_l  = VJP(f_l, theta_l, adj_l)
        g             = VJP(f_l, h_l,   adj_l)   # adjoint handed to the previous layer

No gradient flows through relaxation iterations; each layer is an implicit
layer solved at its equilibrium.  Readout & embedding use their exact local
gradients.  Per-block homeostasis keeps every block contractive so all GMRES
solves converge.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn.functional as F

from .multi_model import MultiBlockERB


@dataclass
class MultiDEQConfig:
    lr: float = 1.5e-3
    lr_out: float = 4e-3
    lr_emb: float = 1.5e-3
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 1e-4
    gmres_k: int = 8
    contractivity: float = 0.7      # per-block homeostasis target
    free_steps: int = 22            # relaxation steps per block
    anderson: bool = True
    anderson_m: int = 5
    anderson_beta: float = 0.7
    total_steps: int = 0
    warmup: int = 300
    device: str = "cuda"


class MultiDEQTrainer:
    def __init__(self, model: MultiBlockERB, cfg: MultiDEQConfig):
        self.model = model
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.m: Dict[str, torch.Tensor] = {}
        self.v: Dict[str, torch.Tensor] = {}
        self.step = 0

    # ------------------------------------------------------------------
    def _adam(self, name: str, p: torch.Tensor, g: torch.Tensor, lr: float):
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

    def _sched(self, base_lr: float) -> float:
        cfg = self.cfg; s = self.step
        if cfg.total_steps <= 0:
            return base_lr
        if s < cfg.warmup:
            return base_lr * (s + 1) / max(1, cfg.warmup)
        prog = (s - cfg.warmup) / max(1, cfg.total_steps - cfg.warmup)
        prog = min(max(prog, 0.0), 1.0)
        return base_lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * prog)))

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
            for i in range(j + 1):
                H[i, j] = torch.dot(Q[i].reshape(-1), w.reshape(-1))
                w = w - H[i, j] * Q[i]
            hn = w.norm(); H[j + 1, j] = hn
            if hn.item() < tol:
                break
            Q.append(w / hn)
        m = j + 1
        e1 = torch.zeros(m + 1, device=b.device, dtype=b.dtype); e1[0] = beta
        y, *_ = torch.linalg.lstsq(H[: m + 1, : m], e1.unsqueeze(-1))
        y = y.squeeze(-1)
        Qm = torch.stack(Q[:m])
        return (y.unsqueeze(1) * Qm).sum(0)

    # ------------------------------------------------------------------
    def update(self, token_ids: torch.Tensor, targets: torch.Tensor) -> Dict:
        model = self.model
        cfg = self.cfg
        V = model.cfg.vocab_size
        L = model.n_blocks

        X = model.tok_emb[token_ids] + model._pos_cache[:token_ids.size(1)].to(model.tok_emb.device).to(model.tok_emb.dtype)

        # ---- 1. forward: relax every block to its equilibrium ----------
        with torch.no_grad():
            out = model.relax_all(X.detach(), steps=cfg.free_steps, anderson=cfg.anderson,
                                   anderson_m=cfg.anderson_m, anderson_beta=cfg.anderson_beta)
            if out["diverged"] or not torch.isfinite(out["states"][-1]).all():
                self._enforce_all()
                self.step += 1
                return {"loss": float("nan"), "res_free": out["residuals"][-1], "skipped": 1}
            h_list = [s.detach() for s in out["states"]]   # [X, h_1, ..., h_L]
        res_free = out["residuals"][-1]
        H_L = h_list[-1]

        # ---- 2. readout loss & top adjoint -----------------------------
        H_L.requires_grad_(True)
        logits = H_L @ model.output_weight + model.b_out
        loss = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1))
        g, = torch.autograd.grad(loss, H_L, retain_graph=False)   # dC/dh_L

        # ---- 3. walk the chain of equilibria top-down ------------------
        per_block_grads: List[List[torch.Tensor]] = [None] * L
        for l in reversed(range(L)):
            blk = model.blocks[l]
            Z = h_list[l + 1].detach().requires_grad_(True)       # this block's equilibrium
            Xin = h_list[l].detach().requires_grad_(True)         # its input (prev equilibrium)
            target = blk.forward_diff(Z, Xin)                     # f_l(Z, Xin; theta_l)

            # adj_l = (I - J_Z^T)^{-1} g   via GMRES on the VJP oracle
            def Av(w, _t=target, _z=Z):
                ws = w.view_as(_z)
                JTw, = torch.autograd.grad(_t, _z, grad_outputs=ws, retain_graph=True)
                return (ws - JTw.detach()).reshape(-1)

            adj = self._gmres(Av, g.detach().reshape(-1), cfg.gmres_k).view_as(Z)

            # param grads = VJP of f w.r.t. this block's params
            grads = torch.autograd.grad(target, blk.params, grad_outputs=adj,
                                        retain_graph=True, create_graph=False)
            per_block_grads[l] = [gd.detach() for gd in grads]

            # propagate adjoint to the previous layer: g = dC/dXin
            g_in, = torch.autograd.grad(target, Xin, grad_outputs=adj,
                                        retain_graph=False, create_graph=False)
            g = g_in.detach()

        # g is now dC/dX -> embedding adjoint

        # ---- 4. readout gradients (local) ------------------------------
        p = torch.softmax(logits.detach(), dim=-1)
        err = p - F.one_hot(targets, V).float()
        g_OW = torch.einsum("btd,btV->dV", H_L.detach(), err) / H_L.size(0)
        g_bout = err.mean(dim=(0, 1))

        # ---- 5. embedding gradient (input adjoint + readout if tied) ---
        agg = torch.zeros_like(model.tok_emb)
        flat_ids = token_ids.reshape(-1)
        agg.index_add_(0, flat_ids, g.reshape(-1, model.tok_emb.size(1)))
        counts = torch.zeros(model.tok_emb.size(0), device=model.tok_emb.device)
        counts.index_add_(0, flat_ids, torch.ones_like(flat_ids, dtype=model.tok_emb.dtype))
        counts = counts.clamp(min=1.0).unsqueeze(1)
        g_tokemb = agg / counts
        if model.cfg.tie_embeddings:
            g_tokemb = g_tokemb + g_OW.t()

        # ---- 6. Adam updates -------------------------------------------
        lr = self._sched(cfg.lr); lr_out = self._sched(cfg.lr_out); lr_emb = self._sched(cfg.lr_emb)
        for l, blk in enumerate(model.blocks):
            pg = per_block_grads[l]
            for name, p_, g_ in zip(
                ["Wq", "Wk", "Wv", "Wo", "W1", "b1", "W2", "b2"], blk.params, pg
            ):
                self._adam(f"b{l}.{name}", p_, g_, lr)
        self._adam("b_out", model.b_out, g_bout, lr_out)
        self._adam("tok_emb", model.tok_emb, g_tokemb, lr_emb)
        if not model.cfg.tie_embeddings:
            self._adam("W_out", model.W_out, g_OW, lr_out)

        # ---- 7. per-block contractivity homeostasis --------------------
        self._enforce_all()

        self.step += 1
        return {"loss": loss.item(), "res_free": res_free, "skipped": 0}

    # ------------------------------------------------------------------
    def _enforce_all(self):
        budget = self.cfg.contractivity / max(self.model.cfg.res_gain, 1e-3)
        for blk in self.model.blocks:
            blk.enforce_contractivity(budget)
