"""Anderson acceleration (AA-m) for fixed-point relaxation.

Anderson acceleration mixes a history of past iterates and residuals to find a
fixed point ``Z = f(Z)`` in far fewer steps than plain Picard / damped
relaxation, and --- crucially for EnergyLM --- it often converges even when the
plain damped iteration is on the edge of contractivity.  That lets us use a
richer recurrent map (larger ``res_gain``) and therefore a more expressive
steady state ``Z*`` while still converging.

This is a type-II AA with damping ``beta`` and Tikhonov regularisation ``lam``
(see Walker & Ni, 2011; Bai, Kolter & Koltun, DEQ 2019).  It operates on the
flattened state so it is dimension-agnostic.  No backpropagation is involved.
"""

from __future__ import annotations

import torch


class AndersonAccel:
    def __init__(self, m: int = 5, beta: float = 0.8, lam: float = 1e-4):
        self.m = m
        self.beta = beta
        self.lam = lam
        self.X: list = []   # past iterates x
        self.G: list = []   # past g(x) = f(x)
        self.R: list = []   # past residuals r = g - x

    def reset(self):
        self.X.clear(); self.G.clear(); self.R.clear()

    @staticmethod
    def _flat(t: torch.Tensor) -> torch.Tensor:
        return t.reshape(t.shape[0], -1)  # keep batch dim, flatten the rest

    def step(self, x: torch.Tensor, gx: torch.Tensor) -> torch.Tensor:
        """Given current iterate ``x`` and its image ``gx = f(x)``, return the
        next iterate (Anderson-accelated when enough history is available)."""
        r = gx - x
        self.X.append(x); self.G.append(gx); self.R.append(r)
        if len(self.X) > self.m + 1:
            self.X.pop(0); self.G.pop(0); self.R.pop(0)

        p = len(self.X) - 1            # number of difference vectors
        if p < 1:
            return x + self.beta * r   # plain damped relaxation

        xf = self._flat(x)
        rf = self._flat(r)
        # difference vectors of iterates (S) and residuals (Y)
        Xf = torch.stack([self._flat(t) for t in self.X])  # (p+1, B, D)
        Rf = torch.stack([self._flat(t) for t in self.R])  # (p+1, B, D)
        S = Xf[1:] - Xf[:-1]          # (p, B, D)
        Y = Rf[1:] - Rf[:-1]          # (p, B, D)

        # Per-batch least squares: gamma (p, B) solving (Y Y^T + lam I) gamma = Y r_k
        # Y Y^T -> (B, p, p);  Y r_k -> (B, p, 1)
        Yy = torch.einsum("pbd,qbd->bpq", Y, Y)            # (B, p, p)
        eye = torch.eye(p, device=x.device, dtype=x.dtype).expand_as(Yy)
        Yy = Yy + self.lam * eye
        rhs = torch.einsum("pbd,bd->bp", Y, rf)            # (B, p)
        gamma = torch.linalg.solve(Yy, rhs.unsqueeze(-1)).squeeze(-1)  # (B, p)

        # AA type-II damped update:  x_{k+1} = x + beta*r - sum_i gamma_i S_i
        corr = torch.einsum("bp,pbd->bd", gamma, S)        # (B, D)
        x_next_flat = xf + self.beta * rf - corr
        return x_next_flat.reshape_as(x)
