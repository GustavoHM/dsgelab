"""The balanced growth path of a model written in levels.

Dynare's ``bgp.write``: every variable is assumed to grow at a constant factor,
``x_t = X g^t``, and the model must hold at two consecutive dates,

    f(X g, X, 0) = 0,        f(X g^2, X g, 0) = 0,

``2n`` equations in the ``n`` levels ``X`` and the ``n`` growth factors ``g``. The
growth factors are identified; the level of a growing variable is not -- only its
ratio to the variables that share its trend is -- so the system is solved by
least squares (Levenberg--Marquardt) from a start, as Dynare's tests do, and the
answer reports which variables grow.

For a ``from_mod`` model the variables are Dynare's, with Dynare's dating: the
level of ``v`` is the one of the period whose equations are evaluated first, and an
auxiliary ``LAG1_v`` grows with ``v``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import least_squares

from .model import Model, Params

__all__ = ["BalancedGrowth", "balanced_growth"]


@dataclass
class BalancedGrowth:
    """Levels and growth factors along a balanced growth path."""

    levels: dict
    growth: dict
    residual: float
    model: Model = field(repr=False)
    success: bool = True
    degenerate: bool = False

    def ratio(self, a: str, b: str) -> float:
        """``a/b`` along the path; meaningful when both share a trend."""
        if abs(self.growth[a] - self.growth[b]) > 1e-8 * max(1.0, abs(self.growth[b])):
            raise ValueError(f"{a} and {b} grow at different factors: their ratio is not constant")
        return self.levels[a] / self.levels[b]

    @property
    def growing(self) -> list[str]:
        return [v for v, g in self.growth.items() if abs(g - 1) > 1e-8]


def bgp_residual(model: Model, params: Params, X: ArrayLike, G: ArrayLike,
                 ctx: Any = None) -> np.ndarray:
    """``[f(X g, X, 0); f(X g^2, X g, 0)]``: the two dates of a balanced growth path."""
    X, G = np.asarray(X, dtype=float), np.asarray(G, dtype=float)
    e = np.zeros(model.n_shocks)
    return np.concatenate([model.residuals(X * G, X, e, params, ctx),
                           model.residuals(X * G * G, X * G, e, params, ctx)])


def balanced_growth(model: Model, params: Params, *,
                    guess: dict[str, float] | ArrayLike | None = None,
                    growth: dict[str, float] | ArrayLike | None = None,
                    fixed: Mapping[str, float] | None = None, ctx: Any = None,
                    tol: float = 1e-12, seed: int | None = 0, max_nfev: int = 100000,
                    tries: int = 20) -> BalancedGrowth:
    """Solve for the balanced growth path.

    ``guess`` and ``growth`` are starting levels and growth factors, as mappings or
    vectors in the model's order (by default, levels drawn around 1 with ``seed``,
    as in Dynare's tests, and growth factors of 1). ``fixed`` maps variables that
    do not grow to their level -- the states that carry exogenous variables, which
    Dynare holds at zero -- and removes them from the unknowns.

    A model in levels often has trivial paths too, on which a factor of production
    is zero and output with it. A solution with a free level at zero is set aside
    and the search restarts from another random start, up to ``tries`` times; if
    every start ends there, the first solution comes back with ``degenerate=True``.
    """
    n = model.n_vars
    rng = np.random.default_rng(seed)
    fixed = dict(fixed or {})
    fix_idx = np.array([model.position(k) for k in fixed], dtype=int)
    free = np.setdiff1d(np.arange(n), fix_idx)
    fix_val = np.array([float(v) for v in fixed.values()])

    def vec(v, default):
        if v is None:
            return default
        if isinstance(v, dict):
            out = default.copy()
            for k, x in v.items():
                out[model.position(k)] = float(x)
            return out
        return np.asarray(v, dtype=float).copy()

    e = np.zeros(model.n_shocks)
    m = free.size

    def full(z):
        X, G = np.empty(n), np.ones(n)
        X[free], G[free] = z[:m], z[m:]
        X[fix_idx] = fix_val
        return X, G

    def F(z):
        return bgp_residual(model, params, *full(z), ctx)

    def jac(z):
        X, G = full(z)
        J1 = model.jacobian(X * G, X, e, params, ctx)
        J2 = model.jacobian(X * G * G, X * G, e, params, ctx)
        A1, B1 = J1[:, :n], J1[:, n:2 * n]
        A2, B2 = J2[:, :n], J2[:, n:2 * n]
        top = np.hstack([(A1 * G + B1)[:, free], (A1 * X)[:, free]])
        bot = np.hstack([(A2 * G * G + B2 * G)[:, free], (A2 * (2 * X * G) + B2 * X)[:, free]])
        return np.vstack([top, bot])

    use_exact = None
    first = None
    for attempt in range(max(1, tries)):
        X0 = vec(guess if attempt == 0 else None, 1 + (rng.random(n) - 0.5) * 0.2)
        G0 = vec(growth, np.ones(n))
        if use_exact is None:
            use_exact = model.jacobian(X0 * G0, X0, e, params, ctx) is not None
        sol = least_squares(F, np.concatenate([X0[free], G0[free]]),
                            jac=jac if use_exact else "2-point", method="lm",
                            xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=max_nfev)
        X, G = full(sol.x)
        res = float(np.max(np.abs(F(sol.x))))
        degenerate = bool(np.min(np.abs(X[free]), initial=np.inf) < 1e-6)
        out = BalancedGrowth(levels=dict(zip(model.names, X)), growth=dict(zip(model.names, G)),
                             residual=res, model=model, success=res < tol, degenerate=degenerate)
        if out.success and not degenerate:
            return out
        if first is None or (out.success and not first.success):
            first = out
    return first
