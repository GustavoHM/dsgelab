"""Nonlinear perfect-foresight transitions by stacked-time Newton.

The same residual function that the perturbation linearises is imposed period by
period, so the deterministic transition and the stochastic block share their
equations by construction rather than by the author's care.

Boundary conditions
-------------------
The ``n_k`` predetermined states at the START are inherited from the old steady
state; the ``n_u`` forward jumps at the END sit at the new one. Fixing the whole
terminal vector instead -- the natural-looking mistake -- overdetermines the
system by ``n_k`` equations, and the only symptom is a Newton that stops
converging.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve

from .model import Model
from .steady import SteadyState

__all__ = ["TransitionPath", "perfect_foresight", "path_of"]


@dataclass
class TransitionPath:
    """Levels along a transition, ``(T+1, n_vars)``, plus convergence evidence."""

    levels: np.ndarray
    model: Model = field(repr=False)
    converged: bool
    iterations: int
    residual: float
    params_path: list = field(repr=False, default_factory=list)
    ctx_path: list = field(repr=False, default_factory=list)

    @property
    def periods(self) -> int:
        return self.levels.shape[0]

    def __getitem__(self, name: str) -> np.ndarray:
        return self.levels[:, self.model.position(name)]

    def values(self) -> dict:
        return {n: self.levels[:, i] for i, n in enumerate(self.model.names)}

    def derived(self, fn: Callable) -> dict:
        """Evaluate a per-period function of the variables along the whole path.

        ``fn(values, params, ctx)`` returns a mapping of named scalars; the
        results are stacked into arrays. This is how the objects that are not
        state variables -- prices, shares, flows defined by identities -- are
        recovered without recomputing them by hand at every date.
        """
        out = None
        for t in range(self.periods):
            vals = {n: float(self.levels[t, i]) for i, n in enumerate(self.model.names)}
            p = self.params_path[t] if self.params_path else None
            c = self.ctx_path[t] if self.ctx_path else None
            got = fn(vals, p, c)
            if out is None:
                out = {k: np.empty(self.periods) for k in got}
            for k, v in got.items():
                out[k][t] = float(v)
        return out or {}


def path_of(value, periods: int) -> list:
    """Normalise a constant, a sequence or a callable into a per-period list."""
    if callable(value):
        return [value(t) for t in range(periods)]
    if isinstance(value, (list, tuple)) or (isinstance(value, np.ndarray) and value.ndim >= 1):
        seq = list(value)
        if len(seq) != periods:
            raise ValueError(f"path has {len(seq)} entries, expected {periods}")
        return seq
    return [value] * periods


def perfect_foresight(model: Model, start: SteadyState, terminal: SteadyState, *,
                      params, T: int = 120, ctx=None, initial_states=None,
                      maxiter: int = 50, tol: float = 1e-10, damping: float = 1.0,
                      jacobian: str = "forward", h: float = 1e-7,
                      guess: np.ndarray | None = None,
                      raise_on_fail: bool = True) -> TransitionPath:
    """Solve the deterministic transition from ``start`` to ``terminal``.

    Parameters
    ----------
    params, ctx : object, sequence or callable
        Constant, a sequence of length ``T+1``, or ``f(t)``. A sequence is how an
        ANNOUNCED policy path enters -- a statutory schedule known to agents from
        the outset -- as against a surprise, which is a change in ``terminal``
        with a constant ``params``.
    initial_states : array_like, optional
        Overrides the states inherited from ``start``, for a transition that does
        not begin at a steady state.
    guess : (T+1, n_vars) array, optional
        Initial iterate. Defaults to the terminal steady state everywhere with
        the initial states pinned, which is adequate for most transitions; a
        previous solution is a better start for a continuation sweep.
    jacobian : {"forward", "central"}
        Difference scheme for the Newton Jacobian. Forward halves the cost and
        does not affect the accuracy of the converged path, which is governed by
        ``tol``; central helps when the residuals are noisy.
    """
    n = model.n_vars
    nk = model.n_states
    nu = model.n_jumps
    params_path = path_of(params, T + 1)
    ctx_path = path_of(ctx, T + 1)
    zero = np.zeros(model.n_shocks)

    y0_states = (np.asarray(initial_states, dtype=float) if initial_states is not None
                 else start.x[:nk].copy())
    if y0_states.size != nk:
        raise ValueError(f"initial_states has {y0_states.size} entries for {nk} states")
    yT = terminal.x.copy()

    if guess is None:
        Y = np.tile(yT, (T + 1, 1))
        Y[0, :nk] = y0_states
    else:
        Y = np.array(guess, dtype=float)
        if Y.shape != (T + 1, n):
            raise ValueError(f"guess must be ({T + 1}, {n}), got {Y.shape}")
        Y[0, :nk] = y0_states

    def pack(Y):
        return np.concatenate([Y[0, nk:], Y[1:].ravel()])

    def unpack(z):
        Yn = np.empty((T + 1, n))
        Yn[0, :nk] = y0_states
        Yn[0, nk:] = z[:nu]
        Yn[1:] = z[nu:].reshape(T, n)
        return Yn

    def residual(Yn):
        r = np.empty(T * n + nu)
        for t in range(T):
            r[t * n:(t + 1) * n] = model.residuals(Yn[t + 1], Yn[t], zero,
                                                   params_path[t], ctx_path[t])
        r[T * n:] = Yn[T, nk:] - yT[nk:]
        return r

    z = pack(Y)
    nrm = np.inf
    it = 0
    for it in range(maxiter):
        Yn = unpack(z)
        r = residual(Yn)
        nrm = float(np.max(np.abs(r)))
        if nrm < tol:
            return TransitionPath(levels=Yn, model=model, converged=True, iterations=it,
                                  residual=nrm, params_path=params_path, ctx_path=ctx_path)
        J = lil_matrix((T * n + nu, z.size))
        for t in range(T):
            yt, ytp = Yn[t], Yn[t + 1]
            pt, ct = params_path[t], ctx_path[t]
            base = model.residuals(ytp, yt, zero, pt, ct)
            for j in range(n):
                if jacobian == "central":
                    a = ytp.copy(); a[j] += h
                    b = ytp.copy(); b[j] -= h
                    col_next = (model.residuals(a, yt, zero, pt, ct)
                                - model.residuals(b, yt, zero, pt, ct)) / (2 * h)
                    a = yt.copy(); a[j] += h
                    b = yt.copy(); b[j] -= h
                    col_now = (model.residuals(ytp, a, zero, pt, ct)
                               - model.residuals(ytp, b, zero, pt, ct)) / (2 * h)
                else:
                    a = ytp.copy(); a[j] += h
                    col_next = (model.residuals(a, yt, zero, pt, ct) - base) / h
                    a = yt.copy(); a[j] += h
                    col_now = (model.residuals(ytp, a, zero, pt, ct) - base) / h
                J[t * n:(t + 1) * n, nu + t * n + j] = col_next.reshape(-1, 1)
                if t == 0:
                    if j >= nk:                    # only the jumps are unknown at t=0
                        J[0:n, j - nk] = col_now.reshape(-1, 1)
                else:
                    J[t * n:(t + 1) * n, nu + (t - 1) * n + j] = col_now.reshape(-1, 1)
        for j in range(nu):
            J[T * n + j, nu + (T - 1) * n + nk + j] = 1.0

        dz = spsolve(csr_matrix(J), -r)
        if not np.all(np.isfinite(dz)):
            raise RuntimeError("perfect_foresight: the Newton step is not finite; the "
                               "stacked Jacobian is singular at this iterate")
        z = z + damping * dz

    Yn = unpack(z)
    if raise_on_fail:
        raise RuntimeError(f"perfect_foresight: no convergence in {maxiter} iterations "
                           f"(residual {nrm:.3e}). Try a longer T, a smaller `damping`, "
                           f"or a `guess` from a nearby solution.")
    return TransitionPath(levels=Yn, model=model, converged=False, iterations=it + 1,
                          residual=nrm, params_path=params_path, ctx_path=ctx_path)
