"""Numerical differentiation of the residual system at the steady state.

Coordinates
-----------
Everything downstream works in the rescaled deviation

    x = xbar + s * yhat,        s_i = |xbar_i| when that is nonzero, else 1

so ``yhat`` is a PROPORTIONAL deviation for variables with a nonzero steady
state and an ABSOLUTE one for the rest. The change of variables is linear, which
is what makes it safe: unlike a log-linearisation it leaves the second-order
tensor a simple rescaling, ``H_hat[e,i,j] = H_level[e,i,j] * s_i * s_j``.

The rescaling is not cosmetic. In an applied model the levels span many orders
of magnitude -- a capital stock of order 40 next to a mean-zero cost shifter --
and the QZ pencil on the unscaled matrices is badly conditioned enough to move
the eigenvalues that Blanchard-Kahn is counting. Row equilibration on top of it
is for the same reason, and neither touches the solution: both are similarity
operations on the pencil.

Testing which variables are proportional by their steady-state LEVEL rather than
by name matters. Adding a mean-zero shock to a model that scaled by name gives
``0 * (1 + yhat) = 0``, and the shock disappears in silence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .model import Model, Params
from .steady import SteadyState

__all__ = ["Coordinates", "LinearSystem", "first_order_system", "second_derivatives",
           "third_derivatives"]


@dataclass
class Coordinates:
    """Affine map between model levels and the rescaled deviations."""

    center: np.ndarray
    scale: np.ndarray

    def to_level(self, yhat: ArrayLike) -> np.ndarray:
        return self.center + self.scale * np.asarray(yhat, dtype=float)

    def to_hat(self, x: ArrayLike) -> np.ndarray:
        return (np.asarray(x, dtype=float) - self.center) / self.scale

    @classmethod
    def from_steady_state(cls, ss: SteadyState, floor: float = 1e-8) -> "Coordinates":
        return cls(center=ss.x.copy(), scale=ss.scale(floor))


@dataclass
class LinearSystem:
    """``A E_t y_{t+1} = B y_t + C eps_{t+1}`` in rescaled deviations."""

    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    coords: Coordinates
    steady_residual: float
    row_scale: np.ndarray

    @property
    def n(self) -> int:
        return self.A.shape[0]


def _residual_at(model: Model, coords: Coordinates, params, ctx):
    """Residual as a function of the rescaled deviations, for differentiation."""
    def f(y_next, y_now, eps):
        return model.residuals(coords.to_level(y_next), coords.to_level(y_now),
                               eps, params, ctx)
    return f


def first_order_system(model: Model, ss: SteadyState, params: Params, *, ctx: Any = "inherit",
                       h: float = 1e-6, equilibrate: bool = True) -> LinearSystem:
    """Build ``A``, ``B``, ``C`` by central differences around the steady state.

    Central rather than forward differences: the forward formula is first-order
    accurate and its truncation error enters the eigenvalues of the pencil,
    which is precisely what the Blanchard-Kahn count is reading.
    """
    if isinstance(ctx, str) and ctx == "inherit":
        ctx = ss.ctx
    coords = Coordinates.from_steady_state(ss)
    f = _residual_at(model, coords, params, ctx)
    n, ne = model.n_vars, model.n_shocks
    zero_y = np.zeros(n)
    zero_e = np.zeros(ne)

    res0 = f(zero_y, zero_y, zero_e)
    steady_residual = float(np.max(np.abs(res0)))

    exact = model.jacobian(coords.center, coords.center, zero_e, params, ctx)
    if exact is not None:
        # Exact derivatives arrive in model LEVELS; the rescaling is one
        # multiplication per column because the change of variables is affine.
        dF_next = exact[:, :n] * coords.scale[None, :]
        dF_now = exact[:, n:2 * n] * coords.scale[None, :]
        dF_eps = exact[:, 2 * n:]
        return _assemble(dF_next, dF_now, dF_eps, coords, steady_residual, equilibrate)

    dF_next = np.zeros((n, n))
    dF_now = np.zeros((n, n))
    dF_eps = np.zeros((n, ne))
    for j in range(n):
        e = np.zeros(n); e[j] = h
        dF_next[:, j] = (f(e, zero_y, zero_e) - f(-e, zero_y, zero_e)) / (2 * h)
        dF_now[:, j] = (f(zero_y, e, zero_e) - f(zero_y, -e, zero_e)) / (2 * h)
    for k in range(ne):
        e = np.zeros(ne); e[k] = h
        dF_eps[:, k] = (f(zero_y, zero_y, e) - f(zero_y, zero_y, -e)) / (2 * h)

    return _assemble(dF_next, dF_now, dF_eps, coords, steady_residual, equilibrate)


def _assemble(dF_next, dF_now, dF_eps, coords, steady_residual, equilibrate):
    """Cast the derivatives into ``A y' = B y + C eps`` and equilibrate the rows."""
    # f(y', y, eps) = 0 linearised is  dF_next y' + dF_now y + dF_eps eps = 0,
    # i.e. A y' = B y + C eps with A = dF_next, B = -dF_now, C = -dF_eps.
    A, B, C = dF_next, -dF_now, -dF_eps
    n = A.shape[0]
    row_scale = np.ones(n)
    if equilibrate:
        row_scale = np.maximum(np.abs(A), np.abs(B)).max(axis=1)
        row_scale[row_scale < 1e-12] = 1.0
        A = A / row_scale[:, None]
        B = B / row_scale[:, None]
        C = C / row_scale[:, None]
    return LinearSystem(A=A, B=B, C=C, coords=coords,
                        steady_residual=steady_residual, row_scale=row_scale)


def second_derivatives(model: Model, ss: SteadyState, params: Params, *, ctx: Any = "inherit",
                       h: float = 1e-4, row_scale: np.ndarray | None = None,
                       at: ArrayLike | None = None) -> np.ndarray:
    """Hessian of every residual with respect to ``w = [y_{t+1}; y_t; eps]``.

    Returns an ``(n_eq, m, m)`` array with ``m = 2 n + n_shocks``, in the same
    rescaled coordinates as :func:`first_order_system` and divided by the same
    row scaling, so the first- and second-order objects are commensurate.

    The step defaults to ``1e-4``. Second differences divide by ``h**2``, so
    round-off grows as ``eps_machine / h**2`` while truncation falls as ``h**2``;
    the balance sits near ``eps_machine ** 0.25``, and the ``1e-6`` that is right
    for a first derivative loses roughly six digits here.
    """
    if isinstance(ctx, str) and ctx == "inherit":
        ctx = ss.ctx
    coords = Coordinates.from_steady_state(ss)
    f = _residual_at(model, coords, params, ctx)
    n, ne = model.n_vars, model.n_shocks
    m = 2 * n + ne

    w0 = np.zeros(m) if at is None else np.asarray(at, dtype=float)
    if w0.size != m:
        raise ValueError(f"`at` must have {m} entries, it has {w0.size}")

    exact = model.hessian(coords.to_level(w0[:n]), coords.to_level(w0[n:2 * n]),
                          w0[2 * n:], params, ctx)
    if exact is not None:
        s_full = np.concatenate([coords.scale, coords.scale, np.ones(ne)])
        H = exact * s_full[None, :, None] * s_full[None, None, :]
        if row_scale is not None:
            H = H / np.asarray(row_scale, dtype=float)[:, None, None]
        return H

    def F(w):
        return f(w[:n], w[n:2 * n], w[2 * n:])

    f0 = F(w0)
    n_eq = f0.size

    # Cache the 2m single-coordinate displacements: every mixed partial reuses
    # them, and without the cache the routine costs 4 m^2 evaluations instead of
    # 2 m^2 + 2 m.
    plus = np.empty((m, n_eq))
    minus = np.empty((m, n_eq))
    for i in range(m):
        e = np.zeros(m); e[i] = h
        plus[i] = F(w0 + e)
        minus[i] = F(w0 - e)

    H = np.zeros((n_eq, m, m))
    for i in range(m):
        H[:, i, i] = (plus[i] - 2.0 * f0 + minus[i]) / (h * h)
    for i in range(m):
        ei = np.zeros(m); ei[i] = h
        for j in range(i + 1, m):
            ej = np.zeros(m); ej[j] = h
            fpp = F(w0 + ei + ej)
            fmm = F(w0 - ei - ej)
            # f_ij = [f(++) - f(+i) - f(+j) + 2 f0 - f(-i) - f(-j) + f(--)] / (2 h^2)
            # The symmetric six-point form: it needs only the two extra corner
            # evaluations per pair, the cached single displacements supplying
            # the rest, and it is second-order accurate like the diagonal.
            val = (fpp + fmm - plus[i] - minus[i] - plus[j] - minus[j] + 2.0 * f0) / (2.0 * h * h)
            H[:, i, j] = val
            H[:, j, i] = val

    if row_scale is not None:
        H = H / np.asarray(row_scale, dtype=float)[:, None, None]
    return H


def third_derivatives(model: Model, ss: SteadyState, params: Params, *, ctx: Any = "inherit",
                      h: float = 1e-3, h_hess: float = 1e-4,
                      row_scale: np.ndarray | None = None) -> np.ndarray:
    """Third derivatives of each residual, ``(n_eq, m, m, m)`` with ``m = 2n + n_shocks``.

    Obtained by central differences of the HESSIAN, not of the function. That choice is what makes
    the third order feasible. Differentiating the function three times costs ``O(m^3)``
    evaluations and divides by the CUBE of the step: for a model with 19 variables and 5 shocks
    that is eighty thousand evaluations, and less than half of the digits survive. Differentiating
    the Hessian costs ``2m`` evaluations of it, and when the Hessian is analytic, as the symbolic
    front end supplies it, the result inherits that exactness in a single central difference.

    The outer step ``h`` is larger than the inner ``h_hess`` on purpose: differentiating an object
    that was itself obtained by differences needs a step that dominates that object's noise, not a
    smaller step, which would amplify it.
    """
    if isinstance(ctx, str) and ctx == "inherit":
        ctx = ss.ctx
    n, ne = model.n_vars, model.n_shocks
    m = 2 * n + ne

    def hess(w):
        return second_derivatives(model, ss, params, ctx=ctx, h=h_hess, at=w)

    T = np.empty((model.n_vars, m, m, m))
    for i in range(m):
        e = np.zeros(m); e[i] = h
        T[:, :, :, i] = (hess(e) - hess(-e)) / (2 * h)

    # The third derivative is symmetric in its three indices. Imposing that removes the asymmetric
    # noise of the differencing instead of carrying it into the Sylvester equation, where it would
    # spread to every coefficient.
    T = (T + np.transpose(T, (0, 2, 1, 3)) + np.transpose(T, (0, 1, 3, 2))
         + np.transpose(T, (0, 3, 1, 2)) + np.transpose(T, (0, 2, 3, 1))
         + np.transpose(T, (0, 3, 2, 1))) / 6.0
    if row_scale is not None:
        T = T / np.asarray(row_scale, dtype=float)[:, None, None, None]
    return T
