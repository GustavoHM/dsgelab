"""Chebyshev projection for a nonlinear rational-expectations residual.

The same ``f(y_{t+1}, y_t, eps) = 0`` used by the perturbation is collocated on a
tensor grid of the predetermined states. Rows of the steady-state Jacobian with
no future jump are the ex-post block: at each shock they determine the next
state, given today's state and today's jump. The remaining rows are integrated
against a Gauss–Hermite rule and set to zero. The jump policy is a tensor-product
Chebyshev polynomial. The initial coefficients are the first-order policy, so the
projection starts from the saddle path rather than from an arbitrary function.

The closed-form stochastic growth model, log utility and full depreciation, is
the check: the polynomial has to reproduce ``k' = αβ a k^α`` and
``c = (1-αβ) a k^α`` on the box it was built on. A tensor grid in more than
three states is not built. That is a different algorithm, not a larger ``order``.
On a :class:`SmolyakGrid` the policy is the grid's polynomial, with one
coefficient per node, and the collocation equations are solved by Newton's
method. Their Jacobian is assembled from the derivatives
of the residual at each node and quadrature point with respect to today's and
next period's jumps, so a Newton step costs a few evaluations of the residual
at every point rather than one per coefficient.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import least_squares, root

from .model import Covariance, Model, Params, Point, VarView
from .smolyak import SmolyakGrid
from .solve import solve
from .steady import SteadyState

__all__ = ["Projection", "solve_projection"]


@dataclass
class Projection:
    """A jump policy ``u(k)`` collocated on a box of the predetermined states.

    On a Smolyak grid, ``grid`` holds it, ``order`` is the highest degree along any state, and
    ``nodes`` and ``bounds`` are in the model's state order.
    """

    model: Model
    params: Params
    bounds: np.ndarray          # (n_k, 2)
    order: int
    coefficients: np.ndarray    # (n_u, n_basis)
    nodes: np.ndarray           # (n_nodes, n_k) in levels
    residual: float
    success: bool
    grid: SmolyakGrid | None = None

    def jumps(self, k: ArrayLike) -> np.ndarray:
        """Jump vector at one state or at a row-stacked array of states."""
        k = np.asarray(k, dtype=float)
        single = k.ndim == 1
        if single:
            k = k[None, :]
        if self.grid is not None:
            basis = self.grid.basis(k[:, _grid_columns(self.model, self.grid)])
        else:
            basis = _basis_matrix(k, self.bounds, self.order)
        out = basis @ self.coefficients.T
        return out[0] if single else out


def solve_projection(model: Model, params: Params, *, bounds: ArrayLike | None = None,
                     guess: Point | None, order: int = 5, cov: Covariance = None,
                     n_quad: int = 7, positive: Sequence[str] = (),
                     steady: SteadyState | None = None,
                     grid: SmolyakGrid | None = None) -> Projection:
    """Collocate the expectational residual on a Chebyshev tensor grid or a Smolyak grid.

    Parameters
    ----------
    bounds :
        ``(low, high)`` for each predetermined state, in levels, in the model's
        state order. The box has to contain the steady state.
    order :
        Polynomial degree along each state. The grid has ``(order + 1)`` nodes
        a side. More than three states is refused: a full tensor is the wrong
        object there.
    cov :
        Shock covariance, as elsewhere in the package. It scales the quadrature.
    grid :
        A :class:`SmolyakGrid` with one axis per predetermined state, named
        after it, in place of ``bounds`` and ``order``. It takes any number of
        states.
    """
    nk = model.n_states
    if grid is not None:
        if bounds is not None:
            raise ValueError("give bounds and order for a tensor grid, or a SmolyakGrid, not both")
        if not isinstance(grid, SmolyakGrid):
            raise TypeError("grid must be a SmolyakGrid")
        cols = _grid_columns(model, grid)
        bounds = np.empty((nk, 2))
        bounds[cols] = grid.bounds
    elif bounds is None:
        raise ValueError("give bounds (with order) for a tensor grid, or a SmolyakGrid")
    bounds = np.asarray(bounds, dtype=float)
    if bounds.shape != (nk, 2):
        raise ValueError(f"bounds must have shape ({nk}, 2), got {bounds.shape}")
    if nk > 3 and grid is None:
        raise ValueError(
            f"{nk} states: a tensor Chebyshev grid is not built past 3. "
            "Pass a SmolyakGrid for more.")
    if model.n_jumps == 0:
        raise ValueError("a projection needs at least one jump; a pure law of motion is already explicit")
    sol = solve(model, params, guess=guess, steady=steady, positive=positive)
    expost, expect = _partition(sol.system.A, nk)
    if expost.size != nk or expect.size != model.n_jumps:
        raise RuntimeError(
            f"the steady-state Jacobian splits into {expost.size} ex-post rows and "
            f"{expect.size} expectational rows, not {nk} and {model.n_jumps}. "
            "The partition is the incidence of future jumps, and this model does not have that split.")
    if grid is not None:
        return _solve_smolyak(model, params, grid, bounds, sol, expost, expect,
                              *_quadrature(model, cov, n_quad))
    nodes_1d = [_chebyshev_nodes(order + 1) for _ in range(nk)]
    grids = np.meshgrid(*[
        _from_unit(nodes_1d[i], bounds[i, 0], bounds[i, 1]) for i in range(nk)
    ], indexing="ij")
    nodes = np.column_stack([g.ravel() for g in grids])
    quad_e, quad_w = _quadrature(model, cov, n_quad)
    basis = _basis_matrix(nodes, bounds, order)
    # Linear policy, in levels, as the start of the nonlinear solve.
    u0 = np.vstack([_linear_jumps(sol, nodes[i]) for i in range(nodes.shape[0])])
    coef0, *_ = np.linalg.lstsq(basis, u0, rcond=None)
    coef0 = coef0.T  # (nu, n_basis)

    def policy_from(coef, k):
        b = _basis_matrix(np.asarray(k, dtype=float)[None, :], bounds, order)
        return (b @ coef.T).ravel()

    ctx = sol.ctx

    def residuals(flat):
        coef = flat.reshape(model.n_jumps, -1)
        out = np.empty(nodes.shape[0] * model.n_jumps)
        for i, k in enumerate(nodes):
            u = policy_from(coef, k)
            acc = np.zeros(model.n_jumps)
            for e, w in zip(quad_e, quad_w):
                kn = _next_state(model, params, ctx, k, u, e, expost, coef, bounds, order, sol)
                if kn is None:
                    acc += w * 1e3
                    continue
                un = policy_from(coef, kn)
                y = _pack(k, u)
                yn = _pack(kn, un)
                acc += w * model.residuals(yn, y, e, params, ctx)[expect]
            out[i * model.n_jumps:(i + 1) * model.n_jumps] = acc
        return out

    fit = least_squares(residuals, coef0.ravel(), method="trf", ftol=1e-10, xtol=1e-10,
                        max_nfev=40)
    coef = fit.x.reshape(model.n_jumps, -1)
    worst = float(np.max(np.abs(fit.fun))) if fit.fun.size else 0.0
    return Projection(model=model, params=params, bounds=bounds, order=order,
                      coefficients=coef, nodes=nodes, residual=worst,
                      success=bool(fit.success or worst < 1e-6))


def _grid_columns(model, grid):
    """The positions in the model's state vector of the grid's axes, in the grid's order."""
    if sorted(grid.names) != sorted(model.states):
        raise ValueError(f"the grid's axes {grid.names} must be the model's states "
                         f"{model.states}, one axis each")
    return np.array([model.states.index(n) for n in grid.names], dtype=int)


def _solve_smolyak(model, params, grid, bounds, sol, expost, expect, quad_e, quad_w,
                   tol=1e-10, max_iter=50):
    """Collocation on a Smolyak grid, by Newton's method on the coefficients.

    The residual at node ``i`` depends on the coefficients through today's jumps ``u_i = B_i c``
    and next period's ``u' = B(k') c``, with ``k'`` given by the ex-post rows from ``k_i``,
    ``u_i`` and the shock. Its derivative is the derivative in ``u_i``, next state and next
    jumps included, times ``B_i``, plus the derivative in ``u'`` at fixed ``k'`` times
    ``B(k')``. Both are taken by finite differences at all nodes at once.
    """
    nk, nu = model.n_states, model.n_jumps
    cols = _grid_columns(model, grid)
    K = np.empty((grid.n, nk))
    K[:, cols] = grid.points
    B = grid.basis(grid.points)
    residuals = _vectorised_residuals(model, params, sol.ctx)
    k_ss = sol.steady.x[:nk]
    k_scale = sol.coords.scale[:nk]
    Q = sol.Q if sol.Q is not None else np.zeros((nk, model.n_shocks))

    def policy(coef, k):
        return grid.basis(k[:, cols]) @ coef.T

    def next_states(k, u, e):
        """The ex-post rows solved for next period's states, by Newton at every point."""
        E = np.broadcast_to(e, (k.shape[0], model.n_shocks))
        kn = k_ss + ((k - k_ss) / k_scale @ sol.P.T + E @ Q.T) * k_scale
        X = np.column_stack([k, u])

        def g(kn):
            return residuals(np.column_stack([kn, u]), X, E)[:, expost]

        for _ in range(30):
            r = g(kn)
            if np.max(np.abs(r)) < 1e-13 * (1.0 + np.max(np.abs(kn))):
                break
            J = np.empty((k.shape[0], nk, nk))
            for j in range(nk):
                h = 1e-7 * (1.0 + np.abs(kn[:, j]))
                kp = kn.copy()
                kp[:, j] += h
                J[:, :, j] = (g(kp) - r) / h[:, None]
            step = np.linalg.solve(J, r[:, :, None])[:, :, 0]
            kn = kn - step
            if np.max(np.abs(step) / (1.0 + np.abs(kn))) < 1e-14:
                break
        return kn

    def at_shock(coef, u, e):
        """The expectational rows at every node for one shock, and ``k'``, ``u'``."""
        kn = next_states(K, u, e)
        un = policy(coef, kn)
        E = np.broadcast_to(e, (grid.n, model.n_shocks))
        R = residuals(np.column_stack([kn, un]), np.column_stack([K, u]), E)[:, expect]
        return R, kn, un, E

    def residual(coef):
        u = B @ coef.T
        return sum(w * at_shock(coef, u, e)[0] for e, w in zip(quad_e, quad_w))

    def jacobian(coef):
        u = B @ coef.T
        today = np.zeros((grid.n, nu, nu))
        J = np.zeros((grid.n, nu, nu, grid.n))
        for e, w in zip(quad_e, quad_w):
            R, kn, un, E = at_shock(coef, u, e)
            X = np.column_stack([K, u])
            for j in range(nu):
                h = 1e-7 * (1.0 + np.abs(u[:, j]))
                up = u.copy()
                up[:, j] += h
                today[:, :, j] += w * (at_shock(coef, up, e)[0] - R) / h[:, None]
                h = 1e-7 * (1.0 + np.abs(un[:, j]))
                unp = un.copy()
                unp[:, j] += h
                Rn = residuals(np.column_stack([kn, unp]), X, E)[:, expect]
                J[:, :, j, :] += (w * (Rn - R) / h[:, None])[:, :, None] * grid.basis(
                    kn[:, cols])[:, None, :]
        J += today[:, :, :, None] * B[:, None, None, :]
        return J.reshape(grid.n * nu, nu * grid.n)

    u0 = np.vstack([_linear_jumps(sol, k) for k in K])
    coef = grid.coefficients(u0).T
    R = residual(coef)
    worst = float(np.max(np.abs(R)))
    converged = worst < tol
    for _ in range(max_iter):
        if converged:
            break
        step = np.linalg.solve(jacobian(coef), R.ravel()).reshape(nu, grid.n)
        t = 1.0
        for _ in range(30):
            trial = coef - t * step
            Rt = residual(trial)
            if np.all(np.isfinite(Rt)) and np.sum(Rt ** 2) < (1.0 - 1e-4 * t) * np.sum(R ** 2):
                break
            t *= 0.5
        else:
            break
        coef, R = trial, Rt
        worst = float(np.max(np.abs(R)))
        converged = worst < tol
    return Projection(model=model, params=params, bounds=bounds,
                      order=int(grid.degrees.max()), coefficients=coef, nodes=K,
                      residual=worst, success=bool(converged or worst < 1e-6), grid=grid)


def _vectorised_residuals(model, params, ctx):
    """The model's residuals at ``P`` points at once, ``(P, n_vars)``.

    The equations are called with each variable a vector. If that fails, or disagrees with the
    call at single points on the first use, they are evaluated point by point from then on.
    """
    mode = {"vector": None}

    def loop(Xn, X, E):
        return np.array([model.residuals(xn, x, e, params, ctx)
                         for xn, x, e in zip(Xn, X, E)]).reshape(X.shape[0], model.n_vars)

    def vector(Xn, X, E):
        nxt = VarView(Xn.T, model._index)
        cur = VarView(X.T, model._index)
        e = VarView(E.T, model._shock_index)
        r = (model.equations(nxt, cur, e, params, ctx) if model._wants_ctx
             else model.equations(nxt, cur, e, params))
        P = X.shape[0]
        rows = [np.broadcast_to(np.asarray(ri, dtype=float), (P,)) for ri in r]
        if len(rows) != model.n_vars:
            raise ValueError(f"{model.name}: equations returned {len(rows)} residuals for "
                             f"{model.n_vars} variables")
        return np.column_stack(rows)

    def call(Xn, X, E):
        if mode["vector"] is not False:
            try:
                R = vector(Xn, X, E)
                if mode["vector"] is None:
                    k = min(3, X.shape[0])
                    mode["vector"] = bool(np.allclose(R[:k], loop(Xn[:k], X[:k], E[:k]),
                                                      rtol=1e-12, atol=1e-14, equal_nan=True))
                return R if mode["vector"] else loop(Xn, X, E)
            except Exception:
                if mode["vector"]:
                    raise
                mode["vector"] = False
        return loop(Xn, X, E)

    return call


def _partition(A, nk, tol=1e-8):
    scale = max(float(np.abs(A).max()), 1.0)
    if A.shape[1] == nk:
        forward = np.zeros(A.shape[0])
    else:
        forward = np.abs(A[:, nk:]).max(axis=1)
    expost = np.where(forward <= tol * scale)[0]
    expect = np.where(forward > tol * scale)[0]
    return expost, expect


def _pack(k, u):
    return np.concatenate([np.asarray(k, dtype=float), np.asarray(u, dtype=float)])


def _linear_jumps(sol, k_level):
    khat = (k_level - sol.steady.x[:sol.model.n_states]) / sol.coords.scale[:sol.model.n_states]
    uhat = sol.F @ khat
    center = sol.steady.x[sol.model.n_states:]
    scale = sol.coords.scale[sol.model.n_states:]
    return center + scale * uhat


def _next_state(model, params, ctx, k, u, eps, expost, coef, bounds, order, sol):
    def policy(kn):
        b = _basis_matrix(np.asarray(kn, dtype=float)[None, :], bounds, order)
        return (b @ coef.T).ravel()

    y = _pack(k, u)
    k_hat = (k - sol.steady.x[:model.n_states]) / sol.coords.scale[:model.n_states]
    Q = sol.Q if sol.Q is not None else np.zeros((model.n_states, model.n_shocks))
    guess = sol.steady.x[:model.n_states] + sol.coords.scale[:model.n_states] * (sol.P @ k_hat + Q @ eps)

    def resid(kn):
        un = policy(kn)
        return model.residuals(_pack(kn, un), y, eps, params, ctx)[expost]

    try:
        ans = root(resid, guess, method="hybr", tol=1e-10)
    except (ValueError, FloatingPointError):
        return None
    if not np.all(np.isfinite(ans.x)) or float(np.max(np.abs(resid(ans.x)))) > 1e-6:
        return None
    return ans.x


def _basis_matrix(states, bounds, order) -> np.ndarray:
    """``states`` is ``(n_points, n_k)``. Columns are the tensor-product Chebyshev basis."""
    nk = bounds.shape[0]
    mapped = np.column_stack([
        _to_unit(states[:, i], bounds[i, 0], bounds[i, 1]) for i in range(nk)
    ])
    factors = [_chebyshev(mapped[:, i], order) for i in range(nk)]  # each (n_points, order+1)
    out = factors[0]
    for nxt in factors[1:]:
        out = (out[:, :, None] * nxt[:, None, :]).reshape(states.shape[0], -1)
    return out


def _chebyshev(x, order) -> np.ndarray:
    cols = [np.ones_like(x), x.copy()]
    for _ in range(2, order + 1):
        cols.append(2.0 * x * cols[-1] - cols[-2])
    return np.column_stack(cols[:order + 1])


def _chebyshev_nodes(n: int) -> np.ndarray:
    k = np.arange(n)
    return np.cos(np.pi * (2 * k + 1) / (2 * n))


def _to_unit(s, lo, hi):
    return 2.0 * (s - lo) / (hi - lo) - 1.0


def _from_unit(x, lo, hi):
    return lo + (x + 1.0) * (hi - lo) / 2.0


def _quadrature(model, cov, n_quad: int):
    from .solution import shock_covariance
    Sigma = shock_covariance(model.shocks, cov)
    # Diagonal covariance: a tensor Gauss–Hermite rule. A full covariance is
    # rotated into independent normals by its symmetric square root.
    vals, vecs = np.linalg.eigh(Sigma)
    vals = np.maximum(vals, 0.0)
    nodes, weights = np.polynomial.hermite.hermgauss(n_quad)
    nodes = nodes * np.sqrt(2.0)
    weights = weights / np.sqrt(np.pi)
    axes = [nodes * np.sqrt(v) if v > 0 else np.zeros(n_quad) for v in vals]
    weight_axes = [weights for _ in vals]
    grids = np.meshgrid(*axes, indexing="ij")
    wgrids = np.meshgrid(*weight_axes, indexing="ij")
    std = np.column_stack([g.ravel() for g in grids])
    w = np.prod(np.column_stack([g.ravel() for g in wgrids]), axis=1)
    shock = std @ vecs.T
    if model.n_shocks > 2:
        raise ValueError(
            f"{model.n_shocks} shocks: the tensor quadrature is built for one or two. "
            "A sparse rule is a different algorithm.")
    return shock, w
