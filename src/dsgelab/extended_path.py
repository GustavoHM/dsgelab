"""Extended path: a stochastic simulation by a perfect-foresight problem every period.

Fair and Taylor (1983) and, for the stochastic version, Adjemian and Juillard
(2013), as Dynare implements them. At each date the shock is drawn, the states it
leads to are computed, and the agents' plan is the solution of a nonlinear
problem that ends ``horizon`` periods later at the steady state; the first date of
the plan is kept and the rest is thrown away.

With ``order = 0`` the agents expect no further shocks: the plan is a perfect-
foresight path. With ``order = p > 0`` they integrate over the shocks of the next
``p`` periods with a Gauss--Hermite quadrature, on a tree of plans. In the FULL
tree every node of the first ``p`` dates branches into one child per quadrature
node; in the SPARSE tree (Dynare's default) only the central branch, the one on
which every shock is zero, keeps branching, and the side branches are carried on
deterministically. Either way, beyond date ``p`` every branch expects no further
shocks.

The equations of a node split in two. A TRANSITION -- a row with no next-period
jump in it -- holds in every child, with that child's shock; an EXPECTATIONAL row
holds on average over the children, with the quadrature weights. In a ``from_mod``
model the transitions are the auxiliary rows that carry lags and exogenous
levels, and the split is Dynare's own: an equation dated ``t`` is averaged over
the values of ``t+1``.

``hybrid=True`` adds, where a child does not branch any further, the second-
order risk correction ``g_ss/2`` to that child's jumps in its parent's
expectation (Dynare's ``hybrid_order = 2``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

from .foresight import _expost_states, perfect_foresight
from .model import Model, Params
from .steady import SteadyState

if TYPE_CHECKING:
    from .secondorder import SecondOrderSolution

__all__ = ["ExtendedPath", "extended_path", "quadrature_nodes"]


@dataclass
class ExtendedPath:
    """The simulated levels, date 0 (the initial point) to ``T``, and the shocks drawn."""

    levels: np.ndarray
    shocks: np.ndarray
    model: Model = field(repr=False)
    iterations: list = field(default_factory=list)

    def __getitem__(self, name: str) -> np.ndarray:
        return self.levels[:, self.model.position(name)]

    def values(self) -> dict[str, np.ndarray]:
        return {n: self.levels[:, i] for i, n in enumerate(self.model.names)}


def quadrature_nodes(cov: ArrayLike, n: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Tensor Gauss--Hermite nodes and weights for the shocks with positive variance.

    Dynare's construction: the standard-normal nodes, scaled by the upper Cholesky
    factor of the covariance of the shocks that have a variance, in a tensor
    product with the first shock varying fastest; the zero node, which must
    exist (``n`` odd), is moved to the front. Returns ``(nodes, weights)``, the
    nodes ``(m, n_shocks)`` with zeros for the shocks without a variance.
    """
    if n % 2 == 0:
        raise ValueError("the number of quadrature nodes must be odd (a zero node is needed)")
    cov = np.atleast_2d(np.asarray(cov, dtype=float))
    ne = cov.shape[0]
    pos = np.where(np.diag(cov) > 0)[0]
    b = np.sqrt(np.arange(1, n) / 2.0)
    vals, vecs = np.linalg.eigh(np.diag(b, 1) + np.diag(b, -1))
    order = np.argsort(vals)
    x0 = np.sqrt(2.0) * vals[order]
    w0 = vecs[0, order] ** 2
    x0[np.abs(x0) < 1e-14] = 0.0
    Omega = np.linalg.cholesky(cov[np.ix_(pos, pos)]).T     # upper triangular
    cols = np.tile(x0[:, None], (1, pos.size)) @ Omega
    grid = list(product(*[range(n)] * pos.size))
    grid = [g[::-1] for g in grid]                           # first index fastest, as ndgrid
    nodes = np.zeros((len(grid), ne))
    weights = np.empty(len(grid))
    for r, g in enumerate(grid):
        nodes[r, pos] = [cols[g[i], i] for i in range(pos.size)]
        weights[r] = np.prod([w0[g[i]] for i in range(pos.size)])
    zero = int(np.where(np.all(np.abs(nodes) < 1e-12, axis=1))[0][0])
    idx = [zero] + [r for r in range(len(grid)) if r != zero]
    return nodes[idx], weights[idx]


def _transition_rows(model: Model, x, params, ctx):
    """Rows with no next-period jump in them: the ones that hold in every child."""
    n, nk = model.n_vars, model.n_states
    rng = np.random.default_rng(0)
    mags = []
    for scale in (0.0, 1e-3):
        z = x * (1 + scale * rng.standard_normal(n)) + scale * rng.standard_normal(n)
        J = model.jacobian(z, z, np.zeros(model.n_shocks), params, ctx)
        if J is None:
            h = 1e-7
            base = model.residuals(z, z, np.zeros(model.n_shocks), params, ctx)
            J = np.empty((n, n))
            for j in range(n):
                a = z.copy(); a[j] += h
                J[:, j] = (model.residuals(a, z, np.zeros(model.n_shocks), params, ctx) - base) / h
        mags.append(np.abs(J[:, :n]))
    m = np.maximum(*mags)
    floor = 1e-12 * max(1.0, float(m.max()))
    rows = np.where((m[:, nk:].max(axis=1) <= floor) & (m[:, :nk].max(axis=1) > floor))[0]
    if rows.size != nk:
        raise ValueError(f"expected {nk} transition rows (no next-period jump), found "
                         f"{rows.size}: the stochastic extended path needs them to split "
                         f"the equations")
    return rows, np.setdiff1d(np.arange(n), rows)


def _tree(order: int, n_nodes: int, T: int, sparse: bool):
    """Nodes ``(date, parent, quadrature index or -1, branches)`` in creation order."""
    nodes = [(0, -1, -1, order > 0)]
    frontier = [0]
    for t in range(1, T + 1):
        nxt = []
        for p in frontier:
            if nodes[p][3]:
                for q in range(n_nodes):
                    branches = t < order and (not sparse or q == 0)
                    nodes.append((t, p, q, branches))
                    nxt.append(len(nodes) - 1)
            else:
                nodes.append((t, p, -1, False))
                nxt.append(len(nodes) - 1)
        frontier = nxt
    return nodes


def _solve_tree(model, k0, steady, params, ctx, tree, nodes_q, weights, rows_s, rows_e,
                correction, T, guess, tol, maxiter):
    """Stacked Newton on the tree: unknowns are the root's jumps and every other node's vector."""
    n, nk = model.n_vars, model.n_states
    nu = n - nk
    N = len(tree)
    children = [[] for _ in range(N)]
    for i, (t, p, q, b) in enumerate(tree):
        if p >= 0:
            children[p].append(i)
    col0 = np.empty(N, dtype=int)            # first unknown of each node's vector (root: -nk)
    col0[0] = -nk
    col0[1:] = nu + n * np.arange(N - 1)
    n_unk = nu + n * (N - 1)

    if guess is None:
        X = np.tile(steady.x, (N, 1))
    else:
        X = guess.copy()
    X[0, :nk] = k0
    zero_e = np.zeros(model.n_shocks)
    ss_u = steady.x[nk:]

    def unpack(z):
        Xn = X.copy()
        Xn[0, nk:] = z[:nu]
        Xn[1:] = z[nu:].reshape(N - 1, n)
        return Xn

    z = np.concatenate([X[0, nk:], X[1:].ravel()])
    it = 0
    for it in range(maxiter + 1):
        Xn = unpack(z)
        res = np.empty(n_unk)
        rows, cols, vals = [], [], []
        r0 = 0
        for i, (t, p, q, b) in enumerate(tree):
            ch = children[i]
            if not ch:                                  # terminal: jumps at the steady state
                res[r0:r0 + nu] = Xn[i, nk:] - ss_u
                rows.append(r0 + np.arange(nu)); cols.append(col0[i] + nk + np.arange(nu))
                vals.append(np.ones(nu))
                r0 += nu
                continue
            branching = len(ch) > 1
            e_avg = np.zeros(nu)
            re_ = r0 + nk * len(ch)
            for m, c in enumerate(ch):
                _, _, qc, bc = tree[c]
                w = weights[qc] if branching else 1.0
                eps = nodes_q[qc] if qc >= 0 else zero_e
                xc = Xn[c]
                # transitions, with this child's shock, at its own values
                rs = r0 + nk * m
                res[rs:rs + nk] = model.residuals(xc, Xn[i], eps, params, ctx)[rows_s]
                Jc = _jac(model, xc, Xn[i], eps, params, ctx)
                _put(rows, cols, vals, rs + np.arange(nk), Jc[rows_s, :n], col0[c], n)
                _put_now(rows, cols, vals, rs + np.arange(nk), Jc[rows_s, n:2 * n], col0[i],
                         nk, n, i == 0)
                # expectations, on average over the children; a child that branches no
                # further carries the risk correction in its parent's expectation
                if correction is not None and branching and not bc:
                    xe = xc.copy()
                    xe[nk:] += correction
                    Je = _jac(model, xe, Xn[i], eps, params, ctx)
                else:
                    xe, Je = xc, Jc
                e_avg += w * model.residuals(xe, Xn[i], eps, params, ctx)[rows_e]
                _put(rows, cols, vals, re_ + np.arange(nu), w * Je[rows_e, :n], col0[c], n)
                _put_now(rows, cols, vals, re_ + np.arange(nu), w * Je[rows_e, n:2 * n],
                         col0[i], nk, n, i == 0)
            res[r0 + nk * len(ch):r0 + nk * len(ch) + nu] = e_avg
            r0 += nk * len(ch) + nu
        nrm = float(np.max(np.abs(res)))
        if nrm < tol:
            return Xn, it
        if it == maxiter:
            break
        J = csr_matrix(coo_matrix((np.concatenate(vals), (np.concatenate(rows),
                                                          np.concatenate(cols))),
                                  shape=(n_unk, n_unk)))
        dz = spsolve(J, -res)
        if not np.all(np.isfinite(dz)):
            raise RuntimeError("extended path: the Newton step on the tree is not finite")
        z = z + dz
    raise RuntimeError(f"extended path: no convergence on the tree in {maxiter} iterations "
                       f"(residual {nrm:.3e})")


def _jac(model, x_next, x_now, eps, params, ctx, h=1e-7):
    J = model.jacobian(x_next, x_now, eps, params, ctx)
    if J is not None:
        return J
    n = model.n_vars
    base = model.residuals(x_next, x_now, eps, params, ctx)
    out = np.empty((n, 2 * n))
    for j in range(n):
        a = x_next.copy(); a[j] += h
        out[:, j] = (model.residuals(a, x_now, eps, params, ctx) - base) / h
        a = x_now.copy(); a[j] += h
        out[:, n + j] = (model.residuals(x_next, a, eps, params, ctx) - base) / h
    return out


def _put(rows, cols, vals, r, block, c0, n):
    rr, cc = np.meshgrid(r, np.arange(n), indexing="ij")
    rows.append(rr.ravel()); cols.append((c0 + cc).ravel()); vals.append(block.ravel())


def _put_now(rows, cols, vals, r, block, c0, nk, n, root):
    if root:                                  # the root's states are known
        block = block[:, nk:]
        rr, cc = np.meshgrid(r, np.arange(nk, n), indexing="ij")
    else:
        rr, cc = np.meshgrid(r, np.arange(n), indexing="ij")
    rows.append(rr.ravel()); cols.append((c0 + cc).ravel()); vals.append(block.ravel())


def extended_path(model: Model, steady: SteadyState, params: Params,
                  shocks: ArrayLike | None = None, *,
                  periods: int | None = None, cov: ArrayLike | None = None,
                  seed: int | None = None,
                  horizon: int = 200, order: int = 0, nodes: int = 3, tree: str = "sparse",
                  hybrid: SecondOrderSolution | bool | None = None,
                  initial_states: ArrayLike | None = None, ctx: Any = None, tol: float = 1e-12,
                  maxiter: int = 50) -> ExtendedPath:
    """Simulate by extended path.

    Parameters
    ----------
    shocks : array ``(T, n_shocks)``, optional
        The innovations, date 1 to ``T``. Without them, ``periods`` draws from a
        normal with covariance ``cov`` (and ``seed``).
    horizon : int
        Periods of each plan before the steady state is imposed (Dynare's
        ``ep.periods``, 200 by default).
    order, nodes, tree
        The stochastic extended path: shocks integrated over in the next
        ``order`` periods, with ``nodes`` Gauss--Hermite nodes per shock, on the
        ``"sparse"`` or ``"full"`` tree (Dynare's ``ep.stochastic.algo`` 1 and 0).
        ``cov`` is then required.
    hybrid : SecondOrderSolution or True, optional
        Add the second-order risk correction where a branch stops branching.
        ``True`` computes it from ``cov``.
    initial_states : array, optional
        States at date 0; the steady state by default.

    Returns an :class:`ExtendedPath` whose row ``t`` is date ``t``: row 0 is the
    initial point, row ``t`` the outcome of the ``t``-th shock (Dynare's column
    ``t + 1``).
    """
    n, nk = model.n_vars, model.n_states
    if shocks is None:
        if periods is None or cov is None:
            raise ValueError("pass `shocks`, or `periods` and `cov` to draw them")
        L = np.linalg.cholesky(np.atleast_2d(np.asarray(cov, dtype=float)) + 0.0)
        shocks = np.random.default_rng(seed).standard_normal((periods, model.n_shocks)) @ L.T
    E = np.atleast_2d(np.asarray(shocks, dtype=float))
    if E.shape[1] != model.n_shocks and E.shape[0] == model.n_shocks:
        E = E.T
    T_sim = E.shape[0]
    if order > 0 and cov is None:
        raise ValueError("the stochastic extended path (order > 0) needs `cov`")
    if tree not in ("sparse", "full"):
        raise ValueError("tree must be 'sparse' or 'full'")

    correction = None
    if hybrid is not None and hybrid is not False:
        if order == 0:
            raise ValueError("the hybrid correction applies to the stochastic extended path")
        if hybrid is True:
            from .secondorder import solve_second_order
            from .solve import solve
            hybrid = solve_second_order(solve(model, params, steady=steady), cov)
        correction = 0.5 * hybrid.g_ss * hybrid.first.coords.scale[nk:]

    levels = np.zeros((T_sim + 1, n))
    levels[0] = steady.x
    if initial_states is not None:
        levels[0, :nk] = np.asarray(initial_states, dtype=float)
        if order == 0:
            p0 = perfect_foresight(model, steady, steady, params=params, T=horizon, ctx=ctx,
                                   initial_states=levels[0, :nk], tol=tol, maxiter=maxiter)
            levels[0] = p0.levels[0]
    if order > 0:
        nodes_q, weights = quadrature_nodes(cov, nodes)
        rows_s, rows_e = _transition_rows(model, steady.x, params, ctx)
        tr = _tree(order, len(weights), horizon, tree == "sparse")
    iterations = []
    guess = None
    for t in range(1, T_sim + 1):
        # the states the realised shock leads to; the previous plan's next date is the guess
        k = _expost_states(model, levels[t - 1], levels[t - 1] if guess is None else guess[1],
                           E[t - 1], params, ctx)
        if order == 0:
            g = None
            if guess is not None:
                g = np.vstack([guess[1:], steady.x[None, :]])
            plan = perfect_foresight(model, steady, steady, params=params, T=horizon, ctx=ctx,
                                     initial_states=k, guess=g, tol=tol, maxiter=maxiter)
            guess = plan.levels
            levels[t] = plan.levels[0]
            iterations.append(plan.iterations)
        else:
            X, it = _solve_tree(model, k, steady, params, ctx, tr, nodes_q, weights, rows_s,
                                rows_e, correction, horizon, guess, tol, maxiter)
            guess = X
            levels[t] = X[0]
            iterations.append(it)
    return ExtendedPath(levels=levels, shocks=E, model=model, iterations=iterations)
