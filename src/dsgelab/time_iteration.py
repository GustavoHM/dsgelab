"""Time iteration on a Cartesian grid, with Markov or quadrature exogenous states and bounded controls.

The residual is the model's own ``f(y_{t+1}, y_t, eps) = 0``, the one the perturbation solves. Its
rows are sorted by what they involve at ``t+1``. A row with next-period states and no next-period
jump is a transition: given today's states and controls and a shock, the transitions give the next
states. A row with a next-period jump, or with nothing dated ``t+1``, is an equilibrium condition,
integrated over the next period and set to zero. With a :class:`MarkovChain` for an exogenous
state, the next value of that state comes from the chain, and the transition row that defines it
(the row with that state at ``t+1`` and no endogenous state there) is left out. The sorting is done
on the model's residual itself, by moving one ``t+1`` variable at a time and seeing which rows
change, so it does not depend on a steady state.

Each iteration takes the policy found so far as the next period's, and solves the equilibrium
conditions for today's controls at every grid point, by Newton's method point by point (Coleman,
1990). It stops when the policy moves by less than ``tol`` at every point. A bounded control is
paired with one equilibrium condition, the one in the same position as the control in the list
of jumps, by complementarity in the standard convention: the condition is zero between the
bounds, nonnegative at the lower bound and nonpositive at the upper one. This is dolo's
convention for its arbitrage equations. The pair is solved through the Fischer–Burmeister
function (Fischer, 1992), in the box form of Billups (1995), which is smooth wherever the
solution is not exactly at a bound with a zero residual.

The policy is interpolated between the grid points, linearly or by a natural cubic spline along
each axis, and extrapolated linearly from the ends of the grid in both cases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike

from .markov import MarkovChain
from .model import Covariance, Model, Params, Point, VarView
from .solution import shock_covariance

__all__ = ["CartesianGrid", "TimeIterationSolution", "time_iteration"]

#: A bound on a control: a number, ``None`` for none, or ``bound(states, params)`` with ``states``
#: a name-indexed view of today's states.
Bound = float | None | Callable[[VarView, Params], Any]


class CartesianGrid:
    """A tensor grid: one increasing vector of nodes for each named axis.

    ``axes`` maps each name to its nodes, or to ``(low, high, n)`` for ``n`` evenly spaced nodes.
    ``points`` lists every node of the grid, one row per node and one column per axis, with the
    last axis varying fastest.
    """

    def __init__(self, axes: Mapping[str, ArrayLike | tuple[float, float, int]]):
        self.names = list(axes)
        if not self.names:
            raise ValueError("a grid needs at least one axis")
        self.axes = []
        for name, spec in axes.items():
            if isinstance(spec, tuple) and len(spec) == 3 and float(spec[2]).is_integer():
                lo, hi, n = float(spec[0]), float(spec[1]), int(spec[2])
                if not hi > lo:
                    raise ValueError(f"axis {name!r}: the upper end must exceed the lower one")
                nodes = np.linspace(lo, hi, n)
            else:
                nodes = np.asarray(spec, dtype=float).ravel()
            if nodes.size < 2 or np.any(np.diff(nodes) <= 0) or not np.all(np.isfinite(nodes)):
                raise ValueError(f"axis {name!r} needs at least two finite, increasing nodes")
            self.axes.append(nodes)
        self.shape = tuple(a.size for a in self.axes)
        mesh = np.meshgrid(*self.axes, indexing="ij")
        self.points = np.column_stack([m.ravel() for m in mesh])

    @property
    def n(self) -> int:
        """Number of grid points."""
        return int(np.prod(self.shape))

    @property
    def d(self) -> int:
        """Number of axes."""
        return len(self.axes)

    @property
    def midpoints(self) -> np.ndarray:
        """The centres of the grid's cells, where the interpolation is least accurate."""
        mids = [0.5 * (a[1:] + a[:-1]) for a in self.axes]
        mesh = np.meshgrid(*mids, indexing="ij")
        return np.column_stack([m.ravel() for m in mesh])

    def interpolate(self, values: ArrayLike, points: ArrayLike, kind: str = "linear") -> np.ndarray:
        """Interpolate ``values``, one row per grid point, at ``points``, one row per point."""
        values = np.asarray(values, dtype=float)
        if values.shape[0] != self.n:
            raise ValueError(f"values need one row per grid point, {self.n}, not {values.shape[0]}")
        return self._evaluate(self._coefficients(values, kind), np.asarray(points, dtype=float),
                              kind)

    # The interpolant is linear in the values. Along each axis it is a combination of two
    # neighbouring values (linear) or of two values and the spline's second derivatives at them
    # (cubic). _coefficients stacks the second derivatives after the values along each axis, once
    # per policy; _evaluate then combines 2**d or 4**d entries at each point.
    def _coefficients(self, values: np.ndarray, kind: str) -> np.ndarray:
        if kind not in ("linear", "cubic"):
            raise ValueError(f"interpolation must be 'linear' or 'cubic', not {kind!r}")
        tail = values.shape[1:]
        G = values.reshape(self.shape + tail)
        if kind == "cubic":
            for axis, nodes in enumerate(self.axes):
                S = _natural_spline_matrix(nodes)
                M = np.tensordot(S, G, axes=([1], [axis]))
                G = np.concatenate([G, np.moveaxis(M, 0, axis)], axis=axis)
        return G

    def _evaluate(self, G: np.ndarray, points: np.ndarray, kind: str) -> np.ndarray:
        single = points.ndim == 1
        if single:
            points = points[None, :]
        if points.shape[1] != self.d:
            raise ValueError(f"points need {self.d} columns, one per axis ({self.names})")
        P = points.shape[0]
        idx = np.zeros((P, 1), dtype=np.intp)
        weight = np.ones((P, 1))
        stride = 1
        sizes = G.shape[:self.d]
        strides = []
        for size in reversed(sizes):
            strides.append(stride)
            stride *= size
        strides = strides[::-1]
        for axis, nodes in enumerate(self.axes):
            i, w = _axis_weights(nodes, points[:, axis], kind)
            idx = (idx[:, :, None] + strides[axis] * i[:, None, :]).reshape(P, -1)
            weight = (weight[:, :, None] * w[:, None, :]).reshape(P, -1)
        flat = G.reshape((-1,) + G.shape[self.d:])
        out = np.einsum("pk,pk...->p...", weight, flat[idx])
        return out[0] if single else out

    def __repr__(self) -> str:
        sides = ", ".join(f"{n}: {a.size} from {a[0]:g} to {a[-1]:g}"
                          for n, a in zip(self.names, self.axes))
        return f"CartesianGrid({sides})"


def _natural_spline_matrix(x: np.ndarray) -> np.ndarray:
    """The matrix that takes values at ``x`` to a natural cubic spline's second derivatives."""
    n = x.size
    S = np.zeros((n, n))
    if n < 3:
        return S
    h = np.diff(x)
    A = np.zeros((n - 2, n - 2))
    B = np.zeros((n - 2, n))
    for r in range(n - 2):
        A[r, r] = (h[r] + h[r + 1]) / 3.0
        if r > 0:
            A[r, r - 1] = h[r] / 6.0
        if r < n - 3:
            A[r, r + 1] = h[r + 1] / 6.0
        B[r, r] = 1.0 / h[r]
        B[r, r + 1] = -1.0 / h[r] - 1.0 / h[r + 1]
        B[r, r + 2] = 1.0 / h[r + 1]
    S[1:-1] = np.linalg.solve(A, B)
    return S


def _axis_weights(x: np.ndarray, s: np.ndarray, kind: str) -> tuple[np.ndarray, np.ndarray]:
    """Indices and weights along one axis, into the values or into ``[values; second derivatives]``."""
    n = x.size
    i = np.clip(np.searchsorted(x, s, side="right") - 1, 0, n - 2)
    h = x[i + 1] - x[i]
    t = (s - x[i]) / h
    if kind == "linear":
        return np.column_stack([i, i + 1]), np.column_stack([1.0 - t, t])
    # Inside the cell, the cubic in values and second derivatives. Beyond an end, the tangent
    # line there, whose value and slope are also combinations of the same four entries.
    a, b = 1.0 - t, t
    c = (a ** 3 - a) * h ** 2 / 6.0
    d = (b ** 3 - b) * h ** 2 / 6.0
    w = np.column_stack([a, b, c, d])
    low, high = t < 0.0, t > 1.0
    if np.any(low):
        dt = s[low] - x[0]
        hl = h[low]
        # value and slope at x[0]: f0, (f1 - f0)/h - h (2 M0 + M1)/6
        w[low] = np.column_stack([1.0 - dt / hl, dt / hl, -dt * hl / 3.0, -dt * hl / 6.0])
    if np.any(high):
        dt = s[high] - x[-1]
        hh = h[high]
        # value and slope at x[-1]: f1, (f1 - f0)/h + h (M0 + 2 M1)/6
        w[high] = np.column_stack([-dt / hh, 1.0 + dt / hh, dt * hh / 6.0, dt * hh / 3.0])
    return np.column_stack([i, i + 1, n + i, n + i + 1]), w


@dataclass
class TimeIterationSolution:
    """A policy for the jumps on a grid of the continuous states, one per node of the Markov chain.

    ``values[m]`` holds the jumps at every grid point when the chain is at node ``m``, one row per
    point in the order of ``grid.points``. Without a Markov chain there is one node.
    """

    model: Model
    params: Params
    grid: CartesianGrid
    chain: MarkovChain | None
    markov_states: list[str]
    values: np.ndarray                 # (n_nodes, grid.n, n_jumps)
    interpolation: str
    iterations: int
    converged: bool
    distance: float
    history: list[float] = field(repr=False)
    lower: np.ndarray = field(repr=False)   # (n_nodes, grid.n, n_jumps), -inf where unbounded
    upper: np.ndarray = field(repr=False)
    _problem: Any = field(repr=False, default=None)

    def controls(self, states: Point) -> np.ndarray:
        """The jumps at one state or at a row-stacked array of states, in the model's state order.

        A state given by a Markov chain must take one of the chain's node values.
        """
        X, single = self._problem.states_array(states)
        node, cont = self._problem.split_states(X)
        out = self._problem.policy(self._problem.coefficients(self.values), node, cont)
        return out[0] if single else out

    __call__ = controls

    def euler_errors(self, states: Point | None = None) -> np.ndarray:
        """The equilibrium conditions, integrated, at ``states``, one row per state.

        For a bounded control the error is the natural residual of the complementarity,
        ``mid(u - lower, F, u - upper)``: the condition itself between the bounds, and only its
        wrong-signed part at a bound. It is in the units in which the condition is written, so
        an Euler equation written as ``1 - beta E[...]`` gives unit-free errors. By default the
        states are the centres of the grid's cells at every node of the chain, the points
        farthest from the grid, where the interpolation is least accurate.
        """
        pb = self._problem
        if states is None:
            mids = self.grid.midpoints
            node = np.repeat(np.arange(pb.n_nodes), mids.shape[0])
            cont = np.tile(mids, (pb.n_nodes, 1))
        else:
            X, _ = pb.states_array(states)
            node, cont = pb.split_states(X)
        G = pb.coefficients(self.values)
        u = pb.policy(G, node, cont)
        F = pb.equilibrium(u, node, cont, G)
        lo, hi = pb.bounds_at(node, cont)
        return np.median(np.stack([u - lo, F, u - hi]), axis=0)

    def simulate(self, periods: int, *, start: Point | None = None, seed: int | None = None,
                 shocks: ArrayLike | None = None) -> dict[str, np.ndarray]:
        """Simulate the policy for ``periods`` periods, from ``start`` (default: the grid's centre).

        The shocks are drawn from the covariance the solution was built with, or given in
        ``shocks``, one row per period after the first; a Markov state follows its chain. Returns
        ``states`` and ``jumps``, one row per period, in levels and in the model's order, and
        ``nodes``, the chain's index each period.
        """
        pb = self._problem
        rng = np.random.default_rng(seed)
        if start is None:
            X0 = np.empty(self.model.n_states)
            X0[pb.cont_idx] = [0.5 * (a[0] + a[-1]) for a in self.grid.axes]
            if pb.chain is not None:
                mid = pb.chain.n // 2
                X0[pb.markov_idx] = pb.node_values[mid]
        else:
            X0, single = pb.states_array(start)
            if not single:
                raise ValueError("start must be a single state")
            X0 = X0[0]
        node, cont = pb.split_states(X0[None, :])
        if pb.chain is not None:
            path = pb.chain.simulate(periods, start=int(node[0]), seed=rng, indices=True)
        else:
            path = np.zeros(periods, dtype=int)
        if shocks is None:
            eps = (rng.standard_normal((max(periods - 1, 0), pb.n_shocks)) @ pb.chol.T
                   if pb.chol is not None else np.zeros((max(periods - 1, 0), pb.n_shocks)))
        else:
            eps = np.asarray(shocks, dtype=float).reshape(max(periods - 1, 0), pb.n_shocks)
        G = pb.coefficients(self.values)
        states = np.empty((periods, self.model.n_states))
        jumps = np.empty((periods, self.model.n_jumps))
        for t in range(periods):
            u = pb.policy(G, node, cont)
            states[t] = pb.full_states(node, cont)[0]
            jumps[t] = u[0]
            if t == periods - 1:
                break
            nxt = np.array([path[t + 1]])
            cont = pb.next_states(cont, node, u, nxt, eps[t][None, :], G)
            node = nxt
        return {"states": states, "jumps": jumps, "nodes": path}


def time_iteration(model: Model, params: Params, *, grid: CartesianGrid | Mapping[str, Any],
                   markov: Mapping[str, MarkovChain] | None = None, cov: Covariance = None,
                   n_quad: int = 5, quadrature: tuple[ArrayLike, ArrayLike] | None = None,
                   bounds: Mapping[str, tuple[Bound, Bound]] | None = None,
                   interpolation: str = "linear",
                   initial: Any = None, guess: Point | None = None, steady: Any = None,
                   positive: Sequence[str] = (), tol: float = 1e-8, max_iter: int = 1000,
                   inner_tol: float = 1e-12, inner_max_iter: int = 50) -> TimeIterationSolution:
    """Solve ``model`` globally by time iteration.

    Parameters
    ----------
    grid :
        A :class:`CartesianGrid` of the continuous states, or the mapping that builds one. Every
        state not given by a Markov chain needs an axis.
    markov :
        Exogenous states given by Markov chains, by name; the chain's nodes are the state's values
        in levels. Several chains are taken as independent.
    cov, n_quad, quadrature :
        The shocks are integrated by a Gauss–Hermite rule with ``n_quad`` nodes per shock, scaled
        by ``cov`` (the package's convention: ``None`` is the identity), or by ``quadrature``,
        ``(nodes, weights)`` with one row of shocks per node. With a Markov chain and neither
        ``cov`` nor ``quadrature``, the shocks are set to zero: the chain carries the uncertainty.
    bounds :
        ``{jump: (lower, upper)}``, each bound a number, ``None``, or a function of today's states
        and the parameters. The equilibrium condition in the jump's position is its complement.
    interpolation :
        ``"linear"`` or ``"cubic"`` (a natural spline along each axis).
    initial :
        The policy the iteration starts from: a :class:`TimeIterationSolution` (to refine on a new
        grid), ``initial(states, params)`` returning the jumps in model order, or a mapping from
        each jump to a number or such a function. ``None`` starts from the first-order
        perturbation, which needs ``guess`` or ``steady``.
    tol :
        The iteration stops when no control moves by more than ``tol`` at any grid point.
    """
    if not isinstance(grid, CartesianGrid):
        grid = CartesianGrid(grid)
    pb = _Problem(model, params, grid, dict(markov or {}), cov, n_quad, quadrature,
                  dict(bounds or {}), interpolation)
    U = pb.initial_policy(initial, guess, steady, positive)
    lo, hi = pb.bounds_at(pb.grid_nodes, pb.grid_cont)
    U = np.clip(U, lo, hi)
    pb.classify(U)
    history: list[float] = []
    distance = np.inf
    it = 0
    while it < max_iter:
        it += 1
        G = pb.coefficients(U.reshape(pb.n_nodes, grid.n, -1))
        U_new = pb.solve_controls(U, G, lo, hi, inner_tol, inner_max_iter)
        distance = float(np.max(np.abs(U_new - U)))
        history.append(distance)
        U = U_new
        if not np.isfinite(distance):
            break
        if distance < tol:
            break
    shape = (pb.n_nodes, grid.n, model.n_jumps)
    return TimeIterationSolution(
        model=model, params=params, grid=grid, chain=pb.chain, markov_states=pb.markov_names,
        values=U.reshape(shape), interpolation=interpolation, iterations=it,
        converged=bool(distance < tol), distance=distance, history=history,
        lower=lo.reshape(shape), upper=hi.reshape(shape), _problem=pb)


class _Problem:
    """The pieces of one time-iteration problem, on arrays with one row per point."""

    def __init__(self, model, params, grid, markov, cov, n_quad, quadrature, bounds,
                 interpolation):
        self.model, self.params, self.grid = model, params, grid
        self.kind = interpolation
        if interpolation not in ("linear", "cubic"):
            raise ValueError(f"interpolation must be 'linear' or 'cubic', not {interpolation!r}")
        if model.n_jumps == 0:
            raise ValueError("time iteration needs at least one jump: a model with no control "
                             "has nothing to solve for")
        self.n_shocks = model.n_shocks
        self.ctx = None
        unknown = [s for s in list(markov) + grid.names if s not in model.states]
        if unknown:
            raise KeyError(f"not states of the model: {unknown}")
        both = sorted(set(markov) & set(grid.names))
        if both:
            raise ValueError(f"{both}: a state is given by a Markov chain or by a grid axis, "
                             "not both")
        missing = [s for s in model.states if s not in markov and s not in grid.names]
        if missing:
            raise ValueError(f"states with neither a grid axis nor a Markov chain: {missing}")
        for name, chain in markov.items():
            if not isinstance(chain, MarkovChain):
                raise TypeError(f"markov[{name!r}] must be a MarkovChain")
        self.markov_names = [s for s in model.states if s in markov]
        self.markov_idx = np.array([model.states.index(s) for s in self.markov_names], dtype=int)
        # the grid's axes, in the grid's order, as positions in the state vector
        self.cont_idx = np.array([model.states.index(s) for s in grid.names], dtype=int)
        self.endo = [s for s in model.states if s not in markov]
        if markov:
            chains = [markov[s] for s in self.markov_names]
            P = chains[0].transition
            vals = chains[0].nodes[:, None]
            for c in chains[1:]:
                P = np.kron(P, c.transition)
                vals = np.column_stack([np.repeat(vals, c.n, axis=0),
                                        np.tile(c.nodes, vals.shape[0])])
            self.chain = MarkovChain(np.arange(P.shape[0], dtype=float), P) \
                if len(chains) > 1 else chains[0]
            self.node_values = vals          # (n_nodes, n_markov)
        else:
            self.chain = None
            self.node_values = np.zeros((1, 0))
        self.n_nodes = self.node_values.shape[0]
        self.transition = self.chain.transition if self.chain is not None else np.ones((1, 1))
        # shocks
        self.chol = None
        if quadrature is not None:
            nodes, weights = quadrature
            nodes = np.asarray(nodes, dtype=float).reshape(-1, model.n_shocks)
            weights = np.asarray(weights, dtype=float).ravel()
            if weights.size != nodes.shape[0] or not np.isclose(weights.sum(), 1.0):
                raise ValueError("quadrature needs one weight per node, summing to one")
            self.eps_nodes, self.eps_weights = nodes, weights
            mean = weights @ nodes
            Sigma = (nodes - mean).T @ ((nodes - mean) * weights[:, None])
            self.chol = _factor(Sigma)
        elif markov and cov is None or model.n_shocks == 0:
            self.eps_nodes, self.eps_weights = np.zeros((1, model.n_shocks)), np.ones(1)
        else:
            Sigma = shock_covariance(model.shocks, cov)
            self.eps_nodes, self.eps_weights = _gauss_hermite(Sigma, n_quad)
            self.chol = _factor(Sigma)
        # controls' bounds
        for name in bounds:
            if name not in model.jumps:
                raise KeyError(f"bounds given for {name!r}, which is not a jump")
        self.bound_specs = bounds
        self.state_index = {s: i for i, s in enumerate(model.states)}
        self.grid_nodes = np.repeat(np.arange(self.n_nodes), grid.n)
        self.grid_cont = np.tile(grid.points, (self.n_nodes, 1))
        self._vectorised = None
        self.affine = None

    # -- states ------------------------------------------------------------
    def full_states(self, node, cont):
        X = np.empty((cont.shape[0], self.model.n_states))
        X[:, self.cont_idx] = cont
        if self.markov_idx.size:
            X[:, self.markov_idx] = self.node_values[node]
        return X

    def states_array(self, states):
        if isinstance(states, Mapping):
            X = np.atleast_1d(np.array([np.asarray(states[s], dtype=float)
                                        for s in self.model.states], dtype=float).T)
        else:
            X = np.asarray(states, dtype=float)
        single = X.ndim == 1
        X = np.atleast_2d(X)
        if X.shape[1] != self.model.n_states:
            raise ValueError(f"a state has {self.model.n_states} values, in the order "
                             f"{self.model.states}")
        return X, single

    def split_states(self, X):
        cont = X[:, self.cont_idx]
        if self.chain is None:
            return np.zeros(X.shape[0], dtype=int), cont
        M = X[:, self.markov_idx]
        match = np.all(np.isclose(M[:, None, :], self.node_values[None, :, :],
                                  rtol=1e-10, atol=1e-12), axis=2)
        if not np.all(match.any(axis=1)):
            raise ValueError(f"the values of {self.markov_names} must be nodes of their chains")
        return match.argmax(axis=1), cont

    # -- the policy ----------------------------------------------------------
    def coefficients(self, values):
        return np.stack([self.grid._coefficients(values[m], self.kind)
                         for m in range(self.n_nodes)])

    def policy(self, G, node, cont):
        out = np.empty((cont.shape[0], self.model.n_jumps))
        for m in np.unique(node):
            sel = node == m
            out[sel] = self.grid._evaluate(G[m], cont[sel], self.kind)
        return out

    def bounds_at(self, node, cont):
        P, nu = cont.shape[0], self.model.n_jumps
        lo, hi = np.full((P, nu), -np.inf), np.full((P, nu), np.inf)
        if not self.bound_specs:
            return lo, hi
        X = self.full_states(node, cont)
        view = VarView(X.T, self.state_index)
        for name, (low, high) in self.bound_specs.items():
            j = self.model.jumps.index(name)
            for target, b in ((lo, low), (hi, high)):
                if b is None:
                    continue
                if callable(b):
                    try:
                        v = np.broadcast_to(np.asarray(b(view, self.params), dtype=float), (P,))
                    except Exception:
                        v = np.array([float(b(VarView(x, self.state_index), self.params))
                                      for x in X])
                else:
                    v = np.full(P, float(b))
                target[:, j] = v
        if np.any(lo > hi):
            raise ValueError("a lower bound exceeds the upper bound at some grid point")
        return lo, hi

    # -- the residual on arrays ----------------------------------------------
    def residuals(self, Xn, X, E):
        """The model's residuals at ``P`` points: ``(P, n_vars)``, rows stacked."""
        if self._vectorised is not False:
            try:
                R = self._residuals_vectorised(Xn, X, E)
                if self._vectorised is None:
                    k = min(3, X.shape[0])
                    loop = self._residuals_loop(Xn[:k], X[:k], E[:k])
                    ok = np.allclose(R[:k], loop, rtol=1e-12, atol=1e-14, equal_nan=True)
                    self._vectorised = bool(ok)
                    if not ok:
                        return self._residuals_loop(Xn, X, E)
                return R
            except Exception:
                if self._vectorised:
                    raise
                self._vectorised = False
        return self._residuals_loop(Xn, X, E)

    def _residuals_vectorised(self, Xn, X, E):
        m = self.model
        nxt = VarView(Xn.T, m._index)
        cur = VarView(X.T, m._index)
        e = VarView(E.T, m._shock_index)
        r = (m.equations(nxt, cur, e, self.params, self.ctx) if m._wants_ctx
             else m.equations(nxt, cur, e, self.params))
        P = X.shape[0]
        rows = [np.broadcast_to(np.asarray(ri, dtype=float), (P,)) for ri in r]
        if len(rows) != m.n_vars:
            raise ValueError(f"{m.name}: equations returned {len(rows)} residuals for "
                             f"{m.n_vars} variables")
        return np.column_stack(rows)

    def _residuals_loop(self, Xn, X, E):
        m = self.model
        return np.array([m.residuals(xn, x, e, self.params, self.ctx)
                         for xn, x, e in zip(Xn, X, E)]).reshape(X.shape[0], m.n_vars)

    # -- sorting the rows ----------------------------------------------------
    def classify(self, U):
        """Sort the rows into transitions, exogenous laws and equilibrium conditions."""
        m = self.model
        ns, nv = m.n_states, m.n_vars
        rng = np.random.default_rng(12345)
        picks = rng.choice(U.shape[0], size=min(4, U.shape[0]), replace=False)
        X = np.column_stack([self.full_states(self.grid_nodes[picks], self.grid_cont[picks]),
                             U[picks]])
        E = np.tile(self.eps_nodes[len(self.eps_nodes) // 2], (picks.size, 1))
        base = self.residuals(X, X, E)
        dep = np.zeros((nv, nv), dtype=bool)          # dep[row, next-period variable]
        for j in range(nv):
            Xn = X.copy()
            Xn[:, j] += 1e-4 * (1.0 + np.abs(Xn[:, j]))
            dep[:, j] = np.any(self.residuals(Xn, X, E) != base, axis=0)
        endo_cols = [m.states.index(s) for s in self.endo]
        on_jump = dep[:, ns:].any(axis=1)
        on_endo = dep[:, endo_cols].any(axis=1) if endo_cols else np.zeros(nv, dtype=bool)
        on_markov = dep[:, self.markov_idx].any(axis=1) if self.markov_idx.size \
            else np.zeros(nv, dtype=bool)
        if self.chain is None:
            transition = ~on_jump & dep[:, :ns].any(axis=1)
            dropped = np.zeros(nv, dtype=bool)
        else:
            transition = ~on_jump & on_endo
            dropped = ~on_jump & ~on_endo & on_markov
        equilibrium = ~transition & ~dropped
        self.transition_rows = np.where(transition)[0]
        self.equilibrium_rows = np.where(equilibrium)[0]
        n_trans = len(self.endo) if self.chain is not None else ns
        if self.transition_rows.size != n_trans or self.equilibrium_rows.size != m.n_jumps \
                or dropped.sum() != self.markov_idx.size:
            raise ValueError(
                f"the rows split into {self.transition_rows.size} transitions (next-period states "
                f"and no next-period jump), {int(dropped.sum())} laws of the Markov states and "
                f"{self.equilibrium_rows.size} equilibrium conditions; time iteration needs "
                f"{n_trans}, {self.markov_idx.size} and {m.n_jumps}. Rows: transitions "
                f"{self.transition_rows.tolist()}, equilibrium {self.equilibrium_rows.tolist()}")
        self.solved_states = np.array([m.states.index(s)
                                       for s in (self.endo if self.chain is not None
                                                 else m.states)], dtype=int)
        self.affine = self._affine_transitions(X, E)

    def _affine_transitions(self, X, E):
        """The inverse of the transitions' Jacobian in the next states, if they are affine in them.

        Transitions are almost always written ``k' - g(k, u, eps)``, and then the next states
        follow from one evaluation instead of a Newton solve. Affine here means a Jacobian that
        is the same for two step sizes and at every sampled point.
        """
        cols, rows = self.solved_states, self.transition_rows
        Xn = X.copy()
        if self.markov_idx.size:
            Xn[:, self.markov_idx] = X[:, self.markov_idx]
        r0 = self.residuals(Xn, X, E)[:, rows]
        A = []
        for scale in (1e-2, 2e-2):
            Aj = np.empty((X.shape[0], rows.size, cols.size))
            for j, c in enumerate(cols):
                h = scale * (1.0 + np.abs(Xn[:, c]))
                Xp = Xn.copy()
                Xp[:, c] += h
                Aj[:, :, j] = (self.residuals(Xp, X, E)[:, rows] - r0) / h[:, None]
            A.append(Aj)
        if not (np.allclose(A[0], A[1], rtol=1e-9, atol=1e-12)
                and np.allclose(A[0], A[0][:1], rtol=1e-9, atol=1e-12)):
            return None
        try:
            return np.linalg.inv(A[0][0])
        except np.linalg.LinAlgError:
            return None

    # -- next period -----------------------------------------------------------
    def next_states(self, cont, node, u, nxt_node, eps, G):
        """The next period's continuous states, from the transitions, at ``P`` points."""
        X = np.column_stack([self.full_states(node, cont), u])
        Xn = X.copy()
        if self.markov_idx.size:
            Xn[:, self.markov_idx] = self.node_values[nxt_node]
        cols = self.solved_states
        rows = self.transition_rows
        x = Xn[:, cols].copy()
        if self.affine is not None:
            r = self.residuals(Xn, X, eps)[:, rows]
            Xn[:, cols] = x - r @ self.affine.T
            return Xn[:, self.cont_idx]
        for _ in range(30):
            Xn[:, cols] = x
            r = self.residuals(Xn, X, eps)[:, rows]
            if np.max(np.abs(r)) < 1e-13 * (1.0 + np.max(np.abs(x))):
                break
            J = np.empty((x.shape[0], rows.size, cols.size))
            for j, c in enumerate(cols):
                h = 1e-7 * (1.0 + np.abs(x[:, j]))
                Xp = Xn.copy()
                Xp[:, c] += h
                J[:, :, j] = (self.residuals(Xp, X, eps)[:, rows] - r) / h[:, None]
            step = np.linalg.solve(J, r[:, :, None])[:, :, 0]
            x = x - step
            if np.max(np.abs(step) / (1.0 + np.abs(x))) < 1e-14:
                Xn[:, cols] = x
                break
        else:
            Xn[:, cols] = x
        return Xn[:, self.cont_idx]

    def equilibrium(self, u, node, cont, G):
        """The equilibrium conditions at ``P`` points, integrated over the next period."""
        P = cont.shape[0]
        X = np.column_stack([self.full_states(node, cont), u])
        out = np.zeros((P, self.model.n_jumps))
        for j in range(self.n_nodes):
            pj = self.transition[node, j]
            if not np.any(pj > 0):
                continue
            nxt_node = np.full(P, j)
            for e, w in zip(self.eps_nodes, self.eps_weights):
                E = np.tile(e, (P, 1))
                cn = self.next_states(cont, node, u, nxt_node, E, G)
                un = self.policy(G, nxt_node, cn)
                Xn = np.column_stack([self.full_states(nxt_node, cn), un])
                out += (pj * w)[:, None] * self.residuals(Xn, X, E)[:, self.equilibrium_rows]
        return out

    # -- one step of time iteration ---------------------------------------------
    def solve_controls(self, U, G, lo, hi, tol, max_iter):
        node, cont = self.grid_nodes, self.grid_cont
        has_lo, has_hi = np.isfinite(lo), np.isfinite(hi)
        bounded = np.any(has_lo | has_hi)

        def phi(u):
            F = self.equilibrium(u, node, cont, G)
            return _fischer_burmeister(u, F, lo, hi, has_lo, has_hi) if bounded else F

        u = U.copy()
        r = phi(u)
        nu = u.shape[1]
        for _ in range(max_iter):
            norm = np.max(np.abs(r), axis=1)
            if np.max(norm) < tol:
                break
            J = np.empty((u.shape[0], nu, nu))
            for j in range(nu):
                h = 1e-7 * (1.0 + np.abs(u[:, j]))
                h = np.where(has_hi[:, j] & (u[:, j] + h > hi[:, j]), -h, h)
                up = u.copy()
                up[:, j] += h
                J[:, :, j] = (phi(up) - r) / h[:, None]
            try:
                step = np.linalg.solve(J, r[:, :, None])[:, :, 0]
            except np.linalg.LinAlgError:
                step = np.einsum("pij,pj->pi", np.linalg.pinv(J), r)
            # Backtracking, point by point, on the sum of squares.
            t = np.ones(u.shape[0])
            todo = norm >= tol
            new_u, new_r = u.copy(), r.copy()
            for _ in range(20):
                if not np.any(todo):
                    break
                trial = u.copy()
                trial[todo] = u[todo] - t[todo, None] * step[todo]
                if bounded:
                    # Keep each control within a bounded side: the Fischer–Burmeister function
                    # is defined beyond the bounds, but the model may not be (log c at c < 0).
                    trial = np.where(has_lo, np.maximum(trial, lo), trial)
                    trial = np.where(has_hi, np.minimum(trial, hi), trial)
                rt = phi(trial)
                better = np.sum(rt ** 2, axis=1) < np.sum(r ** 2, axis=1) * (1.0 - 1e-4 * t)
                better |= ~np.all(np.isfinite(r), axis=1)
                ok = todo & better & np.all(np.isfinite(rt), axis=1)
                new_u[ok], new_r[ok] = trial[ok], rt[ok]
                todo &= ~ok
                t[todo] *= 0.5
            if np.array_equal(new_u, u):
                break
            u, r = new_u, new_r
        return u

    # -- the starting policy -------------------------------------------------------
    def initial_policy(self, initial, guess, steady, positive):
        m = self.model
        X = self.full_states(self.grid_nodes, self.grid_cont)
        P = X.shape[0]
        view = VarView(X.T, self.state_index)

        def call(fn):
            try:
                out = np.asarray(fn(view, self.params), dtype=float)
                if out.ndim == 0:
                    out = np.full(P, float(out))
                return out
            except Exception:
                return np.array([np.asarray(fn(VarView(x, self.state_index), self.params),
                                            dtype=float) for x in X]).T

        if isinstance(initial, TimeIterationSolution):
            return initial.controls(X)
        if isinstance(initial, Mapping):
            missing = [j for j in m.jumps if j not in initial]
            if missing:
                raise KeyError(f"initial has no value for {missing}")
            return np.column_stack([
                call(initial[j]) if callable(initial[j]) else np.full(P, float(initial[j]))
                for j in m.jumps]).reshape(P, m.n_jumps)
        if callable(initial):
            out = call(initial)
            return np.broadcast_to(out.T if out.shape[0] == m.n_jumps and out.ndim == 2
                                   else out.reshape(P, -1), (P, m.n_jumps)).copy()
        if initial is not None:
            raise TypeError("initial must be a TimeIterationSolution, a function or a mapping")
        if guess is None and steady is None:
            raise ValueError("give initial, or guess or steady for the first-order policy to "
                             "start from")
        from .solve import solve
        sol = solve(m, self.params, guess=guess, steady=steady, positive=positive)
        self.ctx = sol.ctx
        ns = m.n_states
        khat = (X - sol.steady.x[:ns]) / sol.coords.scale[:ns]
        return sol.steady.x[ns:] + sol.coords.scale[ns:] * (khat @ sol.F.T)


def _fischer_burmeister(u, F, lo, hi, has_lo, has_hi):
    """``F`` for unbounded controls, its complementarity with the bounds for the others."""
    def fb(a, b):
        return a + b - np.sqrt(a * a + b * b)

    with np.errstate(invalid="ignore"):
        a = np.where(has_lo, u - lo, 0.0)
        b = np.where(has_hi, hi - u, 0.0)
        both = fb(a, -fb(b, -F))
        lower = fb(a, F)
        upper = -fb(b, -F)
    return np.where(has_lo & has_hi, both, np.where(has_lo, lower, np.where(has_hi, upper, F)))


def _gauss_hermite(Sigma, n):
    """Tensor Gauss–Hermite nodes and weights for ``N(0, Sigma)``."""
    vals, vecs = np.linalg.eigh(Sigma)
    vals = np.maximum(vals, 0.0)
    keep = vals > 1e-14 * max(vals.max(), 1e-300)
    x, w = np.polynomial.hermite.hermgauss(n)
    x, w = x * np.sqrt(2.0), w / np.sqrt(np.pi)
    k = int(keep.sum())
    if k == 0:
        return np.zeros((1, Sigma.shape[0])), np.ones(1)
    mesh = np.meshgrid(*([x] * k), indexing="ij")
    wmesh = np.meshgrid(*([w] * k), indexing="ij")
    Z = np.column_stack([g.ravel() for g in mesh])
    W = np.prod(np.column_stack([g.ravel() for g in wmesh]), axis=1)
    return Z @ (vecs[:, keep] * np.sqrt(vals[keep])).T, W


def _factor(Sigma):
    vals, vecs = np.linalg.eigh(Sigma)
    return vecs * np.sqrt(np.maximum(vals, 0.0))
