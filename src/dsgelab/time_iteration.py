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

Two other iterations reach the same fixed point on the same grid. :func:`improved_time_iteration`
applies Newton's method to the conditions of every grid point at once, with the derivative
through next period's policy, as dolo's ``improved_time_iteration`` does. It takes a few steps
where time iteration takes tens or hundreds, and it needs a start close enough to the solution.
:func:`expectations_iteration` iterates on the conditional expectations, declared as jumps, and
solves for the controls given them, which needs no integral inside the solve.

The policy is interpolated between the grid points, linearly or by a natural cubic spline along
each axis, and extrapolated linearly from the ends of the grid in both cases. On a
:class:`SmolyakGrid` it is the grid's polynomial instead, which takes more than three continuous
states.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike

from .markov import MarkovChain
from .model import Covariance, Model, Params, Point, VarView
from .smolyak import SmolyakGrid
from .solution import shock_covariance

__all__ = ["CartesianGrid", "TimeIterationSolution", "time_iteration",
           "improved_time_iteration", "expectations_iteration"]

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
    grid: CartesianGrid | SmolyakGrid
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


def time_iteration(model: Model, params: Params, *,
                   grid: CartesianGrid | SmolyakGrid | Mapping[str, Any],
                   markov: Mapping[str, MarkovChain] | None = None, cov: Covariance = None,
                   n_quad: int = 5, quadrature: tuple[ArrayLike, ArrayLike] | None = None,
                   bounds: Mapping[str, tuple[Bound, Bound]] | None = None,
                   interpolation: str | None = None,
                   initial: Any = None, guess: Point | None = None, steady: Any = None,
                   positive: Sequence[str] = (), tol: float = 1e-8, max_iter: int = 1000,
                   inner_tol: float = 1e-12, inner_max_iter: int = 50) -> TimeIterationSolution:
    """Solve ``model`` globally by time iteration.

    Parameters
    ----------
    grid :
        A :class:`CartesianGrid` of the continuous states, or the mapping that builds one, or a
        :class:`SmolyakGrid`. Every state not given by a Markov chain needs an axis.
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
        ``"linear"`` (the default) or ``"cubic"`` (a natural spline along each axis), on a
        :class:`CartesianGrid`. A :class:`SmolyakGrid` interpolates by its own polynomial and
        takes no value here.
    initial :
        The policy the iteration starts from: a :class:`TimeIterationSolution` (to refine on a new
        grid), ``initial(states, params)`` returning the jumps in model order, or a mapping from
        each jump to a number or such a function. ``None`` starts from the first-order
        perturbation, which needs ``guess`` or ``steady``.
    tol :
        The iteration stops when no control moves by more than ``tol`` at any grid point.
    """
    pb, U, lo, hi = _setup(model, params, grid, markov, cov, n_quad, quadrature, bounds,
                           interpolation, initial, guess, steady, positive)
    history: list[float] = []
    distance = np.inf
    it = 0
    while it < max_iter:
        it += 1
        G = pb.coefficients(U.reshape(pb.n_nodes, pb.grid.n, -1))
        U_new = pb.solve_controls(U, G, lo, hi, inner_tol, inner_max_iter)
        distance = float(np.max(np.abs(U_new - U)))
        history.append(distance)
        U = U_new
        if not np.isfinite(distance):
            break
        if distance < tol:
            break
    return _solution(pb, U, lo, hi, it, distance, tol, history)


def improved_time_iteration(model: Model, params: Params, *,
                            grid: CartesianGrid | SmolyakGrid | Mapping[str, Any],
                            markov: Mapping[str, MarkovChain] | None = None,
                            cov: Covariance = None, n_quad: int = 5,
                            quadrature: tuple[ArrayLike, ArrayLike] | None = None,
                            bounds: Mapping[str, tuple[Bound, Bound]] | None = None,
                            interpolation: str | None = None, initial: Any = None,
                            guess: Point | None = None, steady: Any = None,
                            positive: Sequence[str] = (), tol: float = 1e-8, max_iter: int = 100,
                            linear_tol: float = 1e-10,
                            linear_max_iter: int = 1000) -> TimeIterationSolution:
    """Solve ``model`` globally by improved time iteration (Newton's method on the whole grid).

    Time iteration holds next period's policy fixed while it solves for today's controls, and
    converges at the rate at which a change in the policy feeds back through next period. Improved
    time iteration (dolo's ``improved_time_iteration``) applies Newton's method to the equilibrium
    conditions of every grid point at once, with the policy for next period interpolated from the
    unknowns themselves. The Jacobian has a block for each point, the derivative with next period's
    policy held fixed, and a second term, the derivative through next period's policy, which is
    linear in the policy's values because the interpolant is. That term is applied as an operator,
    without forming the matrix, and each Newton step is solved by GMRES, preconditioned by the
    blocks of the first term. A step that does not reduce the sum of squares of the conditions is
    halved, up to nine times as in dolo, and when none does, the iteration takes one step of time
    iteration instead. Bounds are imposed by the same Fischer–Burmeister complementarity as in
    :func:`time_iteration`, and the fixed point on the grid is the same.

    Newton's method converges from a policy close enough to the solution, and then in a few
    steps. From a distant one, with a bound that binds on part of the grid, it may not converge.
    On dolo's consumption and saving model, with a borrowing limit, it takes 9 steps from
    ``c = min(w, 1)`` and 35 to 46 from ``c = 0.9 w``. With cash on hand from 0.5 instead of
    0.01, on 40 points, it takes 6 to 8 steps from the first start and does not converge within
    100 from the second. A solution of :func:`time_iteration` with a loose ``tol``, given as
    ``initial``, is a start that avoids this.

    The parameters are those of :func:`time_iteration`, and so is the result. ``iterations``
    counts the Newton steps.

    Parameters
    ----------
    tol :
        The iteration stops when a step moves no control by more than ``tol`` at any grid point.
    linear_tol, linear_max_iter :
        GMRES stops at a residual of ``linear_tol`` relative to the right-hand side, or after
        ``linear_max_iter`` iterations.
    """
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1")
    pb, U, lo, hi = _setup(model, params, grid, markov, cov, n_quad, quadrature, bounds,
                           interpolation, initial, guess, steady, positive)
    shape = (pb.n_nodes, pb.grid.n, model.n_jumps)
    has_lo, has_hi = np.isfinite(lo), np.isfinite(hi)
    history: list[float] = []
    distance = np.inf
    it = 0
    G = pb.coefficients(U.reshape(shape))
    r = pb.phi(U, G, lo, hi)
    while it < max_iter:
        it += 1
        step = pb.newton_direction(U, G, r, lo, hi, linear_tol, linear_max_iter)
        base = float(np.sum(r ** 2))
        t = 1.0
        accepted = False
        for _ in range(10):
            trial = U - t * step
            trial = np.where(has_lo, np.maximum(trial, lo), trial)
            trial = np.where(has_hi, np.minimum(trial, hi), trial)
            G_trial = pb.coefficients(trial.reshape(shape))
            r_trial = pb.phi(trial, G_trial, lo, hi)
            if np.all(np.isfinite(r_trial)) and (
                    float(np.sum(r_trial ** 2)) < base * (1.0 - 1e-4 * t) or not np.isfinite(base)):
                accepted = True
                break
            t *= 0.5
        if not accepted:
            trial = pb.solve_controls(U, G, lo, hi, 1e-12, 50)
            G_trial = pb.coefficients(trial.reshape(shape))
            r_trial = pb.phi(trial, G_trial, lo, hi)
        distance = float(np.max(np.abs(trial - U)))
        history.append(distance)
        U, G, r = trial, G_trial, r_trial
        if not np.isfinite(distance) or distance < tol:
            break
    return _solution(pb, U, lo, hi, it, distance, tol, history)


def expectations_iteration(model: Model, params: Params, *, expectations: Sequence[str],
                           grid: CartesianGrid | SmolyakGrid | Mapping[str, Any],
                           markov: Mapping[str, MarkovChain] | None = None,
                           cov: Covariance = None, n_quad: int = 5,
                           quadrature: tuple[ArrayLike, ArrayLike] | None = None,
                           bounds: Mapping[str, tuple[Bound, Bound]] | None = None,
                           interpolation: str | None = None, initial: Any = None,
                           guess: Point | None = None, steady: Any = None,
                           positive: Sequence[str] = (), damping: float = 0.5,
                           tol: float = 1e-8, max_iter: int = 5000, inner_tol: float = 1e-12,
                           inner_max_iter: int = 50) -> TimeIterationSolution:
    """Solve ``model`` globally by iterating on the conditional expectations of its Euler equations.

    The model declares each expectation as a jump, named in ``expectations``, defined by an
    equilibrium condition with next-period variables, such as ``z - beta E[c'^(-gamma) R']``,
    and used in conditions with nothing dated ``t+1``, such as ``c^(-gamma) - z``. This is
    dolo's ``fgh`` form, ``0 = f(s, x, E[h(s', x')])``, in the conditions of a :class:`Model`.
    Each iteration takes the policy found so far, integrates the expectations at every grid
    point with today's controls and next period's policy both taken from it, and then solves the
    conditions without ``t+1`` for today's controls, given the expectations, point by point.
    The second step integrates nothing, and the first solves only for the expectations, in one
    step when their conditions are affine in them, whereas time iteration integrates in every
    evaluation of its Newton solve. This is the iteration that dolo's documentation describes as
    parameterized expectations, on a grid (Christiano and Fisher, 2000). dolo 0.4.9.20 documents
    it and does not implement it.

    The new expectations are averaged with the old ones, with weight ``damping`` on the new.
    Without it, the iteration can diverge. On stochastic growth with log utility and full
    depreciation, ``c = 1/z`` and ``z = alpha beta / ((1 - alpha beta) k')``, with ``k'`` today's
    output less ``c``, so a change in ``c`` changes the next ``c`` by about
    ``-(1 - alpha beta) / (alpha beta)`` times as much, -1.8 at ``alpha = 0.36`` and
    ``beta = 0.99``. With ``damping`` the factor is about ``1 - damping / (alpha beta)``, -0.4 at
    the default of 0.5.

    The conditions are paired with the jumps by position, as in :func:`time_iteration`: the
    condition in the position of an expectation must involve ``t+1``, and the condition in the
    position of a control must not. Bounds apply to the controls, by the same complementarity.
    At its fixed point, the policy solves the same equations on the same grid as
    :func:`time_iteration` on the same model, and the two agree. The other parameters are those
    of :func:`time_iteration`, and so is the result.

    Parameters
    ----------
    expectations :
        The jumps that hold the expectations.
    damping :
        The weight of the new expectations against the old ones, in ``(0, 1]``; 1 is the
        iteration without damping.
    tol :
        The iteration stops when no jump moves by more than ``tol`` at any grid point.
    inner_tol, inner_max_iter :
        The tolerance and the limit on Newton steps in the solve for the controls.
    """
    if not 0.0 < damping <= 1.0:
        raise ValueError("damping must lie in (0, 1]")
    names = list(expectations)
    unknown = [z for z in names if z not in model.jumps]
    if unknown:
        raise KeyError(f"expectations {unknown} are not jumps of the model")
    if not names or len(set(names)) != len(names):
        raise ValueError("expectations needs at least one jump, each named once")
    if len(names) == model.n_jumps:
        raise ValueError("every jump is an expectation: some jump must be a control")
    bounded = [z for z in names if z in (bounds or {})]
    if bounded:
        raise ValueError(f"bounds apply to the controls, not to the expectations {bounded}")
    pb, U, lo, hi = _setup(model, params, grid, markov, cov, n_quad, quadrature, bounds,
                           interpolation, initial, guess, steady, positive)
    zi = np.array([model.jumps.index(z) for z in names], dtype=int)
    xi = np.array([j for j in range(model.n_jumps) if j not in zi], dtype=int)
    forward = pb.forward[pb.equilibrium_rows]
    wrong = ([model.jumps[j] for j in zi if not forward[j]]
             + [model.jumps[j] for j in xi if forward[j]])
    if wrong:
        raise ValueError(
            f"the conditions in the positions of {wrong} do not fit: an expectation's condition "
            "must involve t+1, and a control's must not")
    shape = (pb.n_nodes, pb.grid.n, model.n_jumps)
    history: list[float] = []
    distance = np.inf
    it = 0
    while it < max_iter:
        it += 1
        G = pb.coefficients(U.reshape(shape))
        try:
            Z = pb.solve_expectations(U, G, zi, inner_tol, inner_max_iter)
        except np.linalg.LinAlgError:
            # a policy far enough from the solution that next period cannot be computed: the
            # iteration has diverged
            history.append(np.nan)
            distance = np.nan
            break
        new = U.copy()
        new[:, zi] = U[:, zi] + damping * (Z - U[:, zi])
        new[:, xi] = pb.solve_static(new, xi, lo[:, xi], hi[:, xi], inner_tol, inner_max_iter)
        distance = float(np.max(np.abs(new - U)))
        history.append(distance)
        U = new
        if not np.isfinite(distance) or distance < tol:
            break
    return _solution(pb, U, lo, hi, it, distance, tol, history)


def _setup(model, params, grid, markov, cov, n_quad, quadrature, bounds, interpolation, initial,
           guess, steady, positive):
    """The problem, the starting policy within the bounds, and the bounds at the grid points."""
    if isinstance(grid, SmolyakGrid):
        if interpolation is not None:
            raise ValueError("a SmolyakGrid interpolates by its own polynomial; leave "
                             "interpolation unset")
        interpolation = "smolyak"
    else:
        if not isinstance(grid, CartesianGrid):
            grid = CartesianGrid(grid)
        interpolation = "linear" if interpolation is None else interpolation
    pb = _Problem(model, params, grid, dict(markov or {}), cov, n_quad, quadrature,
                  dict(bounds or {}), interpolation)
    U = pb.initial_policy(initial, guess, steady, positive)
    lo, hi = pb.bounds_at(pb.grid_nodes, pb.grid_cont)
    U = np.clip(U, lo, hi)
    pb.classify(U)
    return pb, U, lo, hi


def _solution(pb, U, lo, hi, it, distance, tol, history):
    shape = (pb.n_nodes, pb.grid.n, pb.model.n_jumps)
    return TimeIterationSolution(
        model=pb.model, params=pb.params, grid=pb.grid, chain=pb.chain,
        markov_states=pb.markov_names, values=U.reshape(shape), interpolation=pb.kind,
        iterations=it, converged=bool(distance < tol), distance=distance, history=history,
        lower=lo.reshape(shape), upper=hi.reshape(shape), _problem=pb)


class _Problem:
    """The pieces of one time-iteration problem, on arrays with one row per point."""

    def __init__(self, model, params, grid, markov, cov, n_quad, quadrature, bounds,
                 interpolation):
        self.model, self.params, self.grid = model, params, grid
        self.kind = interpolation
        if interpolation not in ("linear", "cubic") and not (
                interpolation == "smolyak" and isinstance(grid, SmolyakGrid)):
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
        self.chain, self.node_values = _markov_product([markov[s] for s in self.markov_names])
        self.n_nodes = self.node_values.shape[0]
        self.transition = self.chain.transition if self.chain is not None else np.ones((1, 1))
        self.eps_nodes, self.eps_weights, self.chol = _shock_rule(
            model.shocks, cov, n_quad, quadrature, bool(markov))
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
        self._z_jacobian = None

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
        self.forward = dep.any(axis=1)          # rows with something dated t+1
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

        return _newton_points(phi, U, lo, hi, tol, max_iter)

    # -- improved time iteration -------------------------------------------------
    def phi(self, U, G, lo, hi):
        """The equilibrium conditions at the grid points, in complementarity form."""
        F = self.equilibrium(U, self.grid_nodes, self.grid_cont, G)
        has_lo, has_hi = np.isfinite(lo), np.isfinite(hi)
        if not np.any(has_lo | has_hi):
            return F
        return _fischer_burmeister(U, F, lo, hi, has_lo, has_hi)

    def newton_direction(self, U, G, r, lo, hi, linear_tol, linear_max_iter):
        """The Newton step for the conditions of every grid point, next period's policy included.

        ``phi(U)`` depends on ``U`` at each point directly and through the policy interpolated
        from ``U`` for next period. The first derivative is a block for each point, ``A``, by
        finite differences with the policy fixed. The second is ``D sum_j w_j J_j W_j``: ``W_j``
        evaluates the policy at the next states of quadrature node ``j``, which is linear in
        ``U``, ``J_j`` is the derivative of the conditions in next period's jumps there, and
        ``D`` the derivative of the complementarity in the conditions.
        """
        from scipy.sparse.linalg import LinearOperator

        node, cont = self.grid_nodes, self.grid_cont
        P, nu = U.shape
        ns = self.model.n_states
        has_lo, has_hi = np.isfinite(lo), np.isfinite(hi)
        bounded = np.any(has_lo | has_hi)
        A = np.empty((P, nu, nu))
        for j in range(nu):
            h = 1e-7 * (1.0 + np.abs(U[:, j]))
            h = np.where(has_hi[:, j] & (U[:, j] + h > hi[:, j]), -h, h)
            up = U.copy()
            up[:, j] += h
            A[:, :, j] = (self.phi(up, G, lo, hi) - r) / h[:, None]
        if bounded:
            F = self.equilibrium(U, node, cont, G)
            hF = 1e-7 * (1.0 + np.abs(F))
            D = (_fischer_burmeister(U, F + hF, lo, hi, has_lo, has_hi)
                 - _fischer_burmeister(U, F, lo, hi, has_lo, has_hi)) / hF
        else:
            D = np.ones((P, nu))
        X = np.column_stack([self.full_states(node, cont), U])
        terms = []
        for jn in range(self.n_nodes):
            pj = self.transition[node, jn]
            if not np.any(pj > 0):
                continue
            nxt = np.full(P, jn)
            for e, w in zip(self.eps_nodes, self.eps_weights):
                E = np.tile(e, (P, 1))
                cn = self.next_states(cont, node, U, nxt, E, G)
                Xn = np.column_stack([self.full_states(nxt, cn), self.policy(G, nxt, cn)])
                base = self.residuals(Xn, X, E)[:, self.equilibrium_rows]
                J = np.empty((P, nu, nu))
                for k in range(nu):
                    h = 1e-7 * (1.0 + np.abs(Xn[:, ns + k]))
                    Xp = Xn.copy()
                    Xp[:, ns + k] += h
                    J[:, :, k] = (self.residuals(Xp, X, E)[:, self.equilibrium_rows]
                                  - base) / h[:, None]
                terms.append((nxt, cn, (pj * w)[:, None, None] * J * D[:, :, None]))
        try:
            Ainv = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            Ainv = np.linalg.pinv(A)
        shape = (self.n_nodes, self.grid.n, nu)

        def matvec(v):
            dU = np.asarray(v).reshape(P, nu)
            C = self.coefficients(dU.reshape(shape))
            out = np.zeros((P, nu))
            for nxt, cn, M in terms:
                out += np.einsum("pij,pj->pi", M, self.policy(C, nxt, cn))
            return (dU + np.einsum("pij,pj->pi", Ainv, out)).ravel()

        rhs = np.einsum("pij,pj->pi", Ainv, r).ravel()
        if not np.all(np.isfinite(rhs)) or not np.any(rhs):
            return np.where(np.isfinite(rhs), rhs, 0.0).reshape(P, nu)
        op = LinearOperator((P * nu, P * nu), matvec=matvec, dtype=float)
        return _gmres(op, rhs, linear_tol, linear_max_iter).reshape(P, nu)

    # -- expectations iteration --------------------------------------------------
    def solve_expectations(self, U, G, zi, tol, max_iter):
        """The expectations at the grid points, with today's controls and the policy from ``U``.

        The conditions in the expectations' positions are solved for the expectations, with
        everything else fixed. They are almost always affine in them, ``z - E[h]``, and then
        one Newton step solves them exactly; the step is repeated until it is below ``tol``.
        """
        node, cont = self.grid_nodes, self.grid_cont
        rows = zi       # the columns of equilibrium() follow the jumps they are paired with
        u = U.copy()
        for _ in range(max_iter):
            F = self.equilibrium(u, node, cont, G)[:, rows]
            if not np.all(np.isfinite(F)):
                return np.full((u.shape[0], zi.size), np.nan)
            if self._z_jacobian is None:
                J = np.empty((u.shape[0], zi.size, zi.size))
                for k, j in enumerate(zi):
                    h = 1e-6 * (1.0 + np.abs(u[:, j]))
                    up = u.copy()
                    up[:, j] += h
                    J[:, :, k] = (self.equilibrium(up, node, cont, G)[:, rows] - F) / h[:, None]
                if np.allclose(J, J[:1], rtol=1e-6, atol=1e-9) and np.allclose(
                        J[0], np.round(J[0]), rtol=0, atol=1e-6):
                    self._z_jacobian = np.round(J[0])     # z - E[h], the same at every point
            else:
                J = np.broadcast_to(self._z_jacobian, (u.shape[0], zi.size, zi.size))
            step = np.linalg.solve(J, F[:, :, None])[:, :, 0]
            u[:, zi] -= step
            if np.max(np.abs(step) / (1.0 + np.abs(u[:, zi]))) < tol or \
                    self._z_jacobian is not None:
                break
        return u[:, zi]

    def solve_static(self, U, xi, lo, hi, tol, max_iter):
        """The controls that solve the conditions without ``t+1``, the expectations fixed."""
        node, cont = self.grid_nodes, self.grid_cont
        rows = self.equilibrium_rows[xi]
        S = self.full_states(node, cont)
        has_lo, has_hi = np.isfinite(lo), np.isfinite(hi)
        bounded = np.any(has_lo | has_hi)
        P = U.shape[0]

        def phi(x):
            u = U.copy()
            u[:, xi] = x
            X = np.column_stack([S, u])
            F = np.zeros((P, xi.size))
            for e, w in zip(self.eps_nodes, self.eps_weights):
                F += w * self.residuals(X, X, np.tile(e, (P, 1)))[:, rows]
            return _fischer_burmeister(x, F, lo, hi, has_lo, has_hi) if bounded else F

        return _newton_points(phi, U[:, xi], lo, hi, tol, max_iter)

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


def _newton_points(phi, U, lo, hi, tol, max_iter):
    """Newton's method point by point on ``phi``, with a Jacobian block for each row of ``U``.

    ``phi`` maps the unknowns, one row per point, to as many conditions; the rows are solved
    independently, with backtracking on each row's sum of squares and the unknowns kept within
    their finite bounds.
    """
    has_lo, has_hi = np.isfinite(lo), np.isfinite(hi)
    bounded = np.any(has_lo | has_hi)
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


def _gmres(op, b, rtol, max_iter):
    """GMRES with restarts every 50 iterations, by scipy's keyword for the relative tolerance."""
    import inspect

    from scipy.sparse.linalg import gmres

    restart = min(b.size, 50)
    cycles = max(1, -(-max_iter // restart))
    key = "rtol" if "rtol" in inspect.signature(gmres).parameters else "tol"
    x, _ = gmres(op, b, x0=b.copy(), restart=restart, maxiter=cycles, atol=0.0,
                 **{key: rtol})
    return x


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


def _markov_product(chains):
    """The product of independent chains, the last varying fastest, and its node values.

    Returns ``(chain, values)`` with ``values`` of shape ``(n_nodes, len(chains))``; with no
    chain, ``(None, an empty row)``, so that the code that loops over nodes sees one node.
    """
    if not chains:
        return None, np.zeros((1, 0))
    P = chains[0].transition
    vals = chains[0].nodes[:, None]
    for c in chains[1:]:
        P = np.kron(P, c.transition)
        vals = np.column_stack([np.repeat(vals, c.n, axis=0), np.tile(c.nodes, vals.shape[0])])
    chain = MarkovChain(np.arange(P.shape[0], dtype=float), P) if len(chains) > 1 else chains[0]
    return chain, vals


def _shock_rule(shocks, cov, n_quad, quadrature, has_markov):
    """Nodes and weights for the shocks, one row per node, and a factor of their covariance.

    With a Markov chain and neither ``cov`` nor ``quadrature``, or with no shocks, the shocks
    are zero and the factor is ``None``.
    """
    n = len(shocks)
    if quadrature is not None:
        nodes, weights = quadrature
        nodes = np.asarray(nodes, dtype=float).reshape(-1, n)
        weights = np.asarray(weights, dtype=float).ravel()
        if weights.size != nodes.shape[0] or not np.isclose(weights.sum(), 1.0):
            raise ValueError("quadrature needs one weight per node, summing to one")
        mean = weights @ nodes
        Sigma = (nodes - mean).T @ ((nodes - mean) * weights[:, None])
        return nodes, weights, _factor(Sigma)
    if has_markov and cov is None or n == 0:
        return np.zeros((1, n)), np.ones(1), None
    Sigma = shock_covariance(shocks, cov)
    nodes, weights = _gauss_hermite(Sigma, n_quad)
    return nodes, weights, _factor(Sigma)


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
