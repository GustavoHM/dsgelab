"""Value function iteration on a Cartesian grid, with Howard's improvement steps.

A :class:`DynamicProgram` declares the problem the Bellman equation

    V(s) = max_x  r(s, x) + beta E[V(s')],   s' = g(s, x, e'),   lower(s) <= x <= upper(s)

solves: the states and controls, the reward ``r``, the transition ``g``, the discount factor and
the bounds on the controls. The perturbation's :class:`Model` holds equilibrium conditions, and
a Bellman equation needs these other objects, so the program is declared apart from it.

The value function is kept at the points of a :class:`CartesianGrid` of the continuous states,
one copy per node of a Markov chain when some states follow one, and interpolated between the
points, linearly or by a natural cubic spline along each axis. Each iteration applies the
Bellman operator once, maximizing over the controls at every grid point, and then holds the
policy it found and applies the policy's own operator ``howard`` times (Howard, 1960), which
costs no maximization. With ``howard=0`` this is plain value function iteration; as ``howard``
grows it approaches policy iteration. The iteration stops when one application of the Bellman
operator moves the value by less than ``tol`` at every grid point, so the value is then within
``beta tol / (1 - beta)`` of the operator's fixed point on the grid.

The maximization at each point is a golden-section search when there is one control, and a
projected Newton method (Bertsekas, 1982) with derivatives by finite differences when there are
several, both inside the bounds, which must then be finite. With ``choices``, each control takes
one of a finite set of values and every combination is tried, which is the problem QuantEcon's
``DiscreteDP`` solves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike

from .markov import MarkovChain
from .model import Covariance, Params, Point, VarView, get_param
from .time_iteration import Bound, CartesianGrid, _markov_product, _shock_rule

__all__ = ["DynamicProgram", "ValueIterationSolution", "value_iteration"]

_GOLDEN = (np.sqrt(5.0) - 1.0) / 2.0


@dataclass
class DynamicProgram:
    """A dynamic program in Bellman form.

    Parameters
    ----------
    states : sequence of str
        The state variables. Exogenous states that follow a Markov chain belong here, and so do
        the ones driven by shocks.
    controls : sequence of str
        The choice variables.
    reward : callable
        ``reward(s, x, params)``, the period reward, with ``s`` and ``x`` name-indexed views of
        the states and controls.
    transition : callable
        ``transition(s, x, e, params)``, next period's states: a mapping from each state to its
        value, or a sequence in the order of ``states``. ``e`` holds what is drawn between the
        two periods: the shocks, and the next values of the states that follow a Markov chain,
        under their own names. Those states take their next value from the chain, so the
        transition's entry for them, if any, is not used.
    discount : float, str or callable
        The discount factor: a number, the name of a parameter, or ``discount(params)``.
    shocks : sequence of str, optional
        The shocks, integrated as in :func:`time_iteration`.
    bounds : mapping, optional
        ``{control: (lower, upper)}``, each bound a number, ``None``, or a function of today's
        states and the parameters.
    name : str, optional
    """

    states: Sequence[str]
    controls: Sequence[str]
    reward: Callable
    transition: Callable
    discount: float | str | Callable
    shocks: Sequence[str] = ()
    bounds: Mapping[str, tuple[Bound, Bound]] | None = None
    name: str = "program"

    def __post_init__(self) -> None:
        self.states = list(self.states)
        self.controls = list(self.controls)
        self.shocks = list(self.shocks)
        self.bounds = dict(self.bounds or {})
        if not self.states or not self.controls:
            raise ValueError(f"{self.name}: a program needs at least one state and one control")
        names = self.states + self.controls + self.shocks
        repeated = sorted({n for n in names if names.count(n) > 1})
        if repeated:
            raise ValueError(f"{self.name}: names used more than once: {repeated}")
        unknown = [b for b in self.bounds if b not in self.controls]
        if unknown:
            raise KeyError(f"{self.name}: bounds given for {unknown}, which are not controls")

    def beta(self, params: Params) -> float:
        """The discount factor at ``params``."""
        d = self.discount
        value = d(params) if callable(d) else get_param(params, d) if isinstance(d, str) else d
        value = float(value)
        if not 0.0 <= value < 1.0:
            raise ValueError(f"{self.name}: the discount factor must lie in [0, 1), not {value}")
        return value

    def __repr__(self) -> str:
        return (f"DynamicProgram({self.name!r}, states={self.states}, "
                f"controls={self.controls}, shocks={self.shocks})")


@dataclass
class ValueIterationSolution:
    """The value function and the policy on a grid of the continuous states.

    ``values[m]`` holds the value at every grid point when the chain is at node ``m``, in the
    order of ``grid.points``, and ``policy[m]`` the controls there, one row per point. Without a
    Markov chain there is one node.
    """

    program: DynamicProgram
    params: Params
    grid: CartesianGrid
    chain: MarkovChain | None
    markov_states: list[str]
    values: np.ndarray                 # (n_nodes, grid.n)
    policy: np.ndarray                 # (n_nodes, grid.n, n_controls)
    interpolation: str
    discount: float
    iterations: int
    converged: bool
    distance: float
    history: list[float] = field(repr=False)
    _problem: Any = field(repr=False, default=None)

    def value(self, states: Point) -> np.ndarray | float:
        """The value at one state or at a row-stacked array of states, in the program's order.

        A state given by a Markov chain must take one of the chain's node values.
        """
        pb = self._problem
        S, single = pb.states_array(states)
        node, cont = pb.split_states(S)
        out = pb.interpolate(pb.coefficients(self.values[..., None]), node, cont)[:, 0]
        return float(out[0]) if single else out

    def controls(self, states: Point) -> np.ndarray:
        """The policy, interpolated between the grid points, at one state or at several."""
        pb = self._problem
        S, single = pb.states_array(states)
        node, cont = pb.split_states(S)
        out = pb.interpolate(pb.coefficients(self.policy), node, cont)
        return out[0] if single else out

    __call__ = controls

    def bellman_errors(self, states: Point | None = None) -> np.ndarray:
        """``T V - V`` at ``states``, with ``T`` the Bellman operator, one entry per state.

        At the grid points this is at most the final ``distance``. Between them it measures how
        far the interpolated value is from satisfying the Bellman equation. By default the states
        are the centres of the grid's cells at every node of the chain, the points farthest from
        the grid.
        """
        pb = self._problem
        if states is None:
            mids = self.grid.midpoints
            node = np.repeat(np.arange(pb.n_nodes), mids.shape[0])
            cont = np.tile(mids, (pb.n_nodes, 1))
        else:
            S, _ = pb.states_array(states)
            node, cont = pb.split_states(S)
        G = pb.coefficients(self.values[..., None])
        start = pb.interpolate(pb.coefficients(self.policy), node, cont)
        _, best = pb.maximize(G, node, cont, start)
        return best - pb.interpolate(G, node, cont)[:, 0]

    def simulate(self, periods: int, *, start: Point | None = None, seed: int | None = None,
                 shocks: ArrayLike | None = None) -> dict[str, np.ndarray]:
        """Simulate the policy for ``periods`` periods, from ``start`` (default: the grid's centre).

        The shocks are drawn from the covariance the solution was built with, or given in
        ``shocks``, one row per period after the first; a Markov state follows its chain. Returns
        ``states`` and ``controls``, one row per period in the program's order, ``rewards``, and
        ``nodes``, the chain's index each period.
        """
        pb = self._problem
        rng = np.random.default_rng(seed)
        if start is None:
            S0 = np.empty(len(self.program.states))
            S0[pb.cont_idx] = [0.5 * (a[0] + a[-1]) for a in self.grid.axes]
            if pb.chain is not None:
                S0[pb.markov_idx] = pb.node_values[pb.chain.n // 2]
        else:
            S0, single = pb.states_array(start)
            if not single:
                raise ValueError("start must be a single state")
            S0 = S0[0]
        node, cont = pb.split_states(S0[None, :])
        if pb.chain is not None:
            path = pb.chain.simulate(periods, start=int(node[0]), seed=rng, indices=True)
        else:
            path = np.zeros(periods, dtype=int)
        n_shocks = len(self.program.shocks)
        steps = max(periods - 1, 0)
        if shocks is None:
            eps = (rng.standard_normal((steps, n_shocks)) @ pb.chol.T if pb.chol is not None
                   else np.zeros((steps, n_shocks)))
        else:
            eps = np.asarray(shocks, dtype=float).reshape(steps, n_shocks)
        G = pb.coefficients(self.policy)
        states = np.empty((periods, len(self.program.states)))
        controls = np.empty((periods, len(self.program.controls)))
        for t in range(periods):
            x = pb.interpolate(G, node, cont)
            S = pb.full_states(node, cont)
            states[t], controls[t] = S[0], x[0]
            if t == periods - 1:
                break
            nxt = int(path[t + 1])
            cont = pb.next_states(S, x, eps[t][None, :], pb.node_values[[nxt]])
            node = np.array([nxt])
        rewards = pb.reward(states, controls)
        return {"states": states, "controls": controls, "rewards": rewards, "nodes": path}


def value_iteration(program: DynamicProgram, params: Params, *,
                    grid: CartesianGrid | Mapping[str, Any],
                    markov: Mapping[str, MarkovChain] | None = None, cov: Covariance = None,
                    n_quad: int = 5, quadrature: tuple[ArrayLike, ArrayLike] | None = None,
                    interpolation: str = "linear",
                    choices: Mapping[str, ArrayLike] | None = None, initial: Any = None,
                    howard: int = 20, tol: float = 1e-8, max_iter: int = 10000,
                    inner_tol: float = 1e-10, inner_max_iter: int = 100) -> ValueIterationSolution:
    """Solve ``program`` by value function iteration with Howard's improvement steps.

    Parameters
    ----------
    grid :
        A :class:`CartesianGrid` of the continuous states, or the mapping that builds one. Every
        state not given by a Markov chain needs an axis.
    markov :
        Exogenous states given by Markov chains, by name, as in :func:`time_iteration`.
    cov, n_quad, quadrature :
        The shocks' integration rule, as in :func:`time_iteration`.
    interpolation :
        ``"linear"`` or ``"cubic"`` (a natural spline along each axis), for the value function
        and for the policy between the grid points.
    choices :
        ``{control: values}`` for every control, to choose among finitely many values instead of
        over an interval. Combinations outside the bounds, or with a reward that is not finite,
        are not chosen.
    initial :
        The value the iteration starts from: a number, ``initial(states, params)`` returning the
        value at a row-stacked array of states, or a :class:`ValueIterationSolution` (to refine
        on a new grid, whose policy then also starts the maximization). ``None`` starts from
        zero.
    howard :
        The number of times the policy's own operator is applied after each maximization.
    tol :
        The iteration stops when the Bellman operator moves the value by less than ``tol`` at
        every grid point.
    inner_tol, inner_max_iter :
        The maximization's tolerance on the controls, relative to the width of the bounds for a
        single control and to ``1 + |x|`` for several, and its limit on iterations.
    """
    if not isinstance(grid, CartesianGrid):
        grid = CartesianGrid(grid)
    if howard < 0:
        raise ValueError("howard must be zero or positive")
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1")
    pb = _Program(program, params, grid, dict(markov or {}), cov, n_quad, quadrature,
                  interpolation, choices, inner_tol, inner_max_iter)
    node, cont = pb.grid_nodes, pb.grid_cont
    V = pb.initial_values(initial)
    X: np.ndarray | None = None
    if isinstance(initial, ValueIterationSolution) and initial.policy.shape[2] == pb.nc:
        X = initial.controls(pb.full_states(node, cont))
    history: list[float] = []
    distance = np.inf
    it = 0
    while it < max_iter:
        it += 1
        G = pb.coefficients(V.reshape(pb.n_nodes, grid.n, 1))
        X, TV = pb.maximize(G, node, cont, X)
        distance = float(np.max(np.abs(TV - V)))
        history.append(distance)
        V = TV
        if not np.isfinite(distance) or distance < tol:
            break
        if howard:
            V = pb.evaluate(V, node, cont, X, howard)
    assert X is not None   # max_iter >= 1, so the loop has run
    return ValueIterationSolution(
        program=program, params=params, grid=grid, chain=pb.chain, markov_states=pb.markov_names,
        values=V.reshape(pb.n_nodes, grid.n), policy=X.reshape(pb.n_nodes, grid.n, pb.nc),
        interpolation=interpolation, discount=pb.beta, iterations=it,
        converged=bool(distance < tol), distance=distance, history=history, _problem=pb)


class _Program:
    """The pieces of one value-iteration problem, on arrays with one row per point."""

    def __init__(self, program, params, grid, markov, cov, n_quad, quadrature, interpolation,
                 choices, inner_tol, inner_max_iter):
        self.program, self.params, self.grid = program, params, grid
        if interpolation not in ("linear", "cubic"):
            raise ValueError(f"interpolation must be 'linear' or 'cubic', not {interpolation!r}")
        self.kind = interpolation
        self.beta = program.beta(params)
        self.inner_tol, self.inner_max_iter = inner_tol, inner_max_iter
        states = program.states
        unknown = [s for s in list(markov) + grid.names if s not in states]
        if unknown:
            raise KeyError(f"not states of the program: {unknown}")
        both = sorted(set(markov) & set(grid.names))
        if both:
            raise ValueError(f"{both}: a state is given by a Markov chain or by a grid axis, "
                             "not both")
        missing = [s for s in states if s not in markov and s not in grid.names]
        if missing:
            raise ValueError(f"states with neither a grid axis nor a Markov chain: {missing}")
        for name, chain in markov.items():
            if not isinstance(chain, MarkovChain):
                raise TypeError(f"markov[{name!r}] must be a MarkovChain")
        self.markov_names = [s for s in states if s in markov]
        self.markov_idx = np.array([states.index(s) for s in self.markov_names], dtype=int)
        self.cont_idx = np.array([states.index(s) for s in grid.names], dtype=int)
        self.chain, self.node_values = _markov_product([markov[s] for s in self.markov_names])
        self.n_nodes = self.node_values.shape[0]
        self.transition = self.chain.transition if self.chain is not None else np.ones((1, 1))
        self.eps_nodes, self.eps_weights, self.chol = _shock_rule(
            program.shocks, cov, n_quad, quadrature, bool(markov))
        self.nc = len(program.controls)
        self.state_index = {s: i for i, s in enumerate(states)}
        self.control_index = {x: i for i, x in enumerate(program.controls)}
        # what is drawn between t and t+1: the shocks, then the Markov states' next values
        self.draw_index = {e: i for i, e in enumerate(program.shocks + self.markov_names)}
        self.grid_nodes = np.repeat(np.arange(self.n_nodes), grid.n)
        self.grid_cont = np.tile(grid.points, (self.n_nodes, 1))
        self._vectorised = {"reward": None, "transition": None}
        self.choices = None
        if choices is not None:
            missing = [x for x in program.controls if x not in choices]
            extra = [x for x in choices if x not in program.controls]
            if missing or extra:
                raise ValueError(f"choices need every control and nothing else: missing "
                                 f"{missing}, unknown {extra}")
            sets = [np.asarray(choices[x], dtype=float).ravel() for x in program.controls]
            if any(s.size == 0 for s in sets):
                raise ValueError("every control needs at least one choice")
            mesh = np.meshgrid(*sets, indexing="ij")
            self.choices = np.column_stack([m.ravel() for m in mesh])

    # -- states ------------------------------------------------------------
    def full_states(self, node, cont):
        S = np.empty((cont.shape[0], len(self.program.states)))
        S[:, self.cont_idx] = cont
        if self.markov_idx.size:
            S[:, self.markov_idx] = self.node_values[node]
        return S

    def states_array(self, states):
        names = self.program.states
        if isinstance(states, Mapping):
            S = np.atleast_1d(np.array([np.asarray(states[s], dtype=float) for s in names],
                                       dtype=float).T)
        else:
            S = np.asarray(states, dtype=float)
        single = S.ndim == 1
        S = np.atleast_2d(S)
        if S.shape[1] != len(names):
            raise ValueError(f"a state has {len(names)} values, in the order {names}")
        return S, single

    def split_states(self, S):
        cont = S[:, self.cont_idx]
        if self.chain is None:
            return np.zeros(S.shape[0], dtype=int), cont
        M = S[:, self.markov_idx]
        match = np.all(np.isclose(M[:, None, :], self.node_values[None, :, :],
                                  rtol=1e-10, atol=1e-12), axis=2)
        if not np.all(match.any(axis=1)):
            raise ValueError(f"the values of {self.markov_names} must be nodes of their chains")
        return match.argmax(axis=1), cont

    # -- interpolation -------------------------------------------------------
    def coefficients(self, values):
        return np.stack([self.grid._coefficients(values[m], self.kind)
                         for m in range(self.n_nodes)])

    def interpolate(self, G, node, cont):
        out = np.empty((cont.shape[0],) + G.shape[1 + self.grid.d:])
        for m in np.unique(node):
            sel = node == m
            out[sel] = self.grid._evaluate(G[m], cont[sel], self.kind)
        return out

    def initial_values(self, initial):
        S = self.full_states(self.grid_nodes, self.grid_cont)
        if initial is None:
            return np.zeros(S.shape[0])
        if isinstance(initial, ValueIterationSolution):
            return np.asarray(initial.value(S), dtype=float)
        if callable(initial):
            return np.broadcast_to(np.asarray(initial(S, self.params), dtype=float),
                                   (S.shape[0],)).copy()
        return np.full(S.shape[0], float(initial))

    # -- the program's functions on arrays -----------------------------------
    def _call(self, which, args, views):
        """``program.<which>`` at ``P`` points, vectorised when the function allows it."""
        fn = getattr(self.program, which)
        P = args[0].shape[0]

        def at_once():
            return fn(*[VarView(a.T, idx) for a, idx in zip(args, views)], self.params)

        def one(p):
            return fn(*[VarView(a[p], idx) for a, idx in zip(args, views)], self.params)

        if self._vectorised[which] is not False:
            try:
                out = self._shape(which, at_once(), P)
                if self._vectorised[which] is None:
                    k = min(3, P)
                    loop = np.vstack([self._shape(which, one(p), 1) for p in range(k)])
                    ok = np.allclose(out[:k], loop, rtol=1e-12, atol=1e-14, equal_nan=True)
                    self._vectorised[which] = bool(ok)
                    if not ok:
                        raise _NotVectorised
                return out
            except _NotVectorised:
                pass
            except Exception:
                if self._vectorised[which]:
                    raise
                self._vectorised[which] = False
        return np.vstack([self._shape(which, one(p), 1) for p in range(P)])

    def _shape(self, which, r, P):
        if which == "reward":
            return np.broadcast_to(np.asarray(r, dtype=float), (P,)).reshape(P, 1)
        names = self.program.states
        endo = [names[i] for i in self.cont_idx]
        if isinstance(r, Mapping):
            absent = [s for s in endo if s not in r]
            if absent:
                raise KeyError(f"{self.program.name}: the transition gives no value for {absent}")
            rows = [r[s] for s in endo]
        else:
            r = list(r)
            if len(r) != len(names):
                raise ValueError(f"{self.program.name}: the transition returned {len(r)} values "
                                 f"for {len(names)} states")
            rows = [r[i] for i in self.cont_idx]
        return np.column_stack([np.broadcast_to(np.asarray(v, dtype=float), (P,)) for v in rows])

    def reward(self, S, X):
        with np.errstate(all="ignore"):
            return self._call("reward", (S, X), (self.state_index, self.control_index))[:, 0]

    def next_states(self, S, X, eps, nxt_values):
        """The next continuous states, in the grid's order, given the draw ``[eps, nxt]``."""
        P = S.shape[0]
        E = np.column_stack([np.broadcast_to(eps, (P, eps.shape[1])),
                             np.broadcast_to(nxt_values, (P, nxt_values.shape[1]))])
        with np.errstate(all="ignore"):
            return self._call("transition", (S, X, E),
                              (self.state_index, self.control_index, self.draw_index))

    # -- the Bellman operator ------------------------------------------------
    def successors(self, S, X, node):
        """For each next node with some probability, the next states at every quadrature node."""
        out = []
        for j in range(self.n_nodes):
            prob = self.transition[node, j]
            if not np.any(prob > 0):
                continue
            nxt = [self.next_states(S, X, e[None, :], self.node_values[[j]])
                   for e in self.eps_nodes]
            out.append((j, prob, nxt))
        return out

    def continuation(self, G, succ, P):
        total = np.zeros(P)
        for j, prob, nxt in succ:
            vj = np.zeros(P)
            for w, Sn in zip(self.eps_weights, nxt):
                vj += w * self.grid._evaluate(G[j], Sn, self.kind)[:, 0]
            total += prob * vj
        return total

    def choice_value(self, G, node, S, X):
        with np.errstate(all="ignore"):
            q = self.reward(S, X) + self.beta * self.continuation(
                G, self.successors(S, X, node), S.shape[0])
        return np.where(np.isfinite(q), q, -np.inf)

    def evaluate(self, V, node, cont, X, steps):
        """Apply the policy's own operator ``steps`` times, starting from ``V``."""
        S = self.full_states(node, cont)
        R = self.reward(S, X)
        succ = self.successors(S, X, node)
        for _ in range(steps):
            G = self.coefficients(V.reshape(self.n_nodes, self.grid.n, 1))
            V = R + self.beta * self.continuation(G, succ, S.shape[0])
        return V

    def bounds_at(self, S):
        P = S.shape[0]
        lo, hi = np.full((P, self.nc), -np.inf), np.full((P, self.nc), np.inf)
        view = VarView(S.T, self.state_index)
        for name, (low, high) in self.program.bounds.items():
            j = self.control_index[name]
            for target, b in ((lo, low), (hi, high)):
                if b is None:
                    continue
                if callable(b):
                    try:
                        v = np.broadcast_to(np.asarray(b(view, self.params), dtype=float), (P,))
                    except Exception:
                        v = np.array([float(b(VarView(s, self.state_index), self.params))
                                      for s in S])
                else:
                    v = np.full(P, float(b))
                target[:, j] = v
        if np.any(lo > hi):
            raise ValueError("a lower bound exceeds the upper bound at some state")
        return lo, hi

    def maximize(self, G, node, cont, start):
        """The maximizing controls and the maximum at each point, given the coefficients ``G``."""
        S = self.full_states(node, cont)
        lo, hi = self.bounds_at(S)
        if self.choices is not None:
            return self._exhaustive(G, node, S, lo, hi)
        if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
            raise ValueError("value iteration over an interval needs finite bounds on every "
                             "control; give bounds, or choices")
        if self.nc == 1:
            return self._golden(G, node, S, lo[:, 0], hi[:, 0])
        return self._newton(G, node, S, lo, hi, start)

    def _exhaustive(self, G, node, S, lo, hi):
        P = S.shape[0]
        best = np.full(P, -np.inf)
        arg = np.zeros(P, dtype=int)
        for k, x in enumerate(self.choices):
            X = np.broadcast_to(x, (P, self.nc))
            q = self.choice_value(G, node, S, X)
            q = np.where(np.all((X >= lo) & (X <= hi), axis=1), q, -np.inf)
            better = q > best
            best = np.where(better, q, best)
            arg = np.where(better, k, arg)
        if not np.all(np.isfinite(best)):
            raise ValueError("no choice is feasible at some grid point")
        return self.choices[arg], best

    def _golden(self, G, node, S, lo, hi):
        def q(x):
            return self.choice_value(G, node, S, x[:, None])

        a, b = lo.copy(), hi.copy()
        c, d = b - _GOLDEN * (b - a), a + _GOLDEN * (b - a)
        fc, fd = q(c), q(d)
        width = np.maximum(hi - lo, 1e-300)
        for _ in range(200):
            if np.all(b - a <= self.inner_tol * width):
                break
            left = fc >= fd                 # the maximum is in [a, d]
            b = np.where(left, d, b)
            a = np.where(left, a, c)
            new = np.where(left, b - _GOLDEN * (b - a), a + _GOLDEN * (b - a))
            fn = q(new)
            c, d, fc, fd = (np.where(left, new, d), np.where(left, c, new),
                            np.where(left, fn, fd), np.where(left, fc, fn))
        mid = 0.5 * (a + b)
        cands = np.column_stack([mid, lo, hi])
        vals = np.column_stack([q(mid), q(lo), q(hi)])
        pick = np.argmax(vals, axis=1)
        rows = np.arange(S.shape[0])
        best = vals[rows, pick]
        if not np.all(np.isfinite(best)):
            raise ValueError("the reward is not finite anywhere inside the bounds at some state")
        return cands[rows, pick][:, None], best

    def _newton(self, G, node, S, lo, hi, start):
        P, n = S.shape[0], self.nc

        def q(X):
            return self.choice_value(G, node, S, X)

        # The start: the best of the given controls and a coarse lattice inside the bounds. The
        # lattice keeps a point from staying at a local maximum it reached from an early,
        # inaccurate value function.
        X = np.clip(start, lo, hi) if start is not None else 0.5 * (lo + hi)
        f = q(X)
        ticks = np.linspace(0.1, 0.9, 5)
        mesh = np.meshgrid(*([ticks] * n), indexing="ij")
        for t in np.column_stack([m.ravel() for m in mesh]):
            Y = lo + t * (hi - lo)
            fy = q(Y)
            better = fy > f
            X, f = np.where(better[:, None], Y, X), np.where(better, fy, f)
        if not np.all(np.isfinite(f)):
            raise ValueError("the reward is not finite anywhere inside the bounds at some state")
        eye = np.eye(n)
        for _ in range(self.inner_max_iter):
            g, H = self._derivatives(q, X, f, lo, hi)
            span = np.maximum(1.0, np.abs(X))
            at_lo = (X - lo <= 1e-12 * span) & (g < 0)
            at_hi = (hi - X <= 1e-12 * span) & (g > 0)
            fixed = at_lo | at_hi
            # Newton's step on the free controls, with the Hessian made negative definite
            free = ~fixed
            Hf = np.where(free[:, :, None] & free[:, None, :], H, 0.0)
            Hf = 0.5 * (Hf + np.swapaxes(Hf, 1, 2))
            w, U = np.linalg.eigh(Hf)
            scale = np.maximum(np.max(np.abs(w), axis=1, keepdims=True), 1e-12)
            w = np.minimum(w, -1e-8 * scale)
            Hn = np.einsum("pij,pj,pkj->pik", U, w, U)
            Hn = np.where(free[:, :, None] & free[:, None, :], Hn, -eye)
            gf = np.where(free, g, 0.0)
            step = -np.linalg.solve(Hn, gf[:, :, None])[:, :, 0]
            alpha = np.ones(P)
            todo = np.ones(P, dtype=bool)
            Xn, fn = X.copy(), f.copy()
            for _ in range(40):
                if not todo.any():
                    break
                Y = np.clip(X[todo] + alpha[todo, None] * step[todo], lo[todo], hi[todo])
                fy = self.choice_value(G, node[todo], S[todo], Y)
                ok = fy >= f[todo] + 1e-4 * np.sum(g[todo] * (Y - X[todo]), axis=1)
                idx = np.flatnonzero(todo)
                Xn[idx[ok]], fn[idx[ok]] = Y[ok], fy[ok]
                todo[idx[ok]] = False
                alpha[todo] *= 0.5
            moved = np.max(np.abs(Xn - X) / np.maximum(1.0, np.abs(X)), axis=1)
            X, f = Xn, fn
            if np.all(moved <= self.inner_tol):
                break
        return X, f

    def _derivatives(self, q, X, f, lo, hi):
        """Gradient and Hessian of ``q`` at ``X`` by differences that stay inside the bounds."""
        P, n = X.shape
        span = np.maximum(1.0, np.abs(X))
        h1 = np.cbrt(np.finfo(float).eps) * span
        h2 = np.sqrt(np.sqrt(np.finfo(float).eps)) * span
        g = np.empty((P, n))
        for i in range(n):
            e = np.zeros(n)
            e[i] = 1.0
            hi_ok = X[:, i] + 2 * h1[:, i] <= hi[:, i]
            lo_ok = X[:, i] - 2 * h1[:, i] >= lo[:, i]
            fp = q(X + h1[:, i:i + 1] * e)
            fm = q(X - h1[:, i:i + 1] * e)
            central = (fp - fm) / (2 * h1[:, i])
            fp2 = q(X + 2 * h1[:, i:i + 1] * e)
            forward = (-3 * f + 4 * fp - fp2) / (2 * h1[:, i])
            fm2 = q(X - 2 * h1[:, i:i + 1] * e)
            backward = (3 * f - 4 * fm + fm2) / (2 * h1[:, i])
            g[:, i] = np.where(hi_ok & lo_ok, central, np.where(hi_ok, forward, backward))
        # the Hessian at a centre moved inside the bounds, so that every difference is defined
        C = np.clip(X, lo + 2 * h2, hi - 2 * h2)
        C = np.where(hi - lo > 4 * h2, C, 0.5 * (lo + hi))
        fc = q(C)
        H = np.empty((P, n, n))
        steps = [h2[:, i:i + 1] * np.eye(n)[i] for i in range(n)]
        plus = [q(C + s) for s in steps]
        minus = [q(C - s) for s in steps]
        for i in range(n):
            H[:, i, i] = (plus[i] - 2 * fc + minus[i]) / h2[:, i] ** 2
            for j in range(i):
                fpp = q(C + steps[i] + steps[j])
                fmm = q(C - steps[i] - steps[j])
                H[:, i, j] = H[:, j, i] = ((fpp - plus[i] - plus[j] + 2 * fc - minus[i]
                                            - minus[j] + fmm) / (2 * h2[:, i] * h2[:, j]))
        return (np.where(np.isfinite(g), g, 0.0),
                np.where(np.isfinite(H), H, 0.0))


class _NotVectorised(Exception):
    pass
