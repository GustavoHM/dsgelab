"""The endogenous grid method (Carroll, 2006), for Euler equations that invert in closed form.

An :class:`EGMProgram` declares a problem with one endogenous state ``s``, controls ``x`` and a
poststate ``a``, the endogenous state at the end of the period, once the controls are chosen:

    a = post_decision(s, x),        s' = half_transition(a, e'),
    z(a) = E[expectation(s', x(s'), e')],
    x = direct_response(a, z(a)),   s = reverse_state(a, x).

The Euler equation is written so that its expectation ``z`` depends on today only through the
poststate, and ``direct_response`` inverts it for the controls. For consumption and saving with
cash on hand ``w``, ``a = w - c``, ``w' = R a + y'``, ``z = beta R E[c'^(-gamma)]`` and
``c = z^(-1/gamma)``. These are dolo's ``expectation``, ``half_transition``,
``direct_response_egm`` and ``reverse_state``, with ``post_decision`` added for the Euler errors
and the simulation.

The grid is on the poststate. Each iteration takes the policy found so far as the next
period's, integrates the expectation at every poststate on the grid, inverts it for today's
controls and maps each poststate back to the state from which it is chosen. Those states are the
endogenous grid, and the policy is interpolated between them, linearly or by a natural cubic
spline, extrapolated linearly, and held within the bounds on the controls. Below the lowest
endogenous point the extrapolation overshoots a borrowing limit given as a bound, ``c <= w``, and
the bound then holds the policy at the limit, which is where it binds. The method solves no
equation numerically, whereas time iteration runs Newton's method at every grid point. The
iteration stops when neither the controls nor the endogenous grid move by more than ``tol``.

The other states must follow Markov chains, and the shocks are integrated by quadrature, as in
:func:`time_iteration`. ``reverse_state(a, direct_response(a, z(a)))`` must increase with the
poststate at every node of the chain, which holds when the problem is concave.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike

from .markov import MarkovChain
from .model import Covariance, Params, Point, VarView
from .time_iteration import Bound, CartesianGrid, _markov_product, _shock_rule

__all__ = ["EGMProgram", "EGMSolution", "egm"]


@dataclass
class EGMProgram:
    """A problem in the form the endogenous grid method solves.

    Parameters
    ----------
    states : sequence of str
        The state variables: one endogenous state, on the endogenous grid, and the states that
        follow a Markov chain.
    controls : sequence of str
        The choice variables.
    poststate : str
        The name of the endogenous state at the end of the period, on whose grid the method
        iterates.
    expectations : sequence of str
        The names of the integrated terms of the Euler equation.
    expectation : callable
        ``expectation(s, x, e, params)``, the terms integrated over the next period, at next
        period's states ``s`` and controls ``x``: a mapping from each name in ``expectations`` to
        its value, or a sequence in their order.
    direct_response : callable
        ``direct_response(a, z, params)``, the controls that satisfy the Euler equation given the
        poststate and the integrated terms ``z``: a mapping or a sequence in the order of
        ``controls``.
    reverse_state : callable
        ``reverse_state(a, x, params)``, the endogenous state from which ``x`` leads to the
        poststate ``a``.
    half_transition : callable
        ``half_transition(a, e, params)``, next period's endogenous state, given the poststate.
    post_decision : callable
        ``post_decision(s, x, params)``, the poststate, given the states and the controls. It is
        the inverse of ``reverse_state`` in the state.
    shocks : sequence of str, optional
        The shocks, integrated as in :func:`time_iteration`.
    bounds : mapping, optional
        ``{control: (lower, upper)}``, each bound a number, ``None``, or a function of today's
        states and the parameters. The policy is held within them.
    name : str, optional

    Notes
    -----
    ``a`` is a name-indexed view of the poststate and of today's values of the Markov states;
    ``s``, ``x`` and ``z`` are views of the states, the controls and the integrated terms; ``e``
    holds what is drawn between the two periods: the shocks, and the next values of the Markov
    states, under their own names, as in :class:`DynamicProgram`. ``reverse_state``,
    ``half_transition`` and ``post_decision`` return a value, or a mapping from the variable's
    name to it.
    """

    states: Sequence[str]
    controls: Sequence[str]
    poststate: str
    expectations: Sequence[str]
    expectation: Callable
    direct_response: Callable
    reverse_state: Callable
    half_transition: Callable
    post_decision: Callable
    shocks: Sequence[str] = ()
    bounds: Mapping[str, tuple[Bound, Bound]] | None = None
    name: str = "program"

    def __post_init__(self) -> None:
        self.states = list(self.states)
        self.controls = list(self.controls)
        self.expectations = list(self.expectations)
        self.shocks = list(self.shocks)
        self.bounds = dict(self.bounds or {})
        if not self.states or not self.controls or not self.expectations:
            raise ValueError(f"{self.name}: a program needs at least one state, one control and "
                             "one expectation")
        names = self.states + self.controls + [self.poststate] + self.expectations + self.shocks
        repeated = sorted({n for n in names if names.count(n) > 1})
        if repeated:
            raise ValueError(f"{self.name}: names used more than once: {repeated}")
        unknown = [b for b in self.bounds if b not in self.controls]
        if unknown:
            raise KeyError(f"{self.name}: bounds given for {unknown}, which are not controls")

    def __repr__(self) -> str:
        return (f"EGMProgram({self.name!r}, states={self.states}, controls={self.controls}, "
                f"poststate={self.poststate!r}, shocks={self.shocks})")


@dataclass
class EGMSolution:
    """The policy on the endogenous grid, one grid per node of the Markov chain.

    ``endogenous[m]`` holds the endogenous state that leads to each poststate of ``grid`` when the
    chain is at node ``m``, and ``values[m]`` the controls there, one row per poststate. Without a
    Markov chain there is one node.
    """

    program: EGMProgram
    params: Params
    grid: CartesianGrid
    chain: MarkovChain | None
    markov_states: list[str]
    endogenous: np.ndarray             # (n_nodes, grid.n)
    values: np.ndarray                 # (n_nodes, grid.n, n_controls)
    interpolation: str
    iterations: int
    converged: bool
    distance: float
    history: list[float] = field(repr=False)
    _problem: Any = field(repr=False, default=None)
    _policy: Any = field(repr=False, default=None)

    def controls(self, states: Point) -> np.ndarray:
        """The controls at one state or at a row-stacked array of states, in the program's order.

        A state given by a Markov chain must take one of the chain's node values.
        """
        pb = self._problem
        S, single = pb.states_array(states)
        out = self._policy(pb.split_states(S), S)
        return out[0] if single else out

    __call__ = controls

    def euler_errors(self, states: Point | None = None) -> np.ndarray:
        """The Euler errors at ``states``, relative to the controls, one row per state.

        At each state the policy gives the controls and the poststate, the policy at the next
        states gives the integrated terms, and ``direct_response`` gives the controls they call
        for, held within the bounds. The error is their difference from the policy, divided by
        the policy's absolute value. Where a bound binds and the direct response lies beyond it,
        the error is zero. By default the states are the centres between neighbouring points of
        the endogenous grid, at every node of the chain.
        """
        pb = self._problem
        if states is None:
            mids = 0.5 * (self.endogenous[:, 1:] + self.endogenous[:, :-1])
            node = np.repeat(np.arange(pb.n_nodes), mids.shape[1])
            S = pb.full_states(node, mids.ravel())
        else:
            S, _ = pb.states_array(states)
            node = pb.split_states(S)
        X = self._policy(node, S)
        A = pb.post_decision(S, X)
        target = pb.direct(A, node, pb.expect(A, node, self._policy))
        lo, hi = pb.bounds_at(S)
        return (np.clip(target, lo, hi) - X) / np.abs(X)

    def simulate(self, periods: int, *, start: Point | None = None, seed: int | None = None,
                 shocks: ArrayLike | None = None) -> dict[str, np.ndarray]:
        """Simulate the policy for ``periods`` periods, from ``start``.

        By default the endogenous state starts at the centre of the endogenous grid of the
        chain's middle node. The shocks are drawn from the covariance the solution was built
        with, or given in ``shocks``, one row per period after the first; a Markov state follows
        its chain. Returns ``states`` and ``controls``, one row per period in the program's
        order, ``poststates`` and ``nodes``, the chain's index each period.
        """
        pb = self._problem
        rng = np.random.default_rng(seed)
        if start is None:
            mid = pb.chain.n // 2 if pb.chain is not None else 0
            e = self.endogenous[mid]
            S = pb.full_states(np.array([mid]), np.array([0.5 * (e[0] + e[-1])]))
        else:
            S, single = pb.states_array(start)
            if not single:
                raise ValueError("start must be a single state")
        node = pb.split_states(S)
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
        states = np.empty((periods, len(self.program.states)))
        controls = np.empty((periods, pb.nx))
        post = np.empty(periods)
        for t in range(periods):
            X = self._policy(node, S)
            A = pb.post_decision(S, X)
            states[t], controls[t], post[t] = S[0], X[0], A[0]
            if t == periods - 1:
                break
            nxt = np.array([path[t + 1]])
            S = pb.full_states(nxt, pb.half(A, node, pb.draw(eps[t], int(nxt[0]), 1)))
            node = nxt
        return {"states": states, "controls": controls, "poststates": post, "nodes": path}


def egm(program: EGMProgram, params: Params, *, grid: CartesianGrid | Mapping[str, Any],
        initial: Any, markov: Mapping[str, MarkovChain] | None = None, cov: Covariance = None,
        n_quad: int = 5, quadrature: tuple[ArrayLike, ArrayLike] | None = None,
        interpolation: str = "linear", tol: float = 1e-8, max_iter: int = 1000) -> EGMSolution:
    """Solve ``program`` by the endogenous grid method.

    Parameters
    ----------
    grid :
        A :class:`CartesianGrid` with one axis, the poststate, or the mapping that builds one.
    initial :
        The policy the iteration starts from, as next period's: an :class:`EGMSolution` (to
        refine on a new grid), ``initial(states, params)`` returning the controls, or a mapping
        from each control to a number or such a function.
    markov :
        The states other than the endogenous one, given by Markov chains, as in
        :func:`time_iteration`.
    cov, n_quad, quadrature :
        The shocks' integration rule, as in :func:`time_iteration`.
    interpolation :
        ``"linear"`` or ``"cubic"`` (a natural spline), between the points of the endogenous
        grid.
    tol :
        The iteration stops when neither a control nor a point of the endogenous grid moves by
        more than ``tol``.
    """
    if not isinstance(grid, CartesianGrid):
        grid = CartesianGrid(grid)
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1")
    pb = _EGM(program, params, grid, dict(markov or {}), cov, n_quad, quadrature, interpolation)
    policy = pb.initial_policy(initial)
    A, node = pb.post_points, pb.post_nodes
    shape = (pb.n_nodes, grid.n)
    S_old = X_old = None
    history: list[float] = []
    distance = np.inf
    it = 0
    while it < max_iter:
        it += 1
        X = pb.direct(A, node, pb.expect(A, node, policy))
        S = pb.reverse(A, node, X)
        if it == 1:
            pb.check_inverse(S, X, node)
        policy = pb.interpolant(S.reshape(shape), X.reshape(shape + (pb.nx,)))
        if S_old is not None:
            distance = float(max(np.max(np.abs(X - X_old)), np.max(np.abs(S - S_old))))
            history.append(distance)
            if distance < tol:
                break
        S_old, X_old = S, X
    return EGMSolution(
        program=program, params=params, grid=grid, chain=pb.chain, markov_states=pb.markov_names,
        endogenous=S.reshape(shape), values=X.reshape(shape + (pb.nx,)),
        interpolation=interpolation, iterations=it, converged=bool(distance < tol),
        distance=distance, history=history, _problem=pb, _policy=policy)


class _EGM:
    """The pieces of one endogenous-grid problem, on arrays with one row per point."""

    def __init__(self, program, params, grid, markov, cov, n_quad, quadrature, interpolation):
        self.program, self.params, self.grid = program, params, grid
        if interpolation not in ("linear", "cubic"):
            raise ValueError(f"interpolation must be 'linear' or 'cubic', not {interpolation!r}")
        self.kind = interpolation
        if grid.d != 1 or grid.names[0] != program.poststate:
            raise ValueError(f"the grid needs one axis, the poststate {program.poststate!r}, "
                             f"not {grid.names}")
        states = program.states
        unknown = [s for s in markov if s not in states]
        if unknown:
            raise KeyError(f"not states of the program: {unknown}")
        for name, chain in markov.items():
            if not isinstance(chain, MarkovChain):
                raise TypeError(f"markov[{name!r}] must be a MarkovChain")
        endo = [s for s in states if s not in markov]
        if len(endo) != 1:
            raise ValueError(f"the endogenous grid method needs one state without a Markov chain, "
                             f"not {endo}")
        self.endo = endo[0]
        self.endo_idx = states.index(self.endo)
        self.markov_names = [s for s in states if s in markov]
        self.markov_idx = np.array([states.index(s) for s in self.markov_names], dtype=int)
        self.chain, self.node_values = _markov_product([markov[s] for s in self.markov_names])
        self.n_nodes = self.node_values.shape[0]
        self.transition = self.chain.transition if self.chain is not None else np.ones((1, 1))
        self.eps_nodes, self.eps_weights, self.chol = _shock_rule(
            program.shocks, cov, n_quad, quadrature, bool(markov))
        self.nx = len(program.controls)
        self.state_index = {s: i for i, s in enumerate(states)}
        self.control_index = {x: i for i, x in enumerate(program.controls)}
        self.expectation_index = {z: i for i, z in enumerate(program.expectations)}
        # the poststate's view: the poststate, then today's values of the Markov states
        self.post_index = {n: i for i, n in enumerate([program.poststate] + self.markov_names)}
        # what is drawn between t and t+1: the shocks, then the Markov states' next values
        self.draw_index = {e: i for i, e in enumerate(program.shocks + self.markov_names)}
        self.post_points = np.tile(grid.axes[0], self.n_nodes)
        self.post_nodes = np.repeat(np.arange(self.n_nodes), grid.n)
        self._vectorised: dict[str, bool | None] = {}

    # -- states ------------------------------------------------------------
    def full_states(self, node, s):
        S = np.empty((s.shape[0], len(self.program.states)))
        S[:, self.endo_idx] = s
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
        """The chain's node at each state."""
        if self.chain is None:
            return np.zeros(S.shape[0], dtype=int)
        M = S[:, self.markov_idx]
        match = np.all(np.isclose(M[:, None, :], self.node_values[None, :, :],
                                  rtol=1e-10, atol=1e-12), axis=2)
        if not np.all(match.any(axis=1)):
            raise ValueError(f"the values of {self.markov_names} must be nodes of their chains")
        return match.argmax(axis=1)

    def post_array(self, A, node):
        return np.column_stack([A, self.node_values[node]])

    def draw(self, eps, j, P):
        """The draw ``[shocks, next Markov values]`` at ``P`` points, with the chain at ``j``."""
        return np.column_stack([np.broadcast_to(eps, (P, eps.size)),
                                np.broadcast_to(self.node_values[j], (P, self.markov_idx.size))])

    def bounds_at(self, S):
        P = S.shape[0]
        lo, hi = np.full((P, self.nx), -np.inf), np.full((P, self.nx), np.inf)
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

    # -- the program's functions on arrays -----------------------------------
    def _call(self, which, fn, args, views, names):
        """``fn`` at ``P`` points, vectorised when the function allows it: ``(P, len(names))``."""
        P = args[0].shape[0]

        def at_once():
            return fn(*[VarView(a.T, idx) for a, idx in zip(args, views)], self.params)

        def one(p):
            return fn(*[VarView(a[p], idx) for a, idx in zip(args, views)], self.params)

        state = self._vectorised.get(which)
        if state is not False:
            try:
                out = self._shape(which, at_once(), names, P)
                if state is None:
                    k = min(3, P)
                    loop = np.vstack([self._shape(which, one(p), names, 1) for p in range(k)])
                    ok = np.allclose(out[:k], loop, rtol=1e-12, atol=1e-14, equal_nan=True)
                    self._vectorised[which] = bool(ok)
                    if not ok:
                        raise _NotVectorised
                return out
            except _NotVectorised:
                pass
            except Exception:
                if state:
                    raise
                self._vectorised[which] = False
        return np.vstack([self._shape(which, one(p), names, 1) for p in range(P)])

    def _shape(self, which, r, names, P):
        if isinstance(r, Mapping):
            absent = [n for n in names if n not in r]
            if absent:
                raise KeyError(f"{self.program.name}: {which} gives no value for {absent}")
            rows = [r[n] for n in names]
        elif len(names) == 1 and not isinstance(r, (list, tuple)):
            rows = [r]
        else:
            rows = list(r)
            if len(rows) != len(names):
                raise ValueError(f"{self.program.name}: {which} returned {len(rows)} values for "
                                 f"{names}")
        return np.column_stack([np.broadcast_to(np.asarray(v, dtype=float), (P,)) for v in rows])

    def half(self, A, node, E):
        """Next period's endogenous state at ``P`` poststates, given the draw ``E``."""
        return self._call("half_transition", self.program.half_transition,
                          (self.post_array(A, node), E), (self.post_index, self.draw_index),
                          [self.endo])[:, 0]

    def direct(self, A, node, Z):
        return self._call("direct_response", self.program.direct_response,
                          (self.post_array(A, node), Z), (self.post_index, self.expectation_index),
                          self.program.controls)

    def reverse(self, A, node, X):
        return self._call("reverse_state", self.program.reverse_state,
                          (self.post_array(A, node), X), (self.post_index, self.control_index),
                          [self.endo])[:, 0]

    def post_decision(self, S, X):
        return self._call("post_decision", self.program.post_decision, (S, X),
                          (self.state_index, self.control_index), [self.program.poststate])[:, 0]

    # -- one step ------------------------------------------------------------
    def expect(self, A, node, policy):
        """The integrated terms at ``P`` poststates, with today's chain at ``node``."""
        P = A.shape[0]
        Z = np.zeros((P, len(self.program.expectations)))
        for j in range(self.n_nodes):
            prob = self.transition[node, j]
            if not np.any(prob > 0):
                continue
            nxt = np.full(P, j)
            for e, w in zip(self.eps_nodes, self.eps_weights):
                E = self.draw(e, j, P)
                Sn = self.full_states(nxt, self.half(A, node, E))
                Xn = policy(nxt, Sn)
                Z += (prob * w)[:, None] * self._call(
                    "expectation", self.program.expectation, (Sn, Xn, E),
                    (self.state_index, self.control_index, self.draw_index),
                    self.program.expectations)
        return Z

    def interpolant(self, S, X):
        """The policy between the points of the endogenous grid, held within the bounds."""
        if not (np.all(np.isfinite(S)) and np.all(np.isfinite(X))):
            raise ValueError("the direct response or the reverse state is not finite at some "
                             "poststate")
        grids, coefs = [], []
        for m in range(self.n_nodes):
            if np.any(np.diff(S[m]) <= 0):
                raise ValueError(
                    f"the endogenous grid does not increase with the poststate at node {m}: "
                    "reverse_state(a, direct_response(a, z(a))) must increase with a, which "
                    "fails when the problem is not concave")
            g = CartesianGrid({self.endo: S[m]})
            grids.append(g)
            coefs.append(g._coefficients(X[m], self.kind))

        def policy(node, states):
            out = np.empty((states.shape[0], self.nx))
            s = states[:, [self.endo_idx]]
            for m in np.unique(node):
                sel = node == m
                out[sel] = grids[m]._evaluate(coefs[m], s[sel], self.kind)
            return np.clip(out, *self.bounds_at(states))

        return policy

    def check_inverse(self, S, X, node):
        """``post_decision(reverse_state(a, x), x)`` must give back ``a``."""
        A = self.post_decision(self.full_states(node, S), X)
        a = self.post_points
        if not np.allclose(A, a, rtol=1e-8, atol=1e-10 * max(1.0, np.max(np.abs(a)))):
            worst = float(np.max(np.abs(A - a)))
            raise ValueError(f"{self.program.name}: post_decision does not invert reverse_state; "
                             f"post_decision(reverse_state(a, x), x) differs from a by {worst:.2e}")

    # -- the starting policy -------------------------------------------------------
    def initial_policy(self, initial):
        if isinstance(initial, EGMSolution):
            return lambda node, S: np.clip(initial.controls(S), *self.bounds_at(S))
        if isinstance(initial, Mapping):
            missing = [x for x in self.program.controls if x not in initial]
            if missing:
                raise KeyError(f"initial has no value for {missing}")
            funcs = {x: initial[x] for x in self.program.controls}

            def fn(s, p):
                return {x: f(s, p) if callable(f) else f for x, f in funcs.items()}
        elif callable(initial):
            fn = initial
        else:
            raise TypeError("initial must be an EGMSolution, a function or a mapping")

        def policy(node, S):
            X = self._call("initial", fn, (S,), (self.state_index,), self.program.controls)
            return np.clip(X, *self.bounds_at(S))

        return policy


class _NotVectorised(Exception):
    pass
