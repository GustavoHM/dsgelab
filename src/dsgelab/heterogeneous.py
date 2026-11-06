"""Heterogeneous agents by the sequence-space Jacobian method.

A model here is a set of blocks joined by the names of their inputs and outputs, as in Auclert,
Bardóczy, Rognlie and Straub (2021). Each block maps paths of aggregate inputs to paths of
aggregate outputs, and the model is the directed acyclic graph the names define. A
:class:`HetBlock` holds a continuum of agents on a grid of one asset and the nodes of a Markov
chain, and a distribution over that grid. A :class:`SimpleBlock` is a set of equations in the
aggregates, with lags and leads, and a :class:`SolvedBlock` solves a smaller model of blocks for
some of its inputs. The perturbation's :class:`Model`, with ``vec`` for a finite number of groups,
is unchanged: a distribution over a grid is a different algorithm, and it lives here.

The heterogeneous-agent block
-----------------------------
The user writes one step of the agents' backward iteration, ``step``, which takes the expectation
of the backward variables at ``t + 1``, conditional on today's income state, and returns them at
``t`` with the policy and any other outcome, all on the ``(n_e, n_a)`` grid of income states by
assets. The endogenous grid method (Carroll, 2006) fits this form, and :func:`interpolate_rows`
does the interpolation it needs. The steady state iterates the step to a fixed point of the policy,
and the distribution follows from Young's (2010) lottery: an agent whose policy falls between two
grid points goes to each with the probability that keeps the mean of assets. The stationary
distribution is the null vector of the sparse transition matrix, found by one sparse solve.

The timing is sequence-jacobian's. ``Dbeg_t`` is the distribution over last period's income state
and the assets carried into ``t``; the income shock turns it into ``D_t = Pi_t' Dbeg_t``, over
today's income state, and the policy turns ``D_t`` into ``Dbeg_{t+1}``. An aggregate output is the
mean of an individual outcome under ``D_t``, and its name is the outcome's in capitals. The Markov
matrix may be an output of the hetinputs and move with the inputs along a path: ``Pi_t`` moves the
distribution at ``t``, and the expectation the step takes at ``t`` is ``Pi_{t+1}`` times the
backward variables at ``t + 1``. A transition may start from a distribution ``Dbeg_0`` other than
the steady state's.

The fake-news algorithm
-----------------------
The Jacobian ``J[t, s]`` of an aggregate output at ``t`` with respect to an input at ``s`` takes
four steps (Auclert et al., 2021, section 3). A shock announced ``s`` periods ahead moves the
policy and the outcomes today by what one step of the backward iteration gives, ``s`` times
over: this is ``curlyY_s``, the effect on the aggregate at ``t = 0``, and ``curlyD_s``, the effect
on the distribution at ``t = 1``. The expectation vectors ``curlyE_k`` carry the outcome back
through ``k`` periods of the steady state's transition. The fake-news matrix has ``curlyY`` in its
first row and ``curlyE_{t-1}' curlyD_s`` below it, and ``J`` follows from it by the recursion
``J[t, s] = F[t, s] + J[t - 1, s - 1]``. Each column costs one backward step, so the whole
``T x T`` Jacobian costs ``T`` steps instead of the ``T^2`` of differentiating a path for every
``s``. The derivative of one step is a finite difference, two-sided by default, with step ``h``.
An input that moves the Markov matrix at ``s`` adds three terms, with ``dPi`` the derivative of
the matrix, as in sequence-jacobian 1.0.0: ``Dbeg' dPi y`` to ``curlyY_0``, the lottery of
``dPi' Dbeg`` to ``curlyD_0``, and ``dPi V`` to the expectation the step takes a period earlier.

General equilibrium
-------------------
The Jacobians of the blocks combine by the chain rule along the graph into the Jacobians of the
model's outputs with respect to its inputs. With some inputs chosen as unknowns and some outputs
as targets, ``H_U`` and ``H_Z`` are the targets' Jacobians with respect to the unknowns and to the
exogenous inputs, and ``G_U = -H_U^{-1} H_Z`` is the unknowns' response that keeps the targets at
zero. Every output's general-equilibrium Jacobian ``G`` follows from it, and a linear impulse
response is ``G`` times the shock's path. The nonlinear perfect-foresight transition solves the
targets for the paths of the unknowns by a quasi-Newton iteration with the fixed matrix ``H_U``.
"""

from __future__ import annotations

import inspect
import warnings
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import scipy.sparse as sp
from numpy.typing import ArrayLike
from scipy.linalg import lu_factor, lu_solve
from scipy.optimize import brentq, root
from scipy.sparse.linalg import MatrixRankWarning, spsolve

__all__ = ["HetBlock", "SimpleBlock", "SolvedBlock", "BlockModel", "BlockSteadyState", "simple",
           "interpolate_rows", "log_grid"]

Jacobians = dict[str, dict[str, np.ndarray]]

_COMPLEX_STEP = 1e-20
_MARKOV = "\0markov"        # the key of the Markov matrix's derivative among a step's outcomes
_ComplexWarning = getattr(np, "exceptions", np).ComplexWarning


# ---------------------------------------------------------------------------------------------
# grids and interpolation

def log_grid(lo: float, hi: float, n: int, *, pivot: float | None = None) -> np.ndarray:
    """``n`` points from ``lo`` to ``hi``, evenly spaced in ``log(a + pivot)``.

    The default ``pivot`` is ``|lo| + 0.25``, which is the grid sequence-jacobian's ``agrid``
    builds. The points are denser near ``lo``, where a borrowing limit bends the policy.
    """
    if int(n) != n or n < 2:
        raise ValueError("n, the number of points, must be an integer of at least 2")
    if not hi > lo:
        raise ValueError("hi must be greater than lo")
    if pivot is None:
        pivot = abs(lo) + 0.25
    if lo + pivot <= 0:
        raise ValueError("lo + pivot must be positive")
    grid = np.geomspace(lo + pivot, hi + pivot, int(n)) - pivot
    grid[0] = lo
    return grid


def interpolate_rows(x: ArrayLike, xq: ArrayLike, y: ArrayLike) -> np.ndarray:
    """Linear interpolation of ``y(x)`` at ``xq`` along the last axis, extrapolating linearly.

    ``x`` must increase along its last axis, and ``x`` and ``y`` have the same last dimension.
    The leading dimensions of the three arrays broadcast together, so that each row, an income
    state on the grid of a :class:`HetBlock`, is interpolated on its own. The query points may be
    in any order.
    """
    x, xq, y = (np.asarray(a, dtype=float) for a in (x, xq, y))
    if x.shape[-1] != y.shape[-1]:
        raise ValueError("x and y must have the same last dimension")
    if x.shape[-1] < 2:
        raise ValueError("interpolation needs at least two points")
    lead = np.broadcast_shapes(x.shape[:-1], xq.shape[:-1], y.shape[:-1])
    n, nq = x.shape[-1], xq.shape[-1]
    X = np.broadcast_to(x, lead + (n,)).reshape(-1, n)
    Y = np.broadcast_to(y, lead + (n,)).reshape(-1, n)
    Q = np.broadcast_to(xq, lead + (nq,)).reshape(-1, nq)
    out = np.empty(Q.shape)
    for r in range(Q.shape[0]):
        out[r] = np.interp(Q[r], X[r], Y[r])
    # np.interp holds the end values beyond the points, and the extrapolation is linear
    for end, nxt, beyond in ((0, 1, Q < X[:, :1]), (-1, -2, Q > X[:, -1:])):
        if beyond.any():
            slope = (Y[:, nxt] - Y[:, end]) / (X[:, nxt] - X[:, end])
            line = Y[:, end, None] + (Q - X[:, end, None]) * slope[:, None]
            out = np.where(beyond, line, out)
    return out.reshape(lead + (nq,))


class _Lottery:
    """Young's (2010) lottery of a policy on the asset grid, one row per income state."""

    def __init__(self, policy: np.ndarray, grid: np.ndarray):
        n = grid.shape[0]
        self.shape = policy.shape
        self.i = np.clip(np.searchsorted(grid, policy, side="left") - 1, 0, n - 2)
        self.space = grid[self.i + 1] - grid[self.i]
        self.w = (grid[self.i + 1] - policy) / self.space      # the weight on the lower point
        self.flat = (np.arange(policy.shape[0])[:, None] * n + self.i).ravel()
        self.size = policy.size

    def forward(self, D: np.ndarray) -> np.ndarray:
        """The distribution over the grid after the policy, from ``D`` before it."""
        lower = np.bincount(self.flat, (D * self.w).ravel(), self.size)
        upper = np.bincount(self.flat + 1, (D * (1.0 - self.w)).ravel(), self.size)
        return (lower + upper).reshape(self.shape)

    def expectation(self, X: np.ndarray) -> np.ndarray:
        """The value of ``X`` after the policy, at each point before it."""
        lo = np.take_along_axis(X, self.i, axis=1)
        hi = np.take_along_axis(X, self.i + 1, axis=1)
        return self.w * lo + (1.0 - self.w) * hi

    def shock(self, D: np.ndarray, dpolicy: np.ndarray) -> np.ndarray:
        """The derivative of ``forward(D)`` in the direction ``dpolicy``, the brackets held fixed."""
        moved = (D * dpolicy / self.space).ravel()
        return (np.bincount(self.flat + 1, moved, self.size)
                - np.bincount(self.flat, moved, self.size)).reshape(self.shape)

    def matrix(self) -> sp.csr_matrix:
        """The lottery as a sparse matrix, from the flattened grid before it to after it."""
        cols = np.arange(self.size)
        w = self.w.ravel()
        return sp.csr_matrix((np.concatenate([w, 1.0 - w]),
                              (np.concatenate([self.flat, self.flat + 1]),
                               np.concatenate([cols, cols]))), shape=(self.size, self.size))


# ---------------------------------------------------------------------------------------------
# functions with named outputs

def _arguments(f: Callable, label: str) -> list[str]:
    try:
        params = inspect.signature(f).parameters.values()
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{label}: cannot read the arguments of {f!r}") from exc
    names = []
    for p in params:
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            raise TypeError(f"{label}: the arguments must be named, without *args or **kwargs")
        names.append(p.name)
    return names


def _names(outputs: str | Sequence[str]) -> list[str]:
    names = [outputs] if isinstance(outputs, str) else list(outputs)
    if not names:
        raise ValueError("a function needs at least one output")
    if len(set(names)) != len(names):
        raise ValueError(f"outputs named more than once: {names}")
    return names


class _Function:
    """A function whose arguments are read from a mapping by name, with named outputs."""

    def __init__(self, f: Callable, outputs: str | Sequence[str], label: str):
        if not callable(f):
            raise TypeError(f"{label}: {f!r} is not callable")
        self.f, self.label = f, label
        self.inputs = _arguments(f, label)
        self.outputs = _names(outputs)

    def __call__(self, x: Mapping[str, Any]) -> dict[str, Any]:
        try:
            args = {n: x[n] for n in self.inputs}
        except KeyError as exc:
            raise KeyError(f"{self.label}: no value for {exc.args[0]!r}") from None
        return _as_outputs(self.f(**args), self.outputs, self.label)


def _as_outputs(value: Any, names: list[str], label: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        missing = [n for n in names if n not in value]
        if missing:
            raise KeyError(f"{label}: the returned mapping lacks {missing}")
        return {n: value[n] for n in names}
    if len(names) == 1:
        return {names[0]: value}
    if not isinstance(value, (tuple, list)) or len(value) != len(names):
        raise ValueError(f"{label}: expected {len(names)} outputs, {names}, in that order")
    return dict(zip(names, value))


def _pairs(functions: Iterable, label: str) -> list[_Function]:
    out = []
    for item in functions:
        if isinstance(item, SimpleBlock):
            out.append(_Function(item.f, item.outputs, f"{label} {item.name}"))
            continue
        try:
            f, outputs = item
        except (TypeError, ValueError):
            raise TypeError(f"{label}: each entry must be a pair (function, outputs)") from None
        out.append(_Function(f, outputs, f"{label} {getattr(f, '__name__', f)}"))
    return out


# ---------------------------------------------------------------------------------------------
# steady state

class BlockSteadyState(dict):
    """The steady state of a :class:`BlockModel`: aggregate values by name.

    ``internals[block]`` holds what a :class:`HetBlock` keeps on its grid: the policy and the
    other individual outcomes, the backward variables, the grids and the transition matrix, and
    the distributions ``D`` and ``Dbeg``.
    """

    def __init__(self, values: Mapping[str, Any] | Iterable[tuple[str, Any]] = (),
                 internals: Mapping[str, dict] | None = None):
        super().__init__(values)
        self.internals: dict[str, dict[str, Any]] = dict(internals or {})

    def copy(self) -> "BlockSteadyState":
        return BlockSteadyState(self, {k: dict(v) for k, v in self.internals.items()})

    def __repr__(self) -> str:
        shown = ", ".join(f"{k}={v:.6g}" if np.isscalar(v) else k for k, v in self.items())
        return f"BlockSteadyState({shown})"


def _scalar(value: Any, name: str, label: str) -> float:
    arr = np.asarray(value)
    if arr.size != 1:
        raise TypeError(f"{label}: {name} must be a number, not an array of shape {arr.shape}")
    return float(arr.reshape(-1)[0].real)


# ---------------------------------------------------------------------------------------------
# simple blocks

class _Path(np.ndarray):
    """A path of one aggregate, whose call ``x(k)`` shifts it ``k`` periods.

    Beyond either end the shifted path takes the steady-state value. Arithmetic on a path gives
    a plain array, so only the block's inputs can be shifted.
    """

    def __new__(cls, values, ss, recorder=None):
        obj = np.asarray(values).view(cls)
        obj.ss, obj.recorder = ss, recorder
        return obj

    def __array_finalize__(self, obj):
        self.ss = getattr(obj, "ss", None)
        self.recorder = getattr(obj, "recorder", None)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        inputs = tuple(np.asarray(a) if isinstance(a, _Path) else a for a in inputs)
        if "out" in kwargs:
            kwargs["out"] = tuple(np.asarray(a) if isinstance(a, _Path) else a
                                  for a in kwargs["out"])
        return getattr(ufunc, method)(*inputs, **kwargs)

    def __call__(self, k: int = 0) -> np.ndarray:
        k = int(k)
        if self.recorder is not None:
            self.recorder.append(abs(k))
        values = np.asarray(self)
        if k == 0:
            return values
        out = np.full(values.shape, self.ss, dtype=values.dtype)
        n = values.shape[0]
        if abs(k) < n:
            if k > 0:
                out[:-k] = values[k:]
            else:
                out[-k:] = values[:k]
        return out


class SimpleBlock:
    """A block of equations in the aggregates, with lags and leads.

    Parameters
    ----------
    f : callable
        ``f(x1, x2, ...)``, whose arguments are the block's inputs by name. Each input is a path,
        an array over the periods, and ``x(-1)`` and ``x(+1)`` are its lag and its lead, which take
        the steady-state value beyond the ends. At the steady state every path has one period.
        ``f`` returns the outputs, in the order of ``outputs``, or a mapping from their names.
    outputs : str or sequence of str
        The names of the outputs.
    name : str, optional
        By default the name of ``f``.

    Notes
    -----
    Only the inputs can be shifted: a lag of an expression is written with the lags of its inputs.
    The Jacobian is exact up to rounding by the complex step (Squire and Trapp, 1998), which needs
    ``f`` to accept complex arrays. An output that comes back real, because ``f`` converts it, is
    differentiated by central differences instead.
    """

    def __init__(self, f: Callable, outputs: str | Sequence[str], name: str | None = None):
        self.name = name or getattr(f, "__name__", "simple")
        self.f = f
        self._function = _Function(f, outputs, f"simple block {self.name}")
        self.inputs = self._function.inputs
        self.outputs = self._function.outputs
        both = sorted(set(self.inputs) & set(self.outputs))
        if both:
            raise ValueError(f"simple block {self.name}: {both} are both inputs and outputs")

    def __repr__(self) -> str:
        return f"SimpleBlock({self.name!r}, inputs={self.inputs}, outputs={self.outputs})"

    def __call__(self, *args, **kwargs):
        return self.f(*args, **kwargs)

    def steady_state(self, values: Mapping[str, Any], start: Mapping | None = None):
        """The outputs at the steady-state values of the inputs, and no internals."""
        args = {}
        for n in self.inputs:
            if n not in values:
                raise KeyError(f"simple block {self.name}: no value for {n!r}")
            v = _scalar(values[n], n, f"simple block {self.name}")
            args[n] = _Path(np.array([v]), v)
        out = self._function(args)
        return {o: _scalar(v, o, f"simple block {self.name}") for o, v in out.items()}, {}

    def path(self, ss: Mapping[str, Any], paths: Mapping[str, np.ndarray], T: int):
        """The outputs' paths, in levels, given the paths of some inputs in levels."""
        args = {n: _Path(np.asarray(paths[n], dtype=float), float(ss[n])) if n in paths
                else _Path(np.array([float(ss[n])]), float(ss[n])) for n in self.inputs}
        out = self._function(args)
        return {o: np.broadcast_to(np.asarray(v, dtype=float), (T,)).copy() for o, v in out.items()}

    def jacobian(self, ss: Mapping[str, Any], inputs: Sequence[str], T: int,
                 outputs: Sequence[str] | None = None) -> Jacobians:
        """``J[o][i]``, the ``T x T`` Jacobian of output ``o`` with respect to input ``i``.

        Entries that are zero for every pair of periods are left out.
        """
        outputs = list(outputs or self.outputs)
        reach = self._reach(ss)
        L, c = 2 * reach + 1, reach
        base = {n: float(ss[n]) for n in self.inputs}
        J: Jacobians = {}
        for i in inputs:
            if i not in self.inputs:
                continue
            d = self._derivatives(base, i, L, c, outputs)
            for o, dk in d.items():
                if not np.any(dk):
                    continue
                M = np.zeros((T, T))
                for k in range(-reach, reach + 1):
                    if dk[k + reach] != 0.0 and abs(k) < T:
                        M += dk[k + reach] * np.eye(T, k=k)
                J.setdefault(o, {})[i] = M
        return J

    def _reach(self, ss) -> int:
        """The largest shift the block applies to an input."""
        shifts: list[int] = []
        args = {n: _Path(np.array([float(ss[n])]), float(ss[n]), shifts) for n in self.inputs}
        self._function(args)
        return max(shifts, default=0)

    def _evaluate(self, base, i, values, L):
        args = {n: _Path(np.full(L, base[n]), base[n]) for n in self.inputs if n != i}
        args[i] = _Path(values, base[i])
        return self._function(args)

    def _derivatives(self, base, i, L, c, outputs):
        """``d[o][k + reach]``, the derivative of ``o_t`` with respect to ``i_{t+k}``."""
        idx = c - np.arange(-c, c + 1)          # the period of o that sees i_{t+k} at t = c
        d, real = {}, []
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", _ComplexWarning)
                values = np.full(L, base[i], dtype=complex)
                values[c] += 1j * _COMPLEX_STEP
                out = self._evaluate(base, i, values, L)
            for o in outputs:
                v = np.asarray(out[o])
                if np.iscomplexobj(v):
                    d[o] = np.broadcast_to(v, (L,))[idx].imag / _COMPLEX_STEP
                else:
                    real.append(o)
        except (TypeError, ValueError, _ComplexWarning):
            real = list(outputs)
        if real:
            h = 1e-6 * max(1.0, abs(base[i]))
            up, dn = np.full(L, base[i]), np.full(L, base[i])
            up[c] += h
            dn[c] -= h
            out_up, out_dn = self._evaluate(base, i, up, L), self._evaluate(base, i, dn, L)
            for o in real:
                diff = (np.broadcast_to(np.asarray(out_up[o], dtype=float), (L,))
                        - np.broadcast_to(np.asarray(out_dn[o], dtype=float), (L,)))
                d[o] = diff[idx] / (2 * h)
        return d


def simple(*outputs: str) -> Callable[[Callable], SimpleBlock]:
    """Decorator that turns a function into a :class:`SimpleBlock` with the given outputs."""
    if len(outputs) == 1 and not isinstance(outputs[0], str):
        outputs = tuple(outputs[0])

    def decorate(f: Callable) -> SimpleBlock:
        return SimpleBlock(f, list(outputs))
    return decorate


# ---------------------------------------------------------------------------------------------
# heterogeneous-agent blocks

class HetBlock:
    """A continuum of agents on a grid of assets and the nodes of a Markov chain.

    Parameters
    ----------
    step : callable
        One step of the backward iteration. Its arguments are read by name: ``v_p`` for each
        backward variable ``v`` is its expectation at ``t + 1`` conditional on today's income
        state, an ``(n_e, n_a)`` array, and every other argument is an aggregate input or an
        output of ``hetinputs``. It returns the backward variables at ``t``, the policy and the
        other individual outcomes, each an ``(n_e, n_a)`` array, in the order of ``outputs`` or
        as a mapping.
    outputs : sequence of str
        The names of what ``step`` returns. Every one that is not a backward variable is
        aggregated, under its name in capitals.
    backward : str or sequence of str
        The backward variables, such as the marginal value of assets.
    policy : str
        The asset policy, one of ``outputs``, whose lottery moves the distribution.
    init : callable
        The backward variables the steady-state iteration starts from, as a function of the
        inputs and the outputs of ``hetinputs``, returned in the order of ``backward``.
    markov : str
        The name of the income chain's transition matrix, ``(n_e, n_e)`` with rows that sum to
        one, among the inputs or the outputs of ``hetinputs``.
    grid : str, optional
        The name of the asset grid, by default ``policy + "_grid"``.
    hetinputs : sequence of (callable, outputs), optional
        Functions of the aggregate inputs that make the arrays ``step`` needs, such as the grids
        and income by node, run in order before each step. A :class:`SimpleBlock` is accepted
        for a pair.
    hetoutputs : sequence of (callable, outputs), optional
        Functions of the outcomes of ``step`` that make further individual outcomes, aggregated
        under their names in capitals.
    name : str, optional
        By default the name of ``step``.
    backward_tol, backward_max_iter : float, int
        The steady-state iteration stops when one step moves the policy by less than
        ``backward_tol`` at every point.
    h : float
        The step of the finite differences of ``step`` in the fake-news algorithm.
    twosided : bool
        Central differences when ``True``, forward differences when ``False``.

    Notes
    -----
    The block has one endogenous state, the asset, and one Markov chain. Several chains go into one
    by their Kronecker product. The transition matrix, and the income at each node, may be outputs
    of ``hetinputs`` that move with the inputs along a path, as when the risk of unemployment
    rises in a recession. The asset grid must stay fixed along a path: the lottery at ``t`` puts
    the policy on the grid of ``t + 1``, which one step does not see.
    """

    def __init__(self, step: Callable, outputs: Sequence[str], *, backward: str | Sequence[str],
                 policy: str, init: Callable, markov: str = "Pi", grid: str | None = None,
                 hetinputs: Iterable = (), hetoutputs: Iterable = (), name: str | None = None,
                 backward_tol: float = 1e-10, backward_max_iter: int = 10000, h: float = 1e-4,
                 twosided: bool = True):
        self.name: str = name if name else getattr(step, "__name__", "het")
        label = f"het block {self.name}"
        self.step = _Function(step, outputs, label)
        self.backward = _names(backward)
        for v in self.backward:
            if v not in self.step.outputs:
                raise ValueError(f"{label}: the backward variable {v!r} is not among the outputs")
            if v + "_p" not in self.step.inputs:
                raise ValueError(f"{label}: step must take {v + '_p'!r}, the expectation of {v!r}")
        if policy not in self.step.outputs or policy in self.backward:
            raise ValueError(f"{label}: the policy {policy!r} must be an output that is not a "
                             "backward variable")
        self.policy, self.markov = policy, markov
        self.grid = grid or policy + "_grid"
        self.hetinputs = _pairs(hetinputs, f"{label}: hetinput")
        self.hetoutputs = _pairs(hetoutputs, f"{label}: hetoutput")
        self.init = _Function(init, self.backward, f"{label}: init")
        self.backward_tol, self.backward_max_iter = backward_tol, backward_max_iter
        self.h, self.twosided = h, twosided

        made = [o for f in self.hetinputs for o in f.outputs]
        produced = set(made) | set(self.step.outputs) | {o for f in self.hetoutputs
                                                         for o in f.outputs}
        expected = {v + "_p" for v in self.backward}
        needed: list[str] = []
        for names in ([n for n in self.step.inputs if n not in expected],
                      *[f.inputs for f in self.hetinputs], *[f.inputs for f in self.hetoutputs],
                      self.init.inputs, [self.markov, self.grid]):
            needed += [n for n in names if n not in needed]
        self.inputs = [n for n in needed if n not in produced]
        self.individual = ([o for o in self.step.outputs if o not in self.backward]
                           + [o for f in self.hetoutputs for o in f.outputs])
        self.outputs = [o.upper() for o in self.individual]
        if len(set(self.outputs)) != len(self.outputs):
            raise ValueError(f"{label}: two outcomes have the same name in capitals: "
                             f"{self.individual}")
        clash = sorted(set(self.outputs) & set(self.inputs))
        if clash:
            raise ValueError(f"{label}: the aggregates {clash} are also inputs")
        self._aggregate = dict(zip(self.outputs, self.individual))

    def __repr__(self) -> str:
        return (f"HetBlock({self.name!r}, inputs={self.inputs}, outputs={self.outputs}, "
                f"policy={self.policy!r}, backward={self.backward})")

    # --- pieces ------------------------------------------------------------------------------

    def _inputs(self, values: Mapping[str, Any]) -> dict[str, Any]:
        x = {}
        for n in self.inputs:
            if n not in values:
                raise KeyError(f"het block {self.name}: no value for {n!r}")
            x[n] = values[n]
        return x

    def _run_hetinputs(self, x: dict, changed: set[str] | None = None) -> None:
        """Adds the outputs of the hetinputs to ``x``; with ``changed``, only those that depend on it."""
        for f in self.hetinputs:
            if changed is None or changed & set(f.inputs):
                x.update(f(x))
                if changed is not None:
                    changed |= set(f.outputs)

    def _exogenous(self, x: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        label = f"het block {self.name}"
        Pi = np.asarray(x[self.markov], dtype=float)
        grid = np.asarray(x[self.grid], dtype=float)
        if Pi.ndim != 2 or Pi.shape[0] != Pi.shape[1]:
            raise ValueError(f"{label}: {self.markov} must be a square matrix")
        if np.any(Pi < 0) or not np.allclose(Pi.sum(axis=1), 1.0, rtol=0.0, atol=1e-10):
            raise ValueError(f"{label}: {self.markov} must be nonnegative, with rows that sum to one")
        if grid.ndim != 1 or grid.size < 2 or np.any(np.diff(grid) <= 0):
            raise ValueError(f"{label}: {self.grid} must be an increasing vector of two or more points")
        return Pi, grid

    def _full(self, x: Mapping[str, Any]) -> dict[str, np.ndarray]:
        """``step`` and the hetoutputs at ``x``."""
        out = self.step(x)
        if self.hetoutputs:
            both = {**x, **out}
            for f in self.hetoutputs:
                new = f(both)
                both.update(new)
                out.update(new)
        return out

    def _check_shapes(self, out: Mapping[str, Any], shape: tuple[int, int]) -> None:
        for k in self.step.outputs:
            if np.shape(out[k]) != shape:
                raise ValueError(f"het block {self.name}: step returned {k} with shape "
                                 f"{np.shape(out[k])}, not {shape}, the income states by the grid")

    # --- steady state ------------------------------------------------------------------------

    def steady_state(self, values: Mapping[str, Any], start: Mapping[str, dict] | None = None):
        """The aggregates and the internals at the steady state of the block.

        ``start``, internals by block name such as a previous steady state's, gives the backward
        variables the iteration starts from.
        """
        x = self._inputs(values)
        self._run_hetinputs(x)
        Pi, grid = self._exogenous(x)
        shape = (Pi.shape[0], grid.shape[0])
        prev = (start or {}).get(self.name, {})
        if all(np.shape(prev.get(v)) == shape for v in self.backward):
            V = {v: np.asarray(prev[v], dtype=float) for v in self.backward}
        else:
            V = {v: np.broadcast_to(np.asarray(a, dtype=float), shape).copy()
                 for v, a in self.init(x).items()}
        out = self._backward_steady_state(x, Pi, V, shape)
        lottery = _Lottery(out[self.policy], grid)
        Dbeg = _stationary(Pi, lottery, prev.get("Dbeg"))
        D = Pi.T @ Dbeg
        if self.hetoutputs:
            both = {**x, **out}
            for f in self.hetoutputs:
                new = f(both)
                both.update(new)
                out.update(new)
        aggregates = {O: float(np.vdot(D, out[o])) for O, o in self._aggregate.items()}
        internals = {k: v for k, v in x.items() if k not in self.inputs}
        internals.update(out)
        internals.update(D=D, Dbeg=Dbeg)
        return aggregates, {self.name: internals}

    def _backward_steady_state(self, x, Pi, V, shape):
        x = dict(x)
        old = None
        for _ in range(self.backward_max_iter):
            for v in self.backward:
                x[v + "_p"] = Pi @ V[v]
            out = self.step(x)
            if old is None:
                self._check_shapes(out, shape)
            pol = np.asarray(out[self.policy], dtype=float)
            if not np.all(np.isfinite(pol)):
                raise FloatingPointError(f"het block {self.name}: the policy is not finite")
            if old is not None and np.max(np.abs(pol - old)) < self.backward_tol:
                return {k: np.asarray(v, dtype=float) for k, v in out.items()}
            old = pol
            V = {v: np.asarray(out[v], dtype=float) for v in self.backward}
        raise RuntimeError(f"het block {self.name}: the policy did not converge in "
                           f"{self.backward_max_iter} backward iterations")

    # --- paths -------------------------------------------------------------------------------

    def _at_steady_state(self, ss: BlockSteadyState):
        if self.name not in getattr(ss, "internals", {}):
            raise KeyError(f"het block {self.name}: the steady state has no internals for this "
                           "block; compute it with BlockModel.steady_state")
        internals = ss.internals[self.name]
        x = self._inputs(ss)
        for f in self.hetinputs:
            x.update({o: internals[o] for o in f.outputs})
        return x, internals

    def _moving_hetinputs(self, paths):
        """The inputs that move, with the hetinput outputs that depend on them."""
        changed = set(paths)
        for f in self.hetinputs:
            if changed & set(f.inputs):
                changed |= set(f.outputs)
        return changed

    def _check_fixed(self, x, x0, moved) -> None:
        """Raises if the asset grid at ``x`` differs from the steady state's."""
        if not np.array_equal(np.asarray(x[self.grid]), np.asarray(x0[self.grid])):
            raise NotImplementedError(f"het block {self.name}: {sorted(moved)} move {self.grid}, "
                                      "which must stay fixed along a path")

    def _initial(self, Dbeg: ArrayLike, shape: tuple[int, ...]) -> np.ndarray:
        """``Dbeg`` checked as a distribution on the block's grid."""
        label = f"het block {self.name}"
        D0 = np.asarray(Dbeg, dtype=float)
        if D0.shape != shape:
            raise ValueError(f"{label}: the initial distribution has shape {D0.shape}, not "
                             f"{shape}, the income states by the grid")
        if not np.all(np.isfinite(D0)) or np.any(D0 < 0) or abs(D0.sum() - 1.0) > 1e-10:
            raise ValueError(f"{label}: the initial distribution must be nonnegative and sum to one")
        return D0

    def path(self, ss: BlockSteadyState, paths: Mapping[str, np.ndarray], T: int, *,
             Dbeg: ArrayLike | None = None):
        """The aggregates' paths, in levels, given the paths of some inputs in levels.

        The backward iteration runs from ``T - 1``, where next period is the steady state. The
        distribution starts from ``Dbeg``, over last period's income state and the assets carried
        into the first period, by default the steady state's. The Markov matrix may move with the
        inputs, and the asset grid may not.
        """
        x0, internals = self._at_steady_state(ss)
        changed = self._moving_hetinputs(paths)
        Pi, grid = self._exogenous(x0)
        Pis = [Pi] * T              # the Markov matrix of each period
        V = {v: internals[v] for v in self.backward}
        policies: list[Any] = [None] * T
        outcomes: list[Any] = [None] * T
        for t in reversed(range(T)):
            x = dict(x0)
            for k, p in paths.items():
                x[k] = p[t]
            if changed - set(paths):
                self._run_hetinputs(x, set(paths))
                self._check_fixed(x, x0, paths)
            # the expectation at t is taken with the transition from t to t + 1
            Pi_next = Pis[t + 1] if t + 1 < T else Pi
            if self.markov in changed:
                Pis[t] = self._exogenous(x)[0]
            for v in self.backward:
                x[v + "_p"] = Pi_next @ V[v]
            out = self._full(x)
            policies[t] = np.asarray(out[self.policy], dtype=float)
            outcomes[t] = {o: out[o] for o in self.individual}
            V = {v: np.asarray(out[v], dtype=float) for v in self.backward}
        Y = {O: np.empty(T) for O in self.outputs}
        D_beg = (internals["Dbeg"] if Dbeg is None
                 else self._initial(Dbeg, np.shape(internals["Dbeg"])))
        for t in range(T):
            D = Pis[t].T @ D_beg
            for O, o in self._aggregate.items():
                Y[O][t] = np.vdot(D, outcomes[t][o])
            D_beg = _Lottery(policies[t], grid).forward(D)
        return Y

    # --- Jacobians ---------------------------------------------------------------------------

    def jacobian(self, ss: BlockSteadyState, inputs: Sequence[str], T: int,
                 outputs: Sequence[str] | None = None, *, method: str = "fake_news") -> Jacobians:
        """``J[O][i]``, the ``T x T`` Jacobian of aggregate ``O`` with respect to input ``i``.

        ``method="fake_news"`` uses the fake-news algorithm of Auclert et al. (2021), and
        ``"direct"`` differentiates the nonlinear path once for every period of the input, by
        central differences with step ``h``, which costs ``T`` times more and serves to check it.
        Inputs the block does not use are left out.
        """
        outputs = [o for o in (outputs or self.outputs) if o in self.outputs]
        inputs = [i for i in inputs if i in self.inputs]
        if method == "direct":
            return self._direct_jacobian(ss, inputs, T, outputs)
        if method != "fake_news":
            raise ValueError(f"method must be 'fake_news' or 'direct', not {method!r}")
        x0, internals = self._at_steady_state(ss)
        Pi, grid = self._exogenous(x0)
        for v in self.backward:
            x0[v + "_p"] = Pi @ internals[v]
        D, Dbeg = internals["D"], internals["Dbeg"]
        lottery = _Lottery(internals[self.policy], grid)
        indiv = {O: self._aggregate[O] for O in outputs}
        base = None if self.twosided else {**self._full(x0), _MARKOV: Pi}

        # step 2: the expectation vectors, demeaned (the curlyD sum to zero)
        E = {}
        for O, o in indiv.items():
            e = np.empty((max(T - 1, 0),) + D.shape)
            if T > 1:
                e[0] = Pi @ np.asarray(internals[o], dtype=float)
                e[0] -= e[0].mean()
                for k in range(1, T - 1):
                    e[k] = Pi @ lottery.expectation(e[k - 1])
                    e[k] -= e[k].mean()
            E[O] = e.reshape(max(T - 1, 0), -1)

        J: Jacobians = {O: {} for O in outputs}
        for i in inputs:
            # step 1: curlyY and curlyD for news s periods ahead, by one backward step each
            changed = self._moving_hetinputs({i})
            curlyY = {O: np.empty(T) for O in outputs}
            curlyD = np.empty((T, D.size))
            dV = None
            for s in range(T):
                d = self._step_derivative(x0, base, i if s == 0 else None, dV, changed)
                dV = {v: Pi @ d[v] for v in self.backward}
                curlyD[s] = lottery.shock(D, d[self.policy]).ravel()
                for O, o in indiv.items():
                    curlyY[O][s] = np.vdot(D, d[o])
                if s == 0 and _MARKOV in d:
                    # the input moves the Markov matrix, which moves the distribution today and
                    # the expectation of the backward variables a period earlier
                    dPi = d[_MARKOV]
                    for v in self.backward:
                        dV[v] = dV[v] + dPi @ internals[v]
                    curlyD[0] += lottery.forward(dPi.T @ Dbeg).ravel()
                    for O, o in indiv.items():
                        curlyY[O][0] += np.vdot(Dbeg, dPi @ np.asarray(internals[o], dtype=float))
            # steps 3 and 4: the fake-news matrix, and the Jacobian by its recursion
            for O in outputs:
                F = np.empty((T, T))
                F[0] = curlyY[O]
                if T > 1:
                    F[1:] = E[O] @ curlyD.T
                for t in range(1, T):
                    F[t, 1:] += F[t - 1, :-1]
                J[O][i] = F
        return {O: row for O, row in J.items() if row}

    def _step_derivative(self, x0, base, i, dV, changed):
        """The derivative of every outcome of one step, in the direction of an input or of dV.

        When the input moves the Markov matrix, the matrix's derivative is under ``_MARKOV``.
        """
        h = self.h

        def shocked(sign):
            x = dict(x0)
            if i is not None:
                x[i] = x0[i] + sign * h
                if changed - {i}:
                    self._run_hetinputs(x, {i})
                    self._check_fixed(x, x0, {i})
            if dV is not None:
                for v in self.backward:
                    x[v + "_p"] = x0[v + "_p"] + sign * h * dV[v]
            out = self._full(x)
            if i is not None and self.markov in changed:
                out[_MARKOV] = self._exogenous(x)[0]
            return out

        up = shocked(1.0)
        if self.twosided:
            dn = shocked(-1.0)
            return {k: (np.asarray(up[k]) - np.asarray(dn[k])) / (2 * h) for k in up}
        return {k: (np.asarray(up[k]) - np.asarray(base[k])) / h for k in up}

    def _direct_jacobian(self, ss, inputs, T, outputs):
        J: Jacobians = {O: {} for O in outputs}
        h = self.h
        for i in inputs:
            M = {O: np.empty((T, T)) for O in outputs}
            for s in range(T):
                up, dn = np.full(T, float(ss[i])), np.full(T, float(ss[i]))
                up[s] += h
                dn[s] -= h
                Yu, Yd = self.path(ss, {i: up}, T), self.path(ss, {i: dn}, T)
                for O in outputs:
                    M[O][:, s] = (Yu[O] - Yd[O]) / (2 * h)
            for O in outputs:
                J[O][i] = M[O]
        return J


def _stationary(Pi: np.ndarray, lottery: _Lottery, guess: np.ndarray | None = None) -> np.ndarray:
    """The stationary ``Dbeg``, the null vector of the transition minus the identity.

    The mass at one point ``k`` is set to one and the other equations are solved for the rest by a
    sparse solve, after which the masses are scaled to sum to one. The point must be recurrent, so
    ``k`` is where ``guess`` puts the most mass, or, without one, where 50 steps of the transition
    from a uniform distribution do.
    """
    n_e, n_a = lottery.shape
    N = n_e * n_a
    exog = sp.kron(sp.csr_matrix(Pi.T), sp.identity(n_a), format="csr")
    M = (lottery.matrix() @ exog).tocsc()
    if guess is not None and np.shape(guess) == lottery.shape:
        d = np.asarray(guess, dtype=float).ravel()
    else:
        d = np.full(N, 1.0 / N)
        for _ in range(50):
            d = M @ d
    k = int(np.argmax(d))
    keep = np.r_[0:k, k + 1:N]
    A = (sp.identity(N, format="csc") - M)[keep][:, keep]
    with warnings.catch_warnings():
        warnings.simplefilter("error", MatrixRankWarning)
        try:
            rest = spsolve(A.tocsc(), M[keep][:, [k]].toarray().ravel())
        except MatrixRankWarning:
            rest = np.full(N - 1, np.nan)
    d = np.insert(rest, k, 1.0)
    if not np.all(np.isfinite(d)) or np.max(np.abs(M @ d - d)) > 1e-10 * np.max(np.abs(d)):
        raise np.linalg.LinAlgError("the stationary distribution was not found: the transition "
                                    "may have more than one recurrent class")
    d = np.maximum(d, 0.0)
    return (d / d.sum()).reshape(n_e, n_a)


# ---------------------------------------------------------------------------------------------
# solved blocks and models

class SolvedBlock:
    """A model of blocks solved for some of its inputs, used as one block of a larger model.

    Parameters
    ----------
    blocks : block or sequence of blocks
    unknowns : mapping
        ``{name: guess}`` or ``{name: (lo, hi)}``, the inputs solved for, with a starting value
        or a bracket for the steady state.
    targets : sequence of str
        The outputs held at zero, as many as the unknowns.
    name : str, optional

    Notes
    -----
    The block's Jacobian is the inner model's general-equilibrium Jacobian ``G``, and along a path
    it solves the inner model by the quasi-Newton iteration of :meth:`BlockModel.impulse_nonlinear`.
    """

    def __init__(self, blocks, unknowns: Mapping[str, Any], targets: Sequence[str],
                 name: str | None = None):
        blocks = [blocks] if not isinstance(blocks, (list, tuple)) else list(blocks)
        self.name = name or "solved_" + "_".join(b.name for b in blocks)
        self.model = BlockModel(blocks, name=self.name)
        self.unknowns = dict(unknowns)
        self.targets = list(targets)
        if len(self.unknowns) != len(self.targets):
            raise ValueError(f"solved block {self.name}: {len(self.unknowns)} unknowns and "
                             f"{len(self.targets)} targets")
        bad = [u for u in self.unknowns if u not in self.model.inputs]
        if bad:
            raise ValueError(f"solved block {self.name}: the unknowns {bad} are not inputs")
        bad = [t for t in self.targets if t not in self.model.outputs]
        if bad:
            raise ValueError(f"solved block {self.name}: the targets {bad} are not outputs")
        self.inputs = [i for i in self.model.inputs if i not in self.unknowns]
        self.outputs = list(self.unknowns) + self.model.outputs
        self._lu = None

    def __repr__(self) -> str:
        return (f"SolvedBlock({self.name!r}, unknowns={list(self.unknowns)}, "
                f"targets={self.targets})")

    def steady_state(self, values: Mapping[str, Any], start: Mapping[str, dict] | None = None):
        # a value already known, from an earlier steady state, is a better guess than the given one
        guess = {u: g if isinstance(g, tuple) or u not in values else values[u]
                 for u, g in self.unknowns.items()}
        ss = self.model.steady_state(values, unknowns=guess,
                                     targets={t: 0.0 for t in self.targets}, start=start)
        return {o: ss[o] for o in self.outputs}, ss.internals

    def jacobian(self, ss, inputs, T, outputs=None) -> Jacobians:
        inputs = [i for i in inputs if i in self.inputs]
        if not inputs:
            return {}
        G = self.model.solve_jacobian(ss, list(self.unknowns), self.targets, inputs,
                                      outputs=outputs or self.outputs, T=T)
        return {o: row for o, row in G.items() if o in self.outputs and row}

    def path(self, ss, paths, T, *, initial=None):
        key = (id(ss), T)
        if self._lu is None or self._lu[0] != key or self._lu[1] is not ss:
            H = self.model._stacked(self.model._total(ss, list(self.unknowns), T),
                                    self.targets, list(self.unknowns), T)
            self._lu = (key, ss, lu_factor(H))
        shocks = {k: np.asarray(p, dtype=float) - ss[k] for k, p in paths.items()}
        if not shocks:
            # only an initial distribution moves the block: a path of zeros sets the horizon
            shocks = {self.inputs[0]: np.zeros(T)}
        if initial:
            inner = self.model._het_names()
            initial = {k: v for k, v in initial.items() if k in inner}
        dev = self.model.impulse_nonlinear(ss, list(self.unknowns), self.targets, shocks,
                                           initial=initial, _lu=self._lu[2])
        return {o: ss[o] + dev[o] for o in self.outputs if o in dev}


class _Identity:
    """The identity matrix in the chain rule, kept symbolic to skip the product."""


_I = _Identity()


class BlockModel:
    """A model made of blocks, joined by the names of their inputs and outputs.

    Parameters
    ----------
    blocks : sequence
        :class:`HetBlock`, :class:`SimpleBlock` and :class:`SolvedBlock` objects. Each output has
        one block that makes it, and the graph the names define must have no cycle.
    name : str, optional

    Notes
    -----
    The model's inputs are the inputs no block makes. A steady state evaluates the blocks in the
    graph's order, and can solve for some inputs to hit targets. The Jacobians, impulse responses
    and transitions are with respect to paths of inputs, over ``T`` periods after which the model
    is back at the steady state.
    """

    def __init__(self, blocks: Sequence, name: str = "model"):
        self.name = name
        blocks = list(blocks)
        if not blocks:
            raise ValueError("a model needs at least one block")
        names = [b.name for b in blocks]
        if len(set(names)) != len(names):
            raise ValueError(f"{name}: two blocks have the same name: {names}")
        maker: dict[str, Any] = {}
        for b in blocks:
            for o in b.outputs:
                if o in maker:
                    raise ValueError(f"{name}: {o!r} is an output of both {maker[o].name} and "
                                     f"{b.name}")
                maker[o] = b
        self.blocks = _order(blocks, maker, name)
        self.outputs = [o for b in self.blocks for o in b.outputs]
        self.inputs: list[str] = []
        for b in self.blocks:
            self.inputs += [i for i in b.inputs if i not in maker and i not in self.inputs]

    def __repr__(self) -> str:
        return (f"BlockModel({self.name!r}, blocks={[b.name for b in self.blocks]}, "
                f"inputs={self.inputs})")

    # --- steady state ------------------------------------------------------------------------

    def steady_state(self, calibration: Mapping[str, Any],
                     unknowns: Mapping[str, Any] | None = None,
                     targets: Mapping[str, float] | Sequence[str] | None = None, *,
                     solver: str | None = None, tol: float = 1e-12,
                     start: Mapping[str, dict] | None = None) -> BlockSteadyState:
        """The steady state at ``calibration``, solving for ``unknowns`` to hit ``targets``.

        Parameters
        ----------
        calibration : mapping
            Values of the inputs. A :class:`BlockSteadyState` also passes its internals as the
            starting point of the heterogeneous-agent blocks.
        unknowns : mapping, optional
            ``{name: guess}``, or ``{name: (lo, hi)}`` with one unknown, for a bracket.
        targets : mapping or sequence, optional
            ``{name: value}``, or names whose value is zero, as many as the unknowns.
        solver : str, optional
            ``"brentq"`` (the default with one bracketed unknown) or a method of
            ``scipy.optimize.root`` (``"hybr"`` by default).
        tol : float
            The solver's tolerance.
        start : mapping, optional
            Internals by block name to start the backward iterations from.
        """
        values = dict(calibration)
        if start is None and isinstance(calibration, BlockSteadyState):
            start = calibration.internals
        if not unknowns:
            return self._evaluate(values, start)
        unknowns = dict(unknowns)
        if targets is None:
            raise ValueError("unknowns need targets")
        targets = ({t: 0.0 for t in targets} if not isinstance(targets, Mapping)
                   else dict(targets))
        if len(targets) != len(unknowns):
            raise ValueError(f"{len(unknowns)} unknowns and {len(targets)} targets")
        names = list(unknowns)
        warm = {"internals": dict(start or {})}

        def residual(u):
            values.update(zip(names, np.atleast_1d(u)))
            ss = self._evaluate(values, warm["internals"])
            warm["internals"] = ss.internals
            missing = [t for t in targets if t not in ss]
            if missing:
                raise KeyError(f"{self.name}: the targets {missing} are not in the steady state")
            return np.array([ss[t] - v for t, v in targets.items()])

        bracketed = [u for u, g in unknowns.items() if isinstance(g, tuple)]
        if solver == "brentq" or (solver is None and len(names) == 1 and bracketed):
            if len(names) != 1 or not bracketed:
                raise ValueError("brentq solves for one unknown, given a bracket (lo, hi)")
            lo, hi = unknowns[names[0]]
            x = brentq(lambda u: residual(u)[0], lo, hi, xtol=tol, rtol=4 * np.finfo(float).eps)
            residual(x)
        else:
            if bracketed:
                raise ValueError(f"a bracket is given for {bracketed}, but the solver needs a "
                                 "starting value")
            res = root(residual, np.array([float(unknowns[u]) for u in names]),
                       method=solver or "hybr", tol=tol)
            values.update(zip(names, np.atleast_1d(res.x)))
        ss = self._evaluate(values, warm["internals"])
        err = max(abs(ss[t] - v) for t, v in targets.items())
        if not err < 1e-8:
            raise RuntimeError(f"{self.name}: the steady state was not found; the largest "
                               f"target residual is {err:.2e}")
        return ss

    def _evaluate(self, values: Mapping[str, Any], start=None) -> BlockSteadyState:
        vals = dict(values)
        internals: dict[str, dict] = {}
        for b in self.blocks:
            out, inner = b.steady_state(vals, start)
            vals.update(out)
            internals.update(inner)
        return BlockSteadyState(vals, internals)

    # --- Jacobians ---------------------------------------------------------------------------

    def _total(self, ss, sources: Sequence[str], T: int) -> dict[str, dict[str, Any]]:
        """The total Jacobians of every variable with respect to ``sources``, along the graph."""
        bad = [s for s in sources if s not in self.inputs]
        if bad:
            raise ValueError(f"{self.name}: {bad} are not inputs of the model")
        tot: dict[str, dict[str, Any]] = {s: {s: _I} for s in sources}
        for b in self.blocks:
            ins = [i for i in b.inputs if i in tot]
            if not ins:
                continue
            for o, row in b.jacobian(ss, ins, T).items():
                acc: dict[str, np.ndarray] = {}
                for i, Joi in row.items():
                    for s, Tis in tot[i].items():
                        term = Joi if Tis is _I else Joi @ Tis
                        acc[s] = acc[s] + term if s in acc else term
                if acc:
                    tot[o] = acc
        return tot

    @staticmethod
    def _stacked(tot, rows: Sequence[str], cols: Sequence[str], T: int) -> np.ndarray:
        M = np.zeros((len(rows) * T, len(cols) * T))
        for a, r in enumerate(rows):
            for b, c in enumerate(cols):
                J = tot.get(r, {}).get(c)
                if J is not None:
                    M[a * T:(a + 1) * T, b * T:(b + 1) * T] = np.eye(T) if J is _I else J
        return M

    def jacobian(self, ss: BlockSteadyState, inputs: Sequence[str],
                 outputs: Sequence[str] | None = None, T: int = 300) -> Jacobians:
        """``J[o][i]``, the Jacobians of the outputs with respect to the inputs, with no unknowns
        solved for: the chain rule along the graph."""
        tot = self._total(ss, list(inputs), T)
        outputs = list(outputs or self.outputs)
        return {o: {i: np.eye(T) if J is _I else J for i, J in tot[o].items()}
                for o in outputs if o in tot}

    def solve_jacobian(self, ss: BlockSteadyState, unknowns: Sequence[str],
                       targets: Sequence[str], inputs: Sequence[str],
                       outputs: Sequence[str] | None = None, T: int = 300) -> Jacobians:
        """``G[o][i]``, the general-equilibrium Jacobians: the outputs' response to the inputs
        when the unknowns move to keep the targets at zero.

        ``G`` includes the unknowns. With ``H_U`` and ``H_Z`` the targets' Jacobians with respect
        to the unknowns and to the inputs, ``G_U = -H_U^{-1} H_Z``.
        """
        unknowns, targets, inputs = list(unknowns), list(targets), list(inputs)
        if len(unknowns) != len(targets):
            raise ValueError(f"{len(unknowns)} unknowns and {len(targets)} targets")
        tot = self._total(ss, unknowns + inputs, T)
        H_U = self._stacked(tot, targets, unknowns, T)
        H_Z = self._stacked(tot, targets, inputs, T)
        try:
            G_U = -np.linalg.solve(H_U, H_Z)
        except np.linalg.LinAlgError:
            raise np.linalg.LinAlgError(f"{self.name}: H_U is singular; the targets do not "
                                        "determine the unknowns") from None
        G: Jacobians = {u: {z: G_U[a * T:(a + 1) * T, b * T:(b + 1) * T]
                            for b, z in enumerate(inputs)} for a, u in enumerate(unknowns)}
        outputs = [o for o in (outputs or self.outputs) if o not in unknowns]
        for o in outputs:
            row = tot.get(o, {})
            G[o] = {}
            for z in inputs:
                acc = None
                if z in row:
                    acc = np.eye(T) if row[z] is _I else row[z].copy()
                for u in unknowns:
                    if u in row:
                        term = G[u][z] if row[u] is _I else row[u] @ G[u][z]
                        acc = term if acc is None else acc + term
                G[o][z] = np.zeros((T, T)) if acc is None else acc
        return G

    # --- impulse responses -------------------------------------------------------------------

    @staticmethod
    def _horizon(shocks: Mapping[str, ArrayLike]) -> tuple[dict[str, np.ndarray], int]:
        shocks = {k: np.asarray(v, dtype=float) for k, v in shocks.items()}
        lengths = {v.shape for v in shocks.values()}
        if not shocks or len(lengths) != 1 or len(next(iter(lengths))) != 1:
            raise ValueError("shocks must be paths of one common length")
        return shocks, next(iter(shocks.values())).shape[0]

    def impulse_linear(self, ss: BlockSteadyState, unknowns: Sequence[str],
                       targets: Sequence[str], shocks: Mapping[str, ArrayLike],
                       outputs: Sequence[str] | None = None) -> dict[str, np.ndarray]:
        """The linear general-equilibrium response, in deviations from the steady state, to the
        paths of ``shocks``: ``G`` times the shocks. The horizon is the shocks' length."""
        shocks, T = self._horizon(shocks)
        G = self.solve_jacobian(ss, unknowns, targets, list(shocks), outputs, T)
        out = dict(shocks)
        for o, row in G.items():
            out[o] = sum((row[z] @ dz for z, dz in shocks.items()), np.zeros(T))
        return out

    def impulse_nonlinear(self, ss: BlockSteadyState, unknowns: Sequence[str],
                          targets: Sequence[str], shocks: Mapping[str, ArrayLike], *,
                          tol: float = 1e-10, max_iter: int = 50,
                          jacobian: np.ndarray | None = None,
                          initial: Mapping[str, ArrayLike] | None = None,
                          _lu=None) -> dict[str, np.ndarray]:
        """The nonlinear perfect-foresight transition after the paths of ``shocks``, in deviations
        from the steady state.

        The paths of the unknowns are found by a quasi-Newton iteration with the fixed matrix
        ``H_U``, the targets' Jacobian with respect to the unknowns at the steady state, until
        every target is below ``tol`` in every period. ``jacobian`` may give ``H_U``, stacked with
        one block of ``T`` rows per target and of ``T`` columns per unknown. ``initial`` maps the
        name of a :class:`HetBlock` to the distribution ``Dbeg`` it starts from, over last period's
        income state and the assets carried into the first period, in place of the steady state's,
        such as another steady state's ``internals[name]["Dbeg"]``. With shocks that are zero, the
        transition is the return from that distribution to the steady state. Lags at the first
        period keep their steady-state values.
        """
        shocks, T = self._horizon(shocks)
        unknowns, targets = list(unknowns), list(targets)
        if len(unknowns) != len(targets):
            raise ValueError(f"{len(unknowns)} unknowns and {len(targets)} targets")
        initial = dict(initial or {})
        bad = sorted(set(initial) - self._het_names())
        if bad:
            raise ValueError(f"{self.name}: {bad} in initial are not heterogeneous-agent blocks")
        if _lu is None:
            H = (np.asarray(jacobian, dtype=float) if jacobian is not None
                 else self._stacked(self._total(ss, unknowns, T), targets, unknowns, T))
            _lu = lu_factor(H)
        U = {u: np.zeros(T) for u in unknowns}
        err = np.inf
        for _ in range(max_iter):
            dev = self._paths(ss, {**shocks, **U}, T, initial)
            res = np.concatenate([dev.get(t, np.zeros(T)) for t in targets])
            err = float(np.max(np.abs(res)))
            if not np.isfinite(err):
                break
            if err < tol:
                return {**shocks, **U, **dev}
            step = lu_solve(_lu, res)
            for a, u in enumerate(unknowns):
                U[u] = U[u] - step[a * T:(a + 1) * T]
        raise RuntimeError(f"{self.name}: the transition did not converge in {max_iter} "
                           f"iterations; the largest target residual is {err:.2e}")

    def _het_names(self) -> set[str]:
        """The names of the heterogeneous-agent blocks, inside solved blocks too."""
        names: set[str] = set()
        for b in self.blocks:
            if isinstance(b, HetBlock):
                names.add(b.name)
            elif isinstance(b, SolvedBlock):
                names |= b.model._het_names()
        return names

    def _paths(self, ss, dev_inputs: Mapping[str, np.ndarray], T: int,
               initial: Mapping[str, ArrayLike] | None = None) -> dict[str, np.ndarray]:
        """Every variable's path, in deviations, given the deviations of some inputs and the
        initial distributions of some heterogeneous-agent blocks."""
        dev = dict(dev_inputs)
        for b in self.blocks:
            ins = [i for i in b.inputs if i in dev]
            paths = {i: ss[i] + dev[i] for i in ins}
            if isinstance(b, HetBlock) and initial and b.name in initial:
                levels = b.path(ss, paths, T, Dbeg=initial[b.name])
            elif isinstance(b, SolvedBlock) and initial and set(initial) & b.model._het_names():
                levels = b.path(ss, paths, T, initial=initial)
            elif ins:
                levels = b.path(ss, paths, T)
            else:
                continue
            for o, y in levels.items():
                dev[o] = y - ss[o]
        return dev


def _order(blocks: list, maker: Mapping[str, Any], name: str) -> list:
    """The blocks in an order in which every block comes after the ones that make its inputs."""
    remaining = list(blocks)
    done: set[int] = set()
    ordered = []
    while remaining:
        for b in remaining:
            if all(id(maker[i]) in done for i in b.inputs if i in maker and maker[i] is not b):
                ordered.append(b)
                done.add(id(b))
                remaining.remove(b)
                break
        else:
            raise ValueError(f"{name}: the blocks {[b.name for b in remaining]} form a cycle")
    for b in blocks:
        own = sorted(set(b.inputs) & set(b.outputs))
        if own:
            raise ValueError(f"{name}: block {b.name} takes its own outputs {own} as inputs")
    return ordered
