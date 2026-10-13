"""Model container: variable layout, the equation callback and its calling convention.

A model is declared by three lists of names -- predetermined states, forward
jumps, shocks -- and one function returning the equilibrium residuals::

    f(nxt, cur, eps, params) -> array of length n_states + n_jumps

``nxt`` and ``cur`` are :class:`VarView` objects, so the equations are written
with the variable NAMES rather than with positional indices. That is the whole
point of the layer: an index slip in a hand-packed residual vector is silent,
and it is the single most expensive class of error in this kind of code.

Timing convention. ``eps`` is the innovation dated ``t+1``, unknown at ``t``,
with ``E_t eps_{t+1} = 0``. An exogenous AR(1) is therefore written

    nxt["a"] - rho * cur["a"] - eps["eps_a"]

and the resulting law of motion is ``k_{t+1} = P k_t + Q eps_{t+1}``. Writing it
the other way round -- treating the innovation as known at ``t`` -- makes the
model deterministic within the period and silently kills every risk correction
at second order.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, fields, is_dataclass, replace
from typing import Any, Callable, Iterable, Mapping, Sequence, TypeAlias

import numpy as np
from numpy.typing import ArrayLike

__all__ = ["Model", "VarView", "get_param", "set_params"]

#: Parameters as the routines accept them: a dataclass, a mapping or any object
#: with attributes, read through :func:`get_param` and copied by :func:`set_params`.
Params: TypeAlias = Any
#: A point given by variable name or as a vector in the model's variable order.
Point: TypeAlias = Mapping[str, float] | ArrayLike
#: A shock covariance: standard deviations by shock name, a vector of them, a
#: matrix, or ``None`` for the identity (see :func:`dsgelab.shock_covariance`).
Covariance: TypeAlias = Mapping[str, float] | ArrayLike | None


class VarView:
    """Name-indexed read-only view over one period's variable vector."""

    __slots__ = ("_x", "_idx")

    def __init__(self, x: np.ndarray, index: Mapping[str, int]):
        self._x = x
        self._idx = index

    def __getitem__(self, name: str) -> Any:
        try:
            return self._x[self._idx[name]]
        except KeyError:
            raise KeyError(f"unknown variable {name!r}; known: {list(self._idx)}") from None

    def __getattr__(self, name: str) -> Any:
        try:
            return self._x[self._idx[name]]
        except KeyError:
            raise AttributeError(f"unknown variable {name!r}") from None

    def __contains__(self, name: str) -> bool:
        return name in self._idx

    def get(self, name: str, default: Any = None) -> Any:
        i = self._idx.get(name)
        return default if i is None else self._x[i]

    def vec(self, prefix: str, n: int | None = None) -> np.ndarray:
        """Gather an indexed family ``prefix0, prefix1, ...`` into one array.

        The convenience that makes multi-group models readable: a model with N
        income groups declares ``K0..K{N-1}`` and the equations then work with
        ``cur.vec("K")`` as a vector.
        """
        if n is None:
            n = 0
            while f"{prefix}{n}" in self._idx:
                n += 1
        return np.array([self._x[self._idx[f"{prefix}{i}"]] for i in range(n)])

    def to_dict(self) -> dict[str, float]:
        return {k: float(self._x[i]) for k, i in self._idx.items()}

    @property
    def array(self) -> np.ndarray:
        return self._x

    def __repr__(self) -> str:
        return f"VarView({self.to_dict()})"


def get_param(params: Params, name: str) -> Any:
    """Read a parameter from a dataclass, a mapping or an arbitrary object."""
    if isinstance(params, Mapping):
        return params[name]
    return getattr(params, name)


def set_params(params: Params, updates: Mapping[str, float]) -> Params:
    """Return a COPY of ``params`` with ``updates`` applied.

    Copying rather than mutating is not fastidiousness: calibration and policy
    sweeps evaluate the same parameter object many times, and an in-place update
    makes every earlier result silently refer to the latest values.
    """
    if isinstance(params, Mapping):
        out = dict(params)
        out.update(updates)
        return out
    if is_dataclass(params):
        known = {f.name for f in fields(params)}
        unknown = set(updates) - known
        if unknown:
            raise KeyError(f"not fields of {type(params).__name__}: {sorted(unknown)}")
        return replace(params, **updates)
    import copy
    out = copy.copy(params)
    for k, v in updates.items():
        if not hasattr(out, k):
            raise KeyError(f"{type(params).__name__} has no attribute {k!r}")
        setattr(out, k, v)
    return out


@dataclass
class Model:
    """A DSGE model in implicit form.

    Parameters
    ----------
    states : sequence of str
        Predetermined variables, in the order they occupy the vector. Exogenous
        processes belong here.
    jumps : sequence of str
        Non-predetermined (forward-looking) variables.
    shocks : sequence of str
        Innovation names, dated ``t+1``.
    equations : callable
        ``f(nxt, cur, eps, params)``, or ``f(nxt, cur, eps, params, ctx)`` when
        the residuals need side information (a steady-state anchor, a
        time-varying policy). The signature is inspected once, so both forms are
        accepted without a flag.
    name : str, optional
    """

    states: Sequence[str]
    jumps: Sequence[str]
    shocks: Sequence[str]
    equations: Callable
    name: str = "model"
    #: Optional exact derivatives, in MODEL LEVELS, with respect to the stacked
    #: argument ``w = [x_next; x_now; eps]``. Supplied -- which is what
    #: :mod:`dsgelab.symbolic` does -- they replace finite differences
    #: throughout, which matters most at second order, where a difference
    #: quotient divides by the square of the step and loses roughly six digits.
    jacobian_fn: Callable | None = None
    hessian_fn: Callable | None = None

    def __post_init__(self) -> None:
        self.states = list(self.states)
        self.jumps = list(self.jumps)
        self.shocks = list(self.shocks)
        names = self.states + self.jumps
        dup = {n for n in names if names.count(n) > 1}
        if dup:
            raise ValueError(f"duplicated variable names: {sorted(dup)}")
        dup = {s for s in self.shocks if self.shocks.count(s) > 1}
        if dup:
            raise ValueError(f"duplicated shock names: {sorted(dup)}")
        self._index = {n: i for i, n in enumerate(names)}
        self._shock_index = {s: i for i, s in enumerate(self.shocks)}
        sig = inspect.signature(self.equations)
        n_pos = sum(1 for p in sig.parameters.values()
                    if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD))
        has_var = any(p.kind is p.VAR_POSITIONAL for p in sig.parameters.values())
        self._wants_ctx = has_var or n_pos >= 5
        if not has_var and n_pos < 4:
            raise TypeError(
                "equations must accept at least (nxt, cur, eps, params); "
                f"got {n_pos} positional parameters")

    # -- layout ----------------------------------------------------------
    @property
    def names(self) -> list[str]:
        return list(self.states) + list(self.jumps)

    @property
    def n_states(self) -> int:
        return len(self.states)

    @property
    def n_jumps(self) -> int:
        return len(self.jumps)

    @property
    def n_vars(self) -> int:
        return len(self.states) + len(self.jumps)

    @property
    def n_shocks(self) -> int:
        return len(self.shocks)

    @property
    def index(self) -> dict[str, int]:
        return dict(self._index)

    def position(self, name: str) -> int:
        try:
            return self._index[name]
        except KeyError:
            raise KeyError(f"unknown variable {name!r}") from None

    def shock_position(self, name: str) -> int:
        try:
            return self._shock_index[name]
        except KeyError:
            raise KeyError(f"unknown shock {name!r}") from None

    # -- evaluation ------------------------------------------------------
    def residuals(self, x_next: ArrayLike, x_now: ArrayLike, eps: ArrayLike | None,
                  params: Params, ctx: Any = None) -> np.ndarray:
        """Evaluate the equilibrium residuals at a point."""
        x_next = np.asarray(x_next, dtype=float)
        x_now = np.asarray(x_now, dtype=float)
        eps = np.zeros(self.n_shocks) if eps is None else np.asarray(eps, dtype=float)
        nxt = VarView(x_next, self._index)
        cur = VarView(x_now, self._index)
        e = VarView(eps, self._shock_index)
        r = (self.equations(nxt, cur, e, params, ctx) if self._wants_ctx
             else self.equations(nxt, cur, e, params))
        r = np.asarray(r, dtype=float).ravel()
        if r.size != self.n_vars:
            raise ValueError(
                f"{self.name}: equations returned {r.size} residuals for "
                f"{self.n_vars} variables ({self.n_states} states + {self.n_jumps} jumps)")
        return r

    def jacobian(self, x_next: ArrayLike, x_now: ArrayLike, eps: ArrayLike | None,
                 params: Params, ctx: Any = None) -> np.ndarray | None:
        """Exact ``(n_eq, 2n + n_shocks)`` derivative matrix, or ``None`` if unavailable."""
        if self.jacobian_fn is None:
            return None
        eps = np.zeros(self.n_shocks) if eps is None else np.asarray(eps, dtype=float)
        J = np.asarray(self.jacobian_fn(np.asarray(x_next, dtype=float),
                                        np.asarray(x_now, dtype=float),
                                        eps, params, ctx), dtype=float)
        want = (self.n_vars, 2 * self.n_vars + self.n_shocks)
        if J.shape != want:
            raise ValueError(f"jacobian_fn returned {J.shape}, expected {want}")
        return J

    def hessian(self, x_next: ArrayLike, x_now: ArrayLike, eps: ArrayLike | None,
                params: Params, ctx: Any = None) -> np.ndarray | None:
        """Exact ``(n_eq, m, m)`` second derivatives with ``m = 2n + n_shocks``, or ``None``."""
        if self.hessian_fn is None:
            return None
        eps = np.zeros(self.n_shocks) if eps is None else np.asarray(eps, dtype=float)
        H = np.asarray(self.hessian_fn(np.asarray(x_next, dtype=float),
                                       np.asarray(x_now, dtype=float),
                                       eps, params, ctx), dtype=float)
        m = 2 * self.n_vars + self.n_shocks
        if H.shape != (self.n_vars, m, m):
            raise ValueError(f"hessian_fn returned {H.shape}, expected {(self.n_vars, m, m)}")
        return H

    def vector(self, values: Mapping[str, float]) -> np.ndarray:
        """Pack a name->value mapping into the model's variable order."""
        missing = [n for n in self.names if n not in values]
        if missing:
            raise KeyError(f"missing values for {missing}")
        return np.array([float(values[n]) for n in self.names])

    def dict(self, x: ArrayLike) -> dict[str, float]:
        """Unpack a variable vector into a name->value mapping."""
        x = np.asarray(x, dtype=float)
        return {n: float(x[i]) for i, n in enumerate(self.names)}

    def __repr__(self) -> str:
        return (f"Model({self.name!r}, states={self.n_states}, jumps={self.n_jumps}, "
                f"shocks={self.n_shocks})")
