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
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

from .model import Model
from .steady import SteadyState

__all__ = ["TransitionPath", "perfect_foresight", "path_of", "News",
           "perfect_foresight_with_expectation_errors"]


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


def _shock_path(shocks, T: int, n_shocks: int) -> np.ndarray:
    """One innovation vector per transition, shape ``(T, n_shocks)``."""
    if shocks is None:
        return np.zeros((T, n_shocks))
    arr = np.asarray(shocks, dtype=float)
    if arr.ndim == 1:
        if n_shocks != 1:
            raise ValueError(f"shock path is 1-d but the model has {n_shocks} shocks")
        arr = arr.reshape(-1, 1)
    if arr.shape[0] == T + 1:
        arr = arr[:T]
    if arr.shape != (T, n_shocks):
        raise ValueError(
            f"shocks must have shape ({T}, {n_shocks}) or ({T + 1}, {n_shocks}), "
            f"got {arr.shape}. Entry t is the innovation linking date t to t+1.")
    return arr


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
                      shocks=None, maxiter: int = 50, tol: float = 1e-10,
                      damping: float = 1.0, jacobian: str = "auto", h: float = 1e-7,
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
    jacobian : {"auto", "exact", "forward", "central"}
        The Newton Jacobian: the model's exact derivatives (``"auto"`` uses them
        when the model has them, else forward differences), or a difference
        scheme. Forward halves the cost of central and does not affect the
        accuracy of the converged path, which is governed by ``tol``; central
        helps when the residuals are noisy.
    shocks : array, optional
        Innovations dated with the transition they enter. Shape ``(T, n_shocks)``
        or ``(T+1, n_shocks)``; entry ``t`` is ``eps`` in the residual that links
        date ``t`` to date ``t+1``. Absent, every innovation is zero.
    """
    n = model.n_vars
    nk = model.n_states
    nu = model.n_jumps
    params_path = path_of(params, T + 1)
    ctx_path = path_of(ctx, T + 1)
    shock_path = _shock_path(shocks, T, model.n_shocks)

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
            r[t * n:(t + 1) * n] = model.residuals(Yn[t + 1], Yn[t], shock_path[t],
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
        J = _stacked_jacobian(model, Yn, T, nk, nu, z.size, shock_path, params_path,
                              ctx_path, jacobian, h)
        dz = spsolve(J, -r)
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


def _period_jacobian(model, ytp, yt, et, pt, ct, jacobian, h):
    """``(dF/dy_{t+1}, dF/dy_t)`` of one period's residuals."""
    n = model.n_vars
    if jacobian in ("auto", "exact"):
        Jx = model.jacobian(ytp, yt, et, pt, ct)
        if Jx is not None:
            return Jx[:, :n], Jx[:, n:2 * n]
        if jacobian == "exact":
            raise ValueError("jacobian='exact': the model has no exact derivatives")
    base = model.residuals(ytp, yt, et, pt, ct)
    d_next, d_now = np.empty((n, n)), np.empty((n, n))
    for j in range(n):
        if jacobian == "central":
            a = ytp.copy(); a[j] += h
            b = ytp.copy(); b[j] -= h
            d_next[:, j] = (model.residuals(a, yt, et, pt, ct)
                            - model.residuals(b, yt, et, pt, ct)) / (2 * h)
            a = yt.copy(); a[j] += h
            b = yt.copy(); b[j] -= h
            d_now[:, j] = (model.residuals(ytp, a, et, pt, ct)
                           - model.residuals(ytp, b, et, pt, ct)) / (2 * h)
        else:
            a = ytp.copy(); a[j] += h
            d_next[:, j] = (model.residuals(a, yt, et, pt, ct) - base) / h
            a = yt.copy(); a[j] += h
            d_now[:, j] = (model.residuals(ytp, a, et, pt, ct) - base) / h
    return d_next, d_now


def _stacked_jacobian(model, Yn, T, nk, nu, n_unknowns, shock_path, params_path, ctx_path,
                      jacobian, h):
    """The sparse Jacobian of the stacked residuals in the unknowns ``[u_0, y_1, ..., y_T]``."""
    n = model.n_vars
    rows, cols, vals = [], [], []
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    for t in range(T):
        d_next, d_now = _period_jacobian(model, Yn[t + 1], Yn[t], shock_path[t],
                                         params_path[t], ctx_path[t], jacobian, h)
        rows.append((t * n + ii).ravel()); cols.append((nu + t * n + jj).ravel())
        vals.append(d_next.ravel())
        if t == 0:                           # only the jumps are unknown at t=0
            rows.append((ii[:, nk:]).ravel()); cols.append((jj[:, nk:] - nk).ravel())
            vals.append(d_now[:, nk:].ravel())
        else:
            rows.append((t * n + ii).ravel()); cols.append((nu + (t - 1) * n + jj).ravel())
            vals.append(d_now.ravel())
    rows.append(T * n + np.arange(nu)); cols.append(nu + (T - 1) * n + nk + np.arange(nu))
    vals.append(np.ones(nu))
    return csr_matrix(coo_matrix((np.concatenate(vals), (np.concatenate(rows),
                                                         np.concatenate(cols))),
                                 shape=(T * n + nu, n_unknowns)))


# ------------------------------------------------------- expectation errors
@dataclass
class News:
    """Information that arrives at date ``period`` and replaces what agents expected.

    Every field left at ``None`` keeps the previous information set's value.

    Attributes
    ----------
    period : int
        The date, on the transition's own clock (``0`` is its first date), at
        which agents learn this and re-plan.
    shocks : array, optional
        The innovations agents now expect, on the ABSOLUTE clock: entry ``t``
        links date ``t`` to date ``t+1``, exactly as in :func:`perfect_foresight`.
        Entry ``period - 1`` is the innovation that has just been realised, so a
        surprise at ``period`` is a revision of that entry.
    params : object or sequence, optional
        Parameters from ``period`` on: a constant, or a path on the absolute
        clock. A permanent change -- a new tax rate, a new level of an exogenous
        variable -- is a change here together with a new ``terminal``.
    terminal : SteadyState, optional
        The steady state agents now expect to converge to.
    states : mapping, optional
        Levels of states revealed at ``period`` directly, overriding the ones the
        realised path implies. :mod:`dsgelab.dynare` uses it for an exogenous
        variable whose value at the date of the news is itself news.
    """

    period: int
    shocks: object = None
    params: object = None
    terminal: SteadyState | None = None
    states: dict | None = None


def _expost_states(model: Model, y_prev, y_guess, eps, params, ctx, h: float = 1e-7,
                   tol: float = 1e-12, maxiter: int = 50):
    """States at ``t`` from the realised ``y_{t-1}`` and the innovation into ``t``.

    The rows of the residual with no next-period jump -- the transitions -- pin
    the states down once the past is known. Newton on those rows alone.
    """
    nk = model.n_states
    if nk == 0:
        return np.zeros(0)
    y = np.array(y_guess, dtype=float)
    base = model.residuals(y, y_prev, eps, params, ctx)

    def sens(cols_range):
        cols = []
        for j in cols_range:
            a = y.copy(); a[j] += h
            cols.append((model.residuals(a, y_prev, eps, params, ctx) - base) / h)
        return np.max(np.abs(np.column_stack(cols)), axis=1) if cols else np.zeros(len(base))

    dj = sens(range(nk, model.n_vars))          # next-period jumps
    dk = sens(range(nk))                        # next-period states
    floor = 1e-9 * max(1.0, float(np.max(np.r_[dj, dk], initial=0.0)))
    # a transition: no next-period jump, and it does determine a next-period state
    rows = np.where((dj < floor) & (dk > floor))[0]
    if rows.size != nk:
        raise RuntimeError(f"expected {nk} transition rows free of next-period jumps, "
                           f"found {rows.size}; pass the news' states= explicitly")
    k = y[:nk].copy()
    for _ in range(maxiter):
        y[:nk] = k
        r = model.residuals(y, y_prev, eps, params, ctx)[rows]
        Jx = model.jacobian(y, y_prev, eps, params, ctx)
        if Jx is not None:
            J = Jx[np.ix_(rows, np.arange(nk))]
        else:
            J = np.empty((nk, nk))
            for j in range(nk):
                a = y.copy(); a[j] += h
                J[:, j] = (model.residuals(a, y_prev, eps, params, ctx)[rows] - r) / h
        step = np.linalg.solve(J, r)
        k = k - step
        if np.max(np.abs(r)) < tol or np.max(np.abs(step)) <= 1e-15 * (1 + np.max(np.abs(k))):
            break
    return k


def perfect_foresight_with_expectation_errors(
        model: Model, start: SteadyState, terminal: SteadyState, *, params, T: int,
        news: Sequence[News], shocks=None, ctx=None, initial_states=None,
        constant_simulation_length: bool = False, **kwargs) -> TransitionPath:
    """Perfect foresight in which agents learn, at given dates, that they were wrong.

    At each date in ``news`` the agents take the states the realised path has
    brought them to, form a new perfect-foresight plan with the information
    they now have, and follow it until the next date of news. The path returned
    is the REALISED one, spliced from those plans -- Dynare's
    ``perfect_foresight_with_expectation_errors_solver``.

    ``shocks``, ``params`` and ``terminal`` are the information held before the
    first news; a news item at ``period=0`` replaces them from the start. With
    ``constant_simulation_length`` every plan runs ``T`` periods past its own
    date and the path is longer; otherwise every plan ends at date ``T``, as in
    Dynare by default.

    The plans are kept in ``result.plans``, each a :class:`TransitionPath` on
    the absolute clock offset by ``result.plan_dates``.
    """
    items = sorted(news, key=lambda x: x.period)
    if any(x.period < 0 or x.period >= T for x in items):
        raise ValueError(f"news must arrive at a date in [0, {T})")
    if len({x.period for x in items}) != len(items):
        raise ValueError("two news items at the same date: merge them")
    last = items[-1].period if items else 0
    L = (last + T) if constant_simulation_length else T

    def shocks_abs(arr):
        if arr is None:
            return np.zeros((L, model.n_shocks))
        a = np.asarray(arr, dtype=float)
        if a.ndim == 1:
            a = a.reshape(-1, 1) if model.n_shocks == 1 else a.reshape(1, -1)
        out = np.zeros((L, model.n_shocks))
        out[:min(L, a.shape[0])] = a[:L]
        return out

    def params_abs(p):
        if isinstance(p, (list, tuple)) or (isinstance(p, np.ndarray) and p.ndim >= 1):
            seq = list(p)
            return seq + [seq[-1]] * max(0, L + 1 - len(seq))
        if callable(p):
            return [p(t) for t in range(L + 1)]
        return [p] * (L + 1)

    info = dict(shocks=shocks_abs(shocks), params=params_abs(params), terminal=terminal)
    ctx_abs = path_of(ctx, L + 1) if not callable(ctx) else [ctx(t) for t in range(L + 1)]
    levels = np.zeros((L + 1, model.n_vars))
    dates = [0] + [x.period for x in items if x.period > 0]
    by_date = {x.period: x for x in items}
    plans, prev_plan = [], None
    nk = model.n_states
    for i, tau in enumerate(dates):
        item = by_date.get(tau)
        if item is not None:
            if item.shocks is not None:
                info["shocks"] = shocks_abs(item.shocks)
            if item.params is not None:
                fresh = params_abs(item.params)
                info["params"] = info["params"][:tau] + fresh[tau:] \
                    if len(fresh) == L + 1 else info["params"][:tau] + fresh
            if item.terminal is not None:
                info["terminal"] = item.terminal
        if tau == 0:
            k0 = (np.asarray(initial_states, dtype=float) if initial_states is not None
                  else start.x[:nk].copy())
        else:
            k0 = _expost_states(model, levels[tau - 1], prev_plan.levels[tau - prev_start],
                                info["shocks"][tau - 1], info["params"][tau - 1],
                                ctx_abs[tau - 1])
        if item is not None and item.states:
            for name, v in item.states.items():
                k0[model.states.index(name)] = float(v)
        end = (tau + T) if constant_simulation_length else T
        n_t = end - tau
        guess = None
        if prev_plan is not None:
            off = tau - prev_start
            g = prev_plan.levels[off:]
            if g.shape[0] >= n_t + 1:
                guess = g[:n_t + 1].copy()
            else:
                guess = np.vstack([g, np.tile(info["terminal"].x, (n_t + 1 - g.shape[0], 1))])
        plan = perfect_foresight(model, start, info["terminal"], params=info["params"][tau:end + 1],
                                 T=n_t, ctx=ctx_abs[tau:end + 1], initial_states=k0,
                                 shocks=info["shocks"][tau:end], guess=guess, **kwargs)
        stop = dates[i + 1] if i + 1 < len(dates) else end + 1
        levels[tau:stop] = plan.levels[:stop - tau]
        plans.append(plan)
        prev_plan, prev_start = plan, tau
    out = TransitionPath(levels=levels, model=model, converged=all(p.converged for p in plans),
                         iterations=sum(p.iterations for p in plans),
                         residual=max(p.residual for p in plans),
                         params_path=info["params"], ctx_path=ctx_abs)
    out.plans = plans
    out.plan_dates = dates
    return out
