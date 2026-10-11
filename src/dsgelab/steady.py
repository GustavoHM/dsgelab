"""Deterministic steady state of the declared system.

The steady state is the ``y`` solving ``f(y, y, 0) = 0``. Two facilities beyond
a plain root-find earn their place, because both recur in applied models:

* **Positivity.** Capital stocks, prices and quantities cannot be solved on the
  real line -- the Newton step walks the iterate negative and the residual
  returns a complex number or a NaN. ``positive=`` solves those coordinates in
  logs, which imposes the constraint exactly instead of by a clip.
* **Anchors.** Some residuals need side information that is itself a function of
  the steady state: a portfolio adjustment cost centred on steady-state wealth,
  an exogenous level pinned to equilibrium demand. Substituting the solution
  into that term directly would make it vanish and leave the corresponding
  margin indeterminate, so it is carried in a ``ctx`` and updated by an OUTER
  loop until it stops moving.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np
from scipy.optimize import root

from .model import Model

__all__ = ["SteadyState", "steady_state", "analytic_steady_state", "SteadyStateError"]


@dataclass
class SteadyState:
    """Solved steady state and the evidence that it is one."""

    x: np.ndarray
    values: dict
    residual: float
    ctx: object = None
    outer_iterations: int = 0
    success: bool = True
    message: str = ""
    model: Model | None = field(default=None, repr=False)

    def __getitem__(self, name: str) -> float:
        return self.values[name]

    def __contains__(self, name: str) -> bool:
        return name in self.values

    def get(self, name: str, default=None):
        return self.values.get(name, default)

    def scale(self, floor: float = 1e-8) -> np.ndarray:
        """Column scaling for linearisation: the steady-state level, or 1 where it is zero.

        Variables whose steady state is zero -- a mean-zero cost shifter, a gap
        defined as a deviation -- cannot be expressed in proportional deviations.
        They are carried in ABSOLUTE deviations instead, and this vector is what
        records which is which.
        """
        return np.where(np.abs(self.x) > floor, np.abs(self.x), 1.0)

    def is_proportional(self, floor: float = 1e-8) -> np.ndarray:
        return np.abs(self.x) > floor


def steady_state(model: Model, guess, params, *, ctx=None,
                 ctx_update: Callable | None = None,
                 positive: Sequence[str] = (),
                 method: str = "hybr", tol: float = 1e-11,
                 outer_maxiter: int = 60, outer_tol: float = 1e-12,
                 raise_on_fail: bool = True,
                 calibrate: Mapping[str, str] | None = None,
                 homotopy: Mapping[str, tuple] | None = None,
                 homotopy_steps: int = 10) -> SteadyState:
    """Solve ``f(y, y, 0) = 0``.

    Parameters
    ----------
    guess : mapping, array or None
        Starting point, in the model's variable order if an array. ``None`` uses
        the model's declared ``steady_state_model`` (see
        :func:`analytic_steady_state`), checked and not solved.
    ctx : object, optional
        Side information handed to the equations as their fifth argument.
    ctx_update : callable, optional
        ``ctx_update(values, ctx) -> ctx`` run after each inner solve. The loop
        stops when the returned context stops changing, measured by
        :func:`_ctx_distance`, or after ``outer_maxiter`` rounds.
    positive : sequence of str
        Variables solved in logs, which keeps them strictly positive.
    tol : float
        Accepted maximum absolute residual. Judged on the RESIDUAL, never on the
        solver's success flag: ``hybr`` reports failure whenever its
        finite-difference Jacobian stops improving, and that includes the case
        where the starting point already is the solution.
    calibrate : mapping, optional
        Parameters pinned down by steady-state conditions, solved TOGETHER with
        the steady state -- MacroModelling.jl's calibration equations:
        ``{"beta": "k/(4*y) = 2.5"}`` makes ``beta`` whatever puts the capital-
        output ratio at 2.5. The conditions are written in the model's variable
        and parameter names, at the steady state. The parameters' values in
        ``params`` are the starting point. The result carries ``.calibrated``
        (the solved parameters) and ``.params`` (a copy of ``params`` with them).
        A model built with ``calibration_equations=`` uses its own by default.
    homotopy : mapping, optional
        ``{param: (start, end)}`` -- Dynare's ``homotopy_setup``. The steady state
        is first solved with the parameters at ``start``, where ``guess`` is
        good, and then followed as they move together to ``end`` in
        ``homotopy_steps`` steps, each warm-started from the last; a step that
        fails is halved, one that succeeds lets the next grow. ``end`` replaces
        the value in ``params``. The result carries ``.homotopy_path``, the
        parameter shares visited.
    """
    if homotopy:
        return _continuation(model, guess, params, dict(homotopy), homotopy_steps,
                             dict(ctx=ctx, ctx_update=ctx_update, positive=positive,
                                  method=method, tol=tol, outer_maxiter=outer_maxiter,
                                  outer_tol=outer_tol, calibrate=calibrate))
    if calibrate is None:
        calibrate = getattr(model, "calibration_equations", None)
    if calibrate:
        return _calibrated(model, guess, params, dict(calibrate), ctx=ctx, positive=positive,
                           tol=tol, raise_on_fail=raise_on_fail)
    if guess is None:
        program = getattr(model, "steady_state_model", None)
        if program is None:
            raise ValueError("no guess, and the model declares no steady_state_model")
        return analytic_steady_state(model, params, program, ctx=ctx, tol=tol)
    x0 = model.vector(guess) if isinstance(guess, Mapping) else np.asarray(guess, dtype=float)
    if x0.size != model.n_vars:
        raise ValueError(f"guess has {x0.size} entries for {model.n_vars} variables")

    log_idx = np.array([model.position(n) for n in positive], dtype=int)
    if log_idx.size and np.any(x0[log_idx] <= 0):
        bad = [n for n in positive if x0[model.position(n)] <= 0]
        raise ValueError(f"non-positive starting values for log-solved variables: {bad}")

    def pack(x):
        z = x.copy()
        if log_idx.size:
            z[log_idx] = np.log(x[log_idx])
        return z

    def unpack(z):
        x = z.copy()
        if log_idx.size:
            x[log_idx] = np.exp(z[log_idx])
        return x

    zero = np.zeros(model.n_shocks)
    ctx_now = ctx
    z = pack(x0)
    outer = 0
    sol = None

    for outer in range(1, outer_maxiter + 1):
        def resid(zz, _ctx=None):
            xx = unpack(zz)
            return model.residuals(xx, xx, zero, params, ctx_now)

        sol = root(resid, z, method=method, tol=1e-15)
        z = sol.x
        x = unpack(z)
        if ctx_update is None:
            break
        values = model.dict(x)
        ctx_new = ctx_update(values, ctx_now)
        moved = _ctx_distance(ctx_now, ctx_new)
        ctx_now = ctx_new
        if moved < outer_tol:
            break
    else:
        if raise_on_fail:
            raise RuntimeError(
                f"steady_state: the outer context loop did not settle in {outer_maxiter} "
                f"rounds (last move {moved:.3e})")

    x = unpack(z)
    resid_max = float(np.max(np.abs(model.residuals(x, x, zero, params, ctx_now))))
    ok = resid_max < tol
    if not ok and raise_on_fail:
        raise RuntimeError(
            f"steady_state: residual {resid_max:.3e} above tolerance {tol:.1e}. "
            f"Solver message: {getattr(sol, 'message', '')}")

    return SteadyState(x=x, values=model.dict(x), residual=resid_max, ctx=ctx_now,
                       outer_iterations=outer, success=ok,
                       message=getattr(sol, "message", ""), model=model)


def _ctx_distance(a, b) -> float:
    """Relative movement between two contexts, for the outer loop's stopping rule.

    Handles the shapes a context actually takes -- a mapping of numbers or
    arrays, a bare array, a number. Anything else compares equal-or-not, which
    stops the loop after one round rather than looping forever on an object the
    routine cannot measure.
    """
    if a is None or b is None:
        return 0.0 if a is b else np.inf
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        keys = set(a) | set(b)
        worst = 0.0
        for k in keys:
            if k not in a or k not in b:
                return np.inf
            worst = max(worst, _ctx_distance(a[k], b[k]))
        return worst
    if isinstance(a, (int, float, np.floating, np.ndarray, list, tuple)):
        try:
            av = np.asarray(a, dtype=float).ravel()
            bv = np.asarray(b, dtype=float).ravel()
        except (TypeError, ValueError):
            # A context routinely carries non-numeric entries beside the numeric
            # ones -- variable-name lists, mode flags, a policy rule. Those cannot
            # move by a distance, so they compare equal-or-not rather than
            # aborting the loop that the numeric entries need.
            return _equal_or_infinite(a, b)
        if av.shape != bv.shape:
            return np.inf
        if av.size == 0:
            return 0.0
        denom = np.maximum(np.abs(av), 1.0)
        return float(np.max(np.abs(av - bv) / denom))
    return _equal_or_infinite(a, b)


def _equal_or_infinite(a, b) -> float:
    """Zero when two non-numeric context entries are equal, infinite otherwise."""
    try:
        same = bool(a == b)
    except Exception:
        same = a is b
    return 0.0 if same else np.inf


class SteadyStateError(RuntimeError):
    """A declared (analytic) steady state that does not solve the model."""


def analytic_steady_state(model: Model, params, values, *, ctx=None,
                          tol: float = 1e-10) -> SteadyState:
    """A steady state given in closed form, CHECKED against the model's own equations.

    ``values`` is a mapping of every variable to its steady-state level, or a
    callable ``values(params) -> mapping``. A closed form is a second model of
    the same point -- it is written separately from the equations and can drift
    from them -- so it is used only if it zeroes the dynamic model's residuals
    at ``(y, y, 0)`` to ``tol``; otherwise this raises, naming the equations it
    violates. Nothing is solved: a model whose closed form is right never sees a
    Newton step.
    """
    vals = values(params) if callable(values) else values
    vals = dict(vals)
    aux = getattr(model, "auxiliary", None) or {}
    for name, (kind, of) in aux.items():
        if name not in vals and of in vals:
            vals[name] = vals[of]
    missing = [v for v in model.names if v not in vals]
    if missing:
        raise SteadyStateError(f"the closed form leaves {missing} without a value")
    x = model.vector(vals)
    zero = np.zeros(model.n_shocks)
    r = np.asarray(model.residuals(x, x, zero, params, ctx), float)
    worst = float(np.max(np.abs(r))) if r.size else 0.0
    if not np.isfinite(worst) or worst > tol:
        labels = (getattr(model, "symbolic", None) or {}).get("equations", [])
        order = np.argsort(-np.abs(r))[:3]
        where = "; ".join(f"equation {i + 1} ({labels[i] if i < len(labels) else ''}) "
                          f"residual {r[i]:.3e}" for i in order if abs(r[i]) > tol)
        raise SteadyStateError(f"the closed-form steady state does not solve the model: {where}")
    return SteadyState(x=x, values=model.dict(x), residual=worst, ctx=ctx, success=True,
                       message="closed form, checked", model=model)


def _calibrated(model: Model, guess, params, targets: dict, *, ctx, positive, tol,
                raise_on_fail) -> SteadyState:
    """The steady state and the parameters in ``targets``, as one system."""
    from scipy.optimize import root
    from .model import get_param, set_params
    from .symbolic import calibration_residual

    if guess is None:
        program = getattr(model, "steady_state_model", None)
        if program is None:
            raise ValueError("calibration needs a guess for the steady state")
        guess = program(params)
    x0 = model.vector(guess) if isinstance(guess, Mapping) else np.asarray(guess, float)
    names = list(targets)
    theta0 = np.array([float(get_param(params, p)) for p in names])
    g = calibration_residual(model, targets, params)
    log_idx = np.array([model.position(n) for n in positive], dtype=int)
    zero = np.zeros(model.n_shocks)
    n = model.n_vars

    def unpack(z):
        x = z[:n].copy()
        if log_idx.size:
            x[log_idx] = np.exp(z[log_idx])
        return x, set_params(params, dict(zip(names, map(float, z[n:]))))

    def resid(z):
        x, p = unpack(z)
        return np.r_[model.residuals(x, x, zero, p, ctx), g(x, p)]

    z0 = np.r_[x0, theta0]
    if log_idx.size:
        z0[log_idx] = np.log(x0[log_idx])
    sol = root(resid, z0, method="hybr", tol=1e-15)
    x, p = unpack(sol.x)
    r = resid(sol.x)
    worst = float(np.max(np.abs(r)))
    if worst > tol and raise_on_fail:
        raise RuntimeError(f"steady_state with calibration: residual {worst:.3e} above "
                           f"{tol:.1e} ({sol.message})")
    ss = SteadyState(x=x, values=model.dict(x), residual=worst, ctx=ctx, success=worst <= tol,
                     message=str(sol.message), model=model)
    ss.calibrated = {k: float(get_param(p, k)) for k in names}
    ss.params = p
    return ss


def _continuation(model, guess, params, homotopy, steps, kw) -> SteadyState:
    """Follow the steady state as parameters move from ``start`` to ``end``."""
    from .model import set_params

    def at(share):
        return set_params(params, {k: a + share * (b - a) for k, (a, b) in homotopy.items()})

    ss = steady_state(model, guess, at(0.0), **kw)
    share, step = 0.0, 1.0 / max(1, steps)
    path = [0.0]
    while share < 1.0:
        nxt = min(1.0, share + step)
        try:
            trial = steady_state(model, ss.values, at(nxt), **kw)
        except Exception:
            step /= 2
            if step < 1e-8:
                raise RuntimeError(f"homotopy: stuck at share {share:.6f} of the way; the "
                                   f"steady state may not exist beyond it")
            continue
        ss, share = trial, nxt
        path.append(share)
        step = min(2 * step, 1.0)
    ss.homotopy_path = path
    ss.params = at(1.0)
    return ss
