"""Pin free parameters on steady-state (or model-implied) targets.

Applied calibration is almost never a table of numbers read off a source: a
handful of parameters have no direct counterpart in data and are set so that the
model reproduces moments that do -- a penetration rate, an elasticity, a share of
output. Done by hand this becomes a nest of bespoke bisections, one per
parameter, each hiding the fact that the targets are JOINTLY determined.

This module solves the whole system at once and, just as usefully, reports the
Jacobian of targets with respect to parameters. A calibration whose Jacobian is
near-singular is not identified: some direction in parameter space leaves every
target where it was, and the numbers that come out of it are whichever point the
solver happened to stop at.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares

from .model import Model, set_params
from .steady import SteadyState, steady_state

__all__ = ["CalibrationResult", "calibrate", "target_jacobian"]


@dataclass
class CalibrationResult:
    """Calibrated parameters, the steady state they imply and the fit achieved."""

    params: object
    values: dict
    steady: SteadyState
    targets: dict
    achieved: dict
    max_relative_error: float
    success: bool
    iterations: int
    message: str = ""
    jacobian: np.ndarray | None = field(default=None, repr=False)
    names: list = field(default_factory=list)

    def report(self) -> str:
        w = max((len(k) for k in self.targets), default=8)
        lines = ["calibration:"]
        for k, v in self.values.items():
            lines.append(f"  {k:<{w}} = {v:.6g}")
        lines.append("targets (achieved vs wanted):")
        for k, want in self.targets.items():
            got = self.achieved[k]
            denom = abs(want) if abs(want) > 1e-12 else 1.0
            lines.append(f"  {k:<{w}} {got:.6g}  vs {want:.6g}   "
                         f"({100 * (got - want) / denom:+.3g}%)")
        return "\n".join(lines)


def _observed(model, ss, params, observables):
    if observables is None:
        return {n: ss.values[n] for n in ss.values}
    return dict(observables(ss, params))


def calibrate(model: Model, params, *, free: Mapping[str, float],
              targets: Mapping[str, float],
              guess, observables: Callable | None = None,
              bounds: Mapping[str, tuple] | None = None,
              ctx=None, ctx_update=None, positive: Sequence[str] = (),
              steady_tol: float = 1e-11, tol: float = 1e-10,
              maxiter: int = 200, warm_start: bool = True,
              raise_on_fail: bool = True) -> CalibrationResult:
    """Solve for the ``free`` parameters that make the model hit ``targets``.

    Parameters
    ----------
    free : mapping
        Parameter name -> starting value.
    targets : mapping
        Target name -> wanted value. Names are steady-state variables unless
        ``observables`` supplies them.
    guess : mapping or array
        Starting point for the steady state.
    observables : callable, optional
        ``observables(steady_state, params) -> mapping``, for targets that are
        functions of the solution rather than variables of it (a ratio, an
        elasticity, a share).
    bounds : mapping, optional
        Parameter name -> ``(low, high)``. Bounds are the honest way to keep a
        depreciation rate inside the unit interval; a clip inside the objective
        would flatten the residual and stall the solver on the boundary.
    warm_start : bool
        Reuse the previous steady state as the next guess. Worth an order of
        magnitude in a sweep, and harmless: the accepted solution is still judged
        on its own residual.
    """
    names = list(free)
    x0 = np.array([float(free[k]) for k in names])
    tnames = list(targets)
    want = np.array([float(targets[k]) for k in tnames])
    denom = np.where(np.abs(want) > 1e-12, np.abs(want), 1.0)

    lo = np.full(len(names), -np.inf)
    hi = np.full(len(names), np.inf)
    if bounds:
        for i, k in enumerate(names):
            if k in bounds:
                lo[i], hi[i] = bounds[k]
        x0 = np.clip(x0, lo + 1e-12, hi - 1e-12)

    state = {"guess": guess, "ss": None, "params": None, "achieved": None, "calls": 0}

    def evaluate(x):
        p = set_params(params, {k: float(v) for k, v in zip(names, x)})
        g = state["guess"] if warm_start and state["guess"] is not None else guess
        ss = steady_state(model, g, p, ctx=ctx, ctx_update=ctx_update,
                          positive=positive, tol=steady_tol)
        obs = _observed(model, ss, p, observables)
        missing = [t for t in tnames if t not in obs]
        if missing:
            raise KeyError(
                f"targets {missing} are neither model variables nor produced by "
                f"`observables`; available: {sorted(obs)[:20]}")
        got = np.array([float(obs[t]) for t in tnames])
        state.update(guess=ss.values, ss=ss, params=p, achieved=dict(zip(tnames, got)))
        state["calls"] += 1
        return (got - want) / denom

    res = least_squares(evaluate, x0, bounds=(lo, hi), xtol=1e-14, ftol=1e-14,
                        gtol=1e-14, max_nfev=maxiter)
    # Re-evaluate at the accepted point: least_squares may report the last TRIAL
    # step rather than the best one, and the cached state must match `res.x`.
    err = evaluate(res.x)
    worst = float(np.max(np.abs(err)))
    ok = worst < tol
    if not ok and raise_on_fail:
        detail = ", ".join(f"{t}: {state['achieved'][t]:.6g} vs {targets[t]:.6g}"
                           for t in tnames)
        raise RuntimeError(
            f"calibrate: largest relative target error {worst:.3e} above tolerance "
            f"{tol:.1e} ({detail}). If the residual will not fall, check identification "
            f"with target_jacobian().")

    return CalibrationResult(
        params=state["params"], values={k: float(v) for k, v in zip(names, res.x)},
        steady=state["ss"], targets=dict(targets), achieved=state["achieved"],
        max_relative_error=worst, success=ok, iterations=state["calls"],
        message=res.message, names=names)


def target_jacobian(model: Model, params, *, free: Sequence[str],
                    targets: Sequence[str], guess,
                    observables: Callable | None = None, step: float = 1e-5,
                    ctx=None, ctx_update=None, positive: Sequence[str] = (),
                    steady_tol: float = 1e-11) -> dict:
    """Elasticity of each target with respect to each free parameter.

    Reported both as a level derivative and as an elasticity, together with the
    condition number and the smallest singular value of the elasticity matrix.
    A small singular value is the diagnosis for a calibration that "will not
    converge": the targets do not separate the parameters, and no solver
    tolerance will fix that.
    """
    free = list(free)
    targets = list(targets)

    def obs_at(p, g):
        ss = steady_state(model, g, p, ctx=ctx, ctx_update=ctx_update,
                          positive=positive, tol=steady_tol)
        o = _observed(model, ss, p, observables)
        return np.array([float(o[t]) for t in targets]), ss.values

    base_vals = {k: float(getattr(params, k) if not isinstance(params, Mapping) else params[k])
                 for k in free}
    y0, g0 = obs_at(params, guess)

    J = np.zeros((len(targets), len(free)))
    for j, k in enumerate(free):
        v = base_vals[k]
        dv = step * max(abs(v), 1.0)
        yp, _ = obs_at(set_params(params, {k: v + dv}), g0)
        ym, _ = obs_at(set_params(params, {k: v - dv}), g0)
        J[:, j] = (yp - ym) / (2 * dv)

    scale_y = np.where(np.abs(y0) > 1e-12, np.abs(y0), 1.0)
    scale_x = np.array([max(abs(base_vals[k]), 1e-12) for k in free])
    E = J * scale_x[None, :] / scale_y[:, None]
    U, sv, Vt = np.linalg.svd(E)
    rank_tol = max(E.shape) * (sv[0] if sv.size else 0.0) * np.finfo(float).eps
    rank = int(np.sum(sv > max(rank_tol, 1e-10)))
    # Identification is a statement about the PARAMETERS, so the count that
    # matters is how many directions in parameter space the targets pin down.
    # Reading it off the condition number alone misses the common case of fewer
    # targets than parameters, where the matrix is not square and its own
    # singular values say nothing about the free directions.
    n_free = len(free)
    null_dirs = Vt[rank:] if rank < n_free else np.zeros((0, n_free))
    return dict(targets=targets, free=free, derivative=J, elasticity=E,
                singular_values=sv, rank=rank, n_free=n_free,
                identified=bool(rank == n_free),
                null_directions=null_dirs,
                condition=float(sv[0] / sv[rank - 1]) if rank > 0 and sv[rank - 1] > 0
                else np.inf,
                base_targets=dict(zip(targets, y0)), base_params=base_vals)
