"""Estimation by impulse-response matching, including external IRF estimates.

A target is a tuple ``(variable, shock, period, value)``, with Dynare's impact
period numbered 1. An optional fifth entry is its diagonal weight. Targets may
come from local projections, event studies or another external estimator. Their
joint covariance can be supplied as ``target_cov``; unless an explicit weighting
matrix is also given, its inverse is the efficient weighting matrix.

The first-order model IRFs use columns of the lower Cholesky factor of the shock
covariance, as Dynare's orthogonalised IRFs do. The objective is the quadratic
distance ``Q = (data - model)' W (data - model)``. Standard errors use the
inverse Hessian of ``Q/2``, the Gaussian quasi-likelihood kernel used by Dynare's
frequentist IRF matching; there is no Hansen J test or time-series length factor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike
from scipy import linalg, optimize

from .estimation import hessian, unpack_point
from .model import Covariance, Model, Params, Point

__all__ = ["IRFMatchingResult", "irf_matching"]


@dataclass
class IRFMatchingResult:
    """An estimate by IRF matching. See :func:`irf_matching`.

    ``targets`` and both IRF vectors follow MATLAB's column-major order: period,
    then observable, then shock. ``objective`` is Q, not the Dynare objective
    ``Q/2`` plus its constant. ``hessian`` is the Hessian of Q/2 and ``cov`` its
    scaled inverse. ``target_cov`` is the external covariance, when supplied.
    """

    names: list[str]
    x: np.ndarray
    stderr: dict[str, float]
    cov: np.ndarray
    objective: float
    targets: list[tuple[str, str, int, float]]
    data_irfs: np.ndarray
    model_irfs: np.ndarray
    weighting: np.ndarray
    target_cov: np.ndarray | None
    hessian: np.ndarray
    success: bool = True
    message: str = ""

    @property
    def values(self) -> dict[str, float]:
        return dict(zip(self.names, map(float, self.x)))

    @property
    def irf_labels(self) -> list[str]:
        return [f"{v}[{e}]({h})" for v, e, h, _ in self.targets]

    def summary(self) -> str:
        out = [f"IRF matching, {len(self.targets)} responses, {len(self.names)} parameters",
               f"  {'parameter':16s} {'estimate':>12s} {'s.e.':>12s}"]
        for n, v in zip(self.names, self.x):
            out.append(f"  {n:16s} {v:12.6g} {self.stderr[n]:12.4g}")
        out.append(f"  objective {self.objective:.6g}")
        return "\n".join(out)


def _targets(targets, model: Model, observables: Sequence[str] | None):
    rows = []
    for row in targets:
        if len(row) not in (4, 5):
            raise ValueError("each IRF target must be (variable, shock, period, value[, weight])")
        v, e, h, value = row[:4]
        weight = row[4] if len(row) == 5 else 1.0
        if v not in model.names:
            raise ValueError(f"unknown IRF variable {v!r}")
        if e not in model.shocks:
            raise ValueError(f"unknown IRF shock {e!r}")
        if isinstance(h, (bool, np.bool_)) or int(h) != h or int(h) < 1:
            raise ValueError(f"IRF periods are integers starting at 1, got {h!r}")
        if not np.isfinite(value) or not np.isfinite(weight):
            raise ValueError("IRF values and weights must be finite")
        rows.append((str(v), str(e), int(h), float(value), float(weight)))
    if not rows:
        raise ValueError("no IRF targets")
    keys = [(v, e, h) for v, e, h, _, _ in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("an IRF variable/shock/period target was declared twice")
    obs = list(observables) if observables is not None else list(dict.fromkeys(v for v, *_ in rows))
    missing = sorted({v for v, *_ in rows} - set(obs))
    if missing:
        raise ValueError(f"IRF targets involve {missing}, which are not observables")
    if len(set(obs)) != len(obs) or any(v not in model.names for v in obs):
        raise ValueError("observables must be distinct model variables")
    order = sorted(range(len(rows)), key=lambda i: (model.shocks.index(rows[i][1]),
                                                    obs.index(rows[i][0]), rows[i][2]))
    rows = [rows[i] for i in order]
    canonical = [(v, e, h, value) for v, e, h, value, _ in rows]
    data = np.array([r[3] for r in rows])
    diagonal = np.array([r[4] for r in rows])
    return canonical, data, diagonal, obs


def _matrix(value, n: int, label: str) -> np.ndarray:
    A = np.asarray(value, dtype=float)
    if A.ndim == 1:
        if A.size != n:
            raise ValueError(f"{label} must have {n} entries, got {A.size}")
        A = np.diag(A)
    if A.shape != (n, n):
        raise ValueError(f"{label} must be {n} x {n}, got {A.shape}")
    if not np.all(np.isfinite(A)):
        raise ValueError(f"{label} must be finite")
    A = 0.5 * (A + A.T)
    try:
        linalg.cholesky(A)
    except linalg.LinAlgError as exc:
        raise ValueError(f"{label} must be positive definite") from exc
    return A


def _cholesky(Sigma: np.ndarray, tiny: float) -> np.ndarray:
    live = np.flatnonzero(np.diag(Sigma) != 0)
    L = np.zeros_like(Sigma)
    if not live.size:
        return L
    S = Sigma[np.ix_(live, live)]
    try:
        L[np.ix_(live, live)] = np.linalg.cholesky(S)
    except np.linalg.LinAlgError:
        L[np.ix_(live, live)] = np.linalg.cholesky(S + tiny * np.eye(len(live)))
    return L


def irf_matching(model: Model, targets: Sequence[Sequence], *,
                 params: Params, free: Sequence[str] | Mapping[str, float],
                 observables: Sequence[str] | None = None,
                 bounds: Mapping[str, tuple[float, float]] | None = None,
                 cov: Covariance = None,
                 weighting: ArrayLike | None = None,
                 target_cov: ArrayLike | None = None,
                 relative_irf: bool = False,
                 transform: Callable[[np.ndarray, Any], np.ndarray] | None = None,
                 guess: Point | None = None,
                 steady_fn: Callable[[Params], Any] | None = None,
                 qz_criterium: float | None = None,
                 cholesky_tiny: float = 1e-14,
                 se_step: ArrayLike | None = None,
                 cova_compute: bool = True,
                 optimize_: bool = True,
                 options: Mapping[str, Any] | None = None) -> IRFMatchingResult:
    """Estimate ``free`` by matching the first-order model's impulse responses.

    Parameters
    ----------
    targets : sequence of ``(variable, shock, period, value[, weight])``.
        Period 1 is the impact period. Duplicate targets are refused. The fifth
        entry sets a diagonal element of W; it is a weight, not a variance.
    observables : optional variable order used to sort the targets. By default,
        order of first appearance. Supply the ``varobs`` order for Dynare parity.
    weighting : optional full matrix, diagonal vector, or None. By default the
        individual target weights form a diagonal W.
    target_cov : joint covariance of the external targets, full or diagonal.
        With no explicit ``weighting``, W is its inverse. It is retained in the
        result and is not itself used as W.
    relative_irf : if true, report ``100 * IRF / L[j,j]`` for shock j, Dynare's
        ``relative_irf`` at first order. Otherwise use the Cholesky impulse as is.
    transform : optional ``transform(irfs, solution)`` applied to the full
        ``(horizon, n_variables, n_shocks)`` array before selecting targets.
    se_step : absolute central-difference steps for the Hessian. By default use
        Dynare's ``max(abs(x), sqrt(1e-2)) * eps**(1/6)`` rule.
    optimize_ : False evaluates at the starting point, like ``mode_compute=0``.
    options : passed to :func:`scipy.optimize.least_squares`.
    """
    from .moment_estimation import _start
    from .solve import solve
    from .steady import steady_state

    canonical, data, diagonal, obs = _targets(targets, model, observables)
    n = len(canonical)
    names = list(free)
    theta0 = _start(model, params, cov, names, free)
    lo = np.array([(bounds or {}).get(k, (-np.inf, np.inf))[0] for k in names], float)
    hi = np.array([(bounds or {}).get(k, (-np.inf, np.inf))[1] for k in names], float)
    Omega = None if target_cov is None else _matrix(target_cov, n, "target_cov")
    if weighting is not None:
        W = _matrix(weighting, n, "weighting")
    elif Omega is not None:
        W = np.linalg.inv(Omega)
        W = 0.5 * (W + W.T)
    else:
        W = _matrix(diagonal, n, "weighting")
    Sw = linalg.cholesky(W)
    horizon = max(h for _, _, h, _ in canonical)
    last_ss = {"values": guess}

    def model_irfs(theta) -> np.ndarray:
        p, Sigma, _ = unpack_point(model, params, cov, {}, names, theta)
        ss = steady_fn(p) if steady_fn is not None else steady_state(model, last_ss["values"], p)
        sol = solve(model, p, steady=ss, qz_criterium=qz_criterium)
        last_ss["values"] = ss.values
        L = _cholesky(Sigma, cholesky_tiny)
        full = np.empty((horizon, len(model.names), len(model.shocks)))
        for j, shock in enumerate(model.shocks):
            paths = sol.irf_impulse(L[:, j], periods=horizon, levels=True)
            for i, v in enumerate(model.names):
                # Dynare's IRFs are absolute deviations from the steady state,
                # not the solution object's rescaled/proportional deviations.
                full[:, i, j] = paths[v] - ss.values[v]
            if relative_irf:
                if L[j, j] == 0:
                    raise ValueError(f"relative_irf needs a nonzero Cholesky impact for {shock!r}")
                full[:, :, j] *= 100.0 / L[j, j]
        if transform is not None:
            full = np.asarray(transform(full, sol), dtype=float)
            if full.shape != (horizon, len(model.names), len(model.shocks)):
                raise ValueError("transform must preserve the IRF array's shape")
        pos = {v: model.position(v) for v in obs}
        return np.array([full[h - 1, pos[v], model.shocks.index(e)] for v, e, h, _ in canonical])

    def residuals(theta) -> np.ndarray:
        try:
            r = Sw @ (data - model_irfs(theta))
            return r if np.all(np.isfinite(r)) else np.full(n, 1e7)
        except Exception:
            return np.full(n, 1e7)

    x = np.clip(theta0, lo, hi)
    success, message = True, ""
    if optimize_:
        opts = dict(method="trf", x_scale="jac", ftol=1e-12, xtol=1e-12, gtol=1e-12,
                    max_nfev=5000)
        opts.update(options or {})
        fit = optimize.least_squares(residuals, x, bounds=(lo, hi), **opts)
        x, success, message = fit.x, bool(fit.success), str(fit.message)
    fitted = model_irfs(x)
    gap = data - fitted
    objective = float(gap @ W @ gap)

    H = np.full((len(names), len(names)), np.nan)
    V = np.full_like(H, np.nan)
    if cova_compute and names:
        def distance(theta):
            try:
                r = data - model_irfs(theta)
                return 0.5 * float(r @ W @ r)
            except Exception:
                return np.inf

        steps = se_step
        if steps is None:
            steps = np.maximum(np.abs(x), math.sqrt(1e-2)) * np.finfo(float).eps ** (1 / 6)
        H = hessian(distance, x, steps=steps)
        H = 0.5 * (H + H.T)
        if np.all(np.isfinite(H)) and np.all(np.diag(H) > 0):
            try:
                scale = np.sqrt(np.diag(H))
                V = np.linalg.inv(H / np.outer(scale, scale)) / np.outer(scale, scale)
            except np.linalg.LinAlgError:
                pass
        if not np.all(np.isfinite(V)):
            message = (f"{message}; " if message else "") + "no standard errors: singular Hessian"
    se = np.sqrt(np.where(np.diag(V) > 0, np.diag(V), np.nan))
    return IRFMatchingResult(names=names, x=x, stderr=dict(zip(names, map(float, se))),
                             cov=V, objective=objective, targets=canonical, data_irfs=data,
                             model_irfs=fitted, weighting=W, target_cov=Omega, hessian=H,
                             success=success, message=message)
