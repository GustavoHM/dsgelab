"""Heutel (2011 replication): first-order environmental RBC policy models.

The original hand-linearised systems are retained; this is not a nonlinear
model suitable for second-order welfare or nonlinear perfect foresight.
See ``docs/heutel.md`` for source calibration and timing details.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares, root

from ..model import Model
from ..solve import solve as solve_model
from . import _heutel_algebra as algebra

VARIANTS = ("planner", "tax", "tax_asym", "quota_asym")
BASE_NAMES = ("c", "e", "k", "x", "mu", "y", "z", "i")


@dataclass(frozen=True)
class Params:
    """Tax-script baseline. Use ``source('planner')`` for its distinct defaults."""
    beta: float = 0.98267
    delta: float = 0.025
    rho: float = 0.95
    sd: float = 0.007
    alpha: float = 0.36
    eta: float = 0.9979
    theta1: float = 0.05607
    theta2: float = 2.8
    gamma: float = 1 - 0.696
    phic: float = 2.0
    d2: float = 1.4647e-8 / 5.3024**2
    d1: float = -6.6722e-6 / 5.3024
    d0: float = 1.395e-3
    erowcoef: float = 4.0
    policy_slope: float = 1.4151

    @classmethod
    def source(cls, variant: str) -> Params:
        """Return the actual calibration in the corresponding attached script."""
        _check_variant(variant)
        if variant == "planner":
            return cls(theta2=0.1, gamma=0.8)
        return cls(policy_slope=0.6580 if variant == "quota_asym" else 1.4151)


class SteadyStateError(RuntimeError):
    """No admissible steady state was found; no linear solution is returned."""


def _check_variant(variant):
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")


def _planner_levels(k, e, mu, p):
    x = p.erowcoef * e / (1 - p.eta)
    y = (1 - p.d2*x*x - p.d1*x - p.d0) * k**p.alpha
    z = p.theta1 * mu**p.theta2 * y
    i = p.delta * k
    return dict(k=k, e=e, mu=mu, x=x, y=y, z=z, i=i, c=y-i-z, a=1.0)


def steady_levels(variant="tax", params=None):
    """Solve the source steady-state equations, checking residuals and feasibility.

    The planner source defaults (theta2=.1, gamma=.8) fail this check. To
    compare regimes under the tax scripts' calibration, pass ``Params()``
    explicitly; source defaults are never silently replaced.
    """
    _check_variant(variant)
    p = params or Params.source(variant)
    values = np.array(list(vars(p).values()), dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("all parameters must be finite")
    if not (0 < p.beta < 1 and 0 < p.eta < 1 and 0 <= p.delta < 1
            and 0 < p.alpha < 1 and abs(p.rho) < 1 and p.theta1 > 0
            and p.theta2 > 0 and p.phic > 0 and p.erowcoef > 0 and p.sd >= 0):
        raise ValueError("parameters outside the admissible calibration domain")
    if variant != "planner" and (p.theta2 == 1 or p.erowcoef != 4):
        raise ValueError("source tax equations require theta2 != 1 and erowcoef = 4")
    fn = algebra.planner_ss if variant == "planner" else algebra.tax_ss
    guess = [28.0245, 2.2270, 0.0339] if variant == "planner" else [27.8, 2.29]
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        result = root(lambda x: fn(x, p), guess, tol=1e-11)
    candidate = result.x
    if variant == "planner" and (not np.isfinite(result.fun).all()
            or np.max(np.abs(result.fun)) > 1e-9 or not 0 < candidate[2] < 1
            or np.any(candidate[:2] <= 0)):
        result = least_squares(lambda x: fn(x, p), guess,
            bounds=([1e-3, 1e-6, 1e-9], [1e3, 100, 1-1e-9]),
            xtol=1e-13, ftol=1e-13, gtol=1e-13, max_nfev=5000)
        candidate = result.x
    residual = np.max(np.abs(fn(candidate, p)))
    if not np.isfinite(residual) or residual > 1e-9 or np.any(candidate[:2] <= 0):
        detail = (" The attached planner uses theta2=0.1 and gamma=0.8; "
                  "pass Params() explicitly to use the tax-script baseline."
                  if variant == "planner" else "")
        raise SteadyStateError(f"{variant}: no admissible converged steady state "
                              f"(max residual {residual:.3g})." + detail)
    if variant == "planner":
        ss = _planner_levels(*candidate, p)
    else:
        ss = algebra.tax_levels(*candidate, p)
        ss["a"] = 1.0
    if not all(np.isfinite(v) for v in ss.values()) or not 0 < ss["mu"] < 1 \
            or any(ss[k] <= 0 for k in BASE_NAMES):
        raise SteadyStateError("steady state violates positivity or 0 < mu < 1")
    if variant == "quota_asym":
        ss.pop("tau")
        ss["q"] = ss["e"]
    return ss


def structural_system(variant="tax", params=None):
    """Return ``(names, H, levels)`` in source order: H=[lag,current,lead]."""
    _check_variant(variant)
    p = params or Params.source(variant)
    ss = steady_levels(variant, p)
    if variant == "planner":
        names = BASE_NAMES + ("a",)
        h = algebra.planner_matrix(ss, p)
    else:
        names = BASE_NAMES + ("tau", "r", "lambda", "zeta", "omega", "a")
        # Quota and tax share steady allocations; the pre-policy source matrix
        # still needs the implicit tax price before the quota rows replace it.
        tax_ss = dict(ss)
        if variant == "quota_asym":
            tax_ss["tau"] = p.theta1*p.theta2*ss["mu"]**(p.theta2-1)*ss["y"]**p.gamma
        h = algebra.tax_matrix(tax_ss, p)
        if variant.endswith("asym"):
            h[10:14] = 0
            h[10, 22] = 1
            h[10, 5] = -p.policy_slope
            h[11, 24] = h[12, 25] = h[13, 26] = 1
        if variant == "quota_asym":
            h = algebra.quota_firm_rows(h, ss, p)
            names = BASE_NAMES + ("q", "r", "lambda", "zeta", "omega", "a")
    return names, h, ss


def build(variant="tax", params=None):
    """Return a dsgekit ``(model, params, guess)`` in proportional deviations.

    ``k`` and ``x`` retain the source's end-of-period/contemporaneous timing.
    Auxiliary ``lag_*`` states carry values needed from the previous period;
    ``a`` is current TFP. The model's steady point is zero deviations, while
    actual economic steady levels are in :func:`steady_levels`.
    """
    p = params or Params.source(variant)
    names, h, ss = structural_system(variant, p)
    n = len(names)
    if variant == "tax":
        # The Ramsey multiplier is zero at the optimum. Proportional
        # deviations of it are undefined and make the original pencil nearly
        # singular. Evaluate its linear coefficients per UNIT of multiplier,
        # which is the analytic zero-multiplier limit (no division by zero).
        unit_ss = dict(ss, **{"lambda": 1.0})
        unit_h = algebra.tax_matrix(unit_ss, p)
        columns = [names.index("lambda") + t*n for t in range(3)]
        h[:, columns] = unit_h[:, columns]
    lagged = [name for j, name in enumerate(names)
              if name != "a" and np.any(h[:, j] != 0)]
    states = ["lag_" + name for name in lagged] + ["a"]
    jumps = [name for name in names if name != "a"]
    # All matrix coefficients are fixed at p; reject a later parameter change
    # rather than let calibration silently use stale coefficients.
    def equations(nxt, cur, eps, parameters):
        if parameters != p:
            raise ValueError("rebuild Heutel's structural system after changing parameters")
        lag = np.array([cur["lag_"+name] if name in lagged else 0.0 for name in names])
        now = np.array([cur[name] for name in names])
        future = np.array([nxt[name] for name in names])
        residual = h[:, :n] @ lag + h[:, n:2*n] @ now + h[:, 2*n:] @ future
        # The original first row is a_t = rho*a_(t-1) + eps_t. Shift just
        # this transition to dsgekit's innovation-at-t+1 convention.
        residual[0] = nxt.a - p.rho*cur.a - eps.eps_a
        return np.r_[residual, [nxt["lag_"+name] - cur[name] for name in lagged]]
    model = Model(states=states, jumps=jumps, shocks=["eps_a"],
                  equations=equations, name=f"heutel-{variant}-linear")
    return model, p, {name: 0.0 for name in states+jumps}


def solve(variant="tax", params=None):
    """Solve the source first-order system with dsgekit's Klein/QZ solver.

    ``sol.irf('eps_a', size=.01)`` gives percent responses. Since variables
    already denote deviations, use :func:`to_levels` for economic levels.
    Do not interpret higher-order solutions of this linear model as a
    nonlinear solution of Heutel's economy.
    """
    model, p, guess = build(variant, params)
    return solve_model(model, p, guess=guess)


def to_levels(paths, variant="tax", params=None, *, percent=True):
    """Convert source variables in a deviation path to economic levels."""
    ss = steady_levels(variant, params)
    factor = 100 if percent else 1
    return {name: (ss[name] + np.asarray(path)/factor if name == "lambda"
                   else ss[name]*(1+np.asarray(path)/factor))
            for name, path in paths.items() if name in ss}
