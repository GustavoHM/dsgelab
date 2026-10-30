"""Calibration restrictions on impulse responses and moments.

These are Dynare's ``irf_calibration`` and ``moment_calibration`` blocks: a sign
or an interval that an impulse response, at given horizons, or a theoretical
moment, at given lags, should respect. :func:`check_restrictions` evaluates them
on one first-order solution, and :func:`sample_restrictions` over draws from the
prior, reporting which restrictions hold and in what share of the draws, as the
``map_calibration`` step of Dynare's sensitivity toolbox does.

An IRF restriction is ``(variable, shock, horizons, bound)``, with the impact
period numbered 1. A moment restriction is ``(variable1, variable2, lags,
bound)``: at lag 0 it is the variance when the two variables are the same and
their correlation otherwise, and at lag ``l`` it is the correlation of
``variable1`` at ``t`` with ``variable2`` at ``t + l``, so ``("y", "y", 1, "+")``
asks for a positive first autocorrelation. Horizons and lags are an integer, a
sequence or a ``"first:last"`` string. A bound is ``"+"``, ``"-"`` or ``(lower,
upper)``, and the inequalities are strict.

The conventions are those of Dynare 7.2's ``endogenous_prior_restrictions``:

* Values are absolute deviations from the steady state, as Dynare's are.
* The impulse is one standard deviation of the shock, the square root of the
  diagonal of its covariance with no Cholesky factor. With ``relative_irf`` it is
  100 times a unit innovation.
* A restriction over several horizons or lags holds if any one of them is
  inside the bound (Dynare's logical OR). The restrictions hold together when
  each one holds (AND). Asking every horizon to be inside takes one restriction
  per horizon.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike

from .model import Covariance, Model, Params, Point

__all__ = ["Restriction", "RestrictionCheck", "RestrictionSample",
           "check_restrictions", "sample_restrictions"]

_RANGE = re.compile(r"^\s*([+-]?\d+)\s*(?::\s*([+-]?\d+))?\s*$")


@dataclass(frozen=True)
class Restriction:
    """One restriction. ``kind`` is ``"irf"`` or ``"moment"``.

    For an IRF ``second`` is the shock and ``periods`` the horizons, from 1; for
    a moment ``second`` is the other variable and ``periods`` the leads of it.
    ``periods`` is sorted, as Dynare reports the values.
    """

    kind: str
    first: str
    second: str
    periods: tuple[int, ...]
    lower: float
    upper: float

    @property
    def label(self) -> str:
        p = self.periods
        if len(p) == 1:
            span = str(p[0])
        elif list(p) == list(range(p[0], p[-1] + 1)):
            span = f"{p[0]}:{p[-1]}"
        else:
            span = "[" + ",".join(map(str, p)) + "]"
        if self.kind == "irf":
            return f"{self.first}({span}), {self.second}"
        return f"{self.first}, {self.second}" + ("" if p == (0,) else f"({span})")

    @property
    def bound(self) -> str:
        if self.lower == 0 and self.upper == math.inf:
            return "+"
        if self.lower == -math.inf and self.upper == 0:
            return "-"
        return f"[{self.lower:g}, {self.upper:g}]"

    def inside(self, values: ArrayLike) -> np.ndarray:
        """Which values lie strictly inside the bound (``nan`` never does)."""
        v = np.asarray(values, dtype=float)
        with np.errstate(invalid="ignore"):
            return (v > self.lower) & (v < self.upper)


def _periods(value, kind: str) -> tuple[int, ...]:
    if isinstance(value, str):
        m = _RANGE.match(value)
        if not m:
            raise ValueError(f"cannot read the periods {value!r}: write 3 or '1:4'")
        a = int(m.group(1))
        b = int(m.group(2)) if m.group(2) is not None else a
        out = list(range(a, b + 1))
    elif isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        out = [int(value)]
    else:
        try:
            out = list(value)
        except TypeError:
            raise ValueError(f"periods must be an integer, a sequence or 'a:b', got {value!r}") from None
        if any(isinstance(x, (bool, np.bool_)) or int(x) != x for x in out):
            raise ValueError(f"periods must be integers, got {value!r}")
        out = [int(x) for x in out]
    if not out:
        raise ValueError("a restriction needs at least one period")
    if kind == "irf" and min(out) < 1:
        raise ValueError(f"IRF horizons start at 1 (the impact period), got {value!r}")
    return tuple(sorted(set(out)))


def _bound(value) -> tuple[float, float]:
    if isinstance(value, str):
        if value.strip() == "+":
            return 0.0, math.inf
        if value.strip() == "-":
            return -math.inf, 0.0
        raise ValueError(f"a bound is '+', '-' or (lower, upper), got {value!r}")
    try:
        lo, hi = (float(x) for x in value)
    except (TypeError, ValueError):
        raise ValueError(f"a bound is '+', '-' or (lower, upper), got {value!r}") from None
    if math.isnan(lo) or math.isnan(hi) or not lo < hi:
        raise ValueError(f"the bound needs lower < upper, got {value!r}")
    return lo, hi


def _restrictions(model: Model, irf, moments) -> list[Restriction]:
    out = []
    for row in irf:
        if isinstance(row, Restriction):
            out.append(row)
            continue
        if len(row) != 4:
            raise ValueError("an IRF restriction is (variable, shock, horizons, bound)")
        v, e, h, b = row
        if v not in model.names:
            raise ValueError(f"unknown variable {v!r}")
        if e not in model.shocks:
            raise ValueError(f"unknown shock {e!r}")
        out.append(Restriction("irf", str(v), str(e), _periods(h, "irf"), *_bound(b)))
    for row in moments:
        if isinstance(row, Restriction):
            out.append(row)
            continue
        if len(row) != 4:
            raise ValueError("a moment restriction is (variable1, variable2, lags, bound)")
        a, c, lags, b = row
        for v in (a, c):
            if v not in model.names:
                raise ValueError(f"unknown variable {v!r}")
        out.append(Restriction("moment", str(a), str(c), _periods(lags, "moment"), *_bound(b)))
    if not out:
        raise ValueError("no restrictions: give irf= or moments=")
    return out


# ================================================================ evaluation
def _values(sol, Sigma: np.ndarray, restrictions: Sequence[Restriction],
            relative_irf: bool) -> list[np.ndarray]:
    """Each restriction's values at its periods, in Dynare's units."""
    model = sol.model
    scale = sol.coords.scale
    out: list[np.ndarray] = [np.empty(0)] * len(restrictions)

    irfs = [i for i, r in enumerate(restrictions) if r.kind == "irf"]
    if irfs:
        horizon = max(max(restrictions[i].periods) for i in irfs)
        paths = {}
        for e in dict.fromkeys(restrictions[i].second for i in irfs):
            j = model.shock_position(e)
            impulse = np.zeros(model.n_shocks)
            impulse[j] = 100.0 if relative_irf else math.sqrt(max(Sigma[j, j], 0.0))
            paths[e] = sol.irf_impulse(impulse, periods=horizon, percent=False)
        for i in irfs:
            r = restrictions[i]
            path = paths[r.second][r.first] * scale[model.position(r.first)]
            out[i] = path[np.asarray(r.periods) - 1]

    moms = [i for i, r in enumerate(restrictions) if r.kind == "moment"]
    if moms:
        lags = max(abs(p) for i in moms for p in restrictions[i].periods)
        m = sol.moments(cov=Sigma, lags=lags)
        sd = m["std"]
        for i in moms:
            r = restrictions[i]
            a, b = model.position(r.first), model.position(r.second)
            vals = []
            for l in r.periods:
                if l == 0 and a == b:
                    vals.append(scale[a] ** 2 * m["cov"][a, a])
                elif l == 0:
                    vals.append(m["corr"][a, b])
                else:
                    # autocov[j] is E[x_t x_{t-j}']: corr(a_t, b_{t+l}) is its (b, a)
                    # entry for a lead and its (a, b) entry for a lag
                    G = m["autocov"][abs(l)]
                    g = G[b, a] if l > 0 else G[a, b]
                    with np.errstate(divide="ignore", invalid="ignore"):
                        vals.append(g / (sd[a] * sd[b]) if sd[a] * sd[b] > 0 else math.nan)
            out[i] = np.array(vals, dtype=float)
    return out


@dataclass
class RestrictionCheck:
    """The restrictions evaluated at one parameter point. See :func:`check_restrictions`.

    ``values[i]`` holds restriction ``i`` at each of its periods and ``inside[i]``
    says which of them are inside the bound. ``holds_each[i]`` is restriction
    ``i``'s verdict (any period inside) and ``holds`` is all of them together.
    """

    restrictions: list[Restriction]
    values: list[np.ndarray]

    @property
    def inside(self) -> list[np.ndarray]:
        return [r.inside(v) for r, v in zip(self.restrictions, self.values)]

    @property
    def holds_each(self) -> np.ndarray:
        return np.array([bool(np.any(x)) for x in self.inside])

    @property
    def holds(self) -> bool:
        return bool(np.all(self.holds_each))

    @property
    def failed(self) -> list[str]:
        """Labels of the restrictions that do not hold."""
        return [r.label for r, ok in zip(self.restrictions, self.holds_each) if not ok]

    def summary(self) -> str:
        out = [f"Calibration restrictions: {int(self.holds_each.sum())} of "
               f"{len(self.restrictions)} hold"]
        for r, v, ok in zip(self.restrictions, self.values, self.holds_each):
            shown = ", ".join(f"{x:.4g}" for x in v[:6]) + (", ..." if v.size > 6 else "")
            out.append(f"  {'ok  ' if ok else 'FAIL'} {r.kind:6s} {r.label:22s} "
                       f"{r.bound:>15s}   {shown}")
        return "\n".join(out)


def check_restrictions(solution, *, irf: Sequence[Sequence] = (),
                       moments: Sequence[Sequence] = (), cov: Covariance = None,
                       relative_irf: bool = False) -> RestrictionCheck:
    """Evaluate IRF and moment restrictions on a first-order solution.

    Parameters
    ----------
    solution : FirstOrderSolution
    irf : sequence of ``(variable, shock, horizons, bound)``, Dynare's
        ``irf_calibration``. Horizon 1 is the impact period.
    moments : sequence of ``(variable1, variable2, lags, bound)``, Dynare's
        ``moment_calibration``.
    cov : the shock covariance, as for :meth:`FirstOrderSolution.moments`.
    relative_irf : impulses of 100 times a unit innovation instead of one
        standard deviation, Dynare's ``irf_calibration(relative_irf)``.
    """
    from .solution import shock_covariance

    restrictions = _restrictions(solution.model, irf, moments)
    Sigma = shock_covariance(solution.model.shocks, cov)
    return RestrictionCheck(restrictions, _values(solution, Sigma, restrictions, relative_irf))


# ================================================================== sampling
@dataclass
class RestrictionSample:
    """The restrictions evaluated over parameter draws. See :func:`sample_restrictions`.

    ``draws`` has one row per draw and one column per name in ``names``.
    ``status`` is ``"ok"``, ``"blanchard_kahn"``, ``"no_steady_state"`` or
    ``"other"`` per draw, and ``solved`` is its ``"ok"`` mask. ``values[i]`` is
    ``(draws, periods)`` for restriction ``i``, ``nan`` where the draw was not
    solved. Shares are taken over the solved draws, as Dynare takes them.
    """

    restrictions: list[Restriction]
    names: list[str]
    draws: np.ndarray
    status: np.ndarray
    values: list[np.ndarray] = field(repr=False)

    @property
    def solved(self) -> np.ndarray:
        return self.status == "ok"

    @property
    def inside(self) -> list[np.ndarray]:
        return [r.inside(v) for r, v in zip(self.restrictions, self.values)]

    @property
    def holds_each(self) -> np.ndarray:
        """``(draws, restrictions)``: each restriction's verdict per draw."""
        return np.column_stack([x.any(axis=1) for x in self.inside])

    @property
    def holds(self) -> np.ndarray:
        """Per draw, whether every restriction holds."""
        return self.holds_each.all(axis=1)

    def share(self, which: Sequence[int] | None = None) -> float:
        """Share of the solved draws where the restrictions ``which`` (all, by default) hold."""
        ok = self.solved
        if not ok.any():
            return math.nan
        H = self.holds_each[ok]
        cols = range(H.shape[1]) if which is None else list(which)
        return float(np.mean(H[:, cols].all(axis=1)))

    @property
    def shares(self) -> np.ndarray:
        """Share of the solved draws that satisfies each restriction, in their order."""
        return np.array([self.share([i]) for i in range(len(self.restrictions))])

    def summary(self) -> str:
        n, ok = len(self.status), int(self.solved.sum())
        out = [f"Calibration restrictions over {n} draws, {ok} solved",
               f"  {'share':>7s}  {'kind':6s} {'restriction':22s} {'bound':>15s}"]
        for s, r in zip(self.shares, self.restrictions):
            out.append(f"  {100 * s:6.1f}%  {r.kind:6s} {r.label:22s} {r.bound:>15s}")
        out.append(f"  {100 * self.share():6.1f}%  all together")
        return "\n".join(out)


def sample_restrictions(model: Model, params: Params,
                        priors: Mapping[str, Any] | None = None, *,
                        irf: Sequence[Sequence] = (), moments: Sequence[Sequence] = (),
                        cov: Covariance = None, draws: int = 1000, seed: int | None = 0,
                        points: ArrayLike | None = None, names: Sequence[str] | None = None,
                        relative_irf: bool = False, guess: Point | None = None,
                        steady_fn: Callable[[Params], Any] | None = None,
                        qz_criterium: float | None = None) -> RestrictionSample:
    """Evaluate IRF and moment restrictions over draws from the prior.

    Parameters
    ----------
    priors : ``{name: Prior}`` or ``{name: (shape, mean, sd[, p3, p4])}``, as for
        :class:`EstimationProblem`. A name is a parameter or ``"stderr e"`` for a
        shock's standard deviation.
    irf, moments, relative_irf : as for :func:`check_restrictions`.
    cov : the shock covariance held fixed (the ``"stderr"`` draws replace it).
    draws, seed : how many draws, and the seed of the generator.
    points : evaluate at these rows instead of drawing, one column per name in
        ``names`` (by default the keys of ``priors``).
    guess, steady_fn : the steady state at each draw is ``steady_fn(params)``, or
        :func:`steady_state` from ``guess``.
    qz_criterium : as for :func:`solve`.

    A draw without a determinate solution is kept in the result with its status
    and ``nan`` values, and left out of the shares.
    """
    from .estimation import unpack_point
    from .klein import BlanchardKahnError
    from .priors import Prior, prior as make_prior
    from .solve import solve
    from .steady import steady_state

    restrictions = _restrictions(model, irf, moments)
    pr = {k: v if isinstance(v, Prior) else make_prior(*v) for k, v in dict(priors or {}).items()}
    names = list(names) if names is not None else list(pr)
    if not names:
        raise ValueError("nothing varies: give priors, or points with names")
    if points is None:
        missing = [n for n in names if n not in pr]
        if missing:
            raise ValueError(f"no prior for {missing}")
        if draws < 1:
            raise ValueError("draws must be positive")
        rng = np.random.default_rng(seed)
        X = np.column_stack([np.atleast_1d(pr[n].sample(rng, draws)) for n in names])
    else:
        X = np.atleast_2d(np.asarray(points, dtype=float))
        if X.shape[1] != len(names):
            raise ValueError(f"points have {X.shape[1]} columns for {len(names)} names")

    status = np.full(X.shape[0], "other", dtype=object)
    values = [np.full((X.shape[0], len(r.periods)), np.nan) for r in restrictions]
    for t, x in enumerate(X):
        try:
            p, Sigma, _ = unpack_point(model, params, cov, {}, names, x)
            ss = steady_fn(p) if steady_fn is not None else steady_state(model, guess, p)
            sol = solve(model, p, steady=ss, qz_criterium=qz_criterium)
            vals = _values(sol, Sigma, restrictions, relative_irf)
        except BlanchardKahnError:
            status[t] = "blanchard_kahn"
            continue
        except RuntimeError:
            status[t] = "no_steady_state"
            continue
        except (ValueError, ArithmeticError, np.linalg.LinAlgError):
            continue
        status[t] = "ok"
        for i, v in enumerate(vals):
            values[i][t] = v
    return RestrictionSample(restrictions, names, X, status.astype(str), values)
