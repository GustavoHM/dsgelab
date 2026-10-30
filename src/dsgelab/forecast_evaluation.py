"""Out-of-sample forecast evaluation: recursive estimation, accuracy and calibration.

:func:`forecast_evaluation` repeats, for each forecast origin, what an applied
forecaster does at that date: it takes the sample up to the origin, estimates
the model on it, filters it, and forecasts the observables ``1..horizon``
periods ahead. This is Dynare's recursive estimation (``estimation(nobs=[T1:T2],
forecast=H)``), and with ``window`` the rolling one (a range of ``first_obs``).
The forecasts are then set against the observations that followed:

* :meth:`ForecastEvaluation.rmse` and :meth:`~ForecastEvaluation.relative_rmse`,
  the root mean squared error by horizon, alone and as a ratio to the random walk
  (no change from the last observation), as in the comparisons of Smets and
  Wouters (2007) and Del Negro and Schorfheide (2013);
* :meth:`~ForecastEvaluation.diebold_mariano`, the test of equal accuracy of
  Diebold and Mariano (1995) with the small-sample correction of Harvey,
  Leybourne and Newbold (1997), against the random walk or any other errors;
* :meth:`~ForecastEvaluation.pit`, :meth:`~ForecastEvaluation.coverage_rate` and
  :meth:`~ForecastEvaluation.log_score`, the calibration of the predictive
  density: its probability integral transform, which is uniform when the
  density is right, the share of observations inside the bands, and the log
  predictive score.

The predictive density at each origin is the Kalman filter's Gaussian forecast
at the estimated parameters: mean ``d + Z a_{T+h|T}`` and variance
``Z V_h Z' + H``, with ``V_h`` compounding the filtered state's uncertainty and
the shocks of the horizon. It is conditional on the point estimate and leaves
out parameter uncertainty, as Dynare's forecast after the posterior mode does
(:func:`~dsgelab.posterior_forecast` adds it at one origin).
"""

from __future__ import annotations

import dataclasses
import math
import warnings
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from numpy.typing import ArrayLike

from .estimation import EstimationProblem, ModeResult, maximum_likelihood, posterior_mode
from .kalman import kalman_filter

__all__ = ["ForecastEvaluation", "DieboldMariano", "forecast_evaluation", "diebold_mariano"]


@dataclass
class DieboldMariano:
    """A Diebold-Mariano test, with the Harvey-Leybourne-Newbold correction.

    ``statistic`` is negative when the first forecast has the smaller loss.
    ``horizon`` is the one used for the variance, which falls back to 1 when the
    long-run variance at the requested horizon is not positive.
    """

    statistic: float
    p_value: float
    horizon: int
    power: float
    alternative: str
    n: int


def diebold_mariano(e1: ArrayLike, e2: ArrayLike, horizon: int = 1, *, power: float = 2,
                    alternative: str = "two.sided", varestimator: str = "acf") -> DieboldMariano:
    """Test that two forecasts are equally accurate, from their errors.

    The loss differential is ``d_t = |e1_t|^power - |e2_t|^power``. Its long-run
    variance sums the autocovariances up to lag ``horizon - 1``, with equal
    weights (``"acf"``) or Bartlett weights (``"bartlett"``), since ``h``-step
    errors are serially correlated up to that lag. The statistic is scaled by
    ``sqrt((n + 1 - 2h + h(h - 1)/n) / n)`` and compared with Student's ``t``
    with ``n - 1`` degrees of freedom (Harvey, Leybourne and Newbold, 1997).
    ``alternative`` is ``"two.sided"``, ``"less"`` (the first forecast is more
    accurate) or ``"greater"``. Pairs with a missing error are dropped.

    This is R's ``forecast::dm.test``, with the same fallback: when the variance
    is not positive at ``horizon > 1``, it warns and uses ``horizon = 1``.
    """
    from scipy.stats import t as student

    if alternative not in ("two.sided", "less", "greater"):
        raise ValueError("alternative must be 'two.sided', 'less' or 'greater'")
    if varestimator not in ("acf", "bartlett"):
        raise ValueError("varestimator must be 'acf' or 'bartlett'")
    h = int(horizon)
    if h < 1:
        raise ValueError("horizon must be at least 1")
    a, b = np.asarray(e1, float).ravel(), np.asarray(e2, float).ravel()
    if a.shape != b.shape:
        raise ValueError("the two error series must have the same length")
    d = np.abs(a) ** power - np.abs(b) ** power
    d = d[np.isfinite(d)]
    n = d.size
    if h > n:
        raise ValueError("horizon cannot exceed the number of forecast errors")
    dc = d - d.mean()
    gamma = np.array([dc[k:] @ dc[:n - k] / n for k in range(h)])
    if varestimator == "acf" or h == 1:
        weights = np.ones(h - 1)
    else:
        weights = 1 - np.arange(1, h) / h
    var = (gamma[0] + 2 * weights @ gamma[1:]) / n
    if not var > 0:
        if h == 1:
            raise ValueError("the variance of the loss differential is zero")
        warnings.warn("the long-run variance is not positive; using horizon 1", stacklevel=2)
        return diebold_mariano(a, b, 1, power=power, alternative=alternative,
                               varestimator=varestimator)
    k = math.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = float(d.mean() / math.sqrt(var) * k)
    if alternative == "two.sided":
        p = 2 * student.cdf(-abs(stat), n - 1)
    elif alternative == "less":
        p = student.cdf(stat, n - 1)
    else:
        p = student.sf(stat, n - 1)
    return DieboldMariano(statistic=stat, p_value=float(p), horizon=h, power=float(power),
                          alternative=alternative, n=n)


@dataclass
class ForecastEvaluation:
    """Forecasts made at each origin and the observations that followed.

    The arrays in ``mean``, ``std`` and ``actual`` are ``(origins, horizon)``
    for each observable, in the units of the data; ``actual`` is NaN past the end
    of the sample, and those pairs are left out of every statistic. ``origins``
    are the number of observations up to each origin, counted as the rows of the
    problem's data (Dynare's ``nobs``). ``last`` is the observation at the
    origin, the random walk's forecast. ``theta`` holds the parameters used at
    each origin, in the order of ``names``; when they were estimated by
    ``"mode"`` or ``"ml"``, ``modes`` holds the :class:`~dsgelab.ModeResult` of
    each origin (with the standard deviations) and ``objective`` the posterior
    kernel (or the likelihood) there.
    """

    variables: list
    origins: np.ndarray
    horizon: int
    mean: dict
    std: dict
    actual: dict
    last: dict
    covariance: np.ndarray      # (origins, horizon, n_obs, n_obs)
    names: list
    theta: np.ndarray
    objective: np.ndarray
    coverage: float
    modes: list

    @property
    def errors(self) -> dict:
        """Forecast errors, ``actual - mean``."""
        return {v: self.actual[v] - self.mean[v] for v in self.variables}

    @property
    def random_walk_errors(self) -> dict:
        """Errors of the random walk, ``actual - last``."""
        return {v: self.actual[v] - self.last[v][:, None] for v in self.variables}

    def rmse(self, *, benchmark: bool = False) -> dict:
        """Root mean squared error by horizon; of the random walk with ``benchmark``."""
        e = self.random_walk_errors if benchmark else self.errors
        return {v: _rms(e[v]) for v in self.variables}

    def relative_rmse(self) -> dict:
        """The model's RMSE over the random walk's, by horizon: below 1 is better."""
        m, r = self.rmse(), self.rmse(benchmark=True)
        return {v: m[v] / r[v] for v in self.variables}

    def diebold_mariano(self, variable: str, horizon: int, *,
                        benchmark: str | ArrayLike = "random_walk", power: float = 2,
                        alternative: str = "two.sided",
                        varestimator: str = "acf") -> DieboldMariano:
        """Diebold-Mariano test of the ``horizon``-step forecasts of ``variable``.

        ``benchmark`` is ``"random_walk"`` or the errors of another forecast at
        the same origins, ``(origins,)``. A negative statistic favours the model.
        """
        if not 1 <= horizon <= self.horizon:
            raise ValueError(f"horizon must be between 1 and {self.horizon}")
        e1 = self.errors[variable][:, horizon - 1]
        if isinstance(benchmark, str):
            if benchmark != "random_walk":
                raise ValueError("benchmark must be 'random_walk' or an array of errors")
            e2 = self.random_walk_errors[variable][:, horizon - 1]
        else:
            e2 = np.asarray(benchmark, float).ravel()
        return diebold_mariano(e1, e2, horizon, power=power, alternative=alternative,
                               varestimator=varestimator)

    def pit(self) -> dict:
        """Probability integral transform, ``Phi((actual - mean) / std)``."""
        from scipy.stats import norm
        return {v: norm.cdf((self.actual[v] - self.mean[v]) / self.std[v])
                for v in self.variables}

    def coverage_rate(self, coverage: float | None = None) -> dict:
        """Share of observations inside the central ``coverage`` band, by horizon."""
        from scipy.stats import norm
        z = norm.ppf(0.5 + (self.coverage if coverage is None else coverage) / 2)
        out = {}
        for v in self.variables:
            inside = np.abs(self.actual[v] - self.mean[v]) <= z * self.std[v]
            ok = np.isfinite(self.actual[v])
            out[v] = np.array([inside[ok[:, h], h].mean() if ok[:, h].any() else np.nan
                               for h in range(self.horizon)])
        return out

    def log_score(self, *, joint: bool = False) -> dict | np.ndarray:
        """Log predictive density of the outcomes, ``(origins, horizon)``.

        Per observable by default; with ``joint``, of all the observables
        together under their joint Gaussian density (NaN where any is missing).
        """
        from scipy.stats import norm
        if not joint:
            return {v: norm.logpdf(self.actual[v], self.mean[v], self.std[v])
                    for v in self.variables}
        Y = np.stack([self.actual[v] for v in self.variables], axis=-1)
        M = np.stack([self.mean[v] for v in self.variables], axis=-1)
        out = np.full(Y.shape[:2], np.nan)
        for o in range(Y.shape[0]):
            for h in range(Y.shape[1]):
                e = Y[o, h] - M[o, h]
                if np.all(np.isfinite(e)):
                    S = self.covariance[o, h]
                    sign, logdet = np.linalg.slogdet(S)
                    if sign > 0:
                        out[o, h] = -0.5 * (len(e) * math.log(2 * math.pi) + logdet
                                            + e @ np.linalg.solve(S, e))
        return out


def _rms(e: np.ndarray) -> np.ndarray:
    out = np.full(e.shape[1], np.nan)
    for h in range(e.shape[1]):
        x = e[:, h][np.isfinite(e[:, h])]
        if x.size:
            out[h] = math.sqrt(float(x @ x) / x.size)
    return out


def forecast_evaluation(problem: EstimationProblem, origins: Sequence[int], horizon: int, *,
                        estimate: str | Callable = "mode", theta: ArrayLike | None = None,
                        window: int | None = None, coverage: float = 0.9,
                        include_measurement: bool = True,
                        options: dict | None = None) -> ForecastEvaluation:
    """Re-estimate at each origin and forecast the observables ``1..horizon`` ahead.

    Parameters
    ----------
    problem : EstimationProblem
        The full sample, at first order. Each origin estimates on its rows of
        ``problem``'s data (after ``first_obs``), with the same priors, fixed
        parameters, measurement errors and Kalman initialisation.
    origins : sequence of int
        Sample sizes at which the forecasts are made, ``n`` meaning that the
        estimation sample ends at row ``n`` (Dynare's ``nobs``).
    horizon : int
    estimate : ``"mode"``, ``"ml"``, ``"fixed"`` or a callable
        ``"mode"`` maximises the posterior at each origin, ``"ml"`` the
        likelihood. Each search starts from the estimate at the previous origin,
        and the first from ``theta`` or the problem's starting values. ``"fixed"``
        uses ``theta`` throughout: one vector, or one row per origin. A callable
        ``estimate(problem_at_origin, x0)`` returns the parameters (or a
        :class:`~dsgelab.ModeResult`).
    window : int, optional
        Rolling estimation on the last ``window`` observations before each
        origin. By default the window expands from the start of the sample.
    coverage : float
        Coverage of the bands used by :meth:`ForecastEvaluation.coverage_rate`.
    include_measurement : bool
        Add the measurement errors to the predictive variance, so that it is
        the density of the observation rather than of the model variable.
    options : dict, optional
        Passed to the optimiser for ``"mode"`` and ``"ml"``.
    """
    if problem.order != 1:
        raise ValueError("forecast evaluation works on the first-order (Kalman) problem")
    Y = problem.Y
    T, m = Y.shape
    origins = np.array([int(o) for o in origins])
    if origins.size == 0:
        raise ValueError("give at least one origin")
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    lo = 1 if window is None else int(window)
    if np.any(origins < lo) or np.any(origins > T):
        raise ValueError(f"origins must lie between {lo} and the {T} observations")
    k = len(problem.names)
    if isinstance(estimate, str) and estimate not in ("mode", "ml", "fixed"):
        raise ValueError("estimate must be 'mode', 'ml', 'fixed' or a callable")
    fixed = None
    if estimate == "fixed":
        if theta is None:
            raise ValueError("estimate='fixed' needs theta")
        fixed = np.asarray(theta, float)
        fixed = np.tile(fixed, (origins.size, 1)) if fixed.ndim == 1 else fixed
        if fixed.shape != (origins.size, k):
            raise ValueError(f"theta must have {k} values, or one row of them per origin")
    x0 = None if theta is None or fixed is not None else np.asarray(theta, float)

    H = horizon
    obs = problem.observables
    means = np.zeros((origins.size, H, m))
    covs = np.zeros((origins.size, H, m, m))
    actual = np.full((origins.size, H, m), np.nan)
    thetas = np.zeros((origins.size, k))
    objective = np.full(origins.size, np.nan)
    modes = [None] * origins.size
    for o, n in enumerate(origins):
        start = 0 if window is None else n - window
        sub = dataclasses.replace(problem, data=Y[start:n], first_obs=1)
        if fixed is not None:
            th = fixed[o]
        else:
            if estimate == "mode":
                res = posterior_mode(sub, x0, options=options)
            elif estimate == "ml":
                res = maximum_likelihood(sub, x0, options=options)
            else:
                res = estimate(sub, x0)
            if isinstance(res, ModeResult):
                th, objective[o], modes[o] = res.x, res.log_posterior, res
            else:
                th = np.asarray(res, float)
            x0 = th
        thetas[o] = th
        _, space = sub.solve(th)
        fr = kalman_filter(space, sub.Y, init=sub.init)
        a, V = fr.a_filt[-1].copy(), fr.P_filt[-1].copy()
        RQR = space.R @ space.Sigma @ space.R.T
        for h in range(H):
            a = space.T @ a
            V = space.T @ V @ space.T.T + RQR
            means[o, h] = space.d + space.Z @ a
            S = space.Z @ V @ space.Z.T
            covs[o, h] = 0.5 * (S + S.T) + (space.H if include_measurement else 0.0)
            if n + h < T:
                actual[o, h] = Y[n + h]
    std = np.sqrt(np.maximum(np.diagonal(covs, axis1=2, axis2=3), 0.0))
    return ForecastEvaluation(
        variables=list(obs), origins=origins, horizon=H,
        mean={v: means[:, :, i] for i, v in enumerate(obs)},
        std={v: std[:, :, i] for i, v in enumerate(obs)},
        actual={v: actual[:, :, i] for i, v in enumerate(obs)},
        last={v: Y[origins - 1, i] for i, v in enumerate(obs)},
        covariance=covs, names=list(problem.names), theta=thetas, objective=objective,
        coverage=coverage, modes=modes)
