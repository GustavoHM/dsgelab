"""Plots of the results, as Dynare draws them, and the statistics the plots draw.

Each function takes what the package already returns -- the dictionaries of
:meth:`~dsgelab.FirstOrderSolution.irf` and :meth:`~dsgelab.FirstOrderSolution.simulate`,
:func:`~dsgelab.shock_decomposition`, a :class:`~dsgelab.Forecast`, an
:class:`~dsgelab.MCMCResult`, a :class:`~dsgelab.SmootherResult`, an
:class:`~dsgelab.OccbinResult`, an :class:`~dsgelab.Identification`, a
:class:`~dsgelab.FilteringResult` -- and returns a ``matplotlib`` figure, built
without ``pyplot`` so that nothing global changes: save it with
``fig.savefig(...)``, or let a notebook show it. matplotlib is an optional
dependency (``pip install dsgelab[plot]``).

The figures are those of Dynare's commands: impulse responses and simulated
paths (``stoch_simul``, ``rplot``), shock decompositions
(``plot_shock_decomposition``), forecasts and conditional forecasts, the priors
and the posteriors of ``estimation``, its MCMC diagnostics (``trace_plot``,
``mh_autocorrelation_function`` and the convergence diagnostics of Brooks and
Gelman, 1998), ``mode_check``, the smoothed shocks and variables, posterior
impulse responses with their HPD intervals, the OccBin paths and regimes, the
collinearity patterns, strength, sensitivity and identification patterns of
``identification``, and the distributions compared by Monte Carlo filtering.
Shock decompositions are drawn in real time, by vintage and at annual frequency,
and conditional forecasts with their bands.

The numbers a figure draws that the package does not compute elsewhere are
computed as Dynare computes them, and are available without matplotlib:
:func:`prior_density` is ``draw_prior_density`` (200 points between the
quantiles 0.001 and 0.999), :func:`posterior_density` is the kernel estimate of
``posterior_moments`` (Gaussian kernel, Silverman's bandwidth corrected for
repeated Metropolis draws, 512 points), :func:`brooks_gelman` is the
univariate and multivariate diagnostic of ``mcmc_diagnostics``,
:func:`mh_autocorrelation` is ``sample_autocovariance``, :func:`mode_check` is
``mode_check`` with its grid, and :func:`posterior_irf` is ``PosteriorIRF``,
with the HPD interval and the deciles of ``posterior_moments``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike

__all__ = ["prior_density", "posterior_density", "brooks_gelman", "BrooksGelman",
           "mh_autocorrelation", "mode_check", "posterior_irf",
           "plot_irf", "plot_paths", "plot_decomposition", "plot_forecast", "plot_posterior",
           "plot_priors", "trace_plot", "plot_mh_autocorrelation", "plot_brooks_gelman",
           "plot_mode_check", "plot_smoothed_shocks", "plot_smoothed_observables",
           "plot_posterior_irf", "plot_conditional_forecast", "plot_occbin",
           "plot_occbin_regimes", "plot_identification", "plot_filtering",
           "plot_identification_strength", "plot_identification_sensitivity",
           "plot_identification_patterns"]


# ================================================================ densities
def prior_density(prior, *, steps: int = 200, truncation: float = 1e-3) -> tuple[np.ndarray, np.ndarray]:
    """``(x, f)``: a prior's density on a grid over its bulk, Dynare's ``draw_prior_density``.

    The grid runs between the quantiles ``truncation`` and ``1 - truncation``
    (``10 truncation`` for the inverse gammas, whose right tail is long), within
    the support. Where the density peaks at an end of the grid, values above 10
    are left out (NaN), as Dynare does, so that a pole does not flatten the plot.
    """
    from scipy import stats
    from .priors import Prior, prior as make_prior

    p = prior if isinstance(prior, Prior) else make_prior(*prior)
    s, q = p.shape, truncation
    if s == "beta":
        lo, hi = (p.lower + (p.upper - p.lower) * stats.beta.ppf(u, p.a, p.b) for u in (q, 1 - q))
    elif s == "gamma":
        lo, hi = (p.lower + stats.gamma.ppf(u, p.a, scale=p.b) for u in (q, 1 - q))
    elif s == "normal":
        lo = max(p.lower, stats.norm.ppf(q, p.mean, p.sd))
        hi = min(p.upper, stats.norm.ppf(1 - q, p.mean, p.sd))
    elif s in ("inv_gamma", "inv_gamma2"):
        g = [stats.gamma.ppf(u, p.b / 2, scale=2 / p.a) for u in (1 - 10 * q, 10 * q)]
        if s == "inv_gamma":
            lo, hi = (p.lower + 1 / math.sqrt(v) for v in g)
        else:
            lo, hi = (p.lower + 1 / v for v in g)
    elif s == "uniform":
        lo, hi = p.lower, p.upper
    else:
        raise ValueError(f"unknown prior shape {s!r}")
    x = np.linspace(lo, hi, steps)
    if s == "uniform":
        return x, np.full(steps, 1.0 / (hi - lo))
    f = np.exp([p.logpdf(v) for v in x])
    k = int(np.argmax(f))
    if k in (0, steps - 1):
        f[f > 10] = np.nan
    return x, f


def _repeated_draws(x: np.ndarray) -> float:
    """Dynare's correction for repeated draws: ``mean(2 m_i - 1)``, ``m_i`` the run starting at ``i``."""
    eps = np.finfo(float).eps
    n = x.size
    total = 0
    for i in range(n):
        j = int(np.searchsorted(x, x[i] + 2 * eps, side="left"))
        while j > i + 1 and x[j - 1] - x[i] >= 2 * eps:
            j -= 1
        while j < n and x[j] - x[i] < 2 * eps:
            j += 1
        total += 2 * (j - i) - 1
    return total / n


def posterior_density(draws: ArrayLike, *, grid_points: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """``(x, f)``: kernel density of a parameter's draws, as Dynare's ``posterior_moments``.

    A Gaussian kernel with Silverman's rule-of-thumb bandwidth, multiplied by
    ``c^(1/5)`` where ``c`` corrects for the draws a Metropolis chain repeats
    when it rejects, evaluated by linear binning and a fast Fourier transform on
    ``grid_points`` points (a power of two) spanning the draws and a third of
    their range on each side.
    """
    x = np.sort(np.asarray(draws, dtype=float).ravel())
    n = x.size
    G = int(grid_points)
    if G < 2 or G & (G - 1):
        raise ValueError("grid_points must be a power of two")
    if n < 2 or not np.var(x, ddof=1) > 1e-12:
        return np.full(G, np.nan), np.full(G, np.nan)
    mu02, mu21 = 1 / (2 * math.sqrt(math.pi)), 1.0
    h = 2 * np.std(x, ddof=1) * (math.sqrt(math.pi) * mu02 / (12 * mu21 ** 2 * n)) ** 0.2
    h *= _repeated_draws(x) ** 0.2
    lo = x[0] - (x[-1] - x[0]) / 3
    hi = x[-1] + (x[-1] - x[0]) / 3
    abscissa = np.linspace(lo, hi, G)
    inc = abscissa[1] - abscissa[0]
    xi = np.zeros(G + 1)
    xa = (x - lo) / (hi - lo) * G
    idx = np.floor(xa).astype(int)
    frac = xa - idx
    np.add.at(xi, idx, 1 - frac)
    np.add.at(xi, idx + 1, frac)
    xk = np.arange(-G, G) * inc
    kk = np.exp(-0.5 * (xk / h) ** 2) / math.sqrt(2 * math.pi)
    kk = kk / (kk.sum() * inc * n)
    f = np.fft.ifft(np.fft.fft(np.fft.fftshift(kk)) * np.fft.fft(np.concatenate([xi[:G], np.zeros(G)])))
    return abscissa, np.real(f[:G])


def _round(v: float) -> int:
    """MATLAB's ``round``, half away from zero, for the indices Dynare computes with it."""
    return int(math.floor(abs(v) + 0.5)) * (1 if v >= 0 else -1)


def _chains(draws: Any, names: Sequence[str] | None = None):
    """``(chains, names, log posterior)`` from an MCMCResult or an array of draws.

    An array is ``(draws,)``, one series, ``(draws, parameters)``, one chain, or
    ``(chains, draws, parameters)``. The log posterior is None for an array.
    """
    if hasattr(draws, "chains") and hasattr(draws, "names"):
        lp = getattr(draws, "log_posterior", None)
        return (np.asarray(draws.chains, dtype=float), list(draws.names),
                None if lp is None else np.asarray(lp, dtype=float))
    C = np.asarray(draws, dtype=float)
    if C.ndim == 1:
        C = C[None, :, None]
    elif C.ndim == 2:
        C = C[None]
    labels = [f"x{i}" for i in range(C.shape[2])] if names is None else list(names)
    if len(labels) != C.shape[2]:
        raise ValueError(f"{C.shape[2]} parameters for {len(labels)} names")
    return C, labels, None


# ======================================================== MCMC diagnostics
@dataclass
class BrooksGelman:
    """The convergence diagnostics of Brooks and Gelman (1998). See :func:`brooks_gelman`.

    Attributes
    ----------
    iterations : the last draw of each window.
    names : the parameters.
    univariate : ``(windows, 6, parameters)``, Dynare's ``UDIAG``. For each
        window, the length of the 80% interval of the pooled draws and the mean of
        its length within each chain, the pooled and the mean within-chain
        variance (m2), and the pooled and the mean within-chain third absolute
        moment (m3).
    multivariate : ``(windows, 6)``, Dynare's ``MDIAG``: the same six numbers for
        the log posterior, which aggregates the parameters; None without it.
    """

    iterations: np.ndarray
    names: list
    univariate: np.ndarray
    multivariate: np.ndarray | None = None


def _brooks_gelman_series(X: np.ndarray, iterations: np.ndarray, drop: float,
                          alpha: float) -> np.ndarray:
    """Dynare's six statistics for one quantity, ``X`` of shape ``(chains, draws)``."""
    m = X.shape[0]
    out = np.zeros((iterations.size, 6))
    for w, it in enumerate(iterations):
        # the window keeps draws first..it (from 1), dropping the share ``drop`` of the first it.
        # Dynare's count it - first + 1 is one too many when drop is 0, where first is 0
        first = max(1, math.ceil(drop * it))
        n = it - first + 1
        W = X[:, first - 1:it]
        cinf, csup = max(1, _round(n * alpha / 2)), _round(n * (1 - alpha / 2))
        CINF, CSUP = max(1, _round(m * n * alpha / 2)), _round(m * n * (1 - alpha / 2))
        pooled = np.sort(W.ravel())
        out[w, 0] = pooled[CSUP - 1] - pooled[CINF - 1]
        dev = np.abs(W - pooled.mean())
        out[w, 2] = np.sum(dev ** 2) / (m * n - 1)
        out[w, 4] = np.sum(dev ** 3) / (m * n - 1)
        within = np.sort(W, axis=1)
        out[w, 1] = np.mean(within[:, csup - 1] - within[:, cinf - 1])
        dev = np.abs(W - W.mean(axis=1, keepdims=True))
        out[w, 3] = np.mean(np.sum(dev ** 2, axis=1) / (n - 1))
        out[w, 5] = np.mean(np.sum(dev ** 3, axis=1) / (n - 1))
    return out


def brooks_gelman(draws: Any, *, names: Sequence[str] | None = None,
                  log_posterior: ArrayLike | None = None, drop: float = 0.5, origin: int = 1000,
                  step: int | None = None, alpha: float = 0.2) -> BrooksGelman:
    """Brooks and Gelman's (1998) convergence diagnostics, as Dynare's ``mcmc_diagnostics``.

    ``draws`` is an :class:`~dsgelab.MCMCResult` or a ``(chains, draws,
    parameters)`` array with ``names``. Over windows that end at draws
    ``origin, origin + step, ...`` (``step`` makes about 100 windows by default)
    and drop the first share ``drop`` of the draws up to their end, each
    parameter's draws give three measures of their spread: the length of the
    interval that holds ``1 - alpha`` of them, their variance and their third
    absolute moment. Each is computed on the draws of all chains pooled, and
    within each chain and averaged. When the chains have converged, the pooled
    and the within-chain measures settle at the same value. The multivariate
    diagnostic computes the same on the log posterior (the chains' own for an
    MCMCResult, or ``log_posterior``, ``(chains, draws)``).

    Dynare runs it on the whole chains, burn-in included, with its ``mh_drop``
    (0.5) as ``drop``: :func:`~dsgelab.metropolis_hastings` with ``burn=0``
    keeps the draws it needs.
    """
    C, labels, lp = _chains(draws, names)
    if log_posterior is not None:
        lp = np.atleast_2d(np.asarray(log_posterior, dtype=float))
    m, N, k = C.shape
    if N <= origin:
        raise ValueError(f"{N} draws per chain: the first window ends at draw origin={origin}")
    step = step if step is not None else math.ceil((N - origin) / 100)
    iterations = np.arange(origin, N + 1, step)
    uni = np.stack([_brooks_gelman_series(C[:, :, j], iterations, drop, alpha)
                    for j in range(k)], axis=2)
    multi = None if lp is None else _brooks_gelman_series(lp, iterations, drop, alpha)
    return BrooksGelman(iterations=iterations, names=labels, univariate=uni, multivariate=multi)


def mh_autocorrelation(draws: ArrayLike, lags: int = 30) -> np.ndarray:
    """Autocorrelations of a chain's draws at lags ``0..lags``, Dynare's ``sample_autocovariance``.

    Each autocovariance sums the products of the deviations from the mean over
    the same draws, the last ``N - lags``, and divides by their number. Dynare's
    ``mh_autocorrelation_function`` draws them (``lags`` is its
    ``mh_autocorrelation_function_size``) for the draws of one chain after
    burn-in.
    """
    x = np.asarray(draws, dtype=float).ravel()
    n = x.size - lags
    if n < 1:
        raise ValueError(f"{x.size} draws for {lags} lags")
    d = x - x.mean()
    cov = np.array([d[lags:] @ d[lags - j:x.size - j] for j in range(lags + 1)]) / n
    return cov / cov[0]


def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """Centred mean over ``2 window + 1`` draws, NaN where it does not fit, as ``trace_plot``."""
    out = np.full(x.size, np.nan)
    if x.size > 2 * window:
        out[window:x.size - window] = np.lib.stride_tricks.sliding_window_view(
            x, 2 * window + 1).mean(axis=1)
    return out


# ================================================================ mode check
def _prior_bounds(p, truncation: float) -> tuple[float, float]:
    """The prior's quantiles ``truncation`` and ``1 - truncation``, Dynare's ``prior_bounds``."""
    from scipy import stats
    s, q = p.shape, truncation
    if q == 0:
        return p.lower, p.upper
    if s == "beta":
        return tuple(p.lower + (p.upper - p.lower) * stats.beta.ppf(u, p.a, p.b) for u in (q, 1 - q))
    if s == "gamma":
        return tuple(p.lower + stats.gamma.ppf(u, p.a, scale=p.b) for u in (q, 1 - q))
    if s == "normal":
        return (max(stats.norm.ppf(q, p.mean, p.sd), p.lower),
                min(stats.norm.ppf(1 - q, p.mean, p.sd), p.upper))
    if s in ("inv_gamma", "inv_gamma2"):
        g = [stats.gamma.ppf(u, p.b / 2, scale=2 / p.a) for u in (1 - q, q)]
        if s == "inv_gamma":
            return tuple(p.lower + 1 / math.sqrt(v) for v in g)
        return tuple(p.lower + 1 / v for v in g)
    if s == "uniform":
        return p.lower + (p.upper - p.lower) * q, p.upper - (p.upper - p.lower) * q
    raise ValueError(f"unknown prior shape {s!r}")


def _colon(a: float, d: float, b: float) -> np.ndarray:
    """``a:d:b`` as Octave builds it: ``a + i d``, the last element clipped to ``b``."""
    if d == 0 or not np.isfinite(d) or (b - a) / d < 0:
        return np.empty(0)
    n = int(math.floor((b - a) / d * (1 + 4 * np.finfo(float).eps))) + 1
    z = a + d * np.arange(n)
    if (d > 0 and z[-1] > b) or (d < 0 and z[-1] < b):
        z[-1] = b
    return z


def mode_check(problem: Any, mode: Any, *, neighbourhood: float = 0.5, points: int = 20,
               symmetric: bool = True, prior_trunc: float = 0.0) -> dict:
    """Slices of the log posterior and the log likelihood through the mode, Dynare's ``mode_check``.

    For each parameter of ``problem`` (an :class:`~dsgelab.EstimationProblem`),
    the others held at ``mode`` (a :class:`~dsgelab.ModeResult`, a mapping or an
    array), the log posterior kernel is evaluated on a grid of ``points + 1``
    values, half below the mode and half above. The grid is Dynare's: it spans
    ``neighbourhood`` times the mode on each side, within the bounds of the
    estimation (the support of the prior, or its quantiles ``prior_trunc`` and
    ``1 - prior_trunc``, as Dynare's option of that name), and is made symmetric around the mode when a
    bound cuts one side (``symmetric``, Dynare's ``symmetric_plots``). A zero mode
    or an infinite bound takes Dynare's other rule, which uses the prior standard
    deviation.

    Returns ``{name: {"x": grid, "log_posterior": ..., "log_likelihood": ...,
    "mode": value}}``. The log likelihood is shifted by the log prior at the mode,
    Dynare's log-likelihood kernel, so that the two curves meet at the mode.
    Where the model cannot be solved, both are NaN.
    """
    names = list(problem.names)
    if hasattr(mode, "x"):
        x0 = np.asarray(mode.x, dtype=float)
    elif isinstance(mode, Mapping):
        x0 = np.array([float(mode[n]) for n in names])
    else:
        x0 = np.asarray(mode, dtype=float).ravel()
    if x0.size != len(names):
        raise ValueError(f"{x0.size} values for {len(names)} parameters")
    ll = float(neighbourhood)
    symmetric = symmetric and np.isfinite(ll)
    lp_mode = problem.log_prior(x0)
    out = {}
    for k, name in enumerate(names):
        pr = problem.priors.get(name)
        lb, ub = (-np.inf, np.inf) if pr is None else _prior_bounds(pr, prior_trunc)
        if problem.bounds and name in problem.bounds:
            lb, ub = max(lb, problem.bounds[name][0]), min(ub, problem.bounds[name][1])
        xk = x0[k]
        moved = False
        if xk != 0 and np.isfinite(lb) and np.isfinite(ub):
            l1 = max(lb, (1 - np.sign(xk) * ll) * xk)
            l2 = min(ub, (1 + np.sign(xk) * ll) * xk)
        else:
            # Dynare's code, as it is written: the lower bound stands for the upper and the
            # reverse, -10 when infinite (its huge_number times -1e-6), around the prior sd
            sd = pr.sd if pr is not None else np.nan
            upper = lb if np.isfinite(lb) else -10.0
            lower = ub if np.isfinite(ub) else -10.0
            l1 = lower if not -sd > lower else -sd
            l2 = upper if not sd < upper else sd
        binding = abs(xk - lb) < 1e-8 or abs(xk - ub) < 1e-8
        if symmetric and not binding:
            if l2 < (1 + ll) * xk:
                l1, moved = xk - (l2 - xk), True
            if not moved and l1 > (1 - ll) * xk and xk + (xk - l1) < ub:
                l2 = xk + (xk - l1)
        z = np.union1d(_colon(l1, (xk - l1) / (points / 2), xk),
                       _colon(xk, (l2 - xk) / (points / 2), l2))
        lpost = np.full(z.size, np.nan)
        llik = np.full(z.size, np.nan)
        for i, v in enumerate(z):
            theta = x0.copy()
            theta[k] = v
            lp = problem.log_posterior(theta)
            if np.isfinite(lp):
                lpost[i] = lp
                llik[i] = lp - problem.log_prior(theta) + lp_mode
        out[name] = {"x": z, "log_posterior": lpost, "log_likelihood": llik, "mode": float(xk)}
    return out


# ============================================================= posterior IRFs
def _posterior_moments(X: np.ndarray, mass: float) -> dict:
    """Mean, median, HPD interval and deciles along the first axis, as ``posterior_moments``.

    The HPD interval is the shortest that leaves out ``round((1 - mass) n)``
    draws, and the deciles are the draws ranked ``round(0.1 n), ...,
    round(0.9 n)``, as Dynare takes them.
    """
    xx = np.sort(np.asarray(X, dtype=float), axis=0)
    n = xx.shape[0]
    out = {"mean": xx.mean(axis=0), "median": np.median(xx, axis=0)}
    h = _round((1 - mass) * n)
    if h > 2:
        widths = xx[n - h - 1:n - 1] - xx[:h]
        j = np.argmin(widths, axis=0)[None]
        out["lower"] = np.take_along_axis(xx, j, axis=0)[0]
        out["upper"] = out["lower"] + np.take_along_axis(widths, j, axis=0)[0]
    else:
        out["lower"] = out["upper"] = np.full(xx.shape[1:], np.nan)
    if n > 9:
        out["deciles"] = np.moveaxis(xx[[_round(q / 10 * n) - 1 for q in range(1, 10)]], 0, -1)
    else:
        out["deciles"] = np.full(xx.shape[1:] + (9,), np.nan)
    return out


def posterior_irf(problem: Any, draws: Any, *, shocks: Sequence[str] | None = None,
                  variables: Sequence[str] | None = None, periods: int = 40,
                  n_draws: int | None = 1200, mass: float = 0.9, percent: bool = True,
                  seed: int = 0) -> dict:
    """Impulse responses over the posterior, with their HPD intervals, as Dynare's ``PosteriorIRF``.

    The model of ``problem`` is solved at ``n_draws`` parameter vectors taken
    from ``draws`` (an :class:`~dsgelab.MCMCResult` or a ``(draws, parameters)``
    array; all of them, in order, when ``n_draws`` is None or not smaller), and
    at each the response to a one-standard-deviation orthogonalised impulse of
    each shock is computed: the column of the Cholesky factor of the shock
    covariance at that draw, to which Dynare adds ``1e-14`` on the diagonal, as
    here. Draws at which the model cannot be solved are left out. ``variables``
    are the observables by default, as in Dynare, in the units of
    :meth:`~dsgelab.FirstOrderSolution.irf`.

    For each variable, shock and period the responses give the statistics of
    Dynare's ``posterior_moments``: the mean, the median, the shortest interval
    holding ``mass`` of them (Dynare's ``mh_conf_sig``) and the deciles.

    Returns ``{"mean": {shock: {variable: path}}, "median": ..., "lower": ...,
    "upper": ..., "deciles": {shock: {variable: (periods, 9)}}, "responses":
    {shock: {variable: (draws, periods)}}, "draws": number used, "mass": mass}``.
    """
    X = np.asarray(draws.draws if hasattr(draws, "draws") else draws, dtype=float)
    X = np.atleast_2d(X)
    if n_draws is not None and n_draws < len(X):
        X = X[np.random.default_rng(seed).choice(len(X), size=n_draws, replace=False)]
    names = list(variables) if variables is not None else list(problem.observables)
    stock, shock_names = [], None
    for x in X:
        try:
            sol, space = problem.solve(x)
        except Exception:
            continue
        shock_names = list(shocks) if shocks is not None else list(space.shocks)
        L = np.linalg.cholesky(space.Sigma + 1e-14 * np.eye(len(space.shocks)))
        resp = np.empty((len(shock_names), periods, len(names)))
        for i, e in enumerate(shock_names):
            paths = sol.irf_impulse(L[:, space.shocks.index(e)], periods, percent=percent)
            resp[i] = np.column_stack([paths[v] for v in names])
        stock.append(resp)
    if not stock:
        raise RuntimeError("no draw could be solved")
    assert shock_names is not None
    S = np.array(stock)                              # (draws, shocks, periods, variables)
    st = _posterior_moments(S, mass)
    out: dict[str, Any] = {key: {e: {v: st[key][i, :, j] for j, v in enumerate(names)}
                 for i, e in enumerate(shock_names)}
           for key in ("mean", "median", "lower", "upper", "deciles")}
    out["responses"] = {e: {v: S[:, i, :, j] for j, v in enumerate(names)}
                        for i, e in enumerate(shock_names)}
    out["draws"], out["mass"] = len(stock), mass
    return out


# ================================================================ plotting
def _figure(n: int, columns: int, figsize=None):
    try:
        from matplotlib.figure import Figure
    except ImportError:                                  # pragma: no cover
        raise ImportError("plots need matplotlib: pip install dsgelab[plot]") from None
    columns = max(1, min(columns, n))
    rows = math.ceil(n / columns)
    fig = Figure(figsize=figsize or (3.2 * columns, 2.4 * rows), layout="constrained")
    axes = fig.subplots(rows, columns, squeeze=False).ravel()
    for ax in axes[n:]:
        ax.set_visible(False)
    return fig, axes[:n]


def _overlay(series: Mapping) -> dict:
    """``{label: {variable: path}}``, from one mapping of paths or several."""
    if series and all(isinstance(v, Mapping) for v in series.values()):
        return {str(k): v for k, v in series.items()}
    return {"": series}


def _grid(series, variables, *, columns, start, zero, title, figsize, xlabel):
    sets = _overlay(series)
    names = list(variables) if variables is not None else list(next(iter(sets.values())))
    fig, axes = _figure(len(names), columns, figsize)
    for ax, v in zip(axes, names):
        for label, paths in sets.items():
            if v in paths:
                y = np.asarray(paths[v], dtype=float)
                ax.plot(np.arange(start, start + y.size), y, label=label or None, lw=1.5)
        if zero:
            ax.axhline(0.0, color="0.6", lw=0.8)
        ax.set_title(v)
        ax.set_xlabel(xlabel)
    if len(sets) > 1:
        axes[0].legend(fontsize="small")
    if title:
        fig.suptitle(title)
    return fig


def plot_irf(responses: Mapping, variables: Sequence[str] | None = None, *, columns: int = 3,
             title: str | None = None, figsize=None):
    """Impulse responses, one panel per variable, with the impact period numbered 1.

    ``responses`` is what :meth:`~dsgelab.FirstOrderSolution.irf` returns, or a
    mapping of labels to several of them (shocks, models, parameter values),
    drawn on the same panels.
    """
    return _grid(responses, variables, columns=columns, start=1, zero=True, title=title,
                 figsize=figsize, xlabel="period")


def plot_paths(paths: Mapping, variables: Sequence[str] | None = None, *, columns: int = 3,
               start: int = 1, title: str | None = None, figsize=None):
    """Paths of variables, one panel each: a simulation, smoothed variables, a transition.

    ``paths`` maps variables to series, or labels to such mappings (a model
    against the data, two scenarios), or is a :class:`~dsgelab.TransitionPath`,
    whose first date is the initial condition (``start=0`` numbers it as
    Dynare's ``rplot`` does). ``start`` numbers the first period.
    """
    series = (paths.values() if hasattr(paths, "levels") and hasattr(paths, "model")
              else paths)
    return _grid(series, variables, columns=columns, start=start, zero=False, title=title,
                 figsize=figsize, xlabel="period")


def plot_decomposition(decomposition: Mapping, variable: str, *, start: int = 1, ax=None,
                       title: str | None = None, groups: Mapping[str, Sequence[str]] | None = None,
                       steady_state: float | None = None, detail: bool = False,
                       realtime: int = 0, vintage: int = 0, frequency: str = "qoq",
                       quarter: int = 1):
    """A shock decomposition as stacked bars, with the total as a line.

    ``decomposition`` is what :func:`~dsgelab.shock_decomposition` (or
    :func:`~dsgelab.realtime_shock_decomposition`) returns. The positive
    contributions are stacked up from zero and the negative ones down, as
    Dynare draws them, with the initial condition as one more contribution.

    The options are those of Dynare's ``plot_shock_decomposition``. ``groups``
    maps a label to the shocks it sums (``use_shock_groups``), and the shocks in
    no group join the initial condition as "Others + Initial Values". With
    ``steady_state`` (``steadystate``) the axis is labelled in levels, the
    deviations plus the steady state, with a dashed line at the steady state.
    ``detail`` (``detail_plot``) draws each contribution in a panel of its own,
    with the total and a common scale.

    ``realtime`` and ``vintage`` choose, in what
    :func:`~dsgelab.realtime_shock_decomposition` returns, the decomposition
    Dynare draws with the same options: 1 the real-time pool, 2 the one-step
    conditional pool, 3 the one-step forecast pool, and with a ``vintage`` (a cut
    in its ``vintages``) the decomposition of that cut, of its ``forecast``-step
    conditional forecast and of its forecast. ``frequency`` is Dynare's
    ``type``: ``"yoy"`` and ``"aoa"`` redraw a quarterly decomposition as
    :func:`~dsgelab.annualise_decomposition` changes it, with ``quarter`` the
    quarter of the first date; ``start`` then numbers the first quarter, or for
    ``"aoa"`` the first year, and ``steady_state`` is multiplied by four for
    ``"yoy"``, as Dynare does.
    """
    decomposition, start, label, steady_state = _choose_decomposition(
        decomposition, start, realtime, vintage, frequency, quarter, steady_state)
    if label and title is None:
        title = f"{variable} {label}"
    pieces = {e: np.asarray(d[variable], dtype=float) for e, d in decomposition["shocks"].items()}
    initial = np.asarray(decomposition["initial"][variable], dtype=float)
    total = np.asarray(decomposition["total"][variable], dtype=float)
    if groups:
        grouped = {str(g): sum((pieces[e] for e in members), np.zeros(total.size))
                   for g, members in groups.items()}
        rest = [e for e in pieces if not any(e in members for members in groups.values())]
        others = sum((pieces[e] for e in rest), np.zeros(total.size))
        pieces = grouped
        if np.any(others):
            pieces["Others + Initial Values"] = others + initial
        else:
            pieces["initial"] = initial
    else:
        pieces["initial"] = initial
    t = np.arange(start, start + total.size)
    if len(pieces) > 10:
        from matplotlib import colormaps
        colors = {name: colormaps["tab20"](i % 20) for i, name in enumerate(pieces)}
    else:
        colors = {name: f"C{i}" for i, name in enumerate(pieces)}

    def stack(axis, items, lw=1.5):
        up, down = np.zeros(total.size), np.zeros(total.size)
        for name, y in items.items():
            pos, neg = np.where(y > 0, y, 0.0), np.where(y < 0, y, 0.0)
            axis.bar(t, pos, bottom=up, width=0.8, label=name, color=colors[name])
            axis.bar(t, neg, bottom=down, width=0.8, color=colors[name])
            up, down = up + pos, down + neg
        axis.plot(t, total, color="black", lw=lw, label="total")
        axis.axhline(0.0, color="0.6", lw=0.8)

    if detail:
        if ax is not None:
            raise ValueError("detail draws a figure of its own: leave ax out")
        fig, axes = _figure(len(pieces), 3, (9, 2.2 * math.ceil(len(pieces) / 3) + 0.4))
        for axis, (name, y) in zip(axes, pieces.items()):
            stack(axis, {name: y}, lw=0.8)
            axis.set_title(name)
        low = min(a.get_ylim()[0] for a in axes)
        high = max(a.get_ylim()[1] for a in axes)
        for axis in axes:
            axis.set_ylim(low, high)
        fig.suptitle(title or variable)
        return fig
    if ax is None:
        fig, (ax,) = _figure(1, 1, (8, 3.5))
    else:
        fig = ax.figure
    stack(ax, pieces)
    if steady_state is not None:
        from matplotlib.ticker import FuncFormatter
        ss = float(steady_state)
        low, high = ax.get_ylim()
        if low < -ss < high:
            ax.axhline(-ss, color="black", ls="--", lw=0.8)
        ax.axhline(0.0, color="0.6", ls="--", lw=0.8)
        # a level that is zero up to rounding is labelled 0, as Dynare does
        ax.yaxis.set_major_formatter(FuncFormatter(
            lambda y, _: f"{0.0 if abs(y + ss) < 1e-12 * max(1.0, abs(ss)) else y + ss:.3g}"))
    ax.set_title(title or variable)
    ax.legend(fontsize="small", ncols=min(len(pieces) + 1, 5))
    return fig


def plot_forecast(forecast: Any, variables: Sequence[str] | None = None, *,
                  history: Mapping[str, ArrayLike] | None = None, columns: int = 3,
                  title: str | None = None, figsize=None):
    """Forecasts with their bands, after the history when it is given.

    ``forecast`` is a :class:`~dsgelab.Forecast` or the dictionary of
    :func:`~dsgelab.posterior_forecast`, with ``mean``, ``lower`` and ``upper``.
    ``history`` maps variables to their past values in the same units; the
    forecast then starts the period after the last of them.
    """
    get = (lambda k: forecast[k]) if isinstance(forecast, Mapping) else (lambda k: getattr(forecast, k))
    mean, lower, upper = get("mean"), get("lower"), get("upper")
    names = list(variables) if variables is not None else list(mean)
    fig, axes = _figure(len(names), columns, figsize)
    for ax, v in zip(axes, names):
        past = np.asarray(history[v], dtype=float) if history is not None and v in history else None
        h0 = 0 if past is None else past.size
        t = np.arange(h0 + 1, h0 + 1 + np.size(mean[v]))
        if past is not None:
            ax.plot(np.arange(1, h0 + 1), past, color="black", lw=1.2)
        ax.fill_between(t, lower[v], upper[v], color="C0", alpha=0.25, lw=0)
        ax.plot(t, mean[v], color="C0", lw=1.5)
        ax.set_title(v)
    if title:
        fig.suptitle(title)
    return fig


def plot_posterior(draws: Any, *, names: Sequence[str] | None = None, priors: Mapping | None = None,
                   problem: Any = None, mode: Any = None, columns: int = 3,
                   title: str | None = None, figsize=None):
    """Posterior densities against the priors, as Dynare draws them after Metropolis.

    ``draws`` is an :class:`~dsgelab.MCMCResult` or a ``(draws, parameters)``
    array with ``names``. Each panel has the posterior density
    (:func:`posterior_density`, black), the prior (:func:`prior_density`, grey)
    when ``priors`` or ``problem`` gives it, and the mode (dashed) when ``mode``
    (a :class:`~dsgelab.ModeResult` or a mapping) does. As in Dynare's
    ``PlotPosteriorDistributions``, the horizontal axis spans both densities and
    the vertical one runs to 1.1 times the higher peak.
    """
    if hasattr(draws, "draws") and hasattr(draws, "names"):
        X, names = np.asarray(draws.draws, dtype=float), list(draws.names)
    else:
        X = np.atleast_2d(np.asarray(draws, dtype=float))
        if names is None:
            raise ValueError("give names= with an array of draws")
        names = list(names)
    if X.shape[1] != len(names):
        raise ValueError(f"{X.shape[1]} columns for {len(names)} names")
    priors = dict(priors or (problem.priors if problem is not None else {}))
    modes = dict(mode) if isinstance(mode, Mapping) else dict(getattr(mode, "values", None) or {})
    fig, axes = _figure(len(names), columns, figsize)
    for ax, (j, n) in zip(axes, enumerate(names)):
        x, f = posterior_density(X[:, j])
        lo, hi, top = x[0], x[-1], np.nanmax(f)
        if n in priors:
            xp, fp = prior_density(priors[n])
            ax.plot(xp, fp, color="0.7", lw=2, label="prior")
            lo, hi, top = min(lo, xp[0]), max(hi, xp[-1]), max(top, np.nanmax(fp))
        ax.plot(x, f, color="black", lw=2, label="posterior")
        if n in modes:
            ax.axvline(float(modes[n]), color="green", ls="--", lw=1.5, label="mode")
        if np.isfinite([lo, hi, top]).all():
            ax.set_xlim(lo, hi)
            ax.set_ylim(0.0, 1.1 * top)
        ax.set_title(n)
    axes[0].legend(fontsize="small")
    if title:
        fig.suptitle(title)
    return fig


def plot_priors(priors: Any, names: Sequence[str] | None = None, *, columns: int = 3,
                title: str | None = "Priors", figsize=None):
    """The prior densities, one panel each, as Dynare's ``plot_priors`` draws them.

    ``priors`` maps names to priors (a :class:`~dsgelab.Prior` or a tuple, as in
    :class:`~dsgelab.EstimationProblem`), or is an EstimationProblem. Each
    panel draws :func:`prior_density` in grey.
    """
    pr = dict(priors.priors if hasattr(priors, "priors") else priors)
    names = list(names) if names is not None else list(pr)
    fig, axes = _figure(len(names), columns, figsize)
    for ax, n in zip(axes, names):
        x, f = prior_density(pr[n])
        ax.plot(x, f, color="0.7", lw=2)
        ax.set_title(n)
    if title:
        fig.suptitle(title)
    return fig


# ---------------------------------------------------------- MCMC diagnostics
def trace_plot(draws: Any, names: Sequence[str] | None = None, *,
               chains: Sequence[int] | None = None, window: int = 200, posterior: bool = False,
               title: str | None = None, figsize=None):
    """Trace plots of the draws, as Dynare's ``trace_plot``.

    ``draws`` is an :class:`~dsgelab.MCMCResult` or an array, ``(draws,)``,
    ``(draws, parameters)`` or ``(chains, draws, parameters)``, with ``names``.
    One panel per parameter (``names`` selects them for an MCMCResult). With one
    chain (the first, or the one in ``chains``) a panel shows the draws in grey
    and their centred moving average over ``2 window + 1`` draws (Dynare's
    ``trace_plot_ma``) in black; with several, the chains are drawn over each
    other. ``posterior`` adds a first panel with the log posterior of an
    MCMCResult, Dynare's ``trace_plot`` of ``'PosteriorDensity'``. Dynare draws
    every draw, burn-in included, which :func:`~dsgelab.metropolis_hastings`
    keeps with ``burn=0``.
    """
    C, all_names, lp = _chains(draws, None if hasattr(draws, "chains") else names)
    use = list(names) if names is not None else all_names
    chains = [0] if chains is None else list(chains)
    series = [(n, C[:, :, all_names.index(n)]) for n in use]
    if posterior:
        if lp is None:
            raise ValueError("the log posterior comes with an MCMCResult")
        series.insert(0, ("log posterior", lp))
    fig, axes = _figure(len(series), 1, figsize or (7, 1.9 * len(series) + 0.4))
    for ax, (n, X) in zip(axes, series):
        t = np.arange(1, X.shape[1] + 1)
        if len(chains) == 1:
            x = X[chains[0]]
            ax.plot(t, x, color="0.7", lw=0.8, label="MCMC draw")
            ax.plot(t, _moving_average(x, window), color="black", lw=2,
                    label=f"{window} period moving average")
        else:
            for c in chains:
                ax.plot(t, X[c], lw=0.8, label=f"Chain {c + 1}")
        ax.set_xlim(t[0], t[-1])
        ax.set_title(n)
    axes[0].legend(fontsize="small", loc="upper right")
    if title:
        fig.suptitle(title)
    return fig


def plot_mh_autocorrelation(draws: Any, names: Sequence[str] | None = None, *, chain: int = 0,
                            lags: int = 30, columns: int = 3, title: str | None = None,
                            figsize=None):
    """Autocorrelograms of the draws of one chain, as Dynare's ``mh_autocorrelation_function``.

    ``draws`` is an :class:`~dsgelab.MCMCResult` (its draws after burn-in, as
    Dynare uses them) or an array, as for :func:`trace_plot`. Each panel has the
    autocorrelations of :func:`mh_autocorrelation` at lags ``0..lags`` as bars.
    """
    C, all_names, _ = _chains(draws, None if hasattr(draws, "chains") else names)
    use = list(names) if names is not None else all_names
    fig, axes = _figure(len(use), columns, figsize)
    for ax, n in zip(axes, use):
        ax.bar(np.arange(lags + 1), mh_autocorrelation(C[chain, :, all_names.index(n)], lags),
               color="black", width=0.8)
        ax.set_xlim(-0.6, lags + 0.6)
        ax.set_title(n)
        ax.set_xlabel("lag")
    if title:
        fig.suptitle(title)
    return fig


def plot_brooks_gelman(diagnostics: Any, names: Sequence[str] | None = None, *,
                       multivariate: bool = False, title: str | None = None, figsize=None):
    """Brooks and Gelman's convergence diagnostics, as Dynare draws them.

    ``diagnostics`` is a :class:`BrooksGelman`, or what :func:`brooks_gelman`
    takes, computed with its defaults. The univariate figure has a row per
    parameter and three columns, the interval, m2 and m3, each with the pooled
    measure in blue and the mean within-chain measure in red. With
    ``multivariate`` it is the figure of the log posterior, a row per measure.
    """
    bg = diagnostics if isinstance(diagnostics, BrooksGelman) else brooks_gelman(diagnostics)
    it = bg.iterations
    if multivariate:
        if bg.multivariate is None:
            raise ValueError("no log posterior: the multivariate diagnostic was not computed")
        rows = [("", bg.multivariate)]
        fig, axes = _figure(3, 1, figsize or (6, 6))
    else:
        use = list(names) if names is not None else bg.names
        rows = [(n + " ", bg.univariate[:, :, bg.names.index(n)]) for n in use]
        fig, axes = _figure(3 * len(rows), 3, figsize or (9, 1.9 * len(rows) + 0.4))
    k = 0
    for label, D in rows:
        for c, measure in enumerate(("Interval", "m2", "m3")):
            ax = axes[k]
            ax.plot(it, D[:, 2 * c], color="blue", lw=1.2, label="pooled")
            ax.plot(it, D[:, 2 * c + 1], color="red", lw=1.2, label="within")
            ax.set_xlim(it[0], it[-1])
            ax.set_title(f"{label}({measure})" if label else measure)
            k += 1
    axes[0].legend(fontsize="small")
    if title:
        fig.suptitle(title)
    return fig


def plot_mode_check(slices: Mapping, names: Sequence[str] | None = None, *, columns: int = 3,
                    title: str | None = None, figsize=None):
    """Slices of the log posterior and the log likelihood through the mode, as Dynare's ``mode_check``.

    ``slices`` is what :func:`mode_check` returns. Each panel has the log
    posterior kernel and the log-likelihood kernel against the parameter, the
    mode as a vertical line, and, at the bottom, red points where the model
    could not be solved.
    """
    use = list(names) if names is not None else list(slices)
    fig, axes = _figure(len(use), columns, figsize)
    for ax, n in zip(axes, use):
        s = slices[n]
        ax.plot(s["x"], s["log_posterior"], color="C0", lw=1.5, label="log-post")
        ax.plot(s["x"], s["log_likelihood"], color="C1", lw=1.5, label="log-lik kernel")
        ax.axvline(s["mode"], color="c", lw=1)
        ax.set_xlim(s["x"][0], s["x"][-1])
        bad = np.isnan(s["log_posterior"])
        if bad.any():
            ax.plot(s["x"][bad], np.full(bad.sum(), ax.get_ylim()[0]), "o", color="red", ms=4)
        ax.set_title(n)
    axes[0].legend(fontsize="small")
    if title:
        fig.suptitle(title)
    return fig


# ------------------------------------------------------------------ smoother
def plot_smoothed_shocks(smoothed: Any, names: Sequence[str] | None = None, *,
                         measurement: bool = False, shock_states: Mapping[str, str] | None = None,
                         start: int = 1, columns: int = 3, title: str | None = None,
                         figsize=None):
    """Smoothed shocks, or measurement errors, as Dynare draws them after the smoother.

    ``smoothed`` is a :class:`~dsgelab.SmootherResult`. Each panel has the
    smoothed innovation of a shock (black) against zero (red), Dynare's
    "Smoothed shocks". A model read from a ``.mod`` carries each shock in a
    state, ``SHK_e``, which holds its value at the first date as well:
    ``shock_states`` (:attr:`dsgelab.ModFile.shock_states`) takes the shocks from
    those states, as Dynare reports them. With ``measurement`` the panels are the
    smoothed measurement errors of the observables, those larger than ``1e-6``
    somewhere (Dynare's ``ME_plot_tol``), its "Smoothed observation errors".
    """
    if measurement:
        obs = smoothed.space.observables
        series = {v: smoothed.measurement[:, i] for i, v in enumerate(obs)
                  if np.max(np.abs(smoothed.measurement[:, i])) > 1e-6}
        if not series:
            raise ValueError("no measurement error larger than 1e-6")
    elif shock_states:
        sol = smoothed.space.solution
        hat = sol.variables(smoothed.states[:, :smoothed.space.n_model_states])
        series = {e: hat[st] * sol.coords.scale[sol.model.position(st)]
                  for st, e in shock_states.items()}
    else:
        series = smoothed.shocks_by_name()
    use = list(names) if names is not None else list(series)
    fig, axes = _figure(len(use), columns, figsize)
    for ax, n in zip(axes, use):
        y = np.asarray(series[n], dtype=float)
        t = np.arange(start, start + y.size)
        ax.plot([t[0], t[-1]], [0.0, 0.0], color="red", lw=0.5)
        ax.plot(t, y, color="black", lw=1)
        ax.set_xlim(t[0], t[-1])
        ax.set_title(n)
    if title:
        fig.suptitle(title)
    return fig


def plot_smoothed_observables(smoothed: Any, data: Any = None, names: Sequence[str] | None = None,
                              *, start: int = 1, columns: int = 3, title: str | None = None,
                              figsize=None):
    """Smoothed observables against the data, Dynare's "Historical and smoothed variables".

    ``smoothed`` is a :class:`~dsgelab.SmootherResult`, and ``data`` the
    observed series, a mapping or a ``(T, n_obs)`` array as for
    :func:`~dsgelab.kalman_filter`. Each panel has the smoothed observable,
    ``d + Z k_t`` without the measurement error, in red and the data dashed in
    black, in the units of the observation equation.
    """
    space = smoothed.space
    fitted = smoothed.states @ space.Z.T + space.d
    Y = None if data is None else space.data_matrix(data)
    obs = list(space.observables)
    use = list(names) if names is not None else obs
    fig, axes = _figure(len(use), columns, figsize)
    for ax, n in zip(axes, use):
        i = obs.index(n)
        t = np.arange(start, start + fitted.shape[0])
        ax.plot(t, fitted[:, i], color="red", lw=1, label="smoothed")
        if Y is not None:
            ax.plot(t, Y[:, i], color="black", ls="--", lw=1, label="data")
        ax.set_xlim(t[0], t[-1])
        ax.set_title(n)
    axes[0].legend(fontsize="small")
    if title:
        fig.suptitle(title)
    return fig


# --------------------------------------------------------- IRFs and forecasts
def plot_posterior_irf(irf: Mapping, shock: str, variables: Sequence[str] | None = None, *,
                       threshold: float = 1e-10, columns: int = 3, title: str | None = None,
                       figsize=None):
    """Posterior impulse responses to one shock with their HPD intervals, as Dynare's ``bayesian_irf``.

    ``irf`` is what :func:`posterior_irf` returns. Each panel has the HPD
    interval in grey and the posterior mean in black. Responses whose mean
    stays below ``threshold`` in absolute value are left out, as with Dynare's
    ``irf_plot_threshold``.
    """
    mean, lower, upper = irf["mean"][shock], irf["lower"][shock], irf["upper"][shock]
    use = [v for v in (variables if variables is not None else mean)
           if np.max(np.abs(mean[v])) >= threshold]
    if not use:
        raise ValueError(f"every response to {shock} is below the threshold {threshold:g}")
    fig, axes = _figure(len(use), columns, figsize)
    for ax, v in zip(axes, use):
        t = np.arange(1, np.size(mean[v]) + 1)
        ax.fill_between(t, lower[v], upper[v], color="0.85", lw=0)
        ax.plot(t, mean[v], color="black", lw=2)
        ax.axhline(0.0, color="0.6", lw=0.8)
        ax.set_xlim(1, t[-1])
        ax.set_title(v)
    fig.suptitle(title if title is not None else f"Orthogonalised shock to {shock}")
    return fig


def plot_conditional_forecast(forecast: Mapping, variables: Sequence[str] | None = None, *,
                              unconditional: Mapping | None = None,
                              conditions: Mapping[str, ArrayLike] | None = None, columns: int = 3,
                              title: str | None = None, figsize=None):
    """A conditional forecast against the unconditional one, as Dynare's ``plot_conditional_forecast``.

    ``forecast`` is what :meth:`~dsgelab.FirstOrderSolution.conditional_forecast`
    returns, dates ``0..H``, and ``unconditional`` the paths without
    conditions over the same dates, in the same units. Each panel has the
    conditional path (solid) and the unconditional one (dashed); ``conditions``,
    the paths imposed over dates ``1..H``, are marked in red.

    ``forecast`` may instead be a :class:`~dsgelab.ConditionalForecast`, from
    :func:`~dsgelab.conditional_forecast_bands` or
    :func:`~dsgelab.posterior_conditional_forecast`. Each panel then is Dynare's:
    the band of the conditional paths in grey, their mean (solid), and the mean
    of the unconditional paths (dashed) between their bounds (thin dashed).
    """
    if hasattr(forecast, "unconditional_mean"):
        return _plot_conditional_bands(forecast, variables, conditions=conditions,
                                       columns=columns, title=title, figsize=figsize)
    names = list(variables) if variables is not None else [k for k in forecast if k != "eps"]
    fig, axes = _figure(len(names), columns, figsize)
    for ax, v in zip(axes, names):
        y = np.asarray(forecast[v], dtype=float)
        t = np.arange(y.size)
        ax.plot(t, y, color="black", lw=2, label="conditional")
        if unconditional is not None and v in unconditional:
            ax.plot(t, np.asarray(unconditional[v], dtype=float), color="black", ls="--", lw=2,
                    label="unconditional")
        if conditions is not None and v in conditions:
            c = np.asarray(conditions[v], dtype=float)
            ax.plot(np.arange(1, c.size + 1), c, "o", color="red", ms=4, label="imposed")
        ax.set_xlim(t[0], t[-1])
        ax.set_title(v)
    axes[0].legend(fontsize="small")
    if title:
        fig.suptitle(title)
    return fig


# -------------------------------------------------------------------- OccBin
def plot_occbin(result: Any, variables: Sequence[str] | None = None, *, deviations: bool = False,
                binding: bool = True, columns: int = 3, title: str | None = None, figsize=None):
    """Piecewise-linear against linear paths, as Dynare's ``occbin_graph``.

    ``result`` is an :class:`~dsgelab.OccbinResult`. Each panel has the
    piecewise-linear path in blue and the linear one dashed in red, in levels,
    or in deviations from the steady state with ``deviations`` (Dynare's
    ``steady_state`` option of ``occbin_graph`` set to false), with the
    vertical axis 1% beyond the data, as Dynare sets it. ``binding`` adds a
    panel per constraint with its indicator, 1 at the dates it binds.
    """
    names = list(variables) if variables is not None else list(result.piecewise)
    constraints = list(result.binding) if binding else []
    fig, axes = _figure(len(names) + len(constraints), columns, figsize)
    for ax, v in zip(axes, names):
        pw = np.asarray(result.piecewise[v], dtype=float)
        lin = np.asarray(result.linear[v], dtype=float)
        if deviations:
            pw, lin = pw - result.steady[v], lin - result.steady[v]
        t = np.arange(1, pw.size + 1)
        ax.plot([1, pw.size], [0.0, 0.0], color="black", ls="--", lw=0.5)
        ax.plot(t, pw, color="blue", lw=2, label="Piecewise Linear")
        ax.plot(t, lin, color="red", ls="--", lw=2, label="Linear")
        low, high = min(pw.min(), lin.min()), max(pw.max(), lin.max())
        low, high = low - 0.01 * abs(low), high + 0.01 * abs(high)
        ax.set_ylim(low, high if high > low else low + 1)
        ax.set_xlim(1, pw.size)
        ax.set_title(v)
    for ax, c in zip(axes[len(names):], constraints):
        b = np.asarray(result.binding[c], dtype=float)
        ax.step(np.arange(1, b.size + 1), b, where="mid", color="black", lw=1.5)
        ax.set_ylim(-0.1, 1.1)
        ax.set_yticks([0, 1])
        ax.set_xlim(1, b.size)
        ax.set_title(f"{c} binds")
    if names:
        axes[0].legend(fontsize="small", frameon=False)
    if title:
        fig.suptitle(title)
    return fig


def plot_occbin_regimes(result: Any, *, title: str | None = None, figsize=None):
    """The regimes agents expect at each surprise, as Dynare's ``occbin.plot_regimes``.

    ``result`` is an :class:`~dsgelab.OccbinResult`. One panel per constraint:
    for each date at which a surprise arrives, the dates, counted from it, at
    which each regime of the plan made then starts, red stars for a binding
    regime and blue circles for a slack one, joined to the start of the next.
    """
    from matplotlib.lines import Line2D
    constraints = list(result.binding)
    fig, axes = _figure(len(constraints), 1, figsize or (7, 2.6 * len(constraints) + 0.4))
    periods = len(next(iter(result.binding.values())))
    for ax, c in zip(axes, constraints):
        for date, plan in result.regimes:
            arr = np.asarray(plan[c], dtype=bool)
            starts = [0] + [i for i in range(1, arr.size) if arr[i] != arr[i - 1]]
            for s, e in zip(starts, starts[1:] + [starts[-1]]):
                color, marker = ("red", "*") if arr[s] else ("blue", "o")
                ax.plot([date, date], [s + 1, e + 1], color=color, lw=1)
                ax.plot(date, s + 1, marker, color=color)
        ax.set_xlim(0.5, periods + 0.5)
        ax.set_title(c)
        ax.set_xlabel("Historic period")
        ax.set_ylabel("Regime: expected start")
    axes[0].legend([Line2D([], [], color="blue", marker="o", lw=1),
                    Line2D([], [], color="red", marker="*", lw=1)], ["Slack", "Binding"],
                   fontsize="small")
    if title:
        fig.suptitle(title)
    return fig


# ------------------------------------------------- identification, sensitivity
def plot_identification(ident: Any, groups: Sequence[int] | None = None, *,
                        title: str | None = None, figsize=None):
    """Collinearity patterns, as Dynare's ``identification`` draws them.

    ``ident`` is an :class:`~dsgelab.Identification`. One panel per group size
    ``k`` in :attr:`~dsgelab.Identification.collinearity` (or in ``groups``): row
    ``i`` colours, in the columns of the ``k`` parameters that best reproduce
    the moment derivatives of parameter ``i``, the cosine of that fit, from 0
    to 1, and leaves the other cells grey.
    """
    from matplotlib import colormaps
    ks = sorted(ident.collinearity) if groups is None else list(groups)
    if not ks:
        raise ValueError("no collinearity search in this Identification")
    names = list(ident.names)
    p = len(names)
    side = max(3.0, 0.35 * p + 1.5)
    fig, axes = _figure(len(ks), len(ks), figsize or (side * len(ks) + 0.8, side))
    cmap = colormaps["jet"].with_extremes(bad="0.9")
    for ax, k in zip(axes, ks):
        pax = np.full((p, p), np.nan)
        for i, n in enumerate(names):
            group, cosine = ident.collinearity[k][n]
            for member in group:
                pax[i, names.index(member)] = cosine
        im = ax.imshow(np.ma.masked_invalid(pax), cmap=cmap, vmin=0.0, vmax=1.0)
        ax.set_xticks(range(p), names, rotation=90)
        ax.set_yticks(range(p), names)
        ax.set_title(f"Collinearity patterns with {k} parameter{'s' if k > 1 else ''}")
        fig.colorbar(im, ax=ax, shrink=0.8)
    if title:
        fig.suptitle(title)
    return fig


def _ecdf(ax, x, **style):
    """The empirical distribution function of ``x`` as a staircase, Dynare's ``gsa.cumplot``."""
    xs = np.sort(np.asarray(x, dtype=float))
    ax.step(np.r_[xs[0], xs], np.arange(xs.size + 1) / xs.size, where="post", **style)


def plot_filtering(result: Any, names: Sequence[str] | None = None, *, kind: str = "determinate",
                   scatter: bool = True, title: str | None = None, figsize=None):
    """Monte Carlo filtering, as Dynare's sensitivity toolbox draws it.

    ``result`` is a :class:`~dsgelab.FilteringResult`, or a
    :class:`~dsgelab.StabilityMap` and the ``kind`` of its
    :meth:`~dsgelab.StabilityMap.filtering`. The parameters drawn are those
    whose Smirnov test rejects at ``pvalue_ks``, the most significant first, as
    Dynare chooses them, or ``names``. With ``scatter`` the figure is Dynare's
    ``scatter_mcf``: on the diagonal, the cumulative distributions of each
    parameter in the draws with the behaviour (blue) and without (red); off it,
    the draws of each pair of parameters. Without, it is
    ``stability_mapping_univariate``: one panel per parameter with the two
    distributions and the p-value in the title.
    """
    fr = result.filtering(kind) if hasattr(result, "filtering") else result
    use = list(names) if names is not None else (fr.significant or list(fr.names))
    idx = [fr.names.index(n) for n in use]
    X, Y = fr.points[fr.behaviour], fr.points[fr.nonbehaviour]
    if scatter:
        p = len(use)
        fig, axes = _figure(p * p, p, figsize or (max(4.0, 1.8 * p), max(4.0, 1.8 * p)))
        for row, (nj, j) in enumerate(zip(use, idx)):
            for col, (ni, i) in enumerate(zip(use, idx)):
                ax = axes[row * p + col]
                if i == j:
                    _ecdf(ax, X[:, j], color="blue", lw=1.5)
                    _ecdf(ax, Y[:, j], color="red", lw=1.5)
                    ax.set_yticks([])
                else:
                    ax.plot(X[:, i], X[:, j], ".", color="blue", ms=2)
                    ax.plot(Y[:, i], Y[:, j], ".", color="red", ms=2)
                if row == 0:
                    ax.set_title(ni, fontsize="small")
                if col == 0:
                    ax.set_ylabel(nj, fontsize="small")
                if row < p - 1:
                    ax.set_xticks([])
        fig.suptitle(title if title is not None else
                     f"Monte Carlo filtering: {fr.title} (blue) or not (red)")
        return fig
    fig, axes = _figure(len(use), 4, figsize)
    for ax, n, j in zip(axes, use, idx):
        if X.size:
            _ecdf(ax, X[:, j], color="blue", ls=":", lw=1.5, label="behaviour")
        if Y.size:
            _ecdf(ax, Y[:, j], color="black", lw=1.5, label="non-behaviour")
        ax.grid(True)
        ax.set_title(f"{n}. p-value {fr.p[j]:.2g}")
    axes[0].legend(fontsize="small")
    if title:
        fig.suptitle(title)
    return fig



# ------------------------------------- real-time and annualised decompositions
def _choose_decomposition(decomposition, start, realtime, vintage, frequency, quarter,
                          steady_state):
    """The decomposition Dynare's ``plot_shock_decomposition`` draws, its first date and label."""
    label = ""
    if realtime not in (0, 1, 2, 3):
        raise ValueError("realtime must be 0, 1, 2 or 3")
    if vintage and not realtime:
        raise ValueError("a vintage needs realtime 1, 2 or 3")
    if realtime:
        h = 0
        if vintage:
            key = {1: "vintages", 2: "conditional_vintages", 3: "forecast_vintages"}[realtime]
            if key not in decomposition or vintage not in decomposition[key]:
                raise ValueError(f"vintage {vintage} is not in the {key!r} of this decomposition: "
                                 f"pass it in realtime_shock_decomposition's vintages")
            chosen = decomposition[key][vintage]
            if realtime > 1:
                h = len(next(iter(chosen["total"].values()))) - 1
                start = start + vintage - 1
            label = {1: f"realtime (vintage {vintage})",
                     2: f"{h}-step ahead conditional forecast (given {vintage})",
                     3: f"{h}-step ahead forecast (given {vintage})"}[realtime]
        else:
            key = {1: None, 2: "conditional", 3: "forecast"}[realtime]
            if key is not None and key not in decomposition:
                raise ValueError(f"no {key!r} pool in this decomposition"
                                 + (": pass forecast= to realtime_shock_decomposition"
                                    if key == "forecast" else ""))
            chosen = decomposition if key is None else decomposition[key]
            label = {1: "realtime (rolling)", 2: "1-step ahead conditional forecast (rolling)",
                     3: "1-step ahead forecast (rolling)"}[realtime]
        decomposition = {k: chosen[k] for k in ("shocks", "initial", "total")}
    if frequency != "qoq":
        from .kalman import annualise_decomposition
        decomposition = annualise_decomposition(
            {k: decomposition[k] for k in ("shocks", "initial", "total")}, frequency,
            quarter=quarter)
        if frequency == "yoy":
            start = start + 3
            steady_state = None if steady_state is None else 4 * steady_state
        label = f"{label} {frequency}".strip()
    return decomposition, start, label, steady_state


def _plot_conditional_bands(forecast, variables, *, conditions, columns, title, figsize):
    """Dynare's figure of a conditional forecast with bands, one panel per variable."""
    names = list(variables) if variables is not None else list(forecast.mean)
    fig, axes = _figure(len(names), columns, figsize)
    for ax, v in zip(axes, names):
        m = np.asarray(forecast.mean[v], dtype=float)
        t = np.arange(m.size)
        ax.fill_between(t, forecast.lower[v], forecast.upper[v], color="0.85", lw=0,
                        label=f"{forecast.coverage:.0%} band")
        ax.plot(t, m, color="black", lw=2, label="conditional")
        ax.plot(t, forecast.unconditional_mean[v], color="black", ls="--", lw=2,
                label="unconditional")
        ax.plot(t, forecast.unconditional_lower[v], color="black", ls="--", lw=0.8)
        ax.plot(t, forecast.unconditional_upper[v], color="black", ls="--", lw=0.8)
        if conditions is not None and v in conditions:
            c = np.asarray(conditions[v], dtype=float)
            ax.plot(np.arange(1, c.size + 1), c, "o", color="red", ms=4, label="imposed")
        ax.set_xlim(t[0], t[-1])
        ax.set_title(v)
    axes[0].legend(fontsize="small")
    if title:
        fig.suptitle(title)
    return fig


# ---------------------------------------------------- identification strength
def _strength_of(ident: Any):
    s = getattr(ident, "strength", None)
    if s is None:
        raise ValueError("no identification strength: call identification(..., strength=True)")
    return s


def _log_bars(ax, order, columns, labels):
    """Dynare's bars of ``log`` values, grouped by parameter, with markers for infinite ones.

    A dot marks each finite value, as Dynare draws them over the bars; an infinite
    logarithm is marked at zero, red for a value of zero and white for an infinite
    value.
    """
    n = len(order)
    x = np.arange(1, n + 1)
    width = 0.8 / len(columns)
    for k, (values, label) in enumerate(zip(columns, labels)):
        with np.errstate(divide="ignore", invalid="ignore"):
            y = np.log(np.asarray(values, dtype=float)[order])
        shift = (k - (len(columns) - 1) / 2) * width
        finite = np.isfinite(y)
        ax.bar(x[finite] + shift, y[finite], width=width, color=f"C{k}", label=label)
        ax.plot(x[finite] + shift, y[finite], "o", color="black", ms=4)
        for flag, face in ((np.isneginf(y), "red"), (np.isposinf(y), "white")):
            if flag.any():
                ax.plot(x[flag] + shift, np.zeros(flag.sum()), "o", ms=6, mfc=face, mec="black")
    ax.axhline(0.0, color="0.6", lw=0.8)
    ax.set_xlim(0, n + 1)


def plot_identification_strength(ident: Any, *, prior: bool = True, title: str | None = None,
                                 figsize=None):
    """Identification strength and its sensitivity component, as Dynare's ``identification``.

    ``ident`` is an :class:`~dsgelab.Identification` computed with
    ``strength=True``. The upper panel has the logarithm of the strength of each
    parameter (:attr:`~dsgelab.IdentificationStrength.strength`, its value over its
    asymptotic standard deviation) and the lower one that of its sensitivity
    component (``sensitivity``), the parameters sorted by strength, from the
    weakest. With ``prior`` and the priors given, each parameter has a second bar
    relative to the prior standard deviation. An infinite logarithm is marked at
    zero, red for a strength of zero and white for an infinite one.
    """
    s = _strength_of(ident)
    names = list(ident.names)
    order = np.argsort(s.strength, kind="stable")
    with_prior = prior and not np.all(np.isnan(s.strength_prior))
    labels = ["relative to param value"] + (["relative to prior std"] if with_prior else [])
    fig, axes = _figure(2, 1, figsize or (max(6.0, 0.6 * len(names) + 2), 7))
    panels = ((s.strength, s.strength_prior, "Identification strength"),
              (s.sensitivity, s.sensitivity_prior, "Sensitivity component"))
    for ax, (own, rel, what) in zip(axes, panels):
        _log_bars(ax, order, [own, rel][:len(labels)], labels)
        ax.set_xticks(np.arange(1, len(names) + 1), [names[i] for i in order], rotation=90)
        ax.set_title(f"{what} with asymptotic Information matrix (log-scale)")
        ax.legend(fontsize="small")
    if title:
        fig.suptitle(title)
    return fig


def plot_identification_sensitivity(ident: Any, *, title: str | None = None, figsize=None):
    """The sensitivity of the moments to each parameter, as Dynare's sensitivity bars.

    ``ident`` is an :class:`~dsgelab.Identification` computed with
    ``strength=True`` and priors. The bars are the logarithm of 100 times
    :attr:`~dsgelab.IdentificationStrength.moments_sensitivity` over its largest
    value, the parameters sorted by strength as in
    :func:`plot_identification_strength`: Dynare's "Moments" bars, normalised by
    the largest finite value, so that a parameter whose prior has an infinite
    standard deviation is left out instead of leaving every other bar at minus
    infinity. Dynare's "Model" and "LRE model" bars, from the derivatives of its
    own reduced form and of its dynamic model's Jacobian, are not drawn.
    """
    s = _strength_of(ident)
    si = np.asarray(s.moments_sensitivity, dtype=float)
    finite = np.isfinite(si)
    if not finite.any():
        raise ValueError("no finite sensitivity: give the priors, whose standard deviations "
                         "scale it")
    names = list(ident.names)
    order = np.argsort(s.strength, kind="stable")
    rel = np.where(finite, si / np.max(si[finite]) * 100, np.nan)
    fig, (ax,) = _figure(1, 1, figsize or (max(6.0, 0.6 * len(names) + 2), 4))
    with np.errstate(divide="ignore"):
        y = np.log(rel[order])
    x = np.arange(1, len(names) + 1)
    ax.bar(x[np.isfinite(y)], y[np.isfinite(y)], width=0.8, color="C0", label="Moments")
    ax.axhline(0.0, color="0.6", lw=0.8)
    ax.set_xlim(0, len(names) + 1)
    ax.set_xticks(x, [names[i] for i in order], rotation=90)
    ax.set_title("Sensitivity bars using derivatives (log-scale)")
    ax.legend(fontsize="small")
    if title:
        fig.suptitle(title)
    return fig


def plot_identification_patterns(ident: Any, *, largest: bool = False, count: int = 4,
                                 title: str | None = None, figsize=None):
    """Identification patterns, as Dynare's ``identification`` with ``advanced`` draws them.

    ``ident`` is an :class:`~dsgelab.Identification` computed with
    ``strength=True``. Each panel has the absolute values of the entries of a
    right singular vector of the information matrix
    (:attr:`~dsgelab.IdentificationStrength.patterns`), one bar per parameter, in
    the order of :attr:`~dsgelab.Identification.names`. The ``count`` smallest
    singular values come first, from the smallest, the direction of the
    parameters that the data identify worst; with ``largest`` the ``count``
    largest, from the largest.
    """
    s = _strength_of(ident)
    names = list(ident.names)
    k = min(count, len(names))
    cols = list(range(k)) if largest else [len(names) - 1 - j for j in range(k)]
    fig, axes = _figure(k, 1, figsize or (max(6.0, 0.6 * len(names) + 2), 1.6 * k + 1.4))
    x = np.arange(1, len(names) + 1)
    for j, (ax, c) in enumerate(zip(axes, cols)):
        ax.bar(x, np.abs(s.patterns[:, c]), width=0.8, color="C0")
        ax.set_xlim(0, len(names) + 1)
        ax.set_title(f"Singular value {s.singular_values[c]:.5g}")
        if j == k - 1:
            ax.set_xticks(x, names, rotation=90)
        else:
            ax.set_xticks(x, [])
    which = "HIGHEST" if largest else "SMALLEST"
    fig.suptitle(title if title is not None else
                 f"Identification patterns (Information matrix): {which} SV")
    return fig
