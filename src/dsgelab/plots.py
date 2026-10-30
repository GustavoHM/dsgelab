"""Plots of impulse responses, paths, decompositions, forecasts and posterior distributions.

Each function takes what the package already returns -- the dictionaries of
:meth:`~dsgelab.FirstOrderSolution.irf` and :meth:`~dsgelab.FirstOrderSolution.simulate`,
:func:`~dsgelab.shock_decomposition`, a :class:`~dsgelab.Forecast`, an
:class:`~dsgelab.MCMCResult` -- and returns a ``matplotlib`` figure, built
without ``pyplot`` so that nothing global changes: save it with
``fig.savefig(...)``, or let a notebook show it. matplotlib is an optional
dependency (``pip install dsgelab[plot]``).

The two densities a posterior plot draws are computed as Dynare computes them,
and are available without matplotlib: :func:`prior_density` is
``draw_prior_density`` (200 points between the quantiles 0.001 and 0.999), and
:func:`posterior_density` is the kernel estimate of ``posterior_moments``
(Gaussian kernel, Silverman's bandwidth corrected for repeated Metropolis
draws, 512 points).
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike

__all__ = ["prior_density", "posterior_density", "plot_irf", "plot_paths",
           "plot_decomposition", "plot_forecast", "plot_posterior"]


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
    against the data, two scenarios). ``start`` numbers the first period.
    """
    return _grid(paths, variables, columns=columns, start=start, zero=False, title=title,
                 figsize=figsize, xlabel="period")


def plot_decomposition(decomposition: Mapping, variable: str, *, start: int = 1, ax=None,
                       title: str | None = None):
    """A shock decomposition as stacked bars, with the total as a line.

    ``decomposition`` is what :func:`~dsgelab.shock_decomposition` (or
    :func:`~dsgelab.realtime_shock_decomposition`) returns. The positive
    contributions are stacked up from zero and the negative ones down, as
    Dynare draws them, with the initial condition as one more contribution.
    """
    pieces = {e: np.asarray(d[variable], dtype=float) for e, d in decomposition["shocks"].items()}
    pieces["initial"] = np.asarray(decomposition["initial"][variable], dtype=float)
    total = np.asarray(decomposition["total"][variable], dtype=float)
    if ax is None:
        fig, (ax,) = _figure(1, 1, (8, 3.5))
    else:
        fig = ax.figure
    t = np.arange(start, start + total.size)
    up, down = np.zeros(total.size), np.zeros(total.size)
    for name, y in pieces.items():
        pos, neg = np.where(y > 0, y, 0.0), np.where(y < 0, y, 0.0)
        bars = ax.bar(t, pos, bottom=up, width=0.8, label=name)
        ax.bar(t, neg, bottom=down, width=0.8, color=bars.patches[0].get_facecolor())
        up, down = up + pos, down + neg
    ax.plot(t, total, color="black", lw=1.5, label="total")
    ax.axhline(0.0, color="0.6", lw=0.8)
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
    (a :class:`~dsgelab.ModeResult` or a mapping) does.
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
        if n in priors:
            xp, fp = prior_density(priors[n])
            ax.plot(xp, fp, color="0.6", lw=1.5, label="prior")
        x, f = posterior_density(X[:, j])
        ax.plot(x, f, color="black", lw=1.5, label="posterior")
        if n in modes:
            ax.axvline(float(modes[n]), color="C2", ls="--", lw=1.2, label="mode")
        ax.set_title(n)
    axes[0].legend(fontsize="small")
    if title:
        fig.suptitle(title)
    return fig
