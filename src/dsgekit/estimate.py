"""Bayesian estimation of a first-order solution.

The state space is the linear solution already computed by :func:`dsgekit.solve`::

    k_t = P k_{t-1} + Q eps_t,     y_t = Z k_t + e_t,

with ``eps ~ N(0, Σ)`` and ``e ~ N(0, R)``. ``Z`` reads the observed variables off
``[k; F k]`` in the rescaled deviations, after the data in levels have been centred
on the steady state of the current parameters. The likelihood is the Kalman filter.
A draw that violates Blanchard–Kahn contributes ``-inf`` and is refused.

The posterior mode is a numerical maximum. The posterior draws are a random-walk
Metropolis–Hastings started at that mode. This is the linear-Gaussian problem.
It is not a particle filter, and it does not estimate a nonlinear or pruned law
of motion.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.stats import invgamma, norm

from .model import Model, set_params
from .solve import solve

__all__ = ["EstimationResult", "log_likelihood", "log_posterior", "estimate"]


@dataclass
class EstimationResult:
    """Posterior mode and, when requested, Metropolis–Hastings draws."""

    names: list
    mode: dict
    log_posterior: float
    draws: dict | None
    acceptance: float | None
    success: bool
    message: str


def _prior_logpdf(spec, value: float) -> float:
    """A callable, or ``("normal", mean, sd)`` / ``("invgamma", a, scale)``."""
    if callable(spec):
        return float(spec(value))
    kind, a, b = spec
    if kind == "normal":
        return float(norm.logpdf(value, loc=a, scale=b))
    if kind == "invgamma":
        return float(invgamma.logpdf(value, a, scale=b))
    raise ValueError(f"prior {kind!r} is not 'normal' or 'invgamma'")


def _observation_matrix(sol, names: list[str]) -> np.ndarray:
    """Rows of ``Z`` such that the hat-deviation of each named variable is ``Z k``."""
    model = sol.model
    nk = model.n_states
    Z = np.zeros((len(names), nk))
    for i, name in enumerate(names):
        if name in model.states:
            Z[i, model.states.index(name)] = 1.0
        elif name in model.jumps:
            Z[i] = sol.F[model.jumps.index(name)]
        else:
            raise KeyError(f"{name!r} is not a state or a jump of {model.name!r}")
    return Z


def _measurement(R, n_obs: int) -> np.ndarray:
    if np.isscalar(R):
        return float(R) * np.eye(n_obs)
    arr = np.asarray(R, dtype=float)
    if arr.ndim == 1:
        if arr.size != n_obs:
            raise ValueError(f"measurement variances: expected {n_obs}, got {arr.size}")
        return np.diag(arr)
    if arr.shape != (n_obs, n_obs):
        raise ValueError(f"measurement matrix must be {n_obs}x{n_obs}, got {arr.shape}")
    return arr


def _data_matrix(data, names, sol) -> np.ndarray:
    """Levels in, hat-deviations out, one row per date."""
    cols = []
    index = {n: i for i, n in enumerate(sol.model.names)}
    center = sol.steady.x
    scale = sol.coords.scale
    T = None
    for name in names:
        series = np.asarray(data[name], dtype=float).ravel()
        if T is None:
            T = series.size
        elif series.size != T:
            raise ValueError(f"{name!r} has length {series.size}, expected {T}")
        j = index[name]
        cols.append((series - center[j]) / scale[j])
    return np.column_stack(cols)


def log_likelihood(sol, Y: np.ndarray, Z: np.ndarray, R: np.ndarray, Sigma: np.ndarray) -> float:
    """Gaussian log likelihood of ``Y`` under the linear state space of ``sol``.

    ``Y`` has shape ``(T, n_obs)`` and is already in hat-deviations. The filter is
    initialised at the unconditional distribution of the states.
    """
    from .solution import shock_covariance
    P = sol.P
    Q = sol.Q if sol.Q is not None else np.zeros((P.shape[0], 0))
    if not isinstance(Sigma, np.ndarray):
        Sigma = shock_covariance(sol.model.shocks, Sigma)
    nk = P.shape[0]
    V = _stationary_variance(P, Q, Sigma)
    mean = np.zeros(nk)
    loglik = 0.0
    n_obs = Y.shape[1]
    eye = np.eye(nk)
    for t in range(Y.shape[0]):
        innov = Y[t] - Z @ mean
        S = Z @ V @ Z.T + R
        sign, logdet = np.linalg.slogdet(S)
        if sign <= 0 or not np.isfinite(logdet):
            return -np.inf
        solS = np.linalg.solve(S, innov)
        loglik += -0.5 * (logdet + innov @ solS + n_obs * np.log(2.0 * np.pi))
        K = np.linalg.solve(S, Z @ V).T
        mean = P @ (mean + K @ innov)
        V = P @ (eye - K @ Z) @ V @ P.T + Q @ Sigma @ Q.T
        V = 0.5 * (V + V.T)
    return float(loglik)


def _stationary_variance(P, Q, Sigma) -> np.ndarray:
    from scipy.linalg import solve_discrete_lyapunov
    shock = Q @ Sigma @ Q.T
    shock = 0.5 * (shock + shock.T)
    try:
        V = solve_discrete_lyapunov(P, shock)
    except Exception:
        return np.full(P.shape, np.nan)
    return 0.5 * (V + V.T)


def log_posterior(model: Model, data, *, params, free: dict, priors: dict, guess,
                  observables: list[str], measurement, cov=None,
                  positive=()) -> float:
    """Log posterior at ``free``. ``-inf`` when the model has no saddle path."""
    return _posterior(model, data, params, free, priors, guess, observables,
                      measurement, cov, positive)


def estimate(model: Model, data, *, params, free: dict, priors: dict, guess,
             observables: list[str], measurement, cov=None, positive=(),
             bounds: dict | None = None, draws: int = 0, seed: int = 0,
             proposal: dict | None = None) -> EstimationResult:
    """Maximise the posterior and, if ``draws > 0``, run Metropolis–Hastings.

    Parameters
    ----------
    data :
        Mapping of variable name to a one-dimensional series in **levels**, the
        same units as the model's steady state. Every series has the same length.
    free :
        Parameter name to starting value. These are the only parameters that move.
    priors :
        Parameter name to a ``logpdf(value)`` callable, or to
        ``("normal", mean, sd)`` or ``("invgamma", shape, scale)``.
    observables :
        Names of the variables the series correspond to, in the model's list.
    measurement :
        Variance of the observation error, a scalar for every series, a vector,
        or a full covariance. A pure state-space with no measurement error is
        singular; a small variance is a modelling choice and has to be passed in.
    draws :
        Length of the Metropolis–Hastings chain after the mode. Zero skips it.
    proposal :
        Standard deviation of the random-walk proposal for each free parameter.
        Absent, one tenth of the normal prior scale, or one tenth of the
        starting value when the prior is not normal.
    """
    names = list(free)
    if any(n not in priors for n in names):
        missing = [n for n in names if n not in priors]
        raise KeyError(f"no prior for {missing}")
    x0 = np.array([float(free[n]) for n in names])
    lo = np.full(len(names), -np.inf)
    hi = np.full(len(names), np.inf)
    if bounds:
        for i, n in enumerate(names):
            if n in bounds:
                lo[i], hi[i] = bounds[n]

    def posterior_at(x):
        theta = {n: float(v) for n, v in zip(names, x)}
        return _posterior(model, data, params, theta, priors, guess, observables,
                          measurement, cov, positive)

    def objective(x):
        val = posterior_at(x)
        return 1e12 if not np.isfinite(val) else -val

    opt = minimize(objective, x0, method="L-BFGS-B", bounds=list(zip(lo, hi)))
    mode_x = opt.x
    mode_lp = posterior_at(mode_x)
    mode = {n: float(v) for n, v in zip(names, mode_x)}
    chain = None
    acceptance = None
    if draws:
        chain, acceptance = _metropolis(
            posterior_at, mode_x, lo, hi, names, priors, proposal, draws, seed)
    return EstimationResult(
        names=names, mode=mode, log_posterior=float(mode_lp), draws=chain,
        acceptance=acceptance, success=bool(np.isfinite(mode_lp) and mode_lp > -1e11),
        message=str(opt.message))


def _posterior(model, data, params, theta, priors, guess, observables, measurement, cov, positive):
    from .klein import BlanchardKahnError
    lp = 0.0
    for name, value in theta.items():
        lp += _prior_logpdf(priors[name], float(value))
    if not np.isfinite(lp):
        return -np.inf
    try:
        sol = solve(model, set_params(params, theta), guess=guess, positive=positive)
    except (BlanchardKahnError, RuntimeError, ValueError, np.linalg.LinAlgError):
        return -np.inf
    from .solution import shock_covariance
    Sigma = shock_covariance(model.shocks, cov)
    try:
        Z = _observation_matrix(sol, observables)
        R = _measurement(measurement, len(observables))
        Y = _data_matrix(data, observables, sol)
        ll = log_likelihood(sol, Y, Z, R, Sigma)
    except (np.linalg.LinAlgError, ValueError):
        return -np.inf
    if not np.isfinite(ll):
        return -np.inf
    return float(lp + ll)


def _metropolis(posterior_at, mode, lo, hi, names, priors, proposal, draws, seed):
    rng = np.random.default_rng(seed)
    scale = np.zeros(len(names))
    for i, n in enumerate(names):
        if proposal and n in proposal:
            scale[i] = float(proposal[n])
        else:
            spec = priors[n]
            if isinstance(spec, tuple) and spec[0] == "normal":
                scale[i] = 0.1 * float(spec[2])
            else:
                scale[i] = 0.1 * max(abs(mode[i]), 1e-3)
    current = mode.copy()
    current_lp = posterior_at(current)
    kept = np.empty((draws, len(names)))
    accepted = 0
    for d in range(draws):
        prop = current + rng.normal(0.0, scale)
        if np.any(prop < lo) or np.any(prop > hi):
            kept[d] = current
            continue
        prop_lp = posterior_at(prop)
        if np.log(rng.random()) < prop_lp - current_lp:
            current, current_lp = prop, prop_lp
            accepted += 1
        kept[d] = current
    return {n: kept[:, i] for i, n in enumerate(names)}, accepted / draws
