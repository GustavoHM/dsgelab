"""Estimation of the linear state space: likelihood, posterior, samplers, model comparison.

An :class:`EstimationProblem` ties a model to data: which parameters move and
with which prior, which variables are observed (in levels or in logs) and with
how much measurement error, how the steady state is found at each parameter
vector. Everything else -- the posterior mode, maximum likelihood, the
Metropolis-Hastings and SMC samplers, the marginal likelihood -- is a function
of that one object, so the likelihood the samplers see is the one the mode was
found on.

A parameter vector at which the model has no determinate solution, or no steady
state, has posterior ``-inf`` and is rejected; the prior predictive check
reports how often that happens before any data is seen.

The estimated quantities are named as in Dynare's ``estimated_params``: a
parameter by its name, the standard deviation of a shock as ``"stderr e"``, the
standard deviation of a measurement error on an observed variable as
``"stderr y"``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike

from .kalman import (Data, FilterResult, SmootherResult, StateSpace, kalman_filter,
                     kalman_smoother, state_space)
from .model import Covariance, Model, Params, Point, set_params
from .priors import Prior, prior as make_prior

if TYPE_CHECKING:
    from .inversion import InversionFilterResult
    from .particle import ParticleFilterResult, ParticleSmootherResult
    from .solution import FirstOrderSolution
    from .steady import SteadyState

__all__ = ["EstimationProblem", "ModeResult", "MCMCResult", "SMCResult",
           "posterior_mode", "maximum_likelihood", "metropolis_hastings", "smc",
           "prior_predictive", "model_probabilities", "rhat", "effective_sample_size",
           "posterior_forecast", "hessian"]


def unpack_point(model: Model, params: Params, cov: Covariance,
                 measurement: Mapping[str, float], names: Sequence[str],
                 theta: ArrayLike) -> tuple[Params, np.ndarray, dict[str, float]]:
    """``(params, shock covariance, measurement sds)`` with ``names`` set to ``theta``.

    A name is a parameter, ``"stderr e"`` for a shock, ``"stderr y"`` for the
    measurement error on an observable, or ``"corr e1, e2"`` for the correlation of
    two shocks. A shock's new standard deviation rescales its row and column of
    ``cov``, so its correlations with the other shocks stay; a correlation is set
    after the standard deviations.
    """
    from .solution import shock_covariance
    theta = np.asarray(theta, dtype=float)
    updates, sds, corrs = {}, {}, {}
    meas = dict(measurement)
    for n, v in zip(names, theta):
        if n.startswith("corr "):
            pair = tuple(x.strip() for x in n[5:].split(","))
            if len(pair) != 2 or not all(e in model.shocks for e in pair):
                raise KeyError(f"{n!r}: name two shocks, as 'corr e1, e2'")
            corrs[pair] = float(v)
        elif n.startswith("stderr "):
            target = n[7:]
            if target in model.shocks:
                sds[target] = float(v)
            else:
                meas[target] = float(v)
        else:
            updates[n] = float(v)
    p = set_params(params, updates)
    if isinstance(p, dict):
        p = dict(p)
    S = shock_covariance(model.shocks, cov).copy()
    for e, v in sds.items():
        i = model.shocks.index(e)
        old = math.sqrt(S[i, i]) if S[i, i] > 0 else 0.0
        if old > 0:
            S[i, :] *= v / old
            S[:, i] *= v / old
        else:
            S[i, i] = v * v
    for (a, b), rho in corrs.items():
        i, j = model.shocks.index(a), model.shocks.index(b)
        S[i, j] = S[j, i] = rho * math.sqrt(S[i, i] * S[j, j])
    return p, S, meas


@dataclass
class EstimationProblem:
    """A model, data, and what is estimated.

    Parameters
    ----------
    model : Model
    data : mapping of series, or ``(T, n_obs)`` array in the order of ``observables``
    params : the parameters that do not move (and starting values for those that do)
    observables : the observed model variables
    priors : ``{name: Prior}`` -- see :func:`dsgelab.priors.prior` -- or a tuple
        ``(shape, mean, sd[, p3, p4])``. ``name`` is a parameter, ``"stderr e"``
        for a shock, or ``"stderr y"`` for the measurement error on an observable.
        For maximum likelihood without priors pass ``bounds`` instead and leave
        ``priors`` empty: the free names are then ``free``.
    cov : standard deviations (mapping) or covariance of the shocks held fixed.
    measurement : standard deviations of the measurement errors held fixed.
    transforms : ``"level"`` or ``"log"`` per observable (see :func:`dsgelab.state_space`).
    steady_fn : ``steady_fn(params) -> SteadyState``. By default
        :func:`dsgelab.steady_state` from ``guess``, warm-started from the
        previous trial.
    init : Kalman initialisation, ``"auto"``, ``"stationary"`` or ``"diffuse"``.
    order : 1 for the Kalman likelihood, 2 for a pruned nonlinear likelihood.
    nonlinear_filter : ``"particle"`` or ``"inversion"`` when ``order=2``.
    particles, particle_seed, resample : particle count, common random seed and
        systematic-resampling threshold for ``order=2``.
    first_obs, presample : observations dropped before, and observations used but
        not counted in, the likelihood -- Dynare's options of the same names
        (``first_obs`` counted from 1).
    aggregation : observables seen at a lower frequency, ``{name: (how, every)}``,
        as for :func:`dsgelab.state_space`; first order only.
    """

    model: Model
    data: Data
    params: Params
    observables: Sequence[str]
    priors: Mapping = field(default_factory=dict)
    cov: Covariance = None
    measurement: Mapping[str, float] | float | None = None
    transforms: str | Mapping[str, str] | None = None
    steady_fn: Callable | None = None
    guess: Point | None = None
    init: str = "auto"
    qz_criterium: float | None = None
    first_obs: int = 1
    presample: int = 0
    free: Sequence[str] | None = None
    bounds: Mapping | None = None
    order: int = 1
    nonlinear_filter: str = "particle"
    particles: int = 1000
    particle_seed: int | None = 0
    resample: float | None = 0.5
    aggregation: Mapping[str, tuple[str, int]] | None = None
    _last_ss: SteadyState | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.observables = list(self.observables)
        pr = {}
        for k, v in dict(self.priors).items():
            pr[k] = v if isinstance(v, Prior) else make_prior(*v)
        self.priors = pr
        if self.free is None:
            self.free = list(self.priors)
        self.free = list(self.free)
        if not self.free:
            raise ValueError("nothing to estimate: give priors, or free= with bounds=")
        if self.order not in (1, 2, 3):
            raise ValueError("order must be 1, 2 or 3")
        if self.nonlinear_filter not in ("particle", "inversion"):
            raise ValueError("nonlinear_filter must be 'particle' or 'inversion'")
        if self.order == 3 and self.nonlinear_filter != "particle":
            raise ValueError("third order requires nonlinear_filter='particle'")
        if self.particles < 1:
            raise ValueError("particles must be positive")
        if self.order >= 2 and self.aggregation:
            raise ValueError("temporal aggregation needs order=1 (the Kalman filter)")
        if self.order >= 2 and self.init == "diffuse":
            raise ValueError("the particle filter starts at the stationary distribution")
        Y = self._data_matrix()
        self.Y = Y[self.first_obs - 1:]

    # -- data and parameters ---------------------------------------------
    def _data_matrix(self):
        if isinstance(self.data, Mapping):
            cols = [np.asarray(self.data[n], dtype=float).ravel() for n in self.observables]
            return np.column_stack(cols)
        Y = np.asarray(self.data, dtype=float)
        return Y.reshape(-1, 1) if Y.ndim == 1 else Y

    @property
    def names(self) -> list:
        return list(self.free)

    def start(self) -> np.ndarray:
        """Starting values: the calibrated value, else the prior mean."""
        out = []
        sds = self._base_sds()
        meas = self._base_meas()
        for n in self.names:
            if n.startswith("stderr "):
                target = n[7:]
                v = sds.get(target, meas.get(target, np.nan))
            else:
                try:
                    v = float(self.params[n] if isinstance(self.params, Mapping)
                              else getattr(self.params, n))
                except (KeyError, AttributeError):
                    v = np.nan
            if (not np.isfinite(v) or v <= 0 and n.startswith("stderr ")) and n in self.priors:
                v = self.priors[n].mean
            out.append(float(v))
        return np.array(out)

    def _base_sds(self) -> dict:
        from .solution import shock_covariance
        S = shock_covariance(self.model.shocks, self.cov)
        return {e: float(np.sqrt(S[i, i])) for i, e in enumerate(self.model.shocks)}

    def _base_meas(self) -> dict:
        m = self.measurement
        if m is None:
            return {}
        if isinstance(m, Mapping):
            return {k: float(v) for k, v in m.items()}
        return {n: float(m) for n in self.observables}

    def unpack(self, theta: ArrayLike) -> tuple[Params, np.ndarray, dict[str, float]]:
        """``(params, shock covariance, measurement sds)`` at ``theta``."""
        return unpack_point(self.model, self.params, self.cov, self._base_meas(),
                            self.names, theta)

    # -- densities ---------------------------------------------------------
    def log_prior(self, theta: ArrayLike) -> float:
        lp = 0.0
        for n, v in zip(self.names, np.asarray(theta, dtype=float)):
            if n in self.priors:
                lp += self.priors[n].logpdf(v)
            elif self.bounds and n in self.bounds:
                lo, hi = self.bounds[n]
                if not (lo <= v <= hi):
                    return -np.inf
        return float(lp)

    def solve(self, theta: ArrayLike) -> tuple[FirstOrderSolution, StateSpace]:
        """``(solution, state space)`` at ``theta``; raises if there is none."""
        from .solve import solve
        from .steady import steady_state
        p, S, meas = self.unpack(theta)
        if self.steady_fn is not None:
            ss = self.steady_fn(p)
        else:
            start = self._last_ss.values if self._last_ss is not None else self.guess
            ss = steady_state(self.model, start, p)
        sol = solve(self.model, p, steady=ss, qz_criterium=self.qz_criterium)
        self._last_ss = ss
        space = state_space(sol, self.observables, transforms=self.transforms,
                            measurement=meas or None, cov=S, aggregation=self.aggregation)
        return sol, space

    def log_likelihood(self, theta: ArrayLike) -> float:
        try:
            sol, space = self.solve(theta)
            fr: FilterResult | ParticleFilterResult | InversionFilterResult
            from .secondorder import solve_second_order
            if self.order == 2:
                second = solve_second_order(sol, cov=space.Sigma)
                if self.nonlinear_filter == "inversion":
                    from .inversion import inversion_filter
                    fr = inversion_filter(second, self.Y, observables=self.observables,
                                          transforms=self.transforms, presample=self.presample)
                else:
                    from .particle import particle_filter
                    fr = particle_filter(second, self.Y, observables=self.observables,
                                         transforms=self.transforms, measurement=space.H,
                                         particles=self.particles, seed=self.particle_seed,
                                         resample=self.resample)
            elif self.order == 3:
                from .thirdorder import solve_third_order
                third = solve_third_order(solve_second_order(sol, cov=space.Sigma))
                from .particle import particle_filter
                fr = particle_filter(third, self.Y, observables=self.observables,
                                     transforms=self.transforms, measurement=space.H,
                                     particles=self.particles, seed=self.particle_seed,
                                     resample=self.resample)
            else:
                fr = kalman_filter(space, self.Y, init=self.init)
        except Exception:
            return -np.inf
        ll: float = fr.loglik_t[self.presample:].sum()
        return float(ll) if np.isfinite(ll) else -np.inf

    def log_posterior(self, theta: ArrayLike) -> float:
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        ll = self.log_likelihood(theta)
        return float(lp + ll) if np.isfinite(ll) else -np.inf

    def smoother(self, theta: ArrayLike) -> SmootherResult | ParticleSmootherResult | InversionFilterResult:
        """Smoothed states, shocks and measurement errors at ``theta``."""
        if self.order == 3:
            from .particle import particle_smoother
            return particle_smoother(self.filter(theta))
        if self.order == 2:
            if self.nonlinear_filter == "inversion":
                from .inversion import inversion_smoother
                sol, space = self.solve(theta)
                from .secondorder import solve_second_order
                second = solve_second_order(sol, cov=space.Sigma)
                return inversion_smoother(second, self.Y, observables=self.observables,
                                          transforms=self.transforms, presample=self.presample)
            from .particle import particle_smoother
            return particle_smoother(self.filter(theta))
        _, space = self.solve(theta)
        return kalman_smoother(space, self.Y, init=self.init)

    def filter(self, theta: ArrayLike) -> FilterResult | ParticleFilterResult | InversionFilterResult:
        sol, space = self.solve(theta)
        from .secondorder import solve_second_order
        if self.order == 2:
            second = solve_second_order(sol, cov=space.Sigma)
            if self.nonlinear_filter == "inversion":
                from .inversion import inversion_filter
                return inversion_filter(second, self.Y, observables=self.observables,
                                        transforms=self.transforms, presample=self.presample)
            from .particle import particle_filter
            return particle_filter(second, self.Y, observables=self.observables,
                                   transforms=self.transforms, measurement=space.H,
                                   particles=self.particles, seed=self.particle_seed,
                                   resample=self.resample, store=True)
        if self.order == 3:
            from .thirdorder import solve_third_order
            third = solve_third_order(solve_second_order(sol, cov=space.Sigma))
            from .particle import particle_filter
            return particle_filter(third, self.Y, observables=self.observables,
                                   transforms=self.transforms, measurement=space.H,
                                   particles=self.particles, seed=self.particle_seed,
                                   resample=self.resample, store=True)
        return kalman_filter(space, self.Y, init=self.init)

    # -- support, for unconstrained optimisation --------------------------
    def _support(self):
        lo, hi = [], []
        for n in self.names:
            if n in self.priors:
                pr = self.priors[n]
                lo.append(pr.lower)
                hi.append(pr.upper)
            elif self.bounds and n in self.bounds:
                lo.append(self.bounds[n][0])
                hi.append(self.bounds[n][1])
            else:
                lo.append(-np.inf)
                hi.append(np.inf)
        return np.array(lo, float), np.array(hi, float)


# ===================================================================== mode
@dataclass
class ModeResult:
    """A mode (posterior or likelihood), its curvature, and what follows from it."""

    names: list
    x: np.ndarray
    values: dict
    log_posterior: float          # or the log likelihood, for maximum likelihood
    log_likelihood: float
    hessian: np.ndarray           # of MINUS the objective, at the mode
    covariance: np.ndarray        # its inverse
    std: dict
    laplace: float | None         # log marginal density, Laplace approximation
    success: bool
    message: str
    evaluations: int


def _to_unbounded(x, lo, hi):
    y = np.array(x, float)
    for i in range(len(y)):
        a, b = lo[i], hi[i]
        if np.isfinite(a) and np.isfinite(b):
            u = (x[i] - a) / (b - a)
            y[i] = math.log(u / (1 - u))
        elif np.isfinite(a):
            y[i] = math.log(x[i] - a)
        elif np.isfinite(b):
            y[i] = -math.log(b - x[i])
    return y


def _from_unbounded(y, lo, hi):
    x = np.array(y, float)
    for i in range(len(x)):
        a, b = lo[i], hi[i]
        if np.isfinite(a) and np.isfinite(b):
            x[i] = a + (b - a) / (1 + math.exp(-y[i]))
        elif np.isfinite(a):
            x[i] = a + math.exp(y[i])
        elif np.isfinite(b):
            x[i] = b - math.exp(-y[i])
    return x


def hessian(f: Callable[[np.ndarray], float], x: ArrayLike,
            steps: ArrayLike | None = None) -> np.ndarray:
    """Central-difference Hessian of ``f`` at ``x``, steps relative to ``|x|``."""
    x = np.asarray(x, float)
    k = x.size
    h = np.array(steps if steps is not None else 1e-4 * np.maximum(np.abs(x), 1e-2))
    H = np.zeros((k, k))
    f0 = f(x)
    for i in range(k):
        ei = np.zeros(k); ei[i] = h[i]
        fp, fm = f(x + ei), f(x - ei)
        H[i, i] = (fp - 2 * f0 + fm) / h[i] ** 2
        for j in range(i):
            ej = np.zeros(k); ej[j] = h[j]
            H[i, j] = H[j, i] = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej)
                                 + f(x - ei - ej)) / (4 * h[i] * h[j])
    return H


def _optimise(problem, objective, x0, method, options, gradient=None):
    from scipy.optimize import minimize
    lo, hi = problem._support()
    x0 = np.asarray(x0 if x0 is not None else problem.start(), float)
    count = {"n": 0}

    def g(y):
        count["n"] += 1
        v = objective(_from_unbounded(y, lo, hi))
        return 1e10 if not np.isfinite(v) else -v

    def with_gradient(y):
        value = g(y)
        if value >= 1e10:      # no solution here: the penalty is flat, and the line search backs off
            return value, np.zeros(len(y))
        x = _from_unbounded(y, lo, hi)
        derivative = gradient(x).gradient
        scale = np.ones(len(y))
        for i, (a, b) in enumerate(zip(lo, hi)):
            if np.isfinite(a) and np.isfinite(b):
                scale[i] = (b - a) * math.exp(-y[i]) / (1 + math.exp(-y[i])) ** 2
            elif np.isfinite(a):
                scale[i] = math.exp(y[i])
            elif np.isfinite(b):
                scale[i] = math.exp(-y[i])
        return value, -derivative * scale

    y0 = _to_unbounded(x0, lo, hi)
    use_gradient = gradient is not None and (method is None or method == "BFGS")
    res = minimize(with_gradient if use_gradient else g, y0, jac=use_gradient,
                   method=method or "BFGS", options={"gtol": 1e-8, **(options or {})})
    pol = minimize(g, res.x, method="Nelder-Mead",
                   options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 20000, "maxfev": 20000})
    if pol.fun < res.fun:
        res = pol
    return _from_unbounded(res.x, lo, hi), res, count["n"]


def _mode_result(problem, objective, x, res, n, with_prior):
    H = -hessian(objective, x)
    H = 0.5 * (H + H.T)
    try:
        C = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        C = np.full_like(H, np.nan)
    std = {nm: float(math.sqrt(C[i, i])) if C[i, i] > 0 else float("nan")
           for i, nm in enumerate(problem.names)}
    val = float(objective(x))
    laplace = None
    if with_prior:
        sign, logdet = np.linalg.slogdet(H)
        if sign > 0:
            laplace = val + 0.5 * len(x) * math.log(2 * math.pi) - 0.5 * logdet
    return ModeResult(names=problem.names, x=x, values=dict(zip(problem.names, map(float, x))),
                      log_posterior=val, log_likelihood=problem.log_likelihood(x), hessian=H,
                      covariance=C, std=std, laplace=laplace,
                      success=bool(np.isfinite(val)), message=str(res.message), evaluations=n)


def posterior_mode(problem: EstimationProblem, x0: ArrayLike | None = None, *,
                   method: str | None = None,
                   options: Mapping | None = None) -> ModeResult:
    """Maximise the posterior; report the Hessian and the Laplace marginal density.

    The search runs in an unbounded reparametrisation of each prior's support
    (log for a half line, logit for an interval) with BFGS and a simplex polish;
    the Hessian is a central difference in the original parameters, as Dynare's.
    ``laplace`` is ``log p(Y)`` by the Laplace approximation,
    ``log posterior kernel + k/2 log 2 pi - 1/2 log det H``.
    """
    gradient = None
    if problem.order == 1:
        from .derivatives import log_posterior_gradient
        gradient = lambda theta: log_posterior_gradient(problem, theta)
    x, res, n = _optimise(problem, problem.log_posterior, x0, method, options, gradient)
    return _mode_result(problem, problem.log_posterior, x, res, n, True)


def maximum_likelihood(problem: EstimationProblem, x0: ArrayLike | None = None, *,
                       method: str | None = None,
                       options: Mapping | None = None) -> ModeResult:
    """Maximise the likelihood. Standard errors from the inverse Hessian.

    Priors are ignored except for their support, which keeps the search where the
    model is defined (a standard deviation stays positive, a share inside [0, 1]).
    """
    def ll(x):
        lo, hi = problem._support()
        if np.any(x < lo) or np.any(x > hi):
            return -np.inf
        return problem.log_likelihood(x)

    gradient = None
    if problem.order == 1:
        from .derivatives import log_likelihood_gradient
        gradient = lambda theta: log_likelihood_gradient(problem, theta)
    x, res, n = _optimise(problem, ll, x0, method, options, gradient)
    return _mode_result(problem, ll, x, res, n, False)


# ===================================================================== MCMC
@dataclass
class MCMCResult:
    """Chains of a random-walk Metropolis-Hastings, with their diagnostics."""

    names: list
    chains: np.ndarray            # (n_chains, n_draws, k), after burn-in
    log_posterior: np.ndarray     # (n_chains, n_draws)
    acceptance: np.ndarray        # per chain
    burn: int

    @property
    def draws(self) -> np.ndarray:
        return self.chains.reshape(-1, self.chains.shape[-1])

    def mean(self) -> dict:
        return dict(zip(self.names, self.draws.mean(axis=0)))

    def quantiles(self, q: float | Sequence[float] = (0.05, 0.5, 0.95)) -> dict[str, Any]:
        return {n: np.quantile(self.draws[:, i], q) for i, n in enumerate(self.names)}

    def hpd(self, mass: float = 0.9) -> dict:
        """Shortest interval holding ``mass`` of the draws, per parameter."""
        out = {}
        for i, n in enumerate(self.names):
            x = np.sort(self.draws[:, i])
            m = int(math.floor(mass * len(x)))
            widths = x[m:] - x[:len(x) - m]
            j = int(np.argmin(widths))
            out[n] = (float(x[j]), float(x[j + m]))
        return out

    def rhat(self) -> dict:
        return rhat(self.chains, self.names)

    def ess(self) -> dict:
        return effective_sample_size(self.chains, self.names)

    def log_marginal_mhm(self, tau: Sequence[float] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                                                         0.9)) -> float:
        """Geweke's (1999) modified harmonic mean, averaged over truncations ``tau``,
        as Dynare's ``ModifiedHarmonicMean``."""
        X = self.draws
        lp = self.log_posterior.reshape(-1)
        k = X.shape[1]
        mu = X.mean(axis=0)
        S = np.cov(X.T).reshape(k, k)
        Sinv = np.linalg.inv(S)
        sign, logdetS = np.linalg.slogdet(S)
        dev = X - mu
        q = np.einsum("ij,jk,ik->i", dev, Sinv, dev)
        from scipy.stats import chi2
        lmax = lp.max()
        out = []
        for t in tau:
            crit = chi2.ppf(t, k)
            inside = q <= crit
            logf = (-math.log(t) - 0.5 * k * math.log(2 * math.pi) - 0.5 * logdetS - 0.5 * q)
            terms = np.where(inside, np.exp(logf - lp + lmax), 0.0)
            out.append(lmax - math.log(terms.mean()))
        return float(np.mean(out))


def rhat(chains: np.ndarray, names: Sequence[str] | None = None) -> dict:
    """Split-R-hat of Gelman et al. (2013) for each parameter."""
    m, n, k = chains.shape
    half = n // 2
    split = np.concatenate([chains[:, :half], chains[:, half:2 * half]], axis=0)
    mm, nn = split.shape[0], split.shape[1]
    out = {}
    for i in range(k):
        x = split[:, :, i]
        means = x.mean(axis=1)
        B = nn * means.var(ddof=1)
        W = x.var(axis=1, ddof=1).mean()
        var = (nn - 1) / nn * W + B / nn
        out[(names or range(k))[i]] = float(math.sqrt(var / W)) if W > 0 else float("nan")
    return out


def effective_sample_size(chains: np.ndarray, names: Sequence[str] | None = None) -> dict:
    """Effective sample size from the chains' autocorrelations (Geyer's initial
    positive sequence)."""
    m, n, k = chains.shape
    out = {}
    for i in range(k):
        x = chains[:, :, i]
        xc = x - x.mean()
        var = xc.var()
        if var == 0:
            out[(names or range(k))[i]] = float("nan")
            continue
        acf = []
        for lag in range(1, n):
            c = np.mean([np.mean(xc[j, :-lag] * xc[j, lag:]) for j in range(m)]) / var
            acf.append(c)
            if lag > 2 and lag % 2 == 1 and acf[-1] + acf[-2] < 0:
                break
        s = 0.0
        for j in range(0, len(acf) - 1, 2):
            pair = acf[j] + acf[j + 1]
            if pair < 0:
                break
            s += pair
        tau = 1 + 2 * s
        out[(names or range(k))[i]] = float(m * n / tau)
    return out


def metropolis_hastings(problem: EstimationProblem, mode: ModeResult, *, draws: int = 10000,
                        chains: int = 2, scale: float = 0.8, init_scale: float | None = None,
                        burn: float = 0.5, seed: int = 0) -> MCMCResult:
    """Random-walk Metropolis-Hastings, Dynare's sampler.

    The proposal is ``N(0, scale^2 * C)`` with ``C`` the inverse Hessian at the
    mode (Dynare's ``mh_jscale``). Each chain starts at a draw from
    ``N(mode, init_scale^2 * C)`` (Dynare's ``mh_init_scale``, ``2 * scale``
    by default) that has a finite posterior. ``burn`` is the share of each chain
    dropped (Dynare's ``mh_drop``).
    """
    rng = np.random.default_rng(seed)
    C = mode.covariance
    if not np.all(np.isfinite(C)):
        raise ValueError("the mode has no usable inverse Hessian to build a proposal from")
    L = np.linalg.cholesky(0.5 * (C + C.T) + 1e-12 * np.eye(len(mode.x)))
    init_scale = 2 * scale if init_scale is None else init_scale
    k = len(mode.x)
    kept = np.empty((chains, draws, k))
    lps = np.empty((chains, draws))
    acc = np.zeros(chains)
    for c in range(chains):
        for _ in range(1000):
            cur = mode.x + init_scale * L @ rng.standard_normal(k)
            lp = problem.log_posterior(cur)
            if np.isfinite(lp):
                break
        else:
            cur, lp = mode.x.copy(), problem.log_posterior(mode.x)
        for d in range(draws):
            prop = cur + scale * L @ rng.standard_normal(k)
            plp = problem.log_posterior(prop)
            if np.isfinite(plp) and math.log(rng.random()) < plp - lp:
                cur, lp = prop, plp
                acc[c] += 1
            kept[c, d], lps[c, d] = cur, lp
    b = int(burn * draws)
    return MCMCResult(names=problem.names, chains=kept[:, b:], log_posterior=lps[:, b:],
                      acceptance=acc / draws, burn=b)


# ===================================================================== SMC
@dataclass
class SMCResult:
    """Sequential Monte Carlo with likelihood tempering (Herbst and Schorfheide, 2014)."""

    names: list
    particles: np.ndarray         # (N, k), equally weighted after the last resampling
    weights: np.ndarray
    log_marginal: float           # log p(Y), a by-product of the tempering
    stages: int
    acceptance: list

    def mean(self) -> dict:
        return dict(zip(self.names, np.average(self.particles, axis=0, weights=self.weights)))


def smc(problem: EstimationProblem, *, particles: int = 1000, stages: int = 50,
        bend: float = 2.0, mutation_steps: int = 2, seed: int = 0) -> SMCResult:
    """Tempered SMC from the prior to the posterior.

    Stage ``n`` targets ``prior x likelihood^phi_n`` with ``phi_n = (n/N)^bend``.
    Each stage reweights by the likelihood increment, resamples when the
    effective sample size falls below half, and mutates by a random-walk
    Metropolis step whose covariance is the particles' own and whose scale adapts
    to keep the acceptance near a quarter. The product of the mean incremental
    weights is the marginal likelihood -- no harmonic mean, no mode.
    """
    rng = np.random.default_rng(seed)
    names = problem.names
    k = len(names)
    if any(n not in problem.priors for n in names):
        raise ValueError("SMC samples from the prior: every estimated name needs one")
    X = np.column_stack([problem.priors[n].sample(rng, particles) for n in names])
    ll = np.array([problem.log_likelihood(x) for x in X])
    w = np.full(particles, 1.0 / particles)
    phis = (np.arange(stages + 1) / stages) ** bend
    log_ml = 0.0
    c = 0.5
    accs = []
    for s in range(1, stages + 1):
        dphi = phis[s] - phis[s - 1]
        inc = np.where(np.isfinite(ll), dphi * ll, -np.inf)
        m = np.max(inc[np.isfinite(inc)]) if np.any(np.isfinite(inc)) else 0.0
        wi = w * np.exp(inc - m)
        tot = wi.sum()
        log_ml += m + math.log(tot)
        w = wi / tot
        ess = 1.0 / np.sum(w ** 2)
        if ess < particles / 2:
            idx = rng.choice(particles, size=particles, p=w)
            X, ll = X[idx], ll[idx]
            w = np.full(particles, 1.0 / particles)
        cov = np.cov(X.T, aweights=w).reshape(k, k) + 1e-12 * np.eye(k)
        Lc = np.linalg.cholesky(cov)
        lp = np.array([problem.log_prior(x) for x in X])
        accepted = 0
        for _ in range(mutation_steps):
            prop = X + c * (rng.standard_normal((particles, k)) @ Lc.T)
            for i in range(particles):
                plp = problem.log_prior(prop[i])
                if not np.isfinite(plp):
                    continue
                pll = problem.log_likelihood(prop[i])
                if not np.isfinite(pll):
                    continue
                if math.log(rng.random()) < (plp + phis[s] * pll) - (lp[i] + phis[s] * ll[i]):
                    X[i], ll[i], lp[i] = prop[i], pll, plp
                    accepted += 1
        rate = accepted / (particles * mutation_steps)
        accs.append(rate)
        c *= 0.95 + 0.10 * math.exp(16 * (rate - 0.25)) / (1 + math.exp(16 * (rate - 0.25)))
    return SMCResult(names=names, particles=X, weights=w, log_marginal=float(log_ml),
                     stages=stages, acceptance=accs)


# ============================================================ prior predictive
def prior_predictive(problem: EstimationProblem, draws: int = 1000, *, seed: int = 0,
                     statistics: Callable | None = None) -> dict:
    """What the prior implies before any data: how often the model can be solved.

    Draws from the prior and reports the shares of draws with a determinate
    solution, with no steady state, and with an indeterminate or explosive one.
    ``statistics(sol) -> mapping`` adds any statistic of the solution
    (theoretical moments, impulse responses) collected over the solvable draws.
    """
    from .klein import BlanchardKahnError
    rng = np.random.default_rng(seed)
    names = problem.names
    X = np.column_stack([problem.priors[n].sample(rng, draws) for n in names])
    counts = {"determinate": 0, "no_steady_state": 0, "blanchard_kahn": 0, "other": 0}
    stats = []
    for x in X:
        try:
            sol, _ = problem.solve(x)
        except BlanchardKahnError:
            counts["blanchard_kahn"] += 1
            continue
        except RuntimeError:
            counts["no_steady_state"] += 1
            continue
        except Exception:
            counts["other"] += 1
            continue
        counts["determinate"] += 1
        if statistics is not None:
            stats.append(dict(statistics(sol)))
    out = {k: v / draws for k, v in counts.items()}
    out["draws"] = draws
    if stats:
        out["statistics"] = {k: np.array([s[k] for s in stats]) for k in stats[0]}
    return out


def model_probabilities(log_marginals: Mapping[str, float],
                        prior: Mapping[str, float] | None = None) -> dict:
    """Posterior model probabilities from log marginal likelihoods."""
    names = list(log_marginals)
    lm = np.array([log_marginals[n] for n in names], float)
    lp = np.log(np.array([(prior or {}).get(n, 1.0) for n in names], float))
    z = lm + lp
    z -= z.max()
    p = np.exp(z) / np.exp(z).sum()
    return dict(zip(names, p))


# ===================================================================== forecasts
def posterior_forecast(problem: EstimationProblem, draws: MCMCResult | SMCResult | ArrayLike,
                       horizon: int, *,
                       variables: Sequence[str] | None = None, n_draws: int = 200,
                       shocks: bool = True, coverage: float = 0.9, logs: Sequence[str] = (),
                       seed: int = 0) -> dict:
    """Forecast bands that carry parameter uncertainty as well as shock uncertainty.

    For ``n_draws`` parameter vectors taken from ``draws`` (an :class:`MCMCResult`,
    an :class:`SMCResult`, or a ``(N, k)`` array), the model is solved, the
    sample filtered, and the forecast path simulated from a draw of the last
    filtered state with drawn future shocks (``shocks=False`` keeps only the
    parameter and state uncertainty). The quantiles of the simulated paths are
    the bands; ``mean`` is their average -- Dynare's forecast after
    Metropolis-Hastings.
    """
    from .kalman import kalman_filter
    rng = np.random.default_rng(seed)
    if isinstance(draws, MCMCResult):
        X = draws.draws
    elif isinstance(draws, SMCResult):
        X = draws.particles[rng.choice(len(draws.particles), size=len(draws.particles),
                                       p=draws.weights / draws.weights.sum())]
    else:
        X = np.asarray(draws, float)
    pick = X[rng.choice(len(X), size=min(n_draws, len(X)), replace=False)]
    paths = []
    names = None
    for x in pick:
        try:
            sol, space = problem.solve(x)
            fr = kalman_filter(space, problem.Y, init=problem.init)
        except Exception:
            continue
        names = list(variables) if variables is not None else list(sol.model.names)
        a = rng.multivariate_normal(fr.a_filt[-1], 0.5 * (fr.P_filt[-1] + fr.P_filt[-1].T),
                                    method="eigh")
        L = np.linalg.cholesky(space.Sigma + 1e-300 * np.eye(len(space.shocks))) \
            if np.any(space.Sigma) else np.zeros_like(space.Sigma)
        J = sol._J()
        path = np.zeros((horizon, len(names)))
        for h in range(horizon):
            e = L @ rng.standard_normal(len(space.shocks)) if shocks else 0.0
            a = space.T @ a + (space.R @ e if shocks else 0.0)
            xs = J @ a[:space.n_model_states]
            for i, v in enumerate(names):
                j = sol.model.position(v)
                c, s = sol.coords.center[j], sol.coords.scale[j]
                path[h, i] = (np.log(c) + xs[j]) if v in logs else (c + s * xs[j])
        paths.append(path)
    if not paths:
        raise RuntimeError("no draw could be solved")
    P = np.array(paths)
    lo, hi = (1 - coverage) / 2, 1 - (1 - coverage) / 2
    return dict(mean={v: P[:, :, i].mean(axis=0) for i, v in enumerate(names)},
                lower={v: np.quantile(P[:, :, i], lo, axis=0) for i, v in enumerate(names)},
                upper={v: np.quantile(P[:, :, i], hi, axis=0) for i, v in enumerate(names)},
                paths=P, names=names, draws=len(paths))
