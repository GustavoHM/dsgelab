"""Bayesian VARs: the Sims–Zha prior, and the prior a DSGE model implies.

A VAR with ``p`` lags is written as a regression, ``y_t' = x_t' Phi + u_t'``, with
``x_t = [y_{t-1}; ...; y_{t-p}; 1]`` (the constant when there is one) and
``u_t ~ N(0, Sigma)``. The conjugate prior and posterior are normal-inverse-Wishart::

    Sigma ~ IW(S, df),        vec(Phi) | Sigma ~ N(vec(Phi_mean), Sigma kron XXi),

and :class:`NormalInverseWishart` holds the four numbers, in the names of Dynare's
``oo_.bvar.posterior`` (``PhiHat`` there is ``Phi`` here).

``bvar`` is Dynare's ``bvar_density`` and ``bvar_forecast`` (the ``+bvar`` package,
after Sims's ``varprior.m`` and ``rfvar3.m``). The Minnesota-style prior of Sims and
Zha (1998) is a set of dummy observations appended to the data: ``tau`` sets the
overall tightness, ``decay`` the shrinkage of lag ``l`` as ``l^-decay``, ``omega`` the
weight of the prior on ``Sigma``, ``lambda_`` the co-persistence dummy and ``mu`` the
own-persistence (sum-of-coefficients) dummies. The scale of each variable is the
standard deviation of the ``p + 1`` observations that start the sample. With a
training sample (``train``), the prior is the posterior from the dummies and the
training observations, and the marginal density is that of the rest of the sample.

The DSGE-VAR of Del Negro and Schorfheide (2004) is the same VAR with the prior a
DSGE model implies: ``lambda T`` artificial observations drawn from the model's
population moments. It is estimated through :class:`~dsgelab.EstimationProblem`
with ``dsge_var=`` (the weight ``lambda``) or with ``"dsge_prior_weight"`` among the
estimated names, as in Dynare's ``estimation(dsge_var=...)``. The helpers at the end
of this module compute the model's moments and the likelihood concentrated over the
VAR's parameters, formula (A.2) of Del Negro and Schorfheide (2004).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import gammaln

__all__ = ["NormalInverseWishart", "BVAR", "BVARForecast", "bvar", "bvar_density"]

_LOG2PI = math.log(2.0 * math.pi)


def _generator(seed):
    """A numpy Generator from a seed, or the object itself when it draws normals."""
    return seed if hasattr(seed, "standard_normal") else np.random.default_rng(seed)


# ======================================================== normal-inverse-Wishart
@dataclass
class NormalInverseWishart:
    """``Sigma ~ IW(S, df)`` and ``vec(Phi) | Sigma ~ N(vec(Phi), Sigma kron XXi)``.

    ``Phi`` is ``(k, n)``, one column per equation, with the regressors in the
    order of :mod:`dsgelab.bvar`: the first lag of every variable, then the
    second, and the constant last.
    """

    Phi: np.ndarray          # (k, n), the mean of the coefficients
    XXi: np.ndarray          # (k, k)
    S: np.ndarray            # (n, n)
    df: float

    @property
    def Sigma(self) -> np.ndarray:
        """The mean of ``Sigma``, ``S / (df - n - 1)``; NaN when it does not exist."""
        n = self.S.shape[0]
        if self.df <= n + 1:
            return np.full_like(self.S, np.nan)
        return self.S / (self.df - n - 1)

    @property
    def Phi_std(self) -> np.ndarray:
        """Standard deviations of the coefficients, ``sqrt(XXi_ii E[Sigma]_jj)``."""
        return np.sqrt(np.outer(np.diag(self.XXi), np.diag(self.Sigma)))

    def log_integral(self) -> float:
        """Log of the integral of the kernel, Dynare's ``matrictint``.

        ``k n / 2 log 2 pi + n / 2 log|XXi| - df / 2 log|S| + df n / 2 log 2
        + log Gamma_n(df / 2)``. The log marginal density of a sample is the
        posterior's integral less the prior's, less ``n T / 2 log 2 pi``.
        """
        k, n = self.Phi.shape
        cx = np.linalg.cholesky(self.XXi)
        cs = np.linalg.cholesky(self.S)
        if self.df <= n - 1:
            raise ValueError(f"df = {self.df}: an inverse Wishart in {n} dimensions needs more "
                             f"than {n - 1} degrees of freedom")
        w1 = 0.5 * k * n * _LOG2PI + n * np.sum(np.log(np.diag(cx)))
        w2 = (-self.df * np.sum(np.log(np.diag(cs))) + 0.5 * self.df * n * math.log(2.0)
              + 0.25 * n * (n - 1) * math.log(math.pi)
              + np.sum(gammaln(0.5 * (self.df - np.arange(n)))))
        return float(w1 + w2)

    def _draw(self, rng, S_inv_chol, XXi_chol) -> tuple[np.ndarray, np.ndarray]:
        """One draw of ``(Phi, Sigma)``, consuming normals in the order Dynare does."""
        k, n = self.Phi.shape
        df = self.df
        if float(df).is_integer():
            # Dynare's rand_inverse_wishart: X'X is Wishart(S^-1, df), its inverse IW(S, df)
            X = np.asarray(rng.standard_normal(int(df) * n)).reshape((int(df), n), order="F")
            _, s, Vt = np.linalg.svd(X @ S_inv_chol.T, full_matrices=False)
            Sigma = (Vt.T / s ** 2) @ Vt
        else:
            # Bartlett's decomposition, for the non-integer df of a DSGE-VAR
            A = np.zeros((n, n))
            A[np.diag_indices(n)] = np.sqrt(rng.chisquare(df - np.arange(n)))
            low = np.tril_indices(n, -1)
            A[low] = rng.standard_normal(len(low[0]))
            L = S_inv_chol @ A
            Sigma = np.linalg.inv(L @ L.T)
        Sigma = 0.5 * (Sigma + Sigma.T)
        Sigma_chol = np.linalg.cholesky(Sigma)
        Z = np.asarray(rng.standard_normal(k * n)).reshape((k, n), order="F")
        return self.Phi + XXi_chol @ Z @ Sigma_chol.T, Sigma

    def draw(self, draws: int = 1000, *, seed: Any = 0) -> tuple[np.ndarray, np.ndarray]:
        """``(Phi, Sigma)``, arrays of ``draws`` coefficients ``(k, n)`` and covariances ``(n, n)``.

        ``seed`` is a seed or a numpy ``Generator``. With a whole ``df`` the draws use
        Dynare's construction, ``Sigma`` the inverse of ``X'X`` with ``X`` a
        ``df x n`` normal matrix times the Cholesky factor of ``S^-1``.
        """
        rng = _generator(seed)
        S_inv_chol = np.linalg.cholesky(np.linalg.inv(self.S))
        XXi_chol = np.linalg.cholesky(self.XXi)
        k, n = self.Phi.shape
        Phis, Sigmas = np.empty((draws, k, n)), np.empty((draws, n, n))
        for d in range(draws):
            Phis[d], Sigmas[d] = self._draw(rng, S_inv_chol, XXi_chol)
        return Phis, Sigmas


# ========================================================================= BVAR
@dataclass
class BVARForecast:
    """Forecasts of a :class:`BVAR`, over draws from the posterior.

    ``paths`` holds one path per draw, ``(draws, horizon, n)``, with the shocks
    drawn too, and ``paths_no_shock`` the same coefficients with no shocks: the
    two sets Dynare's ``bvar_forecast`` reports as ``with_shocks`` and
    ``no_shock``. ``point`` is the path at the posterior mean of the coefficients.
    """

    names: list
    point: dict
    paths: np.ndarray
    paths_no_shock: np.ndarray
    coverage: float
    explosive: float              # share of draws with a root of modulus above one

    def _paths(self, shocks: bool) -> np.ndarray:
        return self.paths if shocks else self.paths_no_shock

    def mean(self, *, shocks: bool = True) -> dict:
        P = self._paths(shocks)
        return {v: P[:, :, i].mean(axis=0) for i, v in enumerate(self.names)}

    def median(self, *, shocks: bool = True) -> dict:
        P = self._paths(shocks)
        return {v: np.median(P[:, :, i], axis=0) for i, v in enumerate(self.names)}

    def bands(self, coverage: float | None = None, *,
              shocks: bool = True) -> tuple[dict, dict]:
        """``(lower, upper)``, the central band holding ``coverage`` of the draws."""
        c = self.coverage if coverage is None else coverage
        P = self._paths(shocks)
        lo, hi = np.quantile(P, [(1 - c) / 2, (1 + c) / 2], axis=0)
        return ({v: lo[:, i] for i, v in enumerate(self.names)},
                {v: hi[:, i] for i, v in enumerate(self.names)})


@dataclass
class BVAR:
    """A VAR with the Sims–Zha prior: prior, posterior and marginal density.

    ``prior`` and ``posterior`` are :class:`NormalInverseWishart`. ``log_marginal``
    is the log marginal density of the observations from ``first_obs + presample``
    on, Dynare's ``oo_.bvar.log_marginal_data_density``. ``initial`` holds the last
    ``lags`` observations of the sample, the oldest first, from which forecasts
    start, and ``realized`` the observations after the sample, if any.
    """

    names: list
    lags: int
    constant: bool
    prior: NormalInverseWishart
    posterior: NormalInverseWishart
    log_marginal: float
    nobs: int                     # observations in the marginal density
    initial: np.ndarray
    realized: np.ndarray
    hyperparameters: dict = field(default_factory=dict)

    @property
    def lag_matrices(self) -> np.ndarray:
        """``A[l - 1]``, ``(lags, n, n)``, with ``y_t = c + sum_l A_l y_{t-l}`` at the posterior mean."""
        n = len(self.names)
        Phi = self.posterior.Phi
        return np.array([Phi[l * n:(l + 1) * n].T for l in range(self.lags)])

    @property
    def intercept(self) -> np.ndarray:
        """``c`` at the posterior mean, zero without a constant."""
        return self.posterior.Phi[-1].copy() if self.constant else np.zeros(len(self.names))

    def _companion(self, Phi: np.ndarray) -> np.ndarray:
        n, p = len(self.names), self.lags
        C = np.zeros((n * p, n * p))
        C[:n] = Phi[:n * p].T
        C[n:, :-n] = np.eye(n * (p - 1))
        return C

    def _step(self, history: np.ndarray, Phi: np.ndarray) -> np.ndarray:
        """``x' Phi`` with ``history`` the last ``lags`` values, the oldest first."""
        x = history[::-1].ravel()
        if self.constant:
            x = np.append(x, 1.0)
        return x @ Phi

    def forecast(self, horizon: int, draws: int = 2000, *, seed: Any = 0,
                 coverage: float = 0.6) -> BVARForecast:
        """Forecasts ``1..horizon`` periods after the sample, Dynare's ``bvar_forecast``.

        Each of ``draws`` draws takes ``(Phi, Sigma)`` from the posterior and runs
        the VAR forward from ``initial``, once without shocks and once with shocks
        drawn from ``N(0, Sigma)``. Explosive draws are kept and counted, as Dynare
        counts them. ``coverage`` is the default band of :meth:`BVARForecast.bands`
        (Dynare's ``bvar.conf_sig``, 0.6). ``seed`` is a seed or a numpy
        ``Generator``. With ``draws=0`` only ``point`` is computed.
        """
        rng = _generator(seed)
        post = self.posterior
        n = len(self.names)
        hist = self.initial.copy()
        point = np.empty((horizon, n))
        for t in range(horizon):
            point[t] = self._step(hist, post.Phi)
            hist = np.vstack([hist[1:], point[t]])
        paths = np.empty((draws, horizon, n))
        still = np.empty((draws, horizon, n))
        explosive = 0
        if draws:
            S_inv_chol = np.linalg.cholesky(np.linalg.inv(post.S))
            XXi_chol = np.linalg.cholesky(post.XXi)
        for d in range(draws):
            Phi, Sigma = post._draw(rng, S_inv_chol, XXi_chol)
            if np.any(np.abs(np.linalg.eigvals(self._companion(Phi))) > 1.0000000000001):
                explosive += 1
            hist = self.initial.copy()
            for t in range(horizon):
                still[d, t] = self._step(hist, Phi)
                hist = np.vstack([hist[1:], still[d, t]])
            chol = np.linalg.cholesky(Sigma)
            hist = self.initial.copy()
            for t in range(horizon):
                paths[d, t] = self._step(hist, Phi) + chol @ np.asarray(rng.standard_normal(n))
                hist = np.vstack([hist[1:], paths[d, t]])
        return BVARForecast(names=list(self.names),
                            point={v: point[:, i] for i, v in enumerate(self.names)},
                            paths=paths, paths_no_shock=still, coverage=coverage,
                            explosive=explosive / draws if draws else float("nan"))

    def rmse(self) -> dict:
        """RMSE of the forecast at the posterior mean over ``realized``, as Dynare prints it."""
        m = len(self.realized)
        if m == 0:
            raise ValueError("no observations after the sample: give nobs= shorter than the data")
        path = self.forecast(m, draws=0).point
        return {v: float(np.sqrt(np.mean((path[v] - self.realized[:, i]) ** 2)))
                for i, v in enumerate(self.names)}


def _matrix(data, observables) -> tuple[np.ndarray, list]:
    if isinstance(data, Mapping):
        names = list(observables) if observables is not None else list(data)
        cols = [np.asarray(data[v], dtype=float).ravel() for v in names]
        if len({c.size for c in cols}) != 1:
            raise ValueError("every series must have the same length")
        return np.column_stack(cols), names
    Y = np.asarray(data, dtype=float)
    Y = Y.reshape(-1, 1) if Y.ndim == 1 else Y
    names = list(observables) if observables is not None else [f"y{i + 1}"
                                                               for i in range(Y.shape[1])]
    if len(names) != Y.shape[1]:
        raise ValueError(f"{len(names)} names for {Y.shape[1]} columns")
    return Y, names


def _minnesota_dummies(sig, lags, nx, tau, decay, omega) -> tuple[np.ndarray, np.ndarray]:
    """Dummy observations of Sims's ``varprior``: rows of ``y`` and of ``x``."""
    n = sig.size
    ys, xs = [], []
    for j in range(n):
        for lag in range(1, lags + 1):
            y = np.zeros(n)
            x = np.zeros(n * lags + nx)
            if lag == 1:
                y[j] = tau * sig[j]
            x[(lag - 1) * n + j] = tau * lag ** decay * sig[j]
            ys.append(y)
            xs.append(x)
    for _ in range(int(omega)):          # MATLAB's 1:omega
        for j in range(n):
            y = np.zeros(n)
            y[j] = sig[j]
            ys.append(y)
            xs.append(np.zeros(n * lags + nx))
    return np.array(ys).reshape(-1, n), np.array(xs).reshape(-1, n * lags + nx)


def _regression(Y, lags, nx, dummies, lambda_, mu) -> tuple[np.ndarray, np.ndarray]:
    """Sims's ``rfvar3`` on data ``Y`` and the prior's dummies: the stacked ``(y, X)``."""
    T, n = Y.shape
    rows = [np.concatenate([Y[t - lags:t][::-1].ravel(), np.ones(nx)]) for t in range(lags, T)]
    X = np.array(rows).reshape(-1, n * lags + nx)
    y = Y[lags:]
    ydum, xdum = dummies
    y, X = np.vstack([y, ydum]), np.vstack([X, xdum])
    ybar = Y[:lags].mean(axis=0)
    if lambda_ != 0:
        xbar = np.ones(nx) if lambda_ > 0 else np.zeros(nx)
        lam = abs(lambda_)
        y = np.vstack([y, lam * ybar])
        X = np.vstack([X, lam * np.concatenate([np.tile(ybar, lags), xbar])])
    if mu > 0:
        y = np.vstack([y, mu * np.diag(ybar)])
        X = np.vstack([X, mu * np.hstack([np.tile(np.diag(ybar), lags), np.zeros((n, nx))])])
    return y, X


def _niw(y, X, n, lags, nx, flat) -> NormalInverseWishart:
    """The posterior of the regression ``y = X Phi + u`` under a flat prior, as ``rfvar3``."""
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    W = Vt.T / s
    Phi = W @ (U.T @ y)
    u = y - X @ Phi
    return NormalInverseWishart(Phi=Phi, XXi=W @ W.T, S=u.T @ u,
                                df=float(len(y) - n * lags - nx - flat * (n + 1)))


def bvar(data: Mapping[str, ArrayLike] | ArrayLike, lags: int, *,
         observables: Sequence[str] | None = None, first_obs: int = 1, presample: int = 0,
         nobs: int | None = None, tau: float = 3.0, decay: float = 0.5, lambda_: float = 5.0,
         mu: float = 2.0, omega: int = 1, flat: bool = False, train: int = 0,
         constant: bool = True, prefilter: bool = False) -> BVAR:
    """A VAR with ``lags`` lags and the Sims–Zha (1998) prior, as Dynare's ``bvar_density``.

    Parameters
    ----------
    data : mapping of series, or ``(T, n)`` array
        The observed variables, one column each, in the units the VAR is
        written in (log them first, if the VAR is in logs).
    lags : int
    observables : the names, and their order. By default the mapping's keys, or
        ``y1, y2, ...`` for an array.
    first_obs, presample, nobs : Dynare's options of the same names: the first
        observation used (counted from 1), the observations after it that start
        the VAR without entering the marginal density, and the number of
        observations from ``first_obs`` on (by default, up to the end).
    tau, decay, lambda_, mu, omega, flat, train : Dynare's ``bvar_prior_tau``,
        ``_decay``, ``_lambda``, ``_mu``, ``_omega``, ``_flat`` and ``_train``,
        with Dynare's defaults (``lambda`` is a Python keyword). ``lambda_ < 0``
        leaves the constant out of the co-persistence dummy, ``flat=True`` uses a
        flat prior on ``Sigma`` instead of Jeffreys', and ``train`` observations
        before ``first_obs + presample`` form the training sample.
    constant : include a constant (Dynare's ``noconstant`` is ``constant=False``).
    prefilter : subtract the sample mean of every series first, without a
        constant, as Dynare's ``prefilter``.

    Returns
    -------
    BVAR
    """
    Y, names = _matrix(data, observables)
    Tall, n = Y.shape
    lags = int(lags)
    if lags < 1:
        raise ValueError("lags must be a positive whole number")
    if nobs is None:
        nobs = Tall - first_obs + 1
    if first_obs < 1 or first_obs + nobs - 1 > Tall:
        raise ValueError(f"first_obs={first_obs} and nobs={nobs} do not fit in {Tall} observations")
    if first_obs + presample <= lags:
        raise ValueError("first_obs + presample must exceed lags, to start the VAR")
    if first_obs + presample - train <= lags:
        raise ValueError("first_obs + presample - train must exceed lags, to start the VAR")
    start = first_obs + presample - train - lags - 1           # 0-based, inclusive
    stop = first_obs + nobs - 1                                # 0-based, exclusive
    Y = Y.copy()
    if prefilter:
        Y[start:stop] -= Y[start:stop].mean(axis=0)
    # the scale of each variable: the lags + 1 observations that start the sample
    sig = Y[first_obs + presample - lags - 1:first_obs + presample].std(axis=0, ddof=1)
    nx = 0 if (prefilter or not constant) else 1
    dummies = _minnesota_dummies(sig, lags, nx, tau, decay, omega)
    ydata = Y[start:stop]
    posterior = _niw(*_regression(ydata, lags, nx, dummies, lambda_, mu), n, lags, nx, flat)
    prior = _niw(*_regression(ydata[:train + lags], lags, nx, dummies, lambda_, mu),
                 n, lags, nx, flat)
    if prior.df < n:
        raise ValueError(f"the prior has {prior.df:g} degrees of freedom for {n} variables: "
                         f"lengthen the training sample")
    m = int(round(posterior.df - prior.df))
    log_marginal = posterior.log_integral() - prior.log_integral() - 0.5 * n * m * _LOG2PI
    hyper = dict(tau=tau, decay=decay, lambda_=lambda_, mu=mu, omega=omega, flat=flat,
                 train=train)
    return BVAR(names=names, lags=lags, constant=bool(nx), prior=prior, posterior=posterior,
                log_marginal=float(log_marginal), nobs=m, initial=ydata[-lags:].copy(),
                realized=Y[stop:].copy(), hyperparameters=hyper)


def bvar_density(data: Mapping[str, ArrayLike] | ArrayLike, max_lags: int,
                 **kwargs: Any) -> dict:
    """``{p: log marginal density}`` for ``p = 1..max_lags``, Dynare's ``bvar_density(max_lags)``.

    ``kwargs`` go to :func:`bvar`. Every lag length is scored on the same
    observations only if ``first_obs + presample`` exceeds ``max_lags`` (Dynare
    scores each on the observations from ``first_obs + presample`` on).
    """
    return {p: bvar(data, p, **kwargs).log_marginal for p in range(1, int(max_lags) + 1)}


# ===================================================================== DSGE-VAR
def _var_sample_moments(Y: np.ndarray, lags: int, constant: bool) -> tuple:
    """``(Y'Y, Y'X, X'X)`` of the VAR on ``Y``, its first ``lags`` rows the initial values.

    Dynare's ``var_sample_moments``, with the regressors in this module's order.
    """
    T, n = Y.shape
    rows = [np.concatenate([Y[t - lags:t][::-1].ravel(), np.ones(int(constant))])
            for t in range(lags, T)]
    X = np.array(rows).reshape(-1, n * lags + int(constant))
    y = Y[lags:]
    return y.T @ y, y.T @ X, X.T @ X


def _dsge_var_moments(space, lags: int, constant: bool) -> tuple:
    """``(GYY, GYX, GXX)``, the population moments of the VAR's ``y_t`` and ``x_t``.

    From the state space of a first-order solution: ``Gamma_j = E[y_t y_{t-j}'] =
    Z T^j V Z'`` with ``V`` the stationary variance of the states, plus ``d d'``
    when the VAR has a constant, whose mean is ``d``. Dynare's
    ``dsge_var_likelihood`` builds the same matrices.
    """
    from scipy.linalg import solve_discrete_lyapunov
    if np.any(np.abs(np.linalg.eigvals(space.T)) >= 1.0):
        raise ValueError("the DSGE prior needs stationary states")
    RQR = space.R @ space.Sigma @ space.R.T
    V = solve_discrete_lyapunov(space.T, 0.5 * (RQR + RQR.T))
    V = 0.5 * (V + V.T)
    d = space.d if constant else np.zeros(space.n_obs)
    dd = np.outer(d, d)
    G = [space.Z @ V @ space.Z.T + dd]
    TjV = V
    for _ in range(lags):
        TjV = space.T @ TjV
        G.append(space.Z @ TjV @ space.Z.T + dd)
    n = space.n_obs
    k = n * lags + int(constant)
    GYX = np.zeros((n, k))
    GXX = np.zeros((k, k))
    for i in range(lags):
        GYX[:, i * n:(i + 1) * n] = G[i + 1]
        for j in range(lags):
            GXX[i * n:(i + 1) * n, j * n:(j + 1) * n] = G[j - i] if j >= i else G[i - j].T
    if constant:
        GYX[:, -1] = d
        GXX[:-1, -1] = np.tile(d, lags)
        GXX[-1, :-1] = np.tile(d, lags)
        GXX[-1, -1] = 1.0
    return G[0], GYX, GXX


def _logdet_pd(A: np.ndarray) -> float:
    """``log det A`` for a symmetric positive definite ``A``; raises otherwise."""
    return float(2.0 * np.sum(np.log(np.diag(np.linalg.cholesky(0.5 * (A + A.T))))))


def _dsge_var_log_likelihood(population: tuple, sample: tuple, nobs: int, weight: float) -> float:
    """``log p(Y | theta, lambda)``, the DSGE-VAR likelihood with the VAR's parameters integrated out.

    Formula (A.2) of Del Negro and Schorfheide (2004), as Dynare's
    ``dsge_var_likelihood`` writes it, from the population moments
    ``(GYY, GYX, GXX)``, the sample moments ``(Y'Y, Y'X, X'X)``, the number of
    observations ``nobs`` and the weight ``lambda``. With ``lambda = inf`` it is
    the Gaussian likelihood of the VAR at ``Phi*`` and ``Sigma*``, the VAR the
    model implies. ``-inf`` when ``lambda T`` is below ``k + n``, where the prior
    is improper, or when a covariance is not positive definite.
    """
    GYY, GYX, GXX = population
    mYY, mYX, mXX = sample
    n, k = GYX.shape
    T, lam = float(nobs), float(weight)
    if lam < (k + n) / T:
        return -np.inf
    try:
        Phi_star = np.linalg.solve(GXX, GYX.T)
        Sigma_star = GYY - GYX @ Phi_star
        logdet_star = _logdet_pd(Sigma_star)
        if np.isinf(lam):
            resid = mYY - Phi_star.T @ mYX.T - mYX @ Phi_star + Phi_star.T @ mXX @ Phi_star
            return float(-0.5 * T * (logdet_star + n * _LOG2PI)
                         - 0.5 * np.trace(np.linalg.solve(Sigma_star, resid)))
        lT = lam * T
        A = lT * GXX + mXX
        B = lT * GYX + mYX
        Ssum = lT * GYY + mYY - B @ np.linalg.solve(A, B.T)       # (1 + lambda) T Sigma~
        i = np.arange(1, n + 1)
        ll = (-0.5 * n * _logdet_pd(A)
              - 0.5 * ((1 + lam) * T - k) * _logdet_pd(Ssum)
              + 0.5 * n * (k * math.log(lT) + _logdet_pd(GXX))
              + 0.5 * (lT - k) * (n * math.log(lT) + logdet_star)
              - 0.5 * n * T * _LOG2PI
              + 0.5 * math.log(2.0) * n * ((1 + lam) * T - k)
              - 0.5 * math.log(2.0) * n * (lT - k)
              + np.sum(gammaln(0.5 * ((1 + lam) * T - k + 1 - i)))
              - np.sum(gammaln(0.5 * (lT - k + 1 - i))))
    except np.linalg.LinAlgError:
        return -np.inf
    return float(ll) if np.isfinite(ll) else -np.inf


def _dsge_var_prior(population: tuple, nobs: int, weight: float) -> NormalInverseWishart:
    """The DSGE prior of the VAR's parameters, ``lambda T`` artificial observations.

    ``Phi* = GXX^-1 GYX'``, ``S = lambda T Sigma*`` with ``Sigma* = GYY - GYX Phi*``,
    ``XXi = (lambda T GXX)^-1`` and ``df = lambda T - k`` (Del Negro and
    Schorfheide, 2004, equations 22 to 25).
    """
    GYY, GYX, GXX = population
    n, k = GYX.shape
    lT = float(weight) * nobs
    if not np.isfinite(lT):
        raise ValueError("with an infinite weight the prior is a point mass at Phi*, Sigma*")
    Phi = np.linalg.solve(GXX, GYX.T)
    return NormalInverseWishart(Phi=Phi, XXi=np.linalg.inv(lT * GXX),
                                S=lT * (GYY - GYX @ Phi), df=lT - k)


def _dsge_var_posterior(population: tuple, sample: tuple, nobs: int,
                       weight: float) -> NormalInverseWishart:
    """The posterior of the VAR's parameters given the model's: Dynare's ``PHI_tilde``,
    ``SIGMA_u_tilde`` and ``iXX``.

    ``Phi~ = (lambda T GXX + X'X)^-1 (lambda T GYX + Y'X)'``, ``S = (1 + lambda) T
    Sigma~``, ``XXi = (lambda T GXX + X'X)^-1`` and ``df = (1 + lambda) T - k``
    (Del Negro and Schorfheide, 2004, equations 28 and 29).
    """
    GYY, GYX, GXX = population
    mYY, mYX, mXX = sample
    n, k = GYX.shape
    lT = float(weight) * nobs
    if not np.isfinite(lT):
        raise ValueError("with an infinite weight the posterior is a point mass at Phi*, Sigma*")
    XXi = np.linalg.inv(lT * GXX + mXX)
    B = lT * GYX + mYX
    Phi = XXi @ B.T
    return NormalInverseWishart(Phi=Phi, XXi=XXi, S=lT * GYY + mYY - B @ Phi,
                                df=(1 + float(weight)) * nobs - k)
