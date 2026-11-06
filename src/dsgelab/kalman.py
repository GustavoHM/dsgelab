"""The linear state space of a first-order solution: filter, smoother, likelihood.

The state equation is the solution itself, in this package's dating::

    k_{t+1} = P k_t + Q eps_{t+1},        eps ~ N(0, Sigma)

and the observation equation reads the observed variables off ``[k; F k]``::

    y_t = d + Z k_t + e_t,                e ~ N(0, H)

Each observable is either a LEVEL (``d`` is its steady state, ``Z`` carries the
scale of the rescaled deviation) or a LOG (``d`` is the log of its steady state,
``Z`` the proportional deviation). The second is what Dynare's ``loglinear`` and
``logdata`` options amount to: with this package's proportional coordinates the
log-linear system is the same system, exactly, at first order.

Observation equations are model equations. An observed growth rate, a measured
inflation rate, a variable observed with a different scaling -- each is a
variable of the model with its own equation, as in a Dynare ``.mod``; what is
declared here is only which variables are observed and with how much
measurement error. Missing observations are ``nan`` and are skipped.

The filter
----------
Stationary states start at their unconditional distribution. States with a unit
root -- a price level, a random-walk technology -- cannot, and start DIFFUSE: the
exact initialisation of Koopman and Durbin (2000), in which the variance is
``P_* + kappa P_inf`` with ``kappa`` going to infinity, and the first
observations are spent learning the level. The split between the two parts is
read off an ordered Schur decomposition of the transition.

The filter processes observations one at a time (the univariate treatment of
Koopman and Durbin, 2000) when the initialisation is diffuse; the stationary
case uses the multivariate filter. The two give the same likelihood, which is a
test. A non-diagonal ``H`` is made diagonal by an LDL transformation of the
observations first.

The smoother
------------
The fixed-interval smoother of Durbin and Koopman (2012), in the same univariate
form, exact through the diffuse periods: smoothed states, smoothed innovations
and smoothed measurement errors. Smoothed variances are returned outside the
diffuse periods.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Mapping, Sequence, TypeAlias

import numpy as np
from numpy.typing import ArrayLike

from .model import Covariance

if TYPE_CHECKING:
    from .solution import FirstOrderSolution

__all__ = ["StateSpace", "FilterResult", "SmootherResult", "Forecast", "state_space",
           "kalman_filter", "kalman_smoother", "shock_decomposition",
           "realtime_shock_decomposition", "forecast", "low_frequency",
           "initial_condition_decomposition", "annualise_decomposition",
           "ConditionalForecast", "conditional_forecast_bands"]

_LOG2PI = np.log(2.0 * np.pi)

#: Observed data: a mapping of observable name to series, or a ``(T, n_obs)``
#: array in the order of the observables.
Data: TypeAlias = Mapping[str, ArrayLike] | ArrayLike


@dataclass
class StateSpace:
    """``k_{t+1} = T k_t + R eps_{t+1}``, ``y_t = d + Z k_t + e_t``."""

    T: np.ndarray
    R: np.ndarray
    Sigma: np.ndarray
    Z: np.ndarray
    d: np.ndarray
    H: np.ndarray
    observables: list
    states: list
    shocks: list
    transforms: list
    solution: FirstOrderSolution | None = field(default=None, repr=False)
    #: per observable, ``(how, every)`` for a temporal aggregate, or None
    aggregation: list | None = None

    @property
    def n_states(self) -> int:
        return self.T.shape[0]

    @property
    def n_model_states(self) -> int:
        """The model's own states: the first block of the state vector, before any lags."""
        return self.T.shape[0] if self.solution is None else self.solution.model.n_states

    @property
    def n_obs(self) -> int:
        return self.Z.shape[0]

    def data_matrix(self, data: Data) -> np.ndarray:
        """``(T, n_obs)`` from a mapping of series or an array, in the observables' order."""
        if isinstance(data, Mapping):
            cols = [np.asarray(data[n], dtype=float).ravel() for n in self.observables]
            n = {c.size for c in cols}
            if len(n) != 1:
                raise ValueError("every observed series must have the same length")
            return np.column_stack(cols)
        Y = np.asarray(data, dtype=float)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        if Y.shape[1] != self.n_obs:
            raise ValueError(f"data has {Y.shape[1]} columns for {self.n_obs} observables")
        return Y


def state_space(sol: FirstOrderSolution, observables: Sequence[str], *,
                transforms: str | Mapping[str, str] | None = None,
                measurement: Mapping[str, float] | ArrayLike | None = None,
                cov: Covariance = None,
                aggregation: Mapping[str, tuple[str, int]] | None = None) -> StateSpace:
    """The state space of a :class:`~dsgelab.FirstOrderSolution`.

    Parameters
    ----------
    observables : sequence of str
        Model variables (states or jumps) that are observed.
    transforms : mapping or str, optional
        ``"level"`` (default) or ``"log"`` per observable, or one value for all.
        ``"log"`` means the data are logs of the variable -- Dynare's
        ``loglinear`` with ``logdata`` -- and needs a nonzero steady state.
    measurement : mapping, scalar or array, optional
        Standard deviations of the measurement errors: ``{name: sd}``, one number
        for all, or a full covariance matrix. None by default.
    cov : shock covariance, as for :func:`dsgelab.solve` (identity if absent).
    aggregation : mapping, optional
        Observables seen at a lower frequency than the model's, ``{name: (how,
        every)}``: the data are the ``"mean"`` or the ``"sum"`` of the variable
        over the last ``every`` periods, in the units of its transform (a mean of
        logs for a ``"log"`` observable), and are missing (NaN) in the periods
        between observations (see :func:`low_frequency`). The state then carries
        ``every - 1`` lags of the model's states, as an auxiliary variable
        ``(y + y(-1) + y(-2) + y(-3))/4`` does in a ``.mod`` file. A variable
        observed at the end of each low-frequency period (a stock) needs no
        aggregation, only the missing values.
    """
    from .solution import shock_covariance

    model = sol.model
    observables = list(observables)
    nk = model.n_states
    if isinstance(transforms, str) or transforms is None:
        tr = [transforms or "level"] * len(observables)
    else:
        tr = [transforms.get(n, "level") for n in observables]
    Z = np.zeros((len(observables), nk))
    d = np.zeros(len(observables))
    center, scale = sol.coords.center, sol.coords.scale
    for i, name in enumerate(observables):
        if name in model.states:
            row = np.zeros(nk)
            row[model.states.index(name)] = 1.0
        elif name in model.jumps:
            row = np.asarray(sol.F[model.jumps.index(name)], dtype=float)
        else:
            raise KeyError(f"{name!r} is not a variable of {model.name!r}")
        j = model.position(name)
        if tr[i] == "level":
            Z[i], d[i] = scale[j] * row, center[j]
        elif tr[i] == "log":
            if center[j] <= 0:
                raise ValueError(f"{name!r} has steady state {center[j]}: a log "
                                 f"observation needs a positive one")
            if not np.isclose(scale[j], abs(center[j])):
                raise ValueError(f"{name!r} is not in proportional coordinates")
            Z[i], d[i] = row, np.log(center[j])
        else:
            raise ValueError(f"transform must be 'level' or 'log', got {tr[i]!r}")
    if measurement is None:
        H = np.zeros((len(observables), len(observables)))
    elif isinstance(measurement, Mapping):
        H = np.diag([float(measurement.get(n, 0.0)) ** 2 for n in observables])
    elif np.isscalar(measurement):
        H = float(measurement) ** 2 * np.eye(len(observables))
    else:
        H = np.asarray(measurement, dtype=float)
        if H.shape != (len(observables), len(observables)):
            raise ValueError("measurement covariance has the wrong shape")
    Sigma = shock_covariance(model.shocks, cov)
    Q = sol.Q if sol.Q is not None else np.zeros((nk, model.n_shocks))
    T, R = np.asarray(sol.P, float), np.asarray(Q, float)
    states = list(model.states)
    agg = _aggregation_spec(observables, tr, aggregation)
    if agg is not None:
        T, R, Z, d = _aggregate(T, R, Z, d, agg)
        lags = T.shape[0] // nk - 1
        states += [f"{s}(-{j})" for j in range(1, lags + 1) for s in model.states]
    return StateSpace(T=T, R=R, Sigma=Sigma, Z=Z, d=d, H=H, observables=observables,
                      states=states, shocks=list(model.shocks), transforms=tr, solution=sol,
                      aggregation=agg)


def _aggregation_spec(observables, transforms, aggregation) -> list | None:
    """Per observable, ``(how, every)`` or None; None for the whole list when nothing aggregates."""
    if not aggregation:
        return None
    unknown = set(aggregation) - set(observables)
    if unknown:
        raise ValueError(f"aggregation of {sorted(unknown)}, which are not observed")
    out = []
    for name, tr in zip(observables, transforms):
        if name not in aggregation:
            out.append(None)
            continue
        how, every = aggregation[name]
        if how not in ("mean", "sum"):
            raise ValueError(f"aggregation is 'mean' or 'sum', got {how!r}")
        if int(every) != every or every < 1:
            raise ValueError(f"aggregation of {name!r} needs a positive whole number of periods")
        if how == "sum" and tr == "log":
            raise ValueError(f"{name!r} is observed in logs: a sum of logs is not an aggregate; "
                             f"use 'mean' (and shift the data by log(every) for a log of a sum)")
        out.append((how, int(every)))
    return out if any(a is not None for a in out) else None


def _aggregate(T, R, Z, d, agg, *, derivative=False):
    """Stack ``every - 1`` lags of the states and aggregate the observation rows.

    Linear in ``(T, R, Z, d)`` apart from the identity blocks that shift the
    lags, so with ``derivative=True`` (no identity) it carries their derivatives.
    """
    nk = T.shape[0]
    L = max(a[1] for a in agg if a is not None) - 1
    n = nk * (L + 1)
    Ta = np.zeros((n, n))
    Ta[:nk, :nk] = T
    for j in range(1, L + 1):
        if not derivative:
            Ta[j * nk:(j + 1) * nk, (j - 1) * nk:j * nk] = np.eye(nk)
    Ra = np.zeros((n, R.shape[1]))
    Ra[:nk] = R
    Za = np.zeros((Z.shape[0], n))
    da = np.array(d, dtype=float)
    for i, a in enumerate(agg):
        if a is None:
            Za[i, :nk] = Z[i]
            continue
        how, every = a
        w = 1.0 / every if how == "mean" else 1.0
        for j in range(every):
            Za[i, j * nk:(j + 1) * nk] = w * Z[i]
        if how == "sum":
            da[i] = every * d[i]
    return Ta, Ra, Za, da


def low_frequency(values: ArrayLike, every: int, *, periods: int | None = None,
                  offset: int | None = None) -> np.ndarray:
    """Low-frequency observations on the model's grid, with NaN in between.

    ``values[j]`` lands in period ``offset + j * every``; by default ``offset``
    is ``every - 1``, the last period of the first low-frequency period, where an
    aggregate over the last ``every`` periods is complete. ``periods`` is the
    length of the result, by default just long enough for the values.
    """
    v = np.asarray(values, dtype=float).ravel()
    every = int(every)
    if every < 1:
        raise ValueError("every must be a positive whole number")
    offset = every - 1 if offset is None else int(offset)
    n = offset + every * (v.size - 1) + 1 if periods is None else int(periods)
    out = np.full(n, np.nan)
    pos = offset + every * np.arange(v.size)
    keep = pos < n
    out[pos[keep]] = v[keep]
    return out


# ------------------------------------------------------------------ initial state
def _initial(ss: StateSpace, init: str, qz_criterium: float = 1 - 1e-6):
    """Mean, stationary variance ``P_*`` and diffuse part ``P_inf`` of the first state."""
    from scipy.linalg import schur, solve_discrete_lyapunov

    n = ss.n_states
    RQR = ss.R @ ss.Sigma @ ss.R.T
    RQR = 0.5 * (RQR + RQR.T)
    eig = np.abs(np.linalg.eigvals(ss.T)) if n else np.zeros(0)
    unit = bool(np.any(eig >= qz_criterium))
    if init == "stationary" or (init == "auto" and not unit):
        if unit:
            raise ValueError("a state has a unit root: use init='diffuse' (or 'auto')")
        P = solve_discrete_lyapunov(ss.T, RQR) if n else np.zeros((0, 0))
        return np.zeros(n), 0.5 * (P + P.T), np.zeros((n, n))
    if init not in ("diffuse", "auto"):
        raise ValueError("init must be 'auto', 'stationary' or 'diffuse'")
    # Ordered real Schur form with the UNIT roots first, as Dynare orders it: the
    # first Schur vectors then span the unit-root (right) invariant subspace, and the
    # stable coordinates evolve on their own, so their stationary variance is exact.
    # P_inf is the identity on the unit-root block.
    S, U, sdim = schur(ss.T, output="real",
                       sort=lambda re, im: np.hypot(re, im) >= qz_criterium)
    nu = sdim
    Ub = U.T @ RQR @ U
    P = np.zeros((n, n))
    if nu < n:
        P[nu:, nu:] = solve_discrete_lyapunov(S[nu:, nu:], Ub[nu:, nu:])
    Pinf = np.zeros((n, n))
    Pinf[:nu, :nu] = np.eye(nu)
    P_star = U @ P @ U.T
    P_inf = U @ Pinf @ U.T
    return np.zeros(n), 0.5 * (P_star + P_star.T), 0.5 * (P_inf + P_inf.T)


# --------------------------------------------------------------------------- filter
@dataclass
class FilterResult:
    """What the filter leaves behind, enough for the smoother."""

    loglik: float
    loglik_t: np.ndarray            # contribution of each date
    a_pred: np.ndarray              # (T+1, n) state mean before date t's data
    P_pred: np.ndarray              # (T+1, n, n)
    a_filt: np.ndarray              # (T, n) after date t's data
    P_filt: np.ndarray
    diffuse_periods: int
    univariate: bool
    store: list = field(default_factory=list, repr=False)     # per (t, i) quantities
    Y: np.ndarray | None = field(default=None, repr=False)
    space: StateSpace | None = field(default=None, repr=False)


def _ldl_transform(ss: StateSpace, Y):
    """Diagonalise H: y* = L^{-1} y. Returns transformed (Z, d, h, Y)."""
    H = ss.H
    if np.allclose(H, np.diag(np.diag(H))):
        return ss.Z, ss.d, np.diag(H).copy(), Y, None
    from scipy.linalg import ldl
    lu, D, perm = ldl(H, lower=True)
    Linv = np.linalg.inv(lu)
    Zs = Linv @ ss.Z
    ds = Linv @ ss.d
    Ys = (Linv @ np.nan_to_num(Y.T)).T
    Ys[np.isnan(Y).any(axis=1)] = np.nan           # partial missingness is not mixed here
    return Zs, ds, np.diag(D).copy(), Ys, Linv


def kalman_filter(ss: StateSpace, data: Data, *, init: str = "auto", univariate: bool | None = None,
                  tol: float = 1e-12) -> FilterResult:
    """Kalman filter and Gaussian log likelihood.

    ``init``: ``"auto"`` (stationary if every root is inside the unit circle,
    diffuse for the others), ``"stationary"`` or ``"diffuse"`` (every root
    with modulus 1 or more is diffuse). ``univariate`` forces the observation-
    by-observation filter; it is used whenever the start is diffuse.
    """
    Y = ss.data_matrix(data)
    a0, P0, Pinf0 = _initial(ss, init)
    diffuse = bool(np.any(Pinf0 != 0))
    if univariate is None:
        univariate = diffuse or bool(np.isnan(Y).any())
    if univariate:
        return _univariate(ss, Y, a0, P0, Pinf0, tol)
    return _multivariate(ss, Y, a0, P0, tol)


def _multivariate(ss, Y, a, P, tol):
    Tn, n = Y.shape[0], ss.n_states
    T, Z, d, H = ss.T, ss.Z, ss.d, ss.H
    RQR = ss.R @ ss.Sigma @ ss.R.T
    a_pred = np.zeros((Tn + 1, n)); P_pred = np.zeros((Tn + 1, n, n))
    a_filt = np.zeros((Tn, n)); P_filt = np.zeros((Tn, n, n))
    ll_t = np.zeros(Tn)
    store = []
    for t in range(Tn):
        a_pred[t], P_pred[t] = a, P
        v = Y[t] - d - Z @ a
        F = Z @ P @ Z.T + H
        F = 0.5 * (F + F.T)
        sign, logdet = np.linalg.slogdet(F)
        if sign <= 0:
            raise np.linalg.LinAlgError(f"the forecast-error variance is singular at date {t}")
        Finv_v = np.linalg.solve(F, v)
        ll_t[t] = -0.5 * (len(v) * _LOG2PI + logdet + v @ Finv_v)
        K = np.linalg.solve(F, Z @ P).T          # P Z' F^{-1}
        a = a + K @ v
        P = P - K @ Z @ P
        P = 0.5 * (P + P.T)
        a_filt[t], P_filt[t] = a, P
        store.append((v, F, K))
        a = T @ a
        P = T @ P @ T.T + RQR
        P = 0.5 * (P + P.T)
    a_pred[Tn], P_pred[Tn] = a, P
    return FilterResult(loglik=float(ll_t.sum()), loglik_t=ll_t, a_pred=a_pred, P_pred=P_pred,
                        a_filt=a_filt, P_filt=P_filt, diffuse_periods=0, univariate=False,
                        store=store, Y=Y, space=ss)


def _univariate(ss, Y, a, P, Pinf, tol):
    Z, d, h, Ys, _ = _ldl_transform(ss, Y)
    Tn, n = Ys.shape[0], ss.n_states
    T = ss.T
    RQR = ss.R @ ss.Sigma @ ss.R.T
    a_pred = np.zeros((Tn + 1, n)); P_pred = np.zeros((Tn + 1, n, n))
    a_filt = np.zeros((Tn, n)); P_filt = np.zeros((Tn, n, n))
    ll_t = np.zeros(Tn)
    store = []
    diffuse_until = 0
    for t in range(Tn):
        a_pred[t], P_pred[t] = a, P
        in_diffuse = bool(np.max(np.abs(Pinf), initial=0.0) > tol)
        if in_diffuse:
            diffuse_until = t + 1
        steps = []
        for i in range(Z.shape[0]):
            yi = Ys[t, i]
            if np.isnan(yi):
                steps.append(None)
                continue
            z = Z[i]
            v = yi - d[i] - z @ a
            M = P @ z
            F = z @ M + h[i]
            if in_diffuse:
                Mi = Pinf @ z
                Fi = z @ Mi
                if Fi > tol:
                    K0 = Mi / Fi
                    K1 = M / Fi - Mi * F / Fi ** 2
                    a = a + K0 * v
                    P = P + np.outer(Mi, Mi) * F / Fi ** 2 - (np.outer(M, Mi) + np.outer(Mi, M)) / Fi
                    Pinf = Pinf - np.outer(Mi, Mi) / Fi
                    ll_t[t] += -0.5 * (_LOG2PI + np.log(Fi))
                    steps.append(("inf", z, v, Fi, F, K0, K1, h[i]))
                    continue
            if F > tol:
                K = M / F
                a = a + K * v
                P = P - np.outer(M, M) / F
                ll_t[t] += -0.5 * (_LOG2PI + np.log(F) + v * v / F)
                steps.append(("std", z, v, F, K, h[i]))
            else:
                steps.append(None)
        P = 0.5 * (P + P.T)
        a_filt[t], P_filt[t] = a, P
        store.append(steps)
        a = T @ a
        P = T @ P @ T.T + RQR
        P = 0.5 * (P + P.T)
        if in_diffuse:
            Pinf = T @ Pinf @ T.T
            Pinf = 0.5 * (Pinf + Pinf.T)
    a_pred[Tn], P_pred[Tn] = a, P
    return FilterResult(loglik=float(ll_t.sum()), loglik_t=ll_t, a_pred=a_pred, P_pred=P_pred,
                        a_filt=a_filt, P_filt=P_filt, diffuse_periods=diffuse_until,
                        univariate=True, store=store, Y=Y, space=ss)


# ------------------------------------------------------------------------- smoother
@dataclass
class SmootherResult:
    """Smoothed states, innovations and measurement errors, given all the data."""

    states: np.ndarray              # (T, n_k) hat-deviations, k_t given y_1..y_T
    state_variance: np.ndarray      # (T, n_k, n_k); nan in the diffuse periods
    shocks: np.ndarray              # (T, n_eps): eps_t, the innovation INTO date t (row 0: none)
    measurement: np.ndarray         # (T, n_obs) smoothed measurement errors
    space: StateSpace = field(repr=False, default=None)

    def variables(self) -> dict:
        """Smoothed paths of every model variable, in levels."""
        sol = self.space.solution
        paths = sol.variables(self.states[:, :self.space.n_model_states])
        return sol.to_levels(paths)

    def shocks_by_name(self) -> dict:
        return {e: self.shocks[:, j] for j, e in enumerate(self.space.shocks)}


def kalman_smoother(ss: StateSpace, data: Data | None = None, *,
                    filtered: FilterResult | None = None,
                    init: str = "auto") -> SmootherResult:
    """Fixed-interval smoother (Durbin and Koopman, 2012), exact through a diffuse start."""
    fr = filtered if filtered is not None else kalman_filter(ss, data, init=init)
    Tn, n = fr.a_filt.shape
    T = ss.T
    Sigma_R = ss.Sigma @ ss.R.T
    states = np.zeros((Tn, n))
    var = np.full((Tn, n, n), np.nan)
    shocks = np.zeros((Tn, ss.R.shape[1]))
    meas = np.zeros((Tn, ss.n_obs))
    r0 = np.zeros(n)
    r1 = np.zeros(n)
    N = np.zeros((n, n))
    for t in range(Tn - 1, -1, -1):
        if fr.univariate:
            steps = fr.store[t]
            for i in range(len(steps) - 1, -1, -1):
                st = steps[i]
                if st is None:
                    continue
                if st[0] == "std":
                    _, z, v, F, K, hi = st
                    L = np.eye(n) - np.outer(K, z)
                    meas[t, i] = hi * (v / F - K @ r0)
                    r0 = z * v / F + L.T @ r0
                    r1 = L.T @ r1
                    N = np.outer(z, z) / F + L.T @ N @ L
                else:
                    _, z, v, Fi, F, K0, K1, hi = st
                    L0 = np.eye(n) - np.outer(K0, z)
                    L1 = -np.outer(K1, z)
                    meas[t, i] = -hi * (K0 @ r0)
                    r1 = z * v / Fi + L0.T @ r1 + L1.T @ r0
                    r0 = L0.T @ r0
                    N = L0.T @ N @ L0
            Ppred = fr.P_pred[t]
            states[t] = fr.a_pred[t] + Ppred @ r0
            if t >= fr.diffuse_periods:
                var[t] = Ppred - Ppred @ N @ Ppred
            else:
                states[t] = states[t] + _pinf_at(fr, t) @ r1
        else:
            v, F, K = fr.store[t]
            Z = ss.Z
            Finv = np.linalg.inv(F)
            L = np.eye(n) - K @ Z
            meas[t] = ss.H @ (Finv @ v - K.T @ r0)
            r0 = Z.T @ Finv @ v + L.T @ r0
            N = Z.T @ Finv @ Z + L.T @ N @ L
            Ppred = fr.P_pred[t]
            states[t] = fr.a_pred[t] + Ppred @ r0
            var[t] = Ppred - Ppred @ N @ Ppred
        # r0 is now what all the data say about k_t (Durbin and Koopman's r_{t-1}), and
        # E[eps_t | all data] = Sigma R' r_{t-1}: the innovation INTO date t
        if t > 0:
            shocks[t] = Sigma_R @ r0
        # to date t-1
        r0 = T.T @ r0
        r1 = T.T @ r1
        N = T.T @ N @ T
    return SmootherResult(states=states, state_variance=var, shocks=shocks, measurement=meas,
                          space=ss)


def _pinf_at(fr: FilterResult, t: int) -> np.ndarray:
    """Diffuse variance part before date t's data, rebuilt by rerunning the recursion."""
    if not hasattr(fr, "_pinf_cache"):
        ss = fr.space
        _, _, Pinf = _initial(ss, "diffuse")
        Z, d, h, Ys, _ = _ldl_transform(ss, fr.Y)
        cache = []
        tol = 1e-12
        for s in range(fr.diffuse_periods):
            cache.append(Pinf.copy())
            for i in range(Z.shape[0]):
                if np.isnan(Ys[s, i]):
                    continue
                z = Z[i]
                Mi = Pinf @ z
                Fi = z @ Mi
                if Fi > tol:
                    Pinf = Pinf - np.outer(Mi, Mi) / Fi
            Pinf = ss.T @ Pinf @ ss.T.T
        fr._pinf_cache = cache
    return fr._pinf_cache[t]


# ------------------------------------------------------- decompositions, forecasts
def shock_decomposition(smoothed: SmootherResult, *, variables: Sequence[str] | None = None,
                        initial_shock_states: Mapping[str, str] | None = None) -> dict:
    """Each variable's smoothed path as the sum of what each shock did, plus the start.

    The contribution of a shock is the path the solution produces from the
    smoothed innovations of that shock alone, starting from zero; the initial
    condition is what is left -- Dynare's ``shock_decomposition``. Paths are in
    the solution's rescaled deviations (proportional deviations, i.e. log
    deviations, for variables with a nonzero steady state).

    ``initial_shock_states`` maps a state to the shock whose value it carries at
    the first date. A model read from a ``.mod`` file has one such state per
    shock, ``SHK_e``, and Dynare counts its first value as the shock's, not as
    part of the initial condition; :meth:`dsgelab.ModFile.shock_states` gives
    the mapping.

    Returns ``{"shocks": {shock: {var: path}}, "initial": {var: path},
    "total": {var: path}}``.
    """
    ss = smoothed.space
    sol = ss.solution
    k = smoothed.states
    eps = smoothed.shocks
    Tn, n = k.shape
    P, Q = ss.T, ss.R
    names = list(variables) if variables is not None else list(sol.model.names)
    init_map = dict(initial_shock_states or {})
    pieces = {}
    for m, e in enumerate(ss.shocks):
        piece = np.zeros((Tn, n))
        for st, sh in init_map.items():
            if sh == e:
                i = ss.states.index(st)
                piece[0, i] = k[0, i]
        for t in range(1, Tn):
            piece[t] = P @ piece[t - 1] + Q[:, m] * eps[t, m]
        pieces[e] = sol.variables(piece[:, :ss.n_model_states])
    total = sol.variables(k[:, :ss.n_model_states])
    out = {"shocks": {e: {v: pieces[e][v] for v in names} for e in ss.shocks},
           "total": {v: total[v] for v in names}}
    out["initial"] = {v: total[v] - sum(out["shocks"][e][v] for e in ss.shocks) for v in names}
    return out


def realtime_shock_decomposition(space: StateSpace, data: Data, *, init: str = "auto",
                                 variables: Sequence[str] | None = None,
                                 initial_shock_states: Mapping[str, str] | None = None,
                                 first: int = 2, forecast: int = 0,
                                 vintages: Sequence[int] = ()) -> dict:
    """The shock decomposition each date would have shown with the data up to it.

    For every date ``t`` the sample is cut at ``t``, smoothed and decomposed,
    and what is kept is the decomposition AT ``t``: Dynare's
    ``realtime_shock_decomposition`` pool. The first sample has ``first``
    observations and fills the dates up to it, as Dynare's does (its presample
    of one). Comparing the result with the full-sample decomposition shows how
    much of each date's story later data revised.

    Note on the first date: Dynare's real-time decomposition leaves the shock of
    its first period inside the initial condition, while its full-sample
    ``shock_decomposition`` counts it as a shock. Both are reproduced:
    ``initial_shock_states`` is None here by default and set by
    :meth:`dsgelab.ModFile.shock_states` for the full sample.

    Beside ``"shocks"``, ``"initial"`` and ``"total"`` (the pool), the result has
    the other decompositions of Dynare's command, each a mapping of the same
    form. ``"conditional"`` is its one-step conditional pool: at each date, the
    contribution of that date's innovations, with the rest in ``"initial"`` (NaN
    before the first sample). With ``forecast`` (Dynare's ``forecast``) the pool
    runs ``forecast`` dates past the sample, as the last cut forecasts them, and
    ``"forecast"`` is the pool of one-step forecasts: at date ``t + 1``, what the
    cut at ``t`` forecasts, each contribution carried by the solution without new
    shocks (NaN up to the first sample). ``vintages`` lists cuts, each by the
    number of observations in it (Dynare's ``save_realtime``). For each,
    ``"vintages"`` holds the decomposition of that cut over its dates and the
    ``forecast`` after them, ``"forecast_vintages"`` the decomposition of its
    forecast, from its last date on, and ``"conditional_vintages"`` the
    ``forecast``-step conditional decomposition of the pool from that date on:
    the contributions the cut did not forecast, with the forecast itself counted
    in ``"initial"``. These are what Dynare's ``plot_shock_decomposition`` draws
    with ``realtime`` 1, 3 and 2 and a ``vintage``, and
    :func:`~dsgelab.plots.plot_decomposition` draws them with the same options.
    """
    Y = space.data_matrix(data)
    Tn = Y.shape[0]
    h = max(0, int(forecast))
    sol = space.solution
    assert sol is not None
    names = list(variables) if variables is not None else list(sol.model.names)
    shocks = list(space.shocks)

    def empty(n):
        return {"shocks": {e: {v: np.full(n, np.nan) for v in names} for e in shocks},
                "initial": {v: np.full(n, np.nan) for v in names},
                "total": {v: np.full(n, np.nan) for v in names}}

    def put(out, dec, src, dst):
        for v in names:
            out["total"][v][dst] = dec["total"][v][src]
            out["initial"][v][dst] = dec["initial"][v][src]
            for e in shocks:
                out["shocks"][e][v][dst] = dec["shocks"][e][v][src]

    pool, cond = empty(Tn + h), empty(Tn)
    fpool = empty(Tn + 1)
    saved: dict[int, dict] = {}
    fsaved: dict[int, dict] = {}
    want = {int(v) for v in vintages}
    first = max(1, min(first, Tn))
    nk = space.n_model_states
    for cut in range(first, Tn + 1):
        sm = kalman_smoother(space, Y[:cut], init=init)
        dec = shock_decomposition(sm, variables=names, initial_shock_states=initial_shock_states)
        if h:
            dec = _extend_decomposition(sm, dec, names, h, initial_shock_states)
        if cut == first:
            put(pool, dec, slice(0, cut), slice(0, cut))
        elif cut < Tn:
            put(pool, dec, cut - 1, cut - 1)
        else:
            put(pool, dec, slice(cut - 1, cut + h), slice(cut - 1, cut + h))
        # the one-step conditional decomposition: the impact of this date's innovations
        last = cut - 1
        impact = {e: sol.variables((space.R[:nk, m] * sm.shocks[last, m])[None])
                  for m, e in enumerate(shocks)}
        for v in names:
            cond["total"][v][last] = dec["total"][v][last]
            for e in shocks:
                cond["shocks"][e][v][last] = impact[e][v][0]
            cond["initial"][v][last] = dec["total"][v][last] - sum(impact[e][v][0]
                                                                   for e in shocks)
        if h:
            put(fpool, dec, cut, cut)
        if cut in want:
            saved[cut] = _slice_decomposition(dec, slice(0, cut + h))
            if h:
                fsaved[cut] = _slice_decomposition(dec, slice(cut - 1, cut + h))
    out = dict(pool)
    out["conditional"] = cond
    if h:
        out["forecast"] = fpool
        out["forecast_vintages"] = fsaved
        csaved = {}
        for cut, fc in fsaved.items():
            end = min(cut + h, Tn)
            n = end - cut + 1
            got = _slice_decomposition(pool, slice(cut - 1, end))
            got["initial"] = {v: got["initial"][v] - fc["initial"][v][:n] + fc["total"][v][:n]
                              for v in names}
            got["shocks"] = {e: {v: got["shocks"][e][v] - fc["shocks"][e][v][:n] for v in names}
                             for e in shocks}
            csaved[cut] = got
        out["conditional_vintages"] = csaved
    if want:
        out["vintages"] = saved
    return out


def _slice_decomposition(dec: Mapping, span: slice) -> dict:
    return {"shocks": {e: {v: np.array(p[span]) for v, p in d.items()}
                       for e, d in dec["shocks"].items()},
            "initial": {v: np.array(p[span]) for v, p in dec["initial"].items()},
            "total": {v: np.array(p[span]) for v, p in dec["total"].items()}}


def _extend_decomposition(sm: SmootherResult, dec: Mapping, names, h: int,
                          initial_shock_states) -> dict:
    """``dec`` carried ``h`` dates past the sample by the solution, without new shocks.

    Each shock's contribution and the total follow ``k' = T k``, and the initial
    condition is what is left, as in Dynare's forecast decomposition.
    """
    ss = sm.space
    sol = ss.solution
    assert sol is not None
    nk = ss.n_model_states
    Tn, n = sm.states.shape

    def ahead(k):
        out = np.zeros((h, n))
        for j in range(h):
            k = ss.T @ k
            out[j] = k
        return sol.variables(out[:, :nk])

    tot = ahead(sm.states[-1])
    init_map = dict(initial_shock_states or {})
    out: dict = {"shocks": {}, "total": {}, "initial": {}}
    for m, e in enumerate(ss.shocks):
        piece = np.zeros(n)
        for st, sh in init_map.items():
            if sh == e:
                i = ss.states.index(st)
                piece[i] = sm.states[0, i]
        for t in range(1, Tn):
            piece = ss.T @ piece + ss.R[:, m] * sm.shocks[t, m]
        more = ahead(piece)
        out["shocks"][e] = {v: np.r_[dec["shocks"][e][v], more[v]] for v in names}
    for v in names:
        out["total"][v] = np.r_[dec["total"][v], tot[v]]
        out["initial"][v] = out["total"][v] - sum(out["shocks"][e][v] for e in ss.shocks)
    return out


@dataclass
class Forecast:
    """Point forecast of every variable and bands from the forecast-error variance."""

    mean: dict           # levels (or logs, for log observables), (horizon,)
    std: dict
    lower: dict
    upper: dict
    horizon: int
    coverage: float


def forecast(space: StateSpace, data: Data, horizon: int, *, init: str = "auto",
             coverage: float = 0.9, variables: Sequence[str] | None = None,
             include_measurement: bool = False, logs: Sequence[str] = ()) -> Forecast:
    """Forecast from the end of the sample, with Gaussian bands.

    The mean path starts from the filtered state at the last date, ``a_{T|T}``,
    and the variance compounds the state uncertainty ``P_{T|T}`` with the shocks
    of the forecast horizon: ``V_h = T V_{h-1} T' + R Sigma R'``. Bands are
    ``mean +- z * sd`` with ``z`` the normal quantile for ``coverage``; with
    ``include_measurement`` the observables' measurement errors are added.
    Variables in ``logs`` are reported as logs (Dynare's ``loglinear``);
    the others in levels.
    """
    from scipy.stats import norm

    fr = kalman_filter(space, data, init=init)
    sol = space.solution
    names = list(variables) if variables is not None else list(sol.model.names)
    a, V = fr.a_filt[-1].copy(), fr.P_filt[-1].copy()
    RQR = space.R @ space.Sigma @ space.R.T
    z = norm.ppf(0.5 + coverage / 2)
    mean = {v: np.zeros(horizon) for v in names}
    sd = {v: np.zeros(horizon) for v in names}
    J = sol._J()
    nk = space.n_model_states
    for h in range(horizon):
        a = space.T @ a
        V = space.T @ V @ space.T.T + RQR
        xs = J @ a[:nk]
        Vx = J @ V[:nk, :nk] @ J.T
        for v in names:
            j = sol.model.position(v)
            c, s = sol.coords.center[j], sol.coords.scale[j]
            if v in logs:
                mean[v][h] = np.log(c) + xs[j]
                var = Vx[j, j]
            else:
                mean[v][h] = c + s * xs[j]
                var = s * s * Vx[j, j]
            if include_measurement and v in space.observables:
                i = space.observables.index(v)
                var += space.H[i, i]
            sd[v][h] = np.sqrt(max(var, 0.0))
    lower = {v: mean[v] - z * sd[v] for v in names}
    upper = {v: mean[v] + z * sd[v] for v in names}
    return Forecast(mean=mean, std=sd, lower=lower, upper=upper, horizon=horizon,
                    coverage=coverage)


def initial_condition_decomposition(smoothed: SmootherResult, *,
                                    variables: Sequence[str] | None = None,
                                    initial_shock_states: Mapping[str, str] | None = None) -> dict:
    """The initial condition of a shock decomposition, split by the variables' first values.

    Dynare's ``initial_condition_decomposition``. At the first date each variable
    is its own contribution. From the second on, the first value of each
    variable that the solution carries in a state is carried alone, and the
    other variables contribute nothing. What the first values do not explain is
    what the shocks did after the first date. A state ``LAG1_x``, which a model
    read from a ``.mod`` file or declared by equations keeps for the lag of
    ``x``, holds the first value of ``x`` at the second date, and a state of a
    model declared in Python holds its own first value at the first date. The
    states in ``initial_shock_states`` (:attr:`dsgelab.ModFile.shock_states`)
    carry a shock and count with the shocks.

    Returns ``{"initial": {variable: {var: path}}, "shocks": {var: path},
    "total": {var: path}}``, the first level keyed by the variable whose first
    value contributes, one per variable of the model that is not such a state.
    """
    ss = smoothed.space
    sol = ss.solution
    assert sol is not None
    k = smoothed.states
    Tn, n = k.shape
    nk = ss.n_model_states
    names = list(variables) if variables is not None else list(sol.model.names)
    carriers = set(initial_shock_states or {})
    model_names = list(sol.model.names)
    total = sol.variables(k[:, :nk])
    lags = {st: st[5:] for st in ss.states[:nk]
            if st.startswith("LAG1_") and st[5:] in model_names}
    columns = [v for v in model_names if v not in carriers and v not in lags]
    carried = {lags.get(st, st): (i, 1 if st in lags else 0)
               for i, st in enumerate(ss.states[:nk]) if st not in carriers}
    out: dict = {"initial": {}, "total": {v: total[v] for v in names}}
    for c in columns:
        paths = {v: np.zeros(Tn) for v in names}
        if c in carried and Tn > 1:
            i, start = carried[c]
            piece = np.zeros((Tn, n))
            piece[start, i] = k[start, i]
            for t in range(start + 1, Tn):
                piece[t] = ss.T @ piece[t - 1]
            moved = sol.variables(piece[:, :nk])
            for v in names:
                paths[v][1:] = moved[v][1:]
        if c in paths:
            paths[c][0] = total[c][0]
        out["initial"][c] = paths
    out["shocks"] = {v: total[v] - sum(out["initial"][c][v] for c in columns) for v in names}
    return out


def annualise_decomposition(decomposition: Mapping, kind: str = "yoy", *,
                            quarter: int = 1) -> dict:
    """A quarterly decomposition at another frequency, as ``plot_shock_decomposition``'s ``type``.

    ``decomposition`` maps ``"shocks"``, ``"initial"`` and ``"total"`` (and any
    other entry with the same form, such as the ``"states"`` of
    :func:`initial_condition_decomposition`'s ``"initial"``) to paths.
    ``kind="qoq"`` leaves it as it is, ``"yoy"`` sums each contribution over four
    consecutive quarters, the year-on-year change of a variable in quarterly
    changes, from the fourth quarter of the sample on, and ``"aoa"`` keeps the
    fourth quarter of each year. ``quarter`` is the quarter of the first date
    (Dynare's ``initial_date``). Without Dynare's ``q2a`` structure, this is what
    its ``type=yoy`` and ``type=aoa`` draw. The result has the same form.
    """
    if kind not in ("qoq", "yoy", "aoa"):
        raise ValueError(f"kind must be 'qoq', 'yoy' or 'aoa', not {kind!r}")
    if not 1 <= quarter <= 4:
        raise ValueError("quarter must be 1, 2, 3 or 4")

    def change(y):
        y = np.asarray(y, dtype=float)
        if kind == "yoy":
            return y[:-3] + y[1:-2] + y[2:-1] + y[3:]
        if kind == "aoa":
            return y[4 - quarter::4]
        return y.copy()

    def walk(node):
        if isinstance(node, Mapping):
            return {key: walk(value) for key, value in node.items()}
        return change(node)

    return walk(decomposition)


@dataclass
class ConditionalForecast:
    """A conditional forecast with bands, as Dynare's ``conditional_forecast``.

    See :func:`conditional_forecast_bands`. Every path runs over dates ``0..H``,
    date 0 the starting point.

    Attributes
    ----------
    mean, lower, upper : the mean of the conditional paths over the draws and the
        bounds of the interval holding ``coverage`` of them, by variable
        (Dynare's ``cond.Mean`` and ``cond.ci``).
    unconditional_mean, unconditional_lower, unconditional_upper : the same
        without conditions (``uncond``).
    shocks : the controlled shocks over dates ``1..H``, ``{shock: {"mean": ...,
        "lower": ..., "upper": ...}}`` (``controlled_exo_variables``).
    coverage, replic : the share in the bands and the number of draws.
    """

    mean: dict
    lower: dict
    upper: dict
    unconditional_mean: dict
    unconditional_lower: dict
    unconditional_upper: dict
    shocks: dict
    coverage: float
    replic: int


def _order_bands(X: np.ndarray, coverage: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean and Dynare's interval of the draws along the first axis: ranks ``round(n tt)``."""
    from .plots import _round
    n = X.shape[0]
    tt = (1 - coverage) / 2
    t1, t2 = max(1, _round(n * tt)), min(n, _round(n * (1 - tt)))
    xs = np.sort(X, axis=0)
    return X.mean(axis=0), xs[t1 - 1], xs[t2 - 1]


def _conditional_paths(sol: FirstOrderSolution, targets: np.ndarray, rows: list[int],
                       ctrl: list[int], k0: np.ndarray, shocks: np.ndarray):
    """Dynare's ``get_shock_paths``, for draws at once.

    ``targets`` ``(H, n_cond)`` in deviations (NaN where free), ``shocks``
    ``(replic, H, n_shocks)`` with the controlled ones ignored. At each date the
    controlled shocks paired with the constrained variables in force, condition
    ``i`` with shock ``i``, put them on their paths; with more or fewer controlled
    shocks than conditions, least squares on all of them. Returns the state paths
    ``(replic, H + 1, n_k)`` and the controlled shocks ``(replic, H, n_ctrl)``.
    """
    replic, H, ne = shocks.shape
    nk = sol.klein.n_k
    Q = sol.Q if sol.Q is not None else np.zeros((nk, ne))
    J = sol._J()
    u = shocks.copy()
    u[:, :, ctrl] = 0.0
    k = np.zeros((replic, H + 1, nk))
    k[:, 0] = k0
    e = np.zeros((replic, H, len(ctrl)))
    paired = len(ctrl) == len(rows)
    for t in range(H):
        base = k[:, t] @ sol.P.T + u[:, t] @ Q.T
        live = [i for i in range(len(rows)) if np.isfinite(targets[t, i])]
        if live:
            r = [rows[i] for i in live]
            use = [ctrl[i] for i in live] if paired else ctrl
            M = J[r] @ Q[:, use]
            gap = targets[t, live][None] - base @ J[r].T
            sol_e = (np.linalg.solve(M, gap.T).T if paired
                     else np.linalg.lstsq(M, gap.T, rcond=None)[0].T)
            base = base + sol_e @ Q[:, use].T
            cols = live if paired else list(range(len(ctrl)))
            e[:, t, cols] = sol_e
        k[:, t + 1] = base
    return k, e


def conditional_forecast_bands(solution: FirstOrderSolution, conditions: Mapping[str, ArrayLike],
                               *, controlled: Sequence[str] | None = None,
                               periods: int | None = None, initial: ArrayLike | None = None,
                               cov: Covariance = None, levels: bool = False,
                               replic: int = 5000, coverage: float = 0.8, seed: int | None = 0,
                               normals: ArrayLike | tuple[ArrayLike, ArrayLike] | None = None
                               ) -> ConditionalForecast:
    """A conditional forecast with bands from the shocks left free, as Dynare's.

    ``conditions`` maps variables to their paths over dates ``1..cL`` (a NaN
    leaves that date free) and ``controlled`` names the shocks that put them on
    their paths, one per condition and in the same order, as Dynare's
    ``controlled_varexo`` (all the shocks by default). The forecast runs over
    ``periods`` dates (``cL`` by default) from ``initial`` (hat-deviations of the
    states, the steady state by default). For each of ``replic`` draws, the other
    shocks are drawn over the ``periods`` dates from the covariance ``cov``
    (scaled by its Cholesky factor, Dynare's square root of a diagonal), and the
    controlled ones are chosen date by date to meet the conditions in force at
    that date, as Dynare's ``conditional_forecasts.get_shock_paths``; after
    ``cL`` they are zero. The unconditional paths draw the same shocks with the
    controlled ones at zero. The bands hold ``coverage`` of the draws, ranked as
    Dynare ranks them (its ``conf_sig``, 0.8 by default).

    ``normals``, standard normal draws ``(replic, periods, n_shocks)``, or two
    such arrays for the conditional and the unconditional paths, replace the
    draws of ``seed``, so that the forecast can be compared with Dynare's on the
    same draws. With ``levels`` the paths are in levels in and out, otherwise in
    the rescaled deviations, as for
    :meth:`~dsgelab.FirstOrderSolution.conditional_forecast`.
    """
    from .solution import shock_covariance
    sol = solution
    names = list(conditions)
    series = [np.asarray(conditions[n], dtype=float).ravel() for n in names]
    cl = series[0].size
    if any(s.size != cl for s in series):
        raise ValueError("every conditioned path must have the same length")
    H = cl if periods is None else int(periods)
    if H < cl:
        raise ValueError(f"periods={H} is shorter than the conditions ({cl} dates)")
    if levels:
        series = [(s - sol.coords.center[sol.model.position(n)])
                  / sol.coords.scale[sol.model.position(n)] for n, s in zip(names, series)]
    targets = np.full((H, len(names)), np.nan)
    targets[:cl] = np.column_stack(series)
    shocks_ = list(sol.model.shocks)
    ne = len(shocks_)
    ctrl = list(range(ne)) if controlled is None else [shocks_.index(c) for c in controlled]
    rows = [sol.model.position(n) for n in names]
    k0 = np.zeros(sol.klein.n_k) if initial is None else np.asarray(initial, dtype=float)
    Sigma = shock_covariance(shocks_, cov)
    L = (np.diag(np.sqrt(np.diag(Sigma))) if np.allclose(Sigma, np.diag(np.diag(Sigma)))
         else np.linalg.cholesky(Sigma))
    if normals is None:
        rng = np.random.default_rng(seed)
        Z1 = rng.standard_normal((replic, H, ne))
        Z2 = rng.standard_normal((replic, H, ne))
    elif isinstance(normals, tuple):
        Z1, Z2 = (np.asarray(z, dtype=float) for z in normals)
    else:
        Z1 = Z2 = np.asarray(normals, dtype=float)
    if Z1.shape != (Z1.shape[0], H, ne) or Z2.shape != Z1.shape:
        raise ValueError(f"normals must be (replic, {H}, {ne})")
    k_c, e_c = _conditional_paths(sol, targets, rows, ctrl, k0, Z1 @ L.T)
    k_u, _ = _conditional_paths(sol, np.full_like(targets, np.nan), rows, ctrl, k0, Z2 @ L.T)

    def paths(k):
        flat = sol.variables(k.reshape(-1, k.shape[-1]))
        out = {v: x.reshape(k.shape[0], -1) for v, x in flat.items()}
        return sol.to_levels(out) if levels else out

    pc, pu = paths(k_c), paths(k_u)
    bands_c = {v: _order_bands(x, coverage) for v, x in pc.items()}
    bands_u = {v: _order_bands(x, coverage) for v, x in pu.items()}
    shock_bands = {}
    for j, i in enumerate(ctrl):
        m, lo, hi = _order_bands(e_c[:, :cl, j], coverage)
        shock_bands[shocks_[i]] = {"mean": m, "lower": lo, "upper": hi}
    return ConditionalForecast(
        mean={v: b[0] for v, b in bands_c.items()}, lower={v: b[1] for v, b in bands_c.items()},
        upper={v: b[2] for v, b in bands_c.items()},
        unconditional_mean={v: b[0] for v, b in bands_u.items()},
        unconditional_lower={v: b[1] for v, b in bands_u.items()},
        unconditional_upper={v: b[2] for v, b in bands_u.items()},
        shocks=shock_bands, coverage=coverage, replic=int(Z1.shape[0]))
