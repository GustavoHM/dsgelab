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
           "realtime_shock_decomposition", "forecast"]

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

    @property
    def n_states(self) -> int:
        return self.T.shape[0]

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
                cov: Covariance = None) -> StateSpace:
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
    return StateSpace(T=np.asarray(sol.P, float), R=np.asarray(Q, float), Sigma=Sigma, Z=Z,
                      d=d, H=H, observables=observables, states=list(model.states),
                      shocks=list(model.shocks), transforms=tr, solution=sol)


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
        paths = sol.variables(self.states)
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
        pieces[e] = sol.variables(piece)
    total = sol.variables(k)
    out = {"shocks": {e: {v: pieces[e][v] for v in names} for e in ss.shocks},
           "total": {v: total[v] for v in names}}
    out["initial"] = {v: total[v] - sum(out["shocks"][e][v] for e in ss.shocks) for v in names}
    return out


def realtime_shock_decomposition(space: StateSpace, data: Data, *, init: str = "auto",
                                 variables: Sequence[str] | None = None,
                                 initial_shock_states: Mapping[str, str] | None = None,
                                 first: int = 2) -> dict:
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
    """
    Y = space.data_matrix(data)
    Tn = Y.shape[0]
    names = list(variables) if variables is not None else list(space.solution.model.names)
    shocks = {e: {v: np.full(Tn, np.nan) for v in names} for e in space.shocks}
    initial = {v: np.full(Tn, np.nan) for v in names}
    total = {v: np.full(Tn, np.nan) for v in names}
    first = max(1, min(first, Tn))
    for cut in range(first, Tn + 1):
        sm = kalman_smoother(space, Y[:cut], init=init)
        dec = shock_decomposition(sm, variables=names, initial_shock_states=initial_shock_states)
        dates = range(cut) if cut == first else [cut - 1]
        for t in dates:
            for v in names:
                total[v][t] = dec["total"][v][t]
                initial[v][t] = dec["initial"][v][t]
                for e in space.shocks:
                    shocks[e][v][t] = dec["shocks"][e][v][t]
    return {"shocks": shocks, "initial": initial, "total": total}


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
    for h in range(horizon):
        a = space.T @ a
        V = space.T @ V @ space.T.T + RQR
        xs = J @ a
        Vx = J @ V @ J.T
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
