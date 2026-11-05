"""Likelihood of a Markov-switching state space by Kim's filter, and its estimation.

The state space of a :class:`~dsgelab.MarkovSwitchingSolution` is, with
``s_t`` the regime at ``t``::

    k_t = P[s_{t-1}, s_t] k_{t-1} + p[s_{t-1}, s_t] + Q[s_{t-1}, s_t] eps_t,   eps ~ N(0, Sigma)
    y_t = d[s_t] + Z[s_t] k_t + e_t,                                         e ~ N(0, H)

Observables are levels or logs, as in :func:`dsgelab.state_space`, and ``d``
carries the regime constants of the solution. Given the whole regime path the
model is a linear Gaussian state space, so the exact likelihood is a mixture of
``h^T`` Kalman filters, one per path. Kim's (1994) filter keeps, at each date,
one Gaussian per regime: it runs the ``h^2`` Kalman steps from each regime at
``t-1`` to each at ``t`` and collapses the ``h`` results that end in the same
regime into one, with the mean and variance of their mixture (Kim and Nelson,
1999, ch. 5). The collapse is the approximation. It is exact when the states
carry no information about the past regimes, as in Hamilton's (1989) filter,
where nothing but the regime has dynamics. :func:`enumerated_likelihood`
computes the exact mixture on short samples, which measures the error.

The regime probabilities are the filter's ``Pr(s_t | y_1..y_t)``, the
predicted ``Pr(s_t | y_1..y_{t-1})`` and the smoothed ``Pr(s_t | y_1..y_T)``
by a backward recursion on the filter's joint probabilities of
``(s_{t-1}, s_t)``, with Kim's (1994) approximation.

The first state is drawn, by default, from the ergodic distribution: regime
``i`` with its ergodic probability, and ``k_1`` given ``s_1 = i`` from a normal
with the mean and variance of ``k`` given ``s = i`` in the ergodic distribution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike

from .kalman import Data
from .markov import MarkovChain
from .model import Covariance, Model, Params, Point, get_param, set_params
from .priors import Prior, prior as make_prior
from .switching import (MarkovSwitchingError, MarkovSwitchingSolution, _ergodic, _regime_moments,
                        _transition, solve_markov_switching)

__all__ = ["SwitchingStateSpace", "KimFilterResult", "MarkovSwitchingProblem",
           "switching_state_space", "kim_filter", "enumerated_likelihood"]

_LOG2PI = math.log(2.0 * math.pi)


@dataclass
class SwitchingStateSpace:
    """``k_t = T[i,j] k_{t-1} + c[i,j] + R[i,j] eps_t``, ``y_t = d[j] + Z[j] k_t + e_t``.

    ``i`` is the regime at ``t-1`` and ``j`` that at ``t``.
    """

    T: np.ndarray                 # (h, h, n, n)
    c: np.ndarray                 # (h, h, n)
    R: np.ndarray                 # (h, h, n, n_eps)
    Sigma: np.ndarray             # (n_eps, n_eps)
    Z: np.ndarray                 # (h, n_obs, n)
    d: np.ndarray                 # (h, n_obs)
    H: np.ndarray                 # (n_obs, n_obs)
    transition: np.ndarray        # (h, h)
    observables: list
    states: list = field(default_factory=list)
    transforms: list = field(default_factory=list)
    solution: MarkovSwitchingSolution | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.transition = _transition(self.transition)

    @property
    def h(self) -> int:
        return self.transition.shape[0]

    @property
    def n_states(self) -> int:
        return self.T.shape[2]

    @property
    def n_obs(self) -> int:
        return self.Z.shape[1]

    def ergodic_start(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(probabilities, means, covariances)`` of ``s_1`` and of ``k_1`` given ``s_1``."""
        pi, mean, second = _regime_moments(self.T, self.c, self.R, self.Sigma, self.transition)
        cov = second - np.einsum("ia,ib->iab", mean, mean)
        return pi, mean, 0.5 * (cov + np.transpose(cov, (0, 2, 1)))

    def data_matrix(self, data: Data) -> np.ndarray:
        """``(T, n_obs)`` from a mapping of series or an array, in the observables' order."""
        if isinstance(data, Mapping):
            return np.column_stack([np.asarray(data[n], dtype=float).ravel()
                                    for n in self.observables])
        Y = np.asarray(data, dtype=float)
        Y = Y.reshape(-1, 1) if Y.ndim == 1 else Y
        if Y.shape[1] != self.n_obs:
            raise ValueError(f"data has {Y.shape[1]} columns for {self.n_obs} observables")
        return Y


def switching_state_space(sol: MarkovSwitchingSolution, observables: Sequence[str], *,
                          transforms: str | Mapping[str, str] | None = None,
                          measurement: Mapping[str, float] | ArrayLike | None = None,
                          cov: Covariance = None) -> SwitchingStateSpace:
    """The state space of a :class:`~dsgelab.MarkovSwitchingSolution`.

    ``observables``, ``transforms``, ``measurement`` and ``cov`` are as for
    :func:`dsgelab.state_space`: an observable is a state or a jump, in levels
    or in logs, and the measurement errors are standard deviations by name, one
    for all, or a covariance matrix.
    """
    from .solution import shock_covariance

    model = sol.model
    observables = list(observables)
    h, nk = sol.h, model.n_states
    if isinstance(transforms, str) or transforms is None:
        tr = [transforms or "level"] * len(observables)
    else:
        tr = [transforms.get(n, "level") for n in observables]
    center, scale = sol.coords.center, sol.coords.scale
    Z = np.zeros((h, len(observables), nk))
    d = np.zeros((h, len(observables)))
    for s in range(h):
        for i, name in enumerate(observables):
            if name in model.states:
                row = np.zeros(nk)
                row[model.states.index(name)] = 1.0
                const = 0.0
            elif name in model.jumps:
                m = model.jumps.index(name)
                row, const = np.asarray(sol.F[s][m], dtype=float), float(sol.f[s][m])
            else:
                raise KeyError(f"{name!r} is not a variable of {model.name!r}")
            j = model.position(name)
            if tr[i] == "level":
                Z[s, i], d[s, i] = scale[j] * row, center[j] + scale[j] * const
            elif tr[i] == "log":
                if center[j] <= 0:
                    raise ValueError(f"{name!r} has steady state {center[j]}: a log "
                                     f"observation needs a positive one")
                if not np.isclose(scale[j], abs(center[j])):
                    raise ValueError(f"{name!r} is not in proportional coordinates")
                Z[s, i], d[s, i] = row, np.log(center[j]) + const
            else:
                raise ValueError(f"transform must be 'level' or 'log', got {tr[i]!r}")
    n_obs = len(observables)
    if measurement is None:
        H = np.zeros((n_obs, n_obs))
    elif isinstance(measurement, Mapping):
        H = np.diag([float(measurement.get(n, 0.0)) ** 2 for n in observables])
    elif np.isscalar(measurement):
        H = float(measurement) ** 2 * np.eye(n_obs)
    else:
        H = np.asarray(measurement, dtype=float)
        if H.shape != (n_obs, n_obs):
            raise ValueError("measurement covariance has the wrong shape")
    return SwitchingStateSpace(T=np.real(sol.P), c=np.real(sol.p), R=sol.Q,
                               Sigma=shock_covariance(model.shocks, cov), Z=Z, d=d, H=H,
                               transition=sol.transition, observables=observables,
                               states=list(model.states), transforms=tr, solution=sol)


def _start(space: SwitchingStateSpace, init):
    if isinstance(init, str):
        if init != "ergodic":
            raise ValueError("init must be 'ergodic' or (probabilities, means, covariances)")
        return space.ergodic_start()
    prob, mean, cov = (np.asarray(x, dtype=float) for x in init)
    h, n = space.h, space.n_states
    if prob.shape != (h,) or mean.shape != (h, n) or cov.shape != (h, n, n):
        raise ValueError(f"init needs shapes ({h},), ({h}, {n}) and ({h}, {n}, {n})")
    return prob / prob.sum(), mean, cov


def _update(a, V, y, Z, d, H):
    """Kalman update of ``N(a, V)`` with ``y = d + Z k + e``; returns ``(a, V, log density)``."""
    if y.size == 0:
        return a, V, 0.0
    v = y - d - Z @ a
    S = Z @ V @ Z.T + H
    S = 0.5 * (S + S.T)
    sign, logdet = np.linalg.slogdet(S)
    if sign <= 0:
        return a, V, -np.inf
    K = np.linalg.solve(S, Z @ V).T
    a = a + K @ v
    V = V - K @ Z @ V
    return a, 0.5 * (V + V.T), -0.5 * (y.size * _LOG2PI + logdet + v @ np.linalg.solve(S, v))


def _batch_update(a, V, y, Z, d, H):
    """:func:`_update` for every pair ``(i, j)``, with ``Z[j]`` and ``d[j]`` for regime ``j``."""
    if y.size == 0:
        return a, V, np.zeros(a.shape[:2])
    v = y - d[None] - np.einsum("jab,ijb->ija", Z, a)
    ZV = np.einsum("jab,ijbc->ijac", Z, V)
    S = np.einsum("ijab,jcb->ijac", ZV, Z) + H
    S = 0.5 * (S + np.swapaxes(S, -1, -2))
    sign, logdet = np.linalg.slogdet(S)
    ok = sign > 0
    S_safe = np.where(ok[..., None, None], S, np.eye(y.size))
    Kt = np.linalg.solve(S_safe, ZV)                       # (S^-1 Z V), the gain transposed
    sv = np.linalg.solve(S_safe, v[..., None])[..., 0]
    a = a + np.einsum("ijba,ijb->ija", Kt, v)
    V = V - np.einsum("ijba,ijbc->ijac", ZV, Kt)
    V = 0.5 * (V + np.swapaxes(V, -1, -2))
    ld = np.where(ok, -0.5 * (y.size * _LOG2PI + logdet + np.einsum("ija,ija->ij", v, sv)),
                  -np.inf)
    return a, V, ld


def _logsumexp(x):
    m = np.max(x)
    if not np.isfinite(m):
        return m
    return m + math.log(np.sum(np.exp(x - m)))


@dataclass
class KimFilterResult:
    """Likelihood, regime probabilities and filtered states of Kim's filter."""

    loglik: float
    loglik_t: np.ndarray          # (T,)
    filtered: np.ndarray          # (T, h)  Pr(s_t | y_1..y_t)
    predicted: np.ndarray         # (T, h)  Pr(s_t | y_1..y_{t-1})
    smoothed: np.ndarray          # (T, h)  Pr(s_t | y_1..y_T)
    states: np.ndarray            # (T, n)  E[k_t | y_1..y_t]
    regime_states: np.ndarray     # (T, h, n)  E[k_t | s_t, y_1..y_t]


def kim_filter(space: SwitchingStateSpace, data: Data, *,
               init: str | tuple = "ergodic") -> KimFilterResult:
    """Kim's (1994) filter: approximate likelihood and regime probabilities.

    ``data`` are levels (or logs) of the observables, ``nan`` where missing.
    ``init`` is ``"ergodic"`` or ``(probabilities, means, covariances)`` of
    ``s_1`` and of ``k_1`` given ``s_1``, before the first observation.
    """
    Y = space.data_matrix(data)
    nT, h, n = Y.shape[0], space.h, space.n_states
    P = space.transition
    with np.errstate(divide="ignore"):
        logP = np.log(P)
    prob, a, V = _start(space, init)
    RSR = np.einsum("ijab,bc,ijdc->ijad", space.R, space.Sigma, space.R)
    loglik_t = np.zeros(nT)
    filtered = np.zeros((nT, h))
    predicted = np.zeros((nT, h))
    states = np.zeros((nT, n))
    regime_states = np.zeros((nT, h, n))
    pairs = np.zeros((nT, h, h))                  # Pr(s_{t-1}=i, s_t=j | y_1..y_t)
    for t in range(nT):
        obs = ~np.isnan(Y[t])
        y = Y[t, obs]
        H = space.H[np.ix_(obs, obs)]
        if t == 0:
            # Before the first observation there is one Gaussian per regime and no pair.
            predicted[0] = prob
            lw = np.full(h, -np.inf)
            new_a, new_V = np.array(a, dtype=float), np.array(V, dtype=float)
            for j in range(h):
                if prob[j] > 0:
                    new_a[j], new_V[j], ld = _update(a[j], V[j], y, space.Z[j][obs],
                                                     space.d[j][obs], H)
                    lw[j] = math.log(prob[j]) + ld
            loglik_t[0] = _logsumexp(lw)
            if not np.isfinite(loglik_t[0]):
                loglik_t[t:] = -np.inf
                break
            prob = np.exp(lw - loglik_t[0])
            a, V = new_a, new_V
        else:
            # The h^2 Kalman steps from each regime at t-1 to each at t, in one batch.
            with np.errstate(divide="ignore"):
                lw = np.log(prob)[:, None] + logP
            m = np.einsum("ijab,ib->ija", space.T, a) + space.c
            C = np.einsum("ijab,ibc,ijdc->ijad", space.T, V, space.T) + RSR
            aij, Vij, ld = _batch_update(m, C, y, space.Z[:, obs], space.d[:, obs], H)
            lw = np.where(np.isfinite(lw), lw + ld, -np.inf)
            predicted[t] = (prob[:, None] * P).sum(axis=0)
            loglik_t[t] = _logsumexp(lw.ravel())
            if not np.isfinite(loglik_t[t]):
                loglik_t[t:] = -np.inf
                break
            w = np.exp(lw - loglik_t[t])                       # Pr(s_{t-1}=i, s_t=j | y_1..y_t)
            pairs[t] = w
            prev, prob = prob, w.sum(axis=0)
            # Collapse the h Gaussians that end in each regime into one with the same
            # mean and variance as their mixture.
            for j in range(h):
                if prob[j] > 0:
                    share = w[:, j] / prob[j]
                else:
                    # A regime the data rule out keeps a state, for when it returns.
                    share = (prev > 0) & (P[:, j] > 0)
                    share = share / max(share.sum(), 1)
                a[j] = share @ aij[:, j]
                dev = aij[:, j] - a[j]
                V[j] = np.einsum("i,iab->ab", share, Vij[:, j]) + np.einsum(
                    "i,ia,ib->ab", share, dev, dev)
        filtered[t] = prob
        regime_states[t] = a
        states[t] = prob @ a
    smoothed = _smooth(filtered, pairs)
    return KimFilterResult(loglik=float(loglik_t.sum()), loglik_t=loglik_t, filtered=filtered,
                           predicted=predicted, smoothed=smoothed, states=states,
                           regime_states=regime_states)


def _smooth(filtered, pairs):
    """Backward recursion for ``Pr(s_t | y_1..y_T)``.

    ``Pr(s_{t-1}=i | y_1..y_T) = sum_j Pr(s_{t-1}=i | s_t=j, y_1..y_t) Pr(s_t=j | y_1..y_T)``,
    which takes the data after ``t`` to depend on ``s_{t-1}`` only through ``s_t``. That is Kim's
    (1994) approximation, with the conditioning on ``y_t`` that the filter's joint probabilities
    of ``(s_{t-1}, s_t)`` supply. It is exact when nothing but the regime has dynamics, and when
    the regime and the data reveal the state, as in Hamilton's (1989) autoregression.
    """
    nT = filtered.shape[0]
    out = np.zeros_like(filtered)
    if nT == 0:
        return out
    out[-1] = filtered[-1]
    for t in range(nT - 1, 0, -1):
        ratio = np.divide(out[t], filtered[t], out=np.zeros_like(out[t]), where=filtered[t] > 0)
        out[t - 1] = pairs[t] @ ratio
        total = out[t - 1].sum()
        if total > 0:
            out[t - 1] /= total
        else:
            out[t - 1] = filtered[t - 1]
    return out


def enumerated_likelihood(space: SwitchingStateSpace, data: Data, *,
                          init: str | tuple = "ergodic", max_paths: int = 1 << 20) -> float:
    """Exact log likelihood: the mixture of the Kalman filters of every regime path.

    With ``h`` regimes and ``T`` dates there are ``h^T`` paths, so this is for
    short samples; it refuses more than ``max_paths``. The first state is drawn
    as in :func:`kim_filter`, so the two differ only by Kim's collapse.
    """
    Y = space.data_matrix(data)
    nT, h, n = Y.shape[0], space.h, space.n_states
    if h ** nT > max_paths:
        raise ValueError(f"{h}^{nT} regime paths is more than max_paths={max_paths}")
    P = space.transition
    with np.errstate(divide="ignore"):
        logP = np.log(P)
    prob, a0, V0 = _start(space, init)
    RSR = np.einsum("ijab,bc,ijdc->ijad", space.R, space.Sigma, space.R)
    last = np.arange(h)
    with np.errstate(divide="ignore"):
        logw = np.log(prob)
    a, V = np.array(a0, dtype=float), np.array(V0, dtype=float)
    for t in range(nT):
        if t > 0:
            i = np.repeat(last, h)
            j = np.tile(np.arange(h), last.size)
            a = np.einsum("pab,pb->pa", space.T[i, j], np.repeat(a, h, axis=0)) + space.c[i, j]
            V = np.einsum("pab,pbc,pdc->pad", space.T[i, j], np.repeat(V, h, axis=0),
                          space.T[i, j]) + RSR[i, j]
            logw = np.repeat(logw, h) + logP[i, j]
            last = j
        obs = ~np.isnan(Y[t])
        y = Y[t, obs]
        H = space.H[np.ix_(obs, obs)]
        for k in range(last.size):
            if np.isfinite(logw[k]):
                a[k], V[k], ld = _update(a[k], V[k], y, space.Z[last[k]][obs],
                                         space.d[last[k]][obs], H)
                logw[k] += ld
    return float(_logsumexp(logw))


# ================================================================== estimation
def _bracket(name: str):
    """``("psi", (1,))`` for ``"psi[1]"`` and ``("p", (0, 1))`` for ``"p[0,1]"``; else None."""
    if not name.endswith("]") or "[" not in name:
        return None
    base, inside = name[:-1].split("[", 1)
    try:
        idx = tuple(int(x) for x in inside.split(","))
    except ValueError:
        return None
    return base.strip(), idx


@dataclass
class MarkovSwitchingProblem:
    """A Markov-switching model, data, and what is estimated.

    The fields are those of :class:`dsgelab.EstimationProblem`, with the
    regimes added. ``switching`` gives the value of each switching parameter in
    each regime, and ``transition`` the transition matrix, both as starting
    values for what is estimated and as fixed values for the rest. Besides a
    parameter, ``"stderr e"``, ``"stderr y"`` and ``"corr e1, e2"``, an
    estimated name may be ``"psi[1]"``, a switching parameter in regime 1, or
    ``"p[0,1]"``, the probability of moving from regime 0 to regime 1. In each
    row of the transition matrix, the entries that are not estimated share
    what the estimated ones leave of one, in their original proportions, so
    with two regimes ``"p[0,0]"`` sets ``p[0,1] = 1 - p[0,0]``.

    A parameter vector at which no mean-square stable solution is found has
    likelihood ``-inf``. Newton's method starts from the Klein solutions and
    ``starts`` random points, as in :func:`dsgelab.solve_markov_switching`.
    :func:`dsgelab.posterior_mode`, :func:`dsgelab.maximum_likelihood`,
    :func:`dsgelab.metropolis_hastings` and :func:`dsgelab.smc` accept it.
    """

    model: Model
    data: Data
    params: Params
    observables: Sequence[str]
    switching: Mapping[str, Sequence[float]]
    transition: MarkovChain | ArrayLike
    priors: Mapping = field(default_factory=dict)
    cov: Covariance = None
    measurement: Mapping[str, float] | float | None = None
    transforms: str | Mapping[str, str] | None = None
    next_regime: Sequence[str] = ()
    perturbed: Sequence[str] | None = None
    steady_fn: Callable | None = None
    guess: Point | None = None
    init: str | tuple = "ergodic"
    first_obs: int = 1
    presample: int = 0
    free: Sequence[str] | None = None
    bounds: Mapping | None = None
    starts: int = 0
    seed: int | None = 0
    _last_ss: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.observables = list(self.observables)
        self.priors = {k: v if isinstance(v, Prior) else make_prior(*v)
                       for k, v in dict(self.priors).items()}
        self.free = list(self.priors) if self.free is None else list(self.free)
        if not self.free:
            raise ValueError("nothing to estimate: give priors, or free= with bounds=")
        self.switching = {k: np.asarray(v, dtype=float).ravel() for k, v in self.switching.items()}
        self.transition = _transition(self.transition)
        h = self.transition.shape[0]
        named = np.zeros((h, h), dtype=bool)
        for n in self.free:
            b = _bracket(n)
            if b is None:
                continue
            base, idx = b
            if len(idx) == 2 and base == "p" and all(0 <= i < h for i in idx):
                named[idx] = True
            elif len(idx) == 1 and base in self.switching and 0 <= idx[0] < h:
                pass
            else:
                raise KeyError(f"{n!r}: name a switching parameter in a regime, as 'psi[1]', "
                               f"or a transition probability, as 'p[0,1]', with regimes 0 to "
                               f"{h - 1}")
        if np.any(named.all(axis=1)):
            raise ValueError("every entry of a row of the transition matrix is estimated: "
                             "leave one out, it is what the others leave of one")
        self._named = named
        if isinstance(self.data, Mapping):
            Y = np.column_stack([np.asarray(self.data[n], dtype=float).ravel()
                                 for n in self.observables])
        else:
            Y = np.asarray(self.data, dtype=float)
            Y = Y.reshape(-1, 1) if Y.ndim == 1 else Y
        self.Y = Y[self.first_obs - 1:]

    @property
    def names(self) -> list:
        return list(self.free)

    def _base_meas(self) -> dict:
        m = self.measurement
        if m is None:
            return {}
        if isinstance(m, Mapping):
            return {k: float(v) for k, v in m.items()}
        return {n: float(m) for n in self.observables}

    def start(self) -> np.ndarray:
        """Starting values: the calibrated value, else the prior mean."""
        from .solution import shock_covariance
        S = shock_covariance(self.model.shocks, self.cov)
        sds = {e: float(np.sqrt(S[i, i])) for i, e in enumerate(self.model.shocks)}
        meas = self._base_meas()
        out = []
        for n in self.names:
            b = _bracket(n)
            if b is not None and len(b[1]) == 2:
                v = float(self.transition[b[1]])
            elif b is not None:
                v = float(self.switching[b[0]][b[1][0]])
            elif n.startswith("stderr "):
                v = sds.get(n[7:], meas.get(n[7:], np.nan))
            else:
                try:
                    v = float(get_param(self.params, n))
                except (KeyError, AttributeError):
                    v = np.nan
            if (not np.isfinite(v) or v <= 0 and n.startswith("stderr ")) and n in self.priors:
                v = self.priors[n].mean
            out.append(float(v))
        return np.array(out)

    def unpack(self, theta: ArrayLike):
        """``(params, shock covariance, measurement sds, switching, transition)`` at ``theta``."""
        from .estimation import unpack_point
        theta = np.asarray(theta, dtype=float)
        switching = {k: v.copy() for k, v in self.switching.items()}
        T = self.transition.copy()
        rest_names, rest_values = [], []
        for n, v in zip(self.names, theta):
            b = _bracket(n)
            if b is None:
                rest_names.append(n)
                rest_values.append(v)
            elif len(b[1]) == 2:
                T[b[1]] = v
            else:
                switching[b[0]][b[1][0]] = v
        for i in range(T.shape[0]):
            named = self._named[i]
            if not named.any():
                continue
            left = 1.0 - T[i, named].sum()
            base = self.transition[i, ~named]
            share = base / base.sum() if base.sum() > 0 else np.full(base.size, 1.0 / base.size)
            T[i, ~named] = left * share
        params, S, meas = unpack_point(self.model, self.params, self.cov, self._base_meas(),
                                       rest_names, rest_values)
        return params, S, meas, switching, T

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

    def solve(self, theta: ArrayLike):
        """``(solution, state space)`` at ``theta``; raises if there is none."""
        from .steady import steady_state
        params, S, meas, switching, T = self.unpack(theta)
        if np.any(T < 0) or np.any(T > 1):
            raise MarkovSwitchingError("the transition probabilities leave [0, 1]")
        pi = _ergodic(T)
        mean_params = set_params(params, {k: float(pi @ v) for k, v in switching.items()})
        if self.steady_fn is not None:
            ss = self.steady_fn(mean_params)
        else:
            start = self._last_ss.values if self._last_ss is not None else self.guess
            ss = steady_state(self.model, start, mean_params)
        sol = solve_markov_switching(self.model, params, switching, T, steady=ss,
                                     next_regime=self.next_regime, perturbed=self.perturbed,
                                     starts=self.starts, seed=self.seed)
        self._last_ss = ss
        space = switching_state_space(sol, self.observables, transforms=self.transforms,
                                      measurement=meas or None, cov=S)
        return sol, space

    def filter(self, theta: ArrayLike) -> KimFilterResult:
        """Kim's filter at ``theta``, with the filtered and smoothed regime probabilities."""
        _, space = self.solve(theta)
        return kim_filter(space, self.Y, init=self.init)

    def log_likelihood(self, theta: ArrayLike) -> float:
        try:
            fr = self.filter(theta)
        except Exception:
            return -np.inf
        ll = float(fr.loglik_t[self.presample:].sum())
        return ll if np.isfinite(ll) else -np.inf

    def log_posterior(self, theta: ArrayLike) -> float:
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        ll = self.log_likelihood(theta)
        return float(lp + ll) if np.isfinite(ll) else -np.inf

    def _support(self):
        lo, hi = [], []
        for n in self.names:
            if n in self.priors:
                lo.append(self.priors[n].lower)
                hi.append(self.priors[n].upper)
            elif self.bounds and n in self.bounds:
                lo.append(self.bounds[n][0])
                hi.append(self.bounds[n][1])
            elif (b := _bracket(n)) is not None and len(b[1]) == 2:
                lo.append(0.0)
                hi.append(1.0)
            else:
                lo.append(-np.inf)
                hi.append(np.inf)
        return np.array(lo, float), np.array(hi, float)
