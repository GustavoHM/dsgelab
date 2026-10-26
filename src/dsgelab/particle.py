"""Bootstrap particle filtering for perturbation solutions."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import linalg

from .kalman import Data, state_space
from .model import Covariance
from .moment_estimation import _observation_conversion, _second_order_system
from .secondorder import SecondOrderSolution
from .solution import FirstOrderSolution, _psd_factor
from .thirdorder import ThirdOrderSolution

__all__ = ["ParticleFilterResult", "ParticleSmootherResult", "particle_filter", "particle_smoother"]


@dataclass
class ParticleSmootherResult:
    """Smoothed state means and effective sample sizes."""

    mean: np.ndarray
    ess: np.ndarray


@dataclass
class ParticleFilterResult:
    """Log likelihood and the final particle distribution."""

    loglik: float
    loglik_t: np.ndarray
    states: np.ndarray
    weights: np.ndarray
    ess: np.ndarray
    particles: int
    history: dict | None = field(default=None, repr=False)


def particle_filter(solution: FirstOrderSolution | SecondOrderSolution | ThirdOrderSolution,
                    data: Data, *,
                    observables, transforms=None, measurement=None, cov: Covariance = None,
                    particles: int = 1000, seed: int | None = 0,
                    resample: float | None = 0.5, store: bool = False) -> ParticleFilterResult:
    """Evaluate the likelihood with a bootstrap particle filter.

    ``measurement`` gives the observables' error standard deviations or covariance,
    as in :func:`dsgelab.state_space`; a positive-definite covariance is required.
    ``cov`` is the structural shock covariance at first order; second order uses
    ``solution.Sigma``. ``resample`` is the effective-sample-size threshold as a fraction of the
    particle count, or ``None`` to disable systematic resampling.
    """
    if particles < 1:
        raise ValueError("particles must be positive")
    if resample is not None and not 0.0 <= resample <= 1.0:
        raise ValueError("resample must be between 0 and 1, or None")
    observables = list(observables)
    if isinstance(solution, ThirdOrderSolution):
        return _third_order_particle_filter(solution, data, observables=observables,
                                           transforms=transforms, measurement=measurement,
                                           particles=particles, seed=seed, resample=resample,
                                           store=store)
    first = solution.first if isinstance(solution, SecondOrderSolution) else solution
    ss = state_space(first, observables, transforms=transforms, measurement=measurement,
                     cov=solution.Sigma if isinstance(solution, SecondOrderSolution) else cov)
    Y = ss.data_matrix(data)
    sign, logdet = np.linalg.slogdet(ss.H)
    if sign <= 0:
        raise ValueError("the measurement covariance must be positive definite")

    second = solution if isinstance(solution, SecondOrderSolution) else None
    if second is not None:
        system = _second_order_system(second)
        A, B, c = system["A"], system["B"], system["c"]
        C, D, d = system["C"], system["D"], system["d"]
        rows, base, scale = _observation_conversion(first, observables, transforms)
        C, D, d = C[rows] * scale[:, None], D[rows] * scale[:, None], base + scale * d[rows]
        mean, covariance = system["Ez"], system["Vz"]
        nk = second.model.n_states
    else:
        A, B, C, D, d = ss.T, ss.R, ss.Z, np.zeros((ss.n_obs, ss.R.shape[1])), ss.d
        c = np.zeros(ss.n_states)
        mean = np.zeros(ss.n_states)
        covariance = linalg.solve_discrete_lyapunov(ss.T, ss.R @ ss.Sigma @ ss.R.T)
        nk = first.model.n_states

    rng = np.random.default_rng(seed)
    factor = _psd_factor(covariance)
    shock_factor = _psd_factor(ss.Sigma)
    z = mean + rng.standard_normal((particles, len(mean))) @ factor.T
    weights = np.full(particles, 1.0 / particles)
    loglik_t = np.zeros(Y.shape[0])
    ess = np.zeros(Y.shape[0])
    states = np.zeros((Y.shape[0], particles, nk))
    z_history, weight_history = [], []
    for t in range(Y.shape[0]):
        eps = rng.standard_normal((particles, ss.Sigma.shape[0])) @ shock_factor.T
        if second is not None:
            inov = np.concatenate([
                eps,
                (eps[:, :, None] * eps[:, None, :]).reshape(particles, -1)
                - system["vec_sigma"],
                (z[:, :nk, None] * eps[:, None, :]).reshape(particles, -1),
            ], axis=1)
        else:
            inov = eps
        missing = np.isnan(Y[t])
        if np.all(missing):
            loglik_t[t] = 0.0
            ess[t] = particles
        else:
            if second is not None:
                error = Y[t, ~missing] - (d[~missing] + z @ C[~missing].T + inov @ D[~missing].T)
            if second is None:
                error = Y[t, ~missing] - (d[~missing] + z @ C[~missing].T)
            H = ss.H[np.ix_(~missing, ~missing)]
            logdet = np.linalg.slogdet(H)[1]
            logw = -0.5 * (np.count_nonzero(~missing) * np.log(2 * np.pi) + logdet
                           + np.sum(error @ np.linalg.inv(H) * error, axis=1))
            shift = float(np.max(logw))
            raw = weights * np.exp(logw - shift)
            total = float(np.sum(raw))
            if not np.isfinite(total) or total <= 0.0:
                raise FloatingPointError(f"every particle has zero weight at date {t}")
            loglik_t[t] = np.log(total) + shift
            weights = raw / total
            ess[t] = 1.0 / np.sum(weights ** 2)
        z = c + z @ A.T + inov @ B.T
        if resample is not None and ess[t] < resample * particles:
            positions = (rng.random() + np.arange(particles)) / particles
            indices = np.searchsorted(np.cumsum(weights), positions)
            z = z[indices]
            weights.fill(1.0 / particles)
        z_history.append(z.copy())
        weight_history.append(weights.copy())
        states[t] = z[:, :nk] if second is None else z[:, :nk] + z[:, nk:2 * nk]
    history = None
    if store:
        history = dict(z=z_history, weights=weight_history, A=A, B=B, c=c,
                       Vinov=ss.Sigma if second is None else system["Vinov"],
                       vec_sigma=None if second is None else system["vec_sigma"],
                       nk=nk, second=second is not None)
    return ParticleFilterResult(loglik=float(np.sum(loglik_t)), loglik_t=loglik_t,
                                states=states, weights=weights, ess=ess, particles=particles,
                                history=history)


def _third_order_particle_filter(third: ThirdOrderSolution, data: Data, *, observables,
                                 transforms, measurement, particles, seed, resample, store):
    first = third.second.first
    ss = state_space(first, observables, transforms=transforms, measurement=measurement,
                     cov=third.Sigma)
    Y = ss.data_matrix(data)
    if np.linalg.slogdet(ss.H)[0] <= 0:
        raise ValueError("the measurement covariance must be positive definite")
    model = first.model
    nk, ne = model.n_states, model.n_shocks
    rule = third._pruned_rule(dynare=True)
    rows, base, scale = _observation_conversion(first, observables, transforms)

    def observed(state, jump):
        # every variable at date t, as Dynare orders the rule: states first, then jumps
        full = np.concatenate([rule[state], rule[jump]], axis=0)[rows]
        return full * scale.reshape(-1, *[1] * (full.ndim - 1))

    G = {name: observed(state, jump) for name, state, jump in [
        ("k", "P", "F"), ("u", "Q", "g_u"), ("kk", "h_kk", "g_kk"), ("ku", "h_ku", "g_ku"),
        ("uu", "h_uu", "g_uu"), ("kkk", "h_kkk", "g_kkk"), ("uuu", "h_uuu", "g_uuu"),
        ("kuu", "h_kuu", "g_kuu"), ("kku", "h_kku", "g_kku"), ("xss", "h_xss", "g_xss"),
        ("uss", "h_uss", "g_uss")]}
    d = base + 0.5 * observed("h_s2", "g_ss")
    Sigma = third.Sigma
    shock_factor = _psd_factor(Sigma)
    rng = np.random.default_rng(seed)

    def quad(T, a, b):
        return np.einsum("rab,pa,pb->pr", T, a, b)

    def cubic(T, a, b, c):
        return np.einsum("rabc,pa,pb,pc->pr", T, a, b, c)

    def transition(kf, ks, kt, eps):
        """The pruned layers of Andreasen, Fernández-Villaverde and Rubio-Ramírez."""
        kf_next = kf @ rule["P"].T + eps @ rule["Q"].T
        ks_next = (ks @ rule["P"].T + 0.5 * quad(rule["h_kk"], kf, kf)
                   + quad(rule["h_ku"], kf, eps) + 0.5 * quad(rule["h_uu"], eps, eps)
                   + 0.5 * rule["h_s2"])
        kt_next = (kt @ rule["P"].T + quad(rule["h_kk"], kf, ks)
                   + cubic(rule["h_kkk"], kf, kf, kf) / 6 + 0.5 * kf @ rule["h_xss"].T
                   + quad(rule["h_ku"], ks, eps) + cubic(rule["h_uuu"], eps, eps, eps) / 6
                   + 0.5 * cubic(rule["h_kuu"], kf, eps, eps)
                   + 0.5 * cubic(rule["h_kku"], kf, kf, eps) + 0.5 * eps @ rule["h_uss"].T)
        return kf_next, ks_next, kt_next

    def observation(kf, ks, kt, eps, keep):
        """The same rule for the observables, evaluated before the transition."""
        g = {name: T[keep] for name, T in G.items()}
        return (d[keep] + (kf + ks + kt) @ g["k"].T + eps @ g["u"].T
                + 0.5 * quad(g["kk"], kf, kf) + quad(g["kk"], kf, ks)
                + cubic(g["kkk"], kf, kf, kf) / 6 + 0.5 * kf @ g["xss"].T
                + quad(g["ku"], kf + ks, eps) + 0.5 * quad(g["uu"], eps, eps)
                + cubic(g["uuu"], eps, eps, eps) / 6 + 0.5 * cubic(g["kuu"], kf, eps, eps)
                + 0.5 * cubic(g["kku"], kf, kf, eps) + 0.5 * eps @ g["uss"].T)

    # each particle starts from its own path of 1000 periods, a draw from the ergodic distribution
    kf, ks, kt = (np.zeros((particles, nk)) for _ in range(3))
    for _ in range(1000):
        kf, ks, kt = transition(kf, ks, kt, rng.standard_normal((particles, ne)) @ shock_factor.T)
    z = np.concatenate([kf, ks, kt], axis=1)
    weights = np.full(particles, 1.0 / particles)
    loglik_t = np.zeros(Y.shape[0])
    ess = np.zeros(Y.shape[0])
    states = np.zeros((Y.shape[0], particles, nk))
    z_history, weight_history = [], []
    for t in range(Y.shape[0]):
        kf, ks, kt = z[:, :nk], z[:, nk:2 * nk], z[:, 2 * nk:]
        eps = rng.standard_normal((particles, ne)) @ shock_factor.T
        missing = np.isnan(Y[t])
        if np.all(missing):
            ess[t] = particles
        else:
            error = Y[t, ~missing] - observation(kf, ks, kt, eps, ~missing)
            H = ss.H[np.ix_(~missing, ~missing)]
            logdet = np.linalg.slogdet(H)[1]
            logw = -0.5 * (np.count_nonzero(~missing) * np.log(2 * np.pi) + logdet
                           + np.sum(error @ np.linalg.inv(H) * error, axis=1))
            shift = float(np.max(logw))
            raw = weights * np.exp(logw - shift)
            total_weight = float(np.sum(raw))
            if not np.isfinite(total_weight) or total_weight <= 0.0:
                raise FloatingPointError(f"every particle has zero weight at date {t}")
            loglik_t[t] = np.log(total_weight) + shift
            weights = raw / total_weight
            ess[t] = 1.0 / np.sum(weights ** 2)
        z = np.concatenate(transition(kf, ks, kt, eps), axis=1)
        if resample is not None and ess[t] < resample * particles:
            positions = (rng.random() + np.arange(particles)) / particles
            indices = np.searchsorted(np.cumsum(weights), positions)
            z = z[indices]
            weights.fill(1.0 / particles)
        states[t] = z[:, :nk] + z[:, nk:2 * nk] + z[:, 2 * nk:]
        z_history.append(z.copy())
        weight_history.append(weights.copy())
    A = np.zeros((3 * nk, 3 * nk))
    A[:nk, :nk] = rule["P"]
    B = np.vstack([rule["Q"], np.zeros((2 * nk, ne))])
    c = np.zeros(3 * nk)
    history = dict(z=z_history, weights=weight_history, A=A, B=B, c=c,
                   Vinov=Sigma, vec_sigma=Sigma.reshape(-1), nk=nk,
                   second=False, third=True) if store else None
    return ParticleFilterResult(loglik=float(loglik_t.sum()), loglik_t=loglik_t,
                                states=states, weights=weights, ess=ess,
                                particles=particles, history=history)


def particle_smoother(filtered: ParticleFilterResult) -> ParticleSmootherResult:
    """Backward-information smoothing from a filter built with ``store=True``."""
    from scipy.special import logsumexp
    history = filtered.history
    if history is None:
        raise ValueError("particle_smoother needs particle_filter(..., store=True)")
    A, B, c = history["A"], history["B"], history["c"]
    Vinov = history["Vinov"]
    nk, second = history["nk"], history["second"]
    third = history.get("third", False)
    if second:
        ne = int(round(len(history["vec_sigma"]) ** 0.5))
        inov_mean = np.concatenate([np.zeros(ne), -history["vec_sigma"],
                                    np.zeros(nk * ne)])
    else:
        inov_mean = np.zeros(B.shape[1])
    active = np.arange(nk) if second or third else np.arange(B.shape[0])
    B_active = B[active]
    covariance = B_active @ Vinov @ B_active.T
    sign, logdet = np.linalg.slogdet(covariance)
    if third and sign <= 0:
        eigenvalues = np.linalg.eigvalsh(covariance)
        cutoff = max(float(np.max(np.abs(eigenvalues))) * 1e-12, 1e-14)
        logdet = float(np.log(eigenvalues[eigenvalues > cutoff]).sum())
        inverse = np.linalg.pinv(covariance)
    else:
        if sign <= 0:
            raise np.linalg.LinAlgError("singular transition covariance in particle smoothing")
        inverse = np.linalg.inv(covariance)
    z_history, weight_history = history["z"], history["weights"]
    backward = weight_history[-1].copy()
    mean = np.zeros((len(z_history), nk))
    ess = np.zeros(len(z_history))
    for t in range(len(z_history) - 2, -1, -1):
        current, following = z_history[t], z_history[t + 1]
        mean_next = c + current @ A.T
        if not third:
            mean_next = mean_next + inov_mean @ B.T
        mean_next = mean_next[:, active]
        logw = np.empty(len(current))
        for start in range(0, len(current), 500):
            stop = min(start + 500, len(current))
            error = following[:, active][None, :, :] - mean_next[start:stop, None, :]
            quadratic = np.sum((error @ inverse) * error, axis=-1)
            logw[start:stop] = logsumexp(
                -0.5 * (covariance.shape[0] * np.log(2 * np.pi) + logdet + quadratic)
                + np.log(backward)[None, :], axis=1)
        backward = logw.copy()
        backward -= logsumexp(backward)
        backward = np.exp(backward)
        ess[t] = 1.0 / np.sum(backward ** 2)
        state = (current[:, :nk] + current[:, nk:2 * nk] + current[:, 2 * nk:] if third else
                 current[:, :nk] + current[:, nk:2 * nk] if second else current)
        mean[t] = backward @ state
    ess[-1] = 1.0 / np.sum(backward ** 2)
    final = z_history[-1]
    final_state = (final[:, :nk] + final[:, nk:2 * nk] + final[:, 2 * nk:] if third else
                   final[:, :nk] + final[:, nk:2 * nk] if second else final)
    mean[-1] = backward @ final_state
    return ParticleSmootherResult(mean=mean, ess=ess)
