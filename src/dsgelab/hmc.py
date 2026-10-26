"""Hamiltonian Monte Carlo for differentiable estimation problems."""

from __future__ import annotations

import numpy as np

from .derivatives import log_posterior_gradient
from .estimation import EstimationProblem, MCMCResult, ModeResult

__all__ = ["hmc"]


def _check(draws, chains, step_size, burn, target, init_scale):
    if draws < 1 or chains < 1:
        raise ValueError("draws and chains must be positive")
    if step_size <= 0:
        raise ValueError("step_size must be positive")
    if not 0 <= burn < 1:
        raise ValueError("burn must lie in [0, 1)")
    if not 0 < target < 1:
        raise ValueError("target must lie in (0, 1)")
    if init_scale < 0:
        raise ValueError("init_scale must be nonnegative")


def _mass_matrix(problem, mode):
    """Start, inverse mass matrix ``C`` (the mode covariance) and mass matrix ``M``."""
    k = len(problem.names)
    start = mode.x.copy() if mode is not None else problem.start()
    C = np.eye(k)
    M = np.eye(k)
    if mode is not None and np.all(np.isfinite(mode.covariance)):
        values, vectors = np.linalg.eigh(0.5 * (mode.covariance + mode.covariance.T))
        if np.min(values) > 0:
            C = vectors @ np.diag(values) @ vectors.T
            M = vectors @ np.diag(1 / values) @ vectors.T
    return start, 0.5 * (C + C.T), 0.5 * (M + M.T)


def _start(problem, start, C, init_scale, rng, name):
    """A finite starting point, overdispersed around ``start`` by ``init_scale``."""
    factor = np.linalg.cholesky(C)
    for _ in range(1000):
        candidate = start + init_scale * factor @ rng.standard_normal(len(start))
        evaluated = log_posterior_gradient(problem, candidate)
        if np.isfinite(evaluated.logposterior):
            return candidate, evaluated
        if not init_scale:
            break
    evaluated = log_posterior_gradient(problem, start)
    if not np.isfinite(evaluated.logposterior):
        raise ValueError(f"no finite posterior point to start the {name} chain")
    return start.copy(), evaluated


def _leapfrog(problem, q, p, gradient, step, C):
    p = p + 0.5 * step * gradient
    q = q + step * C @ p
    evaluated = log_posterior_gradient(problem, q)
    if not np.isfinite(evaluated.logposterior) or not np.all(np.isfinite(evaluated.gradient)):
        return q, p, evaluated, False
    return q, p + 0.5 * step * evaluated.gradient, evaluated, True


def _initial_step(problem, q, evaluated, step, C, L, rng):
    """Halve or double the step until one leapfrog step crosses acceptance 1/2.

    Hoffman and Gelman (2014), Algorithm 4, with the mass matrix.
    """
    p = L @ rng.standard_normal(len(q))
    joint0 = evaluated.logposterior - 0.5 * p @ C @ p

    def log_ratio(eps):
        _, p_new, new, ok = _leapfrog(problem, q, p, evaluated.gradient, eps, C)
        return new.logposterior - 0.5 * p_new @ C @ p_new - joint0 if ok else -np.inf

    ratio = log_ratio(step)
    direction = 1.0 if ratio > np.log(0.5) else -1.0
    for _ in range(50):
        if direction * ratio <= direction * np.log(0.5):
            break
        step *= 2.0 ** direction
        ratio = log_ratio(step)
    return step


class _DualAveraging:
    """Step-size adaptation toward a target acceptance (Hoffman and Gelman, 2014, §3.2)."""

    def __init__(self, step, target, gamma=0.05, t0=10.0, kappa=0.75):
        self.mu = np.log(10 * step)
        self.target, self.gamma, self.t0, self.kappa = target, gamma, t0, kappa
        self.m = 0
        self.h_bar = 0.0
        self.log_step = np.log(step)
        self.log_step_bar = 0.0

    def update(self, acceptance):
        self.m += 1
        w = 1.0 / (self.m + self.t0)
        self.h_bar = (1 - w) * self.h_bar + w * (self.target - acceptance)
        self.log_step = self.mu - np.sqrt(self.m) / self.gamma * self.h_bar
        eta = self.m ** -self.kappa
        self.log_step_bar = eta * self.log_step + (1 - eta) * self.log_step_bar
        return float(np.exp(self.log_step))

    def final(self):
        return float(np.exp(self.log_step_bar))


def hmc(problem: EstimationProblem, mode: ModeResult | None = None, *,
        draws: int = 1000, chains: int = 2, step_size: float = 0.05,
        leapfrog: int = 10, burn: float = 0.5, target: float = 0.8,
        adapt: bool = True, init_scale: float = 1.0, seed: int = 0) -> MCMCResult:
    """Sample the posterior with Hamiltonian dynamics and a Metropolis correction.

    The mode's covariance is the inverse mass matrix when available. With ``adapt``,
    ``step_size`` is only a starting value: a heuristic first finds a step with
    acceptance near one half, and dual averaging moves it toward ``target`` during
    burn-in, after which it stays fixed. Each trajectory multiplies the step by a
    uniform draw on [0.8, 1.2], since a fixed trajectory length can return the chain
    close to its start (Neal, 2011, §5.4.2). Every chain contains ``draws`` points, of
    which the fraction ``burn`` is discarded. ``acceptance`` is the mean acceptance
    probability after burn-in.
    """
    _check(draws, chains, step_size, burn, target, init_scale)
    if leapfrog < 1:
        raise ValueError("leapfrog must be positive")
    rng = np.random.default_rng(seed)
    start, C, M = _mass_matrix(problem, mode)
    k = len(problem.names)
    L = np.linalg.cholesky(M)
    kept = np.empty((chains, draws, k))
    lps = np.empty((chains, draws))
    acceptance = np.zeros(chains)
    b = int(burn * draws)

    for c in range(chains):
        current, evaluated = _start(problem, start, C, init_scale, rng, "HMC")
        step = float(step_size)
        if adapt and b:
            step = _initial_step(problem, current, evaluated, step, C, L, rng)
            averaging = _DualAveraging(step, target)
        for d in range(draws):
            p0 = L @ rng.standard_normal(k)
            joint0 = evaluated.logposterior - 0.5 * p0 @ C @ p0
            q, p, new = current, p0, evaluated
            valid = True
            jittered = step * rng.uniform(0.8, 1.2)
            for _ in range(leapfrog):
                q, p, new, valid = _leapfrog(problem, q, p, new.gradient, jittered, C)
                if not valid:
                    break
            log_accept = new.logposterior - 0.5 * p @ C @ p - joint0 if valid else -np.inf
            accept_prob = float(np.exp(min(0.0, log_accept)))
            if rng.random() < accept_prob:
                current, evaluated = q, new
            if adapt and d < b:
                step = averaging.update(accept_prob)
                if d == b - 1:
                    step = averaging.final()
            if d >= b:
                acceptance[c] += accept_prob
            kept[c, d], lps[c, d] = current, evaluated.logposterior
    return MCMCResult(names=problem.names, chains=kept[:, b:], log_posterior=lps[:, b:],
                      acceptance=acceptance / (draws - b), burn=b)
