"""No-U-Turn sampling for differentiable estimation problems."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .estimation import EstimationProblem, MCMCResult, ModeResult
from .hmc import _check, _DualAveraging, _initial_step, _leapfrog, _mass_matrix, _start

__all__ = ["NUTSResult", "nuts"]

_MAX_ENERGY_ERROR = 1000.0


@dataclass
class NUTSResult(MCMCResult):
    """NUTS chains with divergence and tree-depth diagnostics, counted after burn-in."""

    divergences: np.ndarray
    max_depth_hit: np.ndarray
    mean_depth: np.ndarray


@dataclass
class _Point:
    q: np.ndarray
    p: np.ndarray
    evaluated: object


@dataclass
class _Tree:
    """A trajectory segment in the order it was integrated, from ``begin`` to ``end``."""

    begin: _Point
    end: _Point
    proposal: _Point
    log_weight: float
    rho: np.ndarray
    accept_sum: float
    steps: int
    turning: bool
    divergent: bool


def _turning(p_begin, p_end, rho, C):
    """Betancourt's generalised criterion, which does not depend on the direction."""
    return bool((C @ p_begin) @ rho <= 0 or (C @ p_end) @ rho <= 0)


def _build_tree(problem, start, direction, depth, step, joint0, C, rng):
    if depth == 0:
        q, p, evaluated, valid = _leapfrog(problem, start.q, start.p,
                                           start.evaluated.gradient, direction * step, C)
        point = _Point(q, p, evaluated)
        joint = evaluated.logposterior - 0.5 * p @ C @ p if valid else -np.inf
        divergent = not np.isfinite(joint) or joint0 - joint > _MAX_ENERGY_ERROR
        return _Tree(point, point, point, joint - joint0 if not divergent else -np.inf, p.copy(),
                     float(np.exp(min(0.0, joint - joint0))) if np.isfinite(joint) else 0.0,
                     1, False, divergent)
    first = _build_tree(problem, start, direction, depth - 1, step, joint0, C, rng)
    if first.turning or first.divergent:
        return first
    second = _build_tree(problem, first.end, direction, depth - 1, step, joint0, C, rng)
    log_weight = np.logaddexp(first.log_weight, second.log_weight)
    proposal = first.proposal
    if np.log(rng.random()) < second.log_weight - log_weight:
        proposal = second.proposal
    rho = first.rho + second.rho
    turning = second.turning or (
        _turning(first.begin.p, second.end.p, rho, C)
        or _turning(first.begin.p, second.begin.p, first.rho + second.begin.p, C)
        or _turning(first.end.p, second.end.p, first.end.p + second.rho, C))
    return _Tree(first.begin, second.end, proposal, log_weight, rho,
                 first.accept_sum + second.accept_sum, first.steps + second.steps,
                 turning, second.divergent)


def _transition(problem, current, step, max_depth, C, L, rng):
    """One NUTS draw: multinomial sampling over a trajectory doubled until it turns."""
    p0 = L @ rng.standard_normal(len(current.q))
    start = _Point(current.q, p0, current.evaluated)
    joint0 = current.evaluated.logposterior - 0.5 * p0 @ C @ p0
    minus = plus = start
    proposal = start
    log_weight = 0.0
    rho = p0.copy()
    accept_sum, steps, depth = 0.0, 0, 0
    divergent = False
    while depth < max_depth:
        direction = 1.0 if rng.random() < 0.5 else -1.0
        tree = _build_tree(problem, plus if direction > 0 else minus, direction, depth,
                           step, joint0, C, rng)
        accept_sum += tree.accept_sum
        steps += tree.steps
        depth += 1
        if tree.divergent:
            divergent = True
            break
        if tree.turning:
            break
        if np.log(rng.random()) < tree.log_weight - log_weight:
            proposal = tree.proposal
        log_weight = np.logaddexp(log_weight, tree.log_weight)
        # the segment was integrated away from the trajectory, so its end is the new edge
        if direction > 0:
            plus = tree.end
        else:
            minus = tree.end
        rho = rho + tree.rho
        if _turning(minus.p, plus.p, rho, C):
            break
    return proposal, accept_sum / max(steps, 1), depth, divergent


def nuts(problem: EstimationProblem, mode: ModeResult | None = None, *,
         draws: int = 1000, chains: int = 2, step_size: float = 0.05,
         max_depth: int = 8, burn: float = 0.5, target: float = 0.8,
         adapt: bool = True, init_scale: float = 1.0, seed: int = 0) -> NUTSResult:
    """Sample with the No-U-Turn algorithm (Hoffman and Gelman, 2014; Betancourt, 2017).

    Each draw doubles a trajectory in a random direction until the generalised U-turn
    criterion holds or ``max_depth`` doublings are reached, and picks a point from it
    with multinomial weights. The mode's covariance is the inverse mass matrix. With
    ``adapt``, dual averaging tunes the step during burn-in toward ``target``.
    Divergences, maximum-depth hits and depths are reported after burn-in.
    """
    _check(draws, chains, step_size, burn, target, init_scale)
    if max_depth < 1:
        raise ValueError("max_depth must be positive")
    rng = np.random.default_rng(seed)
    start, C, M = _mass_matrix(problem, mode)
    k = len(problem.names)
    L = np.linalg.cholesky(M)
    kept = np.empty((chains, draws, k))
    lps = np.empty((chains, draws))
    acceptance = np.zeros(chains)
    divergences: np.ndarray = np.zeros(chains, dtype=int)
    max_hit: np.ndarray = np.zeros(chains, dtype=int)
    mean_depth = np.zeros(chains)
    b = int(burn * draws)

    for c in range(chains):
        q, evaluated = _start(problem, start, C, init_scale, rng, "NUTS")
        current = _Point(q, np.zeros(k), evaluated)
        step = float(step_size)
        if adapt and b:
            step = _initial_step(problem, q, evaluated, step, C, L, rng)
            averaging = _DualAveraging(step, target)
        for d in range(draws):
            current, accept_stat, depth, divergent = _transition(
                problem, current, step, max_depth, C, L, rng)
            if adapt and d < b:
                step = averaging.update(accept_stat)
                if d == b - 1:
                    step = averaging.final()
            if d >= b:
                acceptance[c] += accept_stat
                divergences[c] += divergent
                max_hit[c] += depth >= max_depth and not divergent
                mean_depth[c] += depth
            kept[c, d], lps[c, d] = current.q, current.evaluated.logposterior
    n = draws - b
    return NUTSResult(names=problem.names, chains=kept[:, b:], log_posterior=lps[:, b:],
                      acceptance=acceptance / n, burn=b, divergences=divergences,
                      max_depth_hit=max_hit, mean_depth=mean_depth / n)
