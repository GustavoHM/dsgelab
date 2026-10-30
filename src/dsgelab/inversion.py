"""Inversion filtering for first- and pruned second-order solutions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import linalg, optimize

from .kalman import Data, state_space
from .model import Covariance
from .moment_estimation import _observation_conversion, _second_order_system
from .secondorder import SecondOrderSolution
from .solution import FirstOrderSolution

__all__ = ["InversionFilterResult", "inversion_filter", "inversion_smoother"]


@dataclass
class InversionFilterResult:
    """Recovered shocks, states and the inversion log likelihood."""

    loglik: float
    loglik_t: np.ndarray
    states: np.ndarray
    shocks: np.ndarray
    jacobians: np.ndarray
    approximate: bool = False


def inversion_filter(solution: FirstOrderSolution | SecondOrderSolution, data: Data, *,
                     observables, transforms=None, cov: Covariance = None,
                     presample: int = 0, tol: float = 1e-9) -> InversionFilterResult:
    """Invert observations using minimum-norm standardized innovations.

    There must be no more observables than shocks. Measurement error is not
    supported. First order gives the exact Gaussian likelihood conditional on
    an initial state of zero, propagating uncertainty through unobserved shocks
    and missing dates. Second order starts at the pruned system's mean. It follows
    one local inverse; with missing or fewer observations than shocks it uses a
    local Gaussian approximation, reported by ``result.approximate``.
    """
    observables = list(observables)
    first = solution.first if isinstance(solution, SecondOrderSolution) else solution
    ss = state_space(first, observables, transforms=transforms,
                     cov=solution.Sigma if isinstance(solution, SecondOrderSolution) else cov)
    Y = ss.data_matrix(data)
    if Y.shape[1] > ss.Sigma.shape[0]:
        raise ValueError("the inversion filter needs at least as many shocks as observables")
    try:
        shock_factor = linalg.cholesky(ss.Sigma, lower=True)
    except linalg.LinAlgError as exc:
        raise ValueError("the structural shock covariance must be positive definite") from exc

    second = solution if isinstance(solution, SecondOrderSolution) else None
    if second is not None:
        system = _second_order_system(second)
        A, B, c = system["A"], system["B"], system["c"]
        C, D, d = system["C"], system["D"], system["d"]
        rows, base, scale = _observation_conversion(first, observables, transforms)
        C, D, d = C[rows] * scale[:, None], D[rows] * scale[:, None], base + scale * d[rows]
        z = system["Ez"].copy()
        nk = second.model.n_states
    else:
        A, B, C, d = ss.T, ss.R, ss.Z, ss.d
        c = np.zeros(ss.n_states)
        D = np.zeros((ss.n_obs, ss.R.shape[1]))
        z = np.zeros(ss.n_states)
        nk = first.model.n_states

    loglik_t = np.zeros(Y.shape[0])
    states = np.zeros((Y.shape[0], nk))
    shocks = np.zeros((Y.shape[0], ss.Sigma.shape[0]))
    jacobians = np.zeros((Y.shape[0], ss.n_obs, ss.Sigma.shape[0]))
    covariance = np.zeros((len(z), len(z)))
    approximate = second is not None and (np.isnan(Y).any() or ss.n_obs < ss.Sigma.shape[0])
    for t in range(Y.shape[0]):
        keep = ~np.isnan(Y[t])
        target = Y[t, keep]
        state_factor = _covariance_factor(covariance)
        rank = state_factor.shape[1]
        dimension = rank + ss.Sigma.shape[0]

        def evaluate(v):
            previous = z + state_factor @ v[:rank]
            u = shock_factor @ v[rank:]
            if second is None:
                following = A @ previous + B @ u
                observation = d + C @ following
                G = np.column_stack([A @ state_factor, B @ shock_factor])
                J = C @ G
                shock_jacobian = C @ B
            else:
                inov = _innovation(previous, u, system)
                Ju = _innovation_jacobian(previous, u, nk)
                Jz = np.zeros((len(inov), len(z)))
                Jz[-nk * len(u):, :nk] = np.kron(np.eye(nk), u[:, None])
                following = c + A @ previous + B @ inov
                observation = d + C @ previous + D @ inov
                G = np.column_stack([(A + B @ Jz) @ state_factor, B @ Ju @ shock_factor])
                shock_jacobian = D @ Ju
                J = np.column_stack([(C + D @ Jz) @ state_factor, shock_jacobian @ shock_factor])
            return following, u, observation[keep] - target, J[keep], G, shock_jacobian[keep]

        v = np.zeros(dimension)
        if target.size:
            _, _, residual0, J0, _, _ = evaluate(v)
            v = linalg.lstsq(J0, -residual0)[0]
            nonlinear_free_directions = second is not None and target.size < dimension
            if nonlinear_free_directions or np.max(np.abs(evaluate(v)[2])) > tol:
                if target.size == dimension:
                    fit = optimize.least_squares(
                        lambda x: evaluate(x)[2], v, jac=lambda x: evaluate(x)[3],
                        method="lm", ftol=tol, xtol=tol, gtol=tol, max_nfev=300)
                else:
                    fit = optimize.minimize(
                        lambda x: 0.5 * x @ x, v, jac=lambda x: x, method="SLSQP",
                        constraints={"type": "eq", "fun": lambda x: evaluate(x)[2],
                                     "jac": lambda x: evaluate(x)[3]},
                        options={"ftol": tol, "maxiter": 300})
                    if not fit.success:
                        raise RuntimeError(f"minimum-norm inversion failed at date {t}: {fit.message}")
                v = fit.x
        following, u, residual, J, G, shock_jacobian = evaluate(v)
        if target.size:
            if np.max(np.abs(residual)) > 100 * tol:
                raise RuntimeError(f"no inversion found at date {t}")
            # J acts on standardized prior uncertainty and shocks. Its nonzero
            # singular values supply the rectangular observation-volume factor.
            _, singular, right = linalg.svd(J, full_matrices=True)
            if singular[-1] <= singular[0] * 1e-12:
                raise np.linalg.LinAlgError(f"singular observation Jacobian at date {t}")
            loglik_t[t] = -0.5 * (2 * np.log(singular).sum() + v @ v
                                 + target.size * np.log(2 * np.pi))
            # The null space is the uncertainty left after conditioning. It
            # vanishes for a square invertible map, without cancellation error.
            remaining = G @ right[target.size:].T
            jacobians[t, keep] = shock_jacobian
        else:
            remaining = G
        covariance = remaining @ remaining.T
        z = following
        shocks[t] = u
        states[t] = z[:nk] + z[nk:2 * nk] if second is not None else z
    return InversionFilterResult(loglik=float(np.sum(loglik_t[presample:])),
                                 loglik_t=loglik_t, states=states, shocks=shocks,
                                 jacobians=jacobians, approximate=bool(approximate))


def _covariance_factor(covariance):
    """A factor containing only the uncertain directions of a PSD matrix."""
    values, vectors = linalg.eigh(0.5 * (covariance + covariance.T))
    keep = values > np.max(values, initial=0.0) * 1e-12
    return vectors[:, keep] * np.sqrt(values[keep])


def inversion_smoother(solution: FirstOrderSolution | SecondOrderSolution, data: Data, *,
                       observables, transforms=None, cov: Covariance = None,
                       presample: int = 0, tol: float = 1e-9) -> InversionFilterResult:
    """Return the inversion filter's paths and approximation flag, without a backward pass."""
    return inversion_filter(solution, data, observables=observables, transforms=transforms,
                            cov=cov, presample=presample, tol=tol)


def _innovation(z, u, system):
    nk = system["nk"]
    return np.concatenate([u, np.outer(u, u).reshape(-1, order="F") - system["vec_sigma"],
                           np.outer(z[:nk], u).ravel()])


def _innovation_jacobian(z, u, nk):
    ne = len(u)
    J_uu = np.zeros((ne * ne, ne))
    for a in range(ne):
        for b in range(ne):
            for j in range(ne):
                J_uu[a + ne * b, j] = int(a == j) * u[b] + int(b == j) * u[a]
    J_xu = np.zeros((nk * ne, ne))
    for a in range(nk):
        for b in range(ne):
            J_xu[a * ne + b, b] = z[a]
    return np.block([[np.eye(ne)], [J_uu], [J_xu]])
