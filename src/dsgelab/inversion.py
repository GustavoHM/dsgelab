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
    """Recovered shocks, states and the exact log likelihood."""

    loglik: float
    loglik_t: np.ndarray
    states: np.ndarray
    shocks: np.ndarray
    jacobians: np.ndarray


def inversion_filter(solution: FirstOrderSolution | SecondOrderSolution, data: Data, *,
                     observables, transforms=None, cov: Covariance = None,
                     presample: int = 0, tol: float = 1e-9) -> InversionFilterResult:
    """Recover minimum-norm shocks that match the data and evaluate their density.

    There must be no more observables than shocks. Measurement error is not
    supported. Higher-order inversions start at the origin, as in MacroModelling.
    """
    observables = list(observables)
    first = solution.first if isinstance(solution, SecondOrderSolution) else solution
    ss = state_space(first, observables, transforms=transforms,
                     cov=solution.Sigma if isinstance(solution, SecondOrderSolution) else cov)
    Y = ss.data_matrix(data)
    if Y.shape[1] > ss.Sigma.shape[0]:
        raise ValueError("the inversion filter needs at least as many shocks as observables")
    sign, logdet_cov = np.linalg.slogdet(ss.Sigma)
    if sign <= 0:
        raise ValueError("the structural shock covariance must be positive definite")

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
    for t in range(Y.shape[0]):
        missing = np.isnan(Y[t])
        if np.all(missing):
            u = np.zeros(ss.Sigma.shape[0])
            jac = np.zeros((0, ss.Sigma.shape[0]))
            ll = 0.0
        else:
            target = Y[t, ~missing]
            if second is None:
                u = linalg.lstsq(C[~missing] @ B, target - d[~missing] - C[~missing] @ A @ z)[0]
                jac = C[~missing] @ B
            else:
                Cw, Dw, dw = C[~missing], D[~missing], d[~missing]

                def residual(u):
                    return dw + Cw @ z + Dw @ _innovation(z, u, system) - target

                def jacobian(u):
                    return Dw @ _innovation_jacobian(z, u, system["nk"])

                fit = optimize.least_squares(residual, np.zeros(ss.Sigma.shape[0]),
                                             jac=jacobian, method="lm", x_scale="jac",
                                             ftol=tol, xtol=tol, gtol=tol, max_nfev=300)
                u = fit.x
                jac = jacobian(u)
                if np.max(np.abs(residual(u))) > 100 * tol:
                    raise RuntimeError(f"no inversion found at date {t}")
            sign_j, logdet_j = np.linalg.slogdet(jac)
            if sign_j == 0:
                raise np.linalg.LinAlgError(f"singular observation Jacobian at date {t}")
            ll = -0.5 * (logdet_cov + 2 * logdet_j + u @ np.linalg.solve(ss.Sigma, u)
                         + len(target) * np.log(2 * np.pi))
        loglik_t[t] = ll
        shocks[t] = u
        if not np.all(missing):
            jacobians[t] = jac
        if second is not None:
            z = c + A @ z + B @ _innovation(z, u, system)
            states[t] = z[:nk] + z[nk:2 * nk]
        else:
            z = c + A @ z + B @ u
            states[t] = z
    return InversionFilterResult(loglik=float(np.sum(loglik_t[presample:])),
                                 loglik_t=loglik_t, states=states, shocks=shocks,
                                 jacobians=jacobians)


def inversion_smoother(solution: FirstOrderSolution | SecondOrderSolution, data: Data, *,
                       observables, transforms=None, cov: Covariance = None,
                       presample: int = 0, tol: float = 1e-9) -> InversionFilterResult:
    """Return the exact inverted state and shock paths, with no backward approximation."""
    return inversion_filter(solution, data, observables=observables, transforms=transforms,
                            cov=cov, presample=presample, tol=tol)


def _innovation(z, u, system):
    nk = system["nk"]
    return np.concatenate([u, np.outer(u, u).reshape(-1, order="F") - system["vec_sigma"],
                           np.outer(z[:nk], u).reshape(-1, order="F")])


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
            J_xu[a + nk * b, b] = z[a]
    return np.block([[np.eye(ne)], [J_uu], [J_xu]])
