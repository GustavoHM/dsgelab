"""Gradients of the Gaussian Kalman likelihood."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .kalman import StateSpace, _initial

__all__ = ["LikelihoodGradient", "kalman_likelihood_gradient"]


@dataclass
class LikelihoodGradient:
    """Likelihood and its derivatives in the order of the supplied names."""

    loglik: float
    loglik_t: np.ndarray
    gradient: np.ndarray
    gradient_t: np.ndarray


def kalman_likelihood_gradient(ss: StateSpace, data, derivatives) -> LikelihoodGradient:
    """Evaluate the likelihood and derivatives of its state-space matrices.

    ``derivatives`` maps a parameter name to a dictionary with ``T``, ``R``,
    ``Sigma``, ``Z``, ``d`` and ``H``. Missing observations use the univariate
    recursion, as does diffuse initialization.
    """
    from scipy.linalg import solve_discrete_lyapunov

    Y = ss.data_matrix(data)
    names = list(derivatives)
    a, P, Pinf = _initial(ss, "auto")
    if np.any(Pinf):
        return _diffuse_gradient(ss, Y, a, P, Pinf, derivatives)
    npar = len(names)
    Tn, n = Y.shape[0], ss.n_states
    ll_t = np.zeros(Tn)
    dll_t = np.zeros((Tn, npar))
    da = np.zeros((npar, n))
    dP = np.zeros((npar, n, n))
    RQR = ss.R @ ss.Sigma @ ss.R.T
    for j, name in enumerate(names):
        D = derivatives[name]
        dRQR = (D["R"] @ ss.Sigma @ ss.R.T + ss.R @ D["Sigma"] @ ss.R.T
                + ss.R @ ss.Sigma @ D["R"].T)
        rhs = (dRQR + D["T"] @ P @ ss.T.T + ss.T @ P @ D["T"].T)
        dP[j] = solve_discrete_lyapunov(ss.T, rhs) if n else np.zeros((0, 0))
    if np.any(Pinf) or np.isnan(Y).any():
        return _univariate_gradient(ss, Y, a, P, Pinf, derivatives, dP)

    for t in range(Tn):
        v = Y[t] - ss.d - ss.Z @ a
        F = ss.Z @ P @ ss.Z.T + ss.H
        F = 0.5 * (F + F.T)
        Finv = np.linalg.inv(F)
        ll_t[t] = -0.5 * (len(v) * np.log(2 * np.pi)
                          + np.linalg.slogdet(F)[1] + v @ Finv @ v)
        K = np.linalg.solve(F, ss.Z @ P).T
        for j, name in enumerate(names):
            D = derivatives[name]
            dv = -D["d"] - D["Z"] @ a - ss.Z @ da[j]
            dF = (D["Z"] @ P @ ss.Z.T + ss.Z @ dP[j] @ ss.Z.T
                  + ss.Z @ P @ D["Z"].T + D["H"])
            dF = 0.5 * (dF + dF.T)
            dll_t[t, j] = -0.5 * (np.trace(Finv @ dF) + 2 * dv @ Finv @ v
                                  - v @ Finv @ dF @ Finv @ v)
            dK = (dP[j] @ ss.Z.T @ Finv + P @ D["Z"].T @ Finv
                  - P @ ss.Z.T @ Finv @ dF @ Finv)
            da[j] = da[j] + dK @ v + K @ dv
            dP[j] = (dP[j] - dK @ ss.Z @ P - K @ D["Z"] @ P
                     - K @ ss.Z @ dP[j])
        a = a + K @ v
        P = P - K @ ss.Z @ P
        P = 0.5 * (P + P.T)
        for j, name in enumerate(names):
            D = derivatives[name]
            da[j] = ss.T @ da[j] + D["T"] @ a
            dRQR = (D["R"] @ ss.Sigma @ ss.R.T + ss.R @ D["Sigma"] @ ss.R.T
                    + ss.R @ ss.Sigma @ D["R"].T)
            dP[j] = (ss.T @ dP[j] @ ss.T.T + D["T"] @ P @ ss.T.T
                     + ss.T @ P @ D["T"].T + dRQR)
            dP[j] = 0.5 * (dP[j] + dP[j].T)
        a = ss.T @ a
        P = ss.T @ P @ ss.T.T + RQR
        P = 0.5 * (P + P.T)
    return LikelihoodGradient(loglik=float(ll_t.sum()), loglik_t=ll_t,
                              gradient=dll_t.sum(axis=0), gradient_t=dll_t)


def _diffuse_gradient(ss, Y, a, P, Pinf, derivatives):
    from scipy.linalg import schur, solve_discrete_lyapunov

    RQR = ss.R @ ss.Sigma @ ss.R.T
    S, U, nu = schur(ss.T, output="real",
                     sort=lambda re, im: np.hypot(re, im) >= 1 - 1e-6)
    Pb = U.T @ P @ U
    dP = np.zeros((len(derivatives), ss.n_states, ss.n_states))
    for j, name in enumerate(derivatives):
        D = derivatives[name]
        dRQR = (D["R"] @ ss.Sigma @ ss.R.T + ss.R @ D["Sigma"] @ ss.R.T
                + ss.R @ ss.Sigma @ D["R"].T)
        dUb = U.T @ dRQR @ U
        dS = U.T @ D["T"] @ U
        stable = slice(nu, ss.n_states)
        rhs = (dUb[stable, stable] + dS[stable, stable] @ Pb[stable, stable] @ S[stable, stable].T
               + S[stable, stable] @ Pb[stable, stable] @ dS[stable, stable].T)
        block = solve_discrete_lyapunov(S[stable, stable], rhs)
        full = np.zeros((ss.n_states, ss.n_states))
        full[stable, stable] = block
        dP[j] = U @ full @ U.T
    return _univariate_gradient(ss, Y, a, P, Pinf, derivatives, dP)


def _univariate_gradient(ss, Y, a, P, Pinf, derivatives, dP, tol=1e-12):
    if not np.allclose(ss.H, np.diag(np.diag(ss.H))):
        raise NotImplementedError("univariate likelihood gradients with non-diagonal H")
    names = list(derivatives)
    npar = len(names)
    Tn, n = Y.shape[0], ss.n_states
    h = np.diag(ss.H)
    ll_t = np.zeros(Tn)
    dll_t = np.zeros((Tn, npar))
    da = np.zeros((npar, n))
    dPinf = np.zeros((npar, n, n))
    RQR = ss.R @ ss.Sigma @ ss.R.T
    for t in range(Tn):
        in_diffuse = bool(np.max(np.abs(Pinf), initial=0.0) > tol)
        for i, z in enumerate(ss.Z):
            yi = Y[t, i]
            if np.isnan(yi):
                continue
            v = yi - ss.d[i] - z @ a
            M = P @ z
            F = z @ M + h[i]
            if in_diffuse:
                Mi = Pinf @ z
                Fi = z @ Mi
                if Fi > tol:
                    K0 = Mi / Fi
                    ll_t[t] += -0.5 * (np.log(2 * np.pi) + np.log(Fi))
                    for j, name in enumerate(names):
                        D = derivatives[name]
                        dv = -D["d"][i] - D["Z"][i] @ a - z @ da[j]
                        dMi = dPinf[j] @ z + Pinf @ D["Z"][i]
                        dFi = D["Z"][i] @ Mi + z @ dMi
                        dM = dP[j] @ z + P @ D["Z"][i]
                        dF = D["Z"][i] @ M + z @ dM + D["H"][i, i]
                        dll_t[t, j] += -0.5 * dFi / Fi
                        dK0 = dMi / Fi - K0 * dFi / Fi
                        da[j] += dK0 * v + K0 * dv
                        dP[j] += ((dMi[:, None] * Mi[None, :] + Mi[:, None] * dMi[None, :])
                                  * F / Fi ** 2 + np.outer(Mi, Mi) * dF / Fi ** 2
                                  - 2 * np.outer(Mi, Mi) * F * dFi / Fi ** 3
                                  - (np.outer(dM, Mi) + np.outer(M, dMi)
                                     + np.outer(dMi, M) + np.outer(Mi, dM)) / Fi
                                  + (np.outer(M, Mi) + np.outer(Mi, M)) * dFi / Fi ** 2)
                        dPinf[j] -= ((dMi[:, None] * Mi[None, :]
                                      + Mi[:, None] * dMi[None, :]) / Fi
                                     - np.outer(Mi, Mi) * dFi / Fi ** 2)
                    a = a + K0 * v
                    P = P + np.outer(Mi, Mi) * F / Fi ** 2 - (np.outer(M, Mi) + np.outer(Mi, M)) / Fi
                    Pinf = Pinf - np.outer(Mi, Mi) / Fi
                    continue
            if F > tol:
                K = M / F
                ll_t[t] += -0.5 * (np.log(2 * np.pi) + np.log(F) + v * v / F)
                for j, name in enumerate(names):
                    D = derivatives[name]
                    dv = -D["d"][i] - D["Z"][i] @ a - z @ da[j]
                    dM = dP[j] @ z + P @ D["Z"][i]
                    dF = D["Z"][i] @ M + z @ dM + D["H"][i, i]
                    dll_t[t, j] += -0.5 * (dF / F + 2 * v * dv / F - v * v * dF / F ** 2)
                    dK = (dM - K * dF) / F
                    da[j] += dK * v + K * dv
                    dP[j] -= ((dM[:, None] * M[None, :] + M[:, None] * dM[None, :]) / F
                              - np.outer(M, M) * dF / F ** 2)
                a = a + K * v
                P = P - np.outer(M, M) / F
        P = 0.5 * (P + P.T)
        dP = 0.5 * (dP + np.transpose(dP, (0, 2, 1)))
        for j, name in enumerate(names):
            D = derivatives[name]
            da[j] = ss.T @ da[j] + D["T"] @ a
            dRQR = (D["R"] @ ss.Sigma @ ss.R.T + ss.R @ D["Sigma"] @ ss.R.T
                    + ss.R @ ss.Sigma @ D["R"].T)
            dP[j] = (ss.T @ dP[j] @ ss.T.T + D["T"] @ P @ ss.T.T
                     + ss.T @ P @ D["T"].T + dRQR)
            dP[j] = 0.5 * (dP[j] + dP[j].T)
            if in_diffuse:
                dPinf[j] = (ss.T @ dPinf[j] @ ss.T.T + D["T"] @ Pinf @ ss.T.T
                            + ss.T @ Pinf @ D["T"].T)
                dPinf[j] = 0.5 * (dPinf[j] + dPinf[j].T)
        a = ss.T @ a
        P = ss.T @ P @ ss.T.T + RQR
        P = 0.5 * (P + P.T)
        if in_diffuse:
            Pinf = ss.T @ Pinf @ ss.T.T
            Pinf = 0.5 * (Pinf + Pinf.T)
    return LikelihoodGradient(loglik=float(ll_t.sum()), loglik_t=ll_t,
                              gradient=dll_t.sum(axis=0), gradient_t=dll_t)
