"""Second-order perturbation (Schmitt-Grohe and Uribe, 2004).

Why it is here rather than left to the first-order solution: a first-order
simulation holds the mean of the states at the deterministic steady state BY
CONSTRUCTION. Every mean effect it then reports is the curvature of the static
mapping alone, and the displacement of the decision rules themselves is missing.
When that displacement is large -- and in models with a convex adjustment cost
and an occasionally-tight margin it routinely is -- the first-order simulation
can get the SIGN of a mean effect wrong, not merely its size. Welfare rankings
and certainty-equivalent comparisons are exactly the objects that depend on it.

Solution form, with ``k`` the states, ``u`` the jumps and ``sigma`` the
perturbation parameter on the innovations::

    u_t     = F k_t + 1/2 g_kk[k_t, k_t] + 1/2 g_ss
    k_{t+1} = P k_t + 1/2 h_kk[k_t, k_t] + 1/2 h_ss + Q eps_{t+1}

all in the rescaled deviations of :mod:`dsgelab.linearize`. The ``g_ss`` and
``h_ss`` terms are the risk corrections: they are what first order sets to zero.

The quadratic coefficients solve a generalised Sylvester equation

    Abar W + Bbar W (P kron P) = -K

whose blocks come from the first and second derivatives of the residuals. For
the model sizes this is written for the equation is solved DIRECTLY; a
matrix-free GMRES takes over above a size threshold, since the direct form costs
``(n * n_k^2)^2`` entries and that grows fast.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import solve_discrete_lyapunov
from scipy.sparse.linalg import LinearOperator, lgmres

from .linearize import second_derivatives
from .model import Covariance, Model
from .solution import FirstOrderSolution, shock_covariance

__all__ = ["SecondOrderSolution", "solve_second_order"]


@dataclass
class SecondOrderSolution:
    """First-order matrices plus the quadratic and risk-correction terms."""

    first: FirstOrderSolution
    g_kk: np.ndarray            # (n_u, n_k, n_k)
    h_kk: np.ndarray            # (n_k, n_k, n_k)
    g_ss: np.ndarray            # (n_u,)
    h_ss: np.ndarray            # (n_k,)
    Sigma: np.ndarray = field(repr=False)
    sylvester_residual: float = 0.0

    @property
    def model(self) -> Model:
        return self.first.model

    @property
    def P(self) -> np.ndarray:
        return self.first.P

    @property
    def F(self) -> np.ndarray:
        return self.first.F

    @property
    def Q(self) -> np.ndarray:
        return self.first.Q

    # -- decision rules --------------------------------------------------
    def jumps(self, k: np.ndarray, risk: bool = True) -> np.ndarray:
        k = np.asarray(k, dtype=float)
        out = self.F @ k + 0.5 * np.einsum("rab,a,b->r", self.g_kk, k, k)
        return out + 0.5 * self.g_ss if risk else out

    def next_states(self, k: np.ndarray, eps: np.ndarray | None = None,
                    risk: bool = True) -> np.ndarray:
        k = np.asarray(k, dtype=float)
        out = self.P @ k + 0.5 * np.einsum("rab,a,b->r", self.h_kk, k, k)
        if risk:
            out = out + 0.5 * self.h_ss
        if eps is not None and self.Q is not None:
            out = out + self.Q @ eps
        return out

    def irf(self, name: str, size: float = 0.01, periods: int = 40, *,
            n_draws: int = 0, seed: int = 0, state: ArrayLike | None = None,
            percent: bool = False) -> dict[str, np.ndarray]:
        """Generalised impulse response of the pruned second-order rule.

        The path with the innovation ``size`` in period 0, minus the path with
        that innovation set to zero, from the same state and with no further
        shocks. ``state`` is a deviation vector of the predetermined variables.
        Absent, the steady state is used when ``n_draws`` is 0, and otherwise
        the difference is averaged over ``n_draws`` states drawn from the
        pruned ergodic distribution. A linear model gives the first-order
        impulse from every state. A curved rule does not: the response depends
        on the state it arrives in, and a positive shock is not the mirror of
        a negative one.
        """
        nk = self.P.shape[0]
        if state is not None:
            starts = [np.asarray(state, dtype=float)]
        elif n_draws <= 0:
            starts = [np.zeros(nk)]
        else:
            drawn = self.simulate(periods=n_draws, seed=seed, burn=50, pruning=True)
            starts = [np.array([drawn[n][i] for n in self.model.states])
                      for i in range(n_draws)]
        acc = None
        for k0 in starts:
            shocked = self._conditional_path(k0, name, size, periods)
            base = self._conditional_path(k0, name, 0.0, periods)
            diff = {n: shocked[n] - base[n] for n in shocked}
            if acc is None:
                acc = diff
            else:
                for n in acc:
                    acc[n] = acc[n] + diff[n]
        out = {n: v / len(starts) for n, v in acc.items()}
        if percent:
            out = {n: v * 100.0 for n, v in out.items()}
        return out

    def _conditional_path(self, k0, shock, size, periods):
        nk = self.P.shape[0]
        k = np.zeros((periods, nk))
        u = np.zeros((periods, self.F.shape[0]))
        k[0] = np.asarray(k0, dtype=float)
        eps = np.zeros(self.model.n_shocks)
        if size != 0.0:
            eps[self.model.shock_position(shock)] = size
        u[0] = self.jumps(k[0])
        if periods > 1:
            k[1] = self.next_states(k[0], eps)
        for t in range(1, periods - 1):
            u[t] = self.jumps(k[t])
            k[t + 1] = self.next_states(k[t], None)
        if periods > 1:
            u[-1] = self.jumps(k[-1])
        paths = {n: k[:, i] for i, n in enumerate(self.model.states)}
        paths.update({n: u[:, i] for i, n in enumerate(self.model.jumps)})
        return paths

    # -- simulation ------------------------------------------------------
    def simulate(self, periods: int = 1000, cov: Covariance = None, *,
                 seed: int | None = None, burn: int = 200, pruning: bool = True,
                 levels: bool = False) -> dict[str, np.ndarray]:
        """Simulate the second-order rules.

        ``pruning`` applies the Kim-Kim-Schaumburg-Sims scheme, which keeps the
        quadratic term a function of the FIRST-order path only. Without it the
        quadratic feeds on itself and a long sample explodes -- not a numerical
        artefact to be tuned away, but a property of iterating a quadratic map,
        and the reason an unpruned second-order simulation cannot be trusted at
        any horizon that makes moments meaningful.
        """
        rng = np.random.default_rng(seed)
        Sigma = shock_covariance(self.model.shocks, cov if cov is not None else self.Sigma)
        vals, vecs = np.linalg.eigh(Sigma)
        L = vecs @ np.diag(np.sqrt(np.maximum(vals, 0.0)))
        total = periods + burn
        nk = self.P.shape[0]
        eps = rng.standard_normal((total, self.model.n_shocks)) @ L.T

        kf = np.zeros((total, nk))
        ks = np.zeros((total, nk))
        k = np.zeros((total, nk))
        u = np.zeros((total, self.F.shape[0]))
        for t in range(1, total):
            if pruning:
                kf[t] = self.P @ kf[t - 1] + self.Q @ eps[t]
                ks[t] = (self.P @ ks[t - 1]
                         + 0.5 * np.einsum("rab,a,b->r", self.h_kk, kf[t - 1], kf[t - 1])
                         + 0.5 * self.h_ss)
                k[t] = kf[t] + ks[t]
                u[t] = (self.F @ k[t]
                        + 0.5 * np.einsum("rab,a,b->r", self.g_kk, kf[t], kf[t])
                        + 0.5 * self.g_ss)
            else:
                k[t] = self.next_states(k[t - 1], eps[t])
                u[t] = self.jumps(k[t])
        k, u = k[burn:], u[burn:]
        paths = {n: k[:, i] for i, n in enumerate(self.model.states)}
        paths.update({n: u[:, i] for i, n in enumerate(self.model.jumps)})
        return self.first.to_levels(paths) if levels else paths

    def mean_correction(self, cov: Covariance = None) -> dict[str, float]:
        """Ergodic mean of each variable, in the rescaled deviation units.

        Zero to first order for every variable by construction, so what this
        returns is the second-order displacement -- the object first order
        cannot see, and the one that decides the sign of a mean effect.

        Computed in CLOSED FORM from the pruned system rather than by averaging
        a simulation. The distinction is practical: the displacement is of order
        ``sigma^2`` while the simulated series has standard deviation of order
        ``sigma``, so a sample long enough to separate them by averaging runs to
        millions of periods in a persistent model, and a shorter one reports
        Monte Carlo noise with a plausible sign.

        With the pruned recursion ``k^f_{t+1} = P k^f_t + Q eps`` and
        ``k^s_{t+1} = P k^s_t + 1/2 h_kk[k^f, k^f] + 1/2 h_ss``, taking
        expectations gives ``E k^s = (I - P)^{-1} (1/2 h_kk : V_f + 1/2 h_ss)``
        with ``V_f`` the first-order state covariance.
        """
        Sigma = shock_covariance(self.model.shocks, cov if cov is not None else self.Sigma)
        Q = self.Q if self.Q is not None else np.zeros((self.P.shape[0], self.model.n_shocks))
        Vf = solve_discrete_lyapunov(self.P, Q @ Sigma @ Q.T)
        nk = self.P.shape[0]
        drift = 0.5 * (np.einsum("rab,ab->r", self.h_kk, Vf) + self.h_ss)
        mean_k = np.linalg.solve(np.eye(nk) - self.P, drift)
        mean_u = (self.F @ mean_k + 0.5 * np.einsum("rab,ab->r", self.g_kk, Vf)
                  + 0.5 * self.g_ss)
        out = {n: float(mean_k[i]) for i, n in enumerate(self.model.states)}
        out.update({n: float(mean_u[i]) for i, n in enumerate(self.model.jumps)})
        return out

    def ergodic_mean(self, cov: Covariance = None, levels: bool = True) -> dict[str, float]:
        """Mean of every variable, in levels by default (steady state plus the correction)."""
        corr = self.mean_correction(cov)
        if not levels:
            return corr
        coords = self.first.coords
        return {n: float(coords.center[self.model.position(n)]
                         + coords.scale[self.model.position(n)] * v)
                for n, v in corr.items()}

    def accuracy(self, sizes: Sequence[float] = (0.02, 0.01, 0.005),
                 seed: int = 0) -> dict[str, Any]:
        """Deterministic accuracy check: the residual must fall as ``||k||^3``.

        The risk terms are switched off, so this reads the QUADRATIC part alone
        against the nonlinear system. Halving the displacement must divide the
        residual by about eight; a ratio near four means the quadratic block is
        wrong and the solution is no better than first order.
        """
        rng = np.random.default_rng(seed)
        nk = self.P.shape[0]
        d = rng.standard_normal(nk)
        d /= np.linalg.norm(d)
        zero_e = np.zeros(self.model.n_shocks)
        coords = self.first.coords
        out = []
        for s in sizes:
            k0 = d * s
            k1 = self.next_states(k0, risk=False)
            x0 = coords.to_level(np.concatenate([k0, self.jumps(k0, risk=False)]))
            x1 = coords.to_level(np.concatenate([k1, self.jumps(k1, risk=False)]))
            r = self.model.residuals(x1, x0, zero_e, self.first.params, self.first.ctx)
            out.append(float(np.max(np.abs(r / np.maximum(self.first.system.row_scale, 1e-300)))))
        from .solution import _order_report
        return _order_report(list(sizes), out)


def solve_second_order(first: FirstOrderSolution, cov: Covariance = None, *, h: float = 1e-4,
                       direct_limit: int = 6000, hessian: np.ndarray | None = None
                       ) -> SecondOrderSolution:
    """Compute the second-order terms around an existing first-order solution.

    Parameters
    ----------
    first : FirstOrderSolution
        Output of :func:`dsgelab.solve`.
    cov : mapping, array or None
        Shock covariance. It enters ONLY the risk corrections ``g_ss`` and
        ``h_ss``; the quadratic terms are independent of it, which is why a
        covariance sweep does not need the Sylvester equation solved again.
    h : float
        Step for the second differences.
    direct_limit : int
        Above this many unknowns (``n_vars * n_states**2``) the Sylvester
        equation is solved iteratively rather than by forming the dense matrix.
    hessian : array, optional
        Precomputed second-derivative tensor, to reuse across a covariance sweep.
    """
    model = first.model
    sys_ = first.system
    nk, nu, n = model.n_states, model.n_jumps, model.n_vars
    ne = model.n_shocks

    f1 = sys_.A                     # df/dy'
    f2 = -sys_.B                    # df/dy
    H = (second_derivatives(model, first.steady, first.params, ctx=first.ctx, h=h,
                            row_scale=sys_.row_scale) if hessian is None else hessian)

    P, F, Q = first.P, first.F, first.Q
    if Q is None:
        Q = np.zeros((nk, ne))

    # First derivatives of the composite map with respect to the states.
    M = np.vstack([P, F @ P])                       # d y'/d k
    N = np.vstack([np.eye(nk), F])                  # d y /d k
    Z = np.vstack([M, N, np.zeros((ne, nk))])       # d w /d k

    # Constant block of the quadratic condition: K[e,i,j] = (Z' H_e Z)[i,j].
    K = np.einsum("ai,eab,bj->eij", Z, H, Z)

    A1 = f1[:, :nk] + f1[:, nk:] @ F                # multiplies h_kk
    A2 = f1[:, nk:]                                 # multiplies g_kk through (P kron P)
    A3 = f2[:, nk:]                                 # multiplies g_kk directly

    Abar = np.hstack([A1, A3])                      # (n, n)
    Bbar = np.hstack([np.zeros((n, nk)), A2])       # (n, n)

    W = _solve_sylvester(Abar, Bbar, P, -K, direct_limit=direct_limit)
    W = 0.5 * (W + np.transpose(W, (0, 2, 1)))      # symmetry in (i, j) is exact in theory
    h_kk = W[:nk]
    g_kk = W[nk:]

    WPP = np.einsum("mab,ai,bj->mij", W, P, P)
    resid = float(np.max(np.abs(np.einsum("ea,aij->eij", Abar, W)
                                + np.einsum("ea,aij->eij", Bbar, WPP) + K)))

    # -- risk corrections -------------------------------------------------
    Sigma = shock_covariance(model.shocks, cov)
    QSQ = Q @ Sigma @ Q.T
    L = np.vstack([Q, F @ Q, np.zeros((n, ne)), np.eye(ne)])     # d w/d sigma, per innovation
    LSL = L @ Sigma @ L.T
    trace_hess = np.einsum("eab,ab->e", H, LSL)
    tr_term = np.einsum("rab,ab->r", g_kk, QSQ)

    lhs = np.hstack([A1, A2 + A3])                  # (n, n) acting on [h_ss; g_ss]
    rhs = -(A2 @ tr_term + trace_hess)
    ss_terms = np.linalg.solve(lhs, rhs)
    h_ss, g_ss = ss_terms[:nk], ss_terms[nk:]

    return SecondOrderSolution(first=first, g_kk=g_kk, h_kk=h_kk, g_ss=g_ss, h_ss=h_ss,
                               Sigma=Sigma, sylvester_residual=resid)


def _solve_sylvester(Abar, Bbar, P, R, direct_limit: int = 6000, order: int = 2) -> np.ndarray:
    """Solve ``Abar W + Bbar W (P kron ... kron P) = R`` for ``W`` of shape ``(n,) + (n_k,)*order``.

    The same equation appears at every perturbation order with one more Kronecker factor, so the
    solver is written once and takes the order as an argument. At second order the direct form is
    ``(n * n_k^2)^2`` entries and fits; at third it is ``(n * n_k^3)^2``, which for a
    twenty-variable model is a hundred gigabytes, and the matrix-free path is not an optimisation
    but the only option.
    """
    n = Abar.shape[0]
    nk = P.shape[0]
    q = nk ** order
    shape = (n,) + (nk,) * order
    R2 = R.reshape(n, q)

    # 'mab...,ai,bj,...->mij...': contracts each state axis with P
    letters = "abcdefgh"[:order]
    target = "ijklmnop"[:order]
    recipe = (f"m{letters}," + ",".join(f"{letters[t]}{target[t]}" for t in range(order))
              + f"->m{target}")

    def kron_P(W):
        return np.einsum(recipe, W, *([P] * order))

    def apply(Wflat):
        W = Wflat.reshape(shape)
        return (Abar @ W.reshape(n, q) + Bbar @ kron_P(W).reshape(n, q)).ravel()

    if n * q <= direct_limit:
        PP = P
        for _ in range(order - 1):
            PP = np.kron(PP, P)
        Mfull = np.kron(Abar, np.eye(q)) + np.kron(Bbar, PP.T)
        return np.linalg.solve(Mfull, R2.ravel()).reshape(shape)

    op = LinearOperator((n * q, n * q), matvec=apply, dtype=float)
    x0 = np.linalg.solve(Abar, R2).ravel()
    sol, info = lgmres(op, R2.ravel(), x0=x0, rtol=1e-12, atol=0.0, maxiter=4000)
    if info != 0:
        raise RuntimeError(
            f"the order-{order} generalised Sylvester equation did not converge "
            f"(lgmres info={info}); raise `direct_limit` to force the direct solve")
    return sol.reshape(shape)
