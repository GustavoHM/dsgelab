"""Third-order perturbation.

What it adds to the second order. The second order corrects the MEAN of the decision rules; the
third corrects their SLOPE in response to risk, and it is what makes a risk premium vary over time
instead of being a constant. Models in which the response to a shock depends on the size of the
shock (asymmetry between expansion and recession, recursive risk aversion, constraints that bind
on one side only) cannot be represented at second order by construction.

Structure
---------
The solution gains the cubic terms::

    u_t     = F k + 1/2 g_kk[k,k] + 1/2 g_ss + 1/6 g_kkk[k,k,k] + 1/2 g_ssk[k]
    k_{t+1} = P k + 1/2 h_kk[k,k] + 1/2 h_ss + 1/6 h_kkk[k,k,k] + 1/2 h_ssk[k] + Q eps

The cubic coefficients ``g_kkk`` and ``h_kkk`` solve the SAME generalised Sylvester equation as
the second order, with one more Kronecker factor::

    Abar W3 + Bbar W3 (P kron P kron P) = -K3

The terms ``g_ssk`` and ``h_ssk`` (risk entering the slope) then solve a linear system of size
``n * n_k``.

What is implemented, and what is NOT
------------------------------------
**Implemented:** the cubic block ``g_kkk`` and ``h_kkk``, verified by the order-of-accuracy test.
Feeding the cubic rules back into the nonlinear system has to leave a residual of order
``||k||^4``, that is, halving the displacement has to divide the residual by sixteen. It is the
same test that separates the second order from the first, one degree higher, and a wrong cubic
coefficient does not survive it.

**Also implemented:** ``g_ssk`` and ``h_ssk``. The innovation still enters at first order through
``Q eps``. The second-order curvature in that innovation, ``h_ee``, is solved for before taking
the expectation, and its derivative with respect to the state, contracted with the covariance, is
``h_ssk``. In the log AR(1) this has to give ``h_ssk[a, a] = ρ σ²``, with a positive sign: the
conditional mean is ``(1+â)^ρ exp(σ²/2)``, and the cross term between ``â`` and ``σ²`` has that
coefficient. An earlier derivation got the magnitude right and the sign wrong because it read the
term through the additive channel, which does not keep it. Here the channel is the curvature in
``eps``, and the sign comes out of the equation, not out of an adjustment.

What this is not: Dynare's full tensor, ``g(y_{-1}, u, σ)``, with ``g_uuu`` and ``g_σσu``. The
term the decision rule uses for the risk premium to vary with the state is ``σ² k``, and that is
the one that enters ``jumps`` and ``next_states``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import solve_discrete_lyapunov

from .linearize import second_derivatives, third_derivatives
from .model import Covariance, Model
from .secondorder import SecondOrderSolution, _solve_sylvester
from .solution import shock_covariance

__all__ = ["ThirdOrderSolution", "solve_third_order"]


@dataclass
class ThirdOrderSolution:
    """Second order, plus the cubic block and the risk slope ``σ² k``."""

    second: SecondOrderSolution
    g_kkk: np.ndarray            # (n_u, n_k, n_k, n_k)
    h_kkk: np.ndarray            # (n_k, n_k, n_k, n_k)
    g_ssk: np.ndarray            # (n_u, n_k)   risk in the slope of the jumps
    h_ssk: np.ndarray            # (n_k, n_k)   risk in the slope of the states
    sylvester_residual: float = 0.0
    risk_slope_residual: float = 0.0
    Sigma: np.ndarray = field(repr=False, default=None)

    @property
    def model(self) -> Model:
        return self.second.model

    @property
    def P(self) -> np.ndarray:
        return self.second.P

    @property
    def F(self) -> np.ndarray:
        return self.second.F

    @property
    def Q(self) -> np.ndarray:
        return self.second.Q

    # -- decision rules ---------------------------------------------------
    def jumps(self, k: ArrayLike, risk: bool = True) -> np.ndarray:
        k = np.asarray(k, dtype=float)
        out = (self.F @ k
               + 0.5 * np.einsum("rab,a,b->r", self.second.g_kk, k, k)
               + (1.0 / 6.0) * np.einsum("rabc,a,b,c->r", self.g_kkk, k, k, k))
        if risk:
            out = out + 0.5 * self.second.g_ss + 0.5 * self.g_ssk @ k
        return out

    def next_states(self, k: ArrayLike, eps: np.ndarray | None = None,
                    risk: bool = True) -> np.ndarray:
        k = np.asarray(k, dtype=float)
        out = (self.P @ k
               + 0.5 * np.einsum("rab,a,b->r", self.second.h_kk, k, k)
               + (1.0 / 6.0) * np.einsum("rabc,a,b,c->r", self.h_kkk, k, k, k))
        if risk:
            out = out + 0.5 * self.second.h_ss + 0.5 * self.h_ssk @ k
        if eps is not None and self.Q is not None:
            out = out + self.Q @ eps
        return out

    # -- simulation -------------------------------------------------------
    def simulate(self, periods: int = 1000, cov: Covariance = None, *,
                 seed: int | None = None, burn: int = 200,
                 levels: bool = False) -> dict[str, np.ndarray]:
        """PRUNED third-order simulation (Andreasen, Fernández-Villaverde and Rubio-Ramírez).

        Pruning splits the state into three layers (first, second and third order), each fed only
        by the ones before it. Without it an iterated cubic map diverges, and faster than the
        quadratic one: this is not an artefact to calibrate, it is what to do with a polynomial of
        degree three in a loop.
        """
        if cov is not None:
            Sigma_new = shock_covariance(self.model.shocks, cov)
            if not (Sigma_new.shape == self.Sigma.shape
                    and np.allclose(Sigma_new, self.Sigma, rtol=1e-12, atol=0.0)):
                # g_ss, h_ss, g_ssk and h_ssk all depend on the covariance: solve again at it
                other = solve_third_order(self.second.at_covariance(Sigma_new), Sigma_new)
                return other.simulate(periods, seed=seed, burn=burn, levels=levels)
        rng = np.random.default_rng(seed)
        Sigma = self.Sigma
        vals, vecs = np.linalg.eigh(Sigma)
        L = vecs @ np.diag(np.sqrt(np.maximum(vals, 0.0)))
        total = periods + burn
        nk = self.P.shape[0]
        eps = rng.standard_normal((total, self.model.n_shocks)) @ L.T

        kf = np.zeros((total, nk))          # first order
        ks = np.zeros((total, nk))          # second-order correction
        kt = np.zeros((total, nk))          # third-order correction
        u = np.zeros((total, self.F.shape[0]))
        gkk, hkk = self.second.g_kk, self.second.h_kk
        for t in range(1, total):
            f_prev, s_prev, t_prev = kf[t - 1], ks[t - 1], kt[t - 1]
            kf[t] = self.P @ f_prev + self.Q @ eps[t]
            ks[t] = (self.P @ s_prev + 0.5 * np.einsum("rab,a,b->r", hkk, f_prev, f_prev)
                     + 0.5 * self.second.h_ss)
            kt[t] = (self.P @ t_prev + np.einsum("rab,a,b->r", hkk, f_prev, s_prev)
                     + (1.0 / 6.0) * np.einsum("rabc,a,b,c->r", self.h_kkk, f_prev, f_prev, f_prev)
                     + 0.5 * self.h_ssk @ f_prev)
            k = kf[t] + ks[t] + kt[t]
            u[t] = (self.F @ k
                    + 0.5 * np.einsum("rab,a,b->r", gkk, kf[t], kf[t])
                    + np.einsum("rab,a,b->r", gkk, kf[t], ks[t])
                    + (1.0 / 6.0) * np.einsum("rabc,a,b,c->r", self.g_kkk, kf[t], kf[t], kf[t])
                    + 0.5 * self.second.g_ss
                    + 0.5 * self.g_ssk @ kf[t])
        k_tot = (kf + ks + kt)[burn:]
        u = u[burn:]
        paths = {n: k_tot[:, i] for i, n in enumerate(self.model.states)}
        paths.update({n: u[:, i] for i, n in enumerate(self.model.jumps)})
        return self.second.first.to_levels(paths) if levels else paths

    # -- verification -----------------------------------------------------
    def accuracy(self, sizes: Sequence[float] = (0.02, 0.01, 0.005),
                 seed: int = 0) -> dict[str, Any]:
        """Deterministic check: the residual has to fall as ``||k||^4``.

        The risk terms are switched off, so what is read is the CUBIC block alone against the
        nonlinear system. Halving the displacement has to divide the residual by sixteen; a ratio
        near eight means that the cubic block is wrong and the solution is no better than the
        second-order one.
        """
        from .solution import _order_report
        rng = np.random.default_rng(seed)
        nk = self.P.shape[0]
        d = rng.standard_normal(nk)
        d /= np.linalg.norm(d)
        zero_e = np.zeros(self.model.n_shocks)
        first = self.second.first
        coords = first.coords
        out = []
        for s in sizes:
            k0 = d * s
            k1 = self.next_states(k0, risk=False)
            x0 = coords.to_level(np.concatenate([k0, self.jumps(k0, risk=False)]))
            x1 = coords.to_level(np.concatenate([k1, self.jumps(k1, risk=False)]))
            r = self.model.residuals(x1, x0, zero_e, first.params, first.ctx)
            out.append(float(np.max(np.abs(r / np.maximum(first.system.row_scale, 1e-300)))))
        return _order_report(list(sizes), out)


def solve_third_order(second: SecondOrderSolution, cov: Covariance = None, *, h: float = 1e-3,
                      h_hess: float = 1e-4, direct_limit: int = 6000,
                      third: np.ndarray | None = None) -> ThirdOrderSolution:
    """Compute the third-order terms from a second-order solution.

    Parameters
    ----------
    second : SecondOrderSolution
        Output of :func:`dsgelab.solve_second_order`.
    cov : mapping, array or None
        Shock covariance. It enters ``g_ssk`` and ``h_ssk`` through the contraction of the
        curvature in ``eps`` with ``Σ``. The simulation uses the same matrix.
    h, h_hess : float
        Outer and inner steps of the third derivative; see
        :func:`dsgelab.linearize.third_derivatives`.
    third : array, optional
        An already computed tensor of third derivatives, to reuse across a sweep.
    """
    first = second.first
    model = first.model
    sys_ = first.system
    nk, nu, n = model.n_states, model.n_jumps, model.n_vars
    ne = model.n_shocks

    f1 = sys_.A
    f2 = -sys_.B
    H = second_derivatives(model, first.steady, first.params, ctx=first.ctx,
                           h=h_hess, row_scale=sys_.row_scale)
    T = (third_derivatives(model, first.steady, first.params, ctx=first.ctx, h=h,
                           h_hess=h_hess, row_scale=sys_.row_scale)
         if third is None else third)

    P, F = first.P, first.F
    Q = first.Q if first.Q is not None else np.zeros((nk, ne))
    g_kk, h_kk = second.g_kk, second.h_kk

    # --- derivatives of the composite map with respect to the states -------
    M = np.vstack([P, F @ P])
    N = np.vstack([np.eye(nk), F])
    Z = np.vstack([M, N, np.zeros((ne, nk))])                    # (m, nk)

    # second derivative of the composite map
    d2_yp_k = h_kk
    d2_yp_u = np.einsum("ra,aij->rij", F, h_kk) + np.einsum("rab,ai,bj->rij", g_kk, P, P)
    Z2 = np.concatenate([d2_yp_k, d2_yp_u,
                         np.zeros((nk, nk, nk)), g_kk,
                         np.zeros((ne, nk, nk))], axis=0)        # (m, nk, nk)

    # --- blocks of the Sylvester equation ----------------------------------
    A1 = f1[:, :nk] + f1[:, nk:] @ F
    A2 = f1[:, nk:]
    A3 = f2[:, nk:]
    Abar = np.hstack([A1, A3])
    Bbar = np.hstack([np.zeros((n, nk)), A2])

    # known term: g_kk crossed with (h_kk, P), summed over the three permutations
    cross = (np.einsum("rab,aij,bl->rijl", g_kk, h_kk, P)
            + np.einsum("rab,ail,bj->rijl", g_kk, h_kk, P)
            + np.einsum("rab,ajl,bi->rijl", g_kk, h_kk, P))

    K3 = (np.einsum("ea,aijl->eijl", A2, cross)
          + np.einsum("eab,ai,bjl->eijl", H, Z, Z2)
          + np.einsum("eab,aj,bil->eijl", H, Z, Z2)
          + np.einsum("eab,al,bij->eijl", H, Z, Z2)
          + np.einsum("eabc,ai,bj,cl->eijl", T, Z, Z, Z))

    W3 = _solve_sylvester(Abar, Bbar, P, -K3, direct_limit=direct_limit, order=3)
    # symmetry in the three indices is exact in theory; imposing it removes numerical noise
    W3 = sum(np.transpose(W3, (0,) + tuple(1 + np.array(perm)))
             for perm in ((0, 1, 2), (0, 2, 1), (1, 0, 2),
                          (1, 2, 0), (2, 0, 1), (2, 1, 0))) / 6.0
    h_kkk, g_kkk = W3[:nk], W3[nk:]

    WPPP = np.einsum("mabc,ai,bj,cl->mijl", W3, P, P, P)
    resid = float(np.max(np.abs(np.einsum("ea,aijl->eijl", Abar, W3)
                                + np.einsum("ea,aijl->eijl", Bbar, WPPP) + K3)))

    Sigma = shock_covariance(model.shocks, cov if cov is not None else second.Sigma)
    # g_ss and h_ss in ``second`` must be those of this covariance, or the third-order rule would
    # mix the risk terms of two economies
    second = second.at_covariance(Sigma)
    g_ssk, h_ssk, slope_resid = _risk_slope(
        nk, nu, n, ne, f1, f2, H, T, P, F, Q, g_kk, g_kkk, Sigma)
    return ThirdOrderSolution(second=second, g_kkk=g_kkk, h_kkk=h_kkk,
                              g_ssk=g_ssk, h_ssk=h_ssk,
                              sylvester_residual=resid, risk_slope_residual=slope_resid,
                              Sigma=Sigma)


def _risk_slope(nk, nu, n, ne, f1, f2, H, T, P, F, Q, g_kk, g_kkk, Sigma):
    """``g_ssk``, ``h_ssk`` from the curvature of the rule with respect to the innovation.

    The first-order rule writes ``k' = P k + Q eps``. The second derivative of that rule in
    ``eps``, ``h_ee``, solves the ex-post residual differentiated twice. The expectation
    ``h_ee : Σ`` is the mean correction. Differentiating that curvature with respect to the state
    and contracting again with ``Σ`` gives ``h_ssk``. ``g_ssk`` closes the same linear system as
    the mean correction, now in each direction of the state.
    """
    A1 = f1[:, :nk] + f1[:, nk:] @ F
    A2 = f1[:, nk:]
    A3 = f2[:, nk:]
    lhs = np.hstack([A1, A2 + A3]) if nu else A1
    Z = np.vstack([np.vstack([P, F @ P]),
                   np.vstack([np.eye(nk), F]),
                   np.zeros((ne, nk))])
    L = np.vstack([Q, F @ Q, np.zeros((n, ne)), np.eye(ne)])

    def solve_state(rhs):
        coef, *_ = np.linalg.lstsq(A1, rhs, rcond=None)
        return coef, float(np.max(np.abs(A1 @ coef - rhs)))

    worst = 0.0
    H_ee = np.einsum("epq,pa,qb->eab", H, L, L)
    g_qq = np.einsum("rab,ia,jb->rij", g_kk, Q, Q) if nu else np.zeros((0, ne, ne))
    h_ee = np.zeros((nk, ne, ne))
    for a in range(ne):
        for b in range(ne):
            rhs = -H_ee[:, a, b]
            if nu:
                rhs = rhs - A2 @ g_qq[:, a, b]
            h_ee[:, a, b], err = solve_state(rhs)
            worst = max(worst, err)
    h_ee = 0.5 * (h_ee + np.transpose(h_ee, (0, 2, 1)))

    h_ke = np.zeros((nk, nk, ne))
    for i in range(nk):
        for a in range(ne):
            Ht = np.einsum("epq,p,q->e", H, Z[:, i], L[:, a])
            rhs = -Ht
            if nu:
                rhs = rhs - A2 @ np.einsum("rab,a,b->r", g_kk, P[:, i], Q[:, a])
            h_ke[:, i, a], err = solve_state(rhs)
            worst = max(worst, err)

    def d2_ee(a, b):
        hk = h_ee[:, a, b]
        hu = F @ hk
        if nu:
            hu = hu + np.einsum("rab,a,b->r", g_kk, Q[:, a], Q[:, b])
        return np.concatenate([hk, hu, np.zeros(n + ne)])

    def d2_ke(i, a):
        hk = h_ke[:, i, a]
        hu = F @ hk
        if nu:
            hu = hu + np.einsum("rab,a,b->r", g_kk, P[:, i], Q[:, a])
        return np.concatenate([hk, hu, np.zeros(n + ne)])

    h_ssk = np.zeros((nk, nk))
    g_ssk = np.zeros((nu, nk))
    for i in range(nk):
        acc = np.zeros(n)
        for a in range(ne):
            for b in range(ne):
                weight = Sigma[a, b]
                if weight == 0.0:
                    continue
                extra = np.zeros(nu)
                if nu:
                    extra = (np.einsum("rab,a,b->r", g_kk, h_ke[:, i, a], Q[:, b])
                             + np.einsum("rab,a,b->r", g_kk, Q[:, a], h_ke[:, i, b])
                             + np.einsum("rab,a,b->r", g_kk, P[:, i], h_ee[:, a, b])
                             + np.einsum("rabc,a,b,c->r", g_kkk, P[:, i], Q[:, a], Q[:, b]))
                Hpart = (np.einsum("epq,p,q->e", H, d2_ke(i, a), L[:, b])
                         + np.einsum("epq,p,q->e", H, d2_ke(i, b), L[:, a])
                         + np.einsum("epq,p,q->e", H, d2_ee(a, b), Z[:, i]))
                Tpart = np.einsum("epqr,p,q,r->e", T, Z[:, i], L[:, a], L[:, b])
                known = Hpart + Tpart
                if nu:
                    known = known + A2 @ extra
                acc += weight * known
        solved = np.linalg.solve(lhs, -acc)
        h_ssk[:, i] = solved[:nk]
        if nu:
            g_ssk[:, i] = solved[nk:]
    return g_ssk, h_ssk, worst
