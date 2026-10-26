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
**Implemented:** the cubic blocks ``g_kkk``, ``h_kkk``, ``h_uuu``, ``h_kuu`` and ``h_kku``. The
state block is verified by the order-of-accuracy test. Feeding the cubic rules back into the
nonlinear system has to leave a residual of order ``||k||^4``, that is, halving the displacement
has to divide the residual by sixteen. The composed ``g_uuu``, ``g_kuu``, ``g_kku`` and ``g_kkk``
are checked against the exact RBC, Dynare's ``ghuuu`` and the state rows of ``ghxuu``, ``ghxxu``
and ``ghxxx``.

**Also implemented:** ``g_ssk``, ``h_ssk``, ``g_xss``, ``h_xss``, ``g_uss`` and ``h_uss``.
The risk blocks are validated against Dynare's ``ghxss`` and ``ghuss`` on ``example1`` and
``example2``. The reduced simulation rule still enters the innovation at first order through
``Q eps``. The full second-order curvature is available in ``second.h_uu``, ``second.h_ku`` and
``second.decision_rule()``. The risk-slope calculation reuses those ex-post blocks and
differentiates them with respect to the state. In the log AR(1) this has to give
``h_ssk[a, a] = ρ σ²``, with a positive sign: the conditional mean is
``(1+â)^ρ exp(σ²/2)``, and the cross term between ``â`` and ``σ²`` has that coefficient. An
earlier derivation got the magnitude right and the sign wrong because it read the term through the
additive channel, which does not keep it. Here the channel is the curvature in ``eps``, and the
sign comes out of the equation, not out of an adjustment.

The reduced decision rule uses ``σ² k`` for the risk premium to vary with the state. Simulation
and IRFs use the full pruned rule with the ``ghxss`` and ``ghuss`` blocks and all cubic innovation
terms. ``stochastic_steady_state`` computes the fixed point of the full unpruned rule, which is
not the ergodic mean of the pruned system.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from itertools import combinations_with_replacement
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import solve_discrete_lyapunov
from scipy.optimize import root

from .linearize import second_derivatives, third_derivatives
from .model import Covariance, Model
from .steady import SteadyState
from .secondorder import (DecisionRule, SecondOrderSolution, _apply_cubic_canonical,
                          _cubic_canonical_maps, _dynare_decision_rule, _solve_sylvester)
from .solution import shock_covariance
from .tensors import SparseSecondDerivatives, SparseThirdDerivatives

__all__ = ["ThirdOrderSolution", "solve_third_order", "stochastic_steady_state"]


@dataclass
class ThirdOrderSolution:
    """Second order, plus cubic blocks and the risk slope ``σ² k``."""

    second: SecondOrderSolution
    g_kkk: np.ndarray | SparseThirdDerivatives  # (n_u, n_k, n_k, n_k)
    h_kkk: np.ndarray | SparseThirdDerivatives  # (n_k, n_k, n_k, n_k)
    h_uuu: np.ndarray            # (n_k, n_eps, n_eps, n_eps)
    h_kuu: np.ndarray            # (n_k, n_k, n_eps, n_eps)
    h_kku: np.ndarray            # (n_k, n_k, n_k, n_eps)
    g_ssk: np.ndarray            # (n_u, n_k)   risk in the slope of the jumps
    h_ssk: np.ndarray            # (n_k, n_k)   risk in the slope of the states
    g_xss: np.ndarray            # (n_u, n_k)   Dynare's risk-state block of the jumps
    h_xss: np.ndarray            # (n_k, n_k)   Dynare's risk-state block of the states
    g_uss: np.ndarray            # (n_u, n_eps) Dynare's risk-innovation block of the jumps
    h_uss: np.ndarray            # (n_k, n_eps) Dynare's risk-innovation block of the states
    Sigma: np.ndarray = field(repr=False)
    sylvester_residual: float = 0.0
    risk_slope_residual: float = 0.0
    risk_state_residual: float = 0.0
    risk_shock_residual: float = 0.0
    innovation_residual: float = 0.0

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
    def decision_rule(self) -> DecisionRule:
        """Compose the state and jump rules, including cubic innovation curvature."""
        return self._decision_rule(state=True)

    def _decision_rule(self, *, state: bool) -> DecisionRule:
        nk, ne = self.P.shape[0], self.model.n_shocks
        Q = self.Q if self.Q is not None else np.zeros((nk, ne))
        P, F, g_kk = self.P, self.F, self.second.g_kk
        g_kkk = self.g_kkk
        jump = np.einsum("ra,aijk->rijk", F, self.h_uuu)
        jump = jump + _contract_third(g_kkk, Q, Q, Q)
        for a in range(ne):
            for b in range(ne):
                for c in range(ne):
                    jump[:, a, b, c] += (
                        np.einsum("rab,a,b->r", g_kk, self.second.h_uu[:, a, b], Q[:, c])
                        + np.einsum("rab,a,b->r", g_kk, self.second.h_uu[:, a, c], Q[:, b])
                        + np.einsum("rab,a,b->r", g_kk, self.second.h_uu[:, b, c], Q[:, a]))
        jump_kuu = np.einsum("ra,aijk->rijk", F, self.h_kuu)
        jump_kku = np.einsum("ra,aijk->rijk", F, self.h_kku)
        for i in range(nk):
            for a in range(ne):
                for b in range(ne):
                    jump_kuu[:, i, a, b] += (
                        np.einsum("rab,a,b->r", g_kk, self.second.h_uu[:, a, b], P[:, i])
                        + np.einsum("rab,a,b->r", g_kk, self.second.h_ku[:, i, a], Q[:, b])
                        + np.einsum("rab,a,b->r", g_kk, self.second.h_ku[:, i, b], Q[:, a])
                        + _contract_third(g_kkk, P[:, i, None], Q[:, a, None],
                                          Q[:, b, None])[:, 0, 0, 0])
                for j in range(nk):
                    jump_kku[:, i, j, a] += (
                        np.einsum("rab,a,b->r", g_kk, self.second.h_kk[:, i, j], Q[:, a])
                        + np.einsum("rab,a,b->r", g_kk, self.second.h_ku[:, i, a], P[:, j])
                        + np.einsum("rab,a,b->r", g_kk, self.second.h_ku[:, j, a], P[:, i])
                        + _contract_third(g_kkk, P[:, i, None], P[:, j, None],
                                          Q[:, a, None])[:, 0, 0, 0])
        cubic_state = None
        if state:
            h_dense = (self.h_kkk.to_dense() if isinstance(self.h_kkk, SparseThirdDerivatives)
                       else self.h_kkk)
            jump_kkk = np.einsum("ra,aijk->rijk", F, h_dense)
            jump_kkk = jump_kkk + _contract_third(g_kkk, P, P, P)
            for i in range(nk):
                for j in range(nk):
                    for k in range(nk):
                        jump_kkk[:, i, j, k] += (
                            np.einsum("rab,a,b->r", g_kk, self.second.h_kk[:, i, j], P[:, k])
                            + np.einsum("rab,a,b->r", g_kk, self.second.h_kk[:, i, k], P[:, j])
                            + np.einsum("rab,a,b->r", g_kk, self.second.h_kk[:, j, k], P[:, i]))
            cubic_state = np.concatenate([h_dense, jump_kkk], axis=0)
        return replace(self.second.decision_rule(),
                       g_uuu=np.concatenate([self.h_uuu, jump], axis=0),
                       g_kuu=np.concatenate([self.h_kuu, jump_kuu], axis=0),
                       g_kku=np.concatenate([self.h_kku, jump_kku], axis=0),
                       g_kkk=cubic_state)

    def jumps(self, k: ArrayLike, eps: np.ndarray | None = None,
              risk: bool = True) -> np.ndarray:
        k = np.asarray(k, dtype=float)
        out = (self.F @ k
               + 0.5 * np.einsum("rab,a,b->r", self.second.g_kk, k, k)
               + (1.0 / 6.0) * _contract_third(self.g_kkk, k[:, None], k[:, None], k[:, None])[:, 0, 0, 0])
        if risk:
            out = out + 0.5 * self.second.g_ss + 0.5 * self.g_xss @ k
        if eps is not None:
            rule = self.decision_rule()
            assert rule.g_uuu is not None and rule.g_kuu is not None and rule.g_kku is not None
            jump = self.model.n_states
            out = out + np.einsum("rab,a,b->r", rule.g_ku[jump:], k, eps)
            out = out + 0.5 * np.einsum("rab,a,b->r", rule.g_uu[jump:], eps, eps)
            out = out + (1.0 / 6.0) * np.einsum("rabc,a,b,c->r", rule.g_uuu[jump:], eps, eps, eps)
            out = out + 0.5 * np.einsum("rabc,a,b,c->r", rule.g_kuu[jump:], k, eps, eps)
            out = out + 0.5 * np.einsum("rabc,a,b,c->r", rule.g_kku[jump:], k, k, eps)
            if risk:
                out = out + 0.5 * self.g_uss @ eps
        return out

    def next_states(self, k: ArrayLike, eps: np.ndarray | None = None,
                    risk: bool = True) -> np.ndarray:
        k = np.asarray(k, dtype=float)
        out = (self.P @ k
               + 0.5 * np.einsum("rab,a,b->r", self.second.h_kk, k, k)
               + (1.0 / 6.0) * _contract_third(self.h_kkk, k[:, None], k[:, None], k[:, None])[:, 0, 0, 0])
        if risk:
            out = out + 0.5 * self.second.h_s2 + 0.5 * self.h_xss @ k
        if eps is not None:
            out = out + self.Q @ eps
            out = out + np.einsum("rab,a,b->r", self.second.h_ku, k, eps)
            out = out + 0.5 * np.einsum("rab,a,b->r", self.second.h_uu, eps, eps)
            out = out + (1.0 / 6.0) * np.einsum("rabc,a,b,c->r", self.h_uuu, eps, eps, eps)
            out = out + 0.5 * np.einsum("rabc,a,b,c->r", self.h_kuu, k, eps, eps)
            out = out + 0.5 * np.einsum("rabc,a,b,c->r", self.h_kku, k, k, eps)
            if risk:
                out = out + 0.5 * self.h_uss @ eps
        return out

    def stochastic_steady_state(self, *, guess=None, tol: float = 1e-10,
                                method: str = "hybr") -> SteadyState:
        """Fixed point of the full third-order decision rule with ``sigma=1``."""
        first = self.second.first
        coords = first.coords
        nk = self.model.n_states
        if guess is None:
            k0 = np.zeros(nk)
        elif isinstance(guess, Mapping):
            k0 = coords.to_hat(self.model.vector(guess))[:nk]
        else:
            k0 = np.asarray(guess, dtype=float)
        if k0.size != nk:
            raise ValueError(f"guess has {k0.size} entries for {nk} states")

        def residual(k):
            return self.next_states(k, risk=True) - k

        solved = root(residual, k0, method=method, tol=1e-12)
        k = solved.x
        error = float(np.max(np.abs(residual(k))))
        if error >= tol:
            raise RuntimeError(f"stochastic_steady_state: residual {error:.3e} above tolerance {tol:.1e}. "
                               f"Solver message: {getattr(solved, 'message', '')}")
        x = coords.to_level(np.r_[k, self.jumps(k)])
        return SteadyState(x=x, values=self.model.dict(x), residual=error, ctx=first.ctx,
                           outer_iterations=int(getattr(solved, "nfev", 0)),
                           message=getattr(solved, "message", ""), model=self.model)

    def _pruned_rule(self, dynare=False):
        nk = self.model.n_states
        second = self.second
        if dynare:
            lower = _dynare_decision_rule(second)
            cubic = _dynare_cubic_rule(self)
            risk = _dynare_risk_rule(self)
        else:
            lower = second.decision_rule()
            cubic = self.decision_rule()
            risk = {"g_xss": np.r_[self.h_xss, self.g_xss],
                    "g_uss": np.r_[self.h_uss, self.g_uss]}
        return {
            "P": lower.g_k[:nk], "Q": lower.g_u[:nk], "F": lower.g_k[nk:],
            "g_u": lower.g_u[nk:],
            "h_s2": lower.g_s2[:nk], "g_ss": lower.g_s2[nk:],
            "h_kk": lower.g_kk[:nk], "g_kk": lower.g_kk[nk:],
            "h_ku": lower.g_ku[:nk], "g_ku": lower.g_ku[nk:],
            "h_uu": lower.g_uu[:nk], "g_uu": lower.g_uu[nk:],
            "h_kkk": cubic["g_kkk"][:nk], "g_kkk": cubic["g_kkk"][nk:],
            "h_uuu": cubic["g_uuu"][:nk], "g_uuu": cubic["g_uuu"][nk:],
            "h_kuu": cubic["g_kuu"][:nk], "g_kuu": cubic["g_kuu"][nk:],
            "h_kku": cubic["g_kku"][:nk], "g_kku": cubic["g_kku"][nk:],
            "h_xss": risk["g_xss"][:nk], "g_xss": risk["g_xss"][nk:],
            "h_uss": risk["g_uss"][:nk], "g_uss": risk["g_uss"][nk:],
        }

    def _pruned_transition(self, f, s, t, eps, rule):
        kf = rule["P"] @ f + rule["Q"] @ eps
        ks = (rule["P"] @ s + 0.5 * np.einsum("rab,a,b->r", rule["h_kk"], f, f)
              + np.einsum("rab,a,b->r", rule["h_ku"], f, eps)
              + 0.5 * np.einsum("rab,a,b->r", rule["h_uu"], eps, eps)
              + 0.5 * rule["h_s2"])
        kt = (rule["P"] @ t + np.einsum("rab,a,b->r", rule["h_kk"], f, s)
              + (1.0 / 6.0) * np.einsum("rabc,a,b,c->r", rule["h_kkk"], f, f, f)
              + 0.5 * rule["h_xss"] @ f
              + np.einsum("rab,a,b->r", rule["h_ku"], s, eps)
              + (1.0 / 6.0) * np.einsum("rabc,a,b,c->r", rule["h_uuu"], eps, eps, eps)
              + 0.5 * np.einsum("rabc,a,b,c->r", rule["h_kuu"], f, eps, eps)
              + 0.5 * np.einsum("rabc,a,b,c->r", rule["h_kku"], f, f, eps)
              + 0.5 * rule["h_uss"] @ eps)
        return kf, ks, kt

    def _pruned_jumps(self, f, s, t, eps, rule):
        k = f + s + t
        return (rule["F"] @ k + rule["g_u"] @ eps
                + 0.5 * np.einsum("rab,a,b->r", rule["g_kk"], f, f)
                + np.einsum("rab,a,b->r", rule["g_kk"], f, s)
                + (1.0 / 6.0) * np.einsum("rabc,a,b,c->r", rule["g_kkk"], f, f, f)
                + 0.5 * rule["g_ss"] + 0.5 * rule["g_xss"] @ f
                + np.einsum("rab,a,b->r", rule["g_ku"], f + s, eps)
                + 0.5 * np.einsum("rab,a,b->r", rule["g_uu"], eps, eps)
                + (1.0 / 6.0) * np.einsum("rabc,a,b,c->r", rule["g_uuu"], eps, eps, eps)
                + 0.5 * np.einsum("rabc,a,b,c->r", rule["g_kuu"], f, eps, eps)
                + 0.5 * np.einsum("rabc,a,b,c->r", rule["g_kku"], f, f, eps)
                + 0.5 * rule["g_uss"] @ eps)

    def irf(self, name: str, size: float = 0.01, periods: int = 40, *,
            n_draws: int = 0, seed: int = 0, state: ArrayLike | None = None,
            percent: bool = False, drop: int = 100) -> dict[str, np.ndarray]:
        """Impulse response of the full pruned third-order rule.

        With ``n_draws=0`` or a supplied state, this is the conditional response
        with no further innovations. Otherwise, it averages generalized responses
        over ``n_draws`` stochastic paths, with the impulse added after ``drop``.
        """
        if state is None and n_draws > 0:
            out = self._generalized_irf(name, size, periods, n_draws, seed, drop)
            return {n: v * 100.0 for n, v in out.items()} if percent else out
        nk = self.P.shape[0]
        starts = [np.asarray(state, dtype=float)] if state is not None else [np.zeros(nk)]
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
        assert acc is not None
        out = {n: v / len(starts) for n, v in acc.items()}
        if percent:
            out = {n: v * 100.0 for n, v in out.items()}
        return out

    def _generalized_irf(self, shock, size, periods, n_draws, seed, drop):
        rng = np.random.default_rng(seed)
        Sigma = self.Sigma
        vals, vecs = np.linalg.eigh(Sigma)
        L = vecs @ np.diag(np.sqrt(np.maximum(vals, 0.0)))
        position = self.model.shock_position(shock)
        factor = np.linalg.cholesky(Sigma)
        impulse = factor[:, position] * (size / factor[position, position])
        rule = self._pruned_rule(dynare=True)
        acc = None
        for _ in range(n_draws):
            eps = rng.standard_normal((periods + drop, self.model.n_shocks)) @ L.T
            shocked_eps = eps.copy()
            shocked_eps[drop] += impulse
            base = self._path_from_shocks(eps, rule)
            shocked = self._path_from_shocks(shocked_eps, rule)
            diff = {n: shocked[n][drop:] - base[n][drop:] for n in shocked}
            if acc is None:
                acc = diff
            else:
                for n in acc:
                    acc[n] = acc[n] + diff[n]
        assert acc is not None
        return {n: v / n_draws for n, v in acc.items()}

    def _path_from_shocks(self, eps, rule):
        total = eps.shape[0]
        nk, nu = self.model.n_states, self.model.n_jumps
        kf = np.zeros((total, nk))
        ks = np.zeros((total, nk))
        kt = np.zeros((total, nk))
        u = np.zeros((total, nu))
        for t in range(total):
            f_prev = kf[t - 1] if t else np.zeros(nk)
            s_prev = ks[t - 1] if t else np.zeros(nk)
            t_prev = kt[t - 1] if t else np.zeros(nk)
            kf[t], ks[t], kt[t] = self._pruned_transition(f_prev, s_prev, t_prev, eps[t], rule)
            u[t] = self._pruned_jumps(f_prev, s_prev, t_prev, eps[t], rule)
        paths = {n: (kf + ks + kt)[:, i] for i, n in enumerate(self.model.states)}
        paths.update({n: u[:, i] for i, n in enumerate(self.model.jumps)})
        return paths

    def _conditional_path(self, k0, shock, size, periods):
        nk = self.P.shape[0]
        kf = np.zeros((periods, nk))
        ks = np.zeros((periods, nk))
        kt = np.zeros((periods, nk))
        u = np.zeros((periods, self.F.shape[0]))
        initial = np.asarray(k0, dtype=float)
        eps = np.zeros(self.model.n_shocks)
        if size != 0.0:
            position = self.model.shock_position(shock)
            factor = np.linalg.cholesky(self.Sigma)
            eps = factor[:, position] * (size / factor[position, position])
        rule = self._pruned_rule(dynare=True)
        zero = np.zeros(nk)
        kf[0], ks[0], kt[0] = self._pruned_transition(initial, zero, zero, eps, rule)
        u[0] = self._pruned_jumps(initial, zero, zero, eps, rule)
        zero_eps = np.zeros(self.model.n_shocks)
        for t in range(1, periods):
            kf[t], ks[t], kt[t] = self._pruned_transition(kf[t - 1], ks[t - 1], kt[t - 1],
                                                          zero_eps, rule)
            u[t] = self._pruned_jumps(kf[t - 1], ks[t - 1], kt[t - 1], zero_eps, rule)
        paths = {n: (kf + ks + kt)[:, i] for i, n in enumerate(self.model.states)}
        paths.update({n: u[:, i] for i, n in enumerate(self.model.jumps)})
        return paths

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
        rule = self._pruned_rule(dynare=True)
        u[0] = self._pruned_jumps(np.zeros(nk), np.zeros(nk), np.zeros(nk), eps[0], rule)
        for t in range(1, total):
            kf[t], ks[t], kt[t] = self._pruned_transition(kf[t - 1], ks[t - 1], kt[t - 1], eps[t], rule)
            u[t] = self._pruned_jumps(kf[t - 1], ks[t - 1], kt[t - 1], eps[t], rule)
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


def stochastic_steady_state(third: ThirdOrderSolution, **kwargs) -> SteadyState:
    return third.stochastic_steady_state(**kwargs)


def _dynare_cubic_rule(third: ThirdOrderSolution) -> dict[str, np.ndarray]:
    """Cubic blocks in Dynare's endogenous-state timing."""
    second = third.second
    first = second.first
    P, Q, F = first.P, first.Q, first.F
    nk, ne = first.model.n_states, first.model.n_shocks
    rule = third.decision_rule()
    assert rule.g_uuu is not None and rule.g_kuu is not None
    assert rule.g_kku is not None and rule.g_kkk is not None
    g_kkk = third.g_kkk.to_dense() if isinstance(third.g_kkk, SparseThirdDerivatives) else third.g_kkk
    h_kkk = third.h_kkk.to_dense() if isinstance(third.h_kkk, SparseThirdDerivatives) else third.h_kkk
    direct = np.array([not np.allclose(Q[i], 0.0, atol=1e-12) for i in range(nk)])
    R = np.eye(nk)
    S = np.zeros_like(Q)
    R[direct], S[direct] = P[direct], Q[direct]
    out = {key: getattr(rule, key).copy() for key in ("g_uuu", "g_kuu", "g_kku", "g_kkk")}
    for row in range(nk):
        if direct[row]:
            continue
        D = np.zeros(nk)
        D[direct] = P[row, direct]
        for a in range(ne):
            for b in range(ne):
                for c in range(ne):
                    out["g_uuu"][row, a, b, c] = (
                        np.einsum("d,d->", D, rule.g_uuu[:nk, a, b, c])
                        + np.einsum("ab,a,b->", second.h_kk[row], S[:, a], rule.g_uu[:nk, b, c])
                        + np.einsum("ab,a,b->", second.h_kk[row], S[:, b], rule.g_uu[:nk, a, c])
                        + np.einsum("ab,a,b->", second.h_kk[row], S[:, c], rule.g_uu[:nk, a, b])
                        + np.einsum("abc,a,b,c->", h_kkk[row], S[:, a], S[:, b], S[:, c]))
        for i in range(nk):
            for a in range(ne):
                for b in range(ne):
                    out["g_kuu"][row, i, a, b] = (
                        np.einsum("d,d->", D, rule.g_kuu[:nk, i, a, b])
                        + np.einsum("ab,a,b->", second.h_kk[row], R[:, i], rule.g_uu[:nk, a, b])
                        + np.einsum("ab,a,b->", second.h_kk[row], S[:, a], rule.g_ku[:nk, i, b])
                        + np.einsum("ab,a,b->", second.h_kk[row], S[:, b], rule.g_ku[:nk, i, a])
                        + np.einsum("abc,a,b,c->", h_kkk[row], R[:, i], S[:, a], S[:, b]))
            for j in range(nk):
                for a in range(ne):
                    inner_ku_i = D * rule.g_ku[:nk, i, a]
                    inner_ku_j = D * rule.g_ku[:nk, j, a]
                    inner_kk = D * rule.g_kk[:nk, i, j]
                    out["g_kku"][row, i, j, a] = (
                        np.einsum("d,d->", D, rule.g_kku[:nk, i, j, a])
                        + np.einsum("ab,a,b->", second.h_kk[row], R[:, i], inner_ku_j)
                        + np.einsum("ab,a,b->", second.h_kk[row], R[:, j], inner_ku_i)
                        + np.einsum("ab,a,b->", second.h_kk[row], S[:, a], inner_kk)
                        + np.einsum("abc,a,b,c->", h_kkk[row], R[:, i], R[:, j], S[:, a]))
            for j in range(nk):
                for k in range(nk):
                    inner_jk = D * rule.g_kk[:nk, j, k]
                    inner_ik = D * rule.g_kk[:nk, i, k]
                    inner_ij = D * rule.g_kk[:nk, i, j]
                    out["g_kkk"][row, i, j, k] = (
                        np.einsum("d,d->", D, rule.g_kkk[:nk, i, j, k])
                        + np.einsum("ab,a,b->", second.h_kk[row], R[:, i], inner_jk)
                        + np.einsum("ab,a,b->", second.h_kk[row], R[:, j], inner_ik)
                        + np.einsum("ab,a,b->", second.h_kk[row], R[:, k], inner_ij)
                        + np.einsum("abc,a,b,c->", h_kkk[row], R[:, i], R[:, j], R[:, k]))
    jump_kuu = (np.einsum("ra,aijk->rijk", F, third.h_kuu)
                + np.einsum("rabc,ai,bj,ck->rijk", g_kkk, R, S, S))
    for i in range(nk):
        for a in range(ne):
            for b in range(ne):
                jump_kuu[:, i, a, b] += (
                    np.einsum("rab,a,b->r", second.g_kk, R[:, i], second.h_uu[:, a, b])
                    + np.einsum("rab,a,b->r", second.g_kk, second.h_ku[:, i, a], S[:, b])
                    + np.einsum("rab,a,b->r", second.g_kk, second.h_ku[:, i, b], S[:, a]))
    out["g_kuu"][nk:] = jump_kuu
    out["g_kku"][nk:] = np.einsum("rabc,ai,bj,ck->rijk", g_kkk, R, R, S)
    out["g_kkk"][nk:] = np.einsum("rabc,ai,bj,ck->rijk", g_kkk, R, R, R)
    return out


def _sparse_sylvester_residual(W3, Abar, Bbar, P, K3, canonical=False):
    n, nk = Abar.shape[0], P.shape[0]
    if canonical:
        difference = (_apply_cubic_canonical(Abar, Bbar, P, W3.to_canonical(),
                                             _cubic_canonical_maps(nk)) + K3)
        return float(np.max(np.abs(difference)))
    identity = np.eye(nk)
    block = max(1, 1_000_000 // max(nk ** 3, 1))
    residual = 0.0
    for start in range(0, n, block):
        rows = slice(start, min(start + block, n))
        if canonical:
            first = W3.contract_canonical(identity, identity, identity, weights=Abar[rows])
            second = W3.contract_canonical(P, P, P, weights=Bbar[rows])
        else:
            first = W3.contract(identity, identity, identity, weights=Abar[rows])
            second = W3.contract(P, P, P, weights=Bbar[rows])
        difference = first + second + K3[rows]
        residual = max(residual, float(np.max(np.abs(difference))))
    return residual


def _contract_third(T, u, v, w):
    if isinstance(T, SparseThirdDerivatives):
        return T.contract(u, v, w)
    return np.einsum("eabc,ai,bj,ck->eijk", T, u, v, w)


def _contract_second(H, u, v):
    if isinstance(H, SparseSecondDerivatives):
        return H.contract(u, v)
    return np.einsum("eab,a...,b...->e...", H, u, v)


def _contract_second_states(H, Z, Z2):
    if isinstance(H, SparseSecondDerivatives):
        return H.contract(Z, Z2)
    return np.einsum("eab,ai,bjl->eijl", H, Z, Z2)


def _canonical_cubic_known(A2, g_kk, h_kk, P, H, T, Z, Z2):
    nk = P.shape[0]
    n = A2.shape[0]
    canonical = list(combinations_with_replacement(range(nk), 3))
    out = np.zeros((n, len(canonical)))
    for column, (i, j, k) in enumerate(canonical):
        cross = (np.einsum("rab,a,b->r", g_kk, h_kk[:, i, j], P[:, k])
                 + np.einsum("rab,a,b->r", g_kk, h_kk[:, i, k], P[:, j])
                 + np.einsum("rab,a,b->r", g_kk, h_kk[:, j, k], P[:, i]))
        first = _contract_second(H, Z[:, i, None], Z2[:, j, k, None]).reshape(n)
        second = _contract_second(H, Z[:, j, None], Z2[:, i, k, None]).reshape(n)
        third = _contract_second(H, Z[:, k, None], Z2[:, i, j, None]).reshape(n)
        cubic = _contract_third(T, Z[:, i, None], Z[:, j, None], Z[:, k, None]).reshape(n)
        out[:, column] = A2 @ cross + first + second + third + cubic
    return out


def _risk_state(third: ThirdOrderSolution, H, T):
    second = third.second
    first = second.first
    model = first.model
    sys_ = first.system
    f1 = sys_.A
    nk, nu, n, ne = model.n_states, model.n_jumps, model.n_vars, model.n_shocks
    P, F = first.P, first.F
    A1 = f1[:, :nk] + f1[:, nk:] @ F
    A2 = f1[:, nk:]
    A3 = -sys_.B[:, nk:]
    Abar = np.hstack([A1, A3])
    Bbar = np.hstack([np.zeros((n, nk)), A2])
    Z = np.vstack([P, F @ P, np.eye(nk), F, np.zeros((ne, nk))])
    rule = third._decision_rule(state=False)
    assert rule.g_kuu is not None
    zup = np.vstack([rule.g_u, np.zeros((n + ne, ne))])
    zuup = np.vstack([rule.g_ku, np.zeros((n + ne, nk, ne))])
    guu = np.vstack([rule.g_uu, np.zeros((n + ne, ne, ne))])
    zss = np.r_[second.h_s2, second.g_ss + F @ second.h_s2,
                np.zeros(nk), second.g_ss, np.zeros(ne)]
    K = np.zeros((n, nk))
    for i in range(nk):
        K[:, i] = (_contract_second(H, Z[:, i, None], zss[:, None]).reshape(n)
                   + A2 @ np.einsum("rab,a,b->r", second.g_kk, P[:, i], second.h_s2))
        for a in range(ne):
            for b in range(ne):
                known = _contract_third(T, Z[:, i, None], zup[:, a, None],
                                        zup[:, b, None])[:, 0, 0, 0]
                known += _contract_second(H, zuup[:, i, a, None],
                                          zup[:, b, None]).reshape(n)
                known += _contract_second(H, zuup[:, i, b, None],
                                          zup[:, a, None]).reshape(n)
                known += _contract_second(H, Z[:, i, None], guu[:, a, b, None]).reshape(n)
                known += f1 @ rule.g_kuu[:, i, a, b]
                K[:, i] += third.Sigma[a, b] * known
    W = _solve_sylvester(Abar, Bbar, P, -K, order=1)
    residual = float(np.max(np.abs(Abar @ W + Bbar @ (W @ P) + K)))
    return W[nk:], W[:nk], residual


def _contract_second_rows(H, u, v):
    if isinstance(H, SparseSecondDerivatives):
        out = np.zeros(H.shape[0])
        for e, (a, b), value in zip(H.equations, H.indices, H.values):
            out[e] += value * u[e, a] * v[e, b]
        return out
    return np.einsum("epq,ep,eq->e", H, u, v)


def _contract_third_rows(T, u, v, w):
    if isinstance(T, SparseThirdDerivatives):
        out = np.zeros(T.shape[0])
        for e, (a, b, c), value in zip(T.equations, T.indices, T.values):
            out[e] += value * u[e, a] * v[b] * w[c]
        return out
    return np.einsum("epqr,ep,q,r->e", T, u, v, w)


def _risk_innovation(third: ThirdOrderSolution, H, T):
    second = third.second
    first = second.first
    model = first.model
    sys_ = first.system
    f1 = sys_.A
    nk, nu, n, ne = model.n_states, model.n_jumps, model.n_vars, model.n_shocks
    P, F, Q = first.P, first.F, first.Q
    A1 = f1[:, :nk] + f1[:, nk:] @ F
    A2 = f1[:, nk:]
    A3 = -sys_.B[:, nk:]
    Abar = np.hstack([A1, A3])
    direct = np.array([not np.allclose(Q[i], 0.0, atol=1e-12) for i in range(nk)])
    forward = np.any(f1[:, nk:] != 0.0, axis=1)
    R = np.eye(nk)
    R[direct] = P[direct]
    U = Q.copy()
    U[~direct] = P[~direct] @ Q
    Ls = np.repeat(np.vstack([U, F @ R @ U, direct[:, None] * U,
                              F @ Q, np.eye(ne)])[None], n, axis=0)
    Ls[np.ix_(np.flatnonzero(forward), np.flatnonzero(direct))] = (P @ U)[direct]
    rule = third._decision_rule(state=False)
    assert rule.g_kuu is not None
    zup = np.vstack([rule.g_u, np.zeros((n + ne, ne))])
    zuup = np.vstack([np.einsum("ria,ib->rab", rule.g_ku, Q),
                      np.zeros((n + ne, ne, ne))])
    guu = np.vstack([rule.g_uu, np.zeros((n + ne, ne, ne))])
    gkuu = np.einsum("riab,ic->rabc", rule.g_kuu, Q)
    h_s2 = second.h_s2
    zss0 = np.r_[h_s2, second.g_ss, direct * h_s2, second.g_ss, np.zeros(ne)]
    zss_forward = np.r_[h_s2 + direct * (P @ h_s2), second.g_ss + F @ h_s2,
                        direct * h_s2, second.g_ss, np.zeros(ne)]
    zss = np.repeat(zss0[None, :], n, axis=0)
    zss[forward] = zss_forward
    g_xss = third.g_xss @ R
    g_kk = _dynare_decision_rule(second).g_kk[nk:]
    K = np.zeros((n, ne))
    for c in range(ne):
        K[:, c] = (_contract_second_rows(H, Ls[:, :, c], zss)
                   + A2 @ (g_xss @ U[:, c]
                           + np.einsum("rab,a,b->r", g_kk, U[:, c], h_s2)))
        for a in range(ne):
            for b in range(ne):
                known = _contract_third_rows(T, Ls[:, :, c], zup[:, a], zup[:, b])
                known += _contract_second(H, zuup[:, a, c, None],
                                          zup[:, b, None]).reshape(n)
                known += _contract_second(H, zuup[:, b, c, None],
                                          zup[:, a, None]).reshape(n)
                known += _contract_second_rows(
                    H, Ls[:, :, c], np.broadcast_to(guu[:, a, b], (n, 2 * n + ne)))
                known += f1 @ gkuu[:, a, b, c]
                K[:, c] += third.Sigma[a, b] * known
    W = np.linalg.solve(Abar, -K)
    residual = float(np.max(np.abs(Abar @ W + K)))
    return W[nk:], W[:nk], residual


def _dynare_risk_rule(third: ThirdOrderSolution) -> dict[str, np.ndarray]:
    first = third.second.first
    P, Q = first.P, first.Q
    nk = first.model.n_states
    direct = np.array([not np.allclose(Q[i], 0.0, atol=1e-12) for i in range(nk)])
    R = np.eye(nk)
    R[direct] = P[direct]
    return {"g_xss": np.concatenate([third.h_xss, third.g_xss]) @ R,
            "g_uss": np.concatenate([third.h_uss, third.g_uss])}


def solve_third_order(second: SecondOrderSolution, cov: Covariance = None, *, h: float = 1e-3,
                      h_hess: float = 1e-4, direct_limit: int = 6000,
                      third: np.ndarray | SparseThirdDerivatives | None = None,
                      hessian: np.ndarray | SparseSecondDerivatives | None = None,
                      sparse: bool = False) -> ThirdOrderSolution:
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
    third : array or SparseThirdDerivatives, optional
        An already computed tensor of third derivatives, to reuse across a sweep.
    hessian : array or SparseSecondDerivatives, optional
        An already computed tensor of second derivatives, to reuse with ``third``.
    sparse : bool
        Keep the model derivatives and cubic state coefficients in coordinate
        form. Rule evaluation and risk terms contract them directly; composed
        decision rules and the residual check use dense arrays on demand.
    direct_limit : int
        Above this many unknowns, use matrix-free LGMRES on the independent
        symmetric state triples instead of forming the dense Sylvester operator.
    """
    first = second.first
    model = first.model
    sys_ = first.system
    nk, nu, n = model.n_states, model.n_jumps, model.n_vars
    ne = model.n_shocks

    f1 = sys_.A
    f2 = -sys_.B
    H = (second_derivatives(model, first.steady, first.params, ctx=first.ctx,
                            h=h_hess, row_scale=sys_.row_scale, sparse=sparse)
         if hessian is None else hessian)
    T = (third_derivatives(model, first.steady, first.params, ctx=first.ctx, h=h,
                           h_hess=h_hess, row_scale=sys_.row_scale, sparse=sparse)
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

    canonical_rhs = sparse and n * nk ** 3 > direct_limit
    if canonical_rhs:
        K3 = _canonical_cubic_known(A2, g_kk, h_kk, P, H, T, Z, Z2)
    else:
        # known term: g_kk crossed with (h_kk, P), summed over the three permutations
        cross = (np.einsum("rab,aij,bl->rijl", g_kk, h_kk, P)
                + np.einsum("rab,ail,bj->rijl", g_kk, h_kk, P)
                + np.einsum("rab,ajl,bi->rijl", g_kk, h_kk, P))
        H_states = _contract_second_states(H, Z, Z2)
        K3 = (np.einsum("ea,aijl->eijl", A2, cross)
              + H_states + np.transpose(H_states, (0, 2, 1, 3))
              + np.transpose(H_states, (0, 2, 3, 1))
              + _contract_third(T, Z, Z, Z))

    W3 = _solve_sylvester(Abar, Bbar, P, -K3, direct_limit=direct_limit, order=3,
                          sparse=sparse, canonical_rhs=canonical_rhs)
    if sparse:
        def row_block(start, stop):
            keep = (W3.equations >= start) & (W3.equations < stop)
            return SparseThirdDerivatives(
                W3.equations[keep] - start, W3.indices[keep], W3.values[keep],
                (stop - start, nk, nk, nk), symmetric=True)
        h_kkk, g_kkk = row_block(0, nk), row_block(nk, n)
        resid = _sparse_sylvester_residual(W3, Abar, Bbar, P, K3, canonical=canonical_rhs)
    else:
        # symmetry in the three indices is exact in theory; imposing it removes numerical noise
        W3 = sum(np.transpose(W3, (0,) + tuple(1 + np.array(perm)))
                 for perm in ((0, 1, 2), (0, 2, 1), (1, 0, 2),
                              (1, 2, 0), (2, 0, 1), (2, 1, 0))) / 6.0
        h_kkk, g_kkk = W3[:nk], W3[nk:]
        WPPP = np.einsum("mabc,ai,bj,cl->mijl", W3, P, P, P)
        resid = float(np.max(np.abs(np.einsum("ea,aijl->eijl", Abar, W3)
                                    + np.einsum("ea,aijl->eijl", Bbar, WPPP) + K3)))

    # -- cubic innovation curvature ---------------------------------------
    L = np.vstack([Q, F @ Q, np.zeros((n, ne)), np.eye(ne)])
    g_qq = np.einsum("rab,ai,aj->rij", g_kk, Q, Q) if nu else np.zeros((0, ne, ne))
    d2_uu = np.concatenate([
        second.h_uu,
        np.einsum("ra,aij->rij", F, second.h_uu) + g_qq,
        np.zeros((n + ne, ne, ne)),
    ], axis=0)
    H_uuu = _contract_third(T, L, L, L)
    rows = first.klein.shock_rows
    Aex = A1[rows]
    h_uuu = np.zeros((nk, ne, ne, ne))
    innovation_resid = 0.0
    for a in range(ne):
        for b in range(ne):
            for c in range(ne):
                known = H_uuu[:, a, b, c].copy()
                known += _contract_second(H, d2_uu[:, a, b], L[:, c, None])[:, 0]
                known += _contract_second(H, d2_uu[:, a, c], L[:, b, None])[:, 0]
                known += _contract_second(H, d2_uu[:, b, c], L[:, a, None])[:, 0]
                rhs = -known[rows]
                coef, *_ = np.linalg.lstsq(Aex, rhs, rcond=None)
                difference = Aex @ coef - rhs
                if difference.size:
                    innovation_resid = max(innovation_resid, float(np.max(np.abs(difference))))
                h_uuu[:, a, b, c] = coef
    h_uuu = sum((np.transpose(h_uuu, (0,) + tuple(1 + np.array(perm)))
                 for perm in ((0, 1, 2), (0, 2, 1), (1, 0, 2),
                              (1, 2, 0), (2, 0, 1), (2, 1, 0))),
                np.zeros_like(h_uuu)) / 6.0

    g_kq = np.einsum("rab,ai,bj->rij", g_kk, P, Q) if nu else np.zeros((0, nk, ne))
    d2_ku = np.concatenate([
        second.h_ku,
        np.einsum("ra,aij->rij", F, second.h_ku) + g_kq,
        np.zeros((n + ne, nk, ne)),
    ], axis=0)
    H_kuu = _contract_third(T, Z, L, L)
    h_kuu = np.zeros((nk, nk, ne, ne))
    for i in range(nk):
        for a in range(ne):
            for b in range(ne):
                known = H_kuu[:, i, a, b].copy()
                known += _contract_second(H, d2_ku[:, i, a], L[:, b, None])[:, 0]
                known += _contract_second(H, d2_ku[:, i, b], L[:, a, None])[:, 0]
                known += _contract_second(H, d2_uu[:, a, b], Z[:, i, None])[:, 0]
                rhs = -known[rows]
                coef, *_ = np.linalg.lstsq(Aex, rhs, rcond=None)
                difference = Aex @ coef - rhs
                if difference.size:
                    innovation_resid = max(innovation_resid, float(np.max(np.abs(difference))))
                h_kuu[:, i, a, b] = coef
    h_kuu = 0.5 * (h_kuu + np.transpose(h_kuu, (0, 1, 3, 2)))

    H_kku = _contract_third(T, Z, Z, L)
    h_kku = np.zeros((nk, nk, nk, ne))
    for i in range(nk):
        for j in range(nk):
            for a in range(ne):
                known = H_kku[:, i, j, a].copy()
                known += _contract_second(H, Z2[:, i, j], L[:, a, None])[:, 0]
                known += _contract_second(H, d2_ku[:, i, a], Z[:, j, None])[:, 0]
                known += _contract_second(H, d2_ku[:, j, a], Z[:, i, None])[:, 0]
                rhs = -known[rows]
                coef, *_ = np.linalg.lstsq(Aex, rhs, rcond=None)
                difference = Aex @ coef - rhs
                if difference.size:
                    innovation_resid = max(innovation_resid, float(np.max(np.abs(difference))))
                h_kku[:, i, j, a] = coef
    h_kku = 0.5 * (h_kku + np.transpose(h_kku, (0, 2, 1, 3)))

    Sigma = shock_covariance(model.shocks, cov if cov is not None else second.Sigma)
    # g_ss and h_ss in ``second`` must be those of this covariance, or the third-order rule would
    # mix the risk terms of two economies
    second = second.at_covariance(Sigma)
    g_ssk, h_ssk, slope_resid = _risk_slope(
        nk, nu, n, ne, f1, f2, H, T, P, F, Q, g_kk, g_kkk,
        second.h_uu, second.h_ku, Sigma)
    solution = ThirdOrderSolution(second=second, g_kkk=g_kkk, h_kkk=h_kkk, h_uuu=h_uuu,
                                  h_kuu=h_kuu, h_kku=h_kku, g_ssk=g_ssk, h_ssk=h_ssk,
                                  g_xss=np.zeros((nu, nk)), h_xss=np.zeros((nk, nk)),
                                  g_uss=np.zeros((nu, ne)), h_uss=np.zeros((nk, ne)),
                                  sylvester_residual=resid, risk_slope_residual=slope_resid,
                                  innovation_residual=innovation_resid, Sigma=Sigma)
    g_xss, h_xss, state_resid = _risk_state(solution, H, T)
    solution = replace(solution, g_xss=g_xss, h_xss=h_xss, risk_state_residual=state_resid)
    g_uss, h_uss, shock_resid = _risk_innovation(solution, H, T)
    return replace(solution, g_uss=g_uss, h_uss=h_uss, risk_shock_residual=shock_resid)


def _risk_slope(nk, nu, n, ne, f1, f2, H, T, P, F, Q, g_kk, g_kkk, h_ee, h_ke, Sigma):
    """``g_ssk``, ``h_ssk`` from the curvature of the rule with respect to the innovation.

    ``h_ee`` and ``h_ke`` are the ex-post second-order innovation blocks. Their
    state derivatives, contracted with ``Σ``, determine the risk slope. The jump
    block closes the same linear system as the mean correction in each state direction.
    """
    A1 = f1[:, :nk] + f1[:, nk:] @ F
    A2 = f1[:, nk:]
    A3 = f2[:, nk:]
    lhs = np.hstack([A1, A2 + A3]) if nu else A1
    Z = np.vstack([np.vstack([P, F @ P]),
                   np.vstack([np.eye(nk), F]),
                   np.zeros((ne, nk))])
    L = np.vstack([Q, F @ Q, np.zeros((n, ne)), np.eye(ne)])
    worst = 0.0

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
                             + _contract_third(g_kkk, P[:, i, None], Q[:, a, None],
                                               Q[:, b, None])[:, 0, 0, 0])
                Hpart = (_contract_second(H, d2_ke(i, a), L[:, b, None])[:, 0]
                         + _contract_second(H, d2_ke(i, b), L[:, a, None])[:, 0]
                         + _contract_second(H, d2_ee(a, b), Z[:, i, None])[:, 0])
                Tpart = _contract_third(T, Z[:, i, None], L[:, a, None],
                                        L[:, b, None])[:, 0, 0, 0]
                known = Hpart + Tpart
                if nu:
                    known = known + A2 @ extra
                acc += weight * known
        solved = np.linalg.solve(lhs, -acc)
        h_ssk[:, i] = solved[:nk]
        if nu:
            g_ssk[:, i] = solved[nk:]
    return g_ssk, h_ssk, worst
