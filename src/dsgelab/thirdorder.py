"""Perturbação de terceira ordem.

O que ela acrescenta à segunda. A segunda ordem corrige a MÉDIA das regras de decisão; a terceira
corrige a INCLINAÇÃO delas em resposta ao risco, e é o que faz um prêmio de risco variar no tempo em
vez de ser uma constante. Modelos em que a resposta a um choque depende do tamanho do choque —
assimetria entre expansão e recessão, aversão a risco recursiva, restrições que apertam de um lado
só — não são representáveis em segunda ordem por construção.

Estrutura
---------
A solução ganha os termos cúbicos::

    u_t     = F k + 1/2 g_kk[k,k] + 1/2 g_ss + 1/6 g_kkk[k,k,k] + 1/2 g_ssk[k]
    k_{t+1} = P k + 1/2 h_kk[k,k] + 1/2 h_ss + 1/6 h_kkk[k,k,k] + 1/2 h_ssk[k] + Q eps

Os coeficientes cúbicos ``g_kkk`` e ``h_kkk`` resolvem a MESMA equação de Sylvester generalizada da
segunda ordem, com um fator de Kronecker a mais::

    Abar W3 + Bbar W3 (P kron P kron P) = -K3

Os termos ``g_ssk`` e ``h_ssk`` — o risco entrando na inclinação — resolvem um sistema linear de
tamanho ``n * n_k`` depois disso.

O que está implementado, e o que NÃO está
-----------------------------------------
**Implementado:** o bloco cúbico ``g_kkk`` e ``h_kkk``, verificado pelo teste de ordem de acurácia —
alimentar as regras cúbicas de volta no sistema não linear tem de deixar resíduo de ordem
``||k||^4``, isto é, dividir o deslocamento pela metade tem de dividir o resíduo por dezesseis. É o
mesmo teste que separa a segunda ordem da primeira, um grau acima, e um coeficiente cúbico errado
não o sobrevive.

**Implementado também:** ``g_ssk`` e ``h_ssk``. A inovação continua a entrar na primeira ordem por
``Q eps``. A curvatura de segunda ordem dessa inovação, ``h_ee``, é resolvida antes de tomar a
esperança, e a sua derivada em relação ao estado, contraída com a covariância, é ``h_ssk``. No
AR(1) logarítmico isso tem de dar ``h_ssk[a, a] = ρ σ²``, com sinal positivo: a média condicional
é ``(1+â)^ρ exp(σ²/2)``, e o termo cruzado entre ``â`` e ``σ²`` tem esse coeficiente. Uma derivação
anterior acertava o módulo e invertia o sinal porque lia o termo pelo canal aditivo, que não o
guarda. Aqui o canal é a curvatura em ``eps``, e o sinal sai da equação, não de um ajuste.

O que isto não é: o tensor completo do Dynare, ``g(y_{-1}, u, σ)``, com ``g_uuu`` e ``g_σσu``.
O termo que a regra de decisão usa para o prêmio de risco variar com o estado é ``σ² k``, e é esse
que entra em ``jumps`` e ``next_states``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import solve_discrete_lyapunov

from .linearize import second_derivatives, third_derivatives
from .model import Model
from .secondorder import SecondOrderSolution, _solve_sylvester
from .solution import shock_covariance

__all__ = ["ThirdOrderSolution", "solve_third_order"]


@dataclass
class ThirdOrderSolution:
    """Segunda ordem, mais o bloco cúbico e a inclinação de risco ``σ² k``."""

    second: SecondOrderSolution
    g_kkk: np.ndarray            # (n_u, n_k, n_k, n_k)
    h_kkk: np.ndarray            # (n_k, n_k, n_k, n_k)
    g_ssk: np.ndarray            # (n_u, n_k)   risco na inclinação dos saltos
    h_ssk: np.ndarray            # (n_k, n_k)   risco na inclinação dos estados
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

    # -- regras de decisão ------------------------------------------------
    def jumps(self, k, risk: bool = True) -> np.ndarray:
        k = np.asarray(k, dtype=float)
        out = (self.F @ k
               + 0.5 * np.einsum("rab,a,b->r", self.second.g_kk, k, k)
               + (1.0 / 6.0) * np.einsum("rabc,a,b,c->r", self.g_kkk, k, k, k))
        if risk:
            out = out + 0.5 * self.second.g_ss + 0.5 * self.g_ssk @ k
        return out

    def next_states(self, k, eps=None, risk: bool = True) -> np.ndarray:
        k = np.asarray(k, dtype=float)
        out = (self.P @ k
               + 0.5 * np.einsum("rab,a,b->r", self.second.h_kk, k, k)
               + (1.0 / 6.0) * np.einsum("rabc,a,b,c->r", self.h_kkk, k, k, k))
        if risk:
            out = out + 0.5 * self.second.h_ss + 0.5 * self.h_ssk @ k
        if eps is not None and self.Q is not None:
            out = out + self.Q @ eps
        return out

    # -- simulação --------------------------------------------------------
    def simulate(self, periods: int = 1000, cov=None, *, seed: int | None = None,
                 burn: int = 200, levels: bool = False) -> dict:
        """Simulação PODADA de terceira ordem (Andreasen, Fernández-Villaverde e Rubio-Ramírez).

        A poda é decomposta em três camadas — primeira, segunda e terceira ordem —, cada uma
        alimentada apenas pelas anteriores. Sem ela um mapa cúbico iterado diverge, e mais depressa
        do que o quadrático: não é um artefato a calibrar, é o que fazer com um polinômio de grau
        três num laço.
        """
        rng = np.random.default_rng(seed)
        Sigma = shock_covariance(self.model.shocks,
                                 cov if cov is not None else self.Sigma)
        vals, vecs = np.linalg.eigh(Sigma)
        L = vecs @ np.diag(np.sqrt(np.maximum(vals, 0.0)))
        total = periods + burn
        nk = self.P.shape[0]
        eps = rng.standard_normal((total, self.model.n_shocks)) @ L.T

        kf = np.zeros((total, nk))          # primeira ordem
        ks = np.zeros((total, nk))          # correção de segunda
        kt = np.zeros((total, nk))          # correção de terceira
        u = np.zeros((total, self.F.shape[0]))
        gkk, hkk = self.second.g_kk, self.second.h_kk
        for t in range(1, total):
            f_ant, s_ant, t_ant = kf[t - 1], ks[t - 1], kt[t - 1]
            kf[t] = self.P @ f_ant + self.Q @ eps[t]
            ks[t] = (self.P @ s_ant + 0.5 * np.einsum("rab,a,b->r", hkk, f_ant, f_ant)
                     + 0.5 * self.second.h_ss)
            kt[t] = (self.P @ t_ant + np.einsum("rab,a,b->r", hkk, f_ant, s_ant)
                     + (1.0 / 6.0) * np.einsum("rabc,a,b,c->r", self.h_kkk, f_ant, f_ant, f_ant)
                     + 0.5 * self.h_ssk @ f_ant)
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

    # -- verificação ------------------------------------------------------
    def accuracy(self, sizes=(0.02, 0.01, 0.005), seed: int = 0) -> dict:
        """Verificação determinística: o resíduo tem de cair como ``||k||^4``.

        Os termos de risco são desligados, de modo que o que se lê é o bloco CÚBICO sozinho contra o
        sistema não linear. Dividir o deslocamento pela metade tem de dividir o resíduo por
        dezesseis; uma razão perto de oito significa que o bloco cúbico está errado e a solução não
        é melhor do que a de segunda ordem.
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


def solve_third_order(second: SecondOrderSolution, cov=None, *, h: float = 1e-3,
                      h_hess: float = 1e-4, direct_limit: int = 6000,
                      third: np.ndarray | None = None) -> ThirdOrderSolution:
    """Calcula os termos de terceira ordem a partir de uma solução de segunda.

    Parameters
    ----------
    second : SecondOrderSolution
        Saída de :func:`dsgelab.solve_second_order`.
    cov : mapping, array ou None
        Covariância dos choques. Entra em ``g_ssk`` e ``h_ssk`` pela contração da curvatura
        em ``eps`` com ``Σ``. A simulação usa a mesma matriz.
    h, h_hess : float
        Passos externo e interno da terceira derivada; ver
        :func:`dsgelab.linearize.third_derivatives`.
    third : array, optional
        Tensor de terceiras derivadas já calculado, para reaproveitar numa varredura.
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

    # --- derivadas do mapa composto em relação aos estados -----------------
    M = np.vstack([P, F @ P])
    N = np.vstack([np.eye(nk), F])
    Z = np.vstack([M, N, np.zeros((ne, nk))])                    # (m, nk)

    # segunda derivada do mapa composto
    d2_yp_k = h_kk
    d2_yp_u = np.einsum("ra,aij->rij", F, h_kk) + np.einsum("rab,ai,bj->rij", g_kk, P, P)
    Z2 = np.concatenate([d2_yp_k, d2_yp_u,
                         np.zeros((nk, nk, nk)), g_kk,
                         np.zeros((ne, nk, nk))], axis=0)        # (m, nk, nk)

    # --- blocos da equação de Sylvester ------------------------------------
    A1 = f1[:, :nk] + f1[:, nk:] @ F
    A2 = f1[:, nk:]
    A3 = f2[:, nk:]
    Abar = np.hstack([A1, A3])
    Bbar = np.hstack([np.zeros((n, nk)), A2])

    # termo conhecido: g_kk cruzado com (h_kk, P), somado sobre as três permutações
    cruz = (np.einsum("rab,aij,bl->rijl", g_kk, h_kk, P)
            + np.einsum("rab,ail,bj->rijl", g_kk, h_kk, P)
            + np.einsum("rab,ajl,bi->rijl", g_kk, h_kk, P))

    K3 = (np.einsum("ea,aijl->eijl", A2, cruz)
          + np.einsum("eab,ai,bjl->eijl", H, Z, Z2)
          + np.einsum("eab,aj,bil->eijl", H, Z, Z2)
          + np.einsum("eab,al,bij->eijl", H, Z, Z2)
          + np.einsum("eabc,ai,bj,cl->eijl", T, Z, Z, Z))

    W3 = _solve_sylvester(Abar, Bbar, P, -K3, direct_limit=direct_limit, order=3)
    # a simetria nos três índices é exata na teoria; impô-la remove o ruído numérico
    W3 = sum(np.transpose(W3, (0,) + tuple(1 + np.array(perm)))
             for perm in ((0, 1, 2), (0, 2, 1), (1, 0, 2),
                          (1, 2, 0), (2, 0, 1), (2, 1, 0))) / 6.0
    h_kkk, g_kkk = W3[:nk], W3[nk:]

    WPPP = np.einsum("mabc,ai,bj,cl->mijl", W3, P, P, P)
    resid = float(np.max(np.abs(np.einsum("ea,aijl->eijl", Abar, W3)
                                + np.einsum("ea,aijl->eijl", Bbar, WPPP) + K3)))

    Sigma = shock_covariance(model.shocks, cov if cov is not None else second.Sigma)
    g_ssk, h_ssk, slope_resid = _risk_slope(
        nk, nu, n, ne, f1, f2, H, T, P, F, Q, g_kk, g_kkk, Sigma)
    return ThirdOrderSolution(second=second, g_kkk=g_kkk, h_kkk=h_kkk,
                              g_ssk=g_ssk, h_ssk=h_ssk,
                              sylvester_residual=resid, risk_slope_residual=slope_resid,
                              Sigma=Sigma)


def _risk_slope(nk, nu, n, ne, f1, f2, H, T, P, F, Q, g_kk, g_kkk, Sigma):
    """``g_ssk``, ``h_ssk`` a partir da curvatura da regra em relação à inovação.

    A regra de primeira ordem escreve ``k' = P k + Q eps``. A segunda derivada dessa regra em
    ``eps``, ``h_ee``, resolve o resíduo ex post diferenciado duas vezes. A esperança
    ``h_ee : Σ`` é a correção de média. Diferenciar essa curvatura em relação ao estado e
    contrair de novo com ``Σ`` é ``h_ssk``. ``g_ssk`` fecha o mesmo sistema linear que a
    correção de média, agora em cada direção do estado.
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
