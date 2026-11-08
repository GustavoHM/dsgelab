"""compare_heterogeneous.py: heterogeneous agents by the sequence-space Jacobian, against
sequence-jacobian and against brute force.

  1. The Krusell-Smith model of sequence-jacobian's krusell_smith.ipynb and the one-asset HANK of
     its hank.ipynb, both at the v1.0.0 tag, solved by sequence-jacobian 1.0.0
     (sequence_jacobian/run_ssj.py, in the environment of sequence_jacobian/setup.sh) and by
     dsgelab's heterogeneous module, with the same calibration, grids, unknowns and targets. The
     household on 7 income states by 500 asset points, the endogenous grid method written again
     with numpy (dsgelab's core does not use numba). Compared, relative to the largest value of
     each: the grids; the steady state's unknowns, policies, distribution and aggregates; the
     household block's Jacobians from the fake-news algorithm, T = 300; the general-equilibrium
     Jacobians; and the linear and nonlinear impulse responses of the notebooks.
     sequence-jacobian differentiates one backward step by forward differences with step 1e-4, so
     its Jacobians differ from the derivative by some 1e-5 of their size. The comparison runs
     dsgelab with the same differences (twosided=False, h=1e-4), and compares the central
     differences that dsgelab uses by default with sequence-jacobian's own (twosided=True).
  2. The household of labor_search.ipynb, at the same tag, whose employment chain is a hetinput
     of the job-finding rate f and the separation rate s, so that its Markov matrix moves along a
     path. sequence-jacobian keeps employment and productivity on two axes with two Markov
     matrices, and dsgelab on one chain of 10 nodes with their Kronecker product. The notebook
     stops at the household, and run_ssj.py closes it with a fixed supply of bonds and the
     interest rate as the unknown, written identically here. Compared as in 1, with the
     Jacobians with respect to f and s, the notebook's linear response to s, and the return to
     the steady state from the distribution of the steady state with s = 0.12, which
     sequence-jacobian computes with ss_initial and dsgelab with initial.
  3. The two-asset HANK of two_asset.ipynb, with a liquid and an illiquid asset, on 3 income
     states by 50 liquid by 70 illiquid points. Compared as in 1, including the two-dimensional
     distribution and the adjustment costs, with both packages' backward and nonlinear iterations
     tightened as in the other models.
  4. dsgelab against itself. The fake-news Jacobian against the direct one, the nonlinear path
     differentiated by central differences for a shock in each period, on ten of the 300 columns
     and with steps of 1e-5, 1e-6 and 1e-7 in both (1e-3, 1e-4 and 1e-5 for the labor-search
     household, with respect to f and s). The linear impulse responses of Krusell-Smith and of
     labor search against the nonlinear transitions, for shocks of 1, 0.1 and 0.01 per cent to
     productivity and of 1, 0.1 and 0.01 points to s: the difference must fall as the square of
     the shock. The labor-search household's path from the distribution with s = 0.12 is checked
     against its transition matrix applied period by period, and the two-asset household's
     fake-news Jacobian is checked on 3 by 15 by 20 points.

Prerequisites:
    bash validation/sequence_jacobian/setup.sh
    validation/sequence_jacobian/.venv/bin/python validation/sequence_jacobian/run_ssj.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

import dsgelab as dsg
from dsgelab import (BlockModel, HetBlock, SimpleBlock, SolvedBlock, interpolate_rows, log_grid,
                     simple)
from dsgelab.heterogeneous import _Lottery

HERE = Path(__file__).resolve().parent
OUT = HERE / "out_heterogeneous"
T = 300
BACKWARD_TOL = 1e-12
NONLINEAR_TOL = 1e-11
COLUMNS = [0, 1, 2, 5, 10, 20, 50, 100, 200, 299]
TWO_AGGREGATES = ["vphi", "mup", "alpha", "Z", "w", "p", "mc", "pi", "Q", "ra", "rb",
                  "tax", "div", "I", "psip", "pshare", "A", "B", "C", "UCE", "CHI", "wealth",
                  "asset_mkt", "goods_mkt", "wnkpc", "fisher", "nkpc", "equity", "inv", "val"]
failures: list[str] = []


def check(label: str, value: float, limit: float) -> None:
    status = "ok" if value <= limit else "FAIL"
    if status == "FAIL":
        failures.append(label)
    print(f"  {status:4s} {label:60s} {value:9.2e}  (limit {limit:.0e})")


def check_at_least(label: str, value: float, floor: float) -> None:
    status = "ok" if value >= floor else "FAIL"
    if status == "FAIL":
        failures.append(label)
    print(f"  {status:4s} {label:60s} {value:9.2f}  (at least {floor:g})")


def rel(x, ref) -> float:
    """The largest difference, relative to the largest absolute value of the reference."""
    ref = np.asarray(ref, dtype=float)
    return float(np.max(np.abs(np.asarray(x, dtype=float) - ref)) / np.max(np.abs(ref)))


def worst(pairs) -> tuple[float, str]:
    """The largest relative difference among the pairs, and its name."""
    return max((rel(x, ref), k) for k, x, ref in pairs)


def worst_in_group(pairs) -> tuple[float, str]:
    """The largest difference among the pairs, relative to the largest value in any of them.

    In general equilibrium some responses are zero, such as assets in the HANK model, where the
    asset market clears at a fixed supply of bonds; a difference relative to their own size would
    compare rounding errors.
    """
    pairs = [(k, np.asarray(x, dtype=float), np.asarray(r, dtype=float)) for k, x, r in pairs]
    scale = max(float(np.max(np.abs(r))) for _, _, r in pairs)
    return max((float(np.max(np.abs(x - r))) / scale, k) for k, x, r in pairs)


# --- the two models, written for dsgelab -----------------------------------------------------

def income_chain(rho_e, sd_e, nE):
    """sequence-jacobian's markov_rouwenhorst: sd_e is the unconditional s.d. of log income, and
    the nodes are scaled to a mean of one under the stationary distribution."""
    chain = dsg.rouwenhorst(rho_e, sd_e * np.sqrt(1 - rho_e ** 2), int(nE))
    pi_e = chain.stationary
    e = np.exp(chain.nodes)
    return e / (pi_e @ e), pi_e, chain.transition


def krusell_smith():
    def household_init(a_grid, e_grid, r, w, eis):
        coh = (1 + r) * a_grid[np.newaxis, :] + w * e_grid[:, np.newaxis]
        return (1 + r) * (0.1 * coh) ** (-1 / eis)

    def household(Va_p, a_grid, e_grid, r, w, beta, eis):
        uc_nextgrid = beta * Va_p
        c_nextgrid = uc_nextgrid ** (-eis)
        coh = (1 + r) * a_grid[np.newaxis, :] + w * e_grid[:, np.newaxis]
        a = np.maximum(interpolate_rows(c_nextgrid + a_grid, coh, a_grid), a_grid[0])
        c = coh - a
        Va = (1 + r) * c ** (-1 / eis)
        return Va, a, c

    def make_grid(rho_e, sd_e, nE, amin, amax, nA):
        e_grid, _, Pi = income_chain(rho_e, sd_e, nE)
        return e_grid, Pi, log_grid(amin, amax, nA)

    hh = HetBlock(household, ["Va", "a", "c"], backward="Va", policy="a", init=household_init,
                  hetinputs=[(make_grid, ["e_grid", "Pi", "a_grid"])],
                  backward_tol=BACKWARD_TOL)

    @simple("r", "w", "Y")
    def firm(K, L, Z, alpha, delta):
        r = alpha * Z * (K(-1) / L) ** (alpha - 1) - delta
        w = (1 - alpha) * Z * (K(-1) / L) ** alpha
        Y = Z * K(-1) ** alpha * L ** (1 - alpha)
        return r, w, Y

    @simple("asset_mkt", "goods_mkt")
    def mkt_clearing(K, A, Y, C, delta):
        return A - K, Y - C - delta * K

    return hh, BlockModel([hh, firm, mkt_clearing], name="Krusell-Smith")


def hank():
    def hh_init(a_grid, we, r, eis, T):
        coh = (1 + r) * a_grid[np.newaxis, :] + we[:, np.newaxis] + T[:, np.newaxis]
        return (1 + r) * (0.1 * coh) ** (-1 / eis)

    def hh(Va_p, a_grid, we, T, r, beta, eis, frisch, vphi):
        uc_nextgrid = beta * Va_p
        c_nextgrid = uc_nextgrid ** (-eis)
        n_nextgrid = (we[:, np.newaxis] * uc_nextgrid / vphi) ** frisch
        lhs = c_nextgrid - we[:, np.newaxis] * n_nextgrid + a_grid[np.newaxis, :] - T[:, np.newaxis]
        rhs = (1 + r) * a_grid
        c = interpolate_rows(lhs, rhs, c_nextgrid)
        n = interpolate_rows(lhs, rhs, n_nextgrid)
        a = rhs + we[:, np.newaxis] * n + T[:, np.newaxis] - c
        con = a < a_grid[0]
        a[con] = a_grid[0]
        if con.any():
            # at the limit, c = w n + T + (1 + r) a - amin, solved by Newton's method on log u'(c),
            # from u'(c) = Va_p, as hetblocks.hh_labor does point by point. It stops at 1e-11 and
            # this at 1e-14, so that the finite differences of section 3, with steps down to
            # 1e-7, do not see where the iteration stopped
            w_c = np.broadcast_to(we[:, np.newaxis], a.shape)[con]
            inc = (rhs[np.newaxis, :] + T[:, np.newaxis] - a_grid[0])[con]
            log_uc = np.log(Va_p[con])
            for _ in range(30):
                cc = np.exp(-eis * log_uc)
                nn = (w_c * np.exp(log_uc) / vphi) ** frisch
                gap = cc - w_c * nn - inc
                if np.max(np.abs(gap)) < 1e-14:
                    break
                log_uc -= gap / (-eis * cc - w_c * frisch * nn)
            else:
                raise RuntimeError("the constrained households' problem did not converge")
            c[con] = np.exp(-eis * log_uc)
            n[con] = (w_c * np.exp(log_uc) / vphi) ** frisch
        Va = (1 + r) * c ** (-1 / eis)
        return Va, a, c, n

    def make_grid(rho_e, sd_e, nE, amin, amax, nA):
        e_grid, pi_e, Pi = income_chain(rho_e, sd_e, nE)
        return e_grid, pi_e, Pi, log_grid(amin, amax, nA)

    def transfers(pi_e, Div, Tax, e_grid):
        tax_rule, div_rule = e_grid, e_grid
        div = Div / np.sum(pi_e * div_rule) * div_rule
        tax = Tax / np.sum(pi_e * tax_rule) * tax_rule
        return div - tax

    def wages(w, e_grid):
        return w * e_grid

    def labor_supply(n, e_grid):
        return e_grid[:, np.newaxis] * n

    household = HetBlock(hh, ["Va", "a", "c", "n"], backward="Va", policy="a", init=hh_init,
                         hetinputs=[(make_grid, ["e_grid", "pi_e", "Pi", "a_grid"]),
                                    (transfers, "T"), (wages, "we")],
                         hetoutputs=[(labor_supply, "ne")], backward_tol=BACKWARD_TOL)

    @simple("L", "Div")
    def firm(Y, w, Z, pi, mu, kappa):
        L = Y / Z
        Div = Y - w * L - mu / (mu - 1) / (2 * kappa) * np.log(1 + pi) ** 2 * Y
        return L, Div

    @simple("r")
    def monetary(pi, rstar, phi):
        return (1 + rstar(-1) + phi * pi(-1)) / (1 + pi) - 1

    @simple("Tax")
    def fiscal(r, B):
        return r * B

    @simple("asset_mkt", "labor_mkt", "goods_mkt")
    def mkt_clearing(A, NE, C, L, Y, B, pi, mu, kappa):
        goods_mkt = Y - C - mu / (mu - 1) / (2 * kappa) * np.log(1 + pi) ** 2 * Y
        return A - B, NE - L, goods_mkt

    @simple("w")
    def nkpc_ss(Z, mu):
        return Z / mu

    @simple("nkpc_res")
    def nkpc(pi, w, Z, Y, r, mu, kappa):
        return (kappa * (w / Z - 1 / mu) + Y(+1) / Y * np.log(1 + pi(+1)) / (1 + r(+1))
                - np.log(1 + pi))

    blocks = [household, firm, monetary, fiscal, mkt_clearing]
    return (household, BlockModel(blocks + [nkpc_ss], name="One-Asset HANK SS"),
            BlockModel(blocks + [nkpc], name="One-Asset HANK"))


def labor_search():
    """labor_search.ipynb's household, whose employment chain moves with f and s, closed by a
    fixed supply of bonds as in run_ssj.py. sequence-jacobian keeps the employment status and the
    productivity on two axes, each with its Markov matrix; here they are one chain of 2 nE nodes,
    employed first, whose matrix is the Kronecker product of the two."""
    def make_grid(rho_e, sd_e, nE, amin, amax, nA):
        e_grid, _, Pi_e = income_chain(rho_e, sd_e, nE)
        return e_grid, Pi_e, log_grid(amin, amax, nA)

    def search_frictions(f, s, Pi_e):
        Pi_s = np.vstack(([1 - s, s], [f, 1 - f]))
        return Pi_s, np.kron(Pi_s, Pi_e)

    def labor_income(e_grid, w, b):
        return np.concatenate((w * e_grid, b * e_grid))

    def household_init(a_grid, y, r, eis):
        c = np.maximum(1e-8, y[:, np.newaxis] + np.maximum(r, 0.04) * a_grid)
        return (1 + r) * (c ** (-1 / eis))

    def household(Va_p, a_grid, y, r, beta, eis):
        c_nextgrid = (beta * Va_p) ** (-eis)
        coh = (1 + r) * a_grid + y[:, np.newaxis]
        a = np.maximum(interpolate_rows(c_nextgrid + a_grid, coh, a_grid), a_grid[0])
        c = coh - a
        Va = (1 + r) * c ** (-1 / eis)
        return Va, a, c

    def unemployment(c):
        u = np.zeros_like(c)
        u[c.shape[0] // 2:] = 1.0
        return u

    hh = HetBlock(household, ["Va", "a", "c"], backward="Va", policy="a", init=household_init,
                  hetinputs=[(make_grid, ["e_grid", "Pi_e", "a_grid"]),
                             (search_frictions, ["Pi_s", "Pi"]), (labor_income, "y")],
                  hetoutputs=[(unemployment, "u")], backward_tol=BACKWARD_TOL)

    @simple("asset_mkt")
    def bond_market(A, B):
        return A - B

    return hh, BlockModel([hh, bond_market], name="Labor search")


def get_Psi_and_deriv(ap, a, ra, chi0, chi1, chi2):
    """The adjustment cost Psi(ap, a) and its derivatives with respect to ap and to a, as
    hetblocks.hh_twoasset writes them."""
    a_with_return = (1 + ra) * a
    a_change = ap - a_with_return
    abs_a_change = np.abs(a_change)
    sign_change = np.sign(a_change)
    adj_denominator = a_with_return + chi0
    core_factor = (abs_a_change / adj_denominator) ** (chi2 - 1)
    Psi = chi1 / chi2 * abs_a_change * core_factor
    Psi1 = chi1 * sign_change * core_factor
    Psi2 = -(1 + ra) * (Psi1 + (chi2 - 1) * Psi / adj_denominator)
    return Psi, Psi1, Psi2


def lhs_equals_rhs(lhs, rhs):
    """For each j, the bracket of the first i at which lhs[..., i] falls below rhs[i, j], and the
    weight on its lower point that makes the two equal by linear interpolation: the result of
    hh_twoasset's lhs_equals_rhs_interpolate, written for numpy. Where lhs starts below rhs, the
    index is 0 and the weight 1, and where it never falls below, the last bracket extrapolates."""
    n = rhs.shape[0]
    below = lhs[..., :, None] < rhs                       # (..., i, j)
    first = np.where(below.any(axis=-2), below.argmax(axis=-2), n - 1)
    lower = np.maximum(first - 1, 0)
    j = np.arange(rhs.shape[1])
    err_upper = rhs[first, j] - np.take_along_axis(lhs, first, axis=-1)
    err_lower = rhs[lower, j] - np.take_along_axis(lhs, lower, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        pi = np.where(first == 0, 1.0, err_upper / (err_upper - err_lower))
    return lower, pi


def apply_coord(i, pi, y):
    """pi y[i] + (1 - pi) y[i + 1] along the last axis, as sequence-jacobian's apply_coord."""
    y = np.broadcast_to(y, i.shape[:-1] + y.shape[-1:])
    return (pi * np.take_along_axis(y, i, axis=-1)
            + (1 - pi) * np.take_along_axis(y, i + 1, axis=-1))


def two_asset():
    """The two-asset HANK of two_asset.ipynb, with the household of hetblocks.hh_twoasset
    written again with numpy, and the aggregate blocks of examples.two_asset."""
    def hh_init(b_grid, a_grid, z_grid, eis):
        Va = ((0.6 + 1.1 * b_grid[:, np.newaxis] + a_grid) ** (-1 / eis)
              * np.ones((z_grid.shape[0], 1, 1)))
        Vb = ((0.5 + b_grid[:, np.newaxis] + 1.2 * a_grid) ** (-1 / eis)
              * np.ones((z_grid.shape[0], 1, 1)))
        return Va, Vb

    def hh(Va_p, Vb_p, a_grid, b_grid, z_grid, e_grid, k_grid, beta, eis, rb, ra, chi0, chi1,
           chi2, Psi1):
        # the expected marginal values of the two assets, on the grid of (z, b', a')
        Wb = beta * Vb_p
        Wa = beta * Va_p
        W_ratio = Wa / Wb
        # unconstrained: a'(z, b', a) from W_ratio = 1 + Psi1(a', a), then b(z, b', a) from the
        # budget, inverted to b'(z, b, a) and a'(z, b, a)
        i, pi = lhs_equals_rhs(W_ratio, 1 + Psi1)
        a_endo_unc = apply_coord(i, pi, a_grid)
        c_endo_unc = apply_coord(i, pi, Wb) ** (-eis)
        b_endo = (c_endo_unc + a_endo_unc
                  + (-z_grid[:, None, None] + b_grid[:, None] - (1 + ra) * a_grid)
                  + get_Psi_and_deriv(a_endo_unc, a_grid, ra, chi0, chi1, chi2)[0]) / (1 + rb)
        a_unc = interpolate_rows(b_endo.swapaxes(1, 2), b_grid,
                                 a_endo_unc.swapaxes(1, 2)).swapaxes(1, 2)
        b_unc = interpolate_rows(b_endo.swapaxes(1, 2), b_grid, b_grid).swapaxes(1, 2)
        # constrained at b' = b_grid[0], with kappa the multiplier of the liquid constraint
        lhs_con = W_ratio[:, 0:1, :] / (1 + k_grid[np.newaxis, :, np.newaxis])
        i, pi = lhs_equals_rhs(lhs_con, 1 + Psi1)
        a_endo_con = apply_coord(i, pi, a_grid)
        c_endo_con = ((1 + k_grid[np.newaxis, :, np.newaxis]) ** (-eis)
                      * apply_coord(i, pi, Wb[:, 0:1, :]) ** (-eis))
        b_endo = (c_endo_con + a_endo_con
                  + (-z_grid[:, None, None] + b_grid[0] - (1 + ra) * a_grid)
                  + get_Psi_and_deriv(a_endo_con, a_grid, ra, chi0, chi1, chi2)[0]) / (1 + rb)
        a_con = interpolate_rows(b_endo.swapaxes(1, 2), b_grid,
                                 a_endo_con.swapaxes(1, 2)).swapaxes(1, 2)
        # the constrained solution where the unconstrained one goes below the liquid limit
        a, b = a_unc.copy(), b_unc.copy()
        b[b <= b_grid[0]] = b_grid[0]
        a[b <= b_grid[0]] = a_con[b <= b_grid[0]]
        Psi, _, Psi2 = get_Psi_and_deriv(a, a_grid, ra, chi0, chi1, chi2)
        c = (z_grid[:, None, None] + (1 + rb) * b_grid[:, None] + (1 + ra) * a_grid
             - Psi - a - b)
        uc = c ** (-1 / eis)
        uce = e_grid[:, np.newaxis, np.newaxis] * uc
        Va = (1 + ra - Psi2) * uc
        Vb = (1 + rb) * uc
        return Va, Vb, a, b, c, uce

    def make_grids(bmax, amax, kmax, nB, nA, nK, nZ, rho_z, sigma_z):
        b_grid = log_grid(0.0, bmax, nB)
        a_grid = log_grid(0.0, amax, nA)
        k_grid = log_grid(0.0, kmax, nK)[::-1].copy()
        e_grid, _, Pi = income_chain(rho_z, sigma_z, nZ)
        return b_grid, a_grid, k_grid, e_grid, Pi

    def income(e_grid, tax, w, N):
        return (1 - tax) * w * N * e_grid

    def marginal_cost_grid(a_grid, ra, chi0, chi1, chi2):
        return get_Psi_and_deriv(a_grid[:, np.newaxis], a_grid[np.newaxis, :], ra, chi0, chi1,
                                 chi2)[1]

    def adjustment_costs(a, a_grid, ra, chi0, chi1, chi2):
        return get_Psi_and_deriv(a, a_grid, ra, chi0, chi1, chi2)[0]

    household = HetBlock(hh, ["Va", "Vb", "a", "b", "c", "uce"], backward=["Va", "Vb"],
                         policy=["b", "a"], init=hh_init,
                         hetinputs=[(make_grids, ["b_grid", "a_grid", "k_grid", "e_grid", "Pi"]),
                                    (income, "z_grid"), (marginal_cost_grid, "Psi1")],
                         hetoutputs=[(adjustment_costs, "chi")], backward_tol=BACKWARD_TOL,
                         backward_max_iter=100_000, name="hh")

    def nkpc_residual(pi, mc, r, Y, kappap, mup):
        return (kappap * (mc - 1 / mup) + Y(+1) / Y * np.log(1 + pi(+1)) / (1 + r(+1))
                - np.log(1 + pi))

    pricing_solved = SolvedBlock(SimpleBlock(nkpc_residual, "nkpc"),
                                 unknowns={"pi": (-0.1, 0.1)}, targets=["nkpc"],
                                 name="pricing_solved")

    @simple("equity")
    def arbitrage(div, p, r):
        return div(+1) + p(+1) - p * (1 + r(+1))

    arbitrage_solved = SolvedBlock(arbitrage, unknowns={"p": (5, 15)}, targets=["equity"],
                                   name="arbitrage_solved")

    @simple("N", "mc")
    def labor(Y, w, K, Z, alpha):
        N = (Y / Z / K(-1) ** alpha) ** (1 / (1 - alpha))
        mc = w * N / (1 - alpha) / Y
        return N, mc

    @simple("inv", "val")
    def investment(Q, K, r, N, mc, Z, delta, epsI, alpha):
        inv = (K / K(-1) - 1) / (delta * epsI) + 1 - Q
        val = (alpha * Z(+1) * (N(+1) / K) ** (1 - alpha) * mc(+1)
               - (K(+1) / K - (1 - delta) + (K(+1) / K - 1) ** 2 / (2 * delta * epsI))
               + K(+1) / K * Q(+1) - (1 + r(+1)) * Q)
        return inv, val

    production_solved = SolvedBlock([labor, investment], unknowns={"Q": 1.0, "K": 10.0},
                                    targets=["inv", "val"], name="production_solved")

    @simple("psip", "I", "div")
    def dividend(Y, w, N, K, pi, mup, kappap, delta, epsI):
        psip = mup / (mup - 1) / 2 / kappap * np.log(1 + pi) ** 2 * Y
        k_adjust = K(-1) * (K / K(-1) - 1) ** 2 / (2 * delta * epsI)
        I = K - (1 - delta) * K(-1) + k_adjust
        div = Y - w * N - I - psip
        return psip, I, div

    @simple("i")
    def taylor(rstar, pi, phi):
        return rstar + phi * pi

    @simple("tax")
    def fiscal(r, w, N, G, Bg):
        return (r * Bg + G) / w / N

    @simple("rb", "ra", "fisher")
    def finance(i, p, pi, r, div, omega, pshare):
        rb = r - omega
        ra = pshare(-1) * (div + p) / p(-1) + (1 - pshare(-1)) * (1 + r) - 1
        fisher = 1 + i(-1) - (1 + r) * (1 + pi)
        return rb, ra, fisher

    @simple("piw")
    def wage(pi, w):
        return (1 + pi) * w / w(-1) - 1

    @simple("wnkpc")
    def union(piw, N, tax, w, UCE, kappaw, muw, vphi, frisch, beta):
        return (kappaw * (vphi * N ** (1 + 1 / frisch) - (1 - tax) * w * N * UCE / muw)
                + beta * np.log(1 + piw(+1)) - np.log(1 + piw))

    @simple("asset_mkt", "wealth", "goods_mkt")
    def mkt_clearing(p, A, B, Bg, C, I, G, CHI, psip, omega, Y):
        wealth = A + B
        asset_mkt = p + Bg - wealth
        goods_mkt = C + I + G + CHI + psip + omega * B - Y
        return asset_mkt, wealth, goods_mkt

    @simple("pshare")
    def share_value(p, tot_wealth, Bh):
        return p / (tot_wealth - Bh)

    @simple("p", "mc", "mup", "alpha", "Z", "w")
    def partial_ss(Y, N, K, r, tot_wealth, Bg, delta):
        p = tot_wealth - Bg
        mc = 1 - r * (p - K) / Y
        mup = 1 / mc
        alpha = (r + delta) * K / Y / mc
        Z = Y * K ** (-alpha) * N ** (alpha - 1)
        w = mc * (1 - alpha) * Y / N
        return p, mc, mup, alpha, Z, w

    @simple("vphi", "wnkpc")
    def union_ss(tax, w, UCE, N, muw, frisch):
        vphi = (1 - tax) * w * UCE / muw / N ** (1 + 1 / frisch)
        wnkpc = vphi * N ** (1 + 1 / frisch) - (1 - tax) * w * UCE / muw
        return vphi, wnkpc

    model = BlockModel([household, production_solved, pricing_solved, arbitrage_solved,
                        dividend, taylor, fiscal, share_value, finance, wage, union,
                        mkt_clearing], name="Two-Asset HANK")
    model_ss = BlockModel([household, partial_ss, union_ss, dividend, taylor, fiscal,
                           share_value, finance, mkt_clearing], name="Two-Asset HANK SS")
    return household, model_ss, model


def calibration_two_asset():
    return {"Y": 1.0, "N": 1.0, "K": 10.0, "r": 0.0125, "rstar": 0.0125, "tot_wealth": 14,
            "delta": 0.02, "pi": 0.0, "kappap": 0.1, "muw": 1.1, "Bh": 1.04, "Bg": 2.8,
            "G": 0.2, "eis": 0.5, "frisch": 1.0, "chi0": 0.25, "chi2": 2, "epsI": 4,
            "omega": 0.005, "kappaw": 0.1, "phi": 1.5, "nZ": 3, "nB": 50, "nA": 70, "nK": 50,
            "bmax": 50, "amax": 4000, "kmax": 1, "rho_z": 0.966, "sigma_z": 0.92}


# --- comparisons -----------------------------------------------------------------------------

def compare_grids(ss, ref, prefix, names):
    internals = ss.internals[next(iter(ss.internals))]
    value, name = worst((k, internals[k], ref[f"{prefix}/internals/{k}"]) for k in names)
    check(f"grids and income chain (largest: {name})", value, 1e-9)


def compare_steady_state(ss, ref, prefix, unknowns, policies, aggregates):
    internals = ss.internals[next(iter(ss.internals))]
    value, name = worst((k, ss[k], ref[f"{prefix}/ss/{k}"]) for k in unknowns)
    check(f"steady state: unknowns (largest: {name})", value, 1e-9)
    value, name = worst((k, internals[k], ref[f"{prefix}/internals/{k}"]) for k in policies)
    check(f"steady state: policies (largest: {name})", value, 1e-9)
    check("steady state: distribution D, relative to its largest mass",
          rel(internals["D"], ref[f"{prefix}/internals/D"]), 1e-7)
    value, name = worst((k, ss[k], ref[f"{prefix}/ss/{k}"]) for k in aggregates)
    check(f"steady state: aggregates (largest: {name})", value, 1e-9)


def compare_jacobians(J, ref, key, label, limit):
    pairs = [(k, J[k.split("/")[0]][k.split("/")[1]], ref[f"{key}/{k}"])
             for k in sorted({k[len(key) + 1:] for k in ref.files if k.startswith(key + "/")})]
    value, name = worst_in_group(pairs)
    check(f"{label} (largest: {name})", value, limit)


def fake_news_vs_direct(block, ss, inputs, label, steps=(1e-5, 1e-6, 1e-7), check_at=-1):
    """The fake-news Jacobian against the direct one, on the columns in COLUMNS.

    Both are finite differences with the same step, of one backward step in the first and of the
    whole path in the second. The policy has kinks, where the borrowing limit starts to bind and
    at every grid point the lottery crosses, so the two differ by the error of the differences,
    which falls with the step until roundoff dominates.
    """
    saved = (block.h, block.twosided)
    gaps = []
    try:
        for h in steps:
            block.h, block.twosided = h, True
            J = block.jacobian(ss, inputs, T)
            largest = 0.0
            for i in inputs:
                for s in COLUMNS:
                    up, dn = np.full(T, float(ss[i])), np.full(T, float(ss[i]))
                    up[s] += h
                    dn[s] -= h
                    Yu, Yd = block.path(ss, {i: up}, T), block.path(ss, {i: dn}, T)
                    for O in J:
                        col = (Yu[O] - Yd[O]) / (2 * h)
                        largest = max(largest, float(np.max(np.abs(col - J[O][i][:, s]))
                                                     / np.max(np.abs(J[O][i]))))
            gaps.append(largest)
    finally:
        block.h, block.twosided = saved
    print(f"       {label}: fake news against direct, {len(COLUMNS)} columns, steps "
          + ", ".join(f"{h:.0e}" for h in steps) + ": " + ", ".join(f"{g:.1e}" for g in gaps))
    check(f"{label}: fake news against direct, step {steps[check_at]:.0e}", gaps[check_at], 1e-7)


def compare_impulses(irf, ref, key, label, limit):
    names = sorted({k[len(key) + 1:] for k in ref.files if k.startswith(key + "/")})
    value, name = worst_in_group([(k, irf[k], ref[f"{key}/{k}"]) for k in names])
    check(f"{label} (largest: {name})", value, limit)


def main() -> None:
    path = OUT / "ssj.npz"
    if not path.exists():
        sys.exit(f"{path} not found: run sequence_jacobian/setup.sh and run_ssj.py first")
    ref = np.load(path)
    meta = json.loads((OUT / "ssj.json").read_text())
    print(f"sequence-jacobian {meta['sequence-jacobian']} (numpy {meta['numpy']}, numba "
          f"{meta['numba']}), T = {meta['T']}")

    print("1. Krusell-Smith (krusell_smith.ipynb), 7 income states by 500 asset points")
    hh, ks = krusell_smith()
    calibration = {"eis": 1, "delta": 0.025, "alpha": 0.11, "rho_e": 0.966, "sd_e": 0.5,
                   "L": 1.0, "nE": 7, "nA": 500, "amin": 0, "amax": 200}
    t0 = time.perf_counter()
    ss = ks.steady_state(calibration, {"beta": 0.98, "Z": 0.85, "K": 3.0},
                         {"r": 0.01, "Y": 1.0, "asset_mkt": 0.0}, solver="hybr")
    t_ss = time.perf_counter() - t0
    compare_grids(ss, ref, "ks", ["a_grid", "e_grid", "Pi"])
    compare_steady_state(ss, ref, "ks", ["beta", "Z", "K"], ["a", "c", "Va"], ["A", "C", "Y"])
    print(f"       goods market (Walras), at asset market clearing: dsgelab "
          f"{ss['goods_mkt']:.1e}, sequence-jacobian {float(ref['ks/ss/goods_mkt']):.1e}")
    hh.twosided = False
    t0 = time.perf_counter()
    J = hh.jacobian(ss, ["r", "w"], T)
    t_j = time.perf_counter() - t0
    compare_jacobians(J, ref, "ks/J", "household Jacobians, forward differences", 1e-7)
    t0 = time.perf_counter()
    G = ks.solve_jacobian(ss, ["K"], ["asset_mkt"], ["Z"], T=T)
    t_g = time.perf_counter() - t0
    compare_jacobians(G, ref, "ks/G", "general-equilibrium Jacobians G", 1e-7)
    shock = {"Z": 0.01 * 0.8 ** np.arange(T)}
    compare_impulses(ks.impulse_linear(ss, ["K"], ["asset_mkt"], shock), ref, "ks/linear",
                     "linear impulse responses to Z", 1e-7)
    t0 = time.perf_counter()
    nonlin = ks.impulse_nonlinear(ss, ["K"], ["asset_mkt"], shock, tol=NONLINEAR_TOL)
    t_nl = time.perf_counter() - t0
    compare_impulses(nonlin, ref, "ks/nonlinear", "nonlinear transition after Z", 1e-8)
    hh.twosided = True
    compare_jacobians(hh.jacobian(ss, ["r", "w"], T), ref, "ks/J2",
                      "household Jacobians, central differences", 1e-7)
    print(f"       seconds, dsgelab / sequence-jacobian: steady state {t_ss:.1f} / "
          f"{meta['seconds']['ks steady state']:.1f}, household Jacobians {t_j:.1f} / "
          f"{meta['seconds']['ks household Jacobians']:.1f}, G {t_g:.1f} / "
          f"{meta['seconds']['ks G']:.1f}, nonlinear {t_nl:.1f} / "
          f"{meta['seconds']['ks nonlinear']:.1f}")

    print("2. One-asset HANK (hank.ipynb), 7 income states by 500 asset points")
    hh, hank_ss, hank_model = hank()
    calibration = {"eis": 0.5, "frisch": 0.5, "rho_e": 0.966, "sd_e": 0.5, "nE": 7,
                   "amin": 0.0, "amax": 150, "nA": 500, "Y": 1.0, "Z": 1.0, "pi": 0.0,
                   "mu": 1.2, "kappa": 0.1, "rstar": 0.005, "phi": 1.5, "B": 5.6}
    t0 = time.perf_counter()
    ss0 = hank_ss.steady_state(calibration, {"beta": 0.986, "vphi": 0.8},
                               {"asset_mkt": 0, "labor_mkt": 0}, solver="hybr")
    ss = hank_model.steady_state(ss0)
    t_ss = time.perf_counter() - t0
    compare_grids(ss, ref, "hank", ["a_grid", "e_grid", "pi_e", "Pi"])
    compare_steady_state(ss, ref, "hank", ["beta", "vphi"], ["a", "c", "n", "Va"],
                         ["A", "C", "N", "NE", "w", "Div", "Tax"])
    print(f"       goods market (Walras), at asset and labour market clearing: dsgelab "
          f"{ss['goods_mkt']:.1e}, sequence-jacobian {float(ref['hank/ss/goods_mkt']):.1e}")
    hh.twosided = False
    inputs = ["r", "w", "Div", "Tax"]
    t0 = time.perf_counter()
    J = hh.jacobian(ss, inputs, T)
    t_j = time.perf_counter() - t0
    compare_jacobians(J, ref, "hank/J", "household Jacobians, forward differences", 1e-7)
    unknowns, targets = ["pi", "w", "Y"], ["nkpc_res", "asset_mkt", "labor_mkt"]
    t0 = time.perf_counter()
    G = hank_model.solve_jacobian(ss, unknowns, targets, ["rstar", "Z"], T=T)
    t_g = time.perf_counter() - t0
    compare_jacobians(G, ref, "hank/G", "general-equilibrium Jacobians G", 1e-7)
    shock = {"rstar": -0.01 / 4 * 0.61 ** np.arange(T)}
    compare_impulses(hank_model.impulse_linear(ss, unknowns, targets, shock), ref,
                     "hank/linear", "linear impulse responses to rstar", 1e-7)
    t0 = time.perf_counter()
    nonlin = hank_model.impulse_nonlinear(ss, unknowns, targets, shock, tol=NONLINEAR_TOL)
    t_nl = time.perf_counter() - t0
    compare_impulses(nonlin, ref, "hank/nonlinear", "nonlinear transition after rstar", 1e-8)
    hh.twosided = True
    compare_jacobians(hh.jacobian(ss, inputs, T), ref, "hank/J2",
                      "household Jacobians, central differences", 1e-7)
    print(f"       seconds, dsgelab / sequence-jacobian: steady state {t_ss:.1f} / "
          f"{meta['seconds']['hank steady state']:.1f}, household Jacobians {t_j:.1f} / "
          f"{meta['seconds']['hank household Jacobians']:.1f}, G {t_g:.1f} / "
          f"{meta['seconds']['hank G']:.1f}, nonlinear {t_nl:.1f} / "
          f"{meta['seconds']['hank nonlinear']:.1f}")
    hank_hh, hank_steady = hh, ss

    print("3. Labor search (labor_search.ipynb), 2 employment states by 5 productivity states by "
          "100 asset points, with the Markov matrix moved by f and s")
    hh, ls = labor_search()
    nA = 100
    calibration = {"beta": 0.95, "r": 0.01, "eis": 0.5, "f": 0.4, "s": 0.1, "w": 1.0, "b": 0.5,
                   "rho_e": 0.95, "sd_e": 0.5, "nE": 5, "amin": 0.0, "amax": 50, "nA": nA}
    t0 = time.perf_counter()
    A = BlockModel([hh]).steady_state(calibration)["A"]
    calibration["B"] = A
    ss = ls.steady_state(calibration)
    t_ss = time.perf_counter() - t0
    internals = ss.internals["household"]

    def flat(key):
        """sequence-jacobian's array on (s, e, a) or (s, e), on the chain of (s, e)."""
        x = np.asarray(ref[key], dtype=float)
        if x.ndim == 3:
            return x.reshape(-1, nA)
        return x.reshape(-1) if key.endswith("/y") else x

    value, name = worst((k, internals[k], flat(f"ls/internals/{k}"))
                        for k in ["a_grid", "e_grid", "Pi_e", "Pi_s", "y"])
    check(f"grids, income and the employment chain (largest: {name})", value, 1e-9)
    check("Markov matrix: Kronecker product of sequence-jacobian's two",
          rel(internals["Pi"], np.kron(ref["ls/internals/Pi_s"], ref["ls/internals/Pi_e"])), 1e-9)
    value, name = worst((k, internals[k], flat(f"ls/internals/{k}")) for k in ["a", "c", "Va"])
    check(f"steady state: policies (largest: {name})", value, 1e-9)
    check("steady state: distribution D, relative to its largest mass",
          rel(internals["D"], flat("ls/internals/D")), 1e-7)
    value, name = worst((k, ss[k], ref[f"ls/ss/{k}"]) for k in ["A", "C", "U", "B"])
    check(f"steady state: aggregates (largest: {name})", value, 1e-9)
    hh.twosided = False
    inputs = ["s", "f", "r", "w"]
    t0 = time.perf_counter()
    J = hh.jacobian(ss, inputs, T)
    t_j = time.perf_counter() - t0
    compare_jacobians(J, ref, "ls/J", "household Jacobians, forward differences", 1e-7)
    J50 = hh.jacobian(ss, ["s"], 50)
    ds = 0.6 ** np.arange(50)
    compare_impulses({O: J50[O]["s"] @ ds for O in ("A", "C", "U")}, ref, "ls/td",
                     "household's linear response to s, the notebook's", 1e-7)
    t0 = time.perf_counter()
    G = ls.solve_jacobian(ss, ["r"], ["asset_mkt"], ["s", "f"], T=T)
    t_g = time.perf_counter() - t0
    compare_jacobians(G, ref, "ls/G", "general-equilibrium Jacobians G", 1e-7)
    shock = {"s": 0.02 * 0.8 ** np.arange(T)}
    compare_impulses(ls.impulse_linear(ss, ["r"], ["asset_mkt"], shock), ref, "ls/linear",
                     "linear impulse responses to s", 1e-7)
    t0 = time.perf_counter()
    nonlin = ls.impulse_nonlinear(ss, ["r"], ["asset_mkt"], shock, tol=NONLINEAR_TOL)
    t_nl = time.perf_counter() - t0
    compare_impulses(nonlin, ref, "ls/nonlinear", "nonlinear transition after s", 1e-8)
    hh.twosided = True
    compare_jacobians(hh.jacobian(ss, inputs, T), ref, "ls/J2",
                      "household Jacobians, central differences", 1e-7)
    ss_initial = ls.steady_state({**calibration, "s": 0.12})
    D0 = ss_initial.internals["household"]["Dbeg"]
    check("distribution of the steady state with s = 0.12, relative to its largest mass",
          rel(D0, flat("ls/initial/Dbeg")), 1e-7)
    back = ls.impulse_nonlinear(ss, ["r"], ["asset_mkt"], {"s": np.zeros(T)},
                                tol=NONLINEAR_TOL, initial={"household": D0})
    compare_impulses(back, ref, "ls/from_initial",
                     "transition from the distribution with s = 0.12", 1e-8)
    print(f"       seconds, dsgelab / sequence-jacobian: steady state {t_ss:.1f} / "
          f"{meta['seconds']['ls steady state']:.1f}, household Jacobians {t_j:.1f} / "
          f"{meta['seconds']['ls household Jacobians']:.1f}, G {t_g:.1f} / "
          f"{meta['seconds']['ls G']:.1f}, nonlinear {t_nl:.1f} / "
          f"{meta['seconds']['ls nonlinear']:.1f}")
    ls_hh, ls_model, ls_steady, ls_D0 = hh, ls, ss, D0

    print("4. Two-asset HANK (two_asset.ipynb), 3 income states by 50 liquid by 70 illiquid "
          "points")
    hh, two_ss, two_model = two_asset()
    t0 = time.perf_counter()
    cali = two_ss.steady_state(calibration_two_asset(), {"beta": 0.976, "chi1": 6.5},
                               {"asset_mkt": 0.0, "B": 1.04}, solver="hybr")
    ss = two_model.steady_state(cali)
    t_ss = time.perf_counter() - t0
    compare_grids(ss, ref, "two", ["a_grid", "b_grid", "k_grid", "e_grid", "Pi", "z_grid",
                                   "Psi1"])
    compare_steady_state(ss, ref, "two", ["beta", "chi1"], ["a", "b", "c", "uce", "chi"],
                         [k for k in TWO_AGGREGATES if abs(float(ref[f"two/ss/{k}"])) > 1e-6])
    internals = ss.internals["hh"]
    value, name = worst((k, internals[k], ref[f"two/internals/{k}"]) for k in ["Va", "Vb"])
    print(f"       marginal values Va and Vb, where each package's backward iteration stopped: "
          f"{value:.1e} ({name})")
    print(f"       goods market (Walras), at asset market clearing: dsgelab "
          f"{ss['goods_mkt']:.1e}, sequence-jacobian {float(ref['two/ss/goods_mkt']):.1e}")
    hh.twosided = False
    inputs = ["N", "ra", "rb", "tax", "w"]
    t0 = time.perf_counter()
    J = hh.jacobian(ss, inputs, T)
    t_j = time.perf_counter() - t0
    compare_jacobians(J, ref, "two/J", "household Jacobians, forward differences", 1e-7)
    unknowns, targets = ["r", "w", "Y"], ["asset_mkt", "fisher", "wnkpc"]
    t0 = time.perf_counter()
    G = two_model.solve_jacobian(ss, unknowns, targets, ["rstar", "Z", "G"], T=T)
    t_g = time.perf_counter() - t0
    compare_jacobians(G, ref, "two/G", "general-equilibrium Jacobians G", 2e-2)
    # the linear approximations differ more than the nonlinear paths; these selected entries are
    # differentiated directly through sequence-jacobian's nonlinear transition in run_ssj.py
    direct = ref["two/direct/K_Z"]
    check("G K/Z, first eight entries against direct sequence-jacobian",
          float(np.max(np.abs(G["K"]["Z"][:8, 0] - direct)) / np.max(np.abs(direct))), 2e-3)
    shock = {"rstar": -0.0025 * 0.6 ** np.arange(T)}
    compare_impulses(two_model.impulse_linear(ss, unknowns, targets, shock), ref, "two/linear",
                     "linear impulse responses to rstar", 2e-4)
    t0 = time.perf_counter()
    nonlin = two_model.impulse_nonlinear(ss, unknowns, targets, shock, tol=NONLINEAR_TOL)
    t_nl = time.perf_counter() - t0
    compare_impulses(nonlin, ref, "two/nonlinear", "nonlinear transition after rstar", 2e-8)
    hh.twosided = True
    J2 = hh.jacobian(ss, inputs, T)
    compare_jacobians(J2, ref, "two/J2", "household Jacobians, central differences", 1e-3)
    direct = float(ref["two/direct/A_ra"])
    check("household A/ra at (299, 299) against direct sequence-jacobian",
          abs(J2["A"]["ra"][-1, -1] - direct) / abs(direct), 1e-4)
    print(f"       seconds, dsgelab / sequence-jacobian: steady state {t_ss:.1f} / "
          f"{meta['seconds']['two steady state']:.1f}, household Jacobians {t_j:.1f} / "
          f"{meta['seconds']['two household Jacobians']:.1f}, G {t_g:.1f} / "
          f"{meta['seconds']['two G']:.1f}, nonlinear {t_nl:.1f} / "
          f"{meta['seconds']['two nonlinear']:.1f}")
    two_steady = ss

    print("5. dsgelab against itself")
    hh, ks = krusell_smith()
    ss = ks.steady_state({**calibration_ks(), "beta": float(ref["ks/ss/beta"]),
                          "Z": float(ref["ks/ss/Z"]), "K": float(ref["ks/ss/K"])})
    fake_news_vs_direct(hh, ss, ["r", "w"], "Krusell-Smith household")
    fake_news_vs_direct(hank_hh, hank_steady, ["r", "w"], "HANK household")
    gaps = []
    for size in (1e-2, 1e-3, 1e-4):
        shock = {"Z": size * 0.8 ** np.arange(T)}
        lin = ks.impulse_linear(ss, ["K"], ["asset_mkt"], shock)
        nonlin = ks.impulse_nonlinear(ss, ["K"], ["asset_mkt"], shock, tol=1e-13)
        gaps.append(max(float(np.max(np.abs(nonlin[k] - lin[k]))) for k in ("K", "r", "w", "C")))
    print("       Krusell-Smith, largest gap between the nonlinear and the linear response, for "
          "shocks of 1, 0.1, 0.01 per cent: " + ", ".join(f"{g:.2e}" for g in gaps))
    check_at_least("Krusell-Smith: smallest ratio of the gaps, shock / 10",
                   min(gaps[0] / gaps[1], gaps[1] / gaps[2]), 90)

    # with steps of 1e-6 and 1e-7 the gap is 1.7e-8 and 1.6e-7: rounding sets it below 1e-5, where
    # it is smallest
    fake_news_vs_direct(ls_hh, ls_steady, ["s", "f"], "labor-search household (moving Markov)",
                        steps=(1e-3, 1e-4, 1e-5))
    gaps = []
    for size in (1e-2, 1e-3, 1e-4):
        shock = {"s": size * 0.8 ** np.arange(T)}
        lin = ls_model.impulse_linear(ls_steady, ["r"], ["asset_mkt"], shock)
        nonlin = ls_model.impulse_nonlinear(ls_steady, ["r"], ["asset_mkt"], shock, tol=1e-13)
        gaps.append(max(float(np.max(np.abs(nonlin[k] - lin[k]))) for k in ("r", "C", "U")))
    print("       labor search, largest gap between the nonlinear and the linear response, for "
          "shocks to s of 1, 0.1, 0.01 points: " + ", ".join(f"{g:.2e}" for g in gaps))
    check_at_least("labor search: smallest ratio of the gaps, shock / 10",
                   min(gaps[0] / gaps[1], gaps[1] / gaps[2]), 90)
    # the household's path from the distribution with s = 0.12, at the steady state's prices,
    # against the steady state's transition matrix applied period by period
    internals = ls_steady.internals["household"]
    path = ls_hh.path(ls_steady, {}, T, Dbeg=ls_D0)
    n_a = internals["a_grid"].size
    M = (_Lottery(internals["a"], internals["a_grid"]).matrix()
         @ sp.kron(sp.csr_matrix(internals["Pi"].T), sp.identity(n_a)))
    Dbeg, largest = ls_D0.ravel(), 0.0
    for t in range(T):
        D = (internals["Pi"].T @ Dbeg.reshape(-1, n_a)).ravel()
        for O, o in (("A", "a"), ("C", "c"), ("U", "u")):
            largest = max(largest, abs(path[O][t] - D @ internals[o].ravel()) / abs(ls_steady[O]))
        Dbeg = M @ Dbeg
    check("labor search: path from an initial distribution against forward simulation",
          largest, 1e-12)

    # the two-asset household on 3 by 15 by 20 points, at the prices of the steady state of 4:
    # the direct Jacobian costs a path of 300 backward steps per column and step; a coarser grid
    # makes consumption invalid at some points during the backward iteration
    hh, _, _ = two_asset()
    small = {**two_steady, "nB": 15, "nA": 20, "nK": 15}
    small_ss = BlockModel([hh]).steady_state(small)
    fake_news_vs_direct(hh, small_ss, ["ra", "rb"], "two-asset household, 3 by 15 by 20",
                        check_at=0)

    if failures:
        print(f"\n{len(failures)} checks failed: {failures}")
        sys.exit(1)
    print("\nall checks passed")


def calibration_ks():
    return {"eis": 1, "delta": 0.025, "alpha": 0.11, "rho_e": 0.966, "sd_e": 0.5, "L": 1.0,
            "nE": 7, "nA": 500, "amin": 0, "amax": 200}


if __name__ == "__main__":
    main()
