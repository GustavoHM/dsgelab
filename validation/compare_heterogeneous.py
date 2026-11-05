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
  2. dsgelab against itself. The fake-news Jacobian against the direct one, the nonlinear path
     differentiated by central differences for a shock in each period, on ten of the 300 columns
     and with steps of 1e-5, 1e-6 and 1e-7 in both. And Krusell-Smith's linear impulse response against the
     nonlinear transition, for shocks to productivity of 1, 0.1 and 0.01 per cent: the
     difference must fall as the square of the shock.

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

import dsgelab as dsg
from dsgelab import BlockModel, HetBlock, interpolate_rows, log_grid, simple

HERE = Path(__file__).resolve().parent
OUT = HERE / "out_heterogeneous"
T = 300
BACKWARD_TOL = 1e-12
NONLINEAR_TOL = 1e-11
COLUMNS = [0, 1, 2, 5, 10, 20, 50, 100, 200, 299]
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


def fake_news_vs_direct(block, ss, inputs, label, steps=(1e-5, 1e-6, 1e-7)):
    """The fake-news Jacobian against the direct one, on the columns in COLUMNS.

    Both are finite differences with the same step, of one backward step in the first and of the
    whole path in the second. The policy has kinks, where the borrowing limit starts to bind and
    at every grid point the lottery crosses, so the two differ by the error of the differences,
    which falls with the step.
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
    check(f"{label}: fake news against direct, step {steps[-1]:.0e}", gaps[-1], 1e-7)


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

    print("3. dsgelab against itself")
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

    if failures:
        print(f"\n{len(failures)} checks failed: {failures}")
        sys.exit(1)
    print("\nall checks passed")


def calibration_ks():
    return {"eis": 1, "delta": 0.025, "alpha": 0.11, "rho_e": 0.966, "sd_e": 0.5, "L": 1.0,
            "nE": 7, "nA": 500, "amin": 0, "amax": 200}


if __name__ == "__main__":
    main()
