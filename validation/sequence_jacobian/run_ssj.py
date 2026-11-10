"""run_ssj.py: sequence-jacobian on four of its own models, for compare_heterogeneous.py.

Runs in the environment that setup.sh creates (Python 3.11, sequence-jacobian 1.0.0) and writes
validation/out_heterogeneous/ssj.npz: for each model, the steady state (aggregates, policies,
distribution, grids), the household block's Jacobians by the fake-news algorithm, the
general-equilibrium Jacobians, and the linear and nonlinear impulse responses of the notebook.

  ks    krusell_smith.ipynb: the household on 7 income states by 500 asset points, the steady
        state solved for beta, Z and K to hit r = 0.01, Y = 1 and asset market clearing; capital
        as the unknown and the asset market as the target; a shock to Z of 0.01 0.8^t.
  hank  hank.ipynb: the one-asset HANK with labour supply on 7 by 500 points, the steady state
        solved for beta and vphi; pi, w and Y as the unknowns and the Phillips curve, the asset
        market and the labour market as the targets; a shock to rstar of -0.0025 0.61^t.
  ls    labor_search.ipynb: the household with unemployment risk on 2 employment states by 5
        productivity states by 100 asset points, whose employment chain is a hetinput of the
        job-finding rate f and the separation rate s; its Jacobians, and its linear response to
        s of 0.6^t, the notebook's. The notebook has no general equilibrium, and the script adds
        the smallest closure (LS_EXTENSION): a fixed supply of bonds equal to the household's
        assets at the steady state, the interest rate as the unknown and the bond market as the
        target, a shock to s of 0.02 0.8^t, and the return to the steady state from the
        distribution of the steady state with s = 0.12 (ss_initial).
  two   two_asset.ipynb: the two-asset HANK of Auclert, Bardoczy, Rognlie and Straub (2021), with
        a liquid and an illiquid asset and a convex cost of adjusting the illiquid one, on 3
        income states by 50 liquid by 70 illiquid points; the steady state solved for beta and
        chi1; r, w and Y as the unknowns and asset market clearing, the Fisher equation and the
        wage Phillips curve as the targets; a shock to rstar of -0.0025 0.6^t.

The model code below is copied from the four notebooks at sequence-jacobian's v1.0.0 tag, which
setup.sh downloads, and the script checks that each piece appears in them verbatim before running
it. The households of the two HANK models, hetblocks.hh_labor.hh and hetblocks.hh_twoasset.hh, and
the aggregate blocks of the two-asset model, examples.two_asset, are part of the package. Some
settings differ from the notebooks: the household's backward iteration stops at 1e-12 instead of
1e-8 and its forward iteration at 1e-14 instead of 1e-10, so that the steady state is close
enough to its fixed point for the comparison, and the nonlinear transitions stop at 1e-11 instead
of 1e-8. In the two-asset model, the steady state is solved by hybr instead of broyden_custom,
and the solved blocks' own transitions stop at 1e-12.

Usage:  validation/sequence_jacobian/.venv/bin/python validation/sequence_jacobian/run_ssj.py
"""
from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "out_heterogeneous"
T = 300
BACKWARD_TOL, FORWARD_TOL = 1e-12, 1e-14
NONLINEAR_TOL = 1e-11

PREAMBLE = """
import numpy as np
from sequence_jacobian import het, simple, solved, combine, create_model
from sequence_jacobian import interpolate, grids, misc, hetblocks
"""

KS_CELLS = [
r'''def household_init(a_grid, e_grid, r, w, eis):
    coh = (1 + r) * a_grid[np.newaxis, :] + w * e_grid[:, np.newaxis]
    Va = (1 + r) * (0.1 * coh) ** (-1 / eis)
    return Va

@het(exogenous='Pi', policy='a', backward='Va', backward_init=household_init)
def household(Va_p, a_grid, e_grid, r, w, beta, eis):
    """Single backward iteration step using endogenous gridpoint method for households with CRRA utility.

    Parameters
    ----------
    Va_p     : array (nE, nA), expected marginal value of assets next period
    a_grid   : array (nA), asset grid
    e_grid   : array (nE), producticity grid
    r        : scalar, ex-post real interest rate
    w        : scalar, wage
    beta     : scalar, discount factor
    eis      : scalar, elasticity of intertemporal substitution

    Returns
    ----------
    Va : array (nE, nA), marginal value of assets today
    a  : array (nE, nA), asset policy today
    c  : array (nE, nA), consumption policy today
    """
    uc_nextgrid = beta * Va_p
    c_nextgrid = uc_nextgrid ** (-eis)
    coh = (1 + r) * a_grid[np.newaxis, :] + w * e_grid[:, np.newaxis]
    a = interpolate.interpolate_y(c_nextgrid + a_grid, coh, a_grid)
    misc.setmin(a, a_grid[0])
    c = coh - a
    Va = (1 + r) * c ** (-1 / eis)
    return Va, a, c''',
r'''def make_grid(rho_e, sd_e, nE, amin, amax, nA):
    e_grid, _, Pi = grids.markov_rouwenhorst(rho=rho_e, sigma=sd_e, N=nE)
    a_grid = grids.agrid(amin=amin, amax=amax, n=nA)
    return e_grid, Pi, a_grid


household_ext = household.add_hetinputs([make_grid])''',
r'''@simple
def firm(K, L, Z, alpha, delta):
    r = alpha * Z * (K(-1) / L) ** (alpha-1) - delta
    w = (1 - alpha) * Z * (K(-1) / L) ** alpha
    Y = Z * K(-1) ** alpha * L ** (1 - alpha)
    return r, w, Y


@simple
def mkt_clearing(K, A, Y, C, delta):
    asset_mkt = A - K
    goods_mkt = Y - C - delta * K
    return asset_mkt, goods_mkt''',
r'''ks = create_model([household_ext, firm, mkt_clearing], name="Krusell-Smith")''',
r'''calibration = {'eis': 1, 'delta': 0.025, 'alpha': 0.11, 'rho_e': 0.966, 'sd_e': 0.5, 'L': 1.0,
               'nE': 7, 'nA': 500, 'amin': 0, 'amax': 200}
unknowns_ss = {'beta': 0.98, 'Z': 0.85, 'K': 3.}
targets_ss = {'r': 0.01, 'Y': 1., 'asset_mkt': 0.}''',
]

HANK_CELLS = [
r'''hh = hetblocks.hh_labor.hh''',
r'''def make_grid(rho_e, sd_e, nE, amin, amax, nA):
    e_grid, pi_e, Pi = grids.markov_rouwenhorst(rho=rho_e, sigma=sd_e, N=nE)
    a_grid = grids.agrid(amin=amin, amax=amax, n=nA)
    return e_grid, pi_e, Pi, a_grid


def transfers(pi_e, Div, Tax, e_grid):
    # hardwired incidence rules are proportional to skill; scale does not matter 
    tax_rule, div_rule = e_grid, e_grid
    div = Div / np.sum(pi_e * div_rule) * div_rule
    tax = Tax / np.sum(pi_e * tax_rule) * tax_rule
    T = div - tax
    return T

def wages(w, e_grid):
    we = w * e_grid
    return we''',
r'''hh1 = hh.add_hetinputs([make_grid, transfers, wages])''',
r'''def labor_supply(n, e_grid):
    ne = e_grid[:, np.newaxis] * n
    return ne''',
r'''hh_ext = hh1.add_hetoutputs([labor_supply])''',
r'''@simple
def firm(Y, w, Z, pi, mu, kappa):
    L = Y / Z
    Div = Y - w * L - mu/(mu-1)/(2*kappa) * (1+pi).apply(np.log)**2 * Y
    return L, Div


@simple
def monetary(pi, rstar, phi):
    r = (1 + rstar(-1) + phi * pi(-1)) / (1 + pi) - 1
    return r


@simple
def fiscal(r, B):
    Tax = r * B
    return Tax


@simple
def mkt_clearing(A, NE, C, L, Y, B, pi, mu, kappa):
    asset_mkt = A - B
    labor_mkt = NE - L
    goods_mkt = Y - C - mu/(mu-1)/(2*kappa) * (1+pi).apply(np.log)**2 * Y
    return asset_mkt, labor_mkt, goods_mkt


@simple
def nkpc_ss(Z, mu):
    w = Z / mu
    return w''',
r'''blocks_ss = [hh_ext, firm, monetary, fiscal, mkt_clearing, nkpc_ss]

hank_ss = create_model(blocks_ss, name="One-Asset HANK SS")''',
r'''calibration = {'eis': 0.5, 'frisch': 0.5, 'rho_e': 0.966, 'sd_e': 0.5, 'nE': 7,
               'amin': 0.0, 'amax': 150, 'nA': 500, 'Y': 1.0, 'Z': 1.0, 'pi': 0.0,
               'mu': 1.2, 'kappa': 0.1, 'rstar': 0.005, 'phi': 1.5, 'B': 5.6}

unknowns_ss = {'beta': 0.986, 'vphi': 0.8}
targets_ss = {'asset_mkt': 0, 'labor_mkt': 0}''',
r'''@simple
def nkpc(pi, w, Z, Y, r, mu, kappa):
    nkpc_res = kappa * (w / Z - 1 / mu) + Y(+1) / Y * (1 + pi(+1)).apply(np.log) / (1 + r(+1))\
               - (1 + pi).apply(np.log)
    return nkpc_res


blocks = [hh_ext, firm, monetary, fiscal, mkt_clearing, nkpc]
hank = create_model(blocks, name="One-Asset HANK")''',
]

LS_CELLS = [
r'''from sequence_jacobian import markov_rouwenhorst, agrid

def make_grid(rho_e, sd_e, nE, amin, amax, nA):
    e_grid, _, Pi_e = markov_rouwenhorst(rho=rho_e, sigma=sd_e, N=nE)
    a_grid = agrid(amin=amin, amax=amax, n=nA)
    return e_grid, Pi_e, a_grid


def search_frictions(f, s):
    Pi_s = np.vstack(([1 - s, s], [f, 1 - f]))
    return Pi_s


def labor_income(e_grid, w, b):
    y = np.vstack((w * e_grid, b * e_grid))
    return y ''',
r'''from sequence_jacobian import het, interpolate_y

def household_init(a_grid, y, r, eis):
    c = np.maximum(1e-8, y[..., np.newaxis] + np.maximum(r, 0.04) * a_grid)
    Va = (1 + r) * (c ** (-1 / eis))
    return Va

@het(exogenous=['Pi_s', 'Pi_e'], policy='a', backward='Va', backward_init=household_init)
def household(Va_p, a_grid, y, r, beta, eis):
    c_nextgrid = (beta * Va_p) ** (-eis)
    coh = (1 + r) * a_grid + y[..., np.newaxis]
    a = interpolate_y(c_nextgrid + a_grid, coh, a_grid)  # (x, xq, y)
    a = np.maximum(a, a_grid[0])
    c = coh - a
    uc = c ** (-1 / eis)
    Va = (1 + r) * uc

    return Va, a, c''',
r'''def unemployment(c):
    u = np.zeros_like(c)
    u[1, ...] = 1.0
    return u''',
r'''hh = household.add_hetinputs([make_grid, search_frictions, labor_income])
hh = hh.add_hetoutputs([unemployment])''',
r'''calibration = dict(beta=0.95, r=0.01, eis=0.5, f=0.4, s=0.1, w=1., b=0.5,
                   rho_e=0.95, sd_e=0.5, nE=5, amin=0., amax=50, nA=100)''',
]

# The notebook stops at the household in partial equilibrium. The extension below, which is not in
# it, closes the model with a fixed supply of bonds B, equal to the household's assets at the
# notebook's steady state, and the interest rate as the unknown that clears the bond market. It
# has no lags, so that a different initial steady state moves only the household's distribution.
LS_EXTENSION = r'''
@simple
def bond_market(A, B):
    asset_mkt = A - B
    return asset_mkt


ls = create_model([hh, bond_market], name="Labor search")
'''

TWO_ASSET_CELLS = [
r'''@solved(unknowns={'pi': (-0.1, 0.1)}, targets=['nkpc'], solver="brentq")
def pricing_solved(pi, mc, r, Y, kappap, mup):
    nkpc = kappap * (mc - 1/mup) + Y(+1) / Y * (1 + pi(+1)).apply(np.log) / \
           (1 + r(+1)) - (1 + pi).apply(np.log)
    return nkpc''',
r'''@solved(unknowns={'p': (5, 15)}, targets=['equity'], solver="brentq")
def arbitrage_solved(div, p, r):
    equity = div(+1) + p(+1) - p * (1 + r(+1))
    return equity''',
r'''@simple
def labor(Y, w, K, Z, alpha):
    N = (Y / Z / K(-1) ** alpha) ** (1 / (1 - alpha))
    mc = w * N / (1 - alpha) / Y
    return N, mc


@simple
def investment(Q, K, r, N, mc, Z, delta, epsI, alpha):
    inv = (K / K(-1) - 1) / (delta * epsI) + 1 - Q
    val = alpha * Z(+1) * (N(+1) / K) ** (1 - alpha) * mc(+1) -\
        (K(+1) / K - (1 - delta) + (K(+1) / K - 1) ** 2 / (2 * delta * epsI)) +\
        K(+1) / K * Q(+1) - (1 + r(+1)) * Q
    return inv, val


production = combine([labor, investment])                              # create combined block
production_solved = production.solved(unknowns={'Q': 1., 'K': 10.},    # turn it into solved block
                                      targets=['inv', 'val'],
                                      solver='broyden_custom')''',
r'''def make_grids(bmax, amax, kmax, nB, nA, nK, nZ, rho_z, sigma_z):
    b_grid = grids.agrid(amax=bmax, n=nB)
    a_grid = grids.agrid(amax=amax, n=nA)
    k_grid = grids.agrid(amax=kmax, n=nK)[::-1].copy()
    e_grid, _, Pi = grids.markov_rouwenhorst(rho=rho_z, sigma=sigma_z, N=nZ)
    return b_grid, a_grid, k_grid, e_grid, Pi


def income(e_grid, tax, w, N):
    z_grid = (1 - tax) * w * N * e_grid
    return z_grid

hh = hetblocks.hh_twoasset.hh
hh_ext = hh.add_hetinputs([income, make_grids])''',
r'''import sequence_jacobian.examples.two_asset as m

blocks = [hh_ext, production_solved, pricing_solved, arbitrage_solved,
          m.dividend, m.taylor, m.fiscal, m.share_value,
          m.finance, m.wage, m.union, m.mkt_clearing]

hank = create_model(blocks, name='Two-Asset HANK')''',
r'''blocks_ss = [hh_ext, m.partial_ss, m.union_ss,
             m.dividend, m.taylor, m.fiscal, m.share_value, m.finance, m.mkt_clearing]

hank_ss = create_model(blocks_ss, name='Two-Asset HANK SS')''',
r'''calibration = {'Y': 1., 'N': 1.0, 'K': 10., 'r': 0.0125, 'rstar': 0.0125, 'tot_wealth': 14,
               'delta': 0.02, 'pi': 0., 'kappap': 0.1, 'muw': 1.1, 'Bh': 1.04, 'Bg': 2.8,
               'G': 0.2, 'eis': 0.5, 'frisch': 1., 'chi0': 0.25, 'chi2': 2, 'epsI': 4,
               'omega': 0.005, 'kappaw': 0.1, 'phi': 1.5, 'nZ': 3, 'nB': 50, 'nA': 70, 'nK': 50,
               'bmax': 50, 'amax': 4000, 'kmax': 1, 'rho_z': 0.966, 'sigma_z': 0.92}

unknowns_ss = {'beta': 0.976, 'chi1': 6.5}
targets_ss = {'asset_mkt': 0., 'B': 'Bh'}''',
r'''exogenous = ['rstar', 'Z', 'G']
unknowns = ['r', 'w', 'Y']
targets = ['asset_mkt', 'fisher', 'wnkpc']''',
]


# what the two-asset model saves: aggregates, the household's internals, the outputs of G and the
# variables of the impulse responses
TWO_SS = ["beta", "chi1", "vphi", "mup", "alpha", "Z", "w", "p", "mc", "pi", "Q", "ra", "rb",
          "tax", "div", "I", "psip", "pshare", "A", "B", "C", "UCE", "CHI", "wealth",
          "asset_mkt", "goods_mkt", "wnkpc", "fisher", "nkpc", "equity", "inv", "val"]
TWO_INTERNALS = ["a", "b", "c", "uce", "chi", "Va", "Vb", "D", "Dbeg", "a_grid", "b_grid",
                 "k_grid", "e_grid", "z_grid", "Pi", "Psi1"]
TWO_G = ["r", "w", "Y", "pi", "p", "K", "Q", "N", "mc", "I", "div", "ra", "rb", "tax", "A",
         "B", "C", "CHI", "goods_mkt"]
TWO_IRF = ["r", "w", "Y", "pi", "p", "K", "N", "I", "ra", "rb", "A", "B", "C", "div"]


def notebook_code(name: str) -> str:
    cells = json.loads((HERE / "notebooks" / name).read_text())["cells"]
    return "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")


def _without_trailing_spaces(code: str) -> str:
    return "\n".join(line.rstrip() for line in code.split("\n"))


def load(name: str, pieces: list[str], extension: str = "") -> dict:
    """Runs the pieces, after checking that each one is in the notebook as published, and then
    the extension, which is not."""
    # the notebooks carry trailing spaces on some lines, which an editor removes from this file
    # without changing the model, so the two are compared without them
    code = _without_trailing_spaces(notebook_code(name))
    for piece in pieces:
        assert _without_trailing_spaces(piece) in code, \
            f"{name}: a piece of the model is not in the notebook:\n{piece}"
    # sequence-jacobian reads the names of the outputs from the source of each function, so the
    # pieces are written to a file and imported from it
    path = OUT / ("model_" + name.replace(".ipynb", ".py"))
    path.write_text(PREAMBLE + "\n\n".join(pieces) + "\n" + extension)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return vars(module)


def jacobians(block, ss, inputs, outputs, **options) -> dict:
    J = block.jacobian(ss, inputs=inputs, outputs=outputs, T=T, **options)
    return {f"{o}/{i}": np.asarray(J[o][i].matrix(T)) if hasattr(J[o][i], "matrix")
            else np.asarray(J[o][i]) for o in outputs for i in inputs}


def ge(model, ss, unknowns, targets, inputs, outputs) -> dict:
    G = model.solve_jacobian(ss, unknowns, targets, inputs, T=T)
    return {f"{o}/{i}": np.asarray(G[o][i].matrix(T)) if hasattr(G[o][i], "matrix")
            else np.asarray(G[o][i]) for o in outputs for i in inputs}


def save(out: dict, prefix: str, values: dict) -> None:
    for k, v in values.items():
        out[f"{prefix}/{k}"] = np.asarray(v, dtype=float)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    out: dict = {}
    times: dict = {}

    ns = load("krusell_smith.ipynb", KS_CELLS)
    ks, household = ns["ks"], ns["household_ext"]
    opts = {"household": {"backward_tol": BACKWARD_TOL, "forward_tol": FORWARD_TOL}}
    t0 = time.perf_counter()
    ss = ks.solve_steady_state(ns["calibration"], ns["unknowns_ss"], ns["targets_ss"],
                               solver="hybr", options=opts)
    times["ks steady state"] = time.perf_counter() - t0
    save(out, "ks/ss", {k: ss[k] for k in ["beta", "Z", "K", "r", "w", "Y", "A", "C",
                                            "asset_mkt", "goods_mkt"]})
    save(out, "ks/internals", {k: ss.internals["household"][k]
                               for k in ["a", "c", "Va", "D", "Dbeg", "a_grid", "e_grid", "Pi"]})
    t0 = time.perf_counter()
    save(out, "ks/J", jacobians(household, ss, ["r", "w"], ["A", "C"]))
    times["ks household Jacobians"] = time.perf_counter() - t0
    save(out, "ks/J2", jacobians(household, ss, ["r", "w"], ["A", "C"], twosided=True))
    t0 = time.perf_counter()
    save(out, "ks/G", ge(ks, ss, ["K"], ["asset_mkt"], ["Z"],
                         ["K", "r", "w", "Y", "A", "C", "goods_mkt"]))
    times["ks G"] = time.perf_counter() - t0
    shock = {"Z": 0.01 * 0.8 ** np.arange(T)}
    lin = ks.solve_impulse_linear(ss, ["K"], ["asset_mkt"], shock)
    t0 = time.perf_counter()
    nonlin = ks.solve_impulse_nonlinear(ss, ["K"], ["asset_mkt"], shock, tol=NONLINEAR_TOL,
                                        maxit=50, verbose=False)
    times["ks nonlinear"] = time.perf_counter() - t0
    names = ["K", "r", "w", "Y", "C", "A"]
    save(out, "ks/linear", {k: lin[k] for k in names})
    save(out, "ks/nonlinear", {k: nonlin[k] for k in names})

    ns = load("hank.ipynb", HANK_CELLS)
    hank_ss, hank, hh = ns["hank_ss"], ns["hank"], ns["hh_ext"]
    opts = {"hh": {"backward_tol": BACKWARD_TOL, "forward_tol": FORWARD_TOL}}
    t0 = time.perf_counter()
    ss0 = hank_ss.solve_steady_state(ns["calibration"], ns["unknowns_ss"], ns["targets_ss"],
                                     solver="hybr", options=opts)
    ss = hank.steady_state(ss0, options=opts)
    times["hank steady state"] = time.perf_counter() - t0
    save(out, "hank/ss", {k: ss[k] for k in ["beta", "vphi", "w", "r", "Div", "Tax", "L", "A",
                                              "C", "N", "NE", "asset_mkt", "labor_mkt",
                                              "goods_mkt", "nkpc_res"]})
    save(out, "hank/internals", {k: ss.internals["hh"][k]
                                 for k in ["a", "c", "n", "Va", "D", "Dbeg", "a_grid", "e_grid",
                                           "pi_e", "Pi", "T", "we"]})
    inputs, outputs = ["r", "w", "Div", "Tax"], ["A", "C", "N", "NE"]
    t0 = time.perf_counter()
    save(out, "hank/J", jacobians(hh, ss, inputs, outputs))
    times["hank household Jacobians"] = time.perf_counter() - t0
    save(out, "hank/J2", jacobians(hh, ss, inputs, outputs, twosided=True))
    unknowns, targets = ["pi", "w", "Y"], ["nkpc_res", "asset_mkt", "labor_mkt"]
    t0 = time.perf_counter()
    save(out, "hank/G", ge(hank, ss, unknowns, targets, ["rstar", "Z"],
                           ["pi", "w", "Y", "r", "C", "A", "NE", "Div", "Tax", "goods_mkt"]))
    times["hank G"] = time.perf_counter() - t0
    shock = {"rstar": -0.01 / 4 * 0.61 ** np.arange(T)}
    lin = hank.solve_impulse_linear(ss, unknowns, targets, shock)
    t0 = time.perf_counter()
    nonlin = hank.solve_impulse_nonlinear(ss, unknowns, targets, shock, tol=NONLINEAR_TOL,
                                          maxit=50, verbose=False)
    times["hank nonlinear"] = time.perf_counter() - t0
    names = ["pi", "w", "Y", "r", "C", "A", "Div"]
    save(out, "hank/linear", {k: lin[k] for k in names})
    save(out, "hank/nonlinear", {k: nonlin[k] for k in names})

    ns = load("labor_search.ipynb", LS_CELLS, LS_EXTENSION)
    hh, ls = ns["hh"], ns["ls"]
    opts = {"household": {"backward_tol": BACKWARD_TOL, "forward_tol": FORWARD_TOL}}
    t0 = time.perf_counter()
    ss_hh = hh.steady_state(ns["calibration"], options=opts)
    calibration = {**ns["calibration"], "B": ss_hh["A"]}
    ss = ls.steady_state(calibration, options=opts)
    times["ls steady state"] = time.perf_counter() - t0
    save(out, "ls/ss", {k: ss[k] for k in ["A", "C", "U", "B", "asset_mkt"]})
    save(out, "ls/internals", {k: ss.internals["household"][k]
                               for k in ["a", "c", "Va", "D", "Dbeg", "a_grid", "e_grid", "Pi_e",
                                         "Pi_s", "y"]})
    inputs, outputs = ["s", "f", "r", "w"], ["A", "C", "U"]
    t0 = time.perf_counter()
    save(out, "ls/J", jacobians(hh, ss, inputs, outputs))
    times["ls household Jacobians"] = time.perf_counter() - t0
    save(out, "ls/J2", jacobians(hh, ss, inputs, outputs, twosided=True))
    td = hh.impulse_linear(ss, {"s": 0.6 ** np.arange(50)})
    save(out, "ls/td", {k: td[k] for k in ["C", "U", "A"]})
    t0 = time.perf_counter()
    save(out, "ls/G", ge(ls, ss, ["r"], ["asset_mkt"], ["s", "f"], ["r", "A", "C", "U"]))
    times["ls G"] = time.perf_counter() - t0
    shock = {"s": 0.02 * 0.8 ** np.arange(T)}
    lin = ls.solve_impulse_linear(ss, ["r"], ["asset_mkt"], shock)
    t0 = time.perf_counter()
    nonlin = ls.solve_impulse_nonlinear(ss, ["r"], ["asset_mkt"], shock, tol=NONLINEAR_TOL,
                                        maxit=50, verbose=False)
    times["ls nonlinear"] = time.perf_counter() - t0
    names = ["r", "A", "C", "U"]
    save(out, "ls/linear", {k: lin[k] for k in names})
    save(out, "ls/nonlinear", {k: nonlin[k] for k in names})
    # the return to the steady state from the distribution of a steady state with s = 0.12
    ss_initial = ls.steady_state({**calibration, "s": 0.12}, options=opts)
    save(out, "ls/initial", {"Dbeg": ss_initial.internals["household"]["Dbeg"]})
    back = ls.solve_impulse_nonlinear(ss, ["r"], ["asset_mkt"], {"s": np.zeros(T)},
                                      tol=NONLINEAR_TOL, maxit=50, verbose=False,
                                      ss_initial=ss_initial)
    save(out, "ls/from_initial", {k: back[k] for k in names})

    ns = load("two_asset.ipynb", TWO_ASSET_CELLS)
    hank_ss, hank, hh = ns["hank_ss"], ns["hank"], ns["hh_ext"]
    # SolvedBlock._steady_state passes its options where solve_steady_state takes dissolve, so
    # options cannot reach the household through the solved blocks: they become its defaults
    for model in (hank_ss, hank):
        model["hh"].steady_state_options = {**model["hh"].steady_state_options,
                                            "backward_tol": BACKWARD_TOL,
                                            "forward_tol": FORWARD_TOL,
                                            "backward_maxit": 100_000, "forward_maxit": 1_000_000}
    t0 = time.perf_counter()
    cali = hank_ss.solve_steady_state(ns["calibration"], ns["unknowns_ss"], ns["targets_ss"],
                                      solver="hybr")
    ss = hank.steady_state(cali)
    times["two steady state"] = time.perf_counter() - t0
    save(out, "two/ss", {k: ss[k] for k in TWO_SS})
    save(out, "two/internals", {k: ss.internals["hh"][k] for k in TWO_INTERNALS})
    inputs, outputs = ["N", "ra", "rb", "tax", "w"], ["A", "B", "C", "UCE", "CHI"]
    t0 = time.perf_counter()
    save(out, "two/J", jacobians(hh, ss, inputs, outputs))
    times["two household Jacobians"] = time.perf_counter() - t0
    save(out, "two/J2", jacobians(hh, ss, inputs, outputs, twosided=True))
    unknowns, targets = ns["unknowns"], ns["targets"]
    t0 = time.perf_counter()
    save(out, "two/G", ge(hank, ss, unknowns, targets, ns["exogenous"], TWO_G))
    times["two G"] = time.perf_counter() - t0
    inner = {k: {"verbose": False, "tol": 1e-12} for k in
             ["pricing_solved", "arbitrage_solved", "labor_to_investment_combined_solved"]}
    # selected entries differentiated through the nonlinear transition, independently of the
    # solve_jacobian approximation used for the full matrices
    h = 1e-4
    up, dn = np.zeros(T), np.zeros(T)
    up[0], dn[0] = h, -h
    Yu = hank.solve_impulse_nonlinear(ss, unknowns, targets, {"Z": up}, tol=NONLINEAR_TOL,
                                      maxit=50, verbose=False, options=inner)
    Yd = hank.solve_impulse_nonlinear(ss, unknowns, targets, {"Z": dn}, tol=NONLINEAR_TOL,
                                      maxit=50, verbose=False, options=inner)
    direct = {"K_Z": (Yu["K"][:8] - Yd["K"][:8]) / (2 * h)}
    up, dn = np.zeros(T), np.zeros(T)
    up[-1], dn[-1] = h, -h
    Yu = hh.impulse_nonlinear(ss, {"ra": up}, outputs=["A"])
    Yd = hh.impulse_nonlinear(ss, {"ra": dn}, outputs=["A"])
    direct["A_ra"] = (Yu["A"][-1] - Yd["A"][-1]) / (2 * h)
    save(out, "two/direct", direct)
    shock = {"rstar": -0.0025 * 0.6 ** np.arange(T)}
    lin = hank.solve_impulse_linear(ss, unknowns, targets, shock)
    t0 = time.perf_counter()
    nonlin = hank.solve_impulse_nonlinear(ss, unknowns, targets, shock, tol=NONLINEAR_TOL,
                                          maxit=50, verbose=False, options=inner)
    times["two nonlinear"] = time.perf_counter() - t0
    save(out, "two/linear", {k: lin[k] for k in TWO_IRF})
    save(out, "two/nonlinear", {k: nonlin[k] for k in TWO_IRF})

    np.savez_compressed(OUT / "ssj.npz", **out)
    from importlib.metadata import version
    meta = {"sequence-jacobian": version("sequence-jacobian"), "numpy": version("numpy"),
            "numba": version("numba"), "T": T, "backward_tol": BACKWARD_TOL,
            "forward_tol": FORWARD_TOL, "nonlinear_tol": NONLINEAR_TOL, "seconds": times}
    (OUT / "ssj.json").write_text(json.dumps(meta, indent=1))
    for k, v in times.items():
        print(f"{k:28s} {v:6.2f} s")


if __name__ == "__main__":
    main()
