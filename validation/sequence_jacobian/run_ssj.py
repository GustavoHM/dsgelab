"""run_ssj.py: sequence-jacobian on two of its own models, for compare_heterogeneous.py.

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

The model code below is copied from the two notebooks at sequence-jacobian's v1.0.0 tag, which
setup.sh downloads, and the script checks that each piece appears in them verbatim before running
it. The household of the HANK model, hetblocks.hh_labor.hh, is part of the package. Two settings
differ from the notebooks: the household's backward iteration stops at 1e-12 instead of 1e-8 and
its forward iteration at 1e-14 instead of 1e-10, so that the steady state is close enough to its
fixed point for the comparison, and the nonlinear transitions stop at 1e-11 instead of 1e-8.

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
from sequence_jacobian import het, simple, create_model
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


def notebook_code(name: str) -> str:
    cells = json.loads((HERE / "notebooks" / name).read_text())["cells"]
    return "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")


def load(name: str, pieces: list[str]) -> dict:
    """Runs the pieces, after checking that each one is in the notebook as published."""
    code = notebook_code(name)
    for piece in pieces:
        assert piece in code, f"{name}: a piece of the model is not in the notebook:\n{piece}"
    # sequence-jacobian reads the names of the outputs from the source of each function, so the
    # pieces are written to a file and imported from it
    path = OUT / ("model_" + name.replace(".ipynb", ".py"))
    path.write_text(PREAMBLE + "\n\n".join(pieces) + "\n")
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
