"""run_dolo.py: dolo's time iteration on three of its own models, for compare_time_iteration.py.

Runs in the environment that setup.sh creates (Python 3.11, dolo 0.4.9.20) and writes
validation/out_time_iteration/dolo.json: for each model and interpolation, the grid, the
quadrature or Markov chain, the parameters, the decision rule at the grid points and at the centres
of the grid's cells, the iterations and the time.

  rbc                  rbc.yaml, states z and k on a 20 x 20 grid (the file asks for 100 x 100),
                       the iid shock integrated by Gauss-Hermite with 5 nodes; labour bounded below.
  consumption_savings  consumption_savings_iid.yaml as published: cash on hand on 100 points, the
                       iid income shock on dolo's 5 equiprobable nodes, consumption between 0 and
                       cash on hand (the borrowing limit).
  rbc_mc               rbc_mc.yaml as published: capital on 50 points, productivity a two-state
                       Markov chain, labour and investment bounded on both sides.

Usage:  validation/dolo/.venv/bin/python validation/dolo/run_dolo.py
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
import dolo.numeric.optimize.ncpsolve as _ncp  # noqa: E402
import dolo.numeric.optimize.newton as _newton  # noqa: E402
from dolo import time_iteration, yaml_import  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "out_time_iteration"
TOL = 1e-11
INNER_TOL = 1e-12

# time_iteration solves each step by ncpsolve (newton without bounds) at their default tolerance,
# 1e-8 on the residual, with at most inner_maxit Newton steps, and offers no option for either.
# At that tolerance the step stops short of the fixed point by some 1e-7 in the controls, and the
# iteration can stall before its own tolerance is met, so both are tightened here. time_iteration
# imports the two solvers when it is called, and picks up these versions.
_ncpsolve, _newton_solve = _ncp.ncpsolve, _newton.newton
_ncp.ncpsolve = lambda f, a, b, x, tol=None, **kw: _ncpsolve(f, a, b, x, tol=INNER_TOL, **kw)
_newton.newton = lambda f, x, tol=None, **kw: _newton_solve(f, x, tol=INNER_TOL, **kw)


def load(name: str, replace: tuple[str, str] | None = None):
    text = (HERE / "models" / name).read_text()
    if replace is not None:
        assert replace[0] in text, f"{name}: {replace[0]!r} not found"
        text = text.replace(*replace)
    tmp = OUT / name
    tmp.write_text(text)
    return yaml_import(str(tmp))


def midpoints(nodes: list[np.ndarray]) -> np.ndarray:
    mids = [0.5 * (a[1:] + a[:-1]) for a in nodes]
    mesh = np.meshgrid(*mids, indexing="ij")
    return np.column_stack([m.ravel() for m in mesh])


def run(model, dprocess, interp: str) -> dict:
    t0 = time.perf_counter()
    res = time_iteration(model, dprocess=dprocess, interp_method=interp, tol=TOL, maxit=5000,
                         inner_maxit=50, verbose=False)
    elapsed = time.perf_counter() - t0
    dr = res.dr
    grid = dr.endo_grid
    axes = [np.linspace(lo, hi, n) for lo, hi, n in zip(grid.min, grid.max, grid.n)]
    nodes, mids = grid.nodes, midpoints(axes)
    n_m = dprocess.n_nodes
    return {
        "interpolation": interp,
        "iterations": int(res.iterations),
        "converged": bool(res.x_converged),
        "error": float(res.err),
        "seconds": elapsed,
        "axes": [a.tolist() for a in axes],
        "at_nodes": [np.asarray(dr.eval_is(i, nodes)).tolist() for i in range(n_m)],
        "at_midpoints": [np.asarray(dr.eval_is(i, mids)).tolist() for i in range(n_m)],
    }


def describe(model, dprocess) -> dict:
    out = {
        "states": list(model.symbols["states"]),
        "controls": list(model.symbols["controls"]),
        "exogenous": list(model.symbols["exogenous"]),
        "parameters": dict(zip(model.symbols["parameters"],
                               map(float, model.calibration["parameters"]))),
    }
    if hasattr(dprocess, "transitions"):
        out["markov"] = {"values": np.asarray(dprocess.values).tolist(),
                         "transitions": np.asarray(dprocess.transitions).tolist()}
    else:
        out["quadrature"] = {
            "nodes": [np.asarray(dprocess.inode(0, j)).tolist()
                      for j in range(dprocess.n_inodes(0))],
            "weights": [float(dprocess.iweight(0, j)) for j in range(dprocess.n_inodes(0))]}
    return out


def main() -> None:
    OUT.mkdir(exist_ok=True)
    cases = {}
    m = load("rbc.yaml", ("n: [100, 100]", "n: [20, 20]"))
    dp = m.exogenous.discretize(to="iid", N=5, method="gauss-hermite")
    cases["rbc"] = describe(m, dp) | {"runs": [run(m, dp, k) for k in ("linear", "cubic")]}
    m = load("consumption_savings_iid.yaml")
    dp = m.exogenous.discretize()
    cases["consumption_savings"] = describe(m, dp) | {
        "runs": [run(m, dp, k) for k in ("linear", "cubic")]}
    m = load("rbc_mc.yaml")
    dp = m.exogenous.discretize()
    cases["rbc_mc"] = describe(m, dp) | {"runs": [run(m, dp, k) for k in ("linear", "cubic")]}
    from importlib.metadata import version
    meta = {"dolo": version("dolo"), "interpolation": version("interpolation"),
            "numpy": version("numpy"), "tol": TOL, "inner_tol": INNER_TOL}
    (OUT / "dolo.json").write_text(json.dumps({"meta": meta, "cases": cases}))
    for name, case in cases.items():
        for r in case["runs"]:
            print(f"{name:20s} {r['interpolation']:7s} {r['iterations']:5d} iterations  "
                  f"converged={r['converged']}  {r['seconds']:.2f} s")


if __name__ == "__main__":
    main()
