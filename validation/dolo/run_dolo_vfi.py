"""run_dolo_vfi.py: dolo's value function iteration on its rbc_mc model, for compare_value_iteration.py.

Runs in the environment that setup.sh creates (Python 3.11, dolo 0.4.9.20) and writes
validation/out_value_iteration/dolo.json: the grid, the Markov chain, the parameters, and the value
function and the decision rule at the grid points and at the centres of the grid's cells.

  rbc_mc   rbc_mc.yaml as published: capital on 50 points, productivity a two-state Markov chain,
           labour and investment chosen within their bounds to maximize the felicity plus the
           discounted value, which dolo interpolates by a cubic spline.

dolo 0.4.9.20's value_iteration needs four changes to run on this model, all made here without
editing dolo:

- It reads the bounds as model.functions["controls_lb"] and ["controls_ub"], which this version of
  dolo names "arbitrage_lb" and "arbitrage_ub"; the two are aliased.
- It starts from the calibrated controls, the same at every grid point. At low capital they leave
  consumption negative, the evaluation of that policy diverges and the run ends in NaN. It starts
  here from dolo's own time-iteration solution of the same model.
- It maximizes by scipy's L-BFGS-B at its default tolerances, which stop some 1e-5 short of the
  maximum. They are tightened.
- Each policy evaluation stops when the value moves by less than 1e-8, so the iteration cannot
  meet a tolerance below that. The evaluation's tolerance is tightened.

Usage:  validation/dolo/.venv/bin/python validation/dolo/run_dolo_vfi.py
"""
from __future__ import annotations

import functools
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.optimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_dolo import TOL, describe, load, midpoints  # noqa: E402  (also tightens ncpsolve)

import dolo.algos.value_iteration  # noqa: E402,F401
from dolo import time_iteration  # noqa: E402

VI = sys.modules["dolo.algos.value_iteration"]
OUT = HERE.parent / "out_value_iteration"
VFI_TOL = 1e-10
EVAL_TOL = 1e-12

_minimize = scipy.optimize.minimize


def _tight_minimize(fun, x0, bounds=None, **kw):
    return _minimize(fun, x0, bounds=bounds, method="L-BFGS-B",
                     options={"ftol": 1e-15, "gtol": 1e-11, "maxiter": 2000})


def main() -> None:
    OUT.mkdir(exist_ok=True)
    (HERE.parent / "out_time_iteration").mkdir(exist_ok=True)   # load() writes there
    m = load("rbc_mc.yaml")
    m.functions["controls_lb"] = m.functions["arbitrage_lb"]
    m.functions["controls_ub"] = m.functions["arbitrage_ub"]
    dp = m.exogenous.discretize()
    start = time_iteration(m, dprocess=dp, interp_method="cubic", tol=TOL, maxit=5000,
                           inner_maxit=50, verbose=False).dr
    VI.constant_policy = lambda model: start
    VI.evaluate_policy = functools.partial(VI.evaluate_policy, tol=EVAL_TOL, maxit=100000)
    scipy.optimize.minimize = _tight_minimize
    try:
        t0 = time.perf_counter()
        res = VI.value_iteration(m, tol=VFI_TOL, maxit=1000, verbose=False)
        elapsed = time.perf_counter() - t0
    finally:
        scipy.optimize.minimize = _minimize
    dr, drv = res.dr, res.drv
    grid = dr.endo_grid
    axes = [np.linspace(lo, hi, n) for lo, hi, n in zip(grid.min, grid.max, grid.n)]
    nodes, mids = grid.nodes, midpoints(axes)
    n_m = dp.n_nodes
    run = {
        "interpolation": "cubic",
        "iterations": int(res.iterations),
        "converged": bool(res.v_converged and res.x_converged),
        "value_error": float(res.v_err),
        "seconds": elapsed,
        "axes": [a.tolist() for a in axes],
        "controls_at_nodes": [np.asarray(dr.eval_is(i, nodes)).tolist() for i in range(n_m)],
        "controls_at_midpoints": [np.asarray(dr.eval_is(i, mids)).tolist() for i in range(n_m)],
        "value_at_nodes": [np.asarray(drv.eval_is(i, nodes))[:, 0].tolist() for i in range(n_m)],
        "value_at_midpoints": [np.asarray(drv.eval_is(i, mids))[:, 0].tolist()
                               for i in range(n_m)],
    }
    from importlib.metadata import version
    meta = {"dolo": version("dolo"), "interpolation": version("interpolation"),
            "numpy": version("numpy"), "scipy": version("scipy"), "tol": VFI_TOL,
            "eval_tol": EVAL_TOL}
    case = describe(m, dp) | {"bounds": {"n": [0.01, 1.0], "i": [0.0, 1.0]}, "run": run}
    (OUT / "dolo.json").write_text(json.dumps({"meta": meta, "cases": {"rbc_mc": case}}))
    print(f"rbc_mc  cubic  {run['iterations']} iterations  converged={run['converged']}  "
          f"value error {run['value_error']:.1e}  {elapsed:.2f} s")


if __name__ == "__main__":
    main()
