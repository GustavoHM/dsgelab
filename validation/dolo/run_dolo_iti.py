"""run_dolo_iti.py: dolo's improved time iteration on three of its own models, for
compare_improved_time_iteration.py.

Runs in the environment that setup.sh creates (Python 3.11, dolo 0.4.9.20) and writes
validation/out_time_iteration/dolo_iti.json: for each model and interpolation, the starting
policy, dolo's improved_time_iteration (its Newton steps, its final residual, the decision rule at
the grid points and at the centres of the grid's cells) and, for the iteration counts, dolo's
time_iteration from the same start. The models, grids and processes are those of run_dolo.py:

  rbc                  rbc.yaml on a 20 x 20 grid, the iid shock by Gauss-Hermite with 5 nodes.
  consumption_savings  consumption_savings_iid.yaml as published.
  rbc_mc               rbc_mc.yaml as published.

improved_time_iteration starts from the controls of the model's calibration, a constant policy,
on rbc and rbc_mc, and time_iteration from the same policy. On consumption_savings, as published,
improved_time_iteration does not converge from the calibration within its 1000 steps (the
residual stays near 1.5). From min(w, 1) it takes 899 steps with linear interpolation and does
not converge with cubic. So time_iteration runs from min(w, 1), and improved_time_iteration starts
from time_iteration's policy at a tolerance of 1e-3 (warm_iterations steps), from which it takes
four Newton steps. The starts are passed as dr0, an object with the two methods the routines
call. Nothing else is changed: improved_time_iteration stops when the largest residual, in
complementarity form, is below tol. time_iteration gets the tighter inner tolerances that
run_dolo.py gives it, by importing run_dolo.py.

Usage:  validation/dolo/.venv/bin/python validation/dolo/run_dolo_iti.py
"""
from __future__ import annotations

import json
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
import run_dolo  # noqa: E402  (tightens time_iteration's inner tolerances)
from dolo import improved_time_iteration, time_iteration  # noqa: E402

OUT = run_dolo.OUT
TOL = 1e-11


class Start:
    """A starting policy that does not depend on the exogenous state, as the two routines call
    it: ``dr0.eval_is(i, s)`` in improved_time_iteration and ``dr0(m, s)`` in time_iteration."""

    def __init__(self, fn):
        self.fn = fn

    def eval_is(self, i, s):
        return self.fn(np.asarray(s))

    def __call__(self, m, s):
        return self.fn(np.asarray(s))


def run(model, dprocess, interp: str, start, warm: float | None = None) -> dict:
    t0 = time.perf_counter()
    warm_iterations = 0
    first = start
    if warm is not None:
        pre = time_iteration(model, dprocess=dprocess, interp_method=interp, tol=warm,
                             maxit=5000, inner_maxit=50, verbose=False, dr0=start)
        first, warm_iterations = pre.dr, int(pre.iterations)
    res = improved_time_iteration(model, dprocess=dprocess, interp_method=interp, tol=TOL,
                                  dr0=first, verbose=False)
    elapsed = time.perf_counter() - t0
    dr = res.dr
    grid = dr.endo_grid
    axes = [np.linspace(lo, hi, n) for lo, hi, n in zip(grid.min, grid.max, grid.n)]
    nodes, mids = grid.nodes, run_dolo.midpoints(axes)
    n_m = max(dprocess.n_nodes, 1)
    ti = time_iteration(model, dprocess=dprocess, interp_method=interp, tol=TOL, maxit=5000,
                        inner_maxit=50, verbose=False, dr0=start)
    return {
        "interpolation": interp,
        "iterations": int(res.N),
        "converged": bool(res.x_converged),
        "residual": float(res.f_x),
        "seconds": elapsed,
        "warm": warm,
        "warm_iterations": warm_iterations,
        "ti_iterations": int(ti.iterations),
        "axes": [a.tolist() for a in axes],
        "at_nodes": [np.asarray(dr.eval_is(i, nodes)).tolist() for i in range(n_m)],
        "at_midpoints": [np.asarray(dr.eval_is(i, mids)).tolist() for i in range(n_m)],
    }


def main() -> None:
    OUT.mkdir(exist_ok=True)
    cases = {}
    m = run_dolo.load("rbc.yaml", ("n: [100, 100]", "n: [20, 20]"))
    dp = m.exogenous.discretize(to="iid", N=5, method="gauss-hermite")
    x0 = np.asarray(m.calibration["controls"], dtype=float)
    start = Start(lambda s: np.tile(x0, (s.shape[0], 1)))
    cases["rbc"] = run_dolo.describe(m, dp) | {
        "start": "calibration", "controls": x0.tolist(),
        "runs": [run(m, dp, k, start) for k in ("linear", "cubic")]}
    m = run_dolo.load("consumption_savings_iid.yaml")
    dp = m.exogenous.discretize()
    start = Start(lambda s: np.minimum(s[:, :1], 1.0))
    cases["consumption_savings"] = run_dolo.describe(m, dp) | {
        "start": "min(w, 1)", "runs": [run(m, dp, k, start, 1e-3) for k in ("linear", "cubic")]}
    m = run_dolo.load("rbc_mc.yaml")
    dp = m.exogenous.discretize()
    x0 = np.asarray(m.calibration["controls"], dtype=float)
    start = Start(lambda s: np.tile(x0, (s.shape[0], 1)))
    cases["rbc_mc"] = run_dolo.describe(m, dp) | {
        "start": "calibration", "controls": x0.tolist(),
        "runs": [run(m, dp, k, start) for k in ("linear", "cubic")]}
    from importlib.metadata import version
    meta = {"dolo": version("dolo"), "interpolation": version("interpolation"),
            "numpy": version("numpy"), "tol": TOL}
    (OUT / "dolo_iti.json").write_text(json.dumps({"meta": meta, "cases": cases}))
    for name, case in cases.items():
        for r in case["runs"]:
            print(f"{name:20s} {r['interpolation']:7s} {r['warm_iterations']:3d} + "
                  f"{r['iterations']:2d} Newton steps (time iteration {r['ti_iterations']})  residual {r['residual']:.1e}  "
                  f"converged={r['converged']}  {r['seconds']:.2f} s")


if __name__ == "__main__":
    main()
