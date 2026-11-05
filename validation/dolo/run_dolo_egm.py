"""run_dolo_egm.py: dolo's endogenous grid method on consumption and saving, for compare_egm.py.

Runs in the environment that setup.sh creates (Python 3.11, dolo 0.4.9.20) and writes
validation/out_egm/dolo.json: for each case, the poststate grid, the quadrature or Markov chain,
the parameters, and the policy at the points of a grid of cash on hand.

  iid     consumption_savings_iid.yaml as published, with its expectation, half_transition,
          direct_response_egm and reverse_state blocks: the iid income shock on dolo's 5
          equiprobable nodes, consumption between 0 and cash on hand (the borrowing limit), and
          the poststate on 100 points from 0 to 4.
  markov  the same file with log income an AR(1) with persistence 0.9 and the same innovation,
          which dolo discretizes as a three-state Markov chain.

dolo 0.4.9.20's egm runs on this model, and three things are changed here, without editing dolo:

- It prints the iteration and the indices of the two chain nodes at every step of its inner loop,
  a debugging line left in the published code. The module's print is replaced by one that prints
  nothing.
- It returns the policy resampled on the model's grid of cash on hand and interpolated there by
  cubic splines, which between the grid's points is not the policy of the endogenous grid method.
  The grid is set here to 199 points, the 100 points of the file and the centres between them, and
  the policy is read at those points only, where the resampled rule equals the method's.
- It stops at the default tolerance of 1e-6 on the change of the controls and the endogenous grid,
  which is tightened to 1e-12. The rule it returns is the one of the iteration before the last,
  within that tolerance of the last.

Usage:  validation/dolo/.venv/bin/python validation/dolo/run_dolo_egm.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_dolo import describe  # noqa: E402

import dolo.algos.egm  # noqa: E402,F401
from dolo import yaml_import  # noqa: E402

EGM = sys.modules["dolo.algos.egm"]
OUT = HERE.parent / "out_egm"
TOL = 1e-12
A_GRID = np.linspace(0.0, 4.0, 100)
FINE = ("orders: [100]", "orders: [199]")
MARKOV = ("exogenous: !UNormal\n    sigma: σ",
          "exogenous: !VAR1\n    rho: 0.9\n    Sigma: [[σ^2]]")


def run(model) -> dict:
    grid, dp = model.discretize()
    t0 = time.perf_counter()
    res = EGM.egm(model, a_grid=A_GRID, η_tol=TOL, maxit=5000, grid=grid, dp=dp)
    elapsed = time.perf_counter() - t0
    w = grid["endo"].nodes
    return describe(model, dp) | {
        "a_grid": A_GRID.tolist(),
        "w": w[:, 0].tolist(),
        "controls": [np.asarray(res.dr.eval_is(i, w))[:, 0].tolist() for i in range(dp.n_nodes)],
        "iterations": int(res.iterations),
        "converged": bool(res.a_converged),
        "error": float(res.err),
        "seconds": elapsed,
    }


def load(name: str, *replacements: tuple[str, str]):
    """consumption_savings_iid.yaml with the given replacements, written under ``name``."""
    text = (HERE / "models" / "consumption_savings_iid.yaml").read_text()
    for old, new in replacements:
        assert old in text, f"{old!r} not found"
        text = text.replace(old, new)
    tmp = OUT / name
    tmp.write_text(text)
    return yaml_import(str(tmp))


def main() -> None:
    OUT.mkdir(exist_ok=True)
    EGM.print = lambda *args, **kwargs: None
    cases = {
        "iid": run(load("consumption_savings_iid.yaml", FINE)),
        "markov": run(load("consumption_savings_markov.yaml", FINE, MARKOV)),
    }
    from importlib.metadata import version
    meta = {"dolo": version("dolo"), "interpolation": version("interpolation"),
            "numpy": version("numpy"), "tol": TOL}
    (OUT / "dolo.json").write_text(json.dumps({"meta": meta, "cases": cases}))
    for name, case in cases.items():
        print(f"{name:7s} {case['iterations']:4d} iterations  converged={case['converged']}  "
              f"{case['seconds']:.2f} s")


if __name__ == "__main__":
    main()
