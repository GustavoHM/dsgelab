"""Pruned second-order means and autocovariances against Dynare 7.2.

Run ``bash run_mom_order2.sh`` first. The comparison covers every endogenous
variable in example1 and example2 and three nonzero autocovariance lags.
"""
from pathlib import Path

import numpy as np
from scipy.io import loadmat

import dsgelab as dsg
import models
from compare_decision_rule import _common_solution, _load, _ordering
from dsgelab.moment_estimation import _second_order_system

HERE = Path(__file__).resolve().parent


def compare(name, builder):
    path = HERE / "out_mom_order2" / f"{name}.mat"
    if not path.exists():
        print(f"{name}: reference missing; run bash run_mom_order2.sh")
        return False
    ex = loadmat(path, squeeze_me=True, struct_as_record=False)["ex"]
    model, params, _, cov, _ = builder()
    reference = _load(name)
    first, endo, residual = _common_solution(model, params, reference)
    second = dsg.solve_second_order(first, cov=cov)
    system = _second_order_system(second, 3)
    _, _, rows, _, _ = _ordering(reference, endo, model)
    scale = first.coords.scale[rows]
    mean = first.coords.center[rows] + scale * system["Ey"][rows]
    Vy = system["Vy"][np.ix_(rows, rows)] * np.outer(scale, scale)
    autocov = [G[np.ix_(rows, rows)] * np.outer(scale, scale)
               for G in system["autocov"]]
    errors = [float(np.max(np.abs(mean - np.atleast_1d(ex.E_y)))),
              float(np.max(np.abs(Vy - ex.Var_y)))]
    errors += [float(np.max(np.abs(autocov[j] - ex.Var_yi[:, :, j]))) for j in range(3)]
    passed = max(errors) < 1e-8
    print(f"{name}: steady residual {residual:.2e}, mean {errors[0]:.2e}, "
          f"var {errors[1]:.2e}, lags {errors[2]:.2e}, {errors[3]:.2e}, {errors[4]:.2e}  "
          f"{'OK' if passed else 'FAIL'}")
    return passed


def compare_imported(name):
    path = HERE / "out_mom_order2" / f"{name}.mat"
    ex = loadmat(path, squeeze_me=True, struct_as_record=False)["ex"]
    mf = dsg.from_mod(HERE / "mods" / f"{name}.mod")
    first = dsg.solve(mf.model, mf.params, steady=mf.steady_state(), tol=2e-6)
    second = dsg.solve_second_order(first, cov=mf.cov)
    system = _second_order_system(second, 3)
    endo = [str(x).strip() for x in np.atleast_1d(ex.endo_names)]
    dr = [endo[i] for i in np.atleast_1d(ex.order_var).astype(int) - 1]
    rows = [mf.model.position(v) for v in dr]
    scale = first.coords.scale[rows]
    mean = first.coords.center[rows] + scale * system["Ey"][rows]
    Vy = system["Vy"][np.ix_(rows, rows)] * np.outer(scale, scale)
    autocov = [G[np.ix_(rows, rows)] * np.outer(scale, scale)
               for G in system["autocov"]]
    errors = [float(np.max(np.abs(mean - np.atleast_1d(ex.E_y)))),
              float(np.max(np.abs(Vy - ex.Var_y)))]
    errors += [float(np.max(np.abs(autocov[j] - ex.Var_yi[:, :, j]))) for j in range(3)]
    passed = max(errors) < 5e-5
    print(f"{name} imported: mean {errors[0]:.2e}, var {errors[1]:.2e}, "
          f"lags {errors[2]:.2e}, {errors[3]:.2e}, {errors[4]:.2e}  "
          f"{'OK' if passed else 'FAIL'}")
    return passed


def main():
    results = [compare(name, builder)
               for name, builder in (("example1", models.example1), ("example2", models.example2))]
    results += [compare_imported("example1"), compare_imported("example2")]
    if all(results):
        print("Pruned second-order means and autocovariances agree within the stated tolerances.")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
