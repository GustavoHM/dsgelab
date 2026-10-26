"""Third-order pruned IRFs against Dynare 7.2."""
from pathlib import Path

import numpy as np
from scipy.io import loadmat

import dsgelab as dsg
import models
from compare_decision_rule import _common_solution

HERE = Path(__file__).resolve().parent
DRAWS = 5000
TOLERANCE = 1e-3


def compare(name, builder):
    path = HERE / "out_third_irf" / f"{name}.mat"
    if not path.exists():
        print(f"{name}: third-order IRF reference missing; run bash run_third_irf.sh")
        return False
    model, params, _, cov, _ = builder()
    ex = loadmat(path, squeeze_me=True, struct_as_record=False)["ex"]
    first, _, _ = _common_solution(model, params, ex)
    third = dsg.solve_third_order(dsg.solve_second_order(first, cov=cov))
    shocks = [str(x).strip() for x in np.atleast_1d(ex.exo_names)]
    worst_error = 0.0
    worst_case = None
    for shock in shocks:
        own_shock = shock if shock in model.shocks else f"eps_{shock}"
        ours = third.irf(own_shock, size=0.009, periods=40, n_draws=DRAWS, seed=0)
        for variable in model.names:
            reference = np.asarray(getattr(ex.irfs, f"{variable}_{shock}"), float)
            converted = ours[variable] * first.coords.scale[model.position(variable)]
            error = float(np.max(np.abs(converted - reference)))
            if error > worst_error:
                worst_error = error
                worst_case = f"{variable}_{shock}"
    passed = worst_error < TOLERANCE
    print(f"{name}: 40 periods, all variables and shocks, max abs {worst_error:.2e} "
          f"at {worst_case}  {'OK' if passed else 'FAIL'}")
    return passed


def main():
    results = [compare("example1", models.example1), compare("example2", models.example2)]
    if all(results):
        print("Third-order pruned IRFs agree with Dynare.")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
