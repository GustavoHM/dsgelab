"""First-order GMM/SMM against Dynare's method_of_moments on An--Schorfheide (2007).

Run fetch.sh, run_mom.sh and then this script. The eight cases compare both methods
at the starting point and after optimization, on Dynare's dataset and on a second
simulated dataset, with eight or five estimated parameters. SMM uses Dynare's own
shock series. For c = 1 + 1/simulation_multiple, SMM's covariance has a factor c
and J a factor 1/c. Dynare 7.2 omits c from its standard errors and multiplies J by
c; the comparison multiplies its standard errors by sqrt(c) and divides its J by c^2.
"""
from __future__ import annotations

import math
from pathlib import Path
import warnings

import numpy as np
import scipy.io as sio

import dsgelab as dsg

HERE = Path(__file__).resolve().parent
CASES = tuple(f"anscho_{method}_{suffix}"
              for suffix in ("0", "4", "d1", "e5") for method in ("GMM", "SMM"))


def _flat(struct):
    out = {}
    for field in struct._fieldnames:
        sub = getattr(struct, field)
        for name in sub._fieldnames:
            value = float(getattr(sub, name))
            if field == "parameters":
                out[name] = value
            elif field == "shocks_std":
                out[f"stderr {name}"] = value
            else:
                out["corr " + ", ".join(name.split("_e_", 1)[0:1] + ["e_" + name.split("_e_", 1)[1]])
                    if "_e_" in name else name] = value
    return out


def _relative_error(ours, reference):
    ours, reference = np.asarray(ours, float), np.asarray(reference, float)
    both_nan = np.isnan(ours) & np.isnan(reference)
    if np.any(np.isnan(ours) != np.isnan(reference)):
        return np.inf
    return np.max(np.abs(ours[~both_nan] - reference[~both_nan]) /
                  np.maximum(np.abs(reference[~both_nan]), 1e-12), initial=0.0)


def compare(name: str):
    driver = (HERE / "work_mom" / f"{name}.mod").read_text()
    include = driver.split('@#include "')[1].split('"')[0]
    text = driver.replace(f'@#include "{include}"', (HERE / "work_mom" / include).read_text())
    mf = dsg.from_mod(text[:text.index("method_of_moments(")])
    ex = sio.loadmat(HERE / "out_mom" / f"{name}.mat", squeeze_me=True,
                     struct_as_record=False)["ex"]
    reference = ex.mom
    data_name = "AnScho_MoM_data_1.mat" if name.endswith(("_d1", "_e5")) else "AnScho_MoM_data_2.mat"
    data = sio.loadmat(HERE / "work_mom" / data_name, squeeze_me=True)
    data = {v: np.asarray(data[v], float).ravel()[:int(ex.nobs)] for v in mf.varobs}
    method = "SMM" if "_SMM_" in name else "GMM"
    kwargs = {}
    if method == "SMM":
        kwargs = dict(shocks=np.asarray(ex.shock_series, float), burnin=500)
    key = method.lower()
    dyn_mode = _flat(getattr(reference, f"{key}_mode"))
    dyn_std = _flat(getattr(reference, f"{key}_std_at_mode"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = mf.method_of_moments(data, method, optimize_=not name.endswith("_0"), **kwargs)
        # Optimizers need not select the same local optimum or stopping point. Check
        # all formulas at Dynare's estimate; the well-conditioned SMM case also
        # checks the complete estimation separately below.
        evaluation = result if name.endswith("_0") else mf.method_of_moments(
            data, method, free=dyn_mode, optimize_=False, **kwargs)

    order = [n for n in result.names if n.startswith("stderr ")] + \
            [n for n in result.names if n.startswith("corr ")] + \
            [n for n in result.names if not n.startswith(("stderr ", "corr "))]
    dyn_x = np.array([dyn_mode[n] for n in order])
    dyn_se = np.array([dyn_std[n] for n in order])
    ours_x = np.array([result.values[n] for n in order])

    correction = 1 + 1 / float(ex.simulation_multiple) if method == "SMM" else 1.0
    expected_se = dyn_se * math.sqrt(correction)
    expected_j = float(reference.J_test.j_stat) / correction ** 2
    data_error = _relative_error(evaluation.data_moments, reference.data_moments)
    model_error = _relative_error(evaluation.model_moments, reference.model_moments)
    estimate_error = np.max(np.abs(ours_x - dyn_x))
    # Dynare's final W comes from the previous stage, not from a fresh evaluation
    # at its estimate. Use that exported W to compare Q and J at the same point.
    Sw = np.asarray(reference.weighting_info.Sw, float)
    gap = evaluation.data_moments - evaluation.model_moments
    objective = float(gap @ Sw.T @ Sw @ gap)
    objective_error = abs(objective - float(reference.Q)) / max(abs(float(reference.Q)), 1e-12)
    ours_j = evaluation.nobs * objective / correction
    j_error = abs(ours_j - expected_j) / max(abs(expected_j), 1e-12)

    tolerance = 1e-7 if name.endswith("_0") else 5e-4
    checks = [data_error < 1e-10, model_error < tolerance,
              objective_error < tolerance, j_error < tolerance,
              evaluation.J["dof"] == int(reference.J_test.degrees_freedom)]
    check_se = name.endswith("_0") or name == "anscho_SMM_e5"
    if check_se:
        ours_se = np.array([result.stderr[n] for n in order])
        se_error = _relative_error(ours_se, expected_se)
        checks.append(se_error < tolerance)
    else:
        se_error = np.nan
    if name == "anscho_SMM_e5":
        checks.append(estimate_error < tolerance)
    se_text = f"{se_error:.1e}" if check_se else "skip"
    print(f"  {name:15s} data {data_error:.1e}  model {model_error:.1e}  "
          f"est. {estimate_error:.1e}  Q {objective_error:.1e}  "
          f"s.e. {se_text}  J {j_error:.1e}  {'OK' if all(checks) else 'FAIL'}")
    return [] if all(checks) else [f"{name}: {checks}"]


def main():
    import sys
    cases = sys.argv[1:] or CASES
    failures = []
    print("case            differences at Dynare's estimate (est.: own optimum; checked for SMM_e5)")
    for name in cases:
        try:
            failures.extend(compare(name))
        except Exception as err:
            failures.append(f"{name}: {err}")
            print(f"  {name:15s} ERROR {err}")
    if failures:
        print("\n".join(failures))
        raise SystemExit(1)
    print(f"All {len(cases)} GMM/SMM cases agree with Dynare after the SMM variance correction.")


if __name__ == "__main__":
    main()
