"""compare_forecast_eval.py: out-of-sample forecast evaluation, against Dynare and R.

  1. Recursive estimation of fs2000.mod (Schorfheide, 2000), read by `from_mod` with model, priors
     and data as they are, on the samples ending at observations 176 to 184, with 8-period
     forecasts (Dynare's estimation(nobs=[176:184], mode_compute=4, forecast=8)).
     a. At Dynare's mode of each sample (estimate="fixed" with one row per origin): the forecast
        means and the 90% bands of both observables.
     b. With dsgelab's own posterior mode at each sample (estimate="mode"): the kernel at the mode
        has to be greater or equal to Dynare's, the parameters within 5% of a posterior standard
        deviation, and the forecasts within 5% of their predictive standard deviation.
  2. The Diebold-Mariano test against R's forecast::dm.test on two autocorrelated error series
     drawn in R: 48 cases (horizons 1 to 4, losses |e| and e^2, both variance estimators, the
     three alternatives), statistic and p-value.

Prerequisites:
    bash fetch.sh
    bash run_forecast_eval.sh
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

from dsgelab import diebold_mariano, forecast_evaluation
from dsgelab.dynare import from_mod

HERE = Path(__file__).resolve().parent
OUT = HERE / "out_forecast_eval"


def _names(x):
    return [str(v).strip() for v in np.atleast_1d(x)]


def check_recursive() -> bool:
    from scipy.io import loadmat
    from scipy.stats import norm
    f = OUT / "fs2000_recursive.mat"
    if not f.exists():
        print("fs2000_recursive: run run_forecast_eval.sh")
        return False
    ex = loadmat(f, squeeze_me=True, struct_as_record=False)["ex"]
    varobs = _names(ex.varobs)
    data = {v: np.asarray(ex.data)[:, i] for i, v in enumerate(varobs)}
    mf = from_mod(HERE / "mods" / "examples" / "fs2000.mod")
    pb = mf.estimation_problem(data)
    origins = np.atleast_1d(ex.nobs).astype(int)
    order = [_names(ex.estim_names).index(n) for n in pb.names]
    modes = np.asarray(ex.mode)[order].T                     # (origins, k)
    H = 8

    at = forecast_evaluation(pb, origins, H, estimate="fixed", theta=modes)
    d_mean = d_band = 0.0
    for v in varobs:
        dm = np.asarray(getattr(ex.mean, v)).T
        lo, hi = np.asarray(getattr(ex.lower, v)).T, np.asarray(getattr(ex.upper, v)).T
        d_mean = max(d_mean, float(np.max(np.abs(at.mean[v] - dm))))
        zq = norm.ppf(0.95)
        d_band = max(d_band, float(np.max(np.abs(at.mean[v] - zq * at.std[v] - lo))),
                     float(np.max(np.abs(at.mean[v] + zq * at.std[v] - hi))))
    print("fs2000 — recursive estimation, samples ending at 176 to 184, 8-period forecasts")
    print(f"  at Dynare's modes: forecast means {d_mean:.1e}, 90% bands {d_band:.1e}")
    ok_a = d_mean < 1e-12 and d_band < 1e-7

    own = forecast_evaluation(pb, origins, H, estimate="mode")
    dens = np.asarray(ex.log_density)
    worst_kernel = float(np.min(own.objective - dens))
    worst_par = 0.0
    for o, m in enumerate(own.modes):
        sd = np.array([m.std[nm] for nm in pb.names])
        worst_par = max(worst_par, float(np.max(np.abs(own.theta[o] - modes[o]) / sd)))
    worst_fc = max(float(np.max(np.abs(own.mean[v] - at.mean[v]) / at.std[v])) for v in varobs)
    print(f"  dsgelab's own modes: kernel minus Dynare's, worst {worst_kernel:+.2e}"
          f"   parameters, worst {worst_par:.3f} posterior sd   forecasts, worst"
          f" {worst_fc:.3f} predictive sd")
    rel = own.relative_rmse()
    for v in varobs:
        print(f"     {v}: RMSE relative to the random walk, h=1..8  "
              + " ".join(f"{r:.2f}" for r in rel[v]))
    ok_b = worst_kernel >= -1e-6 and worst_par < 0.05 and worst_fc < 0.05
    return ok_a and ok_b


def check_dm() -> bool:
    fe, ft = OUT / "dm_errors.csv", OUT / "dm_tests.csv"
    if not (fe.exists() and ft.exists()):
        print("dm: run run_forecast_eval.sh")
        return False
    with open(fe) as fh:
        rows = list(csv.DictReader(fh))
    e1 = np.array([float(r["e1"]) for r in rows])
    e2 = np.array([float(r["e2"]) for r in rows])
    with open(ft) as fh:
        tests = list(csv.DictReader(fh))
    d_stat = d_p = 0.0
    same_h = True
    import warnings
    for r in tests:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t = diebold_mariano(e1, e2, int(r["h"]), power=float(r["power"]),
                                alternative=r["alternative"], varestimator=r["varestimator"])
        d_stat = max(d_stat, abs(t.statistic - float(r["statistic"])))
        d_p = max(d_p, abs(t.p_value - float(r["p_value"])))
        same_h &= t.horizon == int(float(r["h_used"]))
    print(f"\nDiebold-Mariano against R's forecast::dm.test, {len(tests)} cases:"
          f" statistic {d_stat:.1e}, p-value {d_p:.1e}, same horizon used {same_h}")
    return d_stat < 1e-12 and d_p < 1e-12 and same_h


if __name__ == "__main__":
    ok = [check_recursive(), check_dm()]
    print("\n" + ("OK" if all(ok) else "DIVERGE"))
    sys.exit(0 if all(ok) else 1)
