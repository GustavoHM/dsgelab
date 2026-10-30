"""compare_mixed.py: mixed-frequency data, against Dynare.

fs2000.mod from Dynare's examples (Schorfheide, 2000), with output growth observed only as its
four-quarter mean, once a year, and inflation every quarter. Dynare reads it through an auxiliary
variable gy_obs_a = (gy_obs + gy_obs(-1) + gy_obs(-2) + gy_obs(-3))/4 and data with NaN in between
(mixed/prepare.py). dsgelab reads the unchanged fs2000.mod and declares the aggregate with
aggregation={"gy_obs": ("mean", 4)}, on the same numbers. Compared: the likelihood at two points,
the smoothed variables at the first, the analytic gradient against finite differences, and the
posterior mode, whose kernel must be at least Dynare's.

Prerequisites:
    bash fetch.sh
    bash run_mixed.sh
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

from dsgelab.derivatives import log_likelihood_gradient
from dsgelab.dynare import from_mod
from dsgelab.estimation import posterior_mode

HERE = Path(__file__).resolve().parent
OUT = HERE / "out_mixed"


def _load(name):
    from scipy.io import loadmat
    f = OUT / f"{name}.mat"
    return None if not f.exists() else loadmat(f, squeeze_me=True, struct_as_record=False)["ex"]


def _names(x):
    return [str(v).strip() for v in np.atleast_1d(x)]


def main() -> bool:
    ex, exm = _load("fs2000_mf_point"), _load("fs2000_mf_mode")
    if ex is None or exm is None:
        print("run run_mixed.sh")
        return False
    varobs = _names(ex.varobs)
    D = np.asarray(ex.data)
    data = {"gp_obs": D[:, varobs.index("gp_obs")], "gy_obs": D[:, varobs.index("gy_obs_a")]}
    mf = from_mod(HERE / "mods" / "examples" / "fs2000.mod")
    pb = mf.estimation_problem(data, aggregation={"gy_obs": ("mean", 4)})
    names = _names(ex.estim_names)
    order = [names.index(n) for n in pb.names]
    observed = int(np.sum(np.isfinite(data["gy_obs"])))

    d_ll = 0.0
    for xk, lk in (("x1", "ll1"), ("x2", "ll2")):
        th = np.atleast_1d(getattr(ex, xk))[order]
        d_ll = max(d_ll, abs(pb.log_likelihood(th) - float(getattr(ex, lk))))
    th = np.atleast_1d(ex.x1)[order]
    sol, _ = pb.solve(th)
    hat = sol.variables(pb.smoother(th).states[:, :sol.model.n_states])
    d_sv = 0.0
    for v in ("gy_obs", "gp_obs", "y", "c"):
        j = sol.model.position(v)
        ours = math.log(sol.coords.center[j]) + hat[v]
        d_sv = max(d_sv, float(np.max(np.abs(ours - np.asarray(getattr(ex.smoothed_vars, v))))))
    sv_a = np.asarray(ex.smoothed_vars.gy_obs_a)
    j = sol.model.position("gy_obs")
    level = math.log(sol.coords.center[j]) + hat["gy_obs"]
    mean4 = np.array([level[max(t - 3, 0):t + 1].mean() if t >= 3 else np.nan
                      for t in range(len(level))])
    d_agg = float(np.nanmax(np.abs(mean4[3:] - sv_a[3:])))

    g = log_likelihood_gradient(pb, th)
    h = 1e-6
    fd = np.array([(pb.log_likelihood(th + h * e) - pb.log_likelihood(th - h * e)) / (2 * h)
                   for e in np.eye(len(th))])
    d_grad = float(np.max(np.abs(g.gradient - fd) / np.maximum(1.0, np.abs(fd))))

    m = posterior_mode(pb)
    dmode = np.atleast_1d(exm.mode)[order]
    rel = max(abs(m.x[i] - dmode[i]) / m.std[n] for i, n in enumerate(pb.names))
    print("fs2000 with output growth observed as its annual mean "
          f"({observed} of {len(data['gy_obs'])} quarters) and inflation every quarter")
    print(f"  likelihood at two points {d_ll:.1e}   smoothed variables {d_sv:.1e}"
          f"   smoothed annual mean (Dynare's gy_obs_a) {d_agg:.1e}")
    print(f"  analytic gradient against finite differences {d_grad:.1e} (relative)")
    print(f"  posterior kernel at the mode: dsgelab {m.log_posterior:.6f}   Dynare "
          f"{float(exm.log_density):.6f}   parameters within {rel:.3f} posterior sd")
    return (d_ll < 1e-8 and d_sv < 1e-10 and d_agg < 1e-10 and d_grad < 1e-5
            and m.log_posterior >= float(exm.log_density) - 1e-6 and rel < 0.05)


if __name__ == "__main__":
    ok = main()
    print("\n" + ("OK" if ok else "DIVERGE"))
    sys.exit(0 if ok else 1)
