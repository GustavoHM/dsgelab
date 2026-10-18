"""compare_occbin.py: occasionally binding constraints (OccBin) against Dynare.

The two files come from tests/occbin of Dynare 7.2, read by `dsgelab.from_mod` with their tagged
equations (`bind=`/`relax=`), the `occbin_constraints` block, the surprise shocks and the
`occbin_setup`/`occbin_solver` options.

  borrcon.mod   one borrowing constraint, surprise shocks in periods 10 and 30,
                simul_periods=80, simul_maxit=11, simul_curb_retrench
  dynrbc.mod    two constraints (irreversible and nonnegative investment) with four regimes,
                52 surprise shocks between periods 1 and 169, simul_periods=200,
                simul_curb_retrench, simul_check_ahead_periods=300 (scenario option=1)

In levels and in every period, each endogenous variable of the piecewise path
(oo_.occbin.endo_piecewise) and of the linear one (oo_.occbin.endo_linear) is compared. In each
shock period, so is the regime that agents expect (oo_.occbin.simul.regime_history: regimes and
start dates).

Prerequisites:
    bash fetch.sh
    bash run_occbin.sh        # exports out_occbin/<name>.mat and writes work_occbin/<name>.mod
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from dsgelab.dynare import from_mod

HERE = Path(__file__).resolve().parent
TOL = 1e-12


def _load(name):
    from scipy.io import loadmat
    f = HERE / "out_occbin" / f"{name}.mat"
    if not f.exists():
        sys.exit(f"missing {f}: run bash run_occbin.sh")
    return loadmat(f, simplify_cells=True)["ex"]


def _starts(ind):
    """Dynare's map_regime: the regime of each spell and the (1-based) date it starts."""
    ind = np.asarray(ind, dtype=int)
    regime, start = [int(ind[0])], [1]
    for i in range(1, len(ind)):
        if ind[i] != regime[-1]:
            regime.append(int(ind[i]))
            start.append(i + 1)
    return regime, start


def _as_list(x):
    return [int(v) for v in np.atleast_1d(x)]


def compare(name, constraints_order):
    ex = _load(name)
    names = [str(v).strip() for v in ex["endo_names"]]
    mf = from_mod((HERE / "work_occbin" / f"{name}.mod").read_text())
    r = mf.occbin_simulate()
    T = ex["piecewise"].shape[0]
    err = {}
    for kind in ("piecewise", "linear"):
        ours = getattr(r, kind)
        err[kind] = max(float(np.max(np.abs(ours[v][:T] - ex[kind][:, j])))
                        for j, v in enumerate(names))
    # regimes: Dynare stores one history per period up to the last surprise
    suffix = [""] if len(constraints_order) == 1 else ["1", "2"]
    bad = 0
    n_hist = len(np.atleast_1d(ex["rh_regime" + suffix[0]]))
    for t in range(1, n_hist + 1):
        exp = r.expected_regimes(t)
        for c, sfx in zip(constraints_order, suffix):
            reg_d = _as_list(np.atleast_1d(ex["rh_regime" + sfx])[t - 1])
            st_d = _as_list(np.atleast_1d(ex["rh_regimestart" + sfx])[t - 1])
            reg_o, st_o = _starts(exp[c])
            if (reg_o, st_o) != (reg_d, st_d):
                bad += 1
    ok = err["piecewise"] < TOL and err["linear"] < TOL and bad == 0
    print(f"{name:10s} {T:4d} periods  piecewise {err['piecewise']:.1e}  linear "
          f"{err['linear']:.1e}  regimes {n_hist - bad}/{n_hist}  iterations "
          f"{sum(r.iterations)}  {'OK' if ok else 'DIVERGE'}")
    return ok


def main() -> int:
    ok = compare("borrcon", ["borrcon"])
    ok &= compare("dynrbc", ["IRR", "INEG"])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
