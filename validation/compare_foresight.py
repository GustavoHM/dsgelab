"""compare_foresight.py: perfect foresight, with and without expectation errors, against Dynare.

The five files come from tests/deterministic_simulations of Dynare 7.2, read by
`dsgelab.from_mod`.

  rbc_det1.mod   initial condition away from the steady state (histval)
  rbc_det2.mod   a shock in the first period
  rbc_det3.mod   ANTICIPATED shocks in periods 4 to 8, `perfect_foresight(..., shocks=)`
  pfwee.mod                      expectation errors: news in periods 1, 2, 3, 6 and 7,
                                 transitory and permanent shocks (pfwee.csv)
  pfwee_constant_sim_length.mod  the same, with every plan on a constant horizon

Timing. Date t of the dsgelab path is Dynare's period t+1, and date T is the terminal period
T+1. Every Dynare endogenous variable is compared in every period from 1 to T+1.

rbc_det1 computes the initial condition in a piece of MATLAB (`CapitalSS/2`, with CapitalSS read
from oo_.steady_state). from_mod does not execute MATLAB, and the condition is written here with
the same meaning.

Prerequisites:
    bash fetch.sh
    bash run_foresight.sh     # exports out_foresight/<name>.mat
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import dsgelab as dsg
from dsgelab.dynare import from_mod, read_expectation_errors_csv
from dsgelab.foresight import perfect_foresight_with_expectation_errors

HERE = Path(__file__).resolve().parent
MODS = HERE / "mods" / "foresight"


def _load(name):
    from scipy.io import loadmat
    f = HERE / "out_foresight" / f"{name}.mat"
    if not f.exists():
        return None
    ex = loadmat(f, squeeze_me=True, struct_as_record=False)["ex"]
    ex.endo_names = [str(x).strip() for x in np.atleast_1d(ex.endo_names)]
    ex.endo_simul = np.atleast_2d(np.asarray(ex.endo_simul, float))
    return ex


def _compare(path, ex, mf, T):
    """Largest relative gap over every Dynare variable and every period 1..T+1."""
    worst, where = 0.0, ""
    for i, v in enumerate(ex.endo_names):
        if v not in mf.endogenous:
            continue
        dy = ex.endo_simul[i, 1:T + 2]
        ours = path[v][:T + 1]
        scale = max(1.0, float(np.max(np.abs(dy))))
        d = float(np.max(np.abs(ours - dy))) / scale
        if d > worst:
            worst, where = d, v
    return worst, where


def deterministic(name: str, T: int, histval=None):
    ex = _load(name)
    if ex is None:
        return None
    mf = from_mod(MODS / f"{name}.mod")
    ss = mf.steady_state()
    shocks, k0 = mf.deterministic(T, ss)
    if histval:
        k0 = mf.initial_states(ss, histval=histval(ss),
                               exo_first={e: k0[mf.model.states.index(f"SHK_{e}")]
                                          for e in mf.exogenous})
    path = dsg.perfect_foresight(mf.model, ss, ss, params=mf.params, T=T, shocks=shocks,
                                initial_states=k0, tol=1e-12)
    return _compare(path, ex, mf, T)


def expectation_errors(name: str, T: int, constant: bool):
    ex = _load(name)
    if ex is None:
        return None
    mf = from_mod(MODS / f"{name}.mod")
    table = read_expectation_errors_csv(MODS / "pfwee.csv")
    news, start = mf.news(table, T)
    path = perfect_foresight_with_expectation_errors(
        mf.model, start, start, params=mf.params, T=T, news=news,
        initial_states=mf.initial_states(start), constant_simulation_length=constant, tol=1e-12)
    TT = path.levels.shape[0] - 1
    if ex.endo_simul.shape[1] != TT + 2:
        return float("inf"), f"different lengths: Dynare {ex.endo_simul.shape[1] - 2}, dsgelab {TT}"
    return _compare(path, ex, mf, TT)


def main() -> int:
    cases = [
        ("rbc_det1", "initial condition (histval)",
         lambda: deterministic("rbc_det1", 20, histval=lambda ss: {("Capital", 0): ss.values["Capital"] / 2})),
        ("rbc_det2", "shock in period 1", lambda: deterministic("rbc_det2", 300)),
        ("rbc_det3", "anticipated shocks, periods 4 to 8", lambda: deterministic("rbc_det3", 300)),
        ("pfwee", "expectation errors", lambda: expectation_errors("pfwee", 7, False)),
        ("pfwee_constant_sim_length", "idem, constant horizon",
         lambda: expectation_errors("pfwee_constant_sim_length", 7, True)),
    ]
    bad = 0
    for name, what, run in cases:
        r = run()
        if r is None:
            print(f"{name:<27} no out_foresight/{name}.mat, run run_foresight.sh")
            bad += 1
            continue
        d, where = r
        ok = d < 1e-8
        print(f"{name:<27} {what:<38} max relative deviation {d:.1e} ({where})  "
              f"{'OK' if ok else 'DIVERGE'}")
        bad += not ok
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
