"""compare_filtered.py: theoretical moments of filtered variables, against Dynare's stoch_simul.

Five models, read by `dsgelab.from_mod`: example1 (correlated shocks), example2 and RBC_baseline,
the models of run_dynare.sh; fs2000 from Dynare's examples; and estimation/unitroot.mod, whose
price level has a unit root and therefore no unfiltered variance. Dynare's steady state of example2 leaves a residual of
1e-6, and the tolerance for that model is raised to 30 times it, as in compare.py. Each is compared with the
Hodrick-Prescott filter (lambda = 1600) and with the band-pass filter that keeps cycles of 6 to
32 periods, Dynare's `hp_filter=1600` and `bandpass_filter=[6 32]`.

What is compared, for every variable the .mod declares: the covariance matrix of the filtered
variables, their autocorrelations and cross-autocorrelations at lags 1 to 5 (Dynare's
oo_.autocorr, `E[y_t y_{t-j}'] / (sd sd')`), and the variance decomposition. dsgelab works in
proportional deviations for a variable with a nonzero steady state, so its covariances are
multiplied by the steady states before the comparison; the correlations need no conversion.

The decomposition is compared where the shocks are uncorrelated. With correlated shocks Dynare
attributes the variance through the Cholesky factor of their covariance, in the order of
declaration, and dsgelab through each shock's own variance; example1 shows how far apart the
two conventions are.

Prerequisites:
    bash fetch.sh
    bash run_filtered.sh      # exports out_filtered/<name>_<filter>.mat
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import scipy.io as sio

import dsgelab as dsg
from dsgelab.dynare import from_mod

HERE = Path(__file__).resolve().parent
MODELS = {"example1": "mods/example1.mod", "example2": "mods/example2.mod",
          "RBC_baseline": "mods/RBC_baseline.mod", "fs2000": "mods/examples/fs2000.mod",
          "unitroot": "estimation/unitroot.mod"}
FILTERS = {"hp": dict(hp_filter=1600), "bp": dict(bandpass=(6, 32))}
TOL = 1e-9


def _names(x) -> list:
    return [str(v).strip() for v in np.atleast_1d(x)]


def _solution(path: Path, ex):
    mf = from_mod(path)
    ys = dict(zip(_names(ex["endo_names"]), np.atleast_1d(ex["steady_state"]).astype(float)))
    start = {} if mf.linear else {v: ys[v] for v in mf.missing_guess}
    ss = mf.steady_state(guess=start)
    sol = dsg.solve(mf.model, mf.params, steady=ss, qz_criterium=1 + 1e-6)
    exo = _names(ex["exo_names"])
    Sigma = np.zeros((mf.model.n_shocks, mf.model.n_shocks))
    order = [mf.model.shocks.index(e) for e in exo]
    Sigma[np.ix_(order, order)] = np.atleast_2d(np.asarray(ex["Sigma_e"], float))
    # the model's residual at Dynare's steady state: where it is large, what Dynare derives from
    # that point inherits it, as in compare.py
    x_dy = mf.model.vector(mf.full_guess(ys))
    zero = np.zeros(mf.model.n_shocks)
    r_dy = float(np.max(np.abs(mf.model.residuals(x_dy, x_dy, zero, mf.params))))
    return mf, sol, Sigma, exo, r_dy


def compare(name: str, filt: str) -> dict:
    ex = sio.loadmat(HERE / "out_filtered" / f"{name}_{filt}.mat", squeeze_me=True)["ex"]
    ex = {k: ex[k].item() if ex[k].shape == () else ex[k] for k in ex.dtype.names}
    mf, sol, Sigma, exo, r_dy = _solution(HERE / MODELS[name], ex)
    names = _names(ex["names"])
    pos = [mf.model.position(v) for v in names]
    scale = sol.coords.scale[pos]
    mom = sol.moments(Sigma, lags=5, **FILTERS[filt])
    V = mom["cov"][np.ix_(pos, pos)] * np.outer(scale, scale)
    V_dy = np.atleast_2d(np.asarray(ex["var"], float))
    d_var = float(np.max(np.abs(V - V_dy)) / np.max(np.abs(np.diag(V_dy))))
    sd = np.sqrt(np.diag(mom["cov"]))[pos]
    d_auto = 0.0
    for j in range(1, 6):
        A = mom["autocov"][j][np.ix_(pos, pos)] / np.outer(sd, sd)
        A_dy = np.atleast_2d(np.asarray(ex["autocorr"][j - 1], float))
        d_auto = max(d_auto, float(np.max(np.abs(A - A_dy))))
    d_dec = None
    if "decomposition" in ex and len(exo) > 1:
        dec = sol.variance_decomposition(Sigma, **FILTERS[filt])
        cols = [mf.model.shocks.index(e) for e in exo]
        D = 100 * dec["share"][np.ix_(pos, cols)]
        D_dy = np.atleast_2d(np.asarray(ex["decomposition"], float))
        d_dec = float(np.max(np.abs(D - D_dy)))
    diagonal = bool(np.allclose(Sigma, np.diag(np.diag(Sigma))))
    return dict(var=d_var, auto=d_auto, dec=d_dec, diagonal=diagonal, n=len(names),
                tol=max(TOL, 30 * r_dy), r_dy=r_dy)


def main() -> int:
    missing = [f"{n}_{f}" for n in MODELS for f in FILTERS
               if not (HERE / "out_filtered" / f"{n}_{f}.mat").exists()]
    if missing:
        sys.exit(f"missing out_filtered/{missing[0]}.mat: run run_filtered.sh")
    print(f"  {'model':13s} {'filter':7s} {'variables':>9s}  {'covariance':>10s}  "
          f"{'autocorr.':>9s}  {'decomp. (pp)':>12s}")
    ok = True
    for name in MODELS:
        for filt in FILTERS:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = compare(name, filt)
            tol = r["tol"]
            good = r["var"] < tol and r["auto"] < tol
            if r["dec"] is None:
                dec = "one shock"
            elif r["diagonal"]:
                dec = f"{r['dec']:.1e}"
                good &= r["dec"] < 100 * tol
            else:
                dec = f"{r['dec']:.2f} *"
            ok &= good
            print(f"  {name:13s} {filt:7s} {r['n']:9d}  {r['var']:10.1e}  {r['auto']:9.1e}  "
                  f"{dec:>12s}  {'OK' if good else 'FAIL'}"
                  + (f"  (limited by Dynare's s.s., residual {r['r_dy']:.0e})" if tol > TOL else ""))
    print("\n  covariance: largest difference / largest variance; autocorrelations at lags 1 to 5,"
          "\n  cross-correlations included: largest difference; decomposition: percentage points."
          "\n  * correlated shocks: Dynare orthogonalises by Cholesky, dsgelab does not"
          " (see the docstring).")
    print("\nevery comparison passed" if ok else "\nSOME COMPARISONS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
