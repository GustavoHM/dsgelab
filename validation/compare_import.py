"""compare_import.py: Dynare's own examples, read by dsgelab.from_mod and solved.

compare.py checks ports of three models written BY HAND. Here no model is rewritten. Dynare's
`.mod` is read as it is by `dsgelab.dynare.from_mod`, and what is compared is what the two
programs produce from the same file. There are nine models. Three are those of compare.py. Five
come from the examples/ folder of Dynare's repository (tag 6.4) and exercise what a hand port
would not: c(+2) and shocks inside the production function (fs2000), `#` local variables and the
steady_state() operator (Gali_2015), the macro processor (agtrend), correlated shocks (example1,
example3) and a medium-scale model of 28 equations with indexation and wage rigidity
(NK_baseline). The last is fs2000 written in levels, with declared trends (trend_var,
var(deflator=)), from tests/trend_var of Dynare 7.2. Dynare and dsgelab each detrend it on their
own, and what is compared is the stationary model that results.

What comes from Dynare, and why:
  * the parameters that the .mod does NOT define. NK_baseline and RBC_baseline compute some of
    them in a MATLAB steady-state file, which a .mod reader does not execute. The ones the .mod
    defines are read by from_mod and CHECKED against Dynare's.
  * the starting point of the steady state, only for the variables whose value the .mod does not
    give in an evaluable form (a MATLAB helper, a missing steady_state_model). The steady state
    itself is solved by dsgelab and compared.

What is compared, as in compare.py: steady state, orthogonalised (Cholesky) IRFs of every
variable to every shock with nonzero variance, 40 periods, and theoretical standard deviations.
No series is shifted, because from_mod keeps Dynare's timing.

Prerequisites:
    bash fetch.sh
    bash run_examples.sh     # exports out_import/<name>.mat
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

import dsgelab as dsg
from dsgelab.dynare import from_mod

HERE = Path(__file__).resolve().parent
H = 40
TOL = 1e-6


def _load(name, out=HERE / "out_import"):
    from scipy.io import loadmat
    f = Path(out) / f"{name}.mat"
    if not f.exists():
        return None
    ex = loadmat(f, squeeze_me=True, struct_as_record=False)["ex"]
    for k in ("endo_names", "exo_names", "param_names"):
        setattr(ex, k, [str(x).strip() for x in np.atleast_1d(getattr(ex, k))])
    return ex


def _mods():
    out = {}
    for f in sorted((HERE / "mods").glob("*.mod")) + sorted((HERE / "mods" / "examples").glob("*.mod")):
        out[f.stem] = f
    return out


def compare(name: str, path: Path, out: Path = HERE / "out_import",
            ss_floor: float = 1e-8) -> dict:
    """Compare one .mod with the Dynare run exported to ``out/<name>.mat``."""
    ex = _load(name, out)
    if ex is None:
        return dict(name=name, missing=True)
    mf = from_mod(path)
    model = mf.model

    # --- parameters: the .mod's are checked, the missing ones come from Dynare
    dy_par = dict(zip(ex.param_names, np.atleast_1d(ex.params).astype(float)))
    params = dict(mf.params)
    d_par, from_dynare = 0.0, []
    for p, v in dy_par.items():
        if p not in params:
            continue
        if math.isnan(params[p]):
            params[p] = v
            from_dynare.append(p)
        else:
            d_par = max(d_par, abs(params[p] - v) / max(abs(v), 1e-12))

    # --- steady state
    ys = dict(zip(ex.endo_names, np.atleast_1d(ex.ys).astype(float)))
    # a linear model without initval starts from zero, as Dynare itself does
    start = {} if mf.linear else {v: ys[v] for v in mf.missing_guess}
    ss = mf.steady_state(guess=start, params=params)
    # relative error, absolute below ss_floor. mmb/compare.py passes 1: a variable whose steady
    # state is zero (an output gap, a debt ratio) is otherwise compared on its rounding error
    d_ss = max(abs(ss.values[v] - ys[v]) / max(abs(ys[v]), ss_floor) for v in mf.endogenous)
    zero = np.zeros(model.n_shocks)
    x_dy = model.vector(mf.full_guess(ys))
    r_dy = float(np.max(np.abs(model.residuals(x_dy, x_dy, zero, params))))
    r_dk = float(np.max(np.abs(model.residuals(ss.x, ss.x, zero, params))))

    # --- orthogonalised IRFs and moments
    sol = dsg.solve(model, params, steady=ss, qz_criterium=1 + 1e-6)   # Dynare's default
    cov = np.atleast_2d(np.asarray(ex.sigma_e, float))
    live = [i for i in range(len(ex.exo_names)) if cov[i, i] > 0]
    L = np.zeros_like(cov)
    if live:
        L[np.ix_(live, live)] = np.linalg.cholesky(cov[np.ix_(live, live)])
    order = [model.shocks.index(e) for e in ex.exo_names]
    scale = sol.coords.scale
    curves = {}
    for j in live:
        imp = np.zeros(model.n_shocks)
        imp[order] = L[:, j]
        dev = sol.irf_impulse(imp, periods=H, percent=False)
        for v in mf.endogenous:
            attr = f"{v}_{ex.exo_names[j]}"
            dy = (np.atleast_1d(getattr(ex.irfs, attr)).astype(float)[:H]
                  if hasattr(ex.irfs, attr) else np.zeros(H))
            curves[(v, ex.exo_names[j])] = (np.asarray(dev[v])[:H] * scale[model.position(v)], dy)
    ruler = {}
    for (v, _), (a, b) in curves.items():
        ruler[v] = max(ruler.get(v, 0.0), float(np.max(np.abs(b))), float(np.max(np.abs(a))))
    # a variable that does not move at first order (price dispersion, say) responds with rounding
    # error, and is left out when its largest response is below 1e-9 of the model's largest
    quiet = 1e-9 * max(ruler.values(), default=0.0)
    d_irf, worst = 0.0, ""
    for (v, e), (a, b) in curves.items():
        if ruler[v] < max(1e-12, quiet):
            continue
        d = float(np.max(np.abs(a - b))) / ruler[v]
        if d > d_irf:
            d_irf, worst = d, f"{v} <- {e}"

    d_mom = None
    if hasattr(ex, "variance"):
        V = np.atleast_2d(np.asarray(ex.variance, float))
        dy_sd = dict(zip(ex.endo_names, np.sqrt(np.maximum(np.diag(V), 0.0))))
        full = np.zeros((model.n_shocks, model.n_shocks))
        full[np.ix_(order, order)] = cov
        mom = sol.moments(cov=full)
        target = [v for v in mf.endogenous if dy_sd.get(v, 0.0) > 1e-9]
        d_mom = max(abs(mom["std_by_name"][v] * scale[model.position(v)] - dy_sd[v]) / dy_sd[v]
                    for v in target) if target else 0.0

    # When Dynare's s.s. leaves a residual, what follows from it inherits the error, and the
    # tolerance follows.
    floor = max(TOL, 30.0 * r_dy)
    ss_ok = d_ss < 1e-9 or r_dy > 1e3 * max(r_dk, 1e-16)
    ok = ss_ok and d_irf < floor and (d_mom is None or d_mom < floor) and d_par < 1e-12
    return dict(name=name, ok=ok, par=d_par, from_dynare=from_dynare,
                guess_from_dynare=len(start), ss=d_ss, irf=d_irf, worst=worst, mom=d_mom,
                r_dy=r_dy, r_dk=r_dk, floor=floor, n=len(mf.endogenous),
                aux=len(mf.auxiliary), shocks=len(live))


def main() -> int:
    bad = 0
    print(f"{'model':<14}{'vars':>5}{'aux':>5}  {'params':>8} {'s.s.':>8} {'IRFs':>8} "
          f"{'moments':>9}")
    for name, path in _mods().items():
        r = compare(name, path)
        if r.get("missing"):
            print(f"{name:<14} no out_import/{name}.mat, run run_examples.sh")
            bad += 1
            continue
        mom = f"{r['mom']:.1e}" if r["mom"] is not None else "n/a"
        note = ""
        if r["r_dy"] > 1e3 * max(r["r_dk"], 1e-16):
            note = f"  (Dynare's s.s. with residual {r['r_dy']:.0e})"
        print(f"{name:<14}{r['n']:>5}{r['aux']:>5}  {r['par']:>8.1e} {r['ss']:>8.1e} "
              f"{r['irf']:>8.1e} {mom:>9}  {'OK' if r['ok'] else 'DIVERGE ' + r['worst']}{note}")
        if r["from_dynare"]:
            print(f"{'':<14}parameters the .mod leaves to MATLAB, taken from Dynare: "
                  f"{', '.join(r['from_dynare'])}")
        if r["guess_from_dynare"]:
            print(f"{'':<14}s.s. starting point taken from Dynare for "
                  f"{r['guess_from_dynare']} variable(s)")
        bad += not r["ok"]
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
