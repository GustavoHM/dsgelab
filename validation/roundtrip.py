"""roundtrip.py: export from dsgelab to Dynare, run it there, and compare back.

The round-trip test is stronger than `compare.py`. There I wrote the port of a `.mod` to dsgelab by
hand, and a timing error of mine showed up, which the test caught. Here it is the EXPORTER that
writes the `.mod`, and what is checked is whether the automatic translation preserves the model:
same steady state, same impulse responses, same moments.

It uses the same three models of `models.py`, which already came from Dynare. The full path is
therefore Dynare -> dsgelab (by hand, checked by `compare.py`) -> Dynare (automatic, checked
here).

    bash fetch.sh && bash run_dynare.sh      # for compare.py
    python roundtrip.py                      # this one
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dsgelab as dsg
import models
from dsgelab.export import to_dynare

HERE = Path(__file__).resolve().parent
OUT = HERE / "out_roundtrip"
H = 40


def _dynare_dir() -> str:
    d = os.environ.get("DYNARE_MATLAB")
    if d:
        return d
    for c in ("/opt/homebrew/opt/dynare/lib/dynare/matlab",
              "/opt/homebrew/lib/dynare/matlab",
              "/usr/local/opt/dynare/lib/dynare/matlab",
              "/usr/lib/dynare/matlab"):
        if Path(c).is_dir():
            return c
    raise RuntimeError("Dynare's matlab/ directory not found; set DYNARE_MATLAB")


def run_dynare(mod: Path) -> Path:
    """Runs a .mod in Octave and returns the .mat with oo_ exported."""
    octave = os.environ.get("OCTAVE", "/opt/homebrew/bin/octave")
    base = mod.stem
    mat = OUT / f"{base}.mat"
    script = f"""
      pkg load statistics datatypes;
      addpath '{_dynare_dir()}';
      dynare {mod.name} noclearall nograph;
      options_.order = 1; options_.irf = {H}; options_.periods = 0;
      options_.hp_filter = 0; options_.nograph = 1; options_.noprint = 1;
      [info, oo_, options_, M_] = stoch_simul(M_, options_, oo_, M_.endo_names);
      ex = struct();
      ex.steady_state = oo_.steady_state; ex.endo_names = M_.endo_names;
      ex.exo_names = M_.exo_names; ex.irfs = oo_.irfs; ex.info = info;
      if isfield(oo_, 'var'), ex.variance = oo_.var; end
      save('-mat7-binary', '{mat.name}', 'ex');
    """
    r = subprocess.run([octave, "--no-gui", "--eval", script], cwd=OUT,
                       capture_output=True, text=True)
    if not mat.exists():
        raise RuntimeError(f"Dynare failed on {mod.name}:\n{r.stdout[-1800:]}\n{r.stderr[-800:]}")
    return mat


def compare_back(key: str) -> dict:
    model, params, guess, cov, _ = models.ALL[key]()
    ss = dsg.steady_state(model, guess, params)
    sol = dsg.solve(model, params, steady=ss)

    mod = OUT / f"{key}_rt.mod"
    to_dynare(model, params, steady=ss, cov=cov, path=mod, irf=H)
    mat = run_dynare(mod)

    from scipy.io import loadmat
    ex = loadmat(mat, squeeze_me=True, struct_as_record=False)["ex"]
    dy_names = [str(x).strip() for x in np.atleast_1d(ex.endo_names)]
    exo = [str(x).strip() for x in np.atleast_1d(ex.exo_names)]
    dy_ss = dict(zip(dy_names, np.atleast_1d(ex.steady_state).astype(float)))

    print(f"\n{'=' * 84}\n{key} — exported by dsgelab and solved by Dynare\n{'=' * 84}")

    common = [v for v in model.names if v in dy_ss]
    d_ss = max(abs(ss[v] - dy_ss[v]) / max(abs(dy_ss[v]), 1e-8) for v in common)
    zero = np.zeros(model.n_shocks)
    x_dy = np.array([dy_ss[v] for v in model.names])
    r_dy = float(np.max(np.abs(model.residuals(x_dy, x_dy, zero, params))))
    print(f"  steady state ({len(common)} variables)     max relative deviation {d_ss:.3e}")
    print(f"     system residual at the s.s. Dynare returned: {r_dy:.2e}")

    # `predetermined_variables` redates the EQUATIONS, not the reported series. Dynare still
    # publishes the `k` DECIDED in t, which is our k of t+1. The shift is the same as in
    # compare.py, and it applies exactly to the states whose row of Q is zero, the ones that
    # receive no innovation directly. With a shift of 1 the measured deviation falls from 1.4e-02
    # to 3.4e-15.
    shift = {model.states[i] for i in range(model.n_states)
             if np.allclose(sol.Q[i, :], 0.0, atol=1e-12)}
    L = np.linalg.cholesky(np.asarray(cov, float))
    curves, escala = {}, {}
    for m, choque in enumerate(model.shocks):
        dev = sol.irf_impulse(L[:, m], periods=H + 1, percent=False)
        for v in model.names:
            attr = f"{v}_{exo[m]}"
            if not hasattr(ex.irfs, attr):
                continue
            dy = np.atleast_1d(getattr(ex.irfs, attr)).astype(float)[:H]
            bruto = np.asarray(dev[v]) * sol.coords.scale[model.position(v)]
            lvl = bruto[1:H + 1] if v in shift else bruto[:H]
            curves[(v, choque)] = (lvl, dy)
            escala[v] = max(escala.get(v, 0.0), float(np.max(np.abs(dy))))
    worst, onde = 0.0, ""
    for (v, choque), (a, b) in curves.items():
        if escala[v] < 1e-12:
            continue
        d = float(np.max(np.abs(a - b))) / escala[v]
        if d > worst:
            worst, onde = d, f"{v} <- {choque}"
    print(f"  IRFs ({len(curves)} series x {H} periods)          max relative deviation "
          f"{worst:.3e}  ({onde})")

    d_mom = None
    if hasattr(ex, "variance"):
        V = np.atleast_2d(np.asarray(ex.variance, float))
        dy_sd = dict(zip(dy_names, np.sqrt(np.maximum(np.diag(V), 0.0))))
        mom = sol.moments(cov=np.asarray(cov, float))
        cols = [v for v in model.names if v in dy_sd and dy_sd[v] > 1e-9]
        d_mom = max(abs(mom["std_by_name"][v] * sol.coords.scale[model.position(v)] - dy_sd[v])
                    / dy_sd[v] for v in cols)
        print(f"  theoretical standard deviations ({len(cols)} variables)  max relative deviation {d_mom:.3e}")

    piso = max(1e-6, 30.0 * r_dy)
    ok = worst < piso and (d_mom is None or d_mom < piso) and (d_ss < 1e-9 or r_dy > 1e-9)
    if piso > 1e-6:
        print(f"     (tolerance raised to {piso:.1e}, which is what the precision of Dynare's s.s. allows)")
    return {"ok": ok, "ss": d_ss, "irf": worst, "mom": d_mom, "r_dy": r_dy}


def main() -> int:
    OUT.mkdir(exist_ok=True)
    print("=" * 84)
    print("ROUND TRIP — dsgelab writes the .mod, Dynare solves it, we compare back")
    print("=" * 84)
    res = {n: compare_back(n) for n in models.ALL}
    print(f"\n{'=' * 84}\nSUMMARY")
    for n, r in res.items():
        print(f"  {n:<14} s.s. {r['ss']:.1e}   IRFs {r['irf']:.1e}   "
              f"moments {('%.1e' % r['mom']) if r['mom'] is not None else '   n/a'}"
              f"   {'OK' if r['ok'] else 'DIVERGE'}")
    print("=" * 84)
    return 0 if all(r["ok"] for r in res.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
