"""compare_bgp.py: balanced growth, against Dynare's bgp.write and its tests.

The three models come from tests/bgp of Dynare 7.2, read by `dsgelab.from_mod`. They are Solow
with technical progress and a growing population (lags only), a Ramsey model with a deterministic
trend (lags and leads) and a New Keynesian model in levels (leads only).

Part 1, the problem. Dynare writes the function bgpfun, which evaluates the model at two
consecutive periods with x_t = y g^t, and run_bgp.sh evaluates it at five fixed points. dsgelab
evaluates `dsgelab.bgp.bgp_residual` at the same points, with the timing of each. Dynare's level
is that of the lag when the model has lags (and then dsgelab's current level is y g, and that of
the auxiliary LAG1_v is y), and the current one in a model with leads only. The comparison goes
residual by residual.

Part 2, the solution. `ModFile.balanced_growth` from random starting points, as in Dynare's tests,
with their assertions. In Solow, K/Y = s gE gN / (gE gN - 1 + delta), output and capital grow at
gE gN, and the level of efficiency growth is gE (300 starting points). In Ramsey, every growth
factor is 1.02. In the New Keynesian model, levels and factors equal 1.

Prerequisites:
    bash fetch.sh
    bash run_bgp.sh           # exports out_bgp/<name>.mat
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from dsgelab.bgp import bgp_residual
from dsgelab.dynare import from_mod

HERE = Path(__file__).resolve().parent
FILES = {"solow": "solow-1", "ramsey": "ramsey-1", "nk": "nk-1"}


def _modfile(name):
    text = (HERE / "mods" / "bgp" / f"{FILES[name]}.mod").read_text()
    cut = min(i for i in (text.find("if ~isoctave"), text.find("verbatim;")) if i >= 0)
    return from_mod(text[:cut])


def residuals_match(name) -> bool:
    from scipy.io import loadmat
    f = HERE / "out_bgp" / f"{name}.mat"
    if not f.exists():
        sys.exit(f"missing {f}: run bash run_bgp.sh")
    ex = loadmat(f, simplify_cells=True)["ex"]
    names = [str(v).strip() for v in ex["endo_names"]]
    mf = _modfile(name)
    m = mf.model
    nd = len(names)
    lags = any(kind == "lag" for kind, _ in mf.auxiliary.values())
    err = 0.0
    for j in range(ex["Z"].shape[1]):
        y, g = ex["Z"][:nd, j], ex["Z"][nd:, j]
        X, G = np.zeros(m.n_vars), np.ones(m.n_vars)
        for i, v in enumerate(names):
            X[m.position(v)] = y[i] * g[i] if lags else y[i]
            G[m.position(v)] = g[i]
        for aux, (kind, of) in mf.auxiliary.items():
            if kind == "lag":
                i = names.index(of)
                X[m.position(aux)], G[m.position(aux)] = y[i], g[i]
        r = bgp_residual(m, mf.params, X, G)
        n = m.n_vars
        ours = np.concatenate([r[:nd], r[n:n + nd]])      # Dynare's equations, both dates
        err = max(err, float(np.max(np.abs(ours - ex["F"][:, j]))))
    ok = err < 1e-10
    print(f"  {name:7s} residuals at 5 points against bgpfun: max deviation {err:.1e}  "
          f"{'OK' if ok else 'DIVERGE'}")
    return ok


def solutions() -> bool:
    ok = True
    mf = _modfile("solow")
    p = mf.params
    gE, gN = p["EfficiencyGrowth_ss"], p["PopulationGrowth_ss"]
    theory = p["s"] * gE * gN / (gE * gN - 1 + p["delta"])
    dev = np.zeros(4)
    for seed in range(300):
        r = mf.balanced_growth(seed=seed)
        ok &= r.success and not r.degenerate
        dev = np.maximum(dev, np.abs([r.ratio("PhysicalCapitalStock", "Output") - theory,
                                      r.growth["Output"] - gE * gN,
                                      r.growth["PhysicalCapitalStock"] - gE * gN,
                                      r.levels["EfficiencyGrowth"] - gE]))
    good = bool(np.all(dev < 1e-8))
    print(f"  solow   300 starts: K/Y {dev[0]:.1e}, g_Y {dev[1]:.1e}, g_K {dev[2]:.1e}, "
          f"gE {dev[3]:.1e}  {'OK' if good else 'DIVERGE'}")
    ok &= good
    mf = _modfile("ramsey")
    r = mf.balanced_growth(seed=1)
    d = max(abs(r.growth[v] - 1.02) for v in mf.endogenous)
    print(f"  ramsey  growth factors against 1.02: {d:.1e}  {'OK' if d < 1e-6 else 'DIVERGE'}")
    ok &= d < 1e-6
    mf = _modfile("nk")
    r = mf.balanced_growth(seed=2)
    d = max(max(abs(r.levels[v] - 1), abs(r.growth[v] - 1)) for v in mf.endogenous)
    print(f"  nk      levels and factors against 1: {d:.1e}  {'OK' if d < 1e-9 else 'DIVERGE'}")
    return ok and d < 1e-9


def main() -> int:
    print("the problem (bgp.write)")
    ok = all([residuals_match(n) for n in FILES])
    print("the solution (assertions of Dynare's tests)")
    ok &= solutions()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
