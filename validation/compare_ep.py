"""compare_ep.py: extended path, deterministic and stochastic, against Dynare.

The files come from tests/ep of Dynare 7.2, read by `dsgelab.from_mod`. The shocks are the ones
Dynare used, exported by run_ep.sh.

  burnside.mod      Burnside (1998), 100 periods, the seven variants of the test that do not ask
                    for an order-4 perturbation: EP, stochastic EP of order 1 and 2 (three
                    Gauss-Hermite nodes) on the sparse and on the full tree, and the order-2
                    hybrids, which add the second-order risk correction where the tree stops
                    branching
  linearmodel0.mod  linear model with two shocks, 200 periods: the EP and Dynare's own
                    first-order solution (in a linear model the two coincide)

Every endogenous variable is compared in every period. The hybrids use the exact Hessian
(`from_mod(..., derivatives="both")`). The Newton tolerance is 1e-13, below Dynare's (1e-12,
which run_ep.sh imposes), and Dynare's is the one that limits the agreement.

Prerequisites:
    bash fetch.sh
    bash run_ep.sh            # exports out_ep/<name>.mat and writes work_ep/<name>.mod
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from dsgelab.dynare import from_mod
from dsgelab.extended_path import extended_path

HERE = Path(__file__).resolve().parent
TOL = 1e-10

BURNSIDE = [
    ("ep0", "EP", dict(order=0)),
    ("sep1", "stochastic EP, order 1", dict(order=1)),
    ("sep1h", "  idem, hybrid", dict(order=1, hybrid=True)),
    ("sep2", "stochastic EP, order 2", dict(order=2)),
    ("sep2full", "  idem, full tree", dict(order=2, tree="full")),
    ("sep2h", "  idem, hybrid", dict(order=2, hybrid=True)),
    ("sep2hfull", "  idem, hybrid, full tree", dict(order=2, tree="full", hybrid=True)),
]


def _load(name):
    from scipy.io import loadmat
    f = HERE / "out_ep" / f"{name}.mat"
    if not f.exists():
        sys.exit(f"missing {f}: run bash run_ep.sh")
    return loadmat(f, simplify_cells=True)["ex"]


def _report(label, ours, theirs, names, offset=0):
    err = max(float(np.max(np.abs(ours[v][offset:offset + theirs.shape[0]] - theirs[:, j])))
              for j, v in enumerate(names))
    scale = max(float(np.max(np.abs(theirs))), 1.0)
    ok = err / scale < TOL
    print(f"  {label:34s} {theirs.shape[0]:4d} periods  max deviation {err:.1e} "
          f"(relative {err / scale:.1e})  {'OK' if ok else 'DIVERGE'}")
    return ok


def main() -> int:
    ok = True
    ex = _load("burnside")
    names = [str(v).strip() for v in ex["names"]]
    mf = from_mod((HERE / "work_ep" / "burnside.mod").read_text().split("resid;")[0],
                  derivatives="both")
    ss = mf.steady_state()
    E = np.asarray(ex["e"], dtype=float).reshape(-1, 1)[:100]
    print("burnside.mod")
    for key, label, kw in BURNSIDE:
        r = extended_path(mf.model, ss, mf.params, E, cov=mf.cov, tol=1e-13, **kw)
        ok &= _report(label, r, ex[key], names)

    ex = _load("linearmodel0")
    names = [str(v).strip() for v in ex["names"]]
    mf = from_mod((HERE / "work_ep" / "linearmodel0.mod").read_text().split("stoch_simul")[0])
    r = extended_path(mf.model, mf.steady_state(), mf.params, ex["e"][:200], tol=1e-13)
    print("linearmodel0.mod")
    ok &= _report("EP", r, ex["ep0"], names)
    ok &= _report("Dynare's first-order solution", r, ex["linear"], names, offset=1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
