"""compare_plots.py: the densities that posterior plots draw, against Dynare.

The plots draw what the package computes, except for two densities, which are compared here:

  1. prior_density against Dynare's draw_prior_density on the nine priors of fs2000.mod (beta,
     normal and inverse gamma), grid and density;
  2. posterior_density against the kernel density of Dynare's posterior_moments, on two sets of
     draws made in Octave: 6000 or so with repeated values, as a Metropolis chain has after
     rejections (Dynare corrects the bandwidth for them), and 5000 from a lognormal.

Prerequisites:
    bash fetch.sh
    bash run_plots.sh
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from dsgelab.dynare import from_mod
from dsgelab.plots import posterior_density, prior_density

HERE = Path(__file__).resolve().parent


def main() -> bool:
    from scipy.io import loadmat
    f = HERE / "out_plots" / "densities.mat"
    if not f.exists():
        print("run run_plots.sh")
        return False
    ex = loadmat(f, squeeze_me=True, struct_as_record=False)["ex"]
    names = [str(n).strip() for n in np.atleast_1d(ex.names)]
    mf = from_mod(HERE / "mods" / "examples" / "fs2000.mod")
    priors = {e["name"]: e["prior"] for e in mf.estimated if e["prior"] is not None}
    d_x = d_f = 0.0
    shapes = set()
    for i, n in enumerate(names):
        x, fx = prior_density(priors[n])
        shapes.add(priors[n].shape)
        dx, df = np.asarray(ex.prior_x)[:, i], np.asarray(ex.prior_f)[:, i]
        same_nan = np.array_equal(np.isnan(fx), np.isnan(df))
        ok = ~np.isnan(df)
        d_x = max(d_x, float(np.max(np.abs(x - dx) / np.maximum(1.0, np.abs(dx)))))
        d_f = max(d_f, float(np.max(np.abs(fx[ok] - df[ok]) / np.maximum(1.0, np.abs(df[ok])))),
                  0.0 if same_nan else np.inf)
    print(f"prior densities of fs2000 ({len(names)} priors: {', '.join(sorted(shapes))}):"
          f" grid {d_x:.1e}, density {d_f:.1e} (relative)")
    worst = 0.0
    for k in (1, 2):
        draws = np.asarray(getattr(ex, f"draws{k}"), dtype=float)
        D = np.asarray(getattr(ex, f"density{k}"))
        x, fx = posterior_density(draws)
        dk = max(float(np.max(np.abs(x - D[:, 0]))), float(np.max(np.abs(fx - D[:, 1]))) / D[:, 1].max())
        repeats = draws.size - np.unique(draws).size
        print(f"posterior density, {draws.size} draws ({repeats} repeated): {dk:.1e}"
              f" (grid and density, relative to its peak)")
        worst = max(worst, dk)
    return d_x < 1e-12 and d_f < 1e-10 and worst < 1e-12


if __name__ == "__main__":
    ok = main()
    print("\n" + ("OK" if ok else "DIVERGE"))
    sys.exit(0 if ok else 1)
