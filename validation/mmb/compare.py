"""compare.py: central-bank models of the Macroeconomic Model Data Base, read by from_mod.

The thirteen .mod files come from Rep-MMB (fetch.sh) and are read as they are, as
../compare_import.py does for Dynare's own examples, with the same comparison: steady state,
orthogonalised IRFs of every variable to every shock with nonzero variance, 40 periods, and
theoretical standard deviations. Parameters that the .mod leaves undefined, and the starting
point of the steady state where the .mod gives none, come from Dynare, and the output says so.

The steady-state error is relative, and absolute for values below 1. Several of these models
have variables whose steady state is zero (EACZ_GEM03's output gaps, at 7e-13 in dsgelab and 0
in Dynare), and a purely relative error would measure their rounding.

Prerequisites:
    bash fetch.sh
    bash run_dynare.sh     # exports out/<name>.mat
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from compare_import import compare  # noqa: E402


def main() -> int:
    bad = 0
    print(f"{'model':<13}{'vars':>5}{'aux':>5}  {'params':>8} {'s.s.':>8} {'IRFs':>8} "
          f"{'moments':>9}")
    mods = sorted((HERE / "mods").glob("*.mod"))
    if not mods:
        print("no .mod in mods/, run fetch.sh")
        return 1
    for path in mods:
        name = path.stem
        r = compare(name, path, out=HERE / "out", ss_floor=1.0)
        if r.get("missing"):
            print(f"{name:<13} no out/{name}.mat, run run_dynare.sh")
            bad += 1
            continue
        mom = f"{r['mom']:.1e}" if r["mom"] is not None else "n/a"
        note = ""
        if r["r_dy"] > 1e3 * max(r["r_dk"], 1e-16):
            note = f"  (Dynare's s.s. with residual {r['r_dy']:.0e})"
        print(f"{name:<13}{r['n']:>5}{r['aux']:>5}  {r['par']:>8.1e} {r['ss']:>8.1e} "
              f"{r['irf']:>8.1e} {mom:>9}  {'OK' if r['ok'] else 'DIVERGE ' + r['worst']}{note}")
        if r["from_dynare"]:
            print(f"{'':<13}parameters the .mod leaves undefined, taken from Dynare: "
                  f"{', '.join(r['from_dynare'])}")
        if r["guess_from_dynare"]:
            print(f"{'':<13}s.s. starting point taken from Dynare for "
                  f"{r['guess_from_dynare']} variable(s)")
        bad += not r["ok"]
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
