"""Compare selected DSGE_mod models with Dynare through dsgelab.from_mod.

The original files are read without a hand-written Python port. The comparison covers parameters,
steady state, orthogonalised IRFs of every variable to every live shock (40 periods), and
unfiltered theoretical standard deviations. Missing parameters and initial steady-state guesses
come from Dynare and are reported explicitly. This validates the model solution; it does not by
itself reproduce every table, figure, estimation, or higher-order exercise in the source papers.

Prerequisites:
    bash fetch.sh
    bash run_dynare.sh
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from compare_import import compare  # noqa: E402


def main() -> int:
    bad = 0
    print(f"{'model':<24}{'vars':>5}{'aux':>5}  {'params':>8} {'s.s.':>8} {'IRFs':>8} "
          f"{'moments':>9}")
    mods = sorted((HERE / "mods").glob("*.mod"))
    if not mods:
        print("no .mod in mods/, run fetch.sh")
        return 1
    for path in mods:
        name = path.stem
        # Use the same source-specific working copies as run_dynare.sh where the lightweight
        # statement reader needs them: alternative shock blocks (Aguiar–Gopinath) and title
        # assignments without semicolons (Hansen). Equations and calibration remain untouched.
        if name in {"Aguiar_Gopinath_2007", "Hansen_1985"}:
            path = HERE / "prepared" / path.name
        r = compare(name, path, out=HERE / "out", ss_floor=1.0)
        if r.get("missing"):
            print(f"{name:<24} no out/{name}.mat, run run_dynare.sh")
            bad += 1
            continue
        mom = f"{r['mom']:.1e}" if r["mom"] is not None else "n/a"
        note = ""
        if r["r_dy"] > 1e3 * max(r["r_dk"], 1e-16):
            note = f"  (Dynare's s.s. with residual {r['r_dy']:.0e})"
        print(f"{name:<24}{r['n']:>5}{r['aux']:>5}  {r['par']:>8.1e} {r['ss']:>8.1e} "
              f"{r['irf']:>8.1e} {mom:>9}  {'OK' if r['ok'] else 'DIVERGE ' + r['worst']}{note}")
        if r["from_dynare"]:
            print(f"{'':<24}parameters taken from Dynare: {', '.join(r['from_dynare'])}")
        if r["guess_from_dynare"]:
            print(f"{'':<24}s.s. starting point taken from Dynare for "
                  f"{r['guess_from_dynare']} variable(s)")
        bad += not r["ok"]
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
