"""compare.py: central-bank models of the Macroeconomic Model Data Base, read by from_mod.

The .mod files and their data/helper inputs come from Rep-MMB (fetch.sh). Most models are read as
they are; prepare.py makes the MATLAB-supplied values of four models explicit in out/prepared/.
The comparison covers steady state, orthogonalised IRFs of every variable to every shock with
nonzero variance, 40 periods, and theoretical standard deviations. Parameters still left undefined,
and steady-state starting points where needed, come from Dynare, and the output says so.

The steady-state error is relative, and absolute for values below 1. Several of these models
have variables whose steady state is zero (EACZ_GEM03's output gaps, at 7e-13 in dsgelab and 0
in Dynare), and a purely relative error would measure their rounding.

Prerequisites:
    bash fetch.sh
    python prepare.py      # explicit inputs and QUEST3's reference start in out/prepared/
    bash run_dynare.sh     # exports out/<name>.mat
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from compare_import import compare  # noqa: E402
from prepare import SOURCES  # noqa: E402

VALIDATED = (
    "BRA_SAMBA08", "CA_BMZ12", "CA_ToTEM10", "CA_LS07", "CL_MS07", "DEREA_GEAR16",
    "EACZ_GEM03", "EAUS_NAWM08", "EA_ALSV06", "EA_AWM05", "EA_GNSS10", "EA_QUEST3",
    "ESREA_FIMOD12", "G2_SIGMA08", "FI_AINO16", "GPM6_IMF13", "EA_SWW14",
    "HK_FP13", "UK_SM11", "US_DNGS15", "US_FRB03", "US_FRB08", "US_SW07",
)


def main() -> int:
    bad = 0
    print(f"{'model':<13}{'vars':>5}{'aux':>5}  {'params':>8} {'s.s.':>8} {'IRFs':>8} "
          f"{'moments':>9}")
    mods = ([HERE / "mods" / f"{name}.mod" for name in sys.argv[1:]]
            if len(sys.argv) > 1 else [HERE / "mods" / f"{name}.mod" for name in VALIDATED])
    if not mods:
        print("no .mod in mods/, run fetch.sh")
        return 1
    for path in mods:
        name = path.stem
        try:
            source = HERE / "out" / "prepared" / f"{name}.mod" if name in SOURCES else path
            r = compare(name, source, out=HERE / "out", ss_floor=1.0,
                        steady_from_dynare=name in {"GPM6_IMF13", "US_CMR14"},
                        steady_start_from_dynare=name in {"CA_BMZ12", "EA_GNSS10"},
                        steady_tol=1e-5 if name in {"CA_ToTEM10", "EA_GNSS10"} else 1e-11,
                        check_steady=name not in {"CA_ToTEM10", "EA_GNSS10"},
                        steady_analytic=name == "CA_ToTEM10")
        except Exception as exc:
            print(f"{name:<13} DIVERGE {type(exc).__name__}: {exc}")
            bad += 1
            continue
        if r.get("missing"):
            print(f"{name:<13} no out/{name}.mat, run run_dynare.sh")
            bad += 1
            continue
        mom = f"{r['mom']:.1e}" if r["mom"] is not None else "n/a"
        note = "  (Dynare started at the independently solved dsgelab s.s.)" if name == "EA_QUEST3" else ""
        if r["steady_from_dynare"]:
            note = f"  (common Dynare point, residual {r['r_dy']:.0e})"
        elif r["r_dy"] > 1e3 * max(r["r_dk"], 1e-16):
            note = f"  (Dynare's s.s. with residual {r['r_dy']:.0e})"
        elif max(r["r_dy"], r["r_dk"]) > 1e-9:
            note = f"  (s.s. residual {max(r['r_dy'], r['r_dk']):.0e})"
        print(f"{name:<13}{r['n']:>5}{r['aux']:>5}  {r['par']:>8.1e} {r['ss']:>8.1e} "
              f"{r['irf']:>8.1e} {mom:>9}  {'OK' if r['ok'] else 'DIVERGE ' + r['worst']}{note}")
        if name in SOURCES:
            print(f"{'':<13}explicit inputs from pinned Rep-MMB sources: {', '.join(SOURCES[name])}")
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
