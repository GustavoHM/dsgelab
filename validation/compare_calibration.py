"""compare_calibration.py: calibration equations against MacroModelling.jl.

The same RBC in both packages, with beta pinned down by the target k/q = 2.5 in the steady state.
In MacroModelling this is `k[ss] / q[ss] = 2.5 | β` in the @parameters block, and in dsgelab
`calibration_equations={"beta": "k/q = 2.5"}`. The calibrated beta and the steady state are
compared. The closed form (beta = 1/(alpha/2.5 + 1 - delta)) arbitrates.

Prerequisite (once, with the Julia project that has MacroModelling and JSON3):
    julia --project=<project> macromodelling/calibration.jl < /dev/null
    # writes macromodelling/calibration.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import dsgelab as dsg

HERE = Path(__file__).resolve().parent

RBC = ["1/c = beta/c(+1)*(alpha*exp(z(+1))*k(+1)**(alpha-1) + 1 - delta)",
       "c + k(+1) = exp(z)*k**alpha + (1-delta)*k", "z(+1) = rho*z + sigma*e",
       "q = exp(z)*k**alpha"]


def main() -> int:
    f = HERE / "macromodelling" / "calibration.json"
    if not f.exists():
        sys.exit(f"missing {f}: run macromodelling/calibration.jl")
    mm = json.loads(f.read_text())
    m = dsg.from_equations(RBC, states=["k", "z"], jumps=["c", "q"], shocks=["e"],
                          parameters=["alpha", "beta", "delta", "rho", "sigma"],
                          calibration_equations={"beta": "k/q = 2.5"})
    p = dict(alpha=0.5, beta=0.95, delta=0.02, rho=0.2, sigma=0.01)
    ss = dsg.steady_state(m, dict(k=5, z=0, c=2, q=2.2), p)
    exact = 1 / (0.5 / 2.5 + 1 - 0.02)
    beta_mm = mm["calibrated"]["β"]
    ok = True
    print(f"  beta    dsgelab {ss.calibrated['beta']:.15f}  MacroModelling {beta_mm:.15f}  "
          f"closed form {exact:.15f}")
    ok &= abs(ss.calibrated["beta"] - exact) < 1e-13 and abs(beta_mm - exact) < 1e-13
    for v in ("c", "k", "q", "z"):
        d = abs(ss.values[v] - mm["steady_state"][v])
        print(f"  {v:7s} dsgelab {ss.values[v]:.15f}  MacroModelling "
              f"{mm['steady_state'][v]:.15f}  {d:.1e}")
        ok &= d < 1e-12
    print("OK" if ok else "DIVERGE")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
