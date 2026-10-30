"""Recovered shocks of a nonlinear RBC against MacroModelling.jl."""
import json
from pathlib import Path

import numpy as np

import dsgelab as dsg

HERE = Path(__file__).resolve().parent


def main():
    path = HERE / "macromodelling" / "inversion.json"
    if not path.exists():
        print("MacroModelling inversion reference missing; run macromodelling/inversion.jl")
        return 1
    reference = json.loads(path.read_text())
    model = dsg.from_equations(
        ["1/c - beta/c(+1)*(alpha*exp(z(+1))*k(+1)**(alpha-1)+1-delta)",
         "c + k(+1) - exp(z)*k**alpha - (1-delta)*k",
         "z(+1) - rho*z - sigma*eps_x",
         "q - exp(z)*k**alpha"],
        states=["k", "z"], jumps=["c", "q"], shocks=["eps_x"],
        parameters=["alpha", "delta", "rho", "sigma", "beta"], name="RBC")
    params = dict(alpha=0.5, delta=0.02, rho=0.2, sigma=0.01,
                  beta=0.8474576271186441)
    steady = dsg.SteadyState(x=np.array([6.25, 0.0, 2.375, 2.5]),
                             values=dict(k=6.25, z=0.0, c=2.375, q=2.5), residual=0.0)
    first = dsg.solve(model, params, steady=steady)
    second = dsg.solve_second_order(first, cov={"eps_x": 0.01})
    result = dsg.inversion_filter(second, {"k": np.asarray(reference["data"]).ravel()},
                                  observables=["k"])
    error = float(np.max(np.abs(result.shocks[:, 0] - reference["shocks"])))
    passed = error < 0.01
    print(f"MacroModelling inversion: recovered-shock max difference {error:.2e}  "
          f"{'OK' if passed else 'FAIL'}")
    print("  MacroModelling's likelihood uses log|G| once and omits Sigma^-1; "
          "the Dynare convention is used for dsgelab's likelihood.")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
