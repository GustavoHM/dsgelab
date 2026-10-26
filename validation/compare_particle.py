"""Bootstrap particle-filter likelihood against Dynare 7.2."""
from pathlib import Path

import numpy as np
from scipy.io import loadmat

import dsgelab as dsg

HERE = Path(__file__).resolve().parent


def main():
    path = HERE / "out_particle" / "data.mat"
    if not path.exists():
        print("Particle reference missing; run bash run_particle.sh")
        return 1

    def eqs(nxt, cur, eps, p):
        return [nxt.y - p["rho"] * cur.y - eps.e]

    model = dsg.Model(states=["y"], jumps=[], shocks=["e"], equations=eqs, name="ar1")
    first = dsg.solve(model, {"rho": 0.5}, guess={"y": 0.0})
    second = dsg.solve_second_order(first, cov={"e": 0.2})
    y = loadmat(path)["y"].ravel()
    ok = True
    for order, solution in ((1, first), (2, second)):
        ref_path = HERE / "out_particle" / f"order{order}.mat"
        reference = loadmat(ref_path, squeeze_me=True, struct_as_record=False)["ex"]
        result = dsg.particle_filter(
            solution, {"y": y}, observables=["y"], measurement={"y": 0.2},
            cov={} if order == 2 else {"e": 0.2}, particles=100_000, seed=4, resample=1.0)
        error = abs(result.loglik - float(reference.loglik))
        passed = error < 0.05
        ok = ok and passed
        print(f"order {order}: dsgelab {result.loglik:.9f}, Dynare {float(reference.loglik):.9f}, "
              f"difference {error:.2e}  {'OK' if passed else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
