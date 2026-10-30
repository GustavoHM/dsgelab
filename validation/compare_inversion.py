"""OccBin inversion-filter shocks and likelihood against Dynare 7.2."""
from pathlib import Path

import numpy as np
from scipy.io import loadmat

import dsgelab as dsg

HERE = Path(__file__).resolve().parent


def main():
    path = HERE / "out_inversion" / "order1.mat"
    if not path.exists():
        print("Inversion reference missing; run bash run_inversion.sh")
        return 1

    def eqs(nxt, cur, eps, p):
        return [nxt.y - p["rho"] * cur.y - eps.e]

    model = dsg.Model(states=["y"], jumps=[], shocks=["e"], equations=eqs, name="ar1")
    first = dsg.solve(model, {"rho": 0.5}, guess={"y": 0.0})
    y = loadmat(HERE / "out_inversion" / "data.mat")["y"].ravel()
    reference = loadmat(path, squeeze_me=True, struct_as_record=False)["ex"]
    result = dsg.inversion_filter(first, {"y": y}, observables=["y"], cov={"e": 0.2})
    errors = {
        "shocks": float(np.max(np.abs(result.shocks[:, 0] - np.asarray(reference.shocks).ravel()))),
        "jacobian": float(np.max(np.abs(result.jacobians - reference.Emat))),
        "likelihood": abs(result.loglik - float(reference.loglik)),
        "per_date": float(np.max(np.abs(result.loglik_t - reference.loglik_t))),
    }
    ok = max(errors.values()) < 1e-8
    print("  " + ", ".join(f"{key} {value:.2e}" for key, value in errors.items())
          + f"  {'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
