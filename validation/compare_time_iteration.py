"""compare_time_iteration.py: time iteration against dolo and against the closed form.

  1. Three of dolo's own models, solved by dolo 0.4.9.20's time_iteration (dolo/run_dolo.py, in
     the environment of dolo/setup.sh) and by dsgelab's on the same grid, with the same quadrature
     nodes or Markov chain and the same bounds, with linear and with cubic interpolation:
       rbc                  states z and k, an iid shock by Gauss-Hermite, labour bounded below;
       consumption_savings  cash on hand, an iid income shock on dolo's equiprobable nodes, and
                            consumption between 0 and cash on hand, the borrowing limit;
       rbc_mc               capital, productivity a two-state Markov chain, labour and investment
                            bounded on both sides.
     The two solve the same equations on the same grid with the same interpolant: dolo's cubic
     splines (from the interpolation package) also have a zero second derivative at the ends and
     extrapolate linearly. The policies are compared at the grid points and at the centres of the
     grid's cells, relative to the largest control. dolo solves each step to a residual of 1e-8,
     with at most 10 Newton steps, which leaves its policy some 1e-7 from the fixed point;
     run_dolo.py tightens both, as explained there.
  2. Stochastic growth with log utility and full depreciation, whose policy is known:
     c = (1 - alpha beta) a k^alpha. Linear and cubic interpolation, on grids of 10 to 80 capital
     points, at the centres of the grid's cells. Both errors fall as the square of the spacing:
     the natural spline's zero second derivative at the ends, where the policy's is not zero,
     makes the cubic second order there, with a constant about three times smaller.

Prerequisites:
    bash validation/dolo/setup.sh
    validation/dolo/.venv/bin/python validation/dolo/run_dolo.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

import dsgelab as dsg
from dsgelab.examples import rbc

HERE = Path(__file__).resolve().parent
TOL = 1e-11
failures: list[str] = []


def check(label: str, value: float, limit: float) -> None:
    status = "ok" if value <= limit else "FAIL"
    if status == "FAIL":
        failures.append(label)
    print(f"  {status:4s} {label:58s} {value:9.2e}  (limit {limit:.0e})")


def rbc_model():
    def equations(nxt, cur, eps, p):
        def parts(x):
            y = np.exp(x.z) * x.k ** p["alpha"] * x.n ** (1 - p["alpha"])
            return y, y - x.i, p["alpha"] * y / x.k, (1 - p["alpha"]) * y / x.n
        y, c, rk, w = parts(cur)
        y1, c1, rk1, w1 = parts(nxt)
        return [
            nxt.z - (p["rho"] * cur.z + eps.e_z),
            nxt.k - ((1 - p["delta"]) * cur.k + cur.i),
            p["chi"] * cur.n ** p["eta"] * c ** p["sigma"] - w,
            1 - p["beta"] * (c / c1) ** p["sigma"] * (1 - p["delta"] + rk1),
        ]
    return dsg.Model(states=["z", "k"], jumps=["n", "i"], shocks=["e_z"], equations=equations,
                     name="dolo_rbc")


def consumption_savings_model():
    def equations(nxt, cur, eps, p):
        return [
            nxt.w - (np.exp(eps.y) + (cur.w - cur.c) * p["r"]),
            p["β"] * (nxt.c / cur.c) ** (-p["γ"]) * p["r"] - 1,
        ]
    return dsg.Model(states=["w"], jumps=["c"], shocks=["y"], equations=equations,
                     name="dolo_consumption_savings")


def rbc_steady(p):
    rk = 1 / p["beta"] - 1 + p["delta"]
    n = 0.33
    k = n / (rk / p["alpha"]) ** (1 / (1 - p["alpha"]))
    return {"z": 0.0, "k": k, "n": n, "i": p["delta"] * k}


def solve_case(name, case, run):
    p = case["parameters"]
    axes = dict(zip(case["states"] if name != "rbc_mc" else ["k"], run["axes"]))
    kw = dict(interpolation=run["interpolation"], tol=TOL, max_iter=5000)
    if name == "rbc":
        model = rbc_model()
        q = case["quadrature"]
        sol = dsg.time_iteration(model, p, grid=axes, quadrature=(q["nodes"], q["weights"]),
                                 bounds={"n": (0.0, None)}, guess=rbc_steady(p), **kw)
        order = [0, 1]          # dolo's controls are n, i; so are dsgelab's jumps
    elif name == "consumption_savings":
        model = consumption_savings_model()
        q = case["quadrature"]
        sol = dsg.time_iteration(model, p, grid=axes, quadrature=(q["nodes"], q["weights"]),
                                 bounds={"c": (0.0, lambda s, _: s.w)},
                                 initial={"c": lambda s, _: 0.9 * s.w}, **kw)
        order = [0]
    else:
        # z follows the chain, and its law in the model's equations is left out
        model = rbc_model()
        mc = case["markov"]
        values = np.asarray(mc["values"])
        chain = dsg.MarkovChain(values[:, 0], mc["transitions"])
        guess = rbc_steady(p)
        sol = dsg.time_iteration(model, p, grid=axes, markov={"z": chain},
                                 bounds={"n": (0.01, 1.0), "i": (0.0, 1.0)},
                                 initial={"n": guess["n"], "i": guess["i"]}, **kw)
        order = [0, 1]
    return sol, order


def states_from(name, sol, points, node):
    """dolo's endogenous-grid points as dsgelab state vectors at Markov node ``node``."""
    if name == "rbc_mc":
        z = sol.chain.nodes[node]
        return np.column_stack([np.full(points.shape[0], z), points[:, 0]])
    return points


def compare_dolo() -> None:
    path = HERE / "out_time_iteration" / "dolo.json"
    if not path.exists():
        print(f"{path} is missing: run validation/dolo/setup.sh and run_dolo.py first")
        failures.append("dolo reference missing")
        return
    data = json.loads(path.read_text())
    meta = data["meta"]
    print(f"1. dolo {meta['dolo']} (interpolation {meta['interpolation']}), tolerance {TOL:g}")
    for name, case in data["cases"].items():
        runs = {r["interpolation"]: r for r in case["runs"]}
        for kind in ("linear", "cubic"):
            run = runs[kind]
            t0 = time.perf_counter()
            sol, order = solve_case(name, case, run)
            elapsed = time.perf_counter() - t0
            axes = [np.asarray(a) for a in run["axes"]]
            mesh = np.meshgrid(*axes, indexing="ij")
            nodes = np.column_stack([m.ravel() for m in mesh])
            mids = np.column_stack([m.ravel() for m in np.meshgrid(
                *[0.5 * (a[1:] + a[:-1]) for a in axes], indexing="ij")])
            d_nodes = d_mids = 0.0
            scale = 0.0
            for m_idx, (ref_n, ref_m) in enumerate(zip(run["at_nodes"], run["at_midpoints"])):
                ref_n, ref_m = np.asarray(ref_n), np.asarray(ref_m)
                ours_n = sol.controls(states_from(name, sol, nodes, m_idx))[:, order]
                ours_m = sol.controls(states_from(name, sol, mids, m_idx))[:, order]
                d_nodes = max(d_nodes, float(np.max(np.abs(ours_n - ref_n))))
                d_mids = max(d_mids, float(np.max(np.abs(ours_m - ref_m))))
                scale = max(scale, float(np.max(np.abs(ref_n))))
            print(f" {name} ({kind}): dsgelab {sol.iterations} iterations, {elapsed:.2f} s; "
                  f"dolo {run['iterations']} iterations, {run['seconds']:.2f} s")
            if not sol.converged:
                failures.append(f"{name} {kind}: dsgelab did not converge")
            check(f"{name}, {kind}: largest difference at the grid points", d_nodes / scale,
                  1e-8)
            check(f"{name}, {kind}: largest difference between them", d_mids / scale, 1e-8)
            if name == "consumption_savings":
                w = nodes[:, 0]
                c = sol.controls(nodes)[:, 0]
                binding = np.isclose(c, w, rtol=0, atol=1e-10)
                ref_binding = np.isclose(np.asarray(run["at_nodes"][0])[:, 0], w, rtol=0,
                                         atol=1e-8)
                print(f"       the borrowing limit binds at {binding.sum()} of {w.size} grid "
                      f"points in dsgelab, {ref_binding.sum()} in dolo")
                if binding.sum() != ref_binding.sum():
                    failures.append(f"{name} {kind}: binding points differ")
            err = np.abs(sol.euler_errors())
            print(f"       Euler-equation errors at the cells' centres: max {err.max():.2e}, "
                  f"mean {err.mean():.2e}")


def compare_closed_form() -> None:
    print("2. stochastic growth with log utility and full depreciation, against the closed form")
    model, p, guess = rbc.build()
    kss, _ = rbc.steady_levels(p)
    for kind, limit in (("linear", 3e-5), ("cubic", 1e-5)):
        errors = []
        for n in (10, 20, 40, 80):
            sol = dsg.time_iteration(model, p, grid={"k": (0.5 * kss, 1.5 * kss, n),
                                                     "a": (0.95, 1.05, 9)},
                                     cov={"eps_a": 0.007}, guess=guess, interpolation=kind,
                                     tol=TOL)
            mids = sol.grid.midpoints
            c = sol.controls(mids)[:, 0]
            exact = (1 - p.alpha * p.beta) * mids[:, 1] * mids[:, 0] ** p.alpha
            errors.append(float(np.max(np.abs(c / exact - 1))))
        print(f"  {kind}: largest relative error of c with 10, 20, 40, 80 capital points: "
              + ", ".join(f"{e:.2e}" for e in errors))
        check(f"{kind}, 80 points: largest relative error of c", errors[-1], limit)
        off = max(abs(4.0 - a / b) for a, b in zip(errors, errors[1:]))
        check(f"{kind}: error ratio when the spacing halves, distance from 4", off, 0.5)


if __name__ == "__main__":
    compare_dolo()
    compare_closed_form()
    if failures:
        print(f"\n{len(failures)} failure(s): {failures}")
        sys.exit(1)
    print("\nall checks passed")
