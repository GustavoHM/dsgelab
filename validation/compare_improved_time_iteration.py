"""compare_improved_time_iteration.py: improved time iteration and expectations iteration.

  1. Three of dolo's own models, solved by dolo 0.4.9.20's improved_time_iteration
     (dolo/run_dolo_iti.py, in the environment of dolo/setup.sh) and by dsgelab's on the same
     grid, with the same quadrature nodes or Markov chain, the same bounds and the same start,
     with linear and with cubic interpolation. The models are those of
     compare_time_iteration.py. rbc and rbc_mc start from the controls of dolo's calibration.
     consumption_savings starts from time iteration stopped at a tolerance of 1e-3, from
     min(w, 1), in each package, because dolo's improved_time_iteration does not converge from
     the calibration or, with cubic interpolation, from min(w, 1) (run_dolo_iti.py). The
     policies are compared at the grid points and at the centres of the grid's cells, relative to
     the largest control, and so is dsgelab's improved time iteration against its time iteration
     run to 1e-12. The Newton steps are counted against the iterations of time iteration from
     the same start.
  2. Stochastic growth with log utility and full depreciation, whose policy is known:
     c = (1 - alpha beta) a k^alpha. Improved time iteration on grids of 10 to 80 capital points,
     against the closed form at the centres of the grid's cells, and against time iteration.
  3. The same model with the expectation in the Euler equation declared as a jump,
     z = beta E[c'^(-1) alpha y' / k'] and 1/c = z, solved by expectations_iteration, which dolo
     documents and does not implement. Against the closed form and against time iteration on
     the same grid, and without damping, where it diverges.

Prerequisites:
    bash validation/dolo/setup.sh
    validation/dolo/.venv/bin/python validation/dolo/run_dolo_iti.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

import dsgelab as dsg
from compare_time_iteration import consumption_savings_model, rbc_model
from dsgelab.examples import rbc

HERE = Path(__file__).resolve().parent
TOL = 1e-11
failures: list[str] = []


def check(label: str, value: float, limit: float) -> None:
    status = "ok" if value <= limit else "FAIL"
    if status == "FAIL":
        failures.append(label)
    print(f"  {status:4s} {label:58s} {value:9.2e}  (limit {limit:.0e})")


def setup(name, case, run):
    """The model, the arguments both solvers share, the start, and the order of the controls."""
    p = case["parameters"]
    axes = dict(zip(case["states"] if name != "rbc_mc" else ["k"], run["axes"]))
    kw = dict(grid=axes, interpolation=run["interpolation"])
    if name == "rbc":
        model = rbc_model()
        q = case["quadrature"]
        kw |= dict(quadrature=(q["nodes"], q["weights"]), bounds={"n": (0.0, None)})
        n, i = case["controls"]
        return model, p, kw, {"n": n, "i": i}
    if name == "consumption_savings":
        model = consumption_savings_model()
        q = case["quadrature"]
        kw |= dict(quadrature=(q["nodes"], q["weights"]), bounds={"c": (0.0, lambda s, _: s.w)})
        warm = dsg.time_iteration(model, p, initial={"c": lambda s, _: np.minimum(s.w, 1.0)},
                                  tol=run["warm"], max_iter=5000, **kw)
        return model, p, kw, warm
    mc = case["markov"]
    chain = dsg.MarkovChain(np.asarray(mc["values"])[:, 0], mc["transitions"])
    kw |= dict(markov={"z": chain}, bounds={"n": (0.01, 1.0), "i": (0.0, 1.0)})
    n, i = case["controls"]
    return rbc_model(), p, kw, {"n": n, "i": i}


def states_from(name, sol, points, node):
    if name == "rbc_mc":
        z = sol.chain.nodes[node]
        return np.column_stack([np.full(points.shape[0], z), points[:, 0]])
    return points


def compare_dolo() -> None:
    path = HERE / "out_time_iteration" / "dolo_iti.json"
    if not path.exists():
        print(f"{path} is missing: run validation/dolo/setup.sh and run_dolo_iti.py first")
        failures.append("dolo reference missing")
        return
    data = json.loads(path.read_text())
    print(f"1. dolo {data['meta']['dolo']}'s improved_time_iteration, tolerance {TOL:g}")
    for name, case in data["cases"].items():
        for run in case["runs"]:
            kind = run["interpolation"]
            model, p, kw, start = setup(name, case, run)
            t0 = time.perf_counter()
            sol = dsg.improved_time_iteration(model, p, initial=start, tol=TOL, **kw)
            elapsed = time.perf_counter() - t0
            base = (dict(initial={"c": lambda s, _: np.minimum(s.w, 1.0)})
                    if name == "consumption_savings" else dict(initial=start))
            ti = dsg.time_iteration(model, p, tol=1e-12, max_iter=5000, **base, **kw)
            ti_count = dsg.time_iteration(model, p, tol=TOL, max_iter=5000, **base, **kw)
            warm = start.iterations if name == "consumption_savings" else 0
            axes = [np.asarray(a) for a in run["axes"]]
            nodes = np.column_stack([m.ravel() for m in np.meshgrid(*axes, indexing="ij")])
            mids = np.column_stack([m.ravel() for m in np.meshgrid(
                *[0.5 * (a[1:] + a[:-1]) for a in axes], indexing="ij")])
            d_nodes = d_mids = d_ti = scale = 0.0
            for m_idx, (ref_n, ref_m) in enumerate(zip(run["at_nodes"], run["at_midpoints"])):
                ref_n, ref_m = np.asarray(ref_n), np.asarray(ref_m)
                at_n, at_m = states_from(name, sol, nodes, m_idx), states_from(name, sol, mids,
                                                                               m_idx)
                d_nodes = max(d_nodes, float(np.max(np.abs(sol.controls(at_n) - ref_n))))
                d_mids = max(d_mids, float(np.max(np.abs(sol.controls(at_m) - ref_m))))
                d_ti = max(d_ti, float(np.max(np.abs(sol.controls(at_m) - ti.controls(at_m)))))
                scale = max(scale, float(np.max(np.abs(ref_n))))
            print(f" {name} ({kind}): dsgelab {warm} + {sol.iterations} Newton steps "
                  f"({elapsed:.2f} s), time iteration {ti_count.iterations}; dolo "
                  f"{run['warm_iterations']} + {run['iterations']} Newton steps, time iteration "
                  f"{run['ti_iterations']}")
            if not (sol.converged and run["converged"]):
                failures.append(f"{name} {kind}: not converged")
            check(f"{name}, {kind}: against dolo at the grid points", d_nodes / scale, 1e-8)
            check(f"{name}, {kind}: against dolo between them", d_mids / scale, 1e-8)
            check(f"{name}, {kind}: against time iteration", d_ti / scale, 1e-9)
            if warm + sol.iterations >= ti_count.iterations:
                failures.append(f"{name} {kind}: no fewer iterations than time iteration")


def growth_z():
    def equations(nxt, cur, eps, p):
        y = cur.a * cur.k ** p.alpha
        y1 = nxt.a * nxt.k ** p.alpha
        return [
            nxt.k - ((1.0 - p.delta) * cur.k + y - cur.c),
            cur.c ** (-p.gamma) - cur.z,
            cur.z - p.beta * nxt.c ** (-p.gamma) * (p.alpha * y1 / nxt.k + 1.0 - p.delta),
            np.log(nxt.a) - p.rho * np.log(cur.a) - eps.eps_a,
        ]
    return dsg.Model(states=["k", "a"], jumps=["c", "z"], shocks=["eps_a"], equations=equations,
                     name="growth_expectation")


def closed_form(solver, model, guess, label, **extra) -> None:
    _, p, _ = rbc.build()
    kss, _ = rbc.steady_levels(p)
    for kind, limit in (("linear", 3e-5), ("cubic", 1e-5)):
        errors, gaps, counts = [], [], []
        seconds = [0.0, 0.0]
        for n in (10, 20, 40, 80):
            kw = dict(grid={"k": (0.5 * kss, 1.5 * kss, n), "a": (0.95, 1.05, 9)},
                      cov={"eps_a": 0.007}, guess=guess, interpolation=kind)
            t0 = time.perf_counter()
            sol = solver(model, p, tol=TOL, **extra, **kw)
            t1 = time.perf_counter()
            ti = dsg.time_iteration(model, p, tol=TOL, **kw)
            seconds[0] += t1 - t0
            seconds[1] += time.perf_counter() - t1
            mids = sol.grid.midpoints
            c = sol.controls(mids)[:, 0]
            exact = (1 - p.alpha * p.beta) * mids[:, 1] * mids[:, 0] ** p.alpha
            errors.append(float(np.max(np.abs(c / exact - 1))))
            gaps.append(float(np.max(np.abs(sol.values - ti.values))))
            counts.append((sol.iterations, ti.iterations))
            if not sol.converged:
                failures.append(f"{label} {kind} {n}: not converged")
        print(f"  {kind}: largest relative error of c with 10, 20, 40, 80 capital points: "
              + ", ".join(f"{e:.2e}" for e in errors))
        print(f"  {kind}: iterations, {label} and time iteration: "
              + ", ".join(f"{a} and {b}" for a, b in counts)
              + f"; {seconds[0]:.2f} s and {seconds[1]:.2f} s in all")
        check(f"{kind}, 80 points: largest relative error of c", errors[-1], limit)
        off = max(abs(4.0 - a / b) for a, b in zip(errors, errors[1:]))
        check(f"{kind}: error ratio when the spacing halves, distance from 4", off, 0.5)
        check(f"{kind}: largest difference from time iteration", max(gaps), 1e-9)


def compare_growth() -> None:
    model, p, guess = rbc.build()
    print("2. improved time iteration on stochastic growth, against the closed form")
    closed_form(dsg.improved_time_iteration, model, guess, "improved")
    print("3. expectations iteration on stochastic growth, against the closed form")
    mz = growth_z()
    gz = dict(guess) | {"z": 1.0 / guess["c"]}
    closed_form(dsg.expectations_iteration, mz, gz, "expectations", expectations=["z"])
    kss, _ = rbc.steady_levels(p)
    with np.errstate(all="ignore"):
        plain = dsg.expectations_iteration(
            mz, p, expectations=["z"], damping=1.0, tol=TOL, max_iter=50, guess=gz,
            grid={"k": (0.5 * kss, 1.5 * kss, 20), "a": (0.95, 1.05, 9)}, cov={"eps_a": 0.007})
    first = [f"{h:.1e}" for h in plain.history[:5]]
    print(f"  without damping, the first steps move the policy by {', '.join(first)}")
    if plain.converged:
        failures.append("expectations iteration without damping converged")


if __name__ == "__main__":
    compare_dolo()
    compare_growth()
    if failures:
        print(f"\n{len(failures)} failure(s): {failures}")
        sys.exit(1)
    print("\nall checks passed")
