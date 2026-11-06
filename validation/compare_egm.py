"""compare_egm.py: the endogenous grid method against dolo, time iteration and the closed form.

  1. dolo 0.4.9.20's egm on its consumption-savings model (dolo/run_dolo_egm.py, in the
     environment of dolo/setup.sh), with the iid income shock as published and with log income a
     three-state Markov chain. dsgelab's egm solves the same program, written from the file's
     expectation, half_transition, direct_response_egm and reverse_state blocks, on the same
     poststate grid with the same quadrature nodes or chain and the same bounds. Both interpolate
     linearly between the points of the endogenous grid and hold the policy within the bounds,
     so they iterate on the same fixed point. The policies are compared at 199 points of cash on
     hand, the 100 of the file and the centres between them, relative to the largest
     consumption.
  2. dsgelab's time_iteration on the same model, with as many points of cash on hand as egm has
     poststates. The two share the problem's solution but not their discretisation. Time
     iteration interpolates on a fixed grid of cash on hand, on which the kink where the
     borrowing limit stops binding falls inside a cell, and egm on the endogenous grid, whose
     lowest point is that kink. On dolo's grids, both up to 4, next period's cash on hand exceeds
     4 when income is high, time iteration extrapolates its policy there, and with the Markov
     chain the difference does not fall as the grids are refined. With both grids up to 8, the
     largest and the mean difference at cash on hand up to 4 are printed for 100 to 1600 points.
  3. Stochastic growth with log utility and full depreciation, whose policy is known:
     c = (1 - alpha beta) exp(z) k^alpha, with productivity a Rouwenhorst chain. The poststate is
     next period's capital, on 10 to 80 points. The relative error of c at the centres between
     the points of the endogenous grid falls as the square of the spacing, with linear and with
     cubic interpolation, for the reason given in compare_time_iteration.py.

Prerequisites:
    bash validation/dolo/setup.sh
    validation/dolo/.venv/bin/python validation/dolo/run_dolo_egm.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

import dsgelab as dsg

HERE = Path(__file__).resolve().parent
TOL = 1e-12
ALPHA, BETA = 0.36, 0.95
KSS = (ALPHA * BETA) ** (1 / (1 - ALPHA))
failures: list[str] = []


def check(label: str, value: float, limit: float) -> None:
    status = "ok" if value <= limit else "FAIL"
    if status == "FAIL":
        failures.append(label)
    print(f"  {status:4s} {label:58s} {value:9.2e}  (limit {limit:.0e})")


def saving_program(markov: bool) -> dsg.EGMProgram:
    """consumption_savings_iid.yaml's blocks; with a chain, income's next value comes from it."""
    def half_transition(a, e, p):
        return np.exp(e.y) + a.a * p["r"]

    return dsg.EGMProgram(
        states=["w", "y"] if markov else ["w"], controls=["c"], poststate="a",
        expectations=["mr"], shocks=[] if markov else ["y"],
        expectation=lambda s, x, e, p: p["β"] * (x.c / p["cbar"]) ** (-p["γ"]) * p["r"],
        direct_response=lambda a, z, p: p["cbar"] * z.mr ** (-1 / p["γ"]),
        reverse_state=lambda a, x, p: a.a + x.c,
        half_transition=half_transition,
        post_decision=lambda s, x, p: s.w - x.c,
        bounds={"c": (0.0, lambda s, p: s.w)}, name="consumption_savings")


def saving_model(markov: bool) -> dsg.Model:
    """The same problem as a Model, for time_iteration: the yaml's transition and arbitrage."""
    def equations(nxt, cur, eps, p):
        y = nxt.y if markov else eps.y
        rows = [nxt.w - (np.exp(y) + (cur.w - cur.c) * p["r"]),
                p["β"] * (nxt.c / cur.c) ** (-p["γ"]) * p["r"] - 1]
        if markov:
            rows.insert(1, nxt.y - cur.y)      # left out: the chain gives y's next value
        return rows
    return dsg.Model(states=["w", "y"] if markov else ["w"], jumps=["c"],
                     shocks=[] if markov else ["y"], equations=equations,
                     name="consumption_savings")


def integration(case: dict) -> dict:
    if "markov" in case:
        values = np.asarray(case["markov"]["values"])
        return {"markov": {"y": dsg.MarkovChain(values[:, 0], case["markov"]["transitions"])}}
    q = case["quadrature"]
    return {"quadrature": (q["nodes"], q["weights"])}


def states_at(w: np.ndarray, case: dict, node: int) -> np.ndarray:
    if "markov" in case:
        y = np.asarray(case["markov"]["values"])[node, 0]
        return np.column_stack([w, np.full(w.size, y)])
    return w[:, None]


def compare_dolo(data: dict) -> None:
    meta = data["meta"]
    print(f"1. dolo {meta['dolo']} (interpolation {meta['interpolation']}), egm, tolerance {TOL:g}")
    for name, case in data["cases"].items():
        markov = "markov" in case
        p = case["parameters"]
        t0 = time.perf_counter()
        sol = dsg.egm(saving_program(markov), p, grid={"a": np.asarray(case["a_grid"])},
                      initial={"c": lambda s, _: 0.9 * s.w}, tol=TOL, max_iter=5000,
                      **integration(case))
        elapsed = time.perf_counter() - t0
        w = np.asarray(case["w"])
        diff = scale = 0.0
        for node, ref in enumerate(case["controls"]):
            ours = sol.controls(states_at(w, case, node))[:, 0]
            diff = max(diff, float(np.max(np.abs(ours - np.asarray(ref)))))
            scale = max(scale, float(np.max(np.abs(ref))))
        print(f" {name}: dsgelab {sol.iterations} iterations, {elapsed:.2f} s; "
              f"dolo {case['iterations']} iterations, {case['seconds']:.2f} s")
        if not sol.converged:
            failures.append(f"{name}: dsgelab did not converge")
        check(f"{name}: largest difference at 199 points of cash on hand", diff / scale, 1e-10)
        err = np.abs(sol.euler_errors())
        print(f"       Euler errors between the endogenous points, relative to c: max "
              f"{err.max():.2e}, mean {err.mean():.2e}")


def difference(case: dict, n: int, w_hi: float, a_hi: float) -> tuple[float, float]:
    """Time iteration and egm with ``n`` points: the largest and the mean difference in c at the
    points of time iteration's grid where cash on hand is at most 4, dolo's upper end."""
    markov = "markov" in case
    p = case["parameters"]
    w = np.linspace(case["w"][0], w_hi, n)
    ti = dsg.time_iteration(saving_model(markov), p, grid={"w": w},
                            bounds={"c": (0.0, lambda s, _: s.w)},
                            initial={"c": lambda s, _: 0.9 * s.w}, tol=TOL, max_iter=5000,
                            **integration(case))
    sol = dsg.egm(saving_program(markov), p, grid={"a": (case["a_grid"][0], a_hi, n)},
                  initial={"c": lambda s, _: 0.9 * s.w}, tol=TOL, max_iter=5000,
                  **integration(case))
    if not (ti.converged and sol.converged):
        failures.append(f"{n} points up to {w_hi:g}: did not converge")
    inside = w <= case["w"][-1] + 1e-12
    gaps = np.concatenate([
        np.abs(sol.controls(states_at(w, case, node))[:, 0] - ti.values[node, :, 0])[inside]
        for node in range(ti.values.shape[0])])
    return float(gaps.max()), float(gaps.mean())


def compare_time_iteration(data: dict) -> None:
    print("2. time_iteration on the same model")
    for name, case in data["cases"].items():
        largest, mean = difference(case, 100, case["w"][-1], case["a_grid"][-1])
        print(f"  {name}: on dolo's grids, 100 points of cash on hand up to 4 and 100 poststates "
              f"up to 4: largest difference in c {largest:.2e}, mean {mean:.2e}")
        check(f"{name}, dolo's grids: largest difference in c", largest, 5e-3)
        rows = [difference(case, n, 8.0, 8.0) for n in (100, 200, 400, 800, 1600)]
        print(f"  {name}: both grids up to 8, with 100, 200, 400, 800, 1600 points, at cash on "
              "hand up to 4:")
        print("       largest difference in c " + ", ".join(f"{r[0]:.2e}" for r in rows))
        print("       mean difference in c    " + ", ".join(f"{r[1]:.2e}" for r in rows))
        check(f"{name}, up to 8, 1600 points: largest difference in c", rows[-1][0], 1e-4)
        check(f"{name}, up to 8, 1600 points: mean difference in c", rows[-1][1], 5e-6)


def growth_program() -> dsg.EGMProgram:
    """Stochastic growth: the poststate is next period's capital, z the log of productivity."""
    return dsg.EGMProgram(
        states=["k", "z"], controls=["c"], poststate="kp", expectations=["q"],
        expectation=lambda s, x, e, p: BETA * ALPHA * np.exp(s.z) * s.k ** (ALPHA - 1) / x.c,
        direct_response=lambda a, z, p: 1 / z.q,
        reverse_state=lambda a, x, p: ((a.kp + x.c) / np.exp(a.z)) ** (1 / ALPHA),
        half_transition=lambda a, e, p: a.kp,
        post_decision=lambda s, x, p: np.exp(s.z) * s.k ** ALPHA - x.c, name="growth")


def compare_closed_form() -> None:
    print("3. stochastic growth with log utility and full depreciation, against the closed form")
    chain = dsg.rouwenhorst(0.9, 0.02, 5)
    for kind, limit, off in (("linear", 2e-4, 0.5), ("cubic", 1e-4, 1.0)):
        errors = []
        for n in (10, 20, 40, 80):
            sol = dsg.egm(growth_program(), {}, grid={"kp": (0.5 * KSS, 1.5 * KSS, n)},
                          markov={"z": chain}, interpolation=kind, tol=TOL,
                          initial={"c": lambda s, _: 0.5 * np.exp(s.z) * s.k ** ALPHA})
            worst = 0.0
            for m, z in enumerate(chain.nodes):
                k = 0.5 * (sol.endogenous[m, 1:] + sol.endogenous[m, :-1])
                c = sol.controls(np.column_stack([k, np.full(k.size, z)]))[:, 0]
                exact = (1 - ALPHA * BETA) * np.exp(z) * k ** ALPHA
                worst = max(worst, float(np.max(np.abs(c / exact - 1))))
            errors.append(worst)
        print(f"  {kind}: largest relative error of c with 10, 20, 40, 80 poststates: "
              + ", ".join(f"{e:.2e}" for e in errors))
        ratios = [a / b for a, b in zip(errors, errors[1:])]
        print("         error ratios when the spacing halves: "
              + ", ".join(f"{r:.2f}" for r in ratios))
        check(f"{kind}, 80 points: largest relative error of c", errors[-1], limit)
        check(f"{kind}: error ratio when the spacing halves, distance from 4",
              max(abs(4.0 - r) for r in ratios), off)


if __name__ == "__main__":
    path = HERE / "out_egm" / "dolo.json"
    if path.exists():
        data = json.loads(path.read_text())
        compare_dolo(data)
        compare_time_iteration(data)
    else:
        print(f"{path} is missing: run validation/dolo/setup.sh and run_dolo_egm.py first")
        failures.append("dolo reference missing")
    compare_closed_form()
    if failures:
        print(f"\n{len(failures)} failure(s): {failures}")
        sys.exit(1)
    print("\nall checks passed")
