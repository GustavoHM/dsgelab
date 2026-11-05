"""compare_smolyak.py: Smolyak grids against Tasmanian, and their collocation against a closed form.

  1. Tasmanian 8.2's global grids of type "level" on the Clenshaw-Curtis rule
     (tasmanian/run_tasmanian.py, in the environment of tasmanian/setup.sh), isotropic in 1 to 6
     dimensions at levels 0 to 6, and anisotropic with a level cap on each axis, on the canonical
     box and on boxes of their own. For each grid, SmolyakGrid must have the same number of
     points, each of Tasmanian's points must be one of SmolyakGrid's, and the interpolation weights
     at 40 points in the box must agree with getInterpolationWeights once the points are matched.
  2. Stochastic growth with several capital stocks, log utility and full depreciation: a planner
     chooses next period's capital stocks k_j' to maximize E sum beta^t log c_t subject to
     c + k_1' + ... + k_n' = a k_1^alpha_1 ... k_n^alpha_n, with log a an AR(1). The policy is
     k_j' = beta alpha_j y in any dimension, since with c = (1 - beta sum alpha) y each Euler
     equation 1/c = beta E[alpha_j y' / (k_j' c')] holds exactly. With 3, 4 and 5 capital stocks
     (4, 5 and 6 states), solve_projection and time_iteration on isotropic Smolyak grids of
     levels 1 to 4 give the investment policy; the relative error against k_j' = beta alpha_j y
     is measured at 1000 points drawn uniformly in the box. The two methods solve the same
     collocation equations, so they must also agree with each other. The policy is linear in a,
     so an anisotropic grid with level 1 along a and the isotropic level along the capital stocks
     should lose nothing; with 6 states it is solved at levels 3 and 4.

Prerequisites:
    bash validation/tasmanian/setup.sh
    validation/tasmanian/.venv/bin/python validation/tasmanian/run_tasmanian.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

import dsgelab as dsg

HERE = Path(__file__).resolve().parent
failures: list[str] = []


def check(label: str, value: float, limit: float) -> None:
    status = "ok" if value <= limit else "FAIL"
    if status == "FAIL":
        failures.append(label)
    print(f"  {status:4s} {label:62s} {value:9.2e}  (limit {limit:.3g})")


def compare_tasmanian() -> None:
    path = HERE / "out_smolyak" / "tasmanian.json"
    if not path.exists():
        print(f"{path} is missing: run validation/tasmanian/setup.sh and run_tasmanian.py first")
        failures.append("Tasmanian reference missing")
        return
    data = json.loads(path.read_text())
    meta = data["meta"]
    print(f"1. Tasmanian {meta['tasmanian']}, global grids of type {meta['type']!r} on the "
          f"{meta['rule']} rule")
    worst_node = worst_weight = 0.0
    sizes_differ = []
    print(f"  {'levels':20s} {'box':10s} {'points':>7s} {'nodes':>9s} {'weights':>9s}")
    for case in data["cases"]:
        bounds = np.asarray(case["bounds"])
        d = bounds.shape[0]
        grid = dsg.SmolyakGrid({f"x{j}": tuple(b) for j, b in enumerate(bounds)}, case["levels"])
        ref = np.asarray(case["points"]).reshape(-1, d)
        if ref.shape[0] != grid.n:
            sizes_differ.append(case["levels"])
            continue
        # each of Tasmanian's points matched to the nearest of SmolyakGrid's, on the unit box
        unit = 2.0 * (ref - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0]) - 1.0
        dist = np.abs(unit[:, None, :] - grid.unit_points[None, :, :]).max(axis=2)
        match = dist.argmin(axis=1)
        d_node = float(np.max(np.abs(ref - grid.points[match])))
        if np.unique(match).size != grid.n:
            failures.append(f"{case['levels']}: points matched twice")
        W = grid.interpolation_weights(np.asarray(case["at"]))[:, match]
        d_weight = float(np.max(np.abs(W - np.asarray(case["weights"]))))
        worst_node, worst_weight = max(worst_node, d_node), max(worst_weight, d_weight)
        canonical = np.allclose(bounds, [[-1.0, 1.0]] * d)
        print(f"  {str(case['levels']):20s} {'[-1, 1]' if canonical else 'own':10s} "
              f"{grid.n:7d} {d_node:9.1e} {d_weight:9.1e}")
    if sizes_differ:
        failures.append(f"numbers of points differ for {sizes_differ}")
    print(f"  {len(data['cases'])} grids, {len(sizes_differ)} with a different number of points")
    check("largest difference in a node", worst_node, 1e-14)
    check("largest difference in an interpolation weight", worst_weight, 1e-12)


def growth_model(n: int):
    """The planner's problem with ``n`` capital stocks; the jumps are next period's stocks."""
    shares = np.array([0.15, 0.10, 0.07, 0.05, 0.03, 0.02][:n])
    p = {"beta": 0.96, "rho": 0.9, "alpha": 0.36 * shares / shares.sum()}
    ks, inv = [f"k{j}" for j in range(n)], [f"i{j}" for j in range(n)]

    def equations(nxt, cur, eps, p):
        def output(x):
            y = x.a * np.prod([x[k] ** a for k, a in zip(ks, p["alpha"])], axis=0)
            return y, y - sum(x[i] for i in inv)
        y, c = output(cur)
        y1, c1 = output(nxt)
        return ([nxt[k] - cur[i] for k, i in zip(ks, inv)]
                + [1 - p["beta"] * c / c1 * a * y1 / nxt[k] for k, a in zip(ks, p["alpha"])]
                + [np.log(nxt.a) - p["rho"] * np.log(cur.a) - eps.e])

    model = dsg.Model(states=ks + ["a"], jumps=inv, shocks=["e"], equations=equations,
                      name=f"growth_{n}")
    alpha = p["alpha"]
    log_y = np.sum(alpha * np.log(p["beta"] * alpha)) / (1 - alpha.sum())
    kss = p["beta"] * alpha * np.exp(log_y)
    guess = dict(zip(ks, kss)) | {"a": 1.0} | dict(zip(inv, kss))
    axes = {k: (0.75 * v, 1.25 * v) for k, v in zip(ks, kss)} | {"a": (0.97, 1.03)}

    def exact(X):
        y = X[:, -1] * np.prod(X[:, :n] ** alpha, axis=1)
        return p["beta"] * alpha[None, :] * y[:, None]

    return model, p, guess, axes, exact


def solve_both(n, levels, X):
    """Projection and time iteration on one grid: errors, seconds and their difference."""
    model, p, guess, axes, exact = growth_model(n)
    ref = exact(X)
    grid = dsg.SmolyakGrid(axes, levels)
    t0 = time.perf_counter()
    proj = dsg.solve_projection(model, p, grid=grid, guess=guess, cov={"e": 0.01}, n_quad=5)
    t1 = time.perf_counter()
    ti = dsg.time_iteration(model, p, grid=grid, cov={"e": 0.01}, n_quad=5, guess=guess,
                            tol=1e-11)
    t2 = time.perf_counter()
    if not (proj.success and ti.converged):
        failures.append(f"{n + 1} states, levels {levels}: did not converge")
    e_p = float(np.max(np.abs(proj.jumps(X) / ref - 1)))
    e_t = float(np.max(np.abs(ti.controls(X) / ref - 1)))
    gap = float(np.max(np.abs(proj.jumps(X) - ti.controls(X)) / ref))
    return grid.n, e_p, t1 - t0, e_t, t2 - t1, gap


def print_rows(rows) -> None:
    print(f"    {'levels':>18s} {'points':>6s}  {'projection':>18s}  {'time iteration':>18s}"
          f"  {'difference':>10s}")
    for lv, npts, e_p, s_p, e_t, s_t, gap in rows:
        print(f"    {str(lv):>18s} {npts:6d}  {e_p:9.2e} [{s_p:5.1f}]  {e_t:9.2e} [{s_t:5.1f}]"
              f"  {gap:10.1e}")


def compare_closed_form() -> None:
    print("2. stochastic growth with several capital stocks, log utility and full depreciation, "
          "against the closed form")
    print("   relative error of k_j' at 1000 points in the box; seconds in brackets")
    rng = np.random.default_rng(0)
    for n, top, limit in ((3, 4, 3e-8), (4, 4, 6e-8), (5, 4, 6e-8)):
        axes = growth_model(n)[3]
        lo, hi = np.array(list(axes.values())).T
        X = rng.uniform(lo, hi, (1000, n + 1))
        rows = [(lv,) + solve_both(n, lv, X) for lv in range(1, top + 1)]
        print(f"  {n + 1} states ({n} capital stocks), isotropic")
        print_rows(rows)
        check(f"{n + 1} states, level {top}: largest relative error, projection",
              rows[-1][2], limit)
        check(f"{n + 1} states, level {top}: largest relative error, time iteration",
              rows[-1][4], limit)
        fall = min(a[2] / b[2] for a, b in zip(rows, rows[1:]))
        check(f"{n + 1} states: smallest fall of the error per level, inverted", 1 / fall, 1 / 20)
        check(f"{n + 1} states: projection against time iteration",
              max(r[6] for r in rows), 1e-8)
    iso = {r[0]: r for r in rows}
    rows = [(lv,) + solve_both(5, lv, X) for lv in ([3] * 5 + [1], [4] * 5 + [1])]
    print("  6 states, anisotropic: level 1 along a")
    print_rows(rows)
    for r in rows:
        mu = r[0][0]
        print(f"    levels {r[0]}: {r[1]} points against {iso[mu][1]} at isotropic level {mu}")
        check(f"6 states, levels {r[0]}: error over isotropic level {mu}'s",
              max(r[2], r[4]) / max(iso[mu][2], iso[mu][4]), 1.05)


if __name__ == "__main__":
    compare_tasmanian()
    compare_closed_form()
    if failures:
        print(f"\n{len(failures)} failure(s): {failures}")
        sys.exit(1)
    print("\nall checks passed")
