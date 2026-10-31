"""compare_value_iteration.py: value function iteration against QuantEcon, dolo and the closed form.

  1. QuantEcon 0.13.1's DiscreteDP on the discretised stochastic growth model: capital on a grid,
     next period's capital chosen on the same grid, productivity a Rouwenhorst chain. DiscreteDP
     solves it exactly by policy iteration. dsgelab solves the same problem with choices= and
     Howard's steps; on this problem its linear interpolation is exact, since next period's
     capital is always a grid point. The values must agree within the tolerance's bound,
     beta tol / (1 - beta), and the policies must be the same.
  2. dolo 0.4.9.20's value_iteration on its rbc_mc model (dolo/run_dolo_vfi.py, in the
     environment of dolo/setup.sh): capital on 50 points, productivity a two-state Markov chain,
     labour and investment chosen within their bounds, the value interpolated by cubic splines,
     whose second derivative is zero at the ends, with linear extrapolation, as in dsgelab. dolo
     declares the felicity as c^(1-sigma)/(1-sigma), which with sigma = 5 is also finite for
     negative consumption; dolo's maximizer starts near the solution and does not reach that
     region, while dsgelab's searches the whole box, so the program here sets the reward to
     -inf where consumption is not positive. The value is compared relative to its size, and the
     controls in levels: dolo maximizes by L-BFGS-B with gradients by forward differences, which
     leave its controls some 1e-6 from the maximum. The gradient of the maximized objective at
     each policy is printed to show which side the difference comes from.
  3. Stochastic growth with log utility and full depreciation, whose value is known:
     V(k, z) = E + alpha/(1 - alpha beta) log k + H(z), with H solving a linear system in the
     chain's transition matrix. Linear and cubic interpolation, on grids of 10 to 80 capital
     points, at the grid points. The linear error falls as the square of the spacing, and the
     cubic one faster.

Prerequisites:
    bash validation/dolo/setup.sh
    validation/dolo/.venv/bin/python validation/dolo/run_dolo_vfi.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from quantecon.markov import DiscreteDP

import dsgelab as dsg

HERE = Path(__file__).resolve().parent
ALPHA, BETA = 0.36, 0.95
KSS = (ALPHA * BETA) ** (1 / (1 - ALPHA))
failures: list[str] = []


def check(label: str, value: float, limit: float) -> None:
    status = "ok" if value <= limit else "FAIL"
    if status == "FAIL":
        failures.append(label)
    print(f"  {status:4s} {label:58s} {value:9.2e}  (limit {limit:.0e})")


def check_at_least(label: str, value: float, floor: float) -> None:
    status = "ok" if value >= floor else "FAIL"
    if status == "FAIL":
        failures.append(label)
    print(f"  {status:4s} {label:58s} {value:9.2f}  (at least {floor:g})")


def compare_quantecon() -> None:
    import quantecon
    print(f"1. QuantEcon {quantecon.__version__} DiscreteDP, policy iteration")
    tol = 1e-12
    for nz, nk in ((3, 150), (7, 200)):
        chain = dsg.rouwenhorst(0.9, 0.02, nz)
        K = np.linspace(0.5 * KSS, 1.5 * KSS, nk)
        s_idx, a_idx, R, Q = [], [], [], []
        for iz in range(nz):
            for ik in range(nk):
                y = np.exp(chain.nodes[iz]) * K[ik] ** ALPHA
                for ia in np.flatnonzero(K < y):
                    s_idx.append(iz * nk + ik)
                    a_idx.append(ia)
                    R.append(np.log(y - K[ia]))
                    q = np.zeros(nz * nk)
                    q[np.arange(nz) * nk + ia] = chain.transition[iz]
                    Q.append(q)
        ddp = DiscreteDP(np.array(R), np.array(Q), BETA, np.array(s_idx), np.array(a_idx))
        t0 = time.perf_counter()
        res = ddp.solve(method="policy_iteration")
        t_qe = time.perf_counter() - t0
        prog = dsg.DynamicProgram(
            states=["k", "z"], controls=["kp"],
            reward=lambda s, x, p: np.log(np.exp(s.z) * s.k ** ALPHA - x.kp),
            transition=lambda s, x, e, p: {"k": x.kp}, discount=BETA)
        t0 = time.perf_counter()
        sol = dsg.value_iteration(prog, {}, grid={"k": K}, markov={"z": chain},
                                  choices={"kp": K}, tol=tol, howard=100)
        t_ours = time.perf_counter() - t0
        print(f" {nz} productivity nodes, {nk} capital points: dsgelab {sol.iterations} "
              f"iterations, {t_ours:.2f} s; QuantEcon {res.num_iter} iterations, {t_qe:.2f} s")
        dv = float(np.max(np.abs(sol.values - res.v.reshape(nz, nk))))
        same = np.array_equal(sol.policy[..., 0], K[res.sigma.reshape(nz, nk)])
        check(f"{nz} x {nk}: largest difference in the value", dv, BETA * tol / (1 - BETA))
        print(f"       the policies are {'the same' if same else 'DIFFERENT'} at every point")
        if not same:
            failures.append(f"DiscreteDP {nz} x {nk}: policies differ")


def rbc_mc_program():
    def reward(s, x, p):
        y = np.exp(s.z) * s.k ** p["alpha"] * x.n ** (1 - p["alpha"])
        c = np.where(y - x.i > 0, y - x.i, np.nan)
        return (c ** (1 - p["sigma"]) / (1 - p["sigma"])
                - p["chi"] * x.n ** (1 + p["eta"]) / (1 + p["eta"]))

    return dsg.DynamicProgram(
        states=["k", "z"], controls=["n", "i"], reward=reward,
        transition=lambda s, x, e, p: {"k": (1 - p["delta"]) * s.k + x.i},
        discount="beta", bounds={"n": (0.01, 1.0), "i": (0.0, 1.0)}, name="rbc_mc")


def compare_dolo() -> None:
    path = HERE / "out_value_iteration" / "dolo.json"
    if not path.exists():
        print(f"{path} is missing: run validation/dolo/setup.sh and run_dolo_vfi.py first")
        failures.append("dolo reference missing")
        return
    data = json.loads(path.read_text())
    meta = data["meta"]
    case = data["cases"]["rbc_mc"]
    run = case["run"]
    print(f"2. dolo {meta['dolo']} (interpolation {meta['interpolation']}), value_iteration")
    values = np.asarray(case["markov"]["values"])
    chain = dsg.MarkovChain(values[:, 0], np.asarray(case["markov"]["transitions"]))
    K = np.asarray(run["axes"][0])
    t0 = time.perf_counter()
    sol = dsg.value_iteration(rbc_mc_program(), case["parameters"], grid={"k": K},
                              markov={"z": chain}, interpolation="cubic", tol=1e-10, howard=200)
    elapsed = time.perf_counter() - t0
    print(f" rbc_mc (cubic): dsgelab {sol.iterations} iterations from zero, {elapsed:.2f} s; "
          f"dolo {run['iterations']} iterations from its time-iteration policy, "
          f"{run['seconds']:.2f} s")
    if not sol.converged:
        failures.append("rbc_mc: dsgelab did not converge")
    mids = 0.5 * (K[1:] + K[:-1])
    dv_n = dv_m = dx_n = dx_m = 0.0
    for m, z in enumerate(chain.nodes):
        at_n = np.column_stack([K, np.full(K.size, z)])
        at_m = np.column_stack([mids, np.full(mids.size, z)])
        vn, vm = np.asarray(run["value_at_nodes"][m]), np.asarray(run["value_at_midpoints"][m])
        dv_n = max(dv_n, float(np.max(np.abs(sol.value(at_n) / vn - 1))))
        dv_m = max(dv_m, float(np.max(np.abs(sol.value(at_m) / vm - 1))))
        dx_n = max(dx_n, float(np.max(np.abs(sol.controls(at_n)
                                             - np.asarray(run["controls_at_nodes"][m])))))
        dx_m = max(dx_m, float(np.max(np.abs(sol.controls(at_m)
                                             - np.asarray(run["controls_at_midpoints"][m])))))
    check("rbc_mc: value, relative difference at the grid points", dv_n, 1e-9)
    check("rbc_mc: value, relative difference between them", dv_m, 1e-9)
    check("rbc_mc: controls, largest difference at the grid points", dx_n, 1e-5)
    check("rbc_mc: controls, largest difference between them", dx_m, 1e-5)
    # the gradient of the maximized objective, at the interior points of each policy
    pb = sol._problem
    G = pb.coefficients(sol.values[..., None])
    node, cont = pb.grid_nodes, pb.grid_cont
    S = pb.full_states(node, cont)
    lo, hi = pb.bounds_at(S)
    for label, X in (("dsgelab", sol.policy.reshape(-1, 2)),
                     ("dolo", np.vstack([np.asarray(c) for c in run["controls_at_nodes"]]))):
        def q(Y):
            return pb.choice_value(G, node, S, Y)
        g, _ = pb._derivatives(q, X, q(X), lo, hi)
        inside = np.all((X - lo > 1e-6) & (hi - X > 1e-6), axis=1)
        print(f"       gradient of the objective at {label}'s controls, interior points: "
              f"max {np.abs(g[inside]).max():.1e}")


def compare_closed_form() -> None:
    print("3. stochastic growth with log utility and full depreciation, against the closed form")
    chain = dsg.rouwenhorst(0.9, 0.02, 5)
    F = ALPHA / (1 - ALPHA * BETA)
    H = np.linalg.solve(np.eye(chain.n) - BETA * chain.transition,
                        chain.nodes / (1 - ALPHA * BETA))
    E = (np.log(1 - ALPHA * BETA) + BETA * F * np.log(ALPHA * BETA)) / (1 - BETA)
    prog = dsg.DynamicProgram(
        states=["k", "z"], controls=["c"], reward=lambda s, x, p: np.log(x.c),
        transition=lambda s, x, e, p: {"k": np.exp(s.z) * s.k ** ALPHA - x.c}, discount=BETA,
        bounds={"c": (1e-9, lambda s, p: np.exp(s.z) * s.k ** ALPHA - 1e-9)})
    for kind, limit in (("linear", 2e-4), ("cubic", 1e-8)):
        errors = []
        for n in (10, 20, 40, 80):
            sol = dsg.value_iteration(prog, {}, grid={"k": (0.5 * KSS, 1.5 * KSS, n)},
                                      markov={"z": chain}, interpolation=kind, tol=1e-12,
                                      howard=50)
            k = sol.grid.points[:, 0]
            exact = E + F * np.log(k)[None, :] + H[:, None]
            errors.append(float(np.max(np.abs(sol.values - exact))))
        print(f"  {kind}: largest error of V at the grid points with 10, 20, 40, 80 capital "
              "points: " + ", ".join(f"{e:.2e}" for e in errors))
        check(f"{kind}, 80 points: largest error of V", errors[-1], limit)
        ratios = [a / b for a, b in zip(errors, errors[1:])]
        check_at_least(f"{kind}: smallest error ratio when the spacing halves", min(ratios), 3.0)


if __name__ == "__main__":
    compare_quantecon()
    compare_dolo()
    compare_closed_form()
    if failures:
        print(f"\n{len(failures)} failure(s): {failures}")
        sys.exit(1)
    print("\nall checks passed")
