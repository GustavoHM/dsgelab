"""compare_gsa.py: global sensitivity, against Dynare's sensitivity toolbox and SALib.

  1. Stability mapping (Dynare's sensitivity(stab=1)) of Lubik and Schorfheide (2003), from Dynare's
     tests/gsa/ls2003a.mod, on three samples of 1024 draws: from the prior, uniform over the prior's
     range, and from the prior with rho_A uniform on [0.5, 1.1]. dsgelab classifies the draws
     Dynare made, and the comparison covers the class of every draw (unique stable solution,
     indeterminate, explosive, no solution), against the verdict of Dynare's solver at that draw
     and against the lists of its sensitivity toolbox, whose rule sorts the eigenvalues by signed
     value when they are all real (the same rule by modulus gives dsgelab's classes, except at
     the rank failures, which the rule cannot detect and Dynare's solver calls indeterminate,
     info 5); and, for
     each of Dynare's Monte Carlo filtering analyses, on Dynare's own groups, the Smirnov
     statistic and p-value of every parameter and the correlations and their p-values inside
     each group.
  2. Sobol indices against SALib 1.6: the design (SALib's sample.sobol with the same seed draws the
     same points) and the estimators (SALib's analyze.sobol on dsgelab's outputs gives the same
     first-order and total indices), on the Ishigami function, whose indices are known in closed
     form, and on impulse responses and moments of ls2003.

Prerequisites:
    bash fetch.sh
    bash run_gsa.sh
    pip install SALib
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import dsgelab as dsg
from dsgelab.dynare import from_mod

HERE = Path(__file__).resolve().parent
OUT = HERE / "out_gsa"
CASES = ("ls2003_prior", "ls2003_range", "ls2003_explosive")


def _load(name):
    from scipy.io import loadmat
    f = OUT / f"{name}.mat"
    return None if not f.exists() else loadmat(f, squeeze_me=True, struct_as_record=False)["ex"]


def _idx(x):
    return np.atleast_1d(np.asarray(x, dtype=int)) - 1


def _dynare_classes(ex, n):
    """Dynare's classes, from its solver's verdict at each draw (info) and from its GSA rule.

    stability_mapping.m labels a draw indeterminate when the eigenvalue number nspred + 1 of
    sort(eigval) is stable. When every eigenvalue is real, sort orders them by signed value and
    a large negative (unstable) root comes before the stable ones. ``modulus`` applies the same
    rule to the eigenvalues sorted by modulus.
    """
    info = np.atleast_1d(ex.infox).astype(int)
    solver = np.full(n, "no_solution", dtype=object)
    solver[info == 0] = "determinate"
    solver[info == 3] = "explosive"
    solver[(info == 4) | (info == 5)] = "indeterminate"
    gsa = np.full(n, "no_solution", dtype=object)
    gsa[_idx(ex.istable)] = "determinate"
    gsa[_idx(ex.iindeterm)] = "indeterminate"
    gsa[_idx(ex.ixun)] = "explosive"
    E = np.atleast_2d(ex.egg)
    ns = int(ex.nspred)
    forward = int(ex.nboth) or int(ex.nfwrd)
    modulus = gsa.copy()
    for j in np.where((gsa == "indeterminate") | (gsa == "explosive"))[0]:
        stable = forward and np.sort(np.abs(E[:, j]))[ns] <= 1 + 1e-6
        modulus[j] = "indeterminate" if stable else "explosive"
    return solver.astype(str), gsa.astype(str), modulus.astype(str)


def check_stability(name) -> bool:
    ex = _load(name)
    if ex is None:
        print(f"{name}: run run_gsa.sh")
        return False
    mf = from_mod(HERE / "work_gsa" / f"{name}.mod")
    nshock = int(ex.nshock)
    raw = [str(n).strip() for n in np.atleast_1d(ex.names)]
    names = [f"stderr {n}" if i < nshock else n for i, n in enumerate(raw)]
    X = np.atleast_2d(ex.lpmat)
    st = mf.stability_map(points=X, names=names)
    solver, gsa, modulus = _dynare_classes(ex, X.shape[0])
    # dsgelab's rank failures are Dynare's info 5, which its GSA rule sorts by eigenvalue
    d_solver = int(np.sum(st.status != solver))
    d_gsa = int(np.sum(st.status != gsa))
    rank = np.atleast_1d(ex.infox).astype(int) == 5
    off = st.status != modulus
    d_mod = int(np.sum(off))
    rank_only = bool(np.all(rank[off]))          # the rank failures, which the rule cannot see
    sh = st.shares
    # Dynare's Monte Carlo filtering, on its own groups: Smirnov and correlations
    params = names[nshock:]
    d_ks = d_p = d_c = d_cp = 0.0
    analyses = same_groups = 0
    for kind, field in (("determinate", "stable"), ("indeterminate", "indeterminate"),
                        ("explosive", "explosive")):
        g = getattr(ex, field)
        if not hasattr(g, "proba"):
            continue
        analyses += 1
        f = dsg.monte_carlo_filtering(X[:, nshock:], _idx(g.ib), params, nonbehaviour=_idx(g.inb))
        ours = st.filtering(kind)
        same_groups += bool(np.array_equal(ours.behaviour, f.behaviour)
                            and np.array_equal(ours.nonbehaviour, f.nonbehaviour))
        d_ks = max(d_ks, float(np.max(np.abs(f.d - np.atleast_1d(g.dproba)))))
        d_p = max(d_p, float(np.max(np.abs(f.p - np.atleast_1d(g.proba)))))
        if hasattr(g, "c_beha"):
            pos = {n: i for i, n in enumerate(params)}
            for grp, c, p in (("behaviour", g.c_beha, g.p_beha),
                              ("nonbehaviour", g.c_nobeha, g.p_nobeha)):
                for a, b, r, pv in f.correlations(grp, pvalue=1.1):
                    i, j = pos[a], pos[b]
                    d_c = max(d_c, abs(r - c[i, j]))
                    d_cp = max(d_cp, abs(pv - p[i, j]))
    print(f"  {name:<17} {100 * sh['determinate']:5.1f}% stable  {100 * sh['indeterminate']:4.1f}%"
          f" indeterminate  {100 * sh['explosive']:4.1f}% explosive")
    print(f"     classes against Dynare's solver (info): differ in {d_solver};"
          f" against its GSA lists: {d_gsa}; with its rule by modulus: {d_mod}"
          f"{', all rank failures (info 5)' if d_mod and rank_only else ''}")
    print(f"     filtering on Dynare's groups ({analyses} analyses, {same_groups} the same as dsgelab's):"
          f" Smirnov d {d_ks:.1e}, p {d_p:.1e}; correlations {d_c:.1e}, p {d_cp:.1e}")
    for kind in ("determinate", "indeterminate", "explosive"):
        f = st.filtering(kind)
        if f.n_behaviour and f.n_nonbehaviour:
            print(f"     {f.title:<26} driven by "
                  + ", ".join(f"{n} (d {f.d[f.names.index(n)]:.3f})" for n in f.significant))
    return (d_solver == 0 and rank_only and d_ks < 1e-14 and d_p < 1e-12 and d_c < 1e-12
            and d_cp < 1e-12)


def _salib():
    try:
        from SALib.analyze import sobol as analyze
        from SALib.sample import sobol as sample
    except ImportError:
        return None
    return sample, analyze


def compare_sobol(label, run, parameters, n, bounds, salib) -> bool:
    sample, analyze = salib
    res = dsg.sobol(run, parameters, n=n, bootstrap=0)
    k = len(parameters)
    problem = {"num_vars": k, "names": list(parameters), "bounds": bounds}
    X = sample.sample(problem, n, calc_second_order=False, scramble=True, seed=res.seed)
    ours = np.empty((n, k + 2, k))
    ours[:, 0], ours[:, -1] = res.A, res.B
    for i in range(k):
        AB = res.A.copy()
        AB[:, i] = res.B[:, i]
        ours[:, 1 + i] = AB
    design = float(np.max(np.abs(ours.reshape(-1, k) - X)))
    idx = res.indices()
    worst = 0.0
    for out in res.outputs:
        v = res.values[out]
        Y = np.column_stack([v["A"], v["AB"], v["B"]]).ravel()
        Si = analyze.analyze(problem, Y, calc_second_order=False, num_resamples=2,
                             print_to_console=False, seed=1)
        worst = max(worst, float(np.max(np.abs(idx[out]["S1"] - Si["S1"]))),
                    float(np.max(np.abs(idx[out]["ST"] - Si["ST"]))))
    print(f"  {label:<28} design {design:.1e}   indices {worst:.1e}   "
          f"({len(res.outputs)} outputs, {n * (k + 2)} evaluations)")
    return design < 1e-14 and worst < 1e-12, res


def check_sobol() -> bool:
    salib = _salib()
    if salib is None:
        print("SALib is not installed: pip install SALib")
        return False
    pi = np.pi
    ishigami = {f"x{i}": dsg.Uniform(-pi, pi) for i in (1, 2, 3)}

    def f(v):
        return (np.sin(v["x1"]) + 7.0 * np.sin(v["x2"]) ** 2
                + 0.1 * v["x3"] ** 4 * np.sin(v["x1"]))

    print("\nSobol indices against SALib")
    ok1, res = compare_sobol("Ishigami", f, ishigami, 4096, [[-pi, pi]] * 3, salib)
    s1, st = res.indices()["result"]["S1"], res.indices()["result"]["ST"]
    exact_s1, exact_st = np.array([0.3139, 0.4424, 0.0]), np.array([0.5576, 0.4424, 0.2437])
    print(f"     S1 {np.round(s1, 4)}  (exact {exact_s1})   ST {np.round(st, 4)}  (exact {exact_st})")
    ok_exact = np.max(np.abs(s1 - exact_s1)) < 0.02 and np.max(np.abs(st - exact_st)) < 0.02

    mf = from_mod(HERE / "work_gsa" / "ls2003_prior.mod")
    run = dsg.solution_outputs(mf.model, dict(mf.params), cov=mf._shock_cov(),
                               steady_fn=lambda p: mf.steady_state(params=p),
                               irf=[("y", "e_R", [1, 4]), ("pie", "e_R", 1)],
                               moments=[("y", "y", 0), ("pie", "pie", 1)])
    pars = {"psi1": dsg.Uniform(1.1, 2.5), "rho_R": dsg.Uniform(0.2, 0.8),
            "tau": dsg.Uniform(0.2, 0.8)}
    ok2, res2 = compare_sobol("ls2003, IRFs and moments", run, pars, 256,
                              [[1.1, 2.5], [0.2, 0.8], [0.2, 0.8]], salib)
    idx = res2.indices()
    for out in res2.outputs:
        print(f"     {out:<14} S1 {np.round(idx[out]['S1'], 3)}   ST {np.round(idx[out]['ST'], 3)}")
    return ok1 and ok_exact and ok2


if __name__ == "__main__":
    print("Stability mapping against Dynare's sensitivity(stab=1), ls2003a")
    ok = [check_stability(c) for c in CASES]
    ok.append(check_sobol())
    print("\n" + ("OK" if all(ok) else "DIVERGE"))
    sys.exit(0 if all(ok) else 1)
