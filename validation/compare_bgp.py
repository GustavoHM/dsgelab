"""compare_bgp.py: balanced growth, against Dynare's bgp.write and its tests.

The three models come from tests/bgp of Dynare 7.2, read by `dsgelab.from_mod`. They are Solow
with technical progress and a growing population (lags only), a Ramsey model with a deterministic
trend (lags and leads) and a New Keynesian model in levels (leads only).

Part 1, the problem. Dynare writes the function bgpfun, which evaluates the model at two
consecutive periods with x_t = y g^t, and run_bgp.sh evaluates it at five fixed points. dsgelab
evaluates `dsgelab.bgp.bgp_residual` at the same points, with the timing of each. Dynare's level
is that of the lag when the model has lags (and then dsgelab's current level is y g, and that of
the auxiliary LAG1_v is y), and the current one in a model with leads only. The comparison goes
residual by residual.

Part 2, the solution. `ModFile.balanced_growth` from random starting points, as in Dynare's tests,
with their assertions. In Solow, K/Y = s gE gN / (gE gN - 1 + delta), output and capital grow at
gE gN, and the level of efficiency growth is gE (300 starting points). In Ramsey, every growth
factor is 1.02. In the New Keynesian model, levels and factors equal 1.

Part 3, trends declared in `from_equations`. Solow and Ramsey with their trends declared, as
run_bgp.sh rewrites them, and fs2000_nonstationary.mod from tests/trend_var, are written as text
equations in levels with `trend_vars=` and `deflators=`. Each is compared with Dynare, which
detrends the .mod, and with `from_mod` on the same .mod: the steady state (relative), and for Solow
and fs2000 the orthogonalised IRFs over 40 periods and the simulated paths over 200 periods of the
same shocks (each relative to the variable's largest response), and the variance matrix of the
variables (relative to its largest entry). Equal paths under the same shocks mean equal decision
rules, whatever the choice of states. In from_equations the growth factors are states, with the
innovation dated t+1, and fs2000's `del` is `delta`, since `del` is a Python keyword.

Prerequisites:
    bash fetch.sh
    bash run_bgp.sh           # exports out_bgp/<name>.mat and out_bgp/<name>_trend.mat
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from dsgelab.bgp import bgp_residual
from dsgelab.dynare import from_mod

HERE = Path(__file__).resolve().parent
FILES = {"solow": "solow-1", "ramsey": "ramsey-1", "nk": "nk-1"}


def _modfile(name):
    text = (HERE / "mods" / "bgp" / f"{FILES[name]}.mod").read_text()
    cut = min(i for i in (text.find("if ~isoctave"), text.find("verbatim;")) if i >= 0)
    return from_mod(text[:cut])


def residuals_match(name) -> bool:
    from scipy.io import loadmat
    f = HERE / "out_bgp" / f"{name}.mat"
    if not f.exists():
        sys.exit(f"missing {f}: run bash run_bgp.sh")
    ex = loadmat(f, simplify_cells=True)["ex"]
    names = [str(v).strip() for v in ex["endo_names"]]
    mf = _modfile(name)
    m = mf.model
    nd = len(names)
    lags = any(kind == "lag" for kind, _ in mf.auxiliary.values())
    err = 0.0
    for j in range(ex["Z"].shape[1]):
        y, g = ex["Z"][:nd, j], ex["Z"][nd:, j]
        X, G = np.zeros(m.n_vars), np.ones(m.n_vars)
        for i, v in enumerate(names):
            X[m.position(v)] = y[i] * g[i] if lags else y[i]
            G[m.position(v)] = g[i]
        for aux, (kind, of) in mf.auxiliary.items():
            if kind == "lag":
                i = names.index(of)
                X[m.position(aux)], G[m.position(aux)] = y[i], g[i]
        r = bgp_residual(m, mf.params, X, G)
        n = m.n_vars
        ours = np.concatenate([r[:nd], r[n:n + nd]])      # Dynare's equations, both dates
        err = max(err, float(np.max(np.abs(ours - ex["F"][:, j]))))
    ok = err < 1e-10
    print(f"  {name:7s} residuals at 5 points against bgpfun: max deviation {err:.1e}  "
          f"{'OK' if ok else 'DIVERGE'}")
    return ok


def solutions() -> bool:
    ok = True
    mf = _modfile("solow")
    p = mf.params
    gE, gN = p["EfficiencyGrowth_ss"], p["PopulationGrowth_ss"]
    theory = p["s"] * gE * gN / (gE * gN - 1 + p["delta"])
    dev = np.zeros(4)
    for seed in range(300):
        r = mf.balanced_growth(seed=seed)
        ok &= r.success and not r.degenerate
        dev = np.maximum(dev, np.abs([r.ratio("PhysicalCapitalStock", "Output") - theory,
                                      r.growth["Output"] - gE * gN,
                                      r.growth["PhysicalCapitalStock"] - gE * gN,
                                      r.levels["EfficiencyGrowth"] - gE]))
    good = bool(np.all(dev < 1e-8))
    print(f"  solow   300 starts: K/Y {dev[0]:.1e}, g_Y {dev[1]:.1e}, g_K {dev[2]:.1e}, "
          f"gE {dev[3]:.1e}  {'OK' if good else 'DIVERGE'}")
    ok &= good
    mf = _modfile("ramsey")
    r = mf.balanced_growth(seed=1)
    d = max(abs(r.growth[v] - 1.02) for v in mf.endogenous)
    print(f"  ramsey  growth factors against 1.02: {d:.1e}  {'OK' if d < 1e-6 else 'DIVERGE'}")
    ok &= d < 1e-6
    mf = _modfile("nk")
    r = mf.balanced_growth(seed=2)
    d = max(max(abs(r.levels[v] - 1), abs(r.growth[v] - 1)) for v in mf.endogenous)
    print(f"  nk      levels and factors against 1: {d:.1e}  {'OK' if d < 1e-9 else 'DIVERGE'}")
    return ok and d < 1e-9


TRENDED = {
    "solow": dict(
        equations=[
            "EfficiencyGrowth(+1)/EfficiencyGrowth_ss = "
            "(EfficiencyGrowth/EfficiencyGrowth_ss)**rho_x*exp(e_x)",
            "PopulationGrowth(+1)/PopulationGrowth_ss = "
            "(PopulationGrowth/PopulationGrowth_ss)**rho_n*exp(e_n)",
            "Output = PhysicalCapitalStock(-1)**alpha*(Efficiency*Population)**(1-alpha)",
            "PhysicalCapitalStock = (1-delta)*PhysicalCapitalStock(-1) + s*Output"],
        states=["EfficiencyGrowth", "PopulationGrowth"],
        jumps=["Output", "PhysicalCapitalStock"], shocks=["e_x", "e_n"],
        parameters=["alpha", "delta", "s", "rho_x", "rho_n", "EfficiencyGrowth_ss",
                    "PopulationGrowth_ss"],
        trend_vars={"Efficiency": "EfficiencyGrowth", "Population": "PopulationGrowth"},
        deflators={"Output": "Efficiency*Population",
                   "PhysicalCapitalStock": "Efficiency*Population"}),
    "ramsey": dict(
        equations=[
            "c + k - x**(1-alpha)*k(-1)**alpha - (1-delta)*k(-1)",
            "c**(-gam) - (1+beta)**(-1)*(alpha*x(+1)**(1-alpha)*k**(alpha-1) + 1 - delta)"
            "*c(+1)**(-gam)"],
        states=[], jumps=["c", "k"], shocks=[], parameters=["alpha", "gam", "delta", "beta"],
        trend_vars={"x": "1.02"}, deflators={"c": "x", "k": "x"}),
    "fs2000": dict(
        equations=[
            "gA(+1) = exp(gam + e_a)",
            "log(gM(+1)) = (1-rho)*log(mst) + rho*log(gM) + e_m",
            "c + k = k(-1)**alp*(A*n)**(1-alp) + (1-delta)*k(-1)",
            "P*c = M",
            "P/(c(+1)*P(+1)) = bet*P(+1)*(alp*k**(alp-1)*(A(+1)*n(+1))**(1-alp) + (1-delta))"
            "/(c(+2)*P(+2))",
            "(psi/(1-psi))*(c*P/(1-n)) = W",
            "R = P*(1-alp)*k(-1)**alp*A**(1-alp)*n**(-alp)/W",
            "W = l/n",
            "M - M(-1) + d = l",
            "1/(c*P) = bet*R/(c(+1)*P(+1))",
            "y = k(-1)**alp*(A*n)**(1-alp)"],
        states=["gA", "gM"], jumps=["k", "c", "y", "P", "W", "l", "d", "R", "n"],
        shocks=["e_a", "e_m"], parameters=["alp", "bet", "gam", "mst", "rho", "psi", "delta"],
        trend_vars={"A": "gA", "M": "gM"},
        deflators={"k": "A", "c": "A", "y": "A", "P": "M(-1)/A", "W": "M(-1)",
                   "l": "M(-1)", "d": "M(-1)"}),
}


def _paths(sol, model, scale, impulses, shocks):
    """Orthogonalised IRFs and the path under ``shocks``, in levels, by variable."""
    irfs = {j: sol.irf_impulse(imp, periods=40, percent=False) for j, imp in impulses.items()}
    k = sol.state_path(sol.Q @ shocks[0], len(shocks), shocks=shocks)
    path = sol.variables(k)
    lv = lambda v, x: np.asarray(x) * scale[model.position(v)]   # noqa: E731
    return ({(v, j): lv(v, irfs[j][v]) for j in irfs for v in irfs[j]},
            {v: lv(v, path[v]) for v in path})


def _solve_trended(model, params, guess, ex):
    import dsgelab as dsg
    ss = dsg.steady_state(model, guess, params)
    out = dict(ss=ss)
    if not hasattr(ex, "irfs"):
        return out
    sol = dsg.solve(model, params, steady=ss)
    exo = [str(e).strip() for e in np.atleast_1d(ex.exo_names)]
    order = [model.shocks.index(e) for e in exo]
    cov = np.atleast_2d(np.asarray(ex.sigma_e, float))
    L = np.linalg.cholesky(cov)
    impulses = {}
    for j, e in enumerate(exo):
        imp = np.zeros(model.n_shocks)
        imp[order] = L[:, j]
        impulses[e] = imp
    shocks = np.zeros((np.atleast_2d(ex.shocks).shape[0], model.n_shocks))
    shocks[:, order] = np.atleast_2d(ex.shocks)
    scale = sol.coords.scale
    out["irf"], out["path"] = _paths(sol, model, scale, impulses, shocks)
    full = np.zeros((model.n_shocks, model.n_shocks))
    full[np.ix_(order, order)] = cov
    V = sol.covariance(cov=full) * np.outer(scale, scale)
    out["var"] = V
    return out


def _relative(pairs):
    """max |a - b| / max(|b|), per variable, the largest over variables."""
    ruler, worst = {}, 0.0
    for key, (a, b) in pairs.items():
        v = key[0] if isinstance(key, tuple) else key
        ruler[v] = max(ruler.get(v, 0.0), float(np.max(np.abs(b))))
    for key, (a, b) in pairs.items():
        v = key[0] if isinstance(key, tuple) else key
        if ruler[v] > 1e-12:
            worst = max(worst, float(np.max(np.abs(a - b))) / ruler[v])
    return worst


def trended() -> bool:
    import dsgelab as dsg
    from scipy.io import loadmat
    ok = True
    print(f"  {'model':8s}{'against':10s}{'s.s.':>9s}{'IRFs':>9s}{'paths':>9s}{'variance':>10s}")
    for name, spec in TRENDED.items():
        f = HERE / "out_bgp" / f"{name}_trend.mat"
        if not f.exists():
            sys.exit(f"missing {f}: run bash run_bgp.sh")
        ex = loadmat(f, squeeze_me=True, struct_as_record=False)["ex"]
        endo = [str(v).strip() for v in np.atleast_1d(ex.endo_names)]
        ys = dict(zip(endo, np.atleast_1d(ex.ys).astype(float)))
        dy_par = dict(zip([str(p).strip() for p in np.atleast_1d(ex.param_names)],
                          np.atleast_1d(ex.params).astype(float)))
        mod = HERE / "out_bgp" / f"{name}_trend.mod"
        mf = from_mod(mod)
        params = dict(mf.params)
        if name == "fs2000":
            params["delta"] = params["del"]
        eq = dsg.from_equations(spec["equations"], states=spec["states"], jumps=spec["jumps"],
                                shocks=spec["shocks"], parameters=spec["parameters"],
                                trend_vars=spec["trend_vars"], deflators=spec["deflators"])
        # both start from the initval block, as Dynare does
        start = mf.full_guess()
        ours = _solve_trended(eq, params, {v: start[v] for v in endo}, ex)
        theirs = _solve_trended(mf.model, params, start, ex)
        d_par = max(abs(params[p] - v) / abs(v) for p, v in dy_par.items() if v and p in params)
        rows = []
        # against Dynare
        r = dict(ss=max(abs(ours["ss"].values[v] - ys[v]) / abs(ys[v]) for v in endo))
        if "irf" in ours:
            exo = [str(e).strip() for e in np.atleast_1d(ex.exo_names)]
            r["irf"] = _relative({(v, e): (ours["irf"][(v, e)],
                                           np.atleast_1d(getattr(ex.irfs, f"{v}_{e}"))[:40])
                                  for v in endo for e in exo if hasattr(ex.irfs, f"{v}_{e}")})
            path = np.atleast_2d(ex.path)
            T = len(next(iter(ours["path"].values())))
            r["path"] = _relative({v: (ours["path"][v], path[i, 1:T + 1] - ys[v])
                                   for i, v in enumerate(endo)})
            idx = [eq.position(v) for v in endo]
            V = np.atleast_2d(np.asarray(ex.variance, float))
            r["var"] = float(np.max(np.abs(ours["var"][np.ix_(idx, idx)] - V)) / np.max(np.abs(V)))
        rows.append(("Dynare", r))
        # against from_mod on the same .mod
        r = dict(ss=max(abs(ours["ss"].values[v] - theirs["ss"].values[v]) / abs(ys[v])
                        for v in endo))
        if "irf" in ours:
            r["irf"] = _relative({key: (ours["irf"][key], theirs["irf"][key])
                                  for key in ours["irf"] if key[0] in endo})
            r["path"] = _relative({v: (ours["path"][v], theirs["path"][v]) for v in endo})
            idx = [eq.position(v) for v in endo]
            jdx = [mf.model.position(v) for v in endo]
            r["var"] = float(np.max(np.abs(ours["var"][np.ix_(idx, idx)]
                                           - theirs["var"][np.ix_(jdx, jdx)]))
                             / np.max(np.abs(theirs["var"][np.ix_(jdx, jdx)])))
        rows.append(("from_mod", r))
        for who, r in rows:
            good = d_par < 1e-12 and all(x < 1e-10 for x in r.values())
            ok &= good
            cells = "".join(f"{r[k]:9.1e}" if k in r else f"{'-':>9s}"
                            for k in ("ss", "irf", "path"))
            var = f"{r['var']:10.1e}" if "var" in r else f"{'-':>10s}"
            print(f"  {name:8s}{who:10s}{cells}{var}  {'OK' if good else 'DIVERGE'}")
    # Ramsey's detrended steady state in closed form: 1 + beta = g^-gam (alpha g^(1-alpha)
    # k^(alpha-1) + 1 - delta), c = (k/g)^alpha + (1-delta) k/g - k
    p, g = from_mod(HERE / "out_bgp" / "ramsey_trend.mod").params, 1.02
    spec = TRENDED["ramsey"]
    eq = dsg.from_equations(spec["equations"], states=spec["states"], jumps=spec["jumps"],
                            shocks=[], parameters=spec["parameters"],
                            trend_vars=spec["trend_vars"], deflators=spec["deflators"])
    s = dsg.steady_state(eq, {"c": 1.0, "k": 10.0}, p).values
    k = (((1 + p["beta"]) * g ** p["gam"] - 1 + p["delta"]) / (p["alpha"] * g ** (1 - p["alpha"]))
         ) ** (1 / (p["alpha"] - 1))
    c = (k / g) ** p["alpha"] + (1 - p["delta"]) * k / g - k
    d = max(abs(s["k"] - k) / k, abs(s["c"] - c) / c)
    print(f"  ramsey  closed form of the detrended steady state: {d:.1e}  "
          f"{'OK' if d < 1e-12 else 'DIVERGE'}")
    return ok and d < 1e-12


def main() -> int:
    print("the problem (bgp.write)")
    ok = all([residuals_match(n) for n in FILES])
    print("the solution (assertions of Dynare's tests)")
    ok &= solutions()
    print("trends declared in from_equations")
    ok &= trended()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
