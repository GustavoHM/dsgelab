"""compare_estimation.py: estimation, smoothing, decomposition and forecasting, against Dynare.

The central case is fs2000.mod from Dynare's examples/ folder (Schorfheide, 2000), with model,
priors and data as they are, read by `from_mod` (the `estimated_params`, the `varobs` and the
`loglinear` and `logdata` options become an `EstimationProblem`). The data are the ones Dynare
assembled from fs2000_data.m, exported by run_estimation.sh.

  1. At the initial point (the prior means, as Dynare defines it): likelihood, prior density,
     posterior kernel, smoothed variables and shocks, the 8-period forecast with 90% bands,
     shock_decomposition and realtime_shock_decomposition.
  2. The posterior mode: the value of the kernel (dsgelab's has to be greater or equal), the
     parameters, and the Laplace approximation. With the covariance that Dynare ITSELF reports,
     the formula has to give Dynare's number. With dsgelab's Hessian, the difference is that
     between the two numerical Hessians.
  3. Metropolis-Hastings, 2 chains of 10,000 and mh_jscale 0.8, as in Dynare: the posterior
     means, in units of the joint Monte Carlo error of the two chains, and the modified harmonic
     mean. (--no-mh, or its alias --sem-mh, skips this part, which takes a few minutes.)
  4. The diffuse filter: estimation/unitroot.mod, written here, with an observed price level.
     Likelihood at two points, and the smoother.
  5. The conditional forecast: Dynare's tests/conditional_forecasts/1/fs2000_cal.mod.

Prerequisites:
    bash fetch.sh
    bash run_estimation.sh
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

import dsgelab as dsg
from dsgelab.dynare import from_mod
from dsgelab.estimation import metropolis_hastings, posterior_mode
from dsgelab.kalman import forecast, realtime_shock_decomposition, shock_decomposition

HERE = Path(__file__).resolve().parent
OUT = HERE / "out_estimation"


def _load(name):
    from scipy.io import loadmat
    f = OUT / f"{name}.mat"
    return None if not f.exists() else loadmat(f, squeeze_me=True, struct_as_record=False)["ex"]


def _names(x):
    return [str(v).strip() for v in np.atleast_1d(x)]


def _fs2000(ex):
    varobs = _names(ex.varobs)
    data = {v: np.asarray(ex.data)[:, i] for i, v in enumerate(varobs)}
    mf = from_mod(HERE / "mods" / "examples" / "fs2000.mod")
    return mf, mf.estimation_problem(data)


def _ok(flag):
    return "OK" if flag else "DIVERGE"


def check_point() -> bool:
    ex = _load("fs2000_calib")
    if ex is None:
        print("fs2000_calib: run run_estimation.sh")
        return False
    mf, pb = _fs2000(ex)
    names = _names(ex.estim_names)
    dyn = dict(zip(names, np.atleast_1d(ex.calib_point)))
    th = np.array([dyn[n] for n in pb.names])
    same_start = np.allclose(th, pb.start())
    ll, lp = pb.log_likelihood(th), pb.log_prior(th)
    d_ll = abs(ll - float(ex.loglik))
    d_lp = abs(lp - (float(ex.logpost) - float(ex.loglik)))
    sol, space = pb.solve(th)
    sm = pb.smoother(th)
    hat = sol.variables(sm.states)
    endo = _names(ex.endo_names)
    exo = _names(ex.exo_names)
    d_sv = 0.0
    for v in ex.smoothed_vars._fieldnames:
        if v in mf.endogenous:
            j = sol.model.position(v)
            ours = math.log(sol.coords.center[j]) + hat[v]
            d_sv = max(d_sv, float(np.max(np.abs(ours - np.asarray(getattr(ex.smoothed_vars, v))))))
    d_sh = 0.0
    for e in ex.smoothed_shocks._fieldnames:
        j = sol.model.position(f"SHK_{e}")
        ours = hat[f"SHK_{e}"] * sol.coords.scale[j]
        d_sh = max(d_sh, float(np.max(np.abs(ours - np.asarray(getattr(ex.smoothed_shocks, e))))))
    look = ["gp_obs", "gy_obs", "c", "y"]
    f = forecast(space, pb.Y, 8, variables=look, logs=look, coverage=0.9)
    d_fm = d_fb = 0.0
    for v in look:
        d_fm = max(d_fm, float(np.max(np.abs(f.mean[v] - np.asarray(getattr(ex.forecast.Mean, v))))))
        d_fb = max(d_fb, float(np.max(np.abs(f.lower[v] - np.asarray(getattr(ex.forecast.HPDinf, v))))),
                   float(np.max(np.abs(f.upper[v] - np.asarray(getattr(ex.forecast.HPDsup, v))))))
    z = np.asarray(ex.shock_decomposition)
    dec = shock_decomposition(sm, variables=look, initial_shock_states=mf.shock_states)
    pool = np.asarray(ex.realtime_pool)
    rt = realtime_shock_decomposition(space, pb.Y, variables=look)
    d_dec = d_rt = 0.0
    ne = len(exo)
    for v in look:
        i = endo.index(v)
        for m, e in enumerate(exo):
            d_dec = max(d_dec, float(np.max(np.abs(dec["shocks"][e][v] - z[i, m]))))
            d_rt = max(d_rt, float(np.max(np.abs(rt["shocks"][e][v] - pool[i, m]))))
        d_dec = max(d_dec, float(np.max(np.abs(dec["initial"][v] - z[i, ne]))),
                    float(np.max(np.abs(dec["total"][v] - z[i, ne + 1]))))
        d_rt = max(d_rt, float(np.max(np.abs(rt["initial"][v] - pool[i, ne]))),
                   float(np.max(np.abs(rt["total"][v] - pool[i, ne + 1]))))
    print("fs2000 — at the initial point (prior means)")
    print(f"  initial point equal to Dynare's              {same_start}")
    print(f"  likelihood       {ll:.10f}   Dynare {float(ex.loglik):.10f}   {d_ll:.1e}")
    print(f"  log prior        {lp:.10f}   Dynare {float(ex.logpost) - float(ex.loglik):.10f}   {d_lp:.1e}")
    print(f"  smoothed variables {d_sv:.1e}   smoothed shocks {d_sh:.1e}")
    print(f"  forecast: mean {d_fm:.1e}, 90% bands {d_fb:.1e}")
    print(f"  shock_decomposition {d_dec:.1e}   realtime_shock_decomposition {d_rt:.1e}")
    return (same_start and d_ll < 1e-8 and d_lp < 1e-10 and d_sv < 1e-12 and d_sh < 1e-12
            and d_fm < 1e-12 and d_fb < 1e-7 and d_dec < 1e-12 and d_rt < 1e-12)


def check_mode():
    ex = _load("fs2000_mode")
    if ex is None:
        print("fs2000_mode: run run_estimation.sh")
        return False, None, None
    exc = _load("fs2000_calib")
    mf, pb = _fs2000(exc)
    m = posterior_mode(pb)
    names = _names(ex.estim_names)
    dmode = dict(zip(names, np.atleast_1d(ex.mode)))
    V = np.atleast_2d(np.asarray(ex.variance))
    k = V.shape[0]
    lap_formula = float(ex.log_density) + 0.5 * k * math.log(2 * math.pi) \
        + 0.5 * np.linalg.slogdet(V)[1]
    lap_dy = float(ex.marginal.LaplaceApproximation)
    print("\nfs2000 — posterior mode")
    print(f"  posterior kernel     dsgelab {m.log_posterior:.8f}   Dynare {float(ex.log_density):.8f}"
          f"   ({'greater or equal' if m.log_posterior >= float(ex.log_density) - 1e-9 else 'SMALLER'})")
    for n in pb.names:
        print(f"     {n:<11} dsgelab {m.values[n]:>10.6f} ± {m.std[n]:.6f}   "
              f"Dynare {dmode[n]:>10.6f}")
    print(f"  Laplace with Dynare's covariance    {lap_formula:.8f}   Dynare {lap_dy:.8f}"
          f"   {abs(lap_formula - lap_dy):.1e}")
    print(f"  Laplace with dsgelab's Hessian      {m.laplace:.6f}   (difference {m.laplace - lap_dy:+.3f}:"
          f" the two numerical Hessians)")
    rel = max(abs(m.values[n] - dmode[n]) / m.std[n] for n in pb.names)
    ok = (m.log_posterior >= float(ex.log_density) - 1e-9 and abs(lap_formula - lap_dy) < 1e-8
          and rel < 0.05 and abs(m.laplace - lap_dy) < 0.1)
    return ok, pb, m


def check_mh(pb, mode) -> bool:
    ex = _load("fs2000_mh")
    if ex is None:
        print("fs2000_mh: run run_estimation.sh")
        return False
    mh = metropolis_hastings(pb, mode, draws=10000, chains=2, scale=0.8, seed=1)
    pm = ex.posterior_mean
    dyn = {}
    for grp in pm._fieldnames:
        g = getattr(pm, grp)
        if hasattr(g, "_fieldnames"):
            for k in g._fieldnames:
                dyn[(grp, k)] = float(getattr(g, k))
    ess = mh.ess()
    X = mh.draws
    ours = mh.mean()
    print("\nfs2000 — Metropolis-Hastings, 2 x 10,000, mh_jscale 0.8")
    print(f"  acceptance {np.round(mh.acceptance, 3)}   max R-hat {max(mh.rhat().values()):.4f}")
    worst = 0.0
    for i, n in enumerate(mh.names):
        key = ("shocks_std", n[7:]) if n.startswith("stderr ") else ("parameters", n)
        se = X[:, i].std() / math.sqrt(ess[n])
        zval = (ours[n] - dyn[key]) / (math.sqrt(2) * se)
        worst = max(worst, abs(zval))
        print(f"     {n:<11} dsgelab {ours[n]:>10.6f}   Dynare {dyn[key]:>10.6f}   z {zval:+.2f}")
    mhm = mh.log_marginal_mhm()
    mhm_dy = float(ex.marginal.ModifiedHarmonicMean)
    print(f"  modified harmonic mean  dsgelab {mhm:.4f}   Dynare {mhm_dy:.4f}")
    return worst < 4.0 and abs(mhm - mhm_dy) < 0.3


def check_unitroot() -> bool:
    ex = _load("unitroot")
    if ex is None:
        print("unitroot: run run_estimation.sh")
        return False
    D = np.asarray(ex.data)
    varobs = _names(ex.varobs)
    data = {v: D[:, i] for i, v in enumerate(varobs)}
    mf = from_mod(HERE / "estimation" / "unitroot.mod")
    pb = mf.estimation_problem(data, qz_criterium=1 + 1e-6)
    names = _names(ex.names)
    d = 0.0
    for xk, llk in (("x1", "ll1"), ("x2", "ll2")):
        dyn = dict(zip(names, np.atleast_1d(getattr(ex, xk))))
        th = np.array([dyn[n] for n in pb.names])
        d = max(d, abs(pb.log_likelihood(th) - float(getattr(ex, llk))))
    th = np.array([dict(zip(names, np.atleast_1d(ex.x1)))[n] for n in pb.names])
    sol, _ = pb.solve(th)
    sm = pb.smoother(th)
    hat = sol.variables(sm.states)
    d_s = 0.0
    for v in ("y", "p", "pi"):
        j = sol.model.position(v)
        d_s = max(d_s, float(np.max(np.abs(sol.coords.center[j] + sol.coords.scale[j] * hat[v]
                                             - np.asarray(getattr(ex.sv, v))))))
    print("\nDiffuse filter — estimation/unitroot.mod")
    print(f"  likelihood at two points  {d:.1e}   smoother {d_s:.1e}   "
          f"(diffuse periods: {pb.filter(th).diffuse_periods})")
    return d < 1e-8 and d_s < 1e-12


def check_conditional_forecast() -> bool:
    ex = _load("condfc")
    if ex is None:
        print("condfc: run run_estimation.sh")
        return False
    mf = from_mod(HERE / "mods" / "forecast" / "fs2000_cal.mod")
    ss = mf.steady_state()
    sol = dsg.solve(mf.model, mf.params, steady=ss)
    H, nan = 40, np.nan
    cf = sol.conditional_forecast(
        {"gy_obs": np.r_[[0.01, -0.02, 0, 0, 0], [nan] * (H - 5)],
         "gp_obs": np.r_[[0.05] * 5, [nan] * (H - 5)]},
        controlled=["e_a", "e_m"], levels=True)
    d = 0.0
    for v in ex.cond._fieldnames:
        if v in mf.endogenous:
            dy = np.asarray(getattr(ex.cond, v), float)
            d = max(d, float(np.max(np.abs(cf[v] - dy))) / max(1.0, float(np.max(np.abs(dy)))))
    print("\nConditional forecast — tests/conditional_forecasts/1/fs2000_cal.mod")
    print(f"  all variables, 40 periods   max relative deviation {d:.1e}")
    return d < 1e-10


def main() -> int:
    res = {"initial point": check_point()}
    ok_mode, pb, mode = check_mode()
    res["mode and Laplace"] = ok_mode
    # --no-mh skips the Metropolis part; --sem-mh is kept as an alias
    if not {"--no-mh", "--sem-mh"} & set(sys.argv) and pb is not None:
        res["Metropolis"] = check_mh(pb, mode)
    res["diffuse filter"] = check_unitroot()
    res["conditional forecast"] = check_conditional_forecast()
    print("\n" + "  ".join(f"{k}: {_ok(v)}" for k, v in res.items()))
    return 0 if all(res.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
