"""compare_estimation.py — estimação, suavização, decomposição e previsão, contra o Dynare.

O caso central é o fs2000.mod da pasta examples/ do Dynare (Schorfheide, 2000): modelo, priors e
dados como estão, lidos por `from_mod` (o `estimated_params`, o `varobs` e as opções `loglinear`
e `logdata` viram um `EstimationProblem`). Os dados são os que o Dynare montou a partir do
fs2000_data.m, exportados por run_estimation.sh.

  1. No ponto inicial (as médias a priori, como o Dynare o define): verossimilhança, densidade a
     priori, núcleo da posterior; variáveis e choques suavizados; a previsão de 8 períodos com
     bandas de 90%; shock_decomposition e realtime_shock_decomposition.
  2. A moda da posterior: o valor do núcleo (o do dsgelab tem de ser maior ou igual), os
     parâmetros, e a aproximação de Laplace — com a covariância que o PRÓPRIO Dynare reporta, a
     fórmula tem de dar o número do Dynare; com a Hessiana do dsgelab, a diferença é a das duas
     Hessianas numéricas.
  3. Metropolis-Hastings, 2 cadeias de 10 000 e mh_jscale 0,8, como no Dynare: as médias a
     posteriori, em unidades do erro de Monte Carlo conjunto das duas cadeias, e a média
     harmônica modificada. (--sem-mh pula esta parte, que leva alguns minutos.)
  4. O filtro difuso: estimation/unitroot.mod, escrito aqui, um nível de preços observado.
     Verossimilhança em dois pontos e o suavizador.
  5. A previsão condicional: tests/conditional_forecasts/1/fs2000_cal.mod do Dynare.

Pré-requisitos:
    bash fetch.sh
    bash run_estimation.sh
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

import dsgelab as dk
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
        print("fs2000_calib: rode run_estimation.sh")
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
    print("fs2000 — no ponto inicial (médias a priori)")
    print(f"  ponto inicial igual ao do Dynare              {same_start}")
    print(f"  verossimilhança  {ll:.10f}   Dynare {float(ex.loglik):.10f}   {d_ll:.1e}")
    print(f"  log do prior     {lp:.10f}   Dynare {float(ex.logpost) - float(ex.loglik):.10f}   {d_lp:.1e}")
    print(f"  variáveis suavizadas {d_sv:.1e}   choques suavizados {d_sh:.1e}")
    print(f"  previsão: média {d_fm:.1e}, bandas de 90% {d_fb:.1e}")
    print(f"  shock_decomposition {d_dec:.1e}   realtime_shock_decomposition {d_rt:.1e}")
    return (same_start and d_ll < 1e-8 and d_lp < 1e-10 and d_sv < 1e-12 and d_sh < 1e-12
            and d_fm < 1e-12 and d_fb < 1e-7 and d_dec < 1e-12 and d_rt < 1e-12)


def check_mode():
    ex = _load("fs2000_mode")
    if ex is None:
        print("fs2000_mode: rode run_estimation.sh")
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
    print("\nfs2000 — moda da posterior")
    print(f"  núcleo da posterior  dsgelab {m.log_posterior:.8f}   Dynare {float(ex.log_density):.8f}"
          f"   ({'maior ou igual' if m.log_posterior >= float(ex.log_density) - 1e-9 else 'MENOR'})")
    for n in pb.names:
        print(f"     {n:<11} dsgelab {m.values[n]:>10.6f} ± {m.std[n]:.6f}   "
              f"Dynare {dmode[n]:>10.6f}")
    print(f"  Laplace com a covariância do Dynare  {lap_formula:.8f}   Dynare {lap_dy:.8f}"
          f"   {abs(lap_formula - lap_dy):.1e}")
    print(f"  Laplace com a Hessiana do dsgelab    {m.laplace:.6f}   (diferença {m.laplace - lap_dy:+.3f}:"
          f" as duas Hessianas numéricas)")
    rel = max(abs(m.values[n] - dmode[n]) / m.std[n] for n in pb.names)
    ok = (m.log_posterior >= float(ex.log_density) - 1e-9 and abs(lap_formula - lap_dy) < 1e-8
          and rel < 0.05 and abs(m.laplace - lap_dy) < 0.1)
    return ok, pb, m


def check_mh(pb, mode) -> bool:
    ex = _load("fs2000_mh")
    if ex is None:
        print("fs2000_mh: rode run_estimation.sh")
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
    print("\nfs2000 — Metropolis-Hastings, 2 x 10 000, mh_jscale 0,8")
    print(f"  aceitação {np.round(mh.acceptance, 3)}   R-hat máx {max(mh.rhat().values()):.4f}")
    worst = 0.0
    for i, n in enumerate(mh.names):
        key = ("shocks_std", n[7:]) if n.startswith("stderr ") else ("parameters", n)
        se = X[:, i].std() / math.sqrt(ess[n])
        zval = (ours[n] - dyn[key]) / (math.sqrt(2) * se)
        worst = max(worst, abs(zval))
        print(f"     {n:<11} dsgelab {ours[n]:>10.6f}   Dynare {dyn[key]:>10.6f}   z {zval:+.2f}")
    mhm = mh.log_marginal_mhm()
    mhm_dy = float(ex.marginal.ModifiedHarmonicMean)
    print(f"  média harmônica modificada  dsgelab {mhm:.4f}   Dynare {mhm_dy:.4f}")
    return worst < 4.0 and abs(mhm - mhm_dy) < 0.3


def check_unitroot() -> bool:
    ex = _load("unitroot")
    if ex is None:
        print("unitroot: rode run_estimation.sh")
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
    print("\nFiltro difuso — estimation/unitroot.mod")
    print(f"  verossimilhança em dois pontos  {d:.1e}   suavizador {d_s:.1e}   "
          f"(períodos difusos: {pb.filter(th).diffuse_periods})")
    return d < 1e-8 and d_s < 1e-12


def check_conditional_forecast() -> bool:
    ex = _load("condfc")
    if ex is None:
        print("condfc: rode run_estimation.sh")
        return False
    mf = from_mod(HERE / "mods" / "forecast" / "fs2000_cal.mod")
    ss = mf.steady_state()
    sol = dk.solve(mf.model, mf.params, steady=ss)
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
    print("\nPrevisão condicional — tests/conditional_forecasts/1/fs2000_cal.mod")
    print(f"  todas as variáveis, 40 períodos   desvio relativo máx {d:.1e}")
    return d < 1e-10


def main() -> int:
    res = {"ponto inicial": check_point()}
    ok_mode, pb, mode = check_mode()
    res["moda e Laplace"] = ok_mode
    if "--sem-mh" not in sys.argv and pb is not None:
        res["Metropolis"] = check_mh(pb, mode)
    res["filtro difuso"] = check_unitroot()
    res["previsão condicional"] = check_conditional_forecast()
    print("\n" + "  ".join(f"{k}: {_ok(v)}" for k, v in res.items()))
    return 0 if all(res.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
