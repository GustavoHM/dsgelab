"""compare_plots.py: the numbers that the plots draw, against Dynare.

The plots draw what the package computes. What they draw that the package did not compute
before is compared here:

  1. prior_density against Dynare's draw_prior_density on the nine priors of fs2000.mod (beta,
     normal and inverse gamma), grid and density;
  2. posterior_density against the kernel density of Dynare's posterior_moments, on two sets of
     draws made in Octave: 6000 or so with repeated values, as a Metropolis chain has after
     rejections (Dynare corrects the bandwidth for them), and 5000 from a lognormal.

The rest uses an estimation of fs2000.mod in Dynare, with mode_check, two chains of 2,500 draws
and bayesian_irf (run_plots.sh mcmc). The figures Dynare draws are recorded as it draws them, and
each is compared with the function that draws it here, on the same draws, mode and parameters:

  3. brooks_gelman against the univariate and multivariate figures of mcmc_diagnostics;
  4. the moving average of trace_plot, for alp and for the log posterior;
  5. mh_autocorrelation against the bars of mh_autocorrelation_function, for alp;
  6. mode_check against Dynare's mode_check: the grid of each parameter, and the log posterior
     and log-likelihood kernels on it;
  7. posterior_irf on the 1,200 parameter draws of Dynare's PosteriorIRF: the responses at each
     draw, and their mean, median, HPD interval and deciles (oo_.PosteriorIRF.dsge);
  8. the posterior densities of the "Priors and posteriors" figure, and the smoothed shocks and
     observables of the smoother's figures, at the posterior mean.

Prerequisites:
    bash fetch.sh
    bash run_plots.sh
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from dsgelab.dynare import from_mod
from dsgelab.plots import (_moving_average, _posterior_moments, brooks_gelman, mh_autocorrelation, mode_check,
                           posterior_density, posterior_irf, prior_density)

HERE = Path(__file__).resolve().parent


def _load(name):
    from scipy.io import loadmat
    f = HERE / "out_plots" / f"{name}.mat"
    return None if not f.exists() else loadmat(f, squeeze_me=True, struct_as_record=False)["ex"]


def _names(x):
    return [str(v).strip() for v in np.atleast_1d(x)]


def densities(ex) -> bool:
    names = _names(ex.names)
    mf = from_mod(HERE / "mods" / "examples" / "fs2000.mod")
    priors = {e["name"]: e["prior"] for e in mf.estimated if e["prior"] is not None}
    d_x = d_f = 0.0
    shapes = set()
    for i, n in enumerate(names):
        x, fx = prior_density(priors[n])
        shapes.add(priors[n].shape)
        dx, df = np.asarray(ex.prior_x)[:, i], np.asarray(ex.prior_f)[:, i]
        same_nan = np.array_equal(np.isnan(fx), np.isnan(df))
        ok = ~np.isnan(df)
        d_x = max(d_x, float(np.max(np.abs(x - dx) / np.maximum(1.0, np.abs(dx)))))
        d_f = max(d_f, float(np.max(np.abs(fx[ok] - df[ok]) / np.maximum(1.0, np.abs(df[ok])))),
                  0.0 if same_nan else np.inf)
    print(f"prior densities of fs2000 ({len(names)} priors: {', '.join(sorted(shapes))}):"
          f" grid {d_x:.1e}, density {d_f:.1e} (relative)")
    worst = 0.0
    for k in (1, 2):
        draws = np.asarray(getattr(ex, f"draws{k}"), dtype=float)
        D = np.asarray(getattr(ex, f"density{k}"))
        x, fx = posterior_density(draws)
        dk = max(float(np.max(np.abs(x - D[:, 0]))), float(np.max(np.abs(fx - D[:, 1]))) / D[:, 1].max())
        repeats = draws.size - np.unique(draws).size
        print(f"posterior density, {draws.size} draws ({repeats} repeated): {dk:.1e}"
              f" (grid and density, relative to its peak)")
        worst = max(worst, dk)
    return d_x < 1e-12 and d_f < 1e-10 and worst < 1e-12


def _lines(panel):
    """The (x, y) of each line or bar series of a recorded panel, in the order drawn."""
    many = isinstance(panel.y, np.ndarray) and panel.y.dtype == object
    xs, ys = (panel.x, panel.y) if many else ([panel.x], [panel.y])
    return [(np.asarray(x, dtype=float).ravel(), np.asarray(y, dtype=float).ravel())
            for x, y in zip(xs, ys)]


def _rel(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)))


def mcmc(ex) -> bool:
    figs = {str(f.name): list(np.atleast_1d(f.panels)) for f in np.atleast_1d(ex.figures)}
    names = _names(ex.names)                       # Dynare's order: shock sds first
    D = np.asarray(ex.draws)                       # (draws, parameters, chains), burn-in included
    N, k, m = D.shape
    chains = np.transpose(D, (2, 0, 1))
    lp = np.asarray(ex.logpo).T
    drop = float(ex.mh_drop)
    ok = True

    # 3. Brooks and Gelman
    bg = brooks_gelman(chains, names=names, log_posterior=lp, drop=drop)
    pages = [p for name in sorted(n for n in figs if n.startswith("fs2000_udiag")) for p in figs[name]]
    d_it = d_u = 0.0
    for q, panel in enumerate(pages):
        j, c = divmod(q, 3)
        (x1, pooled), (_, within) = _lines(panel)
        d_it = max(d_it, float(np.max(np.abs(x1 - bg.iterations))))
        d_u = max(d_u, _rel(bg.univariate[:, 2 * c, j], pooled),
                  _rel(bg.univariate[:, 2 * c + 1, j], within))
    d_m = 0.0
    for c, panel in enumerate(figs["fs2000_mdiag"]):
        (_, pooled), (_, within) = _lines(panel)
        d_m = max(d_m, _rel(bg.multivariate[:, 2 * c], pooled),
                  _rel(bg.multivariate[:, 2 * c + 1], within))
    print(f"Brooks and Gelman, {m} chains of {N} draws, {bg.iterations.size} windows: windows "
          f"{d_it:.0f}, univariate ({len(pages)} panels) {d_u:.1e}, multivariate {d_m:.1e} "
          f"(relative)")
    ok &= d_it == 0 and d_u < 1e-12 and d_m < 1e-12

    # 4. trace plots
    j = names.index(str(ex.trace_name))
    window = int(ex.trace_plot_ma)
    d_tr = 0.0
    for fig, x in ((f"TracePlot_{ex.trace_name}_blck_1", D[:, j, 0]),
                   ("TracePlot_Posterior_blck_1", lp[0])):
        (_, drawn), (_, ma) = _lines(figs[fig][0])
        ours = _moving_average(x, window)
        same = np.array_equal(np.isnan(ma), np.isnan(ours)) and np.array_equal(drawn, x)
        d_tr = max(d_tr, _rel(ours[~np.isnan(ours)], ma[~np.isnan(ma)]) if same else np.inf)
    print(f"trace_plot, moving average over {2 * window + 1} draws, {ex.trace_name} and the log"
          f" posterior: {d_tr:.1e} (relative)")
    ok &= d_tr < 1e-12

    # 5. autocorrelogram of the draws after burn-in of the first chain
    (lags, bars), = _lines(figs[f"MH_Autocorrelation_{ex.trace_name}"][0])
    kept = D[int(np.floor(drop * N)):, j, 0]
    d_ac = float(np.max(np.abs(mh_autocorrelation(kept, int(ex.acf_size)) - bars)))
    print(f"mh_autocorrelation, {ex.trace_name}, lags 0 to {int(ex.acf_size)}: {d_ac:.1e}")
    ok &= np.array_equal(lags, np.arange(int(ex.acf_size) + 1)) and d_ac < 1e-13

    # the estimation problem, with the data Dynare used
    varobs = _names(ex.varobs)
    data = {v: np.asarray(ex.data)[:, i] for i, v in enumerate(varobs)}
    mf = from_mod(HERE / "mods" / "examples" / "fs2000.mod")
    pb = mf.estimation_problem(data)
    perm = [names.index(n) for n in pb.names]

    # 6. mode_check at Dynare's mode
    slices = mode_check(pb, np.asarray(ex.xparam1)[perm])
    d_grid = 0.0
    worst, mism = {}, {}
    for n in pb.names:
        Dc = np.asarray(getattr(ex.mcheck.cross, n.replace("stderr ", "SE_")))
        s = slices[n]
        if Dc.shape[0] != s["x"].size:
            d_grid = np.inf
            continue
        d_grid = max(d_grid, float(np.max(np.abs(s["x"] - Dc[:, 0]))))
        both = ~np.isnan(Dc[:, 1]) & ~np.isnan(s["log_posterior"])
        mism[n] = int(np.sum(np.isnan(Dc[:, 1]) != np.isnan(s["log_posterior"])))
        worst[n] = max(_rel(s["log_posterior"][both], Dc[both, 1]),
                       _rel(s["log_likelihood"][both], Dc[both, 2]))
    bounded = [n for n in pb.names if pb.priors[n].shape != "normal"]
    normal = [n for n in pb.names if n not in bounded]
    print(f"mode_check, {len(pb.names)} parameters, {s['x'].size} points each: grid {d_grid:.1e};"
          f" log posterior and log-likelihood kernels {max(worst[n] for n in bounded):.1e}"
          f" (relative) for the {len(bounded)} parameters with a bounded prior")
    for n in normal:
        print(f"  {n} (normal prior, Dynare's grid runs from -10 to 10): {worst[n]:.1e} relative"
              f" where both solve the model, and {mism[n]} of {s['x'].size} points that only"
              f" one of them solves")
    ok &= d_grid < 1e-12 and max(worst[n] for n in bounded) < 1e-8
    ok &= all(mism[n] == 0 for n in bounded)

    # 7. posterior IRFs on Dynare's parameter draws
    P = np.asarray(ex.irf_params)[:, perm]
    look = _names(ex.irf_vars)
    exo = _names(ex.exo_names)
    stock = np.asarray(ex.irf_stock)               # (periods, variables, shocks, draws)
    pi = posterior_irf(pb, P, variables=look, periods=stock.shape[0], n_draws=None,
                       mass=float(ex.mh_conf_sig), percent=False)
    scale = float(np.max(np.abs(stock)))
    d_r = max(float(np.max(np.abs(pi["responses"][e][v] - stock[:, j, i, :].T)))
              for i, e in enumerate(exo) for j, v in enumerate(look)) / scale
    # and the same statistics of Dynare's own responses, which isolates posterior_moments
    own = _posterior_moments(np.moveaxis(stock, 3, 0), float(ex.mh_conf_sig))
    d_s, d_own = {}, 0.0
    for key, field in (("mean", "Mean"), ("median", "Median"), ("lower", "HPDinf"),
                       ("upper", "HPDsup"), ("deciles", "deciles")):
        d_s[key] = 0.0
        for i, e in enumerate(exo):
            for j, v in enumerate(look):
                theirs = np.asarray(getattr(getattr(ex.irf, field), f"{v}_{e}"))
                d_s[key] = max(d_s[key], float(np.max(np.abs(pi[key][e][v] - theirs))) / scale)
                d_own = max(d_own, float(np.max(np.abs(own[key][:, j, i] - theirs))))
    print(f"posterior_irf, {pi['draws']} draws, {len(look)} variables, {len(exo)} shocks, "
          f"{stock.shape[0]} periods: responses {d_r:.1e}, " +
          ", ".join(f"{k} {v:.1e}" for k, v in d_s.items()) +
          f" (relative to the largest response); the statistics of Dynare's own responses "
          f"{d_own:.1e}")
    ok &= d_r < 1e-8 and max(d_s.values()) < 1e-8 and d_own < 1e-15

    # 8. posterior densities and the smoother's figures at the posterior mean
    pooled = np.concatenate([D[int(np.floor(drop * N)):, :, b] for b in range(m)])
    d_pd = 0.0
    for q, panel in enumerate(figs["fs2000_PriorsAndPosteriors1"]):
        _, (x1, f1), (xm, _) = _lines(panel)
        x, f = posterior_density(pooled[:, q])
        d_pd = max(d_pd, float(np.max(np.abs(x - x1))), float(np.max(np.abs(f - f1))) / f.max(),
                   float(np.max(np.abs(xm - ex.xparam1[q]))))
    sm = pb.smoother(pooled.mean(axis=0)[perm])
    sol = sm.space.solution
    hat = sol.variables(sm.states[:, :sm.space.n_model_states])
    d_sh = 0.0
    for panel in figs["fs2000_SmoothedShocks1"]:
        e = str(panel.title)
        st_ = {sh: s for s, sh in mf.shock_states.items()}[e]
        ours = hat[st_] * sol.coords.scale[sol.model.position(st_)]
        d_sh = max(d_sh, float(np.max(np.abs(_lines(panel)[1][1] - ours))))
    fitted = sm.states @ sm.space.Z.T + sm.space.d
    Y = sm.space.data_matrix(data)
    d_ob = 0.0
    for panel in figs["fs2000_HistoricalAndSmoothedVariables1"]:
        i = sm.space.observables.index(str(panel.title))
        (_, smooth), (_, observed) = _lines(panel)
        d_ob = max(d_ob, float(np.max(np.abs(smooth - fitted[:, i]))),
                   float(np.max(np.abs(observed - Y[:, i]))))
    print(f"posterior densities and modes of 'Priors and posteriors': {d_pd:.1e};"
          f" smoothed shocks {d_sh:.1e} and observables {d_ob:.1e} at the posterior mean")
    ok &= d_pd < 1e-12 and d_sh < 1e-12 and d_ob < 1e-12
    return bool(ok)


def main() -> bool:
    ok = True
    for name, check in (("densities", densities), ("mcmc", mcmc)):
        ex = _load(name)
        if ex is None:
            print(f"out_plots/{name}.mat is missing: run run_plots.sh {name}")
            ok = False
            continue
        ok &= check(ex)
    return ok


if __name__ == "__main__":
    ok = main()
    print("\n" + ("OK" if ok else "DIVERGE"))
    sys.exit(0 if ok else 1)
