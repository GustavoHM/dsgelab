"""compare_bvar.py: the BVAR and the DSGE-VAR, against Dynare 7.2.

  1. BVAR, on Dynare's tests/bvar_a_la_sims (bvar_standalone.mod and bvar_and_dsge.mod), with the
     data, options and hyperparameters of the tests.
     a. For every lag length of bvar_density (1 to 2 and 1 to 8): the prior's and the posterior's
        df, S, XXi and PhiHat, and the log marginal density.
     b. bvar_forecast: the RMSE over the observations after nobs, and the simulated paths with and
        without shocks, draw by draw. dsgelab takes the same normal draws as Dynare, which
        run_bvar.sh saves from Octave's generator, so the paths can agree to rounding. The share
        of explosive draws and the 60% bands follow from the paths.
     c. bvar_irf(2, 'Cholesky') and bvar_irf(2, 'SquareRoot') in bvar_standalone.mod, 1000 draws
        of 40 periods with the normal draws Octave took: the sampled responses draw by draw, and
        the Mean, Median, Var, Upper_bound and Lower_bound of oo_.bvar.irf. Dynare 7.2 takes the
        rows of the impact matrix as the impulses and names the fields of oo_.bvar.irf shock
        first. With SquareRoot the matrix is symmetric, and BVAR.irf is compared as it is. With
        Cholesky, Dynare's impulses are the rows of the lower factor L, whose covariance is L'L
        rather than Sigma, and BVAR.irf reproduces them with the upper factor L' as its
        identification. The script reports how far the responses to the columns of L, which
        BVAR.irf takes by default, lie from Dynare's.
     d. The BVAR benchmark of forecast_evaluation, in bvar_and_dsge.mod: a BVAR(2) re-estimated at
        origins 100, 130, 160 and 190 (counted from first_obs = 20) on an expanding and on a
        60-observation rolling window, against bvar.toolbox on the same windows and the
        recursion of bvar.forecast at PhiHat, 1 to 4 steps ahead.
  2. DSGE-VAR, on Dynare's tests/dsge-var (dsgevar_forward_calibrated_lambda.mod, lambda = 0.8,
     and dsgevar_forward_estimated_lambda.mod, lambda estimated), read by from_mod.
     a. The log posterior kernel and the log prior at the prior mean, at the mode and at perturbed
        points (in the second file also at lambda = 0.3, 0.5, 1.5 and at 0.2, where the prior is
        improper and both give -inf).
     b. The posterior mode found by dsgelab from Dynare's starting point: the kernel has to be at
        least Dynare's, and the parameters within 5% of a posterior standard deviation. The
        Laplace approximation at Dynare's mode with Dynare's inverse Hessian.
     c. The DSGE-VAR's posterior IRFs (bayesian_irf), on the 1000 parameter draws PosteriorIRF
        took from the 2000 of Metropolis and with the normal draws Octave took: the sampled
        responses draw by draw, and the Mean, Median, Var, HPDinf, HPDsup and deciles of
        oo_.PosteriorIRF.bvardsge. Dynare draws Sigma with floor((1 + lambda) T) - k degrees of
        freedom, and the comparison takes the same.
     Dynare 7.2 counts the dsge_varlag initial observations in the T of the likelihood (94 for
     the 90 rows the VAR is fitted on), while Del Negro and Schorfheide's (2004) T, and
     dsgelab's, is the number of rows. The comparison sets dsgelab's T to Dynare's, and then
     reports how far the mode moves with dsgelab's own T.

Prerequisites:
    bash fetch.sh
    bash run_bvar.sh
"""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import numpy as np

from dsgelab import EstimationProblem, bvar, dsge_var_irf, forecast_evaluation, posterior_mode
from dsgelab.dynare import from_mod

HERE = Path(__file__).resolve().parent
OUT = HERE / "out_bvar"


class Stream:
    """The normal draws Octave took, handed out in order, as a numpy Generator would."""

    def __init__(self, z):
        self.z, self.i = np.asarray(z, float), 0

    def standard_normal(self, size):
        out = self.z[self.i:self.i + size]
        if out.size < size:
            raise RuntimeError("ran out of saved normal draws")
        self.i += size
        return out


def _rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.max(np.abs(a - b)) / max(1.0, float(np.max(np.abs(b)))))


def check_bvar(name: str) -> bool:
    from scipy.io import loadmat
    f = OUT / f"{name}.mat"
    if not f.exists():
        print(f"{name}: run run_bvar.sh")
        return False
    ex = loadmat(f, squeeze_me=True, struct_as_record=False)["ex"]
    mod = (HERE / "mods" / "bvar" / f"{name}.mod").read_text()
    max_lags = int(re.search(r"bvar_density\([^)]*\)\s*(\d+)\s*;", mod).group(1))
    fc = re.search(r"bvar_forecast\(([^)]*)\)\s*(\d+)\s*;", mod)
    horizon = int(re.search(r"forecast\s*=\s*(\d+)", fc.group(1)).group(1))
    fc_lags = int(fc.group(2))
    names = [str(v) for v in np.atleast_1d(ex.data_names)]
    varobs = [str(v) for v in np.atleast_1d(ex.varobs)]
    data = {v: np.asarray(ex.data)[:, names.index(v)] for v in varobs}
    T = len(data[varobs[0]])
    kw = dict(observables=varobs, first_obs=int(ex.first_obs), presample=int(ex.presample),
              tau=float(ex.tau), decay=float(ex.decay), lambda_=float(getattr(ex, "lambda")),
              mu=float(ex.mu), omega=int(ex.omega), flat=bool(ex.flat), train=int(ex.train))
    posts, priors = np.atleast_1d(ex.posterior), np.atleast_1d(ex.prior)
    dens = np.atleast_1d(ex.log_density)
    d_df = d_S = d_X = d_P = d_ld = 0.0
    for p in range(1, max_lags + 1):
        b = bvar(data, p, nobs=T - int(ex.first_obs) + 1, **kw)
        for ours, theirs in ((b.posterior, posts[p - 1]), (b.prior, priors[p - 1])):
            d_df = max(d_df, abs(ours.df - float(theirs.df)))
            d_S = max(d_S, _rel(ours.S, theirs.S))
            d_X = max(d_X, _rel(ours.XXi, np.atleast_2d(theirs.XXi)))
            d_P = max(d_P, _rel(ours.Phi, np.atleast_2d(theirs.PhiHat)))
        d_ld = max(d_ld, abs(b.log_marginal - dens[p - 1]))
    print(f"{name}: bvar_density {max_lags}, first_obs={int(ex.first_obs)}, train={int(ex.train)}"
          f", flat={bool(ex.flat)}")
    print(f"  prior and posterior, lags 1 to {max_lags}: df {d_df:.0e}, S {d_S:.1e}, XXi {d_X:.1e},"
          f" PhiHat {d_P:.1e} (relative)")
    print("  log marginal density: " + " ".join(f"{x:.4f}" for x in dens)
          + f"   deviation {d_ld:.1e}")

    b = bvar(data, fc_lags, nobs=int(ex.nobs), **kw)
    rm = b.rmse()
    d_rmse = max(abs(rm[v] - float(getattr(ex.rmse, v))) for v in varobs)
    draws = np.asarray(ex.sims_with_shocks).shape[-1]          # Dynare draws bvar_replic + 1
    out = b.forecast(horizon, draws, seed=Stream(ex.normals), coverage=float(ex.conf_sig))
    with_shocks = np.transpose(np.asarray(ex.sims_with_shocks), (2, 0, 1))
    no_shock = np.transpose(np.asarray(ex.sims_no_shock), (2, 0, 1))
    d_with = float(np.max(np.abs(out.paths - with_shocks)))
    d_none = float(np.max(np.abs(out.paths_no_shock - no_shock)))
    d_mean = max(float(np.max(np.abs(out.mean()[v]
                                     - np.asarray(getattr(ex.forecast.with_shocks.Mean, v)))))
                 for v in varobs)
    scale = float(np.max(np.abs(with_shocks)))
    print(f"  bvar_forecast {fc_lags}, nobs={int(ex.nobs)}, {horizon} periods, {draws} draws:"
          f" RMSE over {len(b.realized)} observations {d_rmse:.1e}")
    print(f"  paths, draw by draw: with shocks {d_with:.1e}, without {d_none:.1e} (largest value"
          f" {scale:.2f}); mean with shocks {d_mean:.1e}; explosive draws {out.explosive:.3f}")
    ok = (d_df == 0 and max(d_S, d_X, d_P) < 1e-10 and d_ld < 1e-7 and d_rmse < 1e-14
          and max(d_with, d_none, d_mean) < 1e-12)
    if hasattr(ex, "irf_cholesky"):
        ok = check_bvar_irf(ex, b, varobs) and ok
    if hasattr(ex, "bench"):
        ok = check_benchmark(ex, data, varobs) and ok
    return ok


def check_bvar_irf(ex, b, varobs) -> bool:
    """bvar_irf with both identifications, draw by draw and in oo_.bvar.irf."""
    fields = (("Mean", "mean"), ("Median", "median"), ("Var", "variance"),
              ("Upper_bound", "upper_bound"), ("Lower_bound", "lower_bound"))
    ok = True
    upper = (lambda S: np.linalg.cholesky(S).T)
    for tag, ident, label in (("squareroot", "SquareRoot", "SquareRoot"),
                              ("cholesky", upper, "Cholesky, as the upper factor L'"),
                              ("cholesky", "Cholesky", "Cholesky, the columns of L")):
        D = np.asarray(getattr(ex, f"irf_sims_{tag}"))          # (shock, variable, period, draw)
        r = b.irf(int(ex.irf_periods), D.shape[-1], identification=ident,
                  seed=Stream(getattr(ex, f"irf_normals_{tag}")), coverage=float(ex.conf_sig))
        ours = np.array([[r["responses"][e][v].T for v in varobs] for e in varobs])
        scale = float(np.max(np.abs(D)))
        d_sims = float(np.max(np.abs(ours - D))) / scale
        oo = getattr(ex, f"irf_{tag}")
        d_oo = max(float(np.max(np.abs(np.ravel(getattr(getattr(getattr(oo, key), e), v))
                                       - r[f][e][v])))
                   for key, f in fields for e in varobs for v in varobs) / scale
        print(f"  bvar_irf, {label}: {D.shape[-1]} draws of {D.shape[2]} periods, responses draw by"
              f" draw {d_sims:.1e}, oo_.bvar.irf {d_oo:.1e} (relative to {scale:.4f})")
        if ident == "Cholesky":
            ok = ok and d_sims > 1e-3          # Dynare's are not the responses to L's columns
        else:
            ok = ok and max(d_sims, d_oo) < 1e-12
    return ok


def check_benchmark(ex, data, varobs) -> bool:
    """forecast_evaluation's BVAR benchmark against bvar.toolbox on the same windows."""
    bn = ex.bench
    text = (HERE / "mods" / "bvar" / "bvar_and_dsge.mod").read_text()
    # dsgelab's reader takes the prior shapes in lower case, which Dynare also accepts
    mf = from_mod(re.sub(r"\b[A-Z0-9_]+_PDF\b", lambda m: m.group(0).lower(), text))
    pb = mf.estimation_problem({v: data[v] for v in varobs})
    if pb.first_obs != int(bn.first_obs):
        raise RuntimeError("first_obs differs from the estimation's")
    E = np.asarray(bn.errors)                   # (origin, horizon, variable, window)
    origins = [int(o) for o in np.atleast_1d(bn.origins)]
    worst, scale = 0.0, float(np.max(np.abs(E)))
    for w, window in enumerate((None, int(bn.window))):
        fe = forecast_evaluation(pb, origins, int(bn.horizon), estimate="fixed", theta=pb.start(),
                                 window=window, bvar_lags=int(bn.lags))
        ours = np.stack([fe.bvar_errors[v] for v in pb.observables], axis=-1)
        worst = max(worst, float(np.max(np.abs(ours - E[:, :, :, w]))))
    print(f"  forecast_evaluation's BVAR({int(bn.lags)}) benchmark, origins {origins}, expanding"
          f" and {int(bn.window)}-observation windows, 1 to {int(bn.horizon)} steps:"
          f" errors {worst:.1e} (largest error {scale:.4f})")
    return worst < 1e-12


class DynareT(EstimationProblem):
    """Dynare 7.2's T: the rows of the VAR plus its dsge_varlag initial observations."""

    def _dsge_var_nobs(self) -> int:
        return len(self.Y) + int(self.dsge_varlag)


def check_dsge_var(name: str) -> bool:
    from scipy.io import loadmat
    f = OUT / f"{name}.mat"
    if not f.exists():
        print(f"{name}: run run_bvar.sh")
        return False
    ex = loadmat(f, squeeze_me=True, struct_as_record=False)["ex"]
    mf = from_mod(HERE / "mods" / "dsge_var" / f"{name}.mod")
    varobs = [str(v) for v in np.atleast_1d(ex.varobs)]
    p = int(ex.dsge_varlag)
    last = int(ex.first_obs) - 1 + int(ex.nobs_dataset) - p
    data = {v: np.asarray(ex.data)[:last, i] for i, v in enumerate(varobs)}
    own = mf.estimation_problem(data)
    pb = mf.estimation_problem(data)
    pb.__class__ = DynareT
    order = [[str(n) for n in ex.names].index(n) for n in pb.names]
    pts = np.asarray(ex.points)[order].T
    theirs = -np.asarray(ex.minus_log_posterior, float)
    ours = np.array([pb.log_posterior(x) for x in pts])
    finite = np.isfinite(theirs)
    same_inf = bool(np.all(np.isfinite(ours) == finite))
    d_post = float(np.max(np.abs(ours[finite] - theirs[finite])))
    d_prior = float(np.max(np.abs(np.array([pb.log_prior(x) for x in pts])[finite]
                                  - np.asarray(ex.log_prior, float)[finite])))
    print(f"\n{name}: {len(varobs)} observables, {p} lags, T = {len(own.Y)} rows"
          f" (Dynare's T {pb._dsge_var_nobs()})")
    print(f"  log posterior kernel at {finite.sum()} points: {d_post:.1e};  log prior {d_prior:.1e};"
          f"  -inf at the same {(~finite).sum()} point(s): {same_inf}")

    dmode = np.asarray(ex.mode)[order]
    sd = np.sqrt(np.diag(np.asarray(ex.variance)))[order]
    at_dynare = pb.log_posterior(dmode)
    mode = posterior_mode(pb, pb.start())
    gain = mode.log_posterior - float(ex.log_density)
    worst = float(np.max(np.abs(mode.x - dmode) / sd))
    C = np.asarray(ex.variance)[np.ix_(order, order)]
    laplace = at_dynare + 0.5 * len(dmode) * np.log(2 * np.pi) + 0.5 * np.linalg.slogdet(C)[1]
    print(f"  kernel at Dynare's mode {at_dynare - float(ex.log_density):+.1e} from Dynare's;"
          f" dsgelab's mode: kernel {gain:+.2e} over Dynare's, parameters within {worst:.1e}"
          f" posterior sd")
    print(f"  Laplace with Dynare's inverse Hessian {abs(laplace - float(ex.laplace)):.1e};"
          f" with dsgelab's Hessian {mode.laplace - float(ex.laplace):+.1e}")
    mine = posterior_mode(own, mode.x)
    shift = np.abs(mine.x - mode.x) / sd
    line = f"  with dsgelab's T: mode moves by up to {shift.max():.2f} posterior sd"
    if "dsge_prior_weight" in pb.names:
        i = pb.names.index("dsge_prior_weight")
        line += (f", lambda {mine.x[i]:.3f} against {mode.x[i]:.3f} (lambda T = "
                 f"{mine.x[i] * len(own.Y):.1f} against {mode.x[i] * pb._dsge_var_nobs():.1f})")
    print(line)
    ok = (same_inf and d_post < 1e-8 and d_prior < 1e-10 and gain >= -1e-6 and worst < 0.05
          and abs(laplace - float(ex.laplace)) < 1e-8)
    return check_dsge_var_irf(ex, pb, order, varobs) and ok


class DynareIRF(DynareT):
    """Dynare 7.2's draws of Sigma in PosteriorIRF: floor((1 + lambda) T) - k degrees of freedom."""

    def dsge_var_posterior(self, theta):
        post = super().dsge_var_posterior(theta)
        post.df = float(math.floor(post.df + 1e-9))
        return post


def check_dsge_var_irf(ex, pb, order, varobs) -> bool:
    """bayesian_irf of the DSGE-VAR, draw by draw and in oo_.PosteriorIRF.bvardsge."""
    pb.__class__ = DynareIRF
    X = np.atleast_2d(np.asarray(ex.irf_draws))[:, order]
    r = dsge_var_irf(pb, X, periods=int(ex.irf_periods), n_draws=None,
                     mass=float(ex.mh_conf_sig), seed=Stream(ex.irf_normals))
    exo = [str(e) for e in np.atleast_1d(ex.exo_names)]
    D = np.asarray(ex.irf_sims)                                 # (period, variable, shock, draw)
    ours = np.array([[r["responses"][e][v].T for e in exo] for v in varobs])
    ours = np.moveaxis(ours, 2, 0)
    scale = float(np.max(np.abs(D)))
    d_sims = float(np.max(np.abs(ours - D))) / scale
    fields = (("Mean", "mean"), ("Median", "median"), ("Var", "variance"), ("HPDinf", "lower"),
              ("HPDsup", "upper"), ("deciles", "deciles"))
    d_oo = max(float(np.max(np.abs(np.asarray(getattr(getattr(ex.bvardsge, key), f"{v}_{e}"))
                                   - r[f][e][v])))
               for key, f in fields for v in varobs for e in exo) / scale
    print(f"  bayesian_irf of the DSGE-VAR: {r['draws']} parameter draws, {D.shape[0]} periods,"
          f" responses draw by draw {d_sims:.1e}, oo_.PosteriorIRF.bvardsge {d_oo:.1e} (relative"
          f" to {scale:.3f}); VAR draws redrawn as explosive {r['explosive']:.4f}")
    return max(d_sims, d_oo) < 1e-12


if __name__ == "__main__":
    ok = [check_bvar("bvar_standalone"), check_bvar("bvar_and_dsge"),
          check_dsge_var("dsgevar_forward_calibrated_lambda"),
          check_dsge_var("dsgevar_forward_estimated_lambda")]
    print("\n" + ("OK" if all(ok) else "DIVERGE"))
    sys.exit(0 if all(ok) else 1)
