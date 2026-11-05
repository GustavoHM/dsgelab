"""compare_bvar.py: the BVAR and the DSGE-VAR, against Dynare 7.2.

  1. BVAR, on Dynare's tests/bvar_a_la_sims (bvar_standalone.mod and bvar_and_dsge.mod), with the
     data, options and hyperparameters of the tests.
     a. For every lag length of bvar_density (1 to 2 and 1 to 8): the prior's and the posterior's
        df, S, XXi and PhiHat, and the log marginal density.
     b. bvar_forecast: the RMSE over the observations after nobs, and the simulated paths with and
        without shocks, draw by draw. dsgelab takes the same normal draws as Dynare, which
        run_bvar.sh saves from Octave's generator, so the paths can agree to rounding. The share
        of explosive draws and the 60% bands follow from the paths.
  2. DSGE-VAR, on Dynare's tests/dsge-var (dsgevar_forward_calibrated_lambda.mod, lambda = 0.8,
     and dsgevar_forward_estimated_lambda.mod, lambda estimated), read by from_mod.
     a. The log posterior kernel and the log prior at the prior mean, at the mode and at perturbed
        points (in the second file also at lambda = 0.3, 0.5, 1.5 and at 0.2, where the prior is
        improper and both give -inf).
     b. The posterior mode found by dsgelab from Dynare's starting point: the kernel has to be at
        least Dynare's, and the parameters within 5% of a posterior standard deviation. The
        Laplace approximation at Dynare's mode with Dynare's inverse Hessian.
     Dynare 7.2 counts the dsge_varlag initial observations in the T of the likelihood (94 for
     the 90 rows the VAR is fitted on), while Del Negro and Schorfheide's (2004) T, and
     dsgelab's, is the number of rows. The comparison sets dsgelab's T to Dynare's, and then
     reports how far the mode moves with dsgelab's own T.

Prerequisites:
    bash fetch.sh
    bash run_bvar.sh
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

from dsgelab import EstimationProblem, bvar, posterior_mode
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
    return (d_df == 0 and max(d_S, d_X, d_P) < 1e-10 and d_ld < 1e-7 and d_rmse < 1e-14
            and max(d_with, d_none, d_mean) < 1e-12)


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
    return (same_inf and d_post < 1e-8 and d_prior < 1e-10 and gain >= -1e-6 and worst < 0.05
            and abs(laplace - float(ex.laplace)) < 1e-8)


if __name__ == "__main__":
    ok = [check_bvar("bvar_standalone"), check_bvar("bvar_and_dsge"),
          check_dsge_var("dsgevar_forward_calibrated_lambda"),
          check_dsge_var("dsgevar_forward_estimated_lambda")]
    print("\n" + ("OK" if all(ok) else "DIVERGE"))
    sys.exit(0 if all(ok) else 1)
