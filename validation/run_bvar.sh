#!/usr/bin/env bash
# run_bvar.sh: Dynare's references for the BVAR and the DSGE-VAR.
#
# Usage:  bash run_bvar.sh [bvar|dsge_var ...]   (no arguments: both)
#
#   bvar      Dynare 7.2's tests/bvar_a_la_sims: bvar_standalone.mod (bvar_density 2 with
#             first_obs=20, bvar_prior_flat and bvar_prior_train=10, and bvar_forecast 2 on
#             nobs=200) and bvar_and_dsge.mod (bvar_density 8 and bvar_forecast 8 after an
#             estimation that sets first_obs=20). For every lag length, the prior and posterior
#             (df, S, XXi, PhiHat) and the log marginal density; for the forecast, the simulated
#             paths with and without shocks and the RMSE over the observations after nobs.
#             The estimation of bvar_and_dsge.mod runs without Metropolis (mh_replic=0), since
#             its result is not compared. Before bvar_forecast and before each bvar_irf
#             (Cholesky and SquareRoot, in bvar_standalone.mod), a verbatim block resets
#             Octave's generator and saves the normal draws the command is about to take, so
#             that dsgelab can take the same ones, and after each bvar_irf another saves
#             oo_.bvar.irf and the sampled responses. After bvar_and_dsge.mod, bvar.toolbox
#             re-estimates the BVAR(2) on the windows of a forecast evaluation (expanding and
#             60 observations rolling, origins 100, 130, 160 and 190 counted from first_obs)
#             and the recursion of bvar.forecast at PhiHat gives the errors 1 to 4 steps ahead.
#   dsge_var  Dynare 7.2's tests/dsge-var: simul_hybrid.mod simulates the data, and
#             dsgevar_forward_calibrated_lambda.mod (dsge_var=0.8) and
#             dsgevar_forward_estimated_lambda.mod (dsge_prior_weight estimated) run with their
#             models, priors, samples, optimiser (mode_compute=4) and Metropolis (2000 draws)
#             as they are. For each, the posterior mode, the kernel there, its inverse Hessian
#             and the Laplace approximation, and dsge_var_likelihood at the prior mean and at
#             perturbed points. bayesian_irf is removed from the estimation and PosteriorIRF
#             is called after it, with the generators handled as for the BVAR: the parameter
#             draws it takes, the normal draws, the DSGE-VAR's sampled responses and
#             oo_.PosteriorIRF.bvardsge are saved.
#
# Exports out_bvar/<file>.mat.
set -euo pipefail
cd "$(dirname "$0")"

OCTAVE="${OCTAVE:-/opt/homebrew/bin/octave}"
if [ -z "${DYNARE_MATLAB:-}" ]; then
  for c in /opt/homebrew/opt/dynare/lib/dynare/matlab /opt/homebrew/lib/dynare/matlab \
           /usr/local/opt/dynare/lib/dynare/matlab /usr/lib/dynare/matlab \
           /Applications/Dynare/*/matlab; do
    [ -d "$c" ] && DYNARE_MATLAB="$c" && break
  done
fi
[ -n "${DYNARE_MATLAB:-}" ] || { echo "FATAL: Dynare's matlab/ directory not found. Set DYNARE_MATLAB."; exit 1; }

VARIANTS=("$@")
[ ${#VARIANTS[@]} -eq 0 ] && VARIANTS=(bvar dsge_var)
mkdir -p out_bvar

octave_run() {   # octave_run <directory> <code>
  ( cd "$1" && "$OCTAVE" --no-gui --eval "pkg load statistics datatypes; addpath '$DYNARE_MATLAB'; $2" )
}

for v in "${VARIANTS[@]}"; do
  case "$v" in
  bvar)
    for f in bvar_standalone bvar_and_dsge; do
      echo ">>> $f"
      W=work_bvar_$f; rm -rf "$W"; mkdir -p "$W"
      cp mods/bvar/bvar_sample.m "$W/"
      python3 - "mods/bvar/$f.mod" "$W/$f.mod" <<'PY'
import re, sys
text = open(sys.argv[1]).read()
text = re.sub(r"mh_replic\s*=\s*\d+", "mh_replic = 0", text)
seed = ("verbatim;\nrandn('state', 20261103); bvar_normals = randn(1200000, 1);"
        " randn('state', 20261103);\nend;\n")
text = re.sub(r"^(bvar_forecast\()", lambda m: seed + m.group(1), text, flags=re.M)
def irf(m):
    tag = m.group(1).lower()
    return (f"verbatim;\nrandn('state', 20261106); irf_normals_{tag} = randn(1200000, 1);"
            f" randn('state', 20261106);\nend;\n{m.group(0)}\nverbatim;\nirf_{tag} = oo_.bvar.irf;"
            f" s_ = load([M_.dname '/bvar_irf/simulations.mat']);"
            f" irf_sims_{tag} = s_.sampled_irfs;\nend;\n")
text = re.sub(r"^bvar_irf\(\s*\d+\s*,\s*'(\w+)'\s*\)\s*;", irf, text, flags=re.M)
open(sys.argv[2], "w").write(text)
PY
      rm -f "out_bvar/$f.mat"
      octave_run "$W" "dynare $f noclearall nograph;
        ex = struct(); ex.varobs = options_.varobs; ex.first_obs = options_.first_obs;
        ex.nobs = options_.nobs; ex.presample = options_.presample;
        ex.tau = options_.bvar_prior_tau; ex.decay = options_.bvar_prior_decay;
        ex.lambda = options_.bvar_prior_lambda; ex.mu = options_.bvar_prior_mu;
        ex.omega = options_.bvar_prior_omega; ex.flat = options_.bvar_prior_flat;
        ex.train = options_.bvar_prior_train; ex.replic = options_.bvar_replic;
        ex.conf_sig = options_.bvar.conf_sig;
        ex.log_density = oo_.bvar.log_marginal_data_density;
        ex.posterior = oo_.bvar.posterior; ex.prior = oo_.bvar.prior;
        s = load([M_.dname '/bvar_forecast/simulations.mat']);
        ex.sims_no_shock = s.sims_no_shock; ex.sims_with_shocks = s.sims_with_shocks;
        ex.rmse = oo_.bvar.forecast.rmse; ex.forecast = oo_.bvar.forecast;
        ex.normals = bvar_normals; ex.irf_periods = options_.irf;
        for tag = {'cholesky', 'squareroot'}
          if exist(['irf_' tag{1}], 'var')
            ex.(['irf_' tag{1}]) = eval(['irf_' tag{1}]);
            ex.(['irf_sims_' tag{1}]) = eval(['irf_sims_' tag{1}]);
            ex.(['irf_normals_' tag{1}]) = eval(['irf_normals_' tag{1}]);
          end
        end
        if strcmp('$f', 'bvar_and_dsge')
          lags = 2; H = 4; origins = [100 130 160 190]; window = 60; fo = options_.first_obs;
          options_.bvar_prior_train = 0; options_.presample = 0; options_.forecast = H;
          ex.bench = struct('lags', lags, 'horizon', H, 'origins', origins, 'window', window, ...
                            'first_obs', fo, 'errors', NaN(numel(origins), H, 2, 2));
          for w = 1:2
            for o = 1:numel(origins)
              n = origins(o); start = (w == 2) * (n - window);
              options_.first_obs = fo + start + lags; options_.nobs = n - start - lags;
              [ny, nx, posterior, prior, fd] = bvar.toolbox(lags);
              lags_data = fd.initval;
              for t = 1:H
                X = [reshape(flipdim(lags_data, 1)', 1, ny*lags) fd.realized_xdata(t, :)];
                y = X * posterior.PhiHat;
                lags_data(1:end-1, :) = lags_data(2:end, :); lags_data(end, :) = y;
                ex.bench.errors(o, t, :, w) = fd.realized_val(t, :) - y;
              end
            end
          end
        end
        bvar_sample; ex.data = [dx(:), dy(:)]; ex.data_names = {'dx', 'dy'};
        save('-mat7-binary', '../out_bvar/$f.mat', 'ex'); printf('saved out_bvar/$f.mat\n');" \
        > "out_bvar/$f.log" 2>&1 || { echo "FAILED $f, see out_bvar/$f.log"; tail -20 "out_bvar/$f.log"; exit 1; }
      grep '^saved' "out_bvar/$f.log"
      rm -rf "$W"
    done
    ;;
  dsge_var)
    W=work_bvar_dsge_var; rm -rf "$W"; mkdir -p "$W"
    cp mods/dsge_var/*.mod "$W/"
    echo ">>> simul_hybrid (the data)"
    octave_run "$W" "dynare simul_hybrid noclearall nograph;" > out_bvar/simul_hybrid.log 2>&1 \
      || { echo "FAILED simul_hybrid"; tail -20 out_bvar/simul_hybrid.log; exit 1; }
    for f in dsgevar_forward_calibrated_lambda dsgevar_forward_estimated_lambda; do
      echo ">>> $f"
      python3 - "$W/$f.mod" <<'PY'
import re, sys
text = open(sys.argv[1]).read()
def edit(m):
    line = re.sub(r",\s*bayesian_irf", "", m.group(0))
    return line.replace(");", ",nograph,nodisplay,plot_priors=0);")
text = re.sub(r"^estimation\(.*$", edit, text, flags=re.M)
open(sys.argv[1], "w").write(text)
PY
      rm -f "out_bvar/$f.mat"
      octave_run "$W" "dynare $f noclearall nograph;
        ex = struct(); options_ = select_qz_criterium_value(options_);
        options_.sub_draws = set_number_of_subdraws(M_, options_);
        rs_ = rand('state'); ex.irf_draws = get_posterior_subsample(M_, options_, options_.sub_draws);
        rand('state', rs_);
        randn('state', 20261106); ex.irf_normals = randn(1500000, 1); randn('state', 20261106);
        oo_ = PosteriorIRF('posterior', options_, estim_params_, oo_, M_, bayestopt_, dataset_, ...
                           dataset_info, 'estimation');
        s_ = load([M_.dname filesep get_posterior_folder_name(options_) filesep M_.fname ...
                   '_IRF_BVARDSGEs1.mat']);
        ex.irf_sims = s_.STOCK_IRF_BVARDSGE; ex.bvardsge = oo_.PosteriorIRF.bvardsge;
        ex.irf_periods = options_.irf; ex.mh_conf_sig = options_.mh_conf_sig;
        ex.exo_names = M_.exo_names;
        ex.names = bayestopt_.name; ex.varobs = options_.varobs;
        ex.first_obs = options_.first_obs; ex.nobs_dataset = dataset_.nobs;
        ex.dsge_varlag = options_.dsge_varlag;
        ex.mode = oo_.posterior.optimization.mode(:);
        ex.log_density = oo_.posterior.optimization.log_density;
        ex.variance = oo_.posterior.optimization.Variance;
        ex.laplace = oo_.MarginalDensity.LaplaceApproximation;
        info_ = var_sample_moments(options_.dsge_varlag, 0, dataset_, dataset_info);
        bounds_ = prior_bounds(bayestopt_, options_.prior_trunc);
        p1 = bayestopt_.p1; k = numel(p1);
        ex.points = [p1, p1 .* (1 + 0.1 * (-1).^(1:k)'), ex.mode, ex.mode .* (1 - 0.05 * (-1).^(1:k)')];
        if strcmp(bayestopt_.name{end}, 'dsge_prior_weight')
          for w = [0.3, 0.5, 1.5, 0.2]
            ex.points = [ex.points, [ex.mode(1:end-1); w]];
          end
        end
        for j = 1:size(ex.points, 2)
          x = ex.points(:, j);
          ex.minus_log_posterior(j) = dsge_var_likelihood(x, dataset_, info_, options_, M_, ...
            estim_params_, bayestopt_, bounds_, oo_.dr, oo_.steady_state, oo_.exo_steady_state, ...
            oo_.exo_det_steady_state);
          ex.log_prior(j) = priordens(x, bayestopt_.pshape, bayestopt_.p6, bayestopt_.p7, ...
            bayestopt_.p3, bayestopt_.p4);
        end
        datarabanal_hybrid;
        for i = 1:numel(options_.varobs) ex.data(:, i) = eval(options_.varobs{i}); end
        save('-mat7-binary', '../out_bvar/$f.mat', 'ex'); printf('saved out_bvar/$f.mat\n');" \
        > "out_bvar/$f.log" 2>&1 || { echo "FAILED $f, see out_bvar/$f.log"; tail -20 "out_bvar/$f.log"; exit 1; }
      grep '^saved' "out_bvar/$f.log"
    done
    rm -rf "$W"
    ;;
  *) echo "unknown variant $v"; exit 1 ;;
  esac
done
echo "done."
