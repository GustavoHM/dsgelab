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
#             The bvar_irf lines are removed and the estimation of bvar_and_dsge.mod runs
#             without Metropolis (mh_replic=0), since neither result is compared. Before
#             bvar_forecast, a verbatim block resets Octave's generator and saves the normal
#             draws bvar.forecast is about to take, so that dsgelab can take the same ones.
#   dsge_var  Dynare 7.2's tests/dsge-var: simul_hybrid.mod simulates the data, and
#             dsgevar_forward_calibrated_lambda.mod (dsge_var=0.8) and
#             dsgevar_forward_estimated_lambda.mod (dsge_prior_weight estimated) run with their
#             models, priors, samples and optimiser (mode_compute=4) as they are, without
#             Metropolis and IRFs (mh_replic=0, no bayesian_irf). For each, the posterior mode,
#             the kernel there, its inverse Hessian and the Laplace approximation, and
#             dsge_var_likelihood at the prior mean and at perturbed points.
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
text = re.sub(r"^bvar_irf.*$", "", text, flags=re.M)
text = re.sub(r"mh_replic\s*=\s*\d+", "mh_replic = 0", text)
seed = ("verbatim;\nrandn('state', 20261103); bvar_normals = randn(1200000, 1);"
        " randn('state', 20261103);\nend;\n")
text = re.sub(r"^(bvar_forecast\()", lambda m: seed + m.group(1), text, flags=re.M)
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
        ex.normals = bvar_normals;
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
    line = m.group(0)
    line = re.sub(r"mh_replic\s*=\s*\d+", "mh_replic=0", line)
    line = re.sub(r",\s*bayesian_irf", "", line)
    return line.replace(");", ",nograph,nodisplay,plot_priors=0);")
text = re.sub(r"^estimation\(.*$", edit, text, flags=re.M)
open(sys.argv[1], "w").write(text)
PY
      rm -f "out_bvar/$f.mat"
      octave_run "$W" "dynare $f noclearall nograph;
        ex = struct(); ex.names = bayestopt_.name; ex.varobs = options_.varobs;
        ex.first_obs = options_.first_obs; ex.nobs_dataset = dataset_.nobs;
        ex.dsge_varlag = options_.dsge_varlag;
        ex.mode = oo_.posterior.optimization.mode(:);
        ex.log_density = oo_.posterior.optimization.log_density;
        ex.variance = oo_.posterior.optimization.Variance;
        ex.laplace = oo_.MarginalDensity.LaplaceApproximation;
        options_ = select_qz_criterium_value(options_);
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
