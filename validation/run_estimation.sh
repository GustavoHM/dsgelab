#!/usr/bin/env bash
# run_estimation.sh: Dynare's references for the estimation of fs2000.mod (Schorfheide, 2000).
#
# Usage:  bash run_estimation.sh [calib|mode|mh|unitroot|condfc ...]   (no arguments: all)
#
# fs2000.mod from Dynare's examples/ folder runs with the model, the priors and the data as they
# are. Only the `estimation(...)` line changes, in three variants:
#   calib  no optimisation, at the initial point (the prior means, Dynare's rule): likelihood,
#          prior density, posterior kernel (evaluate_likelihood, evaluate_posterior_kernel),
#          the smoother, the forecast (forecast=8), shock_decomposition and
#          realtime_shock_decomposition at that point;
#   mode   the posterior mode (csminwel, mode_compute=4) and the Laplace approximation;
#   mh     Metropolis-Hastings, 2 chains of 10,000, mh_jscale 0.8: posterior means and the
#          modified harmonic mean.
# (The list of variables after estimation(...) avoids Dynare's interactive question about which
# variables to smooth.)
# options_.riccati_tol = 1e-16 turns off the switch to the steady-state gain of the Kalman filter,
# which Dynare makes by default when the variance converges to 1e-6 and which moves the likelihood
# in the seventh decimal place.
# Exports to out_estimation/fs2000_<variant>.mat, together with the data Dynare used.
# Two more variants: unitroot (the diffuse filter, in a model written here) and condfc (Dynare's
# conditional forecast test, tests/conditional_forecasts/1/fs2000_cal.mod).
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

COMMON="order=1, datafile=fs2000_data, loglinear, logdata, nograph, nodisplay, plot_priors=0"
est() {   # the estimation(...) line of each variant (macOS's bash has no associative arrays)
  case "$1" in
    calib) echo "estimation($COMMON, mode_compute=0, mh_replic=0, smoother, forecast=8) m P c e W R k d n l gy_obs gp_obs y dA; shock_decomposition(parameter_set=calibration) gp_obs gy_obs c y; realtime_shock_decomposition(parameter_set=calibration) gp_obs gy_obs c y;" ;;
    mode)  echo "estimation($COMMON, mode_compute=4, mh_replic=0);" ;;
    mh)    echo "estimation($COMMON, mode_compute=4, mh_replic=10000, mh_nblocks=2, mh_jscale=0.8, mh_drop=0.5, nodiagnostic);" ;;
  esac
}

VARIANTS=("$@")
[ ${#VARIANTS[@]} -eq 0 ] && VARIANTS=(calib mode mh unitroot condfc)
mkdir -p out_estimation

octave_run() {   # octave_run <directory> <code>
  ( cd "$1" && "$OCTAVE" --no-gui --eval "pkg load statistics datatypes; addpath '$DYNARE_MATLAB'; $2" )
}

for v in "${VARIANTS[@]}"; do
  if [ "$v" = unitroot ]; then
    # estimation/unitroot.mod, written here: an observed price level (unit root) and Dynare's
    # diffuse_filter. Likelihood at two points, and the smoother.
    echo ">>> unitroot"
    W=work_estimation_unitroot; rm -rf "$W"; mkdir -p "$W"
    python3 - estimation/unitroot.mod "$W/unitroot.mod" <<'PY'
import re, sys
t = open(sys.argv[1]).read()
open(sys.argv[2], "w").write(re.sub(r"^estimation\(", "options_.riccati_tol = 1e-16;\nestimation(", t, flags=re.M))
PY
    cp estimation/unitroot_data.csv "$W/"
    rm -f out_estimation/unitroot.mat
    octave_run "$W" "dynare unitroot noclearall nograph;
      ex = struct(); ex.names = bayestopt_.name;
      ex.x1 = [0.01;0.02;0.5;0.7]; ex.x2 = [0.015;0.025;0.3;0.9];
      ex.ll1 = evaluate_likelihood(ex.x1, M_, estim_params_, oo_, options_, bayestopt_);
      ex.ll2 = evaluate_likelihood(ex.x2, M_, estim_params_, oo_, options_, bayestopt_);
      ex.sv = oo_.SmoothedVariables; ex.ss = oo_.SmoothedShocks; ex.data = dataset_.data;
      ex.varobs = options_.varobs;
      save('-mat7-binary', '../out_estimation/unitroot.mat', 'ex'); printf('saved out_estimation/unitroot.mat\n');" \
      > out_estimation/unitroot.log 2>&1 || { echo "FAILED unitroot"; tail -20 out_estimation/unitroot.log; }
    tail -1 out_estimation/unitroot.log; rm -rf "$W"; continue
  fi
  if [ "$v" = condfc ]; then
    # Dynare's tests/conditional_forecasts/1/fs2000_cal.mod, without the plot: the two controlled
    # shocks, paths for gy_obs and gp_obs in periods 1 to 5.
    echo ">>> condfc"
    W=work_estimation_condfc; rm -rf "$W"; mkdir -p "$W"
    grep -v "plot_conditional_forecast" mods/forecast/fs2000_cal.mod > "$W/fs2000_cal.mod"
    rm -f out_estimation/condfc.mat
    octave_run "$W" "dynare fs2000_cal noclearall nograph;
      ex = struct(); ex.cond = oo_.conditional_forecast.cond.Mean; ex.endo_names = M_.endo_names;
      save('-mat7-binary', '../out_estimation/condfc.mat', 'ex'); printf('saved out_estimation/condfc.mat\n');" \
      > out_estimation/condfc.log 2>&1 || { echo "FAILED condfc"; tail -20 out_estimation/condfc.log; }
    tail -1 out_estimation/condfc.log; rm -rf "$W"; continue
  fi
  echo ">>> fs2000 — $v"
  W=work_estimation_$v; rm -rf "$W"; mkdir -p "$W"
  EST_LINE="$(est "$v")" python3 - mods/examples/fs2000.mod "$W/fs2000.mod" <<'PY'
import os, re, sys
src, dst = sys.argv[1], sys.argv[2]
text = open(src).read()
line = "options_.riccati_tol = 1e-16;\n" + os.environ["EST_LINE"].replace("; ", ";\n")
text = re.sub(r"^estimation\(.*$", lambda m: line, text, flags=re.M)
open(dst, "w").write(text)
PY
  cp mods/examples/fs2000_data.m "$W/"
  rm -f "out_estimation/fs2000_$v.mat"
  ( cd "$W" && "$OCTAVE" --no-gui --eval "
    pkg load statistics datatypes;
    addpath '$DYNARE_MATLAB';
    dynare fs2000 noclearall nograph;
    ex = struct();
    ex.param_names = M_.param_names; ex.params = M_.params;
    ex.exo_names = M_.exo_names; ex.endo_names = M_.endo_names;
    ex.sigma_e = M_.Sigma_e;
    ex.data = dataset_.data; ex.varobs = options_.varobs;
    ex.estim_names = bayestopt_.name;
    if strcmp('$v', 'calib')
      % the calibrated point, in Dynare's order: shock sds, measurement sds, then parameters
      x = [];
      for i = 1:size(estim_params_.var_exo, 1), j = estim_params_.var_exo(i,1); x(end+1,1) = sqrt(M_.Sigma_e(j,j)); end
      for i = 1:size(estim_params_.var_endo, 1), j = estim_params_.var_endo(i,1); x(end+1,1) = sqrt(M_.H(j,j)); end
      for i = 1:size(estim_params_.param_vals, 1), x(end+1,1) = M_.params(estim_params_.param_vals(i,1)); end
      ex.calib_point = x;
      ex.loglik = evaluate_likelihood(x, M_, estim_params_, oo_, options_, bayestopt_);
      ex.logpost = evaluate_posterior_kernel(x, M_, estim_params_, oo_, options_, bayestopt_);
      ex.smoothed_vars = oo_.SmoothedVariables; ex.smoothed_shocks = oo_.SmoothedShocks;
      ex.forecast = oo_.forecast; ex.shock_decomposition = oo_.shock_decomposition;
      ex.realtime_pool = oo_.realtime_shock_decomposition.pool;
      ex.order_var = oo_.dr.order_var;
    end
    if isfield(oo_, 'posterior') && isfield(oo_.posterior, 'optimization')
      ex.mode = oo_.posterior.optimization.mode; ex.log_density = oo_.posterior.optimization.log_density;
      ex.variance = oo_.posterior.optimization.Variance;
    end
    if isfield(oo_, 'MarginalDensity'), ex.marginal = oo_.MarginalDensity; end
    if isfield(oo_, 'posterior_mean'), ex.posterior_mean = oo_.posterior_mean; end
    if isfield(oo_, 'posterior_hpdinf'), ex.hpdinf = oo_.posterior_hpdinf; ex.hpdsup = oo_.posterior_hpdsup; end
    save('-mat7-binary', '../out_estimation/fs2000_$v.mat', 'ex');
    printf('saved out_estimation/fs2000_$v.mat\n');
  " > "../out_estimation/fs2000_$v.log" 2>&1 ) || { echo "FAILED $v, see out_estimation/fs2000_$v.log"; tail -20 "out_estimation/fs2000_$v.log"; }
  tail -1 "out_estimation/fs2000_$v.log"
  rm -rf "$W"
done
echo "done."
