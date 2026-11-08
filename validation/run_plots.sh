#!/usr/bin/env bash
# The numbers that the plots draw and the package did not compute before, from Dynare.
#   densities  draw_prior_density for the nine priors of examples/fs2000.mod, and the kernel
#              density of posterior_moments on two sets of draws made in Octave, one of them with
#              repeated values as a Metropolis chain has after rejections.
#   mcmc       an estimation of fs2000.mod with mode_check, two chains of 2,500 draws and
#              bayesian_irf, which draws Dynare's mode_check, Brooks and Gelman and posterior IRF
#              figures; then its trace plot and autocorrelogram of alp. plots/dyn_saveas.m stands
#              in for Dynare's and records what each figure draws instead of saving it, and
#              plots/export_plots.m gathers it with the draws, the mode and the IRF draws.
# Usage: bash run_plots.sh [densities|mcmc ...]   (no arguments: both)
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

if [ "$#" -eq 0 ]; then set -- densities mcmc; fi
here="$(pwd)"
mkdir -p out_plots
estimation_line() {   # writes fs2000.mod with another estimation(...) line: <destination> <line>
  python3 - mods/examples/fs2000.mod "$1" "$2" <<'PY'
import re, sys
text = open(sys.argv[1]).read()
open(sys.argv[2], "w").write(re.sub(r"^estimation\(.*$", lambda m: sys.argv[3], text, flags=re.M))
PY
}

for part in "$@"; do
  rm -rf work_plots; mkdir -p work_plots
  cp mods/examples/fs2000_data.m work_plots/
  if [ "$part" = densities ]; then
    estimation_line work_plots/fs2000.mod "estimation(order=1, datafile=fs2000_data, loglinear, \
logdata, nograph, nodisplay, plot_priors=0, mode_compute=0, mh_replic=0);"
    ( cd work_plots && "$OCTAVE" --no-gui --eval "
      pkg load statistics datatypes; addpath '$DYNARE_MATLAB';
      dynare fs2000 noclearall nograph;
      ex = struct(); ex.names = bayestopt_.name;
      for i = 1:numel(bayestopt_.name)
        [x, f] = draw_prior_density(i, bayestopt_);
        ex.prior_x(:, i) = x(:); ex.prior_f(:, i) = f(:);
      end
      randn('state', 1); rand('state', 2);
      base = 0.5 + 0.1 * randn(3000, 1);
      ex.draws1 = repelem(base, 1 + floor(3 * rand(3000, 1)));   % repeated, as after rejections
      ex.draws2 = exp(0.4 * randn(5000, 1));                     % skewed, no repeats
      [~, ~, ~, ~, ~, d1] = posterior_moments(ex.draws1, 0.9); ex.density1 = d1;
      [~, ~, ~, ~, ~, d2] = posterior_moments(ex.draws2, 0.9); ex.density2 = d2;
      save('-mat7-binary', '../out_plots/densities.mat', 'ex');
      printf('saved out_plots/densities.mat\n');
    " > ../out_plots/densities.log 2>&1 ) || { echo "FAILED, see out_plots/densities.log"; tail -20 out_plots/densities.log; exit 1; }
    grep '^saved' out_plots/densities.log
  elif [ "$part" = mcmc ]; then
    # nodisplay with graphs on: the figures are drawn, invisible, and recorded by plots/dyn_saveas.m,
    # which the .mod puts on the path (Dynare moves itself to the top when it starts).
    # riccati_tol = 1e-16 keeps the Kalman filter from switching to its steady-state gain, as in
    # run_estimation.sh. The variables after estimation(...) are those of the posterior IRFs.
    estimation_line work_plots/fs2000.mod "options_.riccati_tol = 1e-16;
addpath('$here/plots');
estimation(order=1, datafile=fs2000_data, loglinear, logdata, nodisplay, plot_priors=0, \
mode_compute=4, mode_check, mh_replic=2500, mh_nblocks=2, mh_jscale=0.8, mh_drop=0.5, \
bayesian_irf, irf=20) gp_obs gy_obs y c;"
    ( cd work_plots && "$OCTAVE" --no-gui --eval "
      pkg load statistics datatypes; addpath '$DYNARE_MATLAB';
      global PLOT_CAPTURE; PLOT_CAPTURE = {};
      dynare fs2000 noclearall;
      export_plots('fs2000', '../out_plots/mcmc.mat');
      printf('saved out_plots/mcmc.mat\n');
    " > ../out_plots/mcmc.log 2>&1 ) || { echo "FAILED, see out_plots/mcmc.log"; tail -20 out_plots/mcmc.log; exit 1; }
    grep '^saved' out_plots/mcmc.log
  elif [ "$part" = identification ]; then
    # identification of fs2000 at the prior mean with advanced=1, which draws the strength,
    # sensitivity, collinearity and pattern figures, recorded by plots/dyn_saveas.m. Dynare
    # computes the information matrix on a sample it simulates: a copy of its dsge_likelihood.m,
    # first on the path, keeps that sample when it is called for the asymptotic Hessian, and
    # export_identification.m saves it. riccati_tol as for mcmc.
    python3 identification/working_copy.py mods/examples/fs2000.mod work_plots/fs2000.mod \
      "advanced=1,max_dim_cova_group=2"
    mkdir -p work_plots/capture
    python3 - "$DYNARE_MATLAB/estimation/dsge_likelihood.m" work_plots/capture/dsge_likelihood.m <<'PY'
import sys
lines = open(sys.argv[1]).read().split("\n")
keep = ("global IDENT_SAMPLE; if isfield(options_, 'analytic_Hessian') && "
        "strcmp(options_.analytic_Hessian, 'asymptotic'), IDENT_SAMPLE = dataset_.data; end")
open(sys.argv[2], "w").write("\n".join(lines[:1] + [keep] + lines[1:]))
PY
    python3 - work_plots/fs2000.mod "$here/plots" "$here/work_plots/capture" <<'PY'
import sys
path, plots, capture = sys.argv[1:4]
text = open(path).read()
pre = (f"options_.riccati_tol = 1e-16;\noptions_.nodisplay = true;\naddpath('{plots}');\n"
       f"addpath('{capture}');\n")
open(path, "w").write(text.replace("\nidentification(", "\n" + pre + "identification(", 1))
PY
    ( cd work_plots && "$OCTAVE" --no-gui --eval "
      pkg load statistics datatypes; addpath '$DYNARE_MATLAB';
      global PLOT_CAPTURE; PLOT_CAPTURE = {};
      dynare fs2000 noclearall;
      export_identification('fs2000', '../out_plots/identification.mat');
      printf('saved out_plots/identification.mat\n');
    " > ../out_plots/identification.log 2>&1 ) || { echo "FAILED, see out_plots/identification.log"; tail -20 out_plots/identification.log; exit 1; }
    grep '^saved' out_plots/identification.log
  elif [ "$part" = decomposition ]; then
    # fs2000 at the prior mean (mode_compute=0, the initial point), with the initial condition
    # decomposition, the shock decomposition, the real-time decompositions with a forecast of 4
    # periods, and a conditional forecast of 12 periods on 2,000 draws in which gp_obs follows a
    # path for 4 periods and e_m is the controlled shock, so that e_a is drawn. The state of randn
    # just before conditional_forecast is kept for export_decomposition.m.
    estimation_line work_plots/fs2000.mod "estimation(order=1, datafile=fs2000_data, loglinear, \
logdata, nograph, nodisplay, plot_priors=0, mode_compute=0, mh_replic=0, smoother) gp_obs gy_obs c y;
options_.parameter_set = 'calibration';
initial_condition_decomposition(nodisplay) gp_obs gy_obs c y;
shock_decomposition(parameter_set=calibration, nograph) gp_obs gy_obs c y;
realtime_shock_decomposition(parameter_set=calibration, forecast=4, save_realtime=[40 80 120]) \
gp_obs gy_obs c y;
conditional_forecast_paths;
var gp_obs;
periods 1:4;
values 0.02;
end;
global CF_RNG;
CF_RNG = randn('state');
conditional_forecast(parameter_set=prior_mean, controlled_varexo=(e_m), replic=2000, periods=12);"
    ( cd work_plots && "$OCTAVE" --no-gui --eval "
      pkg load statistics datatypes; addpath '$DYNARE_MATLAB'; addpath '$here/plots';
      dynare fs2000 noclearall nograph;
      export_decomposition('../out_plots/decomposition.mat', {'gp_obs'; 'gy_obs'; 'c'; 'y'}, 2000, 12);
      printf('saved out_plots/decomposition.mat\n');
    " > ../out_plots/decomposition.log 2>&1 ) || { echo "FAILED, see out_plots/decomposition.log"; tail -20 out_plots/decomposition.log; exit 1; }
    grep '^saved' out_plots/decomposition.log
  else
    echo "unknown part $part"; exit 1
  fi
  rm -rf work_plots
done
echo "done."
