#!/usr/bin/env bash
# Mixed-frequency data on Dynare's examples/fs2000.mod: output growth observed only as its annual
# (four-quarter) mean, through an auxiliary variable, and inflation every quarter. The likelihood
# at two points, the smoother at the first, and the posterior mode. See mixed/prepare.py.
# Usage: bash run_mixed.sh
set -euo pipefail
cd "$(dirname "$0")"

OCTAVE="${OCTAVE:-/opt/homebrew/bin/octave}"
PYTHON="${PYTHON:-python}"
if [ -z "${DYNARE_MATLAB:-}" ]; then
  for c in /opt/homebrew/opt/dynare/lib/dynare/matlab /opt/homebrew/lib/dynare/matlab \
           /usr/local/opt/dynare/lib/dynare/matlab /usr/lib/dynare/matlab \
           /Applications/Dynare/*/matlab; do
    [ -d "$c" ] && DYNARE_MATLAB="$c" && break
  done
fi
[ -n "${DYNARE_MATLAB:-}" ] || { echo "FATAL: Dynare's matlab/ directory not found. Set DYNARE_MATLAB."; exit 1; }

rm -rf work_mixed; mkdir -p out_mixed work_mixed
"$PYTHON" mixed/prepare.py
cp mods/examples/fs2000_data.m work_mixed/
for name in fs2000_mf_point fs2000_mf_mode; do
  echo ">>> $name"
  ( cd work_mixed && "$OCTAVE" --no-gui --eval "
    pkg load statistics datatypes; addpath '$DYNARE_MATLAB';
    dynare $name noclearall nograph;
    ex = struct();
    ex.estim_names = bayestopt_.name; ex.data = dataset_.data; ex.varobs = options_.varobs;
    if strcmp('$name', 'fs2000_mf_point')
      x = [];
      for i = 1:size(estim_params_.var_exo, 1), j = estim_params_.var_exo(i,1); x(end+1,1) = sqrt(M_.Sigma_e(j,j)); end
      for i = 1:size(estim_params_.param_vals, 1), x(end+1,1) = M_.params(estim_params_.param_vals(i,1)); end
      ex.x1 = x; ex.x2 = x .* (1 + 0.05 * (-1).^(1:numel(x))');
      ex.ll1 = evaluate_likelihood(ex.x1, M_, estim_params_, oo_, options_, bayestopt_);
      ex.ll2 = evaluate_likelihood(ex.x2, M_, estim_params_, oo_, options_, bayestopt_);
      ex.smoothed_vars = oo_.SmoothedVariables;
    else
      ex.mode = oo_.posterior.optimization.mode; ex.log_density = oo_.posterior.optimization.log_density;
    end
    save('-mat7-binary', '../out_mixed/$name.mat', 'ex');
    printf('saved out_mixed/$name.mat\n');
  " > "../out_mixed/$name.log" 2>&1 ) || { echo "FAILED $name, see out_mixed/$name.log"; tail -20 "out_mixed/$name.log"; exit 1; }
  grep '^saved' "out_mixed/$name.log"
done
rm -rf work_mixed/fs2000_mf_point work_mixed/fs2000_mf_mode work_mixed/+fs2000_mf_point work_mixed/+fs2000_mf_mode
echo "done."
