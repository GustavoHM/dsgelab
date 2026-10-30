#!/usr/bin/env bash
# The densities that posterior plots draw, from Dynare: draw_prior_density for the nine priors of
# examples/fs2000.mod, and the kernel density of posterior_moments on two sets of draws made in
# Octave, one of them with repeated values as a Metropolis chain has after rejections.
# Usage: bash run_plots.sh
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

rm -rf work_plots; mkdir -p out_plots work_plots
python3 - mods/examples/fs2000.mod work_plots/fs2000.mod <<'PY'
import re, sys
text = open(sys.argv[1]).read()
line = ("estimation(order=1, datafile=fs2000_data, loglinear, logdata, nograph, nodisplay, "
        "plot_priors=0, mode_compute=0, mh_replic=0);")
open(sys.argv[2], "w").write(re.sub(r"^estimation\(.*$", lambda m: line, text, flags=re.M))
PY
cp mods/examples/fs2000_data.m work_plots/
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
rm -rf work_plots
echo "done."
