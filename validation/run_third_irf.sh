#!/usr/bin/env bash
# Third-order pruned IRFs from Dynare for examples 1 and 2.
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

mkdir -p out_third_irf work_third_irf
for name in example1 example2; do
  echo ">>> $name, order 3 IRFs"
  rm -rf work_third_irf/*
  sed "s/^stoch_simul.*/stoch_simul(order=3, pruning, irf=40, periods=0, replic=5000);/" \
    "mods/$name.mod" > "work_third_irf/$name.mod"
  ( cd work_third_irf && "$OCTAVE" --no-gui --eval "
    pkg load statistics datatypes;
    addpath '$DYNARE_MATLAB';
    dynare $name.mod noclearall nograph;
    ex = struct();
    ex.ys = oo_.dr.ys;
    ex.irfs = oo_.irfs;
    ex.order_var = oo_.dr.order_var;
    if isfield(oo_.dr, 'state_var'), ex.state_var = oo_.dr.state_var; end
    ex.lead_lag_incidence = M_.lead_lag_incidence;
    ex.endo_names = M_.endo_names; ex.exo_names = M_.exo_names;
    save('-mat7-binary', '../out_third_irf/${name}.mat', 'ex');
  " > "../out_third_irf/${name}.log" 2>&1 ) || {
    echo "FAILED $name, see out_third_irf/${name}.log"
    tail -25 "out_third_irf/${name}.log"
    exit 1
  }
done
rm -rf work_third_irf
echo "done."
