#!/usr/bin/env bash
# Pruned second-order means and autocovariances from Dynare 7.2.
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

mkdir -p out_mom_order2 work_mom_order2
for name in example1 example2; do
  echo ">>> $name, pruned order 2 moments"
  rm -rf work_mom_order2/*
  sed "s/^stoch_simul.*/stoch_simul(order=2, pruning, irf=0, periods=0);/" \
    "mods/$name.mod" > "work_mom_order2/$name.mod"
  ( cd work_mom_order2 && "$OCTAVE" --no-gui --eval "
    pkg load statistics datatypes;
    addpath '$DYNARE_MATLAB';
    dynare $name.mod noclearall nograph;
    options_.order = 2; options_.pruning = 1;
    ps = pruned_SS.pruned_state_space_system(M_, options_, oo_.dr, (1:M_.endo_nbr)', 3, 0, 0);
    ex.E_y = ps.E_y; ex.Var_y = ps.Var_y; ex.Var_yi = ps.Var_yi;
    ex.A = ps.A; ex.B = ps.B; ex.c = ps.c; ex.C = ps.C; ex.D = ps.D; ex.d = ps.d;
    ex.Varinov = ps.Varinov;
    ex.ys = oo_.dr.ys; ex.order_var = oo_.dr.order_var;
    ex.endo_names = M_.endo_names;
    save('-mat7-binary', '../out_mom_order2/$name.mat', 'ex');
  " > "../out_mom_order2/$name.log" 2>&1 ) || {
    echo "FAILED $name, see out_mom_order2/$name.log"
    tail -25 "out_mom_order2/$name.log"
    exit 1
  }
done
rm -rf work_mom_order2
echo "done."
