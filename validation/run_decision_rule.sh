#!/usr/bin/env bash
# run_decision_rule.sh: Dynare's second- and third-order tensors for examples 1 and 2.
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

mkdir -p out_decision_rule work_decision_rule
for order in 2 3; do
  suffix=""
  [ "$order" -eq 2 ] || suffix="_o3"
  for name in example1 example2; do
    echo ">>> $name, order $order"
    rm -rf work_decision_rule/*
    sed "s/^stoch_simul.*/stoch_simul(order=$order, irf=0, periods=0);/" \
      "mods/$name.mod" > "work_decision_rule/$name.mod"
    ( cd work_decision_rule && "$OCTAVE" --no-gui --eval "
      pkg load statistics datatypes;
      addpath '$DYNARE_MATLAB';
      dynare $name.mod noclearall nograph;
      ex = struct();
      ex.ys = oo_.dr.ys;
      ex.ghx = oo_.dr.ghx; ex.ghu = oo_.dr.ghu;
      ex.ghxx = oo_.dr.ghxx; ex.ghuu = oo_.dr.ghuu; ex.ghxu = oo_.dr.ghxu;
      ex.ghs2 = oo_.dr.ghs2;
      if isfield(oo_.dr, 'ghuuu')
        ex.ghuuu = oo_.dr.ghuuu; ex.ghxuu = oo_.dr.ghxuu;
        ex.ghxxu = oo_.dr.ghxxu; ex.ghxxx = oo_.dr.ghxxx;
        ex.ghxss = oo_.dr.ghxss; ex.ghuss = oo_.dr.ghuss;
      end
      ex.order_var = oo_.dr.order_var;
      if isfield(oo_.dr, 'state_var'), ex.state_var = oo_.dr.state_var; end
      ex.lead_lag_incidence = M_.lead_lag_incidence;
      ex.endo_names = M_.endo_names; ex.exo_names = M_.exo_names;
      save('-mat7-binary', '../out_decision_rule/${name}${suffix}.mat', 'ex');
    " > "../out_decision_rule/${name}${suffix}.log" 2>&1 ) || {
      echo "FAILED $name, see out_decision_rule/${name}${suffix}.log"
      tail -25 "out_decision_rule/${name}${suffix}.log"
      exit 1
    }
  done
done
rm -rf work_decision_rule
echo "done."
