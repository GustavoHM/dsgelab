#!/usr/bin/env bash
# Dynare's analytical third-order decision-rule Jacobians for examples 1 and 2.
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
for name in example1 example2; do
  echo ">>> $name third-order derivatives"
  rm -rf work_decision_rule/*
  sed 's/^stoch_simul.*/stoch_simul(order=3, pruning, irf=0, periods=0);\nvarobs y;\nidentification(order=3);/' \
    "mods/$name.mod" > "work_decision_rule/$name.mod"
  ( cd work_decision_rule && "$OCTAVE" --no-gui --eval "
    pkg load statistics datatypes;
    addpath '$DYNARE_MATLAB';
    dynare $name.mod noclearall nograph;
    D = identification.get_perturbation_params_derivs(M_, options_, estim_params_, oo_.dr, oo_.steady_state, oo_.exo_steady_state, oo_.exo_det_steady_state, 1:numel(M_.params), 1:M_.exo_nbr, [1 2], false);
    ex = struct();
    ex.dghxxx = D.dghxxx; ex.dghxxu = D.dghxxu; ex.dghxuu = D.dghxuu; ex.dghuuu = D.dghuuu;
    ex.dghs2 = D.dghs2;
    ex.order_var = oo_.dr.order_var;
    if isfield(oo_.dr, 'state_var'), ex.state_var = oo_.dr.state_var; end
    ex.lead_lag_incidence = M_.lead_lag_incidence;
    ex.endo_names = M_.endo_names; ex.param_names = M_.param_names; ex.ys = oo_.dr.ys;
    ex.exo_names = M_.exo_names;
    save('-mat7-binary', '../out_decision_rule/${name}_jac3.mat', 'ex');
  " > "../out_decision_rule/${name}_jac3.log" 2>&1 ) || {
    echo "FAILED $name, see out_decision_rule/${name}_jac3.log"
    tail -25 "out_decision_rule/${name}_jac3.log"
    exit 1
  }
done
rm -rf work_decision_rule
echo "done."
