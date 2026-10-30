#!/usr/bin/env bash
# run_derivatives.sh: Dynare's analytical dghx, dghu and dYss for examples 1 and 2.
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
  echo ">>> $name derivatives"
  rm -rf work_decision_rule/*
  sed 's/^stoch_simul.*/stoch_simul(order=1, irf=0, periods=0);\nvarobs y;\nidentification(order=1);/' \
    "mods/$name.mod" > "work_decision_rule/$name.mod"
  ( cd work_decision_rule && "$OCTAVE" --no-gui --eval "
    pkg load statistics datatypes;
    addpath '$DYNARE_MATLAB';
    dynare $name.mod noclearall nograph;
    D = identification.get_perturbation_params_derivs(M_, options_, estim_params_, oo_.dr, oo_.steady_state, oo_.exo_steady_state, oo_.exo_det_steady_state, 1:numel(M_.params), [], [], false);
    ex = struct();
    ex.dghx = D.dghx; ex.dghu = D.dghu; ex.dYss = D.dYss;
    ex.order_var = oo_.dr.order_var;
    if isfield(oo_.dr, 'state_var'), ex.state_var = oo_.dr.state_var; end
    ex.lead_lag_incidence = M_.lead_lag_incidence;
    ex.endo_names = M_.endo_names; ex.exo_names = M_.exo_names;
    ex.param_names = M_.param_names; ex.ys = oo_.dr.ys;
    save('-mat7-binary', '../out_decision_rule/${name}_jac.mat', 'ex');
  " > "../out_decision_rule/${name}_jac.log" 2>&1 ) || {
    echo "FAILED $name, see out_decision_rule/${name}_jac.log"
    tail -25 "out_decision_rule/${name}_jac.log"
    exit 1
  }
done
rm -rf work_decision_rule
echo "done."
