#!/usr/bin/env bash
# run_policy.sh: Dynare's references for optimal simple rules, discretion and welfare.
#
# Usage:  bash run_policy.sh
#
#   mods/ramsey/osr_example.mod      Dynare's tests/optimal_policy/OSR, without the 3rd osr call
#   ramsey/nk_osr_planner.mod        osr with planner_objective (loss = minus welfare), written here
#   mods/ramsey/dennis_1.mod         Dynare's tests/discretionary_policy, unchanged
#   mods/ramsey/nk_ramsey_model.mod  Ramsey welfare at 2nd order (the .mod's default order)
#   ramsey/rbc_labortax.mod          idem, with a capital stock, here with order=2
#
# Exports to out_policy/<name>.mat the decision rule (ghx, ghu, order_var, state_var), the
# steady state, the parameters, the covariance, the osr result and
# evaluate_planner_objective.
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

mkdir -p out_policy work
run() {   # run <source.mod> <extra command before the export>
  src="$1"; extra="$2"; m="$(basename "$src")"; base="${m%.mod}"
  echo ">>> $m"
  rm -rf work/*; cp "$src" work/
  ( cd work && "$OCTAVE" --no-gui --eval "
    pkg load statistics datatypes;
    addpath '$DYNARE_MATLAB';
    dynare $m noclearall nograph;
    $extra
    ex = struct();
    ex.ys = oo_.dr.ys; ex.ghx = oo_.dr.ghx; ex.ghu = oo_.dr.ghu;
    ex.order_var = oo_.dr.order_var; ex.lead_lag_incidence = M_.lead_lag_incidence;
    if isfield(oo_.dr, 'state_var'), ex.state_var = oo_.dr.state_var; end
    if isfield(oo_.dr, 'ghxx'), ex.ghxx = oo_.dr.ghxx; ex.ghuu = oo_.dr.ghuu; ex.ghxu = oo_.dr.ghxu; ex.ghs2 = oo_.dr.ghs2; end
    ex.endo_names = M_.endo_names; ex.orig_endo_nbr = M_.orig_endo_nbr; ex.exo_names = M_.exo_names;
    ex.params = M_.params; ex.param_names = M_.param_names; ex.sigma_e = M_.Sigma_e;
    if isfield(oo_, 'osr'), ex.osr = oo_.osr; ex.osr_weights = full(M_.osr.variable_weights); end
    if isfield(oo_, 'planner_objective_value'), ex.welfare = oo_.planner_objective_value; end
    if ~isempty(oo_.exo_simul), ex.exo_simul_first = oo_.exo_simul(1,:); end
    save('-mat7-binary', '../out_policy/${base}.mat', 'ex');
    printf('saved out_policy/${base}.mat\n');
  " > "../out_policy/${base}.log" 2>&1 ) || { echo "FAILED $m, see out_policy/${base}.log"; tail -25 "out_policy/${base}.log"; return 0; }
  tail -1 "out_policy/${base}.log"
}

# The third osr call in the file uses opt_algo=1 (fminunc), which in Octave requires the optim
# package. The copy removes it and keeps the first two (csminwel, with and without the analytical
# derivative).
mkdir -p tmp_osr && grep -v "opt_algo=1" mods/ramsey/osr_example.mod > tmp_osr/osr_example.mod
run tmp_osr/osr_example.mod ""
rm -rf tmp_osr
run ramsey/nk_osr_planner.mod ""
run mods/ramsey/dennis_1.mod "oo_.planner_objective_value = evaluate_planner_objective(M_, options_, oo_);"
run mods/ramsey/nk_ramsey_model.mod ""
sed 's/stoch_simul(order=1, irf=40, nograph);/stoch_simul(order=2, irf=40, nograph);\nevaluate_planner_objective;/' \
    ramsey/rbc_labortax.mod > work_rbc_labortax_o2.mod
mkdir -p tmp_o2 && mv work_rbc_labortax_o2.mod tmp_o2/rbc_labortax_o2.mod
run tmp_o2/rbc_labortax_o2.mod ""
rm -rf tmp_o2 work
echo "done."
