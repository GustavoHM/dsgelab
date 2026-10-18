#!/usr/bin/env bash
# run_foresight.sh: perfect foresight in Dynare, with and without expectation errors.
#
# Usage:  bash run_foresight.sh
#
# The files come from tests/deterministic_simulations of Dynare 7.2 (fetch.sh) and run unchanged,
# except for the plot commands (rplot), which the copy removes, and the solver tolerance. Exports
# to out_foresight/<name>.mat the path oo_.endo_simul, the exogenous oo_.exo_simul and the names.
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

mkdir -p out_foresight work
for m in pfwee pfwee_constant_sim_length rbc_det1 rbc_det2 rbc_det3; do
  echo ">>> $m.mod"
  rm -rf work/*
  # Dynare's solver stops at a residual of 1e-5 by default (options_.dynatol). The comparison is
  # between models, and the copy asks for 1e-12 before each setup and solver. The terminal steady
  # state, which the setup with expectation errors recomputes at each piece of news, stops at
  # options_.simul.steady_tolf (eps^(1/3), copied from solve_tolf at initialisation), and the copy
  # tightens it in the same way.
  grep -v "^\s*rplot" "mods/foresight/$m.mod" \
    | sed -E 's/^(\s*)(perfect_foresight_setup|perfect_foresight_with_expectation_errors_setup|perfect_foresight_solver|perfect_foresight_with_expectation_errors_solver)/\1options_.dynatol.f = 1e-12; options_.dynatol.x = 1e-12; options_.solve_tolf = 1e-14; options_.solve_tolx = 1e-14; options_.simul.steady_tolf = 1e-14; options_.simul.steady_tolx = 1e-14;\n\1\2/' \
    | awk '{print} /^pfwee_simul = oo_.endo_simul;/{exit}' > "work/$m.mod"
  # (pfwee.mod ends with a manual replication and a test of EXACT equality between the two, which
  # no longer holds with another tolerance. The copy stops at pfwee_simul, which is what is
  # exported.)
  rm -f "out_foresight/$m.mat"
  cp mods/foresight/pfwee.csv work/
  ( cd work && "$OCTAVE" --no-gui --eval "
    pkg load statistics datatypes;
    addpath '$DYNARE_MATLAB';
    dynare $m noclearall nograph;
    ex = struct();
    if exist('pfwee_simul', 'var'), ex.endo_simul = pfwee_simul; else ex.endo_simul = oo_.endo_simul; end
    ex.exo_simul = oo_.exo_simul;
    ex.endo_names = M_.endo_names; ex.exo_names = M_.exo_names;
    save('-mat7-binary', '../out_foresight/${m}.mat', 'ex');
    printf('saved out_foresight/${m}.mat\n');
  " > "../out_foresight/${m}.log" 2>&1 ) || { echo "FAILED $m, see out_foresight/${m}.log"; tail -15 "out_foresight/${m}.log"; continue; }
  tail -1 "out_foresight/${m}.log"
done
rm -rf work
echo "done."
