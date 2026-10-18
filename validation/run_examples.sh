#!/usr/bin/env bash
# run_examples.sh: runs in Dynare each .mod of mods/ and mods/examples/ and exports what
# compare_import.py needs: steady state (oo_.dr.ys), parameters, shock covariance, IRFs and
# theoretical moments.
#
# Usage:  bash run_examples.sh [name ...]      (no arguments: all)
#
# Each .mod is cut down by strip_mod.py to what declares the model, and receives the same
# stoch_simul(order=1, irf=40, periods=0). The MATLAB helper files that some examples call
# (NK_baseline's steady state, example3's helper) are copied along.
set -euo pipefail
cd "$(dirname "$0")"

OCTAVE="${OCTAVE:-/opt/homebrew/bin/octave}"
PY="${PYTHON:-python3}"
if [ -z "${DYNARE_MATLAB:-}" ]; then
  for c in /opt/homebrew/opt/dynare/lib/dynare/matlab /opt/homebrew/lib/dynare/matlab \
           /usr/local/opt/dynare/lib/dynare/matlab /usr/lib/dynare/matlab \
           /Applications/Dynare/*/matlab; do
    [ -d "$c" ] && DYNARE_MATLAB="$c" && break
  done
fi
[ -n "${DYNARE_MATLAB:-}" ] || { echo "FATAL: Dynare's matlab/ directory not found. Set DYNARE_MATLAB."; exit 1; }

FILES=()
if [ $# -eq 0 ]; then
  for f in mods/*.mod mods/examples/*.mod; do FILES+=("$f"); done
else
  for n in "$@"; do FILES+=($(ls mods/"$n".mod mods/examples/"$n".mod 2>/dev/null || true)); done
fi

mkdir -p out_import work
for src in "${FILES[@]}"; do
  m="$(basename "$src")"; base="${m%.mod}"
  echo ">>> $m"
  rm -rf work/*
  "$PY" strip_mod.py "$src" "work/$m"
  cp "$(dirname "$src")"/*.m work/ 2>/dev/null || true
  ( cd work && "$OCTAVE" --no-gui --eval "
    pkg load statistics datatypes;
    addpath '$DYNARE_MATLAB';
    dynare $m noclearall nograph;
    ex = struct();
    ex.ys          = oo_.dr.ys;
    ex.endo_names  = M_.endo_names(1:M_.orig_endo_nbr);
    ex.exo_names   = M_.exo_names;
    ex.params      = M_.params;
    ex.param_names = M_.param_names;
    ex.sigma_e     = M_.Sigma_e;
    ex.irfs        = oo_.irfs;
    if isfield(oo_, 'var'), ex.variance = oo_.var; end
    save('-mat7-binary', '../out_import/${base}.mat', 'ex');
    printf('saved out_import/${base}.mat\n');
  " > "../out_import/${base}.log" 2>&1 ) || { echo "FAILED $m, see out_import/${base}.log"; tail -20 "out_import/${base}.log"; continue; }
  tail -1 "out_import/${base}.log"
done
rm -rf work
echo "done."
