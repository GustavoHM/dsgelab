#!/usr/bin/env bash
# run_dynare.sh: runs in Dynare each .mod of mods/ and exports what compare.py needs, as
# ../run_examples.sh does for Dynare's own examples.
#
# Usage:  bash run_dynare.sh [name ...]      (no arguments: all)
#
# Each .mod is cut down by ../strip_mod.py to what declares the model, and receives the same
# stoch_simul(order=1, irf=40, periods=0) in place of the MMB's own commands.
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

# EACZ_GEM03 declares an innovation (interest_) that no equation uses, which Dynare 7 refuses
# without nostrict. Dynare then drops it, and compare.py matches shocks by name.
NOSTRICT=" EACZ_GEM03 "

FILES=()
if [ $# -eq 0 ]; then
  for f in mods/*.mod; do FILES+=("$f"); done
else
  for n in "$@"; do FILES+=("mods/$n.mod"); done
fi

mkdir -p out work
failed=0
for src in "${FILES[@]}"; do
  m="$(basename "$src")"; base="${m%.mod}"
  echo ">>> $m"
  rm -rf work/*
  "$PY" ../strip_mod.py "$src" "work/$m"
  opts="noclearall nograph"
  case "$NOSTRICT" in *" $base "*) opts="$opts nostrict" ;; esac
  ( cd work && "$OCTAVE" --no-gui --eval "
    pkg load statistics datatypes;
    addpath '$DYNARE_MATLAB';
    dynare $m $opts;
    ex = struct();
    ex.ys          = oo_.dr.ys;
    ex.endo_names  = M_.endo_names(1:M_.orig_endo_nbr);
    ex.exo_names   = M_.exo_names;
    ex.params      = M_.params;
    ex.param_names = M_.param_names;
    ex.sigma_e     = M_.Sigma_e;
    ex.irfs        = oo_.irfs;
    if isfield(oo_, 'var'), ex.variance = oo_.var; end
    save('-mat7-binary', '../out/${base}.mat', 'ex');
    printf('saved out/${base}.mat\n');
  " > "../out/${base}.log" 2>&1 ) || { echo "FAILED $m, see out/${base}.log"; tail -20 "out/${base}.log"; failed=1; continue; }
  grep "^saved" "out/${base}.log"
done
rm -rf work
echo "done."
exit $failed
