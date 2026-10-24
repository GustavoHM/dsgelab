#!/usr/bin/env bash
# run_dynare.sh: solves each selected DSGE_mod file and exports the reference solution.
# Usage: bash run_dynare.sh [name ...] (no arguments: all)
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
  for f in mods/*.mod; do FILES+=("$f"); done
else
  for n in "$@"; do FILES+=("mods/$n.mod"); done
fi

mkdir -p out work prepared
failed=0
for src in "${FILES[@]}"; do
  m="$(basename "$src")"; base="${m%.mod}"
  echo ">>> $m"
  rm -rf work/*
  case " $base " in
    *" Aguiar_Gopinath_2007 "*|*" Hansen_1985 "*)
      "$PY" prepare.py "$src" "prepared/$m"
      "$PY" ../strip_mod.py "prepared/$m" "work/$m"
      ;;
    *) "$PY" ../strip_mod.py "$src" "work/$m" ;;
  esac
  cp mods/*.m work/ 2>/dev/null || true
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
    save('-mat7-binary', '../out/${base}.mat', 'ex');
    printf('saved out/${base}.mat\n');
  " > "../out/${base}.log" 2>&1 ) || { echo "FAILED $m, see out/${base}.log"; tail -20 "out/${base}.log"; failed=1; continue; }
  grep "^saved" "out/${base}.log"
done
rm -rf work
echo "done."
exit $failed
