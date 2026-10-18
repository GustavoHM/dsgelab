#!/usr/bin/env bash
# run_dynare.sh: runs each reference .mod in GNU Octave + Dynare and exports oo_ to .mat.
#
# Usage:  bash run_dynare.sh [name.mod ...]      (no arguments: all of those in mods/)
#
# Each .mod carries its own `stoch_simul` options. One filters with HP, another simulates 2000
# periods. So that the comparison is between MODELS and not between reporting options,
# `stoch_simul` is rerun here with the SAME configuration for all: order 1, 40 periods of IRF,
# THEORETICAL moments (periods=0) and no filter. This is the only way to put the three on the
# same yardstick.
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

MODS=("$@")
if [ ${#MODS[@]} -eq 0 ]; then
  MODS=()
  for f in mods/*.mod; do MODS+=("$(basename "$f")"); done
fi

mkdir -p out
for m in "${MODS[@]}"; do
  base="${m%.mod}"
  echo ">>> $m"
  cp "mods/$m" .
  "$OCTAVE" --no-gui --eval "
    pkg load statistics datatypes;
    addpath '$DYNARE_MATLAB';
    dynare $m noclearall nograph;
    % same yardstick for every model, whatever the stoch_simul of the .mod
    options_.order = 1; options_.irf = 40; options_.periods = 0;
    options_.hp_filter = 0; options_.nograph = 1; options_.noprint = 1;
    options_.nomoments = 0; options_.nocorr = 0;
    [info, oo_, options_, M_] = stoch_simul(M_, options_, oo_, M_.endo_names);
    ex = struct();
    ex.steady_state = oo_.steady_state;
    ex.endo_names   = M_.endo_names;
    ex.exo_names    = M_.exo_names;
    ex.irfs         = oo_.irfs;
    ex.info         = info;
    if isfield(oo_, 'var'),  ex.variance = oo_.var;  end
    if isfield(oo_, 'gamma_y'), ex.gamma_y = oo_.gamma_y; end
    if isfield(oo_.dr, 'ghx'), ex.ghx = oo_.dr.ghx; ex.ghu = oo_.dr.ghu;
      ex.order_var = oo_.dr.order_var; ex.nstatic = M_.nstatic;
      ex.npred = M_.npred; ex.nboth = M_.nboth; ex.nfwrd = M_.nfwrd; end
    save('-mat7-binary', 'out/${base}.mat', 'ex');
    printf('saved out/${base}.mat (info=%d)\n', info);
  " > "out/${base}.log" 2>&1 || { echo "FAILED $m, see out/${base}.log"; tail -20 "out/${base}.log"; exit 1; }
  tail -1 "out/${base}.log"
  rm -rf "$m" "$base" "+$base"     # the copied .mod and what Dynare's preprocessor generates
done
echo "done."
