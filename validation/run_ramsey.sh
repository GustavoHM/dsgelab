#!/usr/bin/env bash
# run_ramsey.sh: runs the optimal policy models in Octave + Dynare and exports oo_ to .mat.
#
# Usage:  bash run_ramsey.sh
#
# Three models: ramsey/nk_linear.mod and ramsey/rbc_labortax.mod, written here, and
# mods/ramsey/nk_ramsey_model.mod, from Dynare's own test suite (downloaded by fetch.sh). Dynare's
# .mod runs WITHOUT changes. After it, stoch_simul is rerun with 40 periods of IRF, because the
# original asks for irf=0. The original does not set the order and Dynare uses 2 by default,
# whereas the comparison is at first order. ramsey_model has already left M_ and options_ with the
# planner's problem, and stoch_simul inherits it.
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

mkdir -p out
for src in ramsey/nk_linear.mod ramsey/rbc_labortax.mod mods/ramsey/nk_ramsey_model.mod; do
  [ -f "$src" ] || { echo "missing $src, run fetch.sh"; exit 1; }
  m="$(basename "$src")"; base="${m%.mod}"
  echo ">>> $m"
  cp "$src" .
  "$OCTAVE" --no-gui --eval "
    pkg load statistics datatypes;
    addpath '$DYNARE_MATLAB';
    dynare $m noclearall nograph;
    options_.order = 1; options_.irf = 40; options_.nograph = 1; options_.noprint = 1; options_.periods = 0;
    [info, oo_, options_, M_] = stoch_simul(M_, options_, oo_, M_.endo_names);
    ex = struct();
    ex.steady_state = oo_.steady_state;
    ex.ys = oo_.dr.ys;
    ex.endo_names   = M_.endo_names;
    ex.exo_names    = M_.exo_names;
    ex.orig_endo_nbr = M_.orig_endo_nbr;
    ex.irfs         = oo_.irfs;
    ex.sigma_e      = M_.Sigma_e;
    ex.info         = info;
    save('-mat7-binary', 'out/${base}.mat', 'ex');
    printf('saved out/${base}.mat (info=%d)\n', info);
  " > "out/${base}.log" 2>&1 || { echo "FAILED $m, see out/${base}.log"; tail -30 "out/${base}.log"; exit 1; }
  tail -1 "out/${base}.log"
  rm -rf "$m" "$base" "+$base" "${base}.log"
done
echo "done."
