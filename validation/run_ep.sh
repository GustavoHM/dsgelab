#!/usr/bin/env bash
# run_ep.sh: extended path (deterministic and stochastic) in Dynare, for compare_ep.py.
#
# Usage:  bash run_ep.sh
#
# The files come from tests/ep of Dynare 7.2 (fetch.sh) and run unchanged up to the last
# simulation, except for the solver tolerance. Dynare stops at a residual of 1e-5 by default
# (options_.dynatol), and the copy asks for 1e-12, because the comparison is between algorithms.
# In burnside the copy stops at `toc` (what follows compares with a .mat file that the test does
# not ship). In linearmodel0 the use_dll option is removed, because it requires a compiler with
# GCC's options.
# Exports to out_ep/<name>.mat the paths of each variant and the shocks used.
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

TOL='options_.dynatol.f = 1e-12; options_.dynatol.x = 1e-12;'
mkdir -p out_ep work_ep
awk -v tol="$TOL" '/^T = 100;/{print tol} {print} /^toc/{exit}' mods/ep/burnside.mod > work_ep/burnside.mod
# use_dll requires a compiler with GCC's options, and without it the model is the same
awk -v tol="$TOL" '/^options_.ep.order = 0;/{print tol} {sub(/model\(use_dll\)/, "model")} {print}' \
  mods/ep/linearmodel0.mod > work_ep/linearmodel0.mod

save_burnside="ex = struct(); ex.e = e_1; ex.names = M_.endo_names;
  ex.ep0 = ts0.data; ex.sep1 = ts1.data; ex.sep1h = ts1h.data; ex.sep2 = ts2.data;
  ex.sep2full = ts2__.data; ex.sep2h = ts2h.data; ex.sep2hfull = ts2h__.data;"
save_linear="ex = struct(); ex.e = oo.exo_simul; ex.names = M_.endo_names; ex.ep0 = ts.data;
  ex.linear = oo.endo_simul';"
for m in ${ONLY:-burnside linearmodel0}; do
  echo ">>> $m.mod"
  rm -f "out_ep/$m.mat"
  if [ "$m" = burnside ]; then save="$save_burnside"; else save="$save_linear"; fi
  ( cd work_ep && "$OCTAVE" --no-gui --eval "
    pkg load statistics datatypes;
    addpath '$DYNARE_MATLAB';
    dynare $m noclearall nograph;
    $save
    save('-mat7-binary', '../out_ep/${m}.mat', 'ex');
    printf('saved out_ep/${m}.mat\n');
  " > "../out_ep/${m}.log" 2>&1 ) || { echo "FAILED $m, see out_ep/${m}.log"; tail -15 "out_ep/${m}.log"; continue; }
  grep "saved" "out_ep/${m}.log" || tail -5 "out_ep/${m}.log"
done
echo "done."
