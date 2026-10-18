#!/usr/bin/env bash
# run_occbin.sh: OccBin simulations in Dynare, for compare_occbin.py.
#
# Usage:  bash run_occbin.sh
#
# The files come from tests/occbin of Dynare 7.2 (fetch.sh). The model, the steady state, the
# surprise shocks and the occbin_solver options are those of the test. The working copy
# (work_occbin/, which is also what compare_occbin.py reads) joins the .inc to the .mod and cuts
# what comes after the simulation (plots, smoother). In dynrbc, only the impulse scenario
# (option=1) is kept.
# Exports to out_occbin/<name>.mat the piecewise and linear paths, in levels, and the regime
# history.
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

mkdir -p out_occbin work_occbin
"${PYTHON:-python3}" - <<'PY'
import re
src = "mods/occbin"
# borrcon: the .mod up to the include, then the whole .inc (it ends at occbin_solver)
mod = open(f"{src}/borrcon.mod").read().split('@#include "borrcon_common.inc"')[0]
open("work_occbin/borrcon.mod", "w").write(mod + open(f"{src}/borrcon_common.inc").read())
# dynrbc: the .inc up to "// run occbin simulations", then the option=1 block and the solver
mod = open(f"{src}/dynrbc.mod").read().split('@#include "dynrbc_common.inc"')[0]
inc = open(f"{src}/dynrbc_common.inc").read()
head = inc.split("// run occbin simulations")[0]
shocks = re.search(r"if option==1\s*(shocks\(surprise,overwrite\);.*?end;)", inc, re.S).group(1)
solver = re.search(r"occbin_solver\(simul_debug[^;]*;", inc).group(0).replace("simul_debug,", "")
# occbin_setup prepares the simulator. The test's option (smoother_debug) only applies to the
# smoother.
open("work_occbin/dynrbc.mod", "w").write(mod + head + shocks + "\noccbin_setup;\n" + solver + "\n")
PY
for m in ${ONLY:-borrcon dynrbc}; do
  echo ">>> $m.mod"
  rm -f "out_occbin/$m.mat"
  ( cd work_occbin && "$OCTAVE" --no-gui --eval "
    pkg load statistics datatypes;
    addpath '$DYNARE_MATLAB';
    dynare $m noclearall nograph;
    oo_ = occbin.unpack_simulations(M_, oo_, options_);
    ex = struct();
    n = M_.endo_nbr;
    T = numel(oo_.occbin.endo_piecewise.(strtrim(M_.endo_names{1})));
    ex.piecewise = zeros(T, n); ex.linear = zeros(T, n); ex.ys = oo_.steady_state;
    for j = 1:n
      v = strtrim(M_.endo_names{j});
      ex.piecewise(:, j) = oo_.occbin.endo_piecewise.(v);
      ex.linear(:, j) = oo_.occbin.endo_linear.(v);
    end
    ex.endo_names = M_.endo_names;
    rh = oo_.occbin.simul.regime_history;
    f = fieldnames(rh);
    for j = 1:numel(f)
      ex.(['rh_' f{j}]) = {rh.(f{j})};
    end
    save('-mat7-binary', '../out_occbin/${m}.mat', 'ex');
    printf('saved out_occbin/${m}.mat\n');
  " > "../out_occbin/${m}.log" 2>&1 ) || { echo "FAILED $m, see out_occbin/${m}.log"; tail -15 "out_occbin/${m}.log"; continue; }
  tail -1 "out_occbin/${m}.log"
done
echo "done."
