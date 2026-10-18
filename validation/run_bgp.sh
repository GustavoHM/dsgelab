#!/usr/bin/env bash
# run_bgp.sh: the balanced-growth problem that Dynare sets up (bgp.write), for compare_bgp.py.
#
# Usage:  bash run_bgp.sh
#
# The files come from tests/bgp of Dynare 7.2 (fetch.sh). The working copy stops before the
# test's verbatim block (which solves with MATLAB's fsolve, missing or different in Octave). After
# preprocessing, it calls bgp.write(M_) and evaluates the generated function, bgpfun, at five fixed
# points z = [levels; growth factors]. It exports the points and the residuals to
# out_bgp/<name>.mat.
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

mkdir -p out_bgp work_bgp
for m in solow-1 ramsey-1 nk-1; do
  name="${m%-1}"
  echo ">>> $m.mod"
  rm -rf work_bgp/*
  awk '/^if ~isoctave/ || /^verbatim;/{exit} {print}' "mods/bgp/$m.mod" > "work_bgp/$name.mod"
  ( cd work_bgp && "$OCTAVE" --no-gui --eval "
    pkg load statistics datatypes;
    addpath '$DYNARE_MATLAB';
    dynare $name noclearall nograph;
    bgp.write(M_);
    n = M_.endo_nbr;
    Z = zeros(2*n, 5); F = zeros(2*n, 5);
    for j = 1:5
      Z(:, j) = [1 + 0.1*sin((1:n)' + j); 1 + 0.01*cos((1:n)' + 2*j)];
      F(:, j) = feval([M_.fname '.bgpfun'], Z(:, j));
    end
    ex = struct(); ex.Z = Z; ex.F = F; ex.endo_names = M_.endo_names;
    save('-mat7-binary', '../out_bgp/${name}.mat', 'ex');
    printf('saved out_bgp/${name}.mat\n');
  " > "../out_bgp/${name}.log" 2>&1 ) || { echo "FAILED $m, see out_bgp/${name}.log"; tail -15 "out_bgp/${name}.log"; continue; }
  grep saved "out_bgp/${name}.log" || tail -5 "out_bgp/${name}.log"
done
rm -rf work_bgp
echo "done."
