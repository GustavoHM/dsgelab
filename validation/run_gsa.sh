#!/usr/bin/env bash
# Stability mapping (Dynare's sensitivity toolbox, stab=1) on Dynare's tests/gsa/ls2003a.mod:
# 1024 draws from the prior (prior_range=0), 1024 uniform draws over its range (prior_range=1),
# and 1024 draws from the prior with rho_A uniform on [0.5, 1.1], which makes some draws explosive.
# For each draw, whether it has a unique stable solution, is indeterminate, explosive or has no
# solution; and the Smirnov tests and correlations of the Monte Carlo filtering on those groups.
# Usage: bash run_gsa.sh [ls2003_prior|ls2003_range|ls2003_explosive ...]; without arguments, all three.
set -euo pipefail
cd "$(dirname "$0")"

OCTAVE="${OCTAVE:-/opt/homebrew/bin/octave}"
PYTHON="${PYTHON:-python}"
if [ -z "${DYNARE_MATLAB:-}" ]; then
  for c in /opt/homebrew/opt/dynare/lib/dynare/matlab /opt/homebrew/lib/dynare/matlab \
           /usr/local/opt/dynare/lib/dynare/matlab /usr/lib/dynare/matlab \
           /Applications/Dynare/*/matlab; do
    [ -d "$c" ] && DYNARE_MATLAB="$c" && break
  done
fi
[ -n "${DYNARE_MATLAB:-}" ] || { echo "FATAL: Dynare's matlab/ directory not found. Set DYNARE_MATLAB."; exit 1; }

mkdir -p out_gsa work_gsa
here="$(pwd)"
"$PYTHON" gsa/prepare.py
if [ "$#" -eq 0 ]; then set -- ls2003_prior ls2003_range ls2003_explosive; fi
for name in "$@"; do
  echo ">>> $name"
  rm -rf "work_gsa/$name" "work_gsa/+$name"
  ( cd work_gsa && "$OCTAVE" --no-gui --eval "
    pkg load statistics datatypes; addpath '$DYNARE_MATLAB'; addpath '$here/gsa';
    dynare $name noclearall nograph;
    export_gsa('$name', '../out_gsa/$name.mat');
    printf('saved out_gsa/$name.mat\n');
  " > "../out_gsa/$name.log" 2>&1 ) || { echo "FAILED $name, see out_gsa/$name.log"; tail -20 "out_gsa/$name.log"; exit 1; }
  grep -E "prior support|saved" "out_gsa/$name.log"
done
echo "done."
