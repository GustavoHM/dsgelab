#!/usr/bin/env bash
# Calibration restrictions (irf_calibration, moment_calibration) on Dynare's tests/gsa/ls2003a.mod:
# at the calibration and over 512 prior draws of the sensitivity toolbox.
# Usage: bash run_restrictions.sh [ls2003_rel|ls2003_abs ...]; without arguments, runs both cases.
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

mkdir -p out_restrictions work_restrictions
here="$(pwd)"
"$PYTHON" restrictions/prepare.py
if [ "$#" -eq 0 ]; then set -- ls2003_rel ls2003_abs; fi
for name in "$@"; do
  echo ">>> $name"
  rm -rf "work_restrictions/$name" "work_restrictions/+$name" "work_restrictions/${name}_point.mat"
  ( cd work_restrictions && "$OCTAVE" --no-gui --eval "
    pkg load statistics datatypes; addpath '$DYNARE_MATLAB'; addpath '$here/restrictions';
    dynare $name noclearall nograph;
    export_restrictions('$name', '../out_restrictions/$name.mat');
    printf('saved out_restrictions/$name.mat\n');
  " > "../out_restrictions/$name.log" 2>&1 ) || { echo "FAILED $name, see out_restrictions/$name.log"; tail -20 "out_restrictions/$name.log"; exit 1; }
done
echo "done."
