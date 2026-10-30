#!/usr/bin/env bash
# First-order IRF matching on Dynare's official frequentist RBC example.
# Usage: bash run_irf.sh [0|5 ...]; without arguments, runs both cases.
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

mkdir -p out_mom work_mom
here="$(pwd)"
"$PYTHON" mom/prepare_irf.py
if [ "$#" -eq 0 ]; then set -- 0 5; fi
for suffix in "$@"; do
  name="rbc_irf_$suffix"
  echo ">>> $name"
  rm -rf "work_mom/$name"
  ( cd work_mom && "$OCTAVE" --no-gui --eval "
    pkg load statistics datatypes; addpath '$DYNARE_MATLAB'; addpath '$here/mom';
    dynare $name noclearall nograph;
    export_irf('$name', '../out_mom/$name.mat');
    printf('saved out_mom/$name.mat\n');
  " > "../out_mom/$name.log" 2>&1 ) || { echo "FAILED $name, see out_mom/$name.log"; tail -20 "out_mom/$name.log"; continue; }
done
echo "done."
