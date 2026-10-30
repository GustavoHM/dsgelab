#!/usr/bin/env bash
# First-order GMM and SMM on An and Schorfheide (2007), against Dynare's +mom.
# Usage: bash run_mom.sh [case ...]; without arguments, runs all eight cases.
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
"$PYTHON" mom/prepare.py

if [ "$#" -eq 0 ]; then
  set -- anscho_GMM_0 anscho_SMM_0 anscho_GMM_4 anscho_SMM_4 \
         anscho_GMM_d1 anscho_SMM_d1 anscho_GMM_e5 anscho_SMM_e5
fi

# A second, reproducible dataset complements Dynare's distributed AnScho_MoM_data_2.mat.
( cd work_mom && "$OCTAVE" --no-gui --eval "
  pkg load statistics datatypes; addpath '$DYNARE_MATLAB';
  dynare anscho_sim1 noclearall nograph;
" > "../out_mom/anscho_sim1.log" 2>&1 )

for name in "$@"; do
  echo ">>> $name"
  rm -rf "work_mom/$name" "work_mom/$name.mod"
  "$PYTHON" mom/prepare.py
  ( cd work_mom && "$OCTAVE" --no-gui --eval "
    pkg load statistics datatypes; addpath '$DYNARE_MATLAB'; addpath '$here/mom';
    dynare $name noclearall nograph;
    export_mom('$name', '../out_mom/$name.mat');
    printf('saved out_mom/$name.mat\n');
  " > "../out_mom/$name.log" 2>&1 ) || { echo "FAILED $name, see out_mom/$name.log"; tail -15 "out_mom/$name.log"; continue; }
done
echo "done."
