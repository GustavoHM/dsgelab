#!/usr/bin/env bash
# run_identification.sh: Dynare's identification command at the prior mean, for
# compare_identification.py.
#
# Usage:  bash run_identification.sh
#
# Three models: kim2 and as2007 from tests/identification of Dynare 7.2, and fs2000 from the
# examples of 6.4 (fetch.sh). The working copy of each keeps the file up to its first
# identification or estimation command and ends with one identification command, with the
# brute-force collinearity search (advanced=1). kim2 keeps the tol_rank=1e-8 of Dynare's own test.
# What Dynare found is exported to out_identification/<name>.mat by identification/export_ident.m.
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

mkdir -p out_identification work_identification
here="$(pwd)"
for spec in "identification/kim2.mod:kim2:advanced=1,max_dim_cova_group=3,tol_rank=1e-8" \
            "identification/as2007.mod:as2007:advanced=1,max_dim_cova_group=3" \
            "examples/fs2000.mod:fs2000:advanced=1,max_dim_cova_group=2"; do
  IFS=: read -r src name options <<< "$spec"
  echo ">>> $name"
  rm -rf work_identification/*
  "$PYTHON" identification/working_copy.py "mods/$src" "work_identification/$name.mod" "$options"
  ( cd work_identification && "$OCTAVE" --no-gui --eval "
    pkg load statistics datatypes;
    addpath '$DYNARE_MATLAB';
    addpath '$here/identification';
    dynare $name noclearall nograph;
    export_ident('$name', '../out_identification/$name.mat');
    printf('saved out_identification/$name.mat\n');
  " > "../out_identification/$name.log" 2>&1 ) || { echo "FAILED $name, see out_identification/$name.log"; tail -15 "out_identification/$name.log"; continue; }
  grep saved "out_identification/$name.log" || tail -5 "out_identification/$name.log"
done
rm -rf work_identification
echo "done."
