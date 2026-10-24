#!/usr/bin/env bash
# run_filtered.sh: Dynare's theoretical moments of HP- and band-pass-filtered variables, for
# compare_filtered.py.
#
# Usage:  bash run_filtered.sh
#
# Five models: example1, example2 and RBC_baseline (the models of run_dynare.sh), fs2000 from the
# examples of Dynare 6.4 (fetch.sh), and estimation/unitroot.mod, whose price level has a unit
# root. strip_mod.py keeps each file's model, and stoch_simul then runs twice at order 1 with five
# autocorrelations: with hp_filter=1600 and with bandpass_filter=[6 32]. The moments and the
# variance decomposition go to out_filtered/<name>_<filter>.mat.
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

mkdir -p out_filtered work_filtered
for src in mods/example1.mod mods/example2.mod mods/RBC_baseline.mod mods/examples/fs2000.mod \
           estimation/unitroot.mod; do
  name="$(basename "$src" .mod)"
  echo ">>> $name"
  rm -rf work_filtered/*
  "$PYTHON" strip_mod.py "$src" "work_filtered/$name.mod"
  ( cd work_filtered && "$OCTAVE" --no-gui --eval "
    pkg load statistics datatypes;
    addpath '$DYNARE_MATLAB';
    dynare $name noclearall nograph;
    names = M_.endo_names(1:M_.orig_endo_nbr);
    filters = {'hp', 'bp'};
    for f = 1:2
      opt = options_;
      opt.order = 1; opt.irf = 0; opt.periods = 0; opt.ar = 5; opt.nograph = 1;
      opt.noprint = 1; opt.nomoments = 0; opt.nocorr = 0; opt.nodecomposition = 0;
      opt.hp_filter = 0; opt.bandpass.indicator = false;
      if f == 1
        opt.hp_filter = 1600;
      else
        opt.bandpass.indicator = true; opt.bandpass.passband = [6; 32];
      end
      [info, oo, opt] = stoch_simul(M_, opt, oo_, names);
      ex = struct();
      ex.names = names; ex.exo_names = M_.exo_names; ex.Sigma_e = M_.Sigma_e;
      ex.steady_state = oo.steady_state; ex.endo_names = M_.endo_names;
      ex.var = oo.var; ex.autocorr = oo.autocorr; ex.info = info;
      if isfield(oo, 'variance_decomposition'), ex.decomposition = oo.variance_decomposition; end
      save('-mat7-binary', ['../out_filtered/${name}_' filters{f} '.mat'], 'ex');
      printf('saved out_filtered/${name}_%s.mat (info=%d)\n', filters{f}, info);
    end
  " > "../out_filtered/$name.log" 2>&1 ) || { echo "FAILED $name, see out_filtered/$name.log"; tail -15 "out_filtered/$name.log"; continue; }
  grep saved "out_filtered/$name.log" || tail -5 "out_filtered/$name.log"
done
rm -rf work_filtered
echo "done."
