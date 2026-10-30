#!/usr/bin/env bash
# Dynare OccBin inversion-filter reference on a scalar AR(1).
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

mkdir -p out_inversion work_inversion
cp mods/particle_ar1.mod work_inversion/particle_ar1.mod
( cd work_inversion && "$OCTAVE" --no-gui --eval "
  pkg load statistics datatypes;
  addpath '$DYNARE_MATLAB';
  dynare particle_ar1.mod noclearall nograph;
  options_occbin_ = occbin.set_default_options(struct(), M_, 'all');
  [M_, options_] = occbin.setup(M_, options_, options_occbin_);
  options_.occbin.solver = options_occbin_.solver;
  options_.occbin.likelihood = options_occbin_.likelihood;
  options_.varobs_id = 1;
  set_dynare_seed_local_options([], false, 'default');
  innovations = 0.2*randn(1, 30); y = zeros(1, 30);
  for t = 2:30, y(t) = 0.5*y(t-1) + innovations(t); end
  save('-mat7-binary', '../out_inversion/data.mat', 'y');
  [filtered_errs, resids, Emat, stateval, info, regime_history] = occbin.IVF_core( ...
    M_, oo_.dr, oo_.steady_state, oo_.exo_steady_state, oo_.exo_det_steady_state, ...
    options_, 1, zeros(30, 1), {'y'}, y');
  if info(1), error('IVF_core failed with code %d', info(1)); end
  ex.shocks = filtered_errs; ex.resids = resids; ex.Emat = Emat;
  ex.loglik_t = -0.5*(log(M_.Sigma_e) + 2*log(abs(squeeze(Emat))) + filtered_errs.^2/M_.Sigma_e + log(2*pi));
  ex.loglik = sum(ex.loglik_t);
  save('-mat7-binary', '../out_inversion/order1.mat', 'ex');
" > ../out_inversion/order1.log 2>&1 ) || {
  echo "FAILED inversion, see out_inversion/order1.log"
  tail -30 out_inversion/order1.log
  exit 1
}
rm -rf work_inversion
echo "done."
