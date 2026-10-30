#!/usr/bin/env bash
# Bootstrap particle-filter likelihood on a scalar AR(1), at first and second order.
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

mkdir -p out_particle work_particle
for order in 1 2; do
  echo ">>> particle AR(1), order $order"
  rm -rf work_particle/*
  sed "s/^stoch_simul.*/stoch_simul(order=$order, irf=0, periods=0);/" \
    mods/particle_ar1.mod > work_particle/particle_ar1.mod
  ( cd work_particle && "$OCTAVE" --no-gui --eval "
    pkg load statistics datatypes;
    addpath '$DYNARE_MATLAB';
    dynare particle_ar1.mod noclearall nograph;
    set_dynare_seed_local_options([], false, 'default');
    innovations = 0.2*randn(1, 30); y = zeros(1, 30);
    for t = 2:30, y(t) = 0.5*y(t-1) + innovations(t); end
    y = y + 0.2*randn(1, 30);
    save('-mat7-binary', '../out_particle/data.mat', 'y');
    ReducedForm = struct();
    ReducedForm.mf0 = 1; ReducedForm.mf1 = 1;
    ReducedForm.ghx = oo_.dr.ghx; ReducedForm.ghu = oo_.dr.ghu;
    ReducedForm.constant = oo_.dr.ys;
    ReducedForm.Q = M_.Sigma_e; ReducedForm.H = 0.2^2;
    ReducedForm.StateVectorMean = oo_.dr.ys;
    ReducedForm.StateVectorVariance = 0.04/(1-0.5^2);
    ReducedForm.state_variables_steady_state = oo_.dr.ys;
    ReducedForm.steadystate = oo_.dr.ys;
    ReducedForm.use_k_order_solver = false;
    if $order == 2
      if isfield(oo_.dr, 'ghxx')
        ReducedForm.ghxx = oo_.dr.ghxx; ReducedForm.ghuu = oo_.dr.ghuu;
        ReducedForm.ghxu = oo_.dr.ghxu;
      else
        ReducedForm.ghxx = zeros(M_.endo_nbr, M_.nspred^2);
        ReducedForm.ghuu = zeros(M_.endo_nbr, M_.exo_nbr^2);
        ReducedForm.ghxu = zeros(M_.endo_nbr, M_.nspred*M_.exo_nbr);
      end
    end
    ParticleOptions = struct();
    ParticleOptions.number_of_particles = 100000;
    ParticleOptions.pruning = $order == 2;
    ParticleOptions.resampling.status.generic = false;
    ParticleOptions.resampling.status.systematic = true;
    ParticleOptions.resampling.status.none = false;
    ParticleOptions.resampling.method.kitagawa = true;
    ParticleOptions.resampling.method.stratified = false;
    ParticleOptions.resampling.method.smooth = false;
    ThreadsOptions = struct();
    [LIK, lik] = sequential_importance_particle_filter(ReducedForm, y, 1, ParticleOptions, ThreadsOptions, options_, M_);
    ex.loglik = -LIK; ex.loglik_t = -lik;
    save('-mat7-binary', '../out_particle/order$order.mat', 'ex');
  " > "../out_particle/order$order.log" 2>&1 ) || {
    echo "FAILED order $order, see out_particle/order$order.log"
    tail -30 "out_particle/order$order.log"
    exit 1
  }
done
rm -rf work_particle
echo "done."
