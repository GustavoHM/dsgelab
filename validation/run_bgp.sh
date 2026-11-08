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
#
# The second part is for the trends declared in from_equations. Solow and Ramsey are rewritten
# with their trends declared (trend_var, var(deflator=)), and the trend equations they replace are
# removed. The rewritten models are saved as out_bgp/<name>_trend.mod, which compare_bgp.py also
# reads with from_mod. fs2000_nonstationary.mod (tests/trend_var of Dynare 7.2) is read unchanged
# up to its steady command. Dynare detrends each one and solves it with solve_tolf = 1e-14: the
# steady state, and, for the two with shocks, stoch_simul(order=1, irf=40, periods=0) and 200
# periods of a fixed shock path simulated by simult_. It exports out_bgp/<name>_trend.mat.
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

# --- the same models with their trends declared
strip() { awk '/^if ~isoctave/ || /^verbatim;/{exit} {print}' "$1"; }
strip mods/bgp/solow-1.mod | perl -0pe '
  s/var Efficiency\s+EfficiencyGrowth\s+Population\s+PopulationGrowth\s+Output\s+PhysicalCapitalStock\s*;/var EfficiencyGrowth PopulationGrowth;\ntrend_var(growth_factor=EfficiencyGrowth) Efficiency;\ntrend_var(growth_factor=PopulationGrowth) Population;\nvar(deflator=Efficiency*Population) Output PhysicalCapitalStock;/;
  s/^\s*Efficiency = EfficiencyGrowth\*Efficiency\(-1\);\n//m;
  s/^\s*Population = PopulationGrowth\*Population\(-1\);\n//m' > out_bgp/solow_trend.mod
cat >> out_bgp/solow_trend.mod <<'EOF'

initval;
EfficiencyGrowth = 1.01; PopulationGrowth = 1.01; Output = 2.2; PhysicalCapitalStock = 11.2;
end;

shocks;
var e_x; stderr 0.01;
var e_n; stderr 0.005;
end;
EOF
strip mods/bgp/ramsey-1.mod | perl -0pe '
  s/^var c k x;/trend_var(growth_factor=1.02) x;\nvar(deflator=x) c k;/m;
  s/^\s*x = x\(-1\)\*1\.02;\n//m' > out_bgp/ramsey_trend.mod
cat >> out_bgp/ramsey_trend.mod <<'EOF'

initval;
c = 1; k = 10;
end;
EOF
awk '/^steady;/{exit} {print}' mods/examples/fs2000_nonstationary.mod > out_bgp/fs2000_trend.mod
for name in solow ramsey; do
  grep -q "trend_var" "out_bgp/${name}_trend.mod" && ! grep -Eq "^[[:space:]]*(Efficiency|Population|x) = " \
    "out_bgp/${name}_trend.mod" || { echo "FAILED to declare the trends of $name"; exit 1; }
done
for name in solow ramsey fs2000; do
  echo ">>> ${name}_trend.mod"
  rm -rf work_bgp/*
  cp "out_bgp/${name}_trend.mod" "work_bgp/${name}_trend.mod"
  cat >> "work_bgp/${name}_trend.mod" <<'EOF'

options_.solve_tolf = 1e-14;
steady;
@#if stochastic
stoch_simul(order=1, irf=40, periods=0, nograph);
@#endif
EOF
  stochastic=true; [ "$name" = ramsey ] && stochastic=false
  ( cd work_bgp && "$OCTAVE" --no-gui --eval "
    pkg load statistics datatypes;
    addpath '$DYNARE_MATLAB';
    dynare ${name}_trend noclearall nograph -Dstochastic=$stochastic;
    ex = struct();
    ex.ys = oo_.steady_state; ex.endo_names = M_.endo_names(1:M_.orig_endo_nbr);
    ex.exo_names = M_.exo_names; ex.params = M_.params; ex.param_names = M_.param_names;
    if $stochastic
      ex.sigma_e = M_.Sigma_e; ex.irfs = oo_.irfs; ex.variance = oo_.var;
      T = 200; q = M_.exo_nbr;
      shocks = 0.01*sin(0.7*(1:T)' * (1:q) + (1:q));
      y = simult_(M_, options_, oo_.dr.ys, oo_.dr, shocks, 1);
      ex.shocks = shocks; ex.path = y(1:M_.orig_endo_nbr, :);
    end
    save('-mat7-binary', '../out_bgp/${name}_trend.mat', 'ex');
    printf('saved out_bgp/${name}_trend.mat\n');
  " > "../out_bgp/${name}_trend.log" 2>&1 ) \
    || { echo "FAILED ${name}_trend, see out_bgp/${name}_trend.log"; tail -15 "out_bgp/${name}_trend.log"; continue; }
  grep saved "out_bgp/${name}_trend.log" || tail -5 "out_bgp/${name}_trend.log"
done
rm -rf work_bgp
echo "done."
