#!/usr/bin/env bash
# fetch.sh: downloads the reference .mod files to validation/mods/.
#
# The files are NOT under version control here. They are GPL, and this package is MIT. Vendoring
# them would mix licences with no need, since they are public and stable.
set -euo pipefail
mkdir -p "$(dirname "$0")/mods/ramsey" "$(dirname "$0")/mods/examples" "$(dirname "$0")/mods/foresight" \
         "$(dirname "$0")/mods/forecast"
cd "$(dirname "$0")/mods"

get() { curl -fsSL --retry 5 --retry-all-errors --retry-delay 3 -o "$2" "$1" && echo "ok  $2" || echo "FAILED  $2  ($1)"; }

# Examples of the Dynare tutorial (Collard 2001), cited at dynare.org/resources
get https://www.dynare.org/assets/tutorial/example1.mod example1.mod
get https://www.dynare.org/assets/tutorial/example2.mod example2.mod
# Johannes Pfeifer's collection, also cited at dynare.org/resources
get https://raw.githubusercontent.com/JohannesPfeifer/DSGE_mod/master/RBC_baseline/RBC_baseline.mod RBC_baseline.mod

# Dynare's official repository, at fixed tags so that the reference does not change unnoticed.
# The examples left the tree in the 7 series, so they come from 6.4. The tests come from 7.2, the
# installed version, and the 21 files used are identical to those of 7.1 (checked on 12 October
# 2026).
DY=https://git.dynare.org/Dynare/dynare/-/raw
for f in fs2000 NK_baseline Gali_2015 example3 agtrend; do
  get "$DY/6.4/examples/$f.mod" "examples/$f.mod"
done
# MATLAB helpers that two of them call for the steady state
get "$DY/6.4/examples/NK_baseline_steadystate.m" examples/NK_baseline_steadystate.m
get "$DY/6.4/examples/example3_steady_state_helper.m" examples/example3_steady_state_helper.m
get "$DY/7.2/tests/optimal_policy/nk_ramsey_model.mod"  ramsey/nk_ramsey_model.mod
get "$DY/7.2/tests/optimal_policy/Ramsey/ramsey_ex.mod" ramsey/ramsey_ex.mod
get "$DY/7.2/tests/optimal_policy/OSR/osr_example.mod" ramsey/osr_example.mod
get "$DY/7.2/tests/discretionary_policy/dennis_1.mod"  ramsey/dennis_1.mod
# Perfect foresight, with and without expectation errors (tests/deterministic_simulations, 7.2)
for f in pfwee.mod pfwee.csv pfwee_constant_sim_length.mod rbc_det1.mod rbc_det2.mod rbc_det3.mod; do
  get "$DY/7.2/tests/deterministic_simulations/$f" "foresight/$f"
done
# Estimation: the fs2000 data (MATLAB script that assembles the series) and the conditional
# forecast test
get "$DY/6.4/examples/fs2000_data.m" examples/fs2000_data.m
get "$DY/7.2/tests/conditional_forecasts/1/fs2000_cal.mod" forecast/fs2000_cal.mod
# OccBin: one constraint (borrcon) and two (dynrbc), from tests/occbin of Dynare 7.2
mkdir -p occbin
for f in model_borrcon/borrcon.mod model_borrcon/borrcon_common.inc \
         model_irrcap_twoconstraints/dynrbc.mod model_irrcap_twoconstraints/dynrbc_common.inc; do
  get "$DY/7.2/tests/occbin/$f" "occbin/$(basename "$f")"
done
# Extended path (tests/ep of Dynare 7.2): Burnside (1998), with a closed-form solution, and a
# linear model
mkdir -p ep
for f in burnside.mod linearmodel0.mod; do
  get "$DY/7.2/tests/ep/$f" "ep/$f"
done
# Nonstationary models: fs2000 written with trends (trend_var, var(deflator=)), which Dynare
# detrends, and the bgp.write tests (balanced growth)
get "$DY/7.2/tests/trend_var/fs2000_nonstationary.mod" examples/fs2000_nonstationary.mod
# Local identification (tests/identification of Dynare 7.2): Kim (2003), with two parameters that
# only enter together and one that does not enter at all, and An and Schorfheide (2007)
mkdir -p identification
get "$DY/7.2/tests/identification/kim/kim2.mod" identification/kim2.mod
get "$DY/7.2/tests/identification/as2007/as2007.mod" identification/as2007.mod
mkdir -p bgp
for f in solow-1/solow.mod ramsey-1/ramsey.mod nk-1/nk.mod; do
  get "$DY/7.2/tests/bgp/$f" "bgp/$(dirname "$f").mod"
done
