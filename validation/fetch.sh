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
# Method of moments (tests/estimation/method_of_moments of Dynare 7.2). The
# working cases are generated from this model; the second dataset is simulated.
mkdir -p method_of_moments
for f in AnScho_MoM_common.inc AnScho_MoM_data_2.mat; do
  get "$DY/7.2/tests/estimation/method_of_moments/AnScho/$f" "method_of_moments/$f"
done
get "$DY/7.2/examples/estimation/rbc_irf_matching.mod" method_of_moments/rbc_irf_matching.mod
# Calibration restrictions (tests/gsa of Dynare 7.2): Lubik and Schorfheide (2003), with
# irf_calibration and moment_calibration blocks evaluated over the prior by the sensitivity toolbox
mkdir -p gsa
for f in ls2003a.mod ls2003_model.inc; do
  get "$DY/7.2/tests/gsa/$f" "gsa/$f"
done
mkdir -p bgp
for f in solow-1/solow.mod ramsey-1/ramsey.mod nk-1/nk.mod; do
  get "$DY/7.2/tests/bgp/$f" "bgp/$(dirname "$f").mod"
done
# BVAR and DSGE-VAR (tests/bvar_a_la_sims and tests/dsge-var of Dynare 7.2), checked by SHA-256.
# simul_hybrid.mod simulates the data of the two DSGE-VAR tests.
mkdir -p bvar dsge_var
getsum() {   # getsum <url> <destination> <sha256>
  if get "$1" "$2" > /dev/null && echo "$3  $2" | shasum -a 256 -c --status -; then
    echo "ok  $2"
  else
    echo "FAILED  $2  (download or SHA-256)"; rm -f "$2"
  fi
}
getsum "$DY/7.2/tests/bvar_a_la_sims/bvar_standalone.mod" bvar/bvar_standalone.mod \
  db4f4a7ba86ecf071af26ff6f7be432ec69af2b7e429848c3f2cdabe336d5296
getsum "$DY/7.2/tests/bvar_a_la_sims/bvar_and_dsge.mod" bvar/bvar_and_dsge.mod \
  f856ca48a3691b1479ad9412e79c45c26b8a92c02de1f8233bfc44a71b9dddac
getsum "$DY/7.2/tests/bvar_a_la_sims/bvar_sample.m" bvar/bvar_sample.m \
  f20ff9a540d484afee194f0dce6ce88394c8d682abb0f3022516beab87e23f2d
getsum "$DY/7.2/tests/dsge-var/dsgevar_forward_calibrated_lambda.mod" \
  dsge_var/dsgevar_forward_calibrated_lambda.mod \
  acbe39f1164dcc3e2fceeea2f918bc03a07030e8c4784c910a2c1805103b2c8d
getsum "$DY/7.2/tests/dsge-var/dsgevar_forward_estimated_lambda.mod" \
  dsge_var/dsgevar_forward_estimated_lambda.mod \
  445088f830979369a1d2257325b01e0f90a4beceb0e258fc581eb06c469acfac
getsum "$DY/7.2/tests/dsge-var/simul_hybrid.mod" dsge_var/simul_hybrid.mod \
  634dba42b156e9f8403a3179c0a7411e937ad1ac41b638bb6d3a22edfc223f64
