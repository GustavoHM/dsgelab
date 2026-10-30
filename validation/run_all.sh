#!/usr/bin/env bash
# run_all.sh: the whole external validation, end to end. It downloads the .mod files, runs Dynare
# and MacroModelling.jl, and checks each result against dsgelab.
#
# Usage:  bash run_all.sh            (everything)
#         SKIP_JULIA=1 bash run_all.sh
#         SKIP_MH=1 bash run_all.sh   (skips the estimation's Metropolis, which takes a few minutes)
#
# Requires GNU Octave with the statistics and datatypes packages, Dynare (OCTAVE and DYNARE_MATLAB
# override the search in the usual places) and, for the calibration equations, Julia. The project
# in macromodelling/ pins the versions of MacroModelling.jl. The Python that runs the compare_*.py
# scripts is the one on the PATH, or PYTHON, and needs dsgelab installed.
#
# A step that fails does not stop the following ones. The summary at the end says which failed,
# and the exit status is nonzero if any did. Heutel's (2012) MATLAB files come from the public copy
# that heutel/fetch.sh downloads and verifies (see ../docs/heutel.md).
set -uo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python}"
results=()
failed=0

step() {   # step <label> <command...>
  local label="$1"; shift
  echo
  echo "=== $label"
  local t0=$SECONDS
  if "$@" < /dev/null; then
    results+=("ok       $label  ($((SECONDS - t0)) s)")
  else
    results+=("FAILED   $label  ($((SECONDS - t0)) s)")
    failed=1
  fi
}

"$PYTHON" -c "import dsgelab, sys; print('dsgelab', dsgelab.__version__, 'at', dsgelab.__file__)" \
  || { echo "FATAL: dsgelab does not import with $PYTHON"; exit 1; }

step "fetch.sh"              bash fetch.sh
step "Dynare: examples"      bash run_dynare.sh
step "compare.py"            "$PYTHON" compare.py
step "roundtrip.py"          "$PYTHON" roundtrip.py
step "Dynare: second-order decision rules" bash run_decision_rule.sh
step "compare_decision_rule.py" "$PYTHON" compare_decision_rule.py
step "Dynare: third-order IRFs" bash run_third_irf.sh
step "compare_third_irf.py"  "$PYTHON" compare_third_irf.py
step "Dynare: parameter Jacobians" bash run_derivatives.sh
step "compare_derivatives.py" "$PYTHON" compare_derivatives.py
step "Dynare: second-order parameter Jacobians" bash run_second_derivatives.sh
step "compare_second_derivatives.py" "$PYTHON" compare_second_derivatives.py
step "Dynare: third-order parameter Jacobians" bash run_third_derivatives.sh
step "compare_third_derivatives.py" "$PYTHON" compare_third_derivatives.py
step "sparse third-order Sylvester" "$PYTHON" compare_third_sparse.py
step "sparse third-order full pipeline" "$PYTHON" compare_third_sparse.py --full
step "Dynare: particle filter" bash run_particle.sh
step "compare_particle.py"  "$PYTHON" compare_particle.py
step "Dynare: inversion filter" bash run_inversion.sh
step "compare_inversion.py" "$PYTHON" compare_inversion.py
step "Dynare: Ramsey"        bash run_ramsey.sh
step "compare_ramsey.py"     "$PYTHON" compare_ramsey.py
step "Dynare: .mod files read" bash run_examples.sh
step "compare_import.py"     "$PYTHON" compare_import.py
step "MMB: fetch"            bash mmb/fetch.sh
step "Dynare: MMB models"    bash mmb/run_dynare.sh
step "mmb/compare.py"        "$PYTHON" mmb/compare.py
step "DSGE_mod: fetch"       bash dsge_mod/fetch.sh
step "Dynare: DSGE_mod"      bash dsge_mod/run_dynare.sh
step "dsge_mod/compare.py"   "$PYTHON" dsge_mod/compare.py
step "Dynare: policy"        bash run_policy.sh
step "compare_policy.py"     "$PYTHON" compare_policy.py
step "Dynare: perfect foresight" bash run_foresight.sh
step "compare_foresight.py"  "$PYTHON" compare_foresight.py
step "Dynare: estimation"    bash run_estimation.sh
if [ -n "${SKIP_MH:-}" ]; then
  step "compare_estimation.py --no-mh" "$PYTHON" compare_estimation.py --no-mh
else
  step "compare_estimation.py" "$PYTHON" compare_estimation.py
fi
step "Dynare: OccBin"        bash run_occbin.sh
step "compare_occbin.py"     "$PYTHON" compare_occbin.py
step "Dynare: extended path" bash run_ep.sh
step "compare_ep.py"         "$PYTHON" compare_ep.py
step "Dynare: balanced growth" bash run_bgp.sh
step "compare_bgp.py"        "$PYTHON" compare_bgp.py
step "Dynare: identification" bash run_identification.sh
step "compare_identification.py" "$PYTHON" compare_identification.py
step "Dynare: method of moments" bash run_mom.sh
step "compare_mom.py"       "$PYTHON" compare_mom.py
step "Dynare: pruned second-order moments" bash run_mom_order2.sh
step "compare_mom_order2.py" "$PYTHON" compare_mom_order2.py
step "Dynare: IRF matching" bash run_irf.sh
step "compare_irf.py"       "$PYTHON" compare_irf.py
step "Dynare: filtered moments" bash run_filtered.sh
step "compare_filtered.py"  "$PYTHON" compare_filtered.py
step "Dynare: calibration restrictions" bash run_restrictions.sh
step "compare_restrictions.py" "$PYTHON" compare_restrictions.py
step "Dynare and R: forecast evaluation" bash run_forecast_eval.sh
step "compare_forecast_eval.py" "$PYTHON" compare_forecast_eval.py
step "Dynare: stability mapping" bash run_gsa.sh
step "compare_gsa.py"       "$PYTHON" compare_gsa.py
# after run_examples.sh: the steady_state_model part reads out_import/fs2000.mat
step "compare_diagnostics.py" "$PYTHON" compare_diagnostics.py

if [ -n "${SKIP_JULIA:-}" ]; then
  results+=("skipped  MacroModelling.jl (SKIP_JULIA)")
else
  # calibration.json is under version control. Without deleting it, a Julia failure would let the
  # comparison read the result of an earlier run and pass.
  rm -f macromodelling/calibration.json
  step "MacroModelling.jl: calibration" \
    julia --project=macromodelling -e 'using Pkg; Pkg.instantiate(); include("macromodelling/calibration.jl")'
  step "compare_calibration.py" "$PYTHON" compare_calibration.py
  rm -f macromodelling/inversion.json
  step "MacroModelling.jl: inversion filter" \
    julia --project=macromodelling -e 'using Pkg; Pkg.instantiate(); include("macromodelling/inversion.jl")'
  step "compare_inversion_julia.py" "$PYTHON" compare_inversion_julia.py
fi

step "Heutel (2012): original files" bash heutel/fetch.sh
step "Heutel (2012)" "$PYTHON" heutel/validate.py

echo
echo "=== summary"
printf '%s\n' "${results[@]}"
exit $failed
