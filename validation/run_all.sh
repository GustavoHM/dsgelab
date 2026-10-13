#!/usr/bin/env bash
# run_all.sh — a validação externa inteira, de ponta a ponta: baixa os .mod, roda o Dynare e o
# MacroModelling.jl e confronta cada resultado com o dsgelab.
#
# Uso:  bash run_all.sh            (tudo)
#       SKIP_JULIA=1 bash run_all.sh
#       SKIP_MH=1 bash run_all.sh   (pula o Metropolis da estimação, que leva alguns minutos)
#
# Precisa de GNU Octave com os pacotes statistics e datatypes, do Dynare (OCTAVE e DYNARE_MATLAB
# sobrepõem a busca nos lugares usuais) e, para as equações de calibração, de Julia: o projeto em
# macromodelling/ fixa as versões do MacroModelling.jl. O Python que roda os compare_*.py é o do
# PATH, ou PYTHON, e precisa do dsgelab instalado.
#
# Uma etapa que falha não interrompe as seguintes; o resumo no fim diz quais falharam, e a saída é
# diferente de zero se alguma falhou. Os arquivos MATLAB de Heutel (2012) vêm da cópia pública que
# heutel/fetch.sh baixa e confere (ver ../docs/heutel.md).
set -uo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python}"
results=()
failed=0

step() {   # step <rótulo> <comando...>
  local label="$1"; shift
  echo
  echo "=== $label"
  local t0=$SECONDS
  if "$@" < /dev/null; then
    results+=("ok       $label  ($((SECONDS - t0)) s)")
  else
    results+=("FALHOU   $label  ($((SECONDS - t0)) s)")
    failed=1
  fi
}

"$PYTHON" -c "import dsgelab, sys; print('dsgelab', dsgelab.__version__, 'em', dsgelab.__file__)" \
  || { echo "FATAL: o dsgelab não importa com $PYTHON"; exit 1; }

step "fetch.sh"              bash fetch.sh
step "Dynare: exemplos"      bash run_dynare.sh
step "compare.py"            "$PYTHON" compare.py
step "roundtrip.py"          "$PYTHON" roundtrip.py
step "Dynare: Ramsey"        bash run_ramsey.sh
step "compare_ramsey.py"     "$PYTHON" compare_ramsey.py
step "Dynare: .mod lidos"    bash run_examples.sh
step "compare_import.py"     "$PYTHON" compare_import.py
step "Dynare: política"      bash run_policy.sh
step "compare_policy.py"     "$PYTHON" compare_policy.py
step "Dynare: previsão perfeita" bash run_foresight.sh
step "compare_foresight.py"  "$PYTHON" compare_foresight.py
step "Dynare: estimação"     bash run_estimation.sh
if [ -n "${SKIP_MH:-}" ]; then
  step "compare_estimation.py --sem-mh" "$PYTHON" compare_estimation.py --sem-mh
else
  step "compare_estimation.py" "$PYTHON" compare_estimation.py
fi
step "Dynare: OccBin"        bash run_occbin.sh
step "compare_occbin.py"     "$PYTHON" compare_occbin.py
step "Dynare: trajetória estendida" bash run_ep.sh
step "compare_ep.py"         "$PYTHON" compare_ep.py
step "Dynare: crescimento balanceado" bash run_bgp.sh
step "compare_bgp.py"        "$PYTHON" compare_bgp.py
# depois de run_examples.sh: a parte do steady_state_model lê out_import/fs2000.mat
step "compare_diagnostics.py" "$PYTHON" compare_diagnostics.py

if [ -n "${SKIP_JULIA:-}" ]; then
  results+=("pulado   MacroModelling.jl (SKIP_JULIA)")
else
  # calibration.json é versionado: sem apagá-lo, uma falha do Julia deixaria a comparação ler o
  # resultado de uma rodada anterior e passar.
  rm -f macromodelling/calibration.json
  step "MacroModelling.jl: calibração" \
    julia --project=macromodelling -e 'using Pkg; Pkg.instantiate(); include("macromodelling/calibration.jl")'
  step "compare_calibration.py" "$PYTHON" compare_calibration.py
fi

step "Heutel (2012): arquivos originais" bash heutel/fetch.sh
step "Heutel (2012)" "$PYTHON" heutel/validate.py

echo
echo "=== resumo"
printf '%s\n' "${results[@]}"
exit $failed
