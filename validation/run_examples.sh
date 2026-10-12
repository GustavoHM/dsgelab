#!/usr/bin/env bash
# run_examples.sh — roda no Dynare cada .mod de mods/ e mods/examples/ e exporta o que
# compare_import.py precisa: estado estacionário (oo_.dr.ys), parâmetros, covariância dos choques,
# IRFs e momentos teóricos.
#
# Uso:  bash run_examples.sh [nome ...]      (sem argumentos: todos)
#
# Cada .mod é reduzido por strip_mod.py ao que declara o modelo, e recebe o mesmo
# stoch_simul(order=1, irf=40, periods=0). Os arquivos auxiliares em MATLAB que alguns exemplos
# chamam (o estado estacionário do NK_baseline, o auxiliar do example3) vão junto.
set -euo pipefail
cd "$(dirname "$0")"

OCTAVE="${OCTAVE:-/opt/homebrew/bin/octave}"
PY="${PYTHON:-python3}"
if [ -z "${DYNARE_MATLAB:-}" ]; then
  for c in /opt/homebrew/opt/dynare/lib/dynare/matlab /opt/homebrew/lib/dynare/matlab \
           /usr/local/opt/dynare/lib/dynare/matlab /usr/lib/dynare/matlab \
           /Applications/Dynare/*/matlab; do
    [ -d "$c" ] && DYNARE_MATLAB="$c" && break
  done
fi
[ -n "${DYNARE_MATLAB:-}" ] || { echo "FATAL: diretório matlab/ do Dynare não encontrado. Defina DYNARE_MATLAB."; exit 1; }

FILES=()
if [ $# -eq 0 ]; then
  for f in mods/*.mod mods/examples/*.mod; do FILES+=("$f"); done
else
  for n in "$@"; do FILES+=($(ls mods/"$n".mod mods/examples/"$n".mod 2>/dev/null || true)); done
fi

mkdir -p out_import work
for src in "${FILES[@]}"; do
  m="$(basename "$src")"; base="${m%.mod}"
  echo ">>> $m"
  rm -rf work/*
  "$PY" strip_mod.py "$src" "work/$m"
  cp "$(dirname "$src")"/*.m work/ 2>/dev/null || true
  ( cd work && "$OCTAVE" --no-gui --eval "
    pkg load statistics datatypes;
    addpath '$DYNARE_MATLAB';
    dynare $m noclearall nograph;
    ex = struct();
    ex.ys          = oo_.dr.ys;
    ex.endo_names  = M_.endo_names(1:M_.orig_endo_nbr);
    ex.exo_names   = M_.exo_names;
    ex.params      = M_.params;
    ex.param_names = M_.param_names;
    ex.sigma_e     = M_.Sigma_e;
    ex.irfs        = oo_.irfs;
    if isfield(oo_, 'var'), ex.variance = oo_.var; end
    save('-mat7-binary', '../out_import/${base}.mat', 'ex');
    printf('gravado out_import/${base}.mat\n');
  " > "../out_import/${base}.log" 2>&1 ) || { echo "FALHOU $m — ver out_import/${base}.log"; tail -20 "out_import/${base}.log"; continue; }
  tail -1 "out_import/${base}.log"
done
rm -rf work
echo "pronto."
