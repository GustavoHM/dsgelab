#!/usr/bin/env bash
# run_ramsey.sh — roda os modelos de política ótima no Octave + Dynare e exporta oo_ para .mat.
#
# Uso:  bash run_ramsey.sh
#
# Três modelos: ramsey/nk_linear.mod e ramsey/rbc_labortax.mod, escritos aqui, e mods/ramsey/nk_ramsey_model.mod, do conjunto
# de testes do próprio Dynare (baixado por fetch.sh). O .mod do Dynare roda SEM alteração; depois
# dele, o stoch_simul é reexecutado com 40 períodos de IRF, porque o original pede irf=0. O
# original não fixa a ordem e o Dynare usa 2 por omissão; a comparação é de 1ª ordem. O
# ramsey_model já deixou M_ e options_ com o problema do planejador, e o stoch_simul o herda.
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
[ -n "${DYNARE_MATLAB:-}" ] || { echo "FATAL: diretório matlab/ do Dynare não encontrado. Defina DYNARE_MATLAB."; exit 1; }

mkdir -p out
for src in ramsey/nk_linear.mod ramsey/rbc_labortax.mod mods/ramsey/nk_ramsey_model.mod; do
  [ -f "$src" ] || { echo "falta $src — rode fetch.sh"; exit 1; }
  m="$(basename "$src")"; base="${m%.mod}"
  echo ">>> $m"
  cp "$src" .
  "$OCTAVE" --no-gui --eval "
    pkg load statistics datatypes;
    addpath '$DYNARE_MATLAB';
    dynare $m noclearall nograph;
    options_.order = 1; options_.irf = 40; options_.nograph = 1; options_.noprint = 1; options_.periods = 0;
    [info, oo_, options_, M_] = stoch_simul(M_, options_, oo_, M_.endo_names);
    ex = struct();
    ex.steady_state = oo_.steady_state;
    ex.ys = oo_.dr.ys;
    ex.endo_names   = M_.endo_names;
    ex.exo_names    = M_.exo_names;
    ex.orig_endo_nbr = M_.orig_endo_nbr;
    ex.irfs         = oo_.irfs;
    ex.sigma_e      = M_.Sigma_e;
    ex.info         = info;
    save('-mat7-binary', 'out/${base}.mat', 'ex');
    printf('gravado out/${base}.mat (info=%d)\n', info);
  " > "out/${base}.log" 2>&1 || { echo "FALHOU $m — ver out/${base}.log"; tail -30 "out/${base}.log"; exit 1; }
  tail -1 "out/${base}.log"
  rm -rf "$m" "$base" "+$base" "${base}.log"
done
echo "pronto."
