#!/usr/bin/env bash
# run_ep.sh — trajetória estendida (determinística e estocástica) no Dynare, para compare_ep.py.
#
# Uso:  bash run_ep.sh
#
# Os arquivos vêm de tests/ep do Dynare 7.1 (fetch.sh) e rodam sem alteração até a última
# simulação, exceto a tolerância do solver: o Dynare para em resíduo 1e-5 por omissão
# (options_.dynatol), e a cópia pede 1e-12, porque a comparação é de algoritmos. No burnside,
# a cópia para no `toc` (o que vem depois compara com um arquivo .mat que o teste não traz); no linearmodel0, a
# opção use_dll sai, porque pede um compilador com as opções do GCC.
# Exporta para out_ep/<nome>.mat as trajetórias de cada variante e os choques usados.
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

TOL='options_.dynatol.f = 1e-12; options_.dynatol.x = 1e-12;'
mkdir -p out_ep work_ep
awk -v tol="$TOL" '/^T = 100;/{print tol} {print} /^toc/{exit}' mods/ep/burnside.mod > work_ep/burnside.mod
# use_dll pede um compilador com as opções do GCC; sem ela o modelo é o mesmo
awk -v tol="$TOL" '/^options_.ep.order = 0;/{print tol} {sub(/model\(use_dll\)/, "model")} {print}' \
  mods/ep/linearmodel0.mod > work_ep/linearmodel0.mod

save_burnside="ex = struct(); ex.e = e_1; ex.names = M_.endo_names;
  ex.ep0 = ts0.data; ex.sep1 = ts1.data; ex.sep1h = ts1h.data; ex.sep2 = ts2.data;
  ex.sep2full = ts2__.data; ex.sep2h = ts2h.data; ex.sep2hfull = ts2h__.data;"
save_linear="ex = struct(); ex.e = oo.exo_simul; ex.names = M_.endo_names; ex.ep0 = ts.data;
  ex.linear = oo.endo_simul';"
for m in ${ONLY:-burnside linearmodel0}; do
  echo ">>> $m.mod"
  rm -f "out_ep/$m.mat"
  if [ "$m" = burnside ]; then save="$save_burnside"; else save="$save_linear"; fi
  ( cd work_ep && "$OCTAVE" --no-gui --eval "
    pkg load statistics datatypes;
    addpath '$DYNARE_MATLAB';
    dynare $m noclearall nograph;
    $save
    save('-mat7-binary', '../out_ep/${m}.mat', 'ex');
    printf('gravado out_ep/${m}.mat\n');
  " > "../out_ep/${m}.log" 2>&1 ) || { echo "FALHOU $m — ver out_ep/${m}.log"; tail -15 "out_ep/${m}.log"; continue; }
  grep "gravado" "out_ep/${m}.log" || tail -5 "out_ep/${m}.log"
done
echo "pronto."
