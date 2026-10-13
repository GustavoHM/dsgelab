#!/usr/bin/env bash
# run_occbin.sh — simulações OccBin no Dynare, para compare_occbin.py.
#
# Uso:  bash run_occbin.sh
#
# Os arquivos vêm de tests/occbin do Dynare 7.2 (fetch.sh). O modelo, o estado estacionário,
# os choques-surpresa e as opções do occbin_solver são os do teste; a cópia de trabalho
# (work_occbin/, também o que compare_occbin.py lê) junta o .inc ao .mod e corta o que vem
# depois da simulação (gráficos, suavizador). No dynrbc, só o cenário de impulso (option=1).
# Exporta para out_occbin/<nome>.mat as trajetórias por partes e linear, em níveis, e o
# histórico de regimes.
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

mkdir -p out_occbin work_occbin
"${PYTHON:-python3}" - <<'PY'
import re
src = "mods/occbin"
# borrcon: o .mod até o include, o .inc inteiro (termina no occbin_solver)
mod = open(f"{src}/borrcon.mod").read().split('@#include "borrcon_common.inc"')[0]
open("work_occbin/borrcon.mod", "w").write(mod + open(f"{src}/borrcon_common.inc").read())
# dynrbc: o .inc até "// run occbin simulations", depois o bloco de option=1 e o solver
mod = open(f"{src}/dynrbc.mod").read().split('@#include "dynrbc_common.inc"')[0]
inc = open(f"{src}/dynrbc_common.inc").read()
head = inc.split("// run occbin simulations")[0]
shocks = re.search(r"if option==1\s*(shocks\(surprise,overwrite\);.*?end;)", inc, re.S).group(1)
solver = re.search(r"occbin_solver\(simul_debug[^;]*;", inc).group(0).replace("simul_debug,", "")
# occbin_setup prepara o simulador; a opção do teste (smoother_debug) só vale para o suavizador
open("work_occbin/dynrbc.mod", "w").write(mod + head + shocks + "\noccbin_setup;\n" + solver + "\n")
PY
for m in ${ONLY:-borrcon dynrbc}; do
  echo ">>> $m.mod"
  rm -f "out_occbin/$m.mat"
  ( cd work_occbin && "$OCTAVE" --no-gui --eval "
    pkg load statistics datatypes;
    addpath '$DYNARE_MATLAB';
    dynare $m noclearall nograph;
    oo_ = occbin.unpack_simulations(M_, oo_, options_);
    ex = struct();
    n = M_.endo_nbr;
    T = numel(oo_.occbin.endo_piecewise.(strtrim(M_.endo_names{1})));
    ex.piecewise = zeros(T, n); ex.linear = zeros(T, n); ex.ys = oo_.steady_state;
    for j = 1:n
      v = strtrim(M_.endo_names{j});
      ex.piecewise(:, j) = oo_.occbin.endo_piecewise.(v);
      ex.linear(:, j) = oo_.occbin.endo_linear.(v);
    end
    ex.endo_names = M_.endo_names;
    rh = oo_.occbin.simul.regime_history;
    f = fieldnames(rh);
    for j = 1:numel(f)
      ex.(['rh_' f{j}]) = {rh.(f{j})};
    end
    save('-mat7-binary', '../out_occbin/${m}.mat', 'ex');
    printf('gravado out_occbin/${m}.mat\n');
  " > "../out_occbin/${m}.log" 2>&1 ) || { echo "FALHOU $m — ver out_occbin/${m}.log"; tail -15 "out_occbin/${m}.log"; continue; }
  tail -1 "out_occbin/${m}.log"
done
echo "pronto."
