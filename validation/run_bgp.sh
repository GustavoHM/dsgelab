#!/usr/bin/env bash
# run_bgp.sh — o problema de crescimento balanceado que o Dynare monta (bgp.write), para
# compare_bgp.py.
#
# Uso:  bash run_bgp.sh
#
# Os arquivos vêm de tests/bgp do Dynare 7.2 (fetch.sh). A cópia de trabalho para antes do bloco
# verbatim do teste (que resolve com o fsolve do MATLAB, ausente ou diferente no Octave); depois
# do pré-processamento, chama bgp.write(M_) e avalia a função gerada, bgpfun, em cinco pontos
# fixos z = [níveis; fatores de crescimento]. Exporta para out_bgp/<nome>.mat os pontos e os
# resíduos.
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
    printf('gravado out_bgp/${name}.mat\n');
  " > "../out_bgp/${name}.log" 2>&1 ) || { echo "FALHOU $m — ver out_bgp/${name}.log"; tail -15 "out_bgp/${name}.log"; continue; }
  grep gravado "out_bgp/${name}.log" || tail -5 "out_bgp/${name}.log"
done
rm -rf work_bgp
echo "pronto."
