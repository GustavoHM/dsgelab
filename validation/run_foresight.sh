#!/usr/bin/env bash
# run_foresight.sh — previsão perfeita no Dynare, com e sem erros de expectativa.
#
# Uso:  bash run_foresight.sh
#
# Os arquivos vêm de tests/deterministic_simulations do Dynare 7.2 (fetch.sh) e rodam sem
# alteração, exceto os comandos de gráfico (rplot), que a cópia remove, e a tolerância do solver. Exporta para
# out_foresight/<nome>.mat a trajetória oo_.endo_simul, a exógena oo_.exo_simul e os nomes.
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

mkdir -p out_foresight work
for m in pfwee pfwee_constant_sim_length rbc_det1 rbc_det2 rbc_det3; do
  echo ">>> $m.mod"
  rm -rf work/*
  # O solver do Dynare para em resíduo 1e-5 por omissão (options_.dynatol); a comparação é de
  # modelos, e a cópia pede 1e-12 antes de cada setup e solver. O estado estacionário terminal, que o setup
  # com erros de expectativa recalcula a cada notícia, para em options_.simul.steady_tolf
  # (eps^(1/3), copiado de solve_tolf na inicialização); idem.
  grep -v "^\s*rplot" "mods/foresight/$m.mod" \
    | sed -E 's/^(\s*)(perfect_foresight_setup|perfect_foresight_with_expectation_errors_setup|perfect_foresight_solver|perfect_foresight_with_expectation_errors_solver)/\1options_.dynatol.f = 1e-12; options_.dynatol.x = 1e-12; options_.solve_tolf = 1e-14; options_.solve_tolx = 1e-14; options_.simul.steady_tolf = 1e-14; options_.simul.steady_tolx = 1e-14;\n\1\2/' \
    | awk '{print} /^pfwee_simul = oo_.endo_simul;/{exit}' > "work/$m.mod"
  # (o pfwee.mod termina com uma réplica manual e um teste de igualdade EXATA entre as duas, que
  # deixa de valer com outra tolerância; a cópia para em pfwee_simul, que é o que se exporta)
  rm -f "out_foresight/$m.mat"
  cp mods/foresight/pfwee.csv work/
  ( cd work && "$OCTAVE" --no-gui --eval "
    pkg load statistics datatypes;
    addpath '$DYNARE_MATLAB';
    dynare $m noclearall nograph;
    ex = struct();
    if exist('pfwee_simul', 'var'), ex.endo_simul = pfwee_simul; else ex.endo_simul = oo_.endo_simul; end
    ex.exo_simul = oo_.exo_simul;
    ex.endo_names = M_.endo_names; ex.exo_names = M_.exo_names;
    save('-mat7-binary', '../out_foresight/${m}.mat', 'ex');
    printf('gravado out_foresight/${m}.mat\n');
  " > "../out_foresight/${m}.log" 2>&1 ) || { echo "FALHOU $m — ver out_foresight/${m}.log"; tail -15 "out_foresight/${m}.log"; continue; }
  tail -1 "out_foresight/${m}.log"
done
rm -rf work
echo "pronto."
