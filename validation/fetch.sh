#!/usr/bin/env bash
# fetch.sh — baixa os .mod de referência para validation/mods/.
#
# Os arquivos NÃO são versionados aqui: são GPL, e este pacote é MIT. Vendorizá-los misturaria
# licenças sem necessidade, já que são públicos e estáveis.
set -euo pipefail
cd "$(dirname "$0")/mods"

get() { curl -fsSL --retry 5 --retry-all-errors --retry-delay 3 -o "$2" "$1" && echo "ok  $2" || echo "FALHOU  $2  ($1)"; }

# Exemplos do tutorial do Dynare (Collard 2001), citados em dynare.org/resources
get https://www.dynare.org/assets/tutorial/example1.mod example1.mod
get https://www.dynare.org/assets/tutorial/example2.mod example2.mod
# Coleção de Johannes Pfeifer, também citada em dynare.org/resources
get https://raw.githubusercontent.com/JohannesPfeifer/DSGE_mod/master/RBC_baseline/RBC_baseline.mod RBC_baseline.mod
