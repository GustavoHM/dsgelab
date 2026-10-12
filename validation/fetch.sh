#!/usr/bin/env bash
# fetch.sh — baixa os .mod de referência para validation/mods/.
#
# Os arquivos NÃO são versionados aqui: são GPL, e este pacote é MIT. Vendorizá-los misturaria
# licenças sem necessidade, já que são públicos e estáveis.
set -euo pipefail
mkdir -p "$(dirname "$0")/mods/ramsey" "$(dirname "$0")/mods/examples" "$(dirname "$0")/mods/foresight" \
         "$(dirname "$0")/mods/forecast"
cd "$(dirname "$0")/mods"

get() { curl -fsSL --retry 5 --retry-all-errors --retry-delay 3 -o "$2" "$1" && echo "ok  $2" || echo "FALHOU  $2  ($1)"; }

# Exemplos do tutorial do Dynare (Collard 2001), citados em dynare.org/resources
get https://www.dynare.org/assets/tutorial/example1.mod example1.mod
get https://www.dynare.org/assets/tutorial/example2.mod example2.mod
# Coleção de Johannes Pfeifer, também citada em dynare.org/resources
get https://raw.githubusercontent.com/JohannesPfeifer/DSGE_mod/master/RBC_baseline/RBC_baseline.mod RBC_baseline.mod

# Repositório oficial do Dynare, em tags fixas para que a referência não mude sob os pés.
# Os exemplos saíram da árvore na série 7; vêm da 6.4. Os testes de política ótima vêm da 7.1,
# a versão instalada.
DY=https://git.dynare.org/Dynare/dynare/-/raw
for f in fs2000 NK_baseline Gali_2015 example3 agtrend; do
  get "$DY/6.4/examples/$f.mod" "examples/$f.mod"
done
# auxiliares em MATLAB que dois deles chamam para o estado estacionário
get "$DY/6.4/examples/NK_baseline_steadystate.m" examples/NK_baseline_steadystate.m
get "$DY/6.4/examples/example3_steady_state_helper.m" examples/example3_steady_state_helper.m
get "$DY/7.1/tests/optimal_policy/nk_ramsey_model.mod"  ramsey/nk_ramsey_model.mod
get "$DY/7.1/tests/optimal_policy/Ramsey/ramsey_ex.mod" ramsey/ramsey_ex.mod
get "$DY/7.1/tests/optimal_policy/OSR/osr_example.mod" ramsey/osr_example.mod
get "$DY/7.1/tests/discretionary_policy/dennis_1.mod"  ramsey/dennis_1.mod
# Previsão perfeita, com e sem erros de expectativa (tests/deterministic_simulations, 7.1)
for f in pfwee.mod pfwee.csv pfwee_constant_sim_length.mod rbc_det1.mod rbc_det2.mod rbc_det3.mod; do
  get "$DY/7.1/tests/deterministic_simulations/$f" "foresight/$f"
done
# Estimação: os dados do fs2000 (script MATLAB que monta as séries) e o teste de previsão condicional
get "$DY/6.4/examples/fs2000_data.m" examples/fs2000_data.m
get "$DY/7.1/tests/conditional_forecasts/1/fs2000_cal.mod" forecast/fs2000_cal.mod
# OccBin: uma restrição (borrcon) e duas (dynrbc), de tests/occbin do Dynare 7.1
mkdir -p occbin
for f in model_borrcon/borrcon.mod model_borrcon/borrcon_common.inc \
         model_irrcap_twoconstraints/dynrbc.mod model_irrcap_twoconstraints/dynrbc_common.inc; do
  get "$DY/7.1/tests/occbin/$f" "occbin/$(basename "$f")"
done
# Trajetória estendida (tests/ep do Dynare 7.1): Burnside (1998), com solução fechada, e um
# modelo linear
mkdir -p ep
for f in burnside.mod linearmodel0.mod; do
  get "$DY/7.1/tests/ep/$f" "ep/$f"
done
# Modelos não estacionários: o fs2000 escrito com tendências (trend_var, var(deflator=)), que o
# Dynare destendencia, e os testes de bgp.write (crescimento balanceado)
get "$DY/7.1/tests/trend_var/fs2000_nonstationary.mod" examples/fs2000_nonstationary.mod
mkdir -p bgp
for f in solow-1/solow.mod ramsey-1/ramsey.mod nk-1/nk.mod; do
  get "$DY/7.1/tests/bgp/$f" "bgp/$(dirname "$f").mod"
done
