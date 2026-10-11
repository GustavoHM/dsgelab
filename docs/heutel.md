# Heutel: porte dos scripts MATLAB para o dsgelab

O módulo `dsgelab.examples.heutel` porta os sistemas de primeira ordem dos seis
arquivos MATLAB de replicação do artigo.
O solver Anderson–Moore, ausente no anexo, é substituído pelo Klein/QZ do dsgelab.
O pacote instalado só precisa de NumPy e SciPy; MATLAB, Octave e Dynare são
necessários apenas para refazer a validação independente.

Fonte: Garth Heutel (2012), *How should environmental policy respond to business
cycles? Optimal policy under persistent productivity shocks*, Review of Economic
Dynamics 15, 244–264, DOI 10.1016/j.red.2011.05.002. Os scripts têm README de
23/05/2011 e o título anterior “Optimal Environmental Policy Under Economic
Fluctuations”.

## Uso

```python
from dsgelab.examples import heutel

# Imposto ótimo, informação completa; parâmetros do script de imposto.
sol = heutel.solve("tax")
irf = sol.irf("eps_a", size=0.01, periods=100)   # porcentagem; choque de 1%
ss = heutel.steady_levels("tax")
levels = heutel.to_levels(irf, "tax")

# Regras com informação defasada; inclinações originais 1.4151 e 0.6580.
tax_asym = heutel.solve("tax_asym")
quota_asym = heutel.solve("quota_asym")

# Planejador sob a calibração da Tabela 2: escolha EXPLÍCITA.
planner = heutel.solve("planner", params=heutel.Params())

# Simulações usam desvios fracionários, sem multiplicar por 100.
sim = sol.simulate(periods=1000, cov={"eps_a": 0.007}, seed=42)
sim_levels = heutel.to_levels(sim, "tax", percent=False)
moments = sol.moments(cov={"eps_a": 0.007})
```

A interface padrão também funciona:

```python
import dsgelab as dk
model, params, guess = heutel.build("tax_asym")
sol = dk.solve(model, params, guess=guess)
```

`build` declara o sistema **já linearizado**, em desvios com estado estacionário
zero. Assim, `sol.steady` e `sol.irf(levels=True)` se referem às coordenadas desse
sistema, não aos níveis econômicos. Use `steady_levels` e `heutel.to_levels`
para recuperar os níveis econômicos. Ao mudar parâmetros, reconstrua o modelo;
o callback rejeita parâmetros diferentes daqueles que geraram sua matriz.

## Variantes e calibração

| Variante | Script | Calibração padrão |
|---|---|---|
| `planner` | `SolveSystem.m` + `steadystate.m` | θ₂=0,1; γ=0,8, como no arquivo |
| `tax` | `SolveSystem_tax.m` + `steadystate_tax.m` | θ₂=2,8; γ=0,304 |
| `tax_asym` | `SolveSystem_tax_AsymInfo.m` | anterior; τ̂ₜ=1,4151 ŷₜ₋₁ |
| `quota_asym` | `SolveSystem_quota_AsymInfo.m` | anterior; q̂ₜ=0,6580 ŷₜ₋₁ |

`Params()` contém a calibração dos scripts de imposto. `Params.source(variant)`
recupera o padrão específico do arquivo, inclusive o planejador diferente.

A calibração literal do planejador não produziu um estado estacionário
admissível nos procedimentos executados. O solver limitado chega à fronteira
μ≈1, com resíduo máximo ≈0,00965. O porte levanta `SteadyStateError`, em vez de
aceitar a saída de `fsolve` sem verificar consumo, emissões, μ e os resíduos.
Isso não é uma prova de inexistência global. Para o caso-base da Tabela 2,
`heutel.solve("planner", heutel.Params())` resolve, preservando as equações do
script. As comparações numéricas abaixo usam essa escolha explícita.

## Correções operacionais e coordenadas

- O `k` original é o capital de fim de período. `x` é o estoque contemporâneo.
  Os estados auxiliares `lag_k`, `lag_x` e, quando necessários, `lag_lambda` ou
  `lag_y` guardam os valores defasados. `a` é a produtividade corrente.
- O choque obedece à convenção do dsgelab, εₜ₊₁ na transição de `a`. A IRF
  mantém o mesmo período de impacto dos scripts: â₀=0,01.
- No imposto ótimo, λ estacionário é aproximadamente zero. Log-linearizar
  esse multiplicador torna a matriz quase singular. O porte usa **desvios
  absolutos de λ**, avaliando seus coeficientes por unidade de multiplicador.
  Não divide por um número quase zero. Demais variáveis usam desvios
  proporcionais; ω pode ter nível estacionário negativo. A opção `percent=True`
  multiplica também o desvio absoluto de λ por 100, sem transformá-lo numa
  variação percentual econômica.
- Os scripts assimétricos propagam `state_2(:,t)` a partir de `state(:,t-1)`
  em vez da própria trajetória `state_2(:,t-1)`, tanto na IRF quanto na simulação.
  O porte usa a trajetória do regime escolhido; os testes verificam a regra
  de política a cada período.
- `structural_system` entrega a matriz H bruta, na ordem e coordenadas do
  arquivo, antes da normalização de λ feita em `build`.

## Conferência com o artigo enviado

O PDF foi conferido nas pp. 249–251 e 258 (equações, calibração e sensibilidade).
Há diferenças materiais que o porte literal preserva:

1. A Tabela 2, p. 251, traz θ₂=2,8 e 1−γ=0,696. A p. 258 discute exatamente
   θ₂=0,1 e 1−γ=0,2 como sensibilidade. Portanto, o arquivo do planejador parece
   ter sido salvo configurado para essa sensibilidade.
2. A Tabela 2 e os seis scripts usam β=0,98267. A prosa da p. 250 traz 0,98627.
   O porte segue a tabela e o código, sem resolver essa divergência por suposição.
3. O texto, pp. 250–251, mantém as emissões externas constantes em três vezes
   as emissões domésticas estacionárias. Com x̄=4 ē/(1−η), isso implica
   x̂ₜ=η x̂ₜ₋₁+((1−η)/4) êₜ. As matrizes dos scripts usam `(1−η)` como
   coeficiente de êₜ, quatro vezes esse valor. A matriz original foi preservada
   e testada como tal; não deve ser apresentada como implementação exata dessa
   hipótese do texto.
4. Os coeficientes de danos são escalados por 5,3024 nos scripts. A nota 20,
   p. 251, explica a necessidade de compatibilizar as unidades de carbono com
   as unidades do modelo. `Params()` já contém d₁ e d₂ escalados.

Assim, concordância com o Dynare valida a transcrição e a solução dos sistemas
fornecidos. Ela não estabelece que todas as equações publicadas, a calibração
literal de sensibilidade ou todas as figuras do artigo foram reproduzidas.

## Validação independente

`validation/heutel/validate.py` executa os prefixos dos scripts originais em
Octave até a construção de H, sem depender de AIM. Exporta níveis e matrizes,
gera modelos lineares a partir dessas matrizes no Dynare e compara todas as
variáveis: estado estacionário, IRFs de 100 períodos e desvios-padrão teóricos.
No imposto, a mudança de coordenada de λ também é calculada diretamente em
MATLAB a partir dos coeficientes originais. As trajetórias de referência usam
choque/desvio-padrão de 0,01 em ambos os solvers, sem filtro HP.

| Variante | Erro máximo: níveis | Matriz H | IRFs | Desvios-padrão |
|---|---:|---:|---:|---:|
| Planejador, calibração comum | 5,14e-9 | 2,74e-12 | 3,05e-12 | 1,61e-11 |
| Imposto | 9,55e-11 | 5,92e-13 | 8,35e-13 | 5,80e-12 |
| Imposto assimétrico | 9,55e-11 | 4,99e-14 | 5,93e-16 | 1,05e-14 |
| Cota assimétrica | 9,55e-11 | 4,99e-14 | 3,51e-13 | 3,57e-13 |

Comandos, a partir da raiz do monorepositório:

```bash
pytest -q
python validation/heutel/validate.py
```

O primeiro usa referências Octave/Dynare persistidas em `reference.json`, sem
instalação desses programas. O segundo requer Octave e Dynare; aceita variáveis
`OCTAVE` e `DYNARE_MATLAB`. Os hashes dos anexos estão em `source-sha256.json`.
`translate.py` regenera o módulo privado de álgebra: traduz somente atribuições
escalares e coeficientes das regiões selecionadas, sem executar texto MATLAB.

## Limites

Este porte cobre estado estacionário, dinâmica de primeira ordem, IRFs,
simulações e momentos dos sistemas MATLAB. Não porta as regressões Stata;
não estima a inclinação ótima das regras (os anexos só comentam uma chamada
à função `IRFdiff`, que não foi fornecida); não reproduz séries aleatórias sem
seed dos originais. A diferença entre as trajetórias assimétricas originais e
as corrigidas é intencional e verificada pela equação da política.

Uma função de utilidade aplicada à trajetória linear não constitui uma análise
de bem-estar de segunda ordem. Não use `solve_second_order` sobre este modelo
linear como se ele fornecesse essa análise; isso exige declarar e conferir o
sistema não linear completo e reconciliar as diferenças com o artigo.

Os arquivos originais foram preservados localmente em `validation/heutel/original`
para conferência e permanecem atribuídos a Garth Heutel; não são incluídos como
código MIT do dsgelab. Para refazer a validação em outro computador, copie os seis
`.m` e `ReadMe.txt` fornecidos para essa pasta.
