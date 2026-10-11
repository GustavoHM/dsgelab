# Validação contra o Dynare

O `dsgelab` é verificado em três camadas, e esta é a terceira.

| Camada | Contra o quê | Onde |
|---|---|---|
| Respostas conhecidas no papel | RBC de forma fechada, princípio de Taylor, correção de risco lognormal | `../tests/` |
| O modelo que motivou o pacote | `e50` do artigo, mais Dynare e MacroModelling.jl | `artigo-dinamico-gd/code/e71`, `e72` |
| **Modelos de terceiros** | **os exemplos do próprio Dynare** | **esta pasta** |

As duas primeiras camadas partilham um autor comigo. Esta não: os modelos vêm de
[dynare.org/resources](https://www.dynare.org/resources/) — os exemplos do tutorial de Collard e a
coleção de Johannes Pfeifer — e o solucionador de referência é o Dynare.

## Como rodar

```bash
bash fetch.sh          # baixa os .mod (não versionados: são GPL, o pacote é MIT)
bash run_dynare.sh     # roda cada um no Octave + Dynare, exporta out/<nome>.mat
python compare.py      # confronta o dsgelab com o Dynare
```

Precisa de GNU Octave e Dynare. `run_dynare.sh` procura o diretório `matlab/` do Dynare nos lugares
usuais; `DYNARE_MATLAB=/caminho` força.

## Resultado

```
example1       e.e. 1.2e-14   IRFs 3.2e-14   momentos 2.5e-14   OK
example2       e.e. 2.1e-05   IRFs 4.3e-06   momentos 3.3e-06   OK  (limitado pelo e.e. do Dynare)
rbc_baseline   e.e. 0.0e+00   IRFs 3.8e-14   momentos 3.5e-14   OK
```

## Os modelos

| Arquivo | O que exercita |
|---|---|
| `example1.mod` | Collard (2001). Choques **correlacionados** e dois processos exógenos que se alimentam um do outro, de modo que a transição dos estados não é diagonal. |
| `example2.mod` | O mesmo modelo em **logaritmos**. A alocação real tem de ser a mesma; é um teste da escolha de coordenadas do pacote, que carrega desvios proporcionais para variáveis com e.e. não nulo e absolutos para as demais. |
| `RBC_baseline.mod` | Pfeifer, calibrado a dados dos EUA. Quinze variáveis, seis identidades logarítmicas **puramente estáticas** — linhas nulas em `A`, feixe QZ singular. É onde falha qualquer implementação que inverta `A`. |

## O que a comparação mede, e por que nessa ordem

1. **Estado estacionário**, variável a variável.
2. **Respostas a impulso**, todas as variáveis e todos os choques, 40 períodos.
3. **Desvios-padrão teóricos** — e este é o teste que os dois primeiros não fazem. Uma IRF por
   impulso no estado não passa pelo carregamento de choque `Q`; os momentos passam. Um `Q` errado
   sobrevive aos dois primeiros testes e morre no terceiro.

Cada `.mod` traz as suas próprias opções de `stoch_simul` — um filtra com HP, outro simula 2000
períodos. `run_dynare.sh` reexecuta `stoch_simul` com a mesma configuração para todos (ordem 1, 40
períodos de IRF, momentos teóricos, sem filtro), para que a comparação seja de modelos e não de
opções de relatório.

## Duas convenções que a comparação tem de reconciliar

**Datação do estoque.** O `k` do Dynare é o estoque de FIM de período; o do `dsgelab` é o
DISPONÍVEL no período, que é o `k(-1)` do Dynare. As séries diferem por um período. A comparação
desloca exatamente as variáveis predeterminadas cuja linha de `Q` é nula — que são as que não
recebem inovação diretamente. Trocar as duas datações dá um modelo que resolve, satisfaz
Blanchard-Kahn e responde a choques, e não é o mesmo modelo; só a comparação dinâmica revela.

**Ortogonalização.** O Dynare reporta IRFs a impulsos ORTOGONALIZADOS: a resposta ao choque *i* é a
resposta à coluna *i* do fator de Cholesky da covariância, não a um impulso isolado em *i*. Com
covariância diagonal as duas coincidem; com o `var e, u = phi*0.009^2` do `example1`, não. O
`dsgelab` tem `irf_impulse` para exatamente isto.

## O que a validação encontrou

**No `example2`, o `steady` do Dynare para antes.** O estado estacionário do Dynare deixa um resíduo
de `1,1e-06` nas equações do modelo; o do `dsgelab` deixa `1,0e-44`. Tudo o que decorre do e.e. —
linearização, IRFs, momentos — herda esse erro, e é essa a origem inteira da discrepância de `3e-06`
que a tabela mostra. Quando dois programas discordam, a pergunta útil não é qual número é o certo,
mas qual dos dois PONTOS anula as equações; `compare.py` mede o resíduo em ambos e é assim que o
veredito é formado.

A confirmação independente está à mão: `example1` e `example2` são a mesma economia em coordenadas
diferentes, e `exp(e.e. do example2)` bate com o e.e. do `example1` em nível a `1e-15`.

**Um erro meu, que só a dinâmica revelou.** No porte do `RBC_baseline` escrevi `log_k = log(k)` onde
o `.mod` diz `log_k = log(k)` referindo-se ao `k` CORRENTE do Dynare, que é o estoque de fim de
período — o `k(+1)` na convenção do pacote. O estado estacionário coincidia exatamente; a IRF de
`log_k` estava atrasada um período. É o argumento para comparar trajetórias e não pontos.

## Política ótima: o planejador de Ramsey

```bash
bash fetch.sh            # inclui mods/ramsey/, do conjunto de testes do Dynare 7.1
bash run_ramsey.sh       # ramsey_model + stoch_simul(order=1) no Octave + Dynare
python compare_ramsey.py
```

```
nk_linear         e.e. 0.0e+00   IRFs 1.4e-15   (6 séries; modelo com 8 variáveis, 4 do planejador)   OK
rbc_labortax      e.e. 5.0e-11   IRFs 4.9e-13   (8 séries; modelo com 11 variáveis, 6 do planejador)   OK
nk_ramsey_model   e.e. 2.7e-12   IRFs 2.1e-14   (8 séries; modelo com 13 variáveis, 8 do planejador)   OK
```

| Modelo | Origem | O que exercita |
|---|---|---|
| `ramsey/nk_linear.mod` | escrito aqui | O novo-keynesiano de livro-texto (Galí, 2015, cap. 5). Tem resposta analítica, a regra de metas `π_t = −(λ/κ)(x_t − x_{t−1})` e a forma fechada de `x_t`; `tests/test_optimal.py` confere as duas sem o Dynare. |
| `ramsey/rbc_labortax.mod` | escrito aqui | Um RBC com imposto sobre o trabalho como instrumento. O equilíbrio com `τ = 0` é eficiente e o planejador o mantém em zero. Exercita a condição do planejador para um **estoque**, `k(+1)`, escolhido em `t` e válida em esperança. |
| `mods/ramsey/nk_ramsey_model.mod` | testes do Dynare (`tests/optimal_policy`) | Não linear: Phillips de Rotemberg e Euler com `c(+1)` e `pai(+1)`. As derivadas das restrições em relação a `t+1` dependem de `t`, e a condição precisa de cópias defasadas `lag_c` e `lag_r`. Roda **sem alteração**. |

São comparados o estado estacionário e as IRFs de todas as variáveis privadas **e dos
multiplicadores**. O `MULT_j` do `dsgelab` é o multiplicador da j-ésima equação que não é processo
exógeno; o Dynare numera um por equação, inclusive as exógenas, e dá à mesma equação o mesmo valor,
com o mesmo sinal.

**Duas armadilhas do lado do Dynare.** O `nk_ramsey_model.mod` chama `stoch_simul` sem `order=`, e a
omissão do Dynare é 2: as IRFs que ele produz são impulsos generalizados de segunda ordem, e diferem
dos de primeira em `4e-05`. `run_ramsey.sh` fixa `order=1`. Além disso, `oo_.steady_state` guarda o
`initval` do `.mod` (`c = n = 0,9671684882`), que erra a raiz exata `c = 0,875^{1/4}` na sétima casa;
o ponto em que o Dynare de fato lineariza é `oo_.dr.ys`, e é contra ele que a comparação é feita.

## Os exemplos do Dynare, lidos sem porte: `from_mod`

`compare.py` confronta portes feitos à mão. Aqui o `.mod` é lido como está por
`dsgelab.from_mod`, e o que se compara é o que os dois programas tiram do mesmo arquivo. Além dos
três modelos acima, cinco da pasta `examples/` do repositório do Dynare (tag 6.4).

```bash
bash fetch.sh
bash run_examples.sh       # strip_mod.py reduz cada .mod ao modelo; stoch_simul(order=1) igual para todos
python compare_import.py
```

```
modelo         vars  aux    parâm.     e.e.     IRFs  momentos
RBC_baseline     15    5   3.6e-16  1.1e-13  6.8e-14   7.5e-14  OK
example1          6    5   0.0e+00  1.2e-14  1.8e-14   1.9e-14  OK
example2          6    5   0.0e+00  2.1e-05  4.3e-06   3.3e-06  OK  (e.e. do Dynare com resíduo 1e-06)
Gali_2015        14    7   0.0e+00  0.0e+00  4.9e-15   2.2e-15  OK
NK_baseline      28   16   1.8e-16  9.3e-14  5.9e-11   2.2e-11  OK
              parâmetros que o .mod deixa ao MATLAB, tomados do Dynare: gammma1, Rbar, Lambdax, LambdaYd
              ponto de partida do e.e. tomado do Dynare para 28 variável(is)
agtrend          15    6   0.0e+00  3.9e-08  2.1e-07   2.3e-07  OK  (e.e. do Dynare com resíduo 4e-07)
example3          6    5   0.0e+00  8.3e-15  1.7e-14   9.2e-15  OK
              ponto de partida do e.e. tomado do Dynare para 4 variável(is)
fs2000           14    8   3.2e-16  9.2e-15  2.6e-13   2.5e-13  OK
```

| Modelo | O que o leitor precisa entender |
|---|---|
| `fs2000` | `c(+2)`; choques **dentro** da produção e do crédito, não só num AR(1); parâmetro chamado `del`, palavra reservada do Python |
| `Gali_2015` | `model(linear)`; variáveis locais `#`, uma delas chamada `lambda`; o operador `steady_state(y)`; o **nível de preços** com raiz unitária |
| `agtrend` | o macroprocessador: `@#define`, `@#if`/`@#else` aninhados |
| `example1`, `example3` | choques correlacionados, com a constante `phi` atribuída fora de `parameters` |
| `NK_baseline` | 28 equações, indexação e rigidez de preços e salários, `predetermined_variables` |
| `RBC_baseline` | parâmetros calculados dentro do `steady_state_model` |

**O que veio do Dynare, e está dito na saída.** O `NK_baseline` calcula quatro parâmetros e o
estado estacionário inteiro num arquivo MATLAB (`NK_baseline_steadystate.m`), e o `example3` resolve
as horas num auxiliar MATLAB. Um leitor de `.mod` não executa MATLAB: esses parâmetros e esses
pontos de partida vêm do Dynare. O estado estacionário em si é resolvido pelo `dsgelab` e comparado.
Os parâmetros que o `.mod` define são lidos por `from_mod` e conferidos contra os do Dynare (coluna
`parâm.`).

**Raiz unitária.** O nível de preços do `Gali_2015` (`pi = p - p(-1)`) tem raiz exatamente 1. O
Dynare a conta como estável (`qz_criterium = 1 + 1e-6`); o `dsgelab`, por omissão, não, e recusa
com `BlanchardKahnError`. `solve(..., qz_criterium=1 + 1e-6)` reproduz a convenção do Dynare, e é
o que `compare_import.py` usa.

## Discrição, regras simples ótimas e bem-estar

```bash
bash fetch.sh
bash run_policy.sh         # osr, discretionary_policy e evaluate_planner_objective no Dynare
python compare_policy.py   # --sem-dynare pula a avaliação da perda do Dynare no ótimo do dsgelab
```

```
OSR — osr_example.mod
  perda do dsgelab no ótimo do Dynare   0.4268335923233261
  perda reportada pelo Dynare           0.4268335923233251   desvio relativo 2.5e-15
  ótimo do dsgelab                      0.4266626273124758   menor ou igual que o do Dynare
     gammax0     dsgelab     0.424427   Dynare     0.445744
     gammac0     dsgelab     1.563685   Dynare     1.686970
     gamma_y_    dsgelab    16.666667   Dynare    16.660559
     gamma_inf_  dsgelab     3.977354   Dynare     3.563333
  perda do PRÓPRIO Dynare no ótimo do dsgelab 0.4266626273124780   desvio 5.1e-15

Discrição — dennis_1.mod
  IRFs (5 variáveis x 3 choques x 40)   desvio relativo máx 4.3e-08
  bem-estar incondicional  dsgelab 100.2259173048   Dynare 100.2259172521   5.3e-10
  bem-estar condicional    dsgelab 99.8889224747   Dynare 99.8889224138   6.1e-10
     (com a primeira inovação da simulação do Dynare, que é o que ele usa; sem choque: 99.0953548618)
  iterações de Dennis: 18

Ramsey lido do .mod — nk_ramsey_model.mod
  9 séries comparadas, multiplicadores incluídos (mesma numeração do Dynare)
  estado estacionário  5.9e-12     IRFs de 1ª ordem  2.6e-14
  bem-estar  incondicional -25.213284815613  Dynare -25.213284815578
             condicional   -25.213284815613  Dynare -25.213284815580
             multiplicadores zerados -25.213116382888  Dynare -25.213116382855   desvio relativo máx 1.4e-12

Bem-estar com capital — rbc_labortax.mod, ordem 2
  incondicional -70.761476342807  Dynare -70.761476342831
  condicional   -70.778245895835  Dynare -70.778245895855   desvio relativo máx 3.4e-13

OSR: OK  discrição: OK  Ramsey do .mod: OK  bem-estar com capital: OK
```

Os três arquivos do Dynare são lidos por `from_mod`, sem tradução: `planner_objective`,
`ramsey_model(...)`, `discretionary_policy(instruments=..., planner_discount=...)`, `osr_params`,
`optim_weights` e `osr_params_bounds` viram `ModFile.planner`, `osr_params`, `optim_weights` e
`osr_bounds`.

**O ótimo do `osr`.** A função de perda do `dsgelab`, avaliada no ótimo que o Dynare reporta, dá o
valor do Dynare a 2,5e-15. O ótimo do `dsgelab` tem perda **menor**, 0,4266626 contra 0,4268336, e a
mesma de três otimizadores (BFGS, Nelder–Mead, Powell) a partir do ponto inicial do `.mod` e do
ótimo do Dynare. O próprio Dynare, avaliado no ponto do `dsgelab` (`+osr/objective`), dá a mesma
perda a 5e-15. É o `csminwel` do Dynare que para antes, numa direção quase plana. No ótimo,
`gamma_y_ = 16,667 = 1/|σ|`: a regra neutraliza por inteiro o choque de demanda, que entra na IS
com coeficiente `σ = −0,06` sobre a taxa de juros. A terceira chamada de `osr` no arquivo usa
`opt_algo=1`, que no Octave exige o pacote `optim`, e `run_policy.sh` a remove da cópia.

**O bem-estar condicional da discrição.** O `evaluate_planner_objective` do Dynare, fora do Ramsey,
toma como inovação do primeiro período `oo_.exo_simul(1,:)` quando uma simulação foi feita — e o
`dennis_1.mod` simula 200 períodos. O número condicional do Dynare depende, portanto, de um sorteio.
`run_policy.sh` exporta esse sorteio, e `DiscretionSolution.welfare(shock=...)` o reproduz; sem
choque inicial o valor é outro, e é o que `welfare()` devolve por omissão.

**A regra simples sobre o bem-estar.** `ramsey/nk_osr_planner.mod`, escrito aqui, é o
`osr` com `planner_objective`: a perda é menos o bem-estar incondicional de segunda ordem. A
função de perda do `dsgelab` no ótimo do Dynare reproduz o valor dele a 2,5e-15, e o ótimo do
`dsgelab` tem perda menor ou igual. Limites nos coeficientes (`osr_params_bounds`) exigem no Dynare
um otimizador com restrições que no Octave depende de pacotes extras; o arquivo não os usa.

**O piso da discrição.** As IRFs batem a 4e-08 porque o Dynare para a iteração de Dennis em
`sqrt(eps)`. O `dsgelab` itera até 1e-12 por omissão, e contra a forma fechada de Galí (2015, cap.
5) o erro fica abaixo de 1e-12 (`tests/test_policy.py`).

## Previsão perfeita, com e sem erros de expectativa

```bash
bash fetch.sh                # mods/foresight/, de tests/deterministic_simulations do Dynare 7.1
bash run_foresight.sh
python compare_foresight.py
```

```
rbc_det1                    condição inicial (histval)             desvio relativo máx 6.6e-15 (Capital)  OK
rbc_det2                    choque no período 1                    desvio relativo máx 6.6e-15 (Capital)  OK
rbc_det3                    choques antecipados, períodos 4 a 8    desvio relativo máx 8.0e-14 (Output)  OK
pfwee                       erros de expectativa                   desvio relativo máx 2.8e-12 (c)  OK
pfwee_constant_sim_length   idem, horizonte constante              desvio relativo máx 3.0e-12 (c)  OK
```

`rbc_det1`–`3` validam `perfect_foresight` e `ModFile.deterministic`: condição inicial por `histval`,
choque no primeiro período e choques **antecipados** (`shocks=`). `pfwee` e
`pfwee_constant_sim_length` validam `perfect_foresight_with_expectation_errors`: a tabela de
informação de `pfwee.csv` — notícias nos períodos 1 a 7, choques transitórios na exógena e mudanças
permanentes no seu nível terminal — vira a lista de `News` por `ModFile.news`.

**Três coisas que a comparação teve de acertar.**

1. **A tolerância do Dynare.** O solver de previsão perfeita para em resíduo `1e-5`
   (`options_.dynatol`) e, com erros de expectativa, recalcula o estado estacionário terminal com
   `options_.simul.steady_tolf = eps^(1/3)` — e esta opção é copiada de `solve_tolf` quando as opções
   são inicializadas, de modo que mudar `solve_tolf` depois não a afeta. Com as tolerâncias por
   omissão o `k` terminal do `pfwee` erra a solução fechada `((δ+β)/(aαx))^(1/(α−1))` em `4e-5`, e
   é essa a origem inteira da diferença. `run_foresight.sh` pede `1e-12` (e `1e-14` no estado
   estacionário) antes de cada `setup` e de cada `solver`.
2. **O teste interno do `pfwee.mod`.** O arquivo termina com uma réplica manual e um teste de
   igualdade exata entre as duas simulações, que deixa de valer com outra tolerância e aborta o
   script. A cópia para em `pfwee_simul`, que é o que se exporta.
3. **Notícia que não é notícia.** A tabela tem colunas que só confirmam o que se esperava (períodos
   4, 5 e 7), e o Dynare não replaneja nelas. Com horizonte constante isso muda o comprimento da
   simulação, e `ModFile.news` as descarta pela mesma regra.

**A exógena como nível.** No Dynare `x` é um nível com estado estacionário próprio (`x = 1` no
`initval` do `pfwee`). `from_mod` a carrega no estado `SHK_x`, com `SHK_x(+1) = EXO_SS_x + x`, e
uma mudança permanente no nível é uma mudança no parâmetro `EXO_SS_x`. O `x(+1)` da Euler é o
próximo valor desse estado.

## Estimação, suavização, decomposição e previsão

```bash
bash fetch.sh
bash run_estimation.sh          # fs2000 (calib, mode, mh), unitroot, condfc
python compare_estimation.py    # --sem-mh pula o Metropolis, que leva alguns minutos
```

O `fs2000.mod` do Dynare é lido por `from_mod` com os seus priors e dados. Duas convenções do
Dynare que a comparação teve de reproduzir: sem valor inicial em `estimated_params`, o ponto de
partida é a **média a priori**; e a verossimilhança só bate na última casa com
`options_.riccati_tol = 1e-16`, porque por omissão o Dynare passa ao ganho estacionário do filtro
quando a variância converge a 1e-6. No filtro difuso, o Dynare põe as raízes unitárias **primeiro**
na forma de Schur, com `P_inf` identidade nesse bloco; o `dsgelab` faz o mesmo, e a verossimilhança
difusa bate a 2e-12. A decomposição em tempo real do Dynare deixa o choque do primeiro período na
condição inicial (a decomposição completa, não); as duas convenções são reproduzidas.
