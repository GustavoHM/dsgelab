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
