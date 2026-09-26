# Roadmap

O que o `dsgekit` ainda não faz. Cada item diz por que importa. A maior parte apareceu quando o
pacote encontrou um modelo aplicado: o DSGE de grupos da geração distribuída, com cerca de 71
variáveis, e o porte do Heutel. Nada aqui é promessa nem data. Um item sobe quando um modelo
aplicado precisa dele.

O núcleo a que o pacote se propôs está feito. Estado estacionário com positividade e âncora
externa, Klein/QZ com Blanchard–Kahn imposto, impulso, momentos, decomposição de variância,
segunda ordem com correção de risco, média ergódica em forma fechada e simulação podada,
terceira ordem no bloco determinístico, previsão perfeita com trajetória anunciada de
parâmetros, calibração conjunta com diagnóstico de posto, verificações, equações em texto com
derivadas exatas até a hessiana, derivação de condições de primeira ordem de um problema e
exportação para Dynare e MacroModelling.jl. O que segue é o que esses objetos ainda não
entregam.

## Em seguida: buracos dos objetos que já existem

**Impulso de segunda ordem.** `irf` existe na solução de primeira ordem. Na de segunda há
`simulate`, `mean_correction` e as regras, e não há impulso. O motivo de pagar a segunda ordem é
uma média e um sinal que a primeira não vê. O objeto que um artigo desenha para isso é o impulso
generalizado: simulações podadas com e sem o choque, a diferença média, porque a resposta depende
do estado em que o choque chega e do sinal do choque. O impulso de primeira ordem continua sendo
o caso em que essa média não depende de nenhum dos dois.

**Uma sequência de inovações na previsão perfeita.** `perfect_foresight` chama o resíduo com a
inovação igual a zero em toda data. Uma trajetória anunciada de parâmetros funciona, e foi o que
o modelo aplicado precisou para as rampas. Uma seca de tamanho conhecido no ano 3, ou um choque
noticiado, não entra, a não ser disfarçada de parâmetro ou de estado inicial. A função de
resíduos já recebe a inovação. Falta o argumento, uma sequência de comprimento `T+1`, no mesmo
lugar em que `params` e `ctx` já aceitam sequência.

**Decomposição histórica e previsão condicional, na solução linear.** `state_path` propaga
inovações que o usuário já tem. Falta o caminho de volta: dadas as variáveis observadas, as
inovações que as reproduzem, e uma previsão em que parte das variáveis é fixada numa trajetória.
As duas contas usam `P`, `Q` e `F`, que a solução já devolve. É o que a seção estocástica de um
artigo aplicado pede depois que os momentos incondicionais estão na mesa.

**O bem-estar da função-objetivo que a derivação já viu.** `optimality_conditions` devolve as
equações e larga o retorno do período. O bem-estar de segunda ordem — a utilidade incondicional
na média podada e na variância de primeira ordem — volta a ser derivado à mão no artigo. O
objetivo declarado tem de viajar com o modelo. A variação equivalente de consumo é uma conta em
cima desse número e da utilidade marginal no estado estacionário, que a mesma função já fornece.
A transformação pede a variável que desfaz o buraco de bem-estar. Não se assume log nem uma
variável chamada consumo.

**Vários agentes num equilíbrio só.** `optimality_conditions` trata um problema recursivo. Um
modelo é vários desses mais as condições de fechamento, e a concatenação à mão é onde um sinal
errado volta depois de as condições terem saído certas. Falta um montador fino: problemas mais as
equações de fechamento que o usuário escreve, um `Model` só, pronto para `from_equations` e para
o exportador. Os limites são os que `Problem` já tem. Restrições de igualdade, nada em `t+2`, e
o fechamento de mercado continua sendo texto do modelador. Não é uma linguagem de blocos.

## A lacuna de forma

**Risco que depende do estado.** `g_kkk` e `h_kkk` estão implementados e o teste de ordem de
acurácia os separa de um bloco cúbico errado. `g_ssk` e `h_ssk` não estão nas matrizes. Uma
derivação anterior recuperou o módulo no RBC de referência, `|h_ssk[a, a]| = ρ σ²` a quatro
algarismos, com o sinal trocado. A solução é escrita `k' = h(k, σ) + Q ε`, com a inovação
aditiva. O termo em que o risco muda a inclinação vive na derivada cruzada em relação à inovação,
e essa forma não tem onde guardá-la. O Dynare escreve `y = g(y(-1), u, σ)`.

O trabalho, se for feito, é a representação: regras que recebem a inovação como argumento, com
`g_u`, `g_uu` e `g_ku`, e só então `g_σσk`. Corrigir o sinal à mão embarca um coeficiente que se
sabe errado. Até essa forma existir, a terceira ordem daqui melhora a aproximação determinística,
e o prêmio de risco variável no tempo sai pelo exportador, com `stoch_simul(order=3)`. A prova
de que o sinal deixou de ser um palpite é a ida e volta contra o Dynare, no mesmo padrão que a
segunda ordem já tem.

A documentação de `solve_third_order` dizia que `cov` entrava em `g_ssk` e `h_ssk`. `cov` é a
covariância guardada para a simulação. Os coeficientes de inclinação não são calculados.

**Terceira ordem no tamanho do modelo aplicado.** No sistema de 71 variáveis, `m = 2n + n_ε` fica
perto de 150. O tensor denso de terceiras derivadas tem ordem `n m³` e não cabe como objeto
simbólico, nem como diferença finita repetida de uma hessiana. O front end simbólico para na
hessiana de propósito: ela já custa `n m²` expressões. Para a terceira ordem rodar aqui nesse
tamanho, o tensor tem de ser esparso. Enquanto não for, o caminho desse modelo é o `.mod`. Nos
modelos pequenos, o bloco cúbico determinístico já roda e já é verificado.

## Só se um modelo aplicado pedir

**Restrições que às vezes seguram.** `check_slack` já diz quando um `max` suavizado saiu da
região em que a perturbação vale, ao longo do impulso e não só no estado estacionário. Não
resolve o modelo nessa região. O algoritmo que cabe no que o pacote tem é o linear por partes de
Guerrieri e Iacoviello, em primeira ordem, sobre o mesmo Klein. Entra no dia em que o diagnóstico
falhar num impulso que o artigo precisa reportar.

**Trajetória estendida.** Previsão perfeita com a expectativa de choque a um período, iterada.
Fica entre a perturbação e um método global. Faz sentido quando a transição precisa ser não
linear e estocástica ao mesmo tempo e a restrição ativa no ponto errado não é o problema, porque
a não linearidade é suave e o choque é grande.

## De propósito, fora

Cada um destes é uma célula vazia numa tabela de capacidades, e o lugar certo de fazê-lo já está
nomeado em [`docs/comparison.md`](docs/comparison.md).

* **Estimação bayesiana, filtro de Kalman e suavizador.** gEconpy, via PyMC, ou pydsge. A
  identificação pela verossimilhança é o mesmo projeto. `target_jacobian` continua sendo o
  diagnóstico de identificação da calibração, que é o objeto que este pacote resolve.
* **Métodos globais e de projeção.** dolo.
* **Agentes heterogêneos no sentido de uma distribuição.** sequence-jacobian ou econpizza.
  `vec` escreve um número finito de grupos, que é o que o modelo aplicado é. Uma distribuição
  sobre os grupos é outro algoritmo.
* **Um segundo bloco de estado estacionário analítico.** `solve(..., steady=)` já aceita o estado
  estacionário que o usuário calculou, e o exportador escreve esse ponto num `initval`. Um bloco
  simbólico paralelo seria um segundo modelo do mesmo estado estacionário.
* **Defasagem `(-1)` e avanço `(+2)` no texto.** Recusados de propósito. Defasagem é um estado
  com a sua equação de transição. Um termo em `t+2` significa que o problema não cabe em dois
  períodos, e `optimality_conditions` já recusa dizendo qual estado o causou.
* **A calibração a partir da matriz de contabilidade social, da POF, do ONS e da ANEEL.** Isso é
  o pipeline do artigo. O pacote começa na função de resíduos.

## O cgekit como chamador

O equilíbrio de um período, calibrado numa matriz de contabilidade social, continua no
`cgekit`. O que passaria para cá é o resíduo intertemporal, nos pontos em que um agente olha
para a frente. A matriz não entra neste pacote. A dependência seria opcional: só o módulo-ponte
do `cgekit` importaria o `dsgekit`, e um modelo estático seguiria com numpy, scipy e pandas.

Quatro usos, todos do lado de lá e resolvidos com o que este pacote já tem:

* uma trajetória anunciada de imposto, tarifa ou preço de carbono, por `perfect_foresight`, que
  já aceita uma sequência de parâmetros;
* choques agregados sobre os estoques de capital que a dinâmica recursiva já carrega, por
  `solve` e `solve_second_order`, com os saldos contábeis substituídos pelo equilíbrio do
  período;
* o deslocamento de bem-estar que vem do risco agregado, por `mean_correction` sobre o consumo
  que o período devolve;
* o estado estacionário em que a Euler fecha, por `steady_state`. Esse ponto é outro que o
  ano-base da matriz, e os dois têm de continuar distintos.

A lista completa, e o que segue sendo conta do `cgekit` (ninhos, tarifas, regiões, fechamento do
governo), está no roadmap dele, na seção *Extensions that call dsgekit*.

## Em volta do pacote

O `cgekit` e o `dsgekit` são repositórios separados. Este está em
`https://github.com/GustavoHM/dsgekit`, versão 0.1.0. Os testes rodam na integração contínua
deste repositório.

A página de replicação está em [`site/replications.html`](site/replications.html). Ela executa
quatro contas que a suíte já confere: o crescimento estocástico com utilidade logarítmica e
depreciação total, contra a solução fechada; a correção de risco do AR(1) lognormal, que sai
`σ²`; o princípio de Taylor no modelo de três equações; e o Heutel (2012) contra a Tabela 2, os
coeficientes publicados das regras com informação defasada e o sistema linear do autor.
