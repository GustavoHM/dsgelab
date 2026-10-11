# Roadmap

Plano revisto em 11 de outubro de 2026. O objetivo é deixar o `dsgelab` **o mais completo
possível**: a **versão 1.0** cobre o que as ferramentas de referência oferecem — Dynare,
MacroModelling.jl, gEconpy, dolo, sequence-jacobian — e ainda os buracos que os modelos aplicados
revelaram nos objetos que o pacote já tem. O **release público** acontece na **versão 0.5**.

Este plano substitui o de 25 de setembro, que deixava de fora, de propósito, a estimação completa,
os métodos globais, os agentes heterogêneos, o estado estacionário analítico declarado e as
defasagens e avanços no texto, apontando para as ferramentas que já os têm. Em 11 de outubro a
decisão foi revertida: tudo isso entra. Os argumentos da exclusão não foram descartados. Viraram
**condições de desenho** dos itens correspondentes, registradas junto de cada um.

A comparação item a item com as outras ferramentas está em
[`docs/comparison.md`](docs/comparison.md). O `cgelab` avança pelos mesmos marcos, no roadmap dele:
"0.3" quer dizer a 0.3 de cada um.

## Onde estamos

Versão 0.3.0, em `https://github.com/GustavoHM/dsgelab`; reúne os marcos 0.2 e 0.3. Estado estacionário com positividade e
âncora externa; Klein/QZ com Blanchard–Kahn imposto e, opcionalmente, o critério de raiz unitária
do Dynare; impulso, momentos e decomposição de variância; segunda ordem com correção de risco,
média ergódica em forma fechada, simulação podada e impulso generalizado; terceira ordem com o
bloco cúbico e a inclinação de risco; previsão perfeita com trajetória anunciada de parâmetros e de
inovações; decomposição histórica e previsão condicional; calibração conjunta com diagnóstico de
posto; estimação linear por Kalman e Metropolis; projeção de Chebyshev; equações em texto com
derivadas exatas até a hessiana; derivação de CPOs; **o planejador de Ramsey**; **leitura de
`.mod`**; exportação para Dynare e MacroModelling.jl.

A validação contra o Dynare cobre onze modelos: três portados à mão, oito `.mod` lidos sem porte
(entre eles cinco da pasta `examples/` do próprio Dynare) e três problemas de Ramsey, um deles do
conjunto de testes de política ótima do Dynare. Detalhes em [`validation/`](validation/README.md).

## O que conta como pronto

Um item só sai do roadmap quando cumpre as seis condições abaixo. A garantia que o pacote oferece
é que cada número foi conferido contra uma referência independente, e um recurso sem essa
conferência rebaixaria o pacote inteiro.

1. Implementado com API pública em `__all__`.
2. **Validado contra uma referência independente**: outra implementação (Dynare,
   MacroModelling.jl, gEconpy, dolo, sequence-jacobian), uma solução fechada ou um resultado
   publicado. A validação fica num script que qualquer pessoa consegue rodar.
3. Testes na suíte do pacote, sem dependência de ferramentas externas. O que precisa do Dynare ou
   do Julia fica em `validation/`.
4. Documentado na referência da API, com exemplo executável no site, conferido por
   `tools/check_site_examples.py`.
5. Linha correspondente atualizada em `docs/comparison.md`.
6. Entrada no [`CHANGELOG.md`](CHANGELOG.md).

## Marcos

### 0.2 — política ótima, bem-estar e os buracos dos objetos que já existem

- [x] **Ramsey com compromisso**, perspectiva atemporal, e o seu estado estacionário
  (`ramsey_model`, `ramsey_steady_state`). Validado contra a regra de metas e a forma fechada de
  Galí (2015, cap. 5) e contra o `ramsey_model` do Dynare em três modelos, inclusive o
  `tests/optimal_policy/nk_ramsey_model.mod` do Dynare, sem alteração: estado estacionário a 5e-11,
  IRFs das variáveis e dos multiplicadores a 5e-13.
- [x] **Regras simples ótimas** (`osr`): perda quadrática nos momentos incondicionais, com aviso
  quando o ótimo cai na fronteira. No `osr_example.mod` do Dynare a perda bate com a do Dynare a
  2,5e-15 no ótimo dele, e o `dsgelab` acha um ótimo de perda menor, que o próprio Dynare confirma.
  Também sobre o bem-estar (`objective=`), o `osr` com `planner_objective` do Dynare.
- [x] **Política discricionária** (`discretionary_policy`), linear-quadrática, pelo algoritmo de
  Dennis. Validada contra a forma fechada de Galí e contra o `dennis_1.mod` do Dynare.
- [x] **Bem-estar** (`welfare`, `consumption_equivalent`): a função-objetivo viaja com o modelo;
  bem-estar de segunda ordem, condicional e incondicional, e, no Ramsey, com os multiplicadores
  defasados em zero; equivalente proporcional numa variável nomeada, sem assumir log nem consumo.
  Bate com o `evaluate_planner_objective` do Dynare a 1e-12.
- [x] **Impulso generalizado de segunda ordem** (`SecondOrderSolution.irf`): simulações podadas com
  e sem o choque, porque a resposta depende do estado em que o choque chega e do seu sinal.
- [x] **Sequência de inovações na previsão perfeita** (`perfect_foresight(..., shocks=)`), validada
  contra o `perfect_foresight_solver` do Dynare nos `rbc_det1`–`3` dele a 1e-13.
- [x] **Previsão perfeita com erros de expectativa** (`perfect_foresight_with_expectation_errors`),
  validada contra o solver do Dynare nos `pfwee.mod` e `pfwee_constant_sim_length.mod` a 3e-12.
- [x] **Ramsey e discrição a partir do `.mod`**: `from_mod` lê `planner_objective`,
  `ramsey_model`, `discretionary_policy`, `osr_params` e `optim_weights`; os testes de política
  ótima do Dynare rodam sem tradução manual.

### 0.3 — estimação, previsão e decomposição

Validado contra o Dynare no `fs2000.mod` da pasta `examples/` dele, com os seus priors e dados
(`validation/compare_estimation.py`).

- [x] Filtro de Kalman sobre a solução linear, moda da posteriori e Metropolis–Hastings
  (`kalman_filter`, `posterior_mode`, `metropolis_hastings`). Verossimilhança a 6e-12 e densidade a
  priori a 4e-13 contra o Dynare; moda com núcleo maior ou igual ao do Dynare; médias do Metropolis
  a menos de 2,5 erros de Monte Carlo.
- [x] **Decomposição histórica** (`shock_decomposition`) e **previsão condicional**
  (`conditional_forecast`, com `controlled`), contra o `shock_decomposition` (5e-15) e o
  `conditional_forecast` do Dynare (`tests/conditional_forecasts`, 1e-13).
- [x] **Equações de observação** e **erro de medida** (`state_space`, `EstimationProblem`):
  observáveis em nível ou em log (`loglinear` com `logdata`), desvios-padrão de medida fixos ou
  estimados (`"stderr y"`), dados faltantes.
- [x] **Suavizador de Kalman** (`kalman_smoother`) e **inicialização difusa exata** para raízes
  unitárias, contra o `diffuse_filter` do Dynare: verossimilhança a 2e-12, suavizador a 1e-15.
- [x] **Máxima verossimilhança** (`maximum_likelihood`), com erros-padrão pela hessiana.
- [x] **Estimação bayesiana completa**: priors do Dynare por média e desvio (`prior`), SMC
  (`smc`), R-hat e tamanho efetivo de amostra, verificação preditiva a priori com a fração de
  sorteios determinados (`prior_predictive`). Validada contra formas fechadas (o caso conjugado
  normal) e contra o `estimation` do Dynare no `fs2000.mod`.
- [x] **Verossimilhança marginal**: Laplace (a fórmula reproduz o número do Dynare a 2e-13 com a
  covariância dele) e média harmônica modificada; `model_probabilities` para a comparação.
- [x] **Decomposição histórica em tempo real** (`realtime_shock_decomposition`, 5e-15 contra o
  Dynare) e **previsão com bandas** (`forecast`, com incerteza do estado e dos choques, contra o
  `forecast` do Dynare; `posterior_forecast`, com a incerteza dos parâmetros).

### 0.4 — não linearidades e declaração do modelo

- [x] **Importar `.mod`** (`from_mod`), com macros, variáveis locais, `steady_state()`, avanços e
  defasagens de qualquer ordem. Validado em oito `.mod`, lidos sem alteração, contra o Dynare.
- [x] **Defasagens e avanços arbitrários**, no caminho do `.mod`, com toda variável auxiliar listada
  em `ModFile.auxiliary`. *Falta o mesmo em `from_equations`. Condição de desenho, herdada da
  exclusão anterior: o que é predeterminado continua declarado num lugar só, e nenhuma auxiliar
  nasce em silêncio.*
- [ ] **Restrições ocasionalmente ativas** pelo método linear por partes de Guerrieri e Iacoviello,
  sobre o mesmo Klein. Validação contra o `+occbin` do Dynare.
- [ ] **Trajetória estendida** (*extended path*). Validação contra o `extended_path` do Dynare.
- [ ] **Estado estacionário analítico declarado**, inclusive o `steady_state_model` de um `.mod`.
  *Condição de desenho:* um bloco analítico é um segundo modelo do mesmo ponto, e por isso o pacote
  confere que ele anula o resíduo do modelo dinâmico antes de usá-lo; se não anular, recusa.
- [ ] **Equações de calibração dentro do modelo**, à maneira do MacroModelling.jl.
- [ ] **Vários agentes num equilíbrio só**: um montador que junta problemas de otimização às
  equações de fechamento num único `Model`. Os limites são os de `Problem`; não é uma linguagem de
  blocos.
- [ ] **Homotopia e continuação** para estados estacionários difíceis.
- [ ] **Modelos não estacionários**: tendências declaradas, com verificação do crescimento
  balanceado. Validação contra o `+bgp` do Dynare. A raiz unitária já é aceita por
  `qz_criterium`; falta a declaração da tendência.
- [ ] **Diagnóstico do modelo**: singularidades, equações redundantes, variáveis que não aparecem.

### 0.5 — release público

Nenhum recurso novo. Os itens abaixo são condição para publicar.

**Bloqueantes**

- [x] **Renomear.** `dsgekit` estava tomado no PyPI por um pacote concorrente, *"A Python toolkit
  for DSGE models"*. O pacote agora é `dsgelab`, livre no PyPI em 11 de outubro de 2026. Falta
  reservar o nome com um primeiro upload ao TestPyPI e ao PyPI.
- [x] **Um repositório canônico.** `https://github.com/GustavoHM/dsgelab`. As cópias do repositório de pesquisa foram arquivadas.
- [ ] **Idioma.** Código, docstrings e `docs/api.md` estão em inglês; o site, o roadmap, a
  comparação e a validação, em português. O site público precisa de versão em inglês, como o do
  `cgelab` já tem.
- [ ] **Validação reproduzível por terceiros.** A evidência publicada roda na mão de qualquer um:
  os `.mod` do Dynare, as soluções fechadas e o Heutel (2012). Falta tirar da documentação pública
  as afirmações que só o modelo do artigo sustenta, ou marcá-las como tal.

**Engenharia de publicação**

- [ ] Integração contínua: testes em Python 3.10 a 3.14, em Linux, macOS e Windows;
  `check_site_examples.py`. Há um `tests.yml`; falta a matriz completa.
- [ ] Validações externas como tarefa agendada na CI: Octave e Dynare no Ubuntu; Julia com
  MacroModelling.jl.
- [ ] Instalação testada a partir da *wheel*: no Python 3.14 a instalação editável pode quebrar em
  silêncio no macOS.
- [ ] `python -m build`, `twine check`, publicação por *trusted publishing*, TestPyPI antes.
- [ ] DOI no Zenodo a cada release; `CITATION.cff` com ORCID e DOI.
- [ ] `CONTRIBUTING.md`, código de conduta, modelos de *issue* e de *pull request*.
- [ ] `py.typed` e anotações de tipo na API pública.
- [ ] Seção de limitações conhecidas no README.

**Política de versões a partir da 0.5.** Versionamento semântico. Antes da 1.0 uma versão menor
ainda pode mudar a API, mas toda remoção ou mudança incompatível passa antes por um aviso de
descontinuação que dura pelo menos uma versão menor. A API pública é o que está em `__all__`.

### 0.6 — identificação, momentos e dados

- [ ] **Identificação local** (Iskrev; Komunjer e Ng). Validação contra o `+identification` do Dynare.
- [ ] **Sensibilidade global**: região de estabilidade e determinação no espaço de parâmetros,
  filtragem de Monte Carlo. Validação contra o `+gsa` do Dynare.
- [ ] **Método dos momentos**: GMM, SMM e ajuste de IRF. Validação contra o `+mom` do Dynare.
- [ ] **Restrições de calibração sobre IRFs e momentos** (`irf_calibration`, `moment_calibration`).
- [ ] **Momentos teóricos filtrados** (Hodrick–Prescott e passa-banda).
- [ ] **Dados de frequência mista**.
- [ ] **Gráficos**: impulsos, simulações, decomposições, previsões, posteriores.

### 0.7 — terceira ordem completa e derivadas

- [ ] **Núcleo de perturbação na forma de regra de decisão do Dynare**, com a inovação como
  argumento (`g_u`, `g_uu`, `g_ku`). A inclinação de risco `g_σσk` já está; o tensor com a inovação
  em todos os monômios, não.
- [ ] **Estado estacionário estocástico** e IRFs generalizadas analíticas. A prova é a ida e volta
  contra o Dynare em `order=3`.
- [ ] **Terceira ordem no tamanho do modelo aplicado**: tensores esparsos. No sistema de 71
  variáveis, `m = 2n + n_ε` fica perto de 150 e o tensor denso não cabe.
- [ ] **Derivadas da solução em relação aos parâmetros**, por diferenciação automática, e
  **estimação com gradiente** (HMC/NUTS).
- [ ] **Filtros não lineares**: inversão e partículas, para estimar em segunda e terceira ordem.
  Validação contra o MacroModelling.jl.

### 0.8 — métodos globais

- [ ] **Iteração de tempo, de função-valor, grade endógena, grade esparsa.** A colocação de
  Chebyshev até três estados já existe e reproduz a solução fechada do crescimento estocástico.
  Validação contra o dolo.
- [ ] **Discretização de processos** (Tauchen, Rouwenhorst).

### 0.9 — a fronteira

- [ ] **Troca de regime de Markov** na solução e na estimação. Validação contra o RISE.
- [ ] **Agentes heterogêneos** pelo método do jacobiano no espaço de sequências. Validação contra o
  sequence-jacobian. *Condição de desenho, herdada da exclusão anterior:* `vec` continua sendo a
  forma de escrever um número finito de grupos; uma distribuição é outro algoritmo, num módulo
  próprio.
- [ ] **BVAR e DSGE-VAR**. Validação contra o `+bvar` e o `dsge_var` do Dynare.

### 1.0

Toda a lista acima coberta, API congelada, e a partir daí versionamento semântico estrito.

## O `cgelab` como chamador

O equilíbrio de um período, calibrado numa matriz de contabilidade social, continua no `cgelab`.
O que passaria para cá é o resíduo intertemporal, nos pontos em que um agente olha para a frente.
A matriz não entra neste pacote, e a dependência seria opcional: só o módulo-ponte do `cgelab`
importaria o `dsgelab`. A lista completa está no roadmap do `cgelab`, na seção *Extensions that
call dsgelab*.

## Riscos

**Escopo.** Vários itens equivalem cada um a um pacote inteiro: agentes heterogêneos é o escopo do
sequence-jacobian; métodos globais, o do dolo; troca de regime, o do RISE. Com o padrão de
validação deste roadmap, cada um é trabalho de meses. Se algum travar, a saída é publicar primeiro
uma ponte validada para a ferramenta que já o faz — como o exportador para o Dynare já faz com a
terceira ordem — e manter o item aberto até a implementação própria passar na mesma validação.

**Dependências.** Estimação, diferenciação automática e métodos globais puxam dependências pesadas.
O núcleo continua com numpy e scipy; o resto entra como extra opcional, para que quem só resolve
modelos não pague por isso.

**Validação dependente de ferramentas externas.** Quase toda a evidência vem de comparar com o
Dynare e o MacroModelling.jl. Se uma dessas ferramentas mudar de comportamento entre versões, as
validações quebram sem que o pacote tenha mudado. Por isso os `.mod` vêm de tags fixas do Dynare
(6.4 para os exemplos, 7.1 para os testes), e a versão usada fica registrada nos scripts.
