# Roadmap

*English version: [ROADMAP.md](ROADMAP.md).*

Plano revisado em 11 de outubro de 2026 e atualizado em 17 de outubro de 2026. O objetivo é
deixar o `dsgelab` **o mais completo possível**. A **versão 1.0** cobre o que oferecem as
ferramentas de referência (Dynare, MacroModelling.jl, gEconpy, dolo, sequence-jacobian) e preenche
as lacunas que os modelos aplicados revelaram nos objetos que o pacote já tem. O **lançamento
público** ocorre na **versão 0.5**.

Este plano substitui o de 25 de setembro, que deixava de fora a estimação completa, os métodos
globais, os agentes heterogêneos, o estado estacionário analítico declarado e as defasagens e
avanços nas equações em texto. Para esses itens, o plano anterior apontava as ferramentas que já os
têm. Em 11 de outubro essa decisão foi revertida, e todos esses itens passaram a fazer parte do
plano. Os argumentos da exclusão passaram a ser **requisitos de projeto** dos itens
correspondentes, registrados junto de cada um.

A comparação item a item com as outras ferramentas está em
[`docs/comparison.md`](docs/comparison.md). O `cgelab` avança pelos mesmos marcos no roadmap dele,
de modo que "0.3" quer dizer a 0.3 de cada um.

## Onde estamos

A versão 0.5.0, em `https://github.com/GustavoHM/dsgelab`, reúne os marcos 0.2 a 0.4 e o trabalho
de publicação do marco 0.5. Foi lançada em 18 de outubro de 2026, no PyPI e no Zenodo (DOI
[10.5281/zenodo.23456475](https://doi.org/10.5281/zenodo.23456475)). O estado estacionário é
encontrado com positividade e âncora externa, aceito em forma fechada depois de conferido e
calculado com equações de calibração e homotopia. A solução de primeira ordem é a de Klein/QZ, com
a condição de Blanchard–Kahn imposta e, opcionalmente, o critério de raiz unitária do Dynare, e
fornece respostas a impulso, momentos e decomposição de variância. Em segunda ordem, o pacote
calcula a correção de risco, a média ergódica em forma fechada, a simulação podada e as respostas a
impulso generalizadas, e em terceira ordem o bloco cúbico e os termos de risco na inclinação das
regras. A previsão perfeita aceita uma trajetória anunciada de parâmetros e de inovações, e erros
de expectativa. O pacote também trata **restrições ocasionalmente ativas** (OccBin) e a
**trajetória estendida**, determinística e estocástica. **Modelos em níveis** são cobertos, com
tendências declaradas e o caminho de crescimento balanceado, assim como a projeção de Chebyshev.
Para dados, há decomposição histórica e previsão condicional, calibração conjunta com diagnóstico
de posto, filtro e suavizador de Kalman, e estimação bayesiana e por máxima verossimilhança. Os
modelos podem ser escritos como equações em texto, com derivadas exatas até a hessiana e
defasagens e avanços de qualquer ordem. Também podem ser montados pela derivação de CPOs e pela
reunião de vários agentes, ou lidos de um `.mod`, e são exportados para o Dynare e o
MacroModelling.jl. A política ótima abrange o planejador de Ramsey, a política discricionária, as
regras simples ótimas e o bem-estar, e o diagnóstico do modelo completa o conjunto.

A validação contra o Dynare cobre três modelos portados à mão, nove `.mod` lidos sem conversão
manual (cinco da pasta `examples/` do próprio Dynare) e, dos testes do próprio Dynare, os de
política ótima, previsão perfeita, estimação, OccBin, trajetória estendida, crescimento balanceado
e diagnóstico. As equações de calibração são validadas contra o MacroModelling.jl. Os detalhes
estão em [`validation/`](validation/README.md).

A versão 0.7.0 traz os marcos 0.6 e 0.7. Em terceira ordem, a regra de decisão está completa na
forma do Dynare, com todos os blocos cúbicos e de risco, o estado estacionário estocástico e tensores
esparsos para modelos grandes, e a solução tem derivadas analíticas em relação aos parâmetros nas
três ordens. A estimação ganha HMC e NUTS sobre o gradiente da verossimilhança de Kalman, e filtros
de partículas e de inversão em segunda e terceira ordem. Para identificação e dados, há
identificação local, GMM, SMM e ajuste de IRF, projeções locais, momentos filtrados, restrições de
calibração sobre IRFs e momentos, sensibilidade global (mapeamento de estabilidade, filtragem de
Monte Carlo e índices de Sobol), avaliação de previsões fora da amostra, dados de frequência mista
e gráficos. Cada um é validado contra uma referência independente: o Dynare na maioria dos casos, o
SALib para os índices de Sobol, o `forecast::dm.test` do R para o teste de Diebold–Mariano e o
resultado populacional de Plagborg-Møller e Wolf (2021) para as projeções locais. Os detalhes
estão em [`validation/`](validation/README.md).

## O que conta como pronto

Um item só sai do roadmap quando cumpre as seis condições abaixo. O padrão do pacote é que cada
recurso seja validado contra uma referência independente quando ela existe (outra implementação,
uma forma fechada ou um resultado publicado), e alguns recursos se apoiam apenas em testes
internos. Um recurso novo sem essa conferência comprometeria esse padrão para o pacote inteiro.

1. Implementado com API pública em `__all__`.
2. **Validado contra uma referência independente**, que pode ser outra implementação (por ora o
   Dynare e o MacroModelling.jl, e o dolo, o sequence-jacobian e o RISE para os itens planejados
   contra eles), uma solução fechada ou um resultado publicado. A validação fica num script que
   qualquer pessoa consegue rodar.
3. Testes na suíte do pacote, sem dependência de ferramentas externas. O que precisa do Dynare ou
   do Julia fica em `validation/`.
4. Documentado na referência da API, com exemplo executável no site, conferido por
   `tools/check_site_examples.py`.
5. Linha correspondente atualizada em `docs/comparison.md`.
6. Entrada no [`CHANGELOG.md`](CHANGELOG.md).

## Marcos

### 0.2 — política ótima, bem-estar e as lacunas dos objetos que já existem

- [x] **Ramsey com compromisso**, perspectiva atemporal, e o seu estado estacionário
  (`ramsey_model`, `ramsey_steady_state`). Validado contra a regra de metas e a forma fechada de
  Galí (2015, cap. 5) e contra o `ramsey_model` do Dynare em três modelos, inclusive o
  `tests/optimal_policy/nk_ramsey_model.mod` do Dynare, sem alteração. Nos três modelos, o estado
  estacionário coincide até 5e-11, e as IRFs das variáveis e dos multiplicadores, até 5e-13.
- [x] **Regras simples ótimas** (`osr`), com perda quadrática nos momentos incondicionais e aviso
  quando o ótimo cai na fronteira. No `osr_example.mod` do Dynare, a perda coincide com a do
  próprio Dynare até 2,6e-16 no ótimo encontrado por ele, e o `dsgelab` acha um ótimo de perda
  menor, confirmado pelo Dynare. Também otimiza sobre o bem-estar (`objective=`), como o `osr` com
  `planner_objective` do Dynare.
- [x] **Política discricionária** (`discretionary_policy`), linear-quadrática, pelo algoritmo de
  Dennis. Validada contra a forma fechada de Galí e contra o `dennis_1.mod` do Dynare.
- [x] **Bem-estar** (`welfare`, `consumption_equivalent`). A função-objetivo fica armazenada no
  modelo. O bem-estar é calculado em segunda ordem, condicional e incondicional, e, no Ramsey, com
  os multiplicadores defasados em zero. O equivalente proporcional é tomado numa variável nomeada,
  sem supor utilidade logarítmica nem que a variável seja o consumo. Coincide com o
  `evaluate_planner_objective` do Dynare até 1,4e-12 no Ramsey e até 6,1e-10 na política
  discricionária.
- [x] **Respostas a impulso generalizadas de segunda ordem** (`SecondOrderSolution.irf`), a partir
  de simulações podadas com e sem o choque, porque a resposta depende do estado em que o choque
  chega e do seu sinal.
- [x] **Sequência de inovações na previsão perfeita** (`perfect_foresight(..., shocks=)`), validada
  contra o `perfect_foresight_solver` do Dynare nos `rbc_det1`–`3` dele, cujos resultados
  reproduz até 1e-13.
- [x] **Previsão perfeita com erros de expectativa** (`perfect_foresight_with_expectation_errors`),
  validada contra o solver do Dynare nos `pfwee.mod` e `pfwee_constant_sim_length.mod`, com
  diferença de até 3e-12.
- [x] **Ramsey e política discricionária a partir do `.mod`**. O `from_mod` lê `planner_objective`,
  `ramsey_model`, `discretionary_policy`, `osr_params` e `optim_weights`, e os testes de política
  ótima do Dynare rodam sem tradução manual.

### 0.3 — estimação, previsão e decomposição

Validado contra o Dynare no `fs2000.mod` da pasta `examples/` dele, com os seus priors e dados
(`validation/compare_estimation.py`).

- [x] Filtro de Kalman sobre a solução linear, moda a posteriori e Metropolis–Hastings
  (`kalman_filter`, `posterior_mode`, `metropolis_hastings`). Contra o Dynare, a verossimilhança
  coincide até 6e-12 e a densidade a priori até 4e-13, a moda tem núcleo maior ou igual ao do
  Dynare, e as médias do Metropolis ficam a menos de 3 erros-padrão de Monte Carlo das do Dynare
  (o maior, 2,9).
- [x] **Decomposição histórica** (`shock_decomposition`) e **previsão condicional**
  (`conditional_forecast`, com `controlled`), contra o `shock_decomposition` (5e-15) e o
  `conditional_forecast` do Dynare (`tests/conditional_forecasts`, 1,6e-13).
- [x] **Equações de observação** e **erro de medida** (`state_space`, `EstimationProblem`), com
  observáveis em nível ou em log (`loglinear` com `logdata`), desvios-padrão de medida fixos ou
  estimados (`"stderr y"`) e dados faltantes.
- [x] **Suavizador de Kalman** (`kalman_smoother`) e **inicialização difusa exata** para raízes
  unitárias, contra o `diffuse_filter` do Dynare, com a verossimilhança até 2,5e-12 e o suavizador
  até 1e-15.
- [x] **Máxima verossimilhança** (`maximum_likelihood`), com erros-padrão pela hessiana.
- [x] **Estimação bayesiana completa**, com os priors do Dynare por média e desvio (`prior`), SMC
  (`smc`), R-hat e tamanho efetivo de amostra, e a verificação preditiva a priori com a fração de
  sorteios com equilíbrio determinado (`prior_predictive`). Validada contra formas fechadas (o caso
  conjugado normal) e contra o `estimation` do Dynare no `fs2000.mod`.
- [x] **Verossimilhança marginal** por Laplace (a fórmula reproduz o número do Dynare até 2e-13 com
  a covariância dele) e por média harmônica modificada, e `model_probabilities` para comparar
  modelos.
- [x] **Decomposição histórica em tempo real** (`realtime_shock_decomposition`, 5e-15 contra o
  Dynare) e **previsão com bandas** (`forecast`, com incerteza do estado e dos choques, contra o
  `forecast` do Dynare, e `posterior_forecast`, com a incerteza dos parâmetros).

### 0.4 — não linearidades e declaração do modelo

Concluído na versão 0.4.0, em 11 de outubro de 2026.

- [x] **Importar `.mod`** (`from_mod`), com macros, variáveis locais, `steady_state()`, avanços e
  defasagens de qualquer ordem. Validado em nove `.mod`, lidos sem alteração, contra o Dynare.
- [x] **Defasagens e avanços arbitrários**, no `.mod` e em `from_equations`, com toda variável
  auxiliar listada em `auxiliary`. *Nos dois caminhos, como pede o requisito de projeto, o que
  é predeterminado continua declarado num lugar só. A defasagem de uma variável de salto é um
  estado novo com equação própria, e nenhuma variável auxiliar é criada sem aviso.*
- [x] **Restrições ocasionalmente ativas** pelo método linear por partes de Guerrieri e Iacoviello,
  sobre o mesmo solver de Klein (`OccbinModel`, `occbin_simulate`). Conferidas contra o `+occbin`
  do Dynare com uma e com duas restrições (`borrcon.mod`, `dynrbc.mod`), cujos resultados
  reproduzem até 2e-15, com os mesmos históricos de regime.
- [x] **Trajetória estendida**, determinística e estocástica (árvores esparsa e completa, correção
  híbrida de segunda ordem), em `extended_path`. Conferida contra o `extended_path` do Dynare em
  sete variantes do `burnside.mod`, com diferença de até 1,1e-11. *Falta a correção híbrida de
  ordem maior que 2, que exige a perturbação de ordem 4 (0.7).*
- [x] **Estado estacionário analítico declarado**, inclusive o `steady_state_model` de um `.mod`
  (`analytic_steady_state`, `ModFile.steady_state(analytic=True)`). *O requisito de projeto está
  cumprido, porque* o bloco analítico é conferido contra o resíduo do modelo dinâmico antes de ser
  usado, e recusado se não o anula, como no Dynare.
- [x] **Equações de calibração dentro do modelo**, à maneira do MacroModelling.jl, coincidindo com
  o MacroModelling até 4e-15.
- [x] **Vários agentes num equilíbrio só** (`assemble`). Os limites são os de `Problem`, e o recurso
  não é uma linguagem de blocos.
- [x] **Homotopia e continuação** para estados estacionários difíceis, com passo adaptativo, e o
  `homotopy_setup` de um `.mod`.
- [x] **Modelos não estacionários**. `trend_var`, `log_trend_var`, `var(deflator=)` e
  `var(log_deflator=)` são destendenciados como o Dynare faz, com a verificação do crescimento
  balanceado. O `balanced_growth` resolve o problema do `+bgp`, com resíduos que coincidem com
  os do Dynare até 2,2e-16. *Falta declarar tendências em `from_equations`, e por ora o caminho
  é o `.mod`.*
- [x] **Diagnóstico do modelo** (`diagnose`), o `model_diagnostics` do Dynare, com o mesmo
  relatório.

### 0.5 — lançamento público (lançada em 18 de outubro de 2026)

Nenhum recurso novo. Os itens abaixo são condições para publicar.

**Pré-requisitos**

- [x] **Renomear.** `dsgekit` já estava em uso no PyPI por outro pacote, *"A Python toolkit for
  DSGE models"*. O pacote agora é `dsgelab`, que estava disponível no PyPI em 11 de outubro de
  2026. Falta reservar o nome com um primeiro upload ao TestPyPI e ao PyPI.
- [x] **Um repositório canônico.** `https://github.com/GustavoHM/dsgelab`. As cópias do
  repositório de pesquisa foram arquivadas.
- [x] **Idioma.** O site tem uma versão em inglês na raiz e a versão em português em
  `site/pt-br/`, como o do `cgelab`, e os exemplos rodam nas duas (12 de outubro de 2026). O
  código, a documentação (`docs/`, com o PDF do whitepaper regenerado), o `validation/README.md`, o
  `CITATION.cff` e as mensagens que os scripts de `validation/` imprimem estão em inglês (13 a 16 de
  outubro). O roadmap existe nas duas línguas, com o `ROADMAP.md` em inglês e o `ROADMAP.pt-BR.md`
  em português.
- [x] **Validação reproduzível por terceiros.** A evidência publicada, formada pelos `.mod` do
  Dynare, pelas soluções fechadas e pelo Heutel (2012), pode ser reproduzida por qualquer
  pessoa. Desde 12 de outubro de 2026, o `validation/run_all.sh` roda tudo de uma vez. As
  versões ficam fixadas (pacotes do Octave por sha256 em `setup_octave_packages.sh`,
  MacroModelling.jl em `validation/macromodelling/`, `.mod` em tags do Dynare), e os arquivos
  do Heutel vêm da cópia pública da Georgia State University (`validation/heutel/fetch.sh`),
  idêntica byte a byte nos seis `.m`. Tudo passa no Dynare 7.2, no Octave 11.3.0 e nos pacotes
  statistics 2.0.0 e datatypes 1.5.0. Em 16 de outubro, a única comparação feita num modelo que
  não é público (o modelo aplicado de um artigo inédito do autor, um DSGE de 71 variáveis) foi
  marcada no site como não reproduzível a partir do repositório. O README e o
  `validation/README.md` a marcam da mesma forma.
- [x] **Revisão de segurança antes do lançamento público.**
- [x] **Revisão do autor de todo texto público.** Nada é publicado (o repositório tornado público, o
  site, um release no GitHub, o PyPI, o Zenodo) antes de o autor revisar todos os textos que vão
  junto, a saber o README, o CHANGELOG, o `CONTRIBUTING.md`, o `CODE_OF_CONDUCT.md`, os modelos de
  *issue* e de *pull request*, o `CITATION.cff`, a descrição no `pyproject.toml`, o site nas duas
  línguas, `docs/` e o `validation/README.md`. Os 25 textos foram aprovados em 18 de outubro de
  2026, depois de uma revisão final das suas afirmações contra o código.

**Engenharia de publicação**

- [x] Integração contínua, com testes em Python 3.11 a 3.14, em Linux, macOS e Windows, e
  `check_site_examples.py`. *Escrito* (`tests.yml`). A suíte roda contra a *wheel* instalada em 12
  combinações, o sdist é testado a partir da própria árvore e os exemplos do site rodam. A primeira
  execução no GitHub, em 13 de outubro de 2026, passou nos 17 jobs, inclusive no Windows. O Python
  3.10 deixou de ser suportado em 12 de outubro de 2026 porque o SciPy 1.15.3, a última versão para
  ele, não carrega no macOS 27 (`__thread_bss`).
- [x] Validações externas como tarefa agendada na CI. *Escrito* (`validation.yml`, semanal e sob
  demanda). Roda no macOS, com Octave e Dynare do Homebrew, os pacotes do Octave fixados e o Julia
  1.12.7 com o MacroModelling.jl. No Ubuntu, o Dynare dos repositórios da distribuição está
  defasado em relação à versão corrente, e compilá-lo toda semana somaria o tempo de compilação a
  cada execução. O Dynare do Homebrew acompanha a versão corrente, e por isso a tarefa serve de
  aviso antecipado do risco descrito na seção Riscos. A primeira execução foi disparada à mão em
  16 de outubro de 2026, e as suas 25 etapas passaram em 28 minutos.
- [x] Instalação testada a partir da *wheel*, com 148 testes contra a *wheel* instalada em Python
  3.14 no macOS (`DSGELAB_TEST_INSTALLED=1`), e o sdist com os arquivos que os próprios testes leem.
- [x] `python -m build`, `twine check`, publicação por *trusted publishing*, TestPyPI antes
  (`release.yml`, disparado por uma tag `v*`). O fluxo confere a tag contra a versão, publica no
  TestPyPI, roda a suíte contra o que o TestPyPI serve e só publica no PyPI depois da aprovação do
  revisor obrigatório. A versão 0.5.0 saiu assim em 18 de outubro de 2026.
- [x] Site no GitHub Pages, publicado a partir de `site/` pelo `.github/workflows/pages.yml` a cada
  mudança, com os links para `docs/` apontados para o GitHub (`https://gustavohm.github.io/dsgelab/`).
- [x] DOI no Zenodo a cada release, e `CITATION.cff` com ORCID e DOI. A versão 0.5.0 é
  10.5281/zenodo.23456475, e o DOI conceitual 10.5281/zenodo.23456474, que leva sempre à versão
  mais recente, está no `CITATION.cff`.
- [x] `CONTRIBUTING.md`, código de conduta, modelos de *issue* e de *pull request*.
- [x] `py.typed` e anotações de tipo na API pública, sem nenhuma função ou método público sem
  anotação e sem erro novo acusado pelo `mypy`. Os wrappers de importação preguiçosa (*lazy*) de
  `__init__.py` (`from_equations`, `assemble`, `to_dynare`...) aceitam `*args, **kwargs`, e o
  verificador de tipos não confere os argumentos deles.
- [x] Seção de limitações conhecidas no README.

**Política de versões a partir da 0.5.** As versões seguem o versionamento semântico. Antes da 1.0,
uma versão menor ainda pode mudar a API, mas toda remoção ou mudança incompatível é precedida por
um aviso de descontinuação que dura pelo menos uma versão menor. A API pública é o que está em
`__all__`.

### 0.6 — identificação, momentos e dados

- [x] **Identificação local** (Iskrev; Komunjer e Ng; também Qu e Tkachenko, e a busca de
  colinearidade por força bruta), com `identification` e `ModFile.identification`. Validada
  contra o `+identification` do Dynare em três modelos (`kim2`, `as2007`, `fs2000`), com os
  mesmos vereditos nos três critérios e derivadas a menos de 2e-8 (18 de outubro de 2026).
  Ficam de fora a repetição sobre sorteios da priori (`prior_mc`), a força da identificação
  pela matriz de informação e os gráficos do Dynare.
- [x] **Sensibilidade global**, cobrindo a região de estabilidade e determinação no espaço de
  parâmetros e a filtragem de Monte Carlo. A referência é o `+gsa` do Dynare. Índices de Sobol
  de primeira ordem e totais para respostas a impulso e momentos, validados contra o SALib. As
  distribuições têm a mesma interface que as de `cgelab.sensitivity` (`Uniform`, `Normal`,
  `Triangular`), sem que um pacote importe o outro. `stability_map` separa os sorteios em
  determinados, indeterminados, explosivos e sem solução; `monte_carlo_filtering` (também
  `StabilityMap.filtering` e `RestrictionSample.filtering`) dá o teste de Smirnov de cada
  parâmetro e as correlações dentro de cada grupo; `sobol` e `solution_outputs` dão os índices de
  IRFs e momentos. Validação no `tests/gsa/ls2003a.mod` do Dynare 7.2, em três amostras de 1024
  sorteios (30 de outubro de 2026): cada sorteio tem a classe que o solver do Dynare lhe dá, e,
  nos grupos do Dynare, as estatísticas de Smirnov concordam exatamente, os p-valores até
  `3.3e-16` e os das correlações até `1.7e-13`. As listas da própria caixa de sensibilidade
  diferem em 37 sorteios da amostra explosiva: em 36 porque ela ordena os autovalores pelo valor
  com sinal quando todos são reais, e numa falha de posto que a regra dela, baseada nos
  autovalores, não enxerga. Pelo módulo, a regra dela dá as classes do `dsgelab` em todos os
  sorteios, exceto nas falhas de posto. O desenho e os índices de Sobol concordam com o SALib 1.6
  até `8.9e-16` na função de Ishigami e em IRFs e momentos do mesmo modelo. Ficam de fora o
  mapeamento da forma reduzida (`redform`), a filtragem por RMSE (`rmse`) e o método de Morris.
- [x] **Método dos momentos** (GMM, SMM e ajuste de IRF). GMM e SMM funcionam em primeira e
  segunda ordem com poda; o ajuste de IRF funciona em primeira ordem. Validação contra o `+mom`
  do Dynare 7.2 em oito casos de GMM/SMM em primeira ordem e dois de ajuste de IRF (24 de outubro
  de 2026). As médias e autocovariâncias de segunda ordem concordam em `example1` e `example2`
  até `1e-14`. Os modelos importados, com suas variáveis auxiliares, concordam até `2.93e-8` e
  `1.59e-6` nas médias e até `6.08e-9` e `2.87e-8` nos segundos momentos.
  O ajuste de IRF é frequentista e aceita alvos externos, inclusive projeções locais ou estudos
  de evento, com a sua covariância conjunta. `local_projection` fornece IRFs de um choque observado
  e covariância HAC conjunta; em dados AR(1) simulados, recupera as IRFs do modelo na população
  (Plagborg-Møller e Wolf, 2021). Ficam de fora o ajuste bayesiano de IRF e a execução de arquivos
  de transformação MATLAB; transformações Python podem ser passadas explicitamente.
- [x] **Avaliação de previsão fora da amostra**: estimação recursiva em janela crescente, RMSE
  contra o passeio aleatório, o teste de Diebold–Mariano, e a transformada integral de
  probabilidade e a cobertura da densidade preditiva. As referências são as comparações de
  previsão de Smets e Wouters (2007) e de Del Negro e Schorfheide (2013). A comparação com um
  BVAR espera o BVAR da 0.9. `forecast_evaluation` reestima em cada origem, com janela
  crescente ou deslizante, e `ForecastEvaluation` dá o RMSE, a razão dele para o do passeio
  aleatório, `diebold_mariano` com a correção de Harvey, Leybourne e Newbold, a PIT, a cobertura
  e o escore logarítmico. Validação contra a estimação recursiva do Dynare 7.2 no `fs2000`, em
  nove amostras (30 de outubro de 2026): nas modas do Dynare, as previsões concordam até
  `2.6e-15` e as bandas de 90% até `6.3e-9`. Reestimada pelo `dsgelab`, cada moda tem kernel pelo
  menos tão alto quanto o do Dynare e fica a menos de `0.003` desvio-padrão posterior dela. O
  teste concorda com o `forecast::dm.test` do R até `3.1e-15` em 48 casos. A densidade
  preditiva é condicional à estimativa pontual, como a do Dynare depois da moda; a incerteza dos
  parâmetros em cada origem fica de fora.
- [x] **Restrições de calibração sobre IRFs e momentos** (`irf_calibration`, `moment_calibration`),
  com `check_restrictions` num ponto e `sample_restrictions` em sorteios da priori, e os blocos do
  `.mod` lidos por `from_mod`. Validação contra o Dynare 7.2 no seu `tests/gsa/ls2003a.mod`, com e
  sem `relative_irf` (30 de outubro de 2026): na calibração e nos 512 sorteios da priori da sua
  caixa de ferramentas de sensibilidade, os valores concordam até `1.3e-12`, os mesmos sorteios
  têm solução, e cada restrição tem o mesmo veredito em cada sorteio. Como as do Dynare, as
  restrições funcionam em primeira ordem. A filtragem de Monte Carlo dos parâmetros por trás de
  cada restrição pertence ao item de sensibilidade global, acima.
- [x] **Momentos teóricos filtrados** (Hodrick–Prescott e passa-banda), com `hp_filter` e
  `bandpass` em `moments`, `covariance` e `variance_decomposition`. Validados contra o
  `stoch_simul` do Dynare em cinco modelos, um deles com raiz unitária (24 de outubro de 2026).
- [x] **Dados de frequência mista**. `aggregation={nome: ("mean" | "sum", every)}` em `state_space` e
  em `EstimationProblem` empilha as defasagens do estado, de modo que uma variável observada pode
  ser a média ou a soma de uma variável nos últimos `every` períodos, e `low_frequency` posiciona as
  observações. Filtro, suavizador, gradiente da verossimilhança, decomposições e previsões
  funcionam sobre o estado empilhado. Validação contra o Dynare 7.2 no `fs2000.mod` com o
  crescimento do produto observado só como média anual, escrito no Dynare como a variável auxiliar
  que um usuário acrescentaria (30 de outubro de 2026): a verossimilhança em dois pontos concorda
  até `8.0e-13`, as variáveis suavizadas até `1.3e-14`, e a moda posterior tem kernel pelo menos
  tão alto quanto o do Dynare, a menos de `0.003` desvio-padrão posterior. Ficam de fora os
  agregados de taxas de crescimento de níveis (o crescimento anual de um nível trimestral) e os
  filtros não lineares.
- [x] **Gráficos** de respostas a impulso, simulações, decomposições, previsões e distribuições a
  posteriori: `plot_irf`, `plot_paths`, `plot_decomposition`, `plot_forecast` e `plot_posterior`,
  que devolvem figuras do matplotlib, com o matplotlib como dependência opcional
  (`dsgelab[plot]`). As duas densidades que o gráfico da posteriori desenha são as do Dynare:
  `prior_density` concorda com `draw_prior_density` nas nove prioris do `fs2000` até `1.9e-12`, e
  `posterior_density` com a estimativa de núcleo de `posterior_moments`, com a largura de banda
  corrigida para sorteios repetidos, até `8.9e-16` (30 de outubro de 2026). A suíte de testes
  confere que cada figura desenha os números que recebe. Ficam de fora os gráficos de diagnóstico
  do MCMC e os gráficos de identificação e de sensibilidade do Dynare.

### 0.7 — terceira ordem completa e derivadas

- [x] **Núcleo de perturbação na forma de regra de decisão do Dynare**, com a inovação como
  argumento (`g_u`, `g_uu`, `g_ku`). Os termos de risco na inclinação das regras (`g_ssk` e
  `h_ssk`) já estão implementados. Os blocos de inovação de segunda ordem e a `DecisionRule`
  composta foram validados contra o Dynare 7.2 em `example1` e `example2` (24 de outubro de 2026).
  Os blocos cúbicos `h_uuu`, `h_kuu` e `h_kku`, com suas composições, também concordam com
  `ghuuu` e com todas as linhas de `ghxuu`, `ghxxu` e `ghxxx`. A constante completa de risco de
  segunda ordem também concorda com `ghs2`. `ghxss` e `ghuss` concordam em todas as linhas, com
  erros abaixo de `1,5e-16` nos dois modelos.
- [x] **Estado estacionário estocástico** e IRFs generalizadas analíticas. As IRFs condicionais
  da regra completa com poda estão implementadas em terceira ordem, incluindo `ghxss`, `ghuss` e
  os blocos cúbicos de inovação, com testes unitários da recursão. As IRFs generalizadas usam
  trajetórias estocásticas com `drop=100` e concordam com o Dynare em `example1` e `example2`.
  `stochastic_steady_state` resolve o ponto fixo da regra completa sem poda e se distingue da
  média ergódica.
- [x] **Terceira ordem no tamanho do modelo aplicado**, com tensores esparsos. O frontend
  simbólico fornece derivadas segunda e terceira exatas com `derivatives="third"`, guardadas em coordenadas.
  `solve_third_order(sparse=True)` também guarda os coeficientes cúbicos de estado por triplo
  simétrico independente e os contrai diretamente na avaliação da regra e nos termos de risco.
  O termo conhecido e o residual iterativos usam coordenadas canônicas, com uma equação expandida
  por matvec e caminho de contração otimizado. Uma pipeline sintética não linear completa, com 71
  variáveis, 35 estados e cinco choques, tem residual máximo de `3,05e-20` e pico de 321 MB.
  A composição explícita da regra de decisão ainda expande arrays densos. O modelo aplicado inédito
  em si não é público.
- [x] **Derivadas da solução em relação aos parâmetros**, por diferenciação implícita e numérica, e
  **estimação com gradiente** (HMC/NUTS). `solution_jacobian` calcula derivadas de `P`, `F`, `Q`
  e do estado estacionário. Soluções QZ gerais diferenciam as condições implícitas da saddle path,
  sem outra decomposição QZ nos parâmetros perturbados; soluções puramente backward usam as fórmulas
  do sistema inverso. `second_order_jacobian` também calcula `g_k`, `g_u`, `g_kk`, `g_uu`, `g_ku`
  e `g_s2`, para parâmetros do modelo, desvios-padrão e correlações dos choques.
  `third_order_jacobian` adiciona `g_kkk`, `g_kku`, `g_kuu` e `g_uuu`. Todas são validadas contra
  as derivadas analíticas do Dynare; os erros relativos ficam abaixo de `0.552%` em segunda ordem
  e `0.159%` em terceira. Derivadas exatas do modelo reduzem os erros da regra cúbica abaixo de
  `3e-13` em `example1` e `2e-14` em `example2`. `log_likelihood_gradient` diferencia a
  verossimilhança de Kalman de primeira ordem pela recursão, verificada contra diferenças finitas
  para parâmetros, desvios e correlações dos choques, erros de medição, dados faltantes, início
  difuso, transformações logarítmicas e modelos com saltos. `hmc` usa o gradiente posterior com
  integração leapfrog, matriz de massa baseada na moda e adaptação do passo por média dual durante o
  burn-in. `nuts` adiciona amostragem multinomial na trajetória, o critério generalizado de U-turn
  e diagnósticos de divergência. Na suíte de testes, os dois reproduzem a posterior exata de um
  coeficiente AR(1) e uma cadeia longa de Metropolis–Hastings.
  `posterior_mode` e `maximum_likelihood` usam esses gradientes no BFGS.
- [x] **Filtros não lineares** para estimação em segunda e terceira ordem. O filtro de partículas
  bootstrap funciona em primeira, segunda e terceira ordens, com poda nas ordens não lineares e
  reamostragem sistemática, e reproduz a verossimilhança do Dynare no AR(1) com diferença de
  `0.006` (24 de outubro de 2026). `EstimationProblem(order=2)` expõe a mesma verossimilhança ao
  modo posterior e aos amostradores, com sorteios comuns. O filtro de inversão também funciona em
  primeira e segunda ordem com poda e reproduz exatamente os choques, o Jacobiano e a
  verossimilhança do OccBin no AR(1). `EstimationProblem(nonlinear_filter=)` escolhe a rota. Os
  choques também concordam com o MacroModelling.jl dentro de `0.006` em um RBC não linear; como
  sua convenção de verossimilhança difere, apenas os choques são comparados. O `particle_smoother`
  funciona com `store=True` nas três ordens. Transições de primeira ordem com posto completo usam
  a recursão gaussiana backward; transições não lineares ou de posto deficiente usam a ancestralidade
  armazenada das partículas, cujo tamanho efetivo de amostra pode degenerar em históricos longos.
  O smoothing de inversão retorna as trajetórias do filtro sem passo backward: elas são exatas
  quando os dados identificam todos os choques e filtradas nos demais casos. O filtro de partículas de terceira ordem usa a regra completa,
  incluindo `ghxss` e `ghuss`, e reproduz a verossimilhança de Kalman num modelo linear. `EstimationProblem(order=3)` expõe a verossimilhança e o smoothing.

### 0.8 — métodos globais

- [ ] **Iteração temporal, iteração da função valor, método da grade endógena, grades esparsas.**
  A colocação de Chebyshev com até três estados já existe e reproduz a solução fechada do
  crescimento estocástico. A comparação será com o dolo.
- [ ] **Discretização de processos** (Tauchen, Rouwenhorst).

### 0.9 — a fronteira

- [ ] **Mudança de regime markoviana** na solução e na estimação. Validação contra o RISE.
- [ ] **Agentes heterogêneos** pelo método do jacobiano no espaço de sequências, validados contra
  o sequence-jacobian. *O requisito de projeto, herdado da exclusão anterior, é que* `vec`
  continue sendo a forma de escrever um número finito de grupos, e que uma distribuição seja outro
  algoritmo, num módulo próprio.
- [ ] **BVAR e DSGE-VAR**, a serem conferidos contra o `+bvar` e o `dsge_var` do Dynare.

### 1.0

Tudo o que está na lista acima coberto, a API congelada e, a partir daí, versionamento semântico
estrito.

## Integração com o `cgelab`

O equilíbrio de um período, calibrado numa matriz de contabilidade social, continua no `cgelab`.
O resíduo intertemporal, nos pontos em que as decisões de um agente dependem de expectativas sobre
o futuro, é resolvido aqui. A matriz não entra neste pacote, e a dependência é opcional, porque só
o módulo-ponte do `cgelab` importa o `dsgelab` (`pip install "cgelab[dsge]"`). A ponte já existe
no `cgelab.bridge`, como um EGC de Ramsey em que o investimento segue uma equação de Euler, com
política anunciada e incerteza agregada de primeira ordem. As próximas extensões estão no roadmap
do `cgelab`, na seção *Extensions that call dsgelab*. Os sites dos dois pacotes apontam um para o
outro.

## Riscos

**Escopo.** Vários itens equivalem, cada um, a um pacote inteiro. Agentes heterogêneos são o
escopo do sequence-jacobian, métodos globais o do dolo, e a mudança de regime o do RISE. Com o
padrão de validação deste roadmap, cada um é trabalho de meses. Se algum não avançar, a saída é
publicar primeiro uma ponte validada para a ferramenta que já o faz (como o exportador para o
Dynare faz hoje com a terceira ordem) e manter o item aberto até a implementação própria passar na
mesma validação.

**Dependências.** Estimação, diferenciação automática e métodos globais exigem dependências
grandes. O núcleo continua com numpy e scipy, e o resto entra como extra opcional, para que quem
só resolve modelos não precise instalá-las.

**Validação dependente de ferramentas externas.** Quase toda a evidência vem da comparação com o
Dynare e o MacroModelling.jl. Se uma dessas ferramentas mudar de comportamento entre versões, as
validações quebram sem que o pacote tenha mudado. Por isso os `.mod` vêm de tags fixas do Dynare
(6.4 para os exemplos e 7.2 para os testes, com os arquivos usados idênticos aos da 7.1), e a
versão usada fica registrada nos scripts.
