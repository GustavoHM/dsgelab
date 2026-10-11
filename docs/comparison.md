# Como estes pacotes se comparam aos que já existem

Levantamento feito em 9 de setembro de 2026 e revisto em 11 de outubro de 2026, depois de o
`dsgelab` ganhar o planejador de Ramsey, a leitura de `.mod`, o impulso generalizado de segunda
ordem, a decomposição histórica e a previsão condicional. As linhas que mudaram em 11 de outubro
estão marcadas com ‡; as de 9 de setembro, com †. A seção [Lacunas, também contra ferramentas de
outras linguagens](#lacunas-também-contra-ferramentas-de-outras-linguagens) amplia a régua para o
Dynare e o MacroModelling.jl, e cada lacuna ali tem uma versão no [`ROADMAP.md`](../ROADMAP.md).

A pergunta que originou o levantamento — *já existia pacote para DSGE e CGE em Python?* — tem
resposta diferente dos dois lados, e vale registrar por escrito em vez de deixar implícito. A
tabela do lado CGE está no repositório do `cgelab`.

## Como ler as tabelas

| Marca | Significa |
|---|---|
| **✅** | verificado por mim, instalando o pacote e inspecionando a API ou o código |
| **📄** | consta da documentação oficial; **não** verifiquei no código |
| **—** | ausente até onde apurei |
| **?** | não apurado |

A distinção não é decorativa. Instalei e inspecionei `gEconpy` e `sequence-jacobian`; clonei e li o
código de `pycge`, `PSLmodels/CGE` e `python-cge`; para `dolo`, `pydsge`, `econpizza` e `PyMacLab`
li a documentação. Uma célula **📄** pode estar errada, e prefiro dizê-lo a fingir precisão.

---

# DSGE

O campo é povoado. Um resumo antes da tabela: se eu estivesse começando do zero hoje e precisasse
apenas de primeira ordem com estimação bayesiana, instalaria o **gEconpy**; se precisasse de métodos
globais ou terceira ordem, o **dolo**; de agentes heterogêneos, o **sequence-jacobian** ou o
**econpizza**. O `dsgelab` não cobre nenhum desses três casos melhor do que eles.

| Capacidade | dsgelab | gEconpy 2.2 | dolo | pydsge | econpizza | PyMacLab | Dynare (ref.) |
|---|---|---|---|---|---|---|---|
| **Declaração do modelo** | função de resíduos, equações em texto, ou problema de otimização † | ✅ arquivo `.GCN` por blocos de otimização | 📄 arquivo YAML | 📄 formato do dolo | 📄 YAML | 📄 arquivo de modelo próprio | `.mod` |
| **Deriva as CPOs do problema do agente** | ✅ † lagrangiano declarado | ✅ | — | — | — | — | — |
| Estado estacionário numérico | ✅ | ✅ | 📄 | 📄 | 📄 | 📄 | ✅ |
| Estado estacionário analítico declarado | — | ✅ bloco `STEADY_STATE` | ? | ? | ? | 📄 | ✅ |
| **Perturbação de 1ª ordem** | ✅ Klein/QZ | ✅ | 📄 | 📄 | — | 📄 Klein | ✅ |
| **Perturbação de 2ª ordem + correções de risco** | ✅ | **—** | 📄 | — | — | 📄 Klein–Gomme | ✅ |
| Perturbação de 3ª ordem | ✅ † cúbico e inclinação de risco | — | 📄 | — | — | — | ✅ |
| Métodos globais / projeção | ✅ † colocação de Chebyshev | — | 📄 iteração de tempo, valor, expectativas | — | 📄 | — | — |
| Agentes heterogêneos (HANK) | — | — | — | — | 📄 | — | 📄 |
| Restrições ocasionalmente ativas | — | — | 📄 | 📄 ZLB | 📄 | — | 📄 OccBin |
| Raiz unitária aceita (`qz_criterium`) | ✅ ‡ opcional; estrito por omissão | ? | ? | ? | ? | ? | ✅ por omissão |
| **Previsão perfeita não linear** | ✅ Newton de tempo empilhado | ✅ `solve_perfect_foresight` | 📄 | ? | 📄 *shooting* | ? | ✅ |
| Trajetória ANUNCIADA de parâmetros | ✅ | ✅ `param_paths` | ? | ? | ? | ? | 📄 |
| Trajetória anunciada de INOVAÇÕES | ✅ ‡ | ? | ? | ? | ? | ? | ✅ |
| IRFs | ✅ | ✅ | ? | 📄 | 📄 | 📄 | ✅ |
| IRFs ortogonalizadas (Cholesky) | ✅ | ✅ `orthogonalize_shocks` | ? | ? | ? | ? | ✅ |
| Momentos por Lyapunov | ✅ | ✅ `stationary_covariance_matrix` | ? | ? | — | 📄 | ✅ |
| Autocorrelações | ✅ | ✅ `autocorrelation_matrix` | ? | ? | — | 📄 | ✅ |
| Decomposição de variância | ✅ | ? | ? | ? | — | ? | ✅ |
| Simulação com poda (*pruning*) | ✅ | — (só 1ª ordem) | ? | ? | — | ? | ✅ |
| Impulso generalizado de 2ª ordem | ✅ ‡ | — | ? | ? | — | ? | ✅ |
| Decomposição histórica e previsão condicional | ✅ ‡ na solução linear | ? | — | ? | — | — | ✅ |
| **Política ótima: Ramsey com compromisso** | ✅ ‡ atemporal, contra o Dynare a 5e-13 | — | — | — | — | — | ✅ |
| Regras simples ótimas, discrição | ✅ ‡ contra o Dynare | — | — | — | — | — | ✅ |
| Bem-estar de 2ª ordem, condicional e incondicional | ✅ ‡ contra o Dynare a 1e-12 | — | — | — | — | — | ✅ |
| **Estimação bayesiana** | ✅ † Kalman e Metropolis | ✅ via PyMC | — | ✅ com ZLB | — | — 📄 declara não ter | ✅ |
| Filtro de Kalman / *smoothing* | ✅ † filtro, sem suavizador | ✅ `statespace_from_gcn` | — | 📄 | — | — | ✅ |
| **Calibração por alvos** | ✅ conjunta, com limites | ✅ parâmetros calibrados por razões-alvo | ? | ? | ? | ? | 📄 |
| **Diagnóstico de identificação da calibração** | ✅ posto e direções nulas | ? | ? | ? | ? | ? | 📄 (identificação do modelo) |
| Blanchard-Kahn | ✅ **imposto**, com margem | ✅ `check_bk_condition` | ? | ? | — | ? | ✅ relatado |
| **Ordem de acurácia autoverificada** | ✅ ordens 2, 3 e 4 | ? | ? | ? | ? | ? | — |
| Exportar `.mod` do Dynare | ✅ † ida e volta a 5e-14 | ✅ `make_mod_file` | 📄 | — | — | — | — |
| **Ler `.mod` do Dynare** | ✅ ‡ oito exemplos do Dynare lidos sem porte | — | — | — | — | — | n/a |
| Dependências | numpy, scipy (sympy opcional) | sympy, pytensor, PyMC, preliz | numpy, scipy, pandas (pin antigo) | dolo, numpy | **JAX** | numpy, scipy | — |
| Instala hoje (Python 3.14) | ✅ | ✅ | ❌ falha: fixa um pandas que não compila | ? | ? | ❌ o sdist não tem `setup.py` nem `pyproject.toml` | n/a |
| Licença | MIT | GPL-3.0 | BSD-2 | ? | ? | ? | GPL |

### O que a tabela diz

**O `gEconpy` é o vizinho mais próximo, e é maior em quase tudo.** Declaração por blocos com derivação
automática das condições de primeira ordem, estado estacionário analítico, estimação bayesiana,
exportação para Dynare. Em três células o `dsgelab` tem o que ele não tem: **segunda ordem**, que o
gEconpy simplesmente não faz; a **decomposição de variância**; e a **autoverificação da ordem de
acurácia**. Se o seu trabalho não precisa de segunda ordem, o gEconpy é a escolha melhor.

**O `PyMacLab` é o que mais se parece com o `dsgelab` metodologicamente** — Klein de 1ª ordem e
Klein–Gomme de 2ª — e é o único que declara não ter estimação. Também é o único que **não instala**:
o pacote de 0.95.9 no PyPI não traz `setup.py` nem `pyproject.toml`, e a documentação data de janeiro
de 2013.

**O `dolo` é maior em método** (até 3ª ordem, métodos globais, iteração de tempo e de valor) e não
instalou aqui por fixar uma versão de pandas que já não compila; deve funcionar em Python 3.11 ou
anterior, mas não confirmei.

**O `econpizza` e o `sequence-jacobian` resolvem outro problema** — agentes heterogêneos —, e não
concorrem com o `dsgelab` em modelo de agente representativo.

**Um concorrente direto escapou ao primeiro levantamento.** Há no PyPI um pacote chamado `dsgekit`,
*"A Python toolkit for DSGE models"*, de Gustavo Vargas (0.1.0rc1). Não foi instalado nem lido, e
por isso não está na tabela. Foi por causa dele que este pacote, que se chamava `dsgekit`, passou
a se chamar `dsgelab` em 11 de outubro de 2026.

### O que mudou, e o que continua faltando

Três das cinco lacunas que a versão anterior desta tabela mostrava foram fechadas: **derivação de
CPOs** a partir de um lagrangiano declarado, **terceira ordem** e **exportadores** para Dynare e
MacroModelling.jl — este último com ida e volta verificada, isto é, o pacote escreve o `.mod`, o
Dynare resolve, e as duas soluções batem a 5e-14.

As três lacunas que essa tabela deixava em aberto foram fechadas no que cada uma tem de verificável,
e o alcance de cada fechamento é o do teste, não o de uma plataforma.

* **Terceira ordem, incluindo a inclinação de risco.** ``g_kkk`` e ``h_kkk`` continuam verificados
  pela ordem de acurácia. ``h_ssk`` e ``g_ssk`` saem da curvatura da regra em relação à inovação,
  contraída com ``Σ``. No AR(1) logarítmico, ``h_ssk[a, a] = ρ σ²`` com o sinal positivo, que é o
  termo cruzado de ``(1+â)^ρ exp(σ²/2)``. A linha do capital, cuja regra é exata, sai em zero.
  O tensor completo do Dynare, com a inovação como argumento em todos os monômios, não é este
  objeto. O termo ``σ² k`` da regra de decisão é.
* **Estimação bayesiana do modelo linear.** Filtro de Kalman sobre a solução de primeira ordem,
  moda da posteriori e Metropolis–Hastings de passeio aleatório. No AR(1) gaussiano a verossimilhança
  bate com a fórmula condicional, e a moda de ``ρ`` numa amostra simulada volta ao valor que a gerou.
  Não há filtro de partículas, nem suavizador, nem estimação da solução podada.
* **Projeção por colocação de Chebyshev.** O mesmo resíduo, numa grade tensorial de até três estados.
  As linhas sem salto futuro determinam o estado seguinte a cada choque. As outras são integradas por
  Gauss–Hermite. No crescimento estocástico de solução fechada, a política reproduz
  ``c = (1-αβ) a k^α``. Uma grade esparsa, iteração de valor e mais de dois choques na quadratura
  tensorial não estão neste solver.

### Avaliação honesta

O `dsgelab` não é primeiro de tipo nem o mais completo. O nicho defensável é estreito e específico:

0. **Um caminho contínuo do problema do agente ao `.mod`.** Declara-se o lagrangiano, derivam-se as
   CPOs, resolve-se, e exporta-se para o Dynare ou para o MacroModelling — com a datação traduzida
   automaticamente, que é onde um porte manual erra e onde a ida e volta prova que não errou.
1. **A mesma função de resíduos serve à perturbação e à transição determinística.** Não afirmo que os
   outros não façam isso — não auditei os seus internos. Afirmo que era o problema concreto do
   repositório que o motivou, em que a perturbação e a transição repousavam sobre equações
   diferentes.
2. **Segunda ordem em processo**, onde o vizinho mais próximo não tem nenhuma e a alternativa era
   sair para Octave e Dynare.
3. **As verificações são da biblioteca**: Blanchard-Kahn imposto e não relatado, ordem de acurácia
   medida, identidades que o modelo não impõe tratadas como verificação, identificação da calibração
   diagnosticada.
4. **Duas dependências.** numpy e scipy. Nem sympy, nem PyMC, nem JAX, nem pytensor.

Se algum dia a segunda ordem deixar de ser necessária, migrar para o `gEconpy` é o caminho, e as
verificações do item 3 são finas o bastante para virar uma camada por cima dele. Para estimação
bayesiana, hoje, é o `gEconpy` ou o `pydsge`; para métodos globais, o `dolo`; para risco variável no
tempo, o Dynare pela porta do exportador.

---

# Lacunas, também contra ferramentas de outras linguagens

A tabela acima compara com pacotes em Python. Esta seção amplia a régua para o que existe em
qualquer linguagem. A régua foi conferida nas versões instaladas nesta máquina: os subsistemas e os
arquivos de função do Dynare 7.1 (`lib/dynare/matlab/`) e a lista de funções exportadas do
MacroModelling.jl. Desde 11 de outubro de 2026 o objetivo é cobrir **toda** esta lista até a 1.0;
a versão de cada item é a do [`ROADMAP.md`](../ROADMAP.md).

| Item | Quem tem | Estado no `dsgelab` | Versão |
|---|---|---|---|
| Ramsey (compromisso) | ✅ Dynare (`ramsey_model`) | **feito**: `ramsey_model`, contra o Dynare em três modelos | 0.2 |
| Regras simples ótimas; discrição | ✅ Dynare (`+osr`, `discretionary_policy`) | **feito**: `osr`, `discretionary_policy`, contra os testes do Dynare | 0.2 |
| Bem-estar de segunda ordem e variação equivalente | ✅ Dynare (`evaluate_planner_objective`) | **feito**: `welfare`, `consumption_equivalent` | 0.2 |
| Impulso generalizado de segunda ordem | ✅ MacroModelling (`get_girf`) | **feito** | 0.2 |
| Inovações em sequência na previsão perfeita | ✅ Dynare (`perfect_foresight_solver`) | **feito**; falta a validação contra o Dynare | 0.2 |
| Erros de expectativa na previsão perfeita | ✅ Dynare (`perfect_foresight_with_expectation_errors_solver`) | falta | 0.2 |
| Kalman, moda, Metropolis | ✅ Dynare (`+estimate`), MacroModelling, gEconpy | **feito** no modelo linear | 0.3 |
| Suavizador, máxima verossimilhança, SMC, diagnósticos | ✅ Dynare | falta | 0.3 |
| Verossimilhança marginal e comparação de modelos | ✅ Dynare | falta | 0.3 |
| Decomposição histórica; previsão condicional | ✅ Dynare (`shock_decomposition`, `+conditional_forecasts`), MacroModelling | **feito** na solução linear; falta a versão em tempo real e a validação | 0.3 |
| **Ler `.mod`** | ✅ MacroModelling (`translate_mod_file`) | **feito**: `from_mod`, oito `.mod` do Dynare | 0.4 |
| Defasagens e avanços arbitrários | 📄 Dynare, 📄 MacroModelling | **feito** pelo caminho do `.mod`; falta em `from_equations` | 0.4 |
| Restrições ocasionalmente ativas | ✅ Dynare (`+occbin`) | falta; `check_slack` só diagnostica | 0.4 |
| Trajetória estendida | ✅ Dynare (`extended_path`) | falta | 0.4 |
| Estado estacionário analítico declarado | 📄 Dynare (`steady_state_model`), ✅ gEconpy | falta como objeto; `from_mod` usa o bloco como ponto de partida | 0.4 |
| Equações de calibração no modelo | ✅ MacroModelling | falta; `calibrate` faz o mesmo de fora | 0.4 |
| Homotopia; crescimento balanceado; diagnóstico do modelo | ✅ Dynare | falta; a raiz unitária já é aceita | 0.4 |
| Identificação local; sensibilidade global | ✅ Dynare (`+identification`, `+gsa`) | falta; `target_jacobian` cobre alvos de estado estacionário | 0.6 |
| Método dos momentos; restrições sobre IRFs e momentos | ✅ Dynare (`+mom`, `irf_calibration`) | falta | 0.6 |
| Momentos filtrados; frequência mista; gráficos | ✅ Dynare, gEconpy, MacroModelling | falta | 0.6 |
| Terceira ordem na forma de regra do Dynare; estado estacionário estocástico | ✅ Dynare, MacroModelling | inclinação de risco feita; o resto falta | 0.7 |
| Terceira ordem esparsa; derivadas em relação aos parâmetros; filtros não lineares | 📄 Dynare (k-order), ✅ MacroModelling | falta | 0.7 |
| Métodos globais | 📄 dolo | colocação de Chebyshev até três estados; o resto falta | 0.8 |
| Troca de regime; agentes heterogêneos; BVAR e DSGE-VAR | 📄 RISE, ✅ Dynare, sequence-jacobian | falta | 0.9 |

---

## Fontes

- [gEconpy](https://github.com/jessegrabowski/gEconpy) — instalado (2.2.0) e inspecionado
- [sequence-jacobian](https://github.com/shade-econ/sequence-jacobian) — instalado e inspecionado
- [dolo](https://github.com/EconForge/dolo.py) e [a sua documentação](https://www.econforge.org/dolo.py/)
- [pydsge](https://github.com/gboehl/pydsge) e [os seus docs](https://pydsge.readthedocs.io/en/latest/index.html)
- [econpizza](https://github.com/gboehl/econpizza)
- [PyMacLab](https://pythonhosted.org/pymaclab/) — documentação de janeiro de 2013
- [Dynare](https://www.dynare.org/) 7.1, instalado; os `.mod` de validação vêm das tags 6.4 e 7.1
- [MacroModelling.jl](https://github.com/thorek1/MacroModelling.jl), instalado
