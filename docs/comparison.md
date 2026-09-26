# Como estes pacotes se comparam aos que já existem

Levantamento feito em 9 de setembro de 2026 e **revisto no mesmo dia**, depois de o
`dsgekit` ganhar derivação de CPOs, terceira ordem determinística e exportadores para
Dynare e MacroModelling.jl. As linhas que mudaram estão marcadas com †. A pergunta que o originou — *já existia pacote para
DSGE e CGE em Python?* — tem resposta diferente dos dois lados, e vale registrar por escrito em vez
de deixar implícito.

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
**econpizza**. O `dsgekit` não cobre nenhum desses três casos melhor do que eles.

| Capacidade | dsgekit | gEconpy 2.2 | dolo | pydsge | econpizza | PyMacLab | Dynare (ref.) |
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
| **Previsão perfeita não linear** | ✅ Newton de tempo empilhado | ✅ `solve_perfect_foresight` | 📄 | ? | 📄 *shooting* | ? | ✅ |
| Trajetória ANUNCIADA de parâmetros | ✅ | ✅ `param_paths` | ? | ? | ? | ? | 📄 |
| IRFs | ✅ | ✅ | ? | 📄 | 📄 | 📄 | ✅ |
| IRFs ortogonalizadas (Cholesky) | ✅ | ✅ `orthogonalize_shocks` | ? | ? | ? | ? | ✅ |
| Momentos por Lyapunov | ✅ | ✅ `stationary_covariance_matrix` | ? | ? | — | 📄 | ✅ |
| Autocorrelações | ✅ | ✅ `autocorrelation_matrix` | ? | ? | — | 📄 | ✅ |
| Decomposição de variância | ✅ | ? | ? | ? | — | ? | ✅ |
| Simulação com poda (*pruning*) | ✅ | — (só 1ª ordem) | ? | ? | — | ? | ✅ |
| **Estimação bayesiana** | ✅ † Kalman e Metropolis | ✅ via PyMC | — | ✅ com ZLB | — | — 📄 declara não ter | ✅ |
| Filtro de Kalman / *smoothing* | ✅ † filtro, sem suavizador | ✅ `statespace_from_gcn` | — | 📄 | — | — | ✅ |
| **Calibração por alvos** | ✅ conjunta, com limites | ✅ parâmetros calibrados por razões-alvo | ? | ? | ? | ? | 📄 |
| **Diagnóstico de identificação da calibração** | ✅ posto e direções nulas | ? | ? | ? | ? | ? | 📄 (identificação do modelo) |
| Blanchard-Kahn | ✅ **imposto**, com margem | ✅ `check_bk_condition` | ? | ? | — | ? | ✅ relatado |
| **Ordem de acurácia autoverificada** | ✅ ordens 2, 3 e 4 | ? | ? | ? | ? | ? | — |
| Exportar `.mod` do Dynare | ✅ † ida e volta a 5e-14 | ✅ `make_mod_file` | 📄 | — | — | — | — |
| Dependências | numpy, scipy (sympy opcional) | sympy, pytensor, PyMC, preliz | numpy, scipy, pandas (pin antigo) | dolo, numpy | **JAX** | numpy, scipy | — |
| Instala hoje (Python 3.14) | ✅ | ✅ | ❌ falha: fixa um pandas que não compila | ? | ? | ❌ o sdist não tem `setup.py` nem `pyproject.toml` | n/a |
| Licença | MIT | GPL-3.0 | BSD-2 | ? | ? | ? | GPL |

### O que a tabela diz

**O `gEconpy` é o vizinho mais próximo, e é maior em quase tudo.** Declaração por blocos com derivação
automática das condições de primeira ordem, estado estacionário analítico, estimação bayesiana,
exportação para Dynare. Em três células o `dsgekit` tem o que ele não tem: **segunda ordem**, que o
gEconpy simplesmente não faz; a **decomposição de variância**; e a **autoverificação da ordem de
acurácia**. Se o seu trabalho não precisa de segunda ordem, o gEconpy é a escolha melhor.

**O `PyMacLab` é o que mais se parece com o `dsgekit` metodologicamente** — Klein de 1ª ordem e
Klein–Gomme de 2ª — e é o único que declara não ter estimação. Também é o único que **não instala**:
o pacote de 0.95.9 no PyPI não traz `setup.py` nem `pyproject.toml`, e a documentação data de janeiro
de 2013.

**O `dolo` é maior em método** (até 3ª ordem, métodos globais, iteração de tempo e de valor) e não
instalou aqui por fixar uma versão de pandas que já não compila; deve funcionar em Python 3.11 ou
anterior, mas não confirmei.

**O `econpizza` e o `sequence-jacobian` resolvem outro problema** — agentes heterogêneos —, e não
concorrem com o `dsgekit` em modelo de agente representativo.

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

O `dsgekit` não é primeiro de tipo nem o mais completo. O nicho defensável é estreito e específico:

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

# CGE

Aqui a situação é outra, e a diferença não é de opinião: é de escala.

| Capacidade | cgekit | pycge | PSLmodels/CGE | python-cge | GAMS / GEMPACK (ref.) |
|---|---|---|---|---|---|
| **Origem** | escrito para uma MCS aplicada | ✅ porte do `splcge.gams` da biblioteca do GAMS | ✅ "o modelo CGE mais simples de Hosoe, Gawana e Hashimoto (2010)" | ✅ transcrição direta do livro de Hosoe, p. 72 | plataformas de propósito geral |
| **Escala demonstrada** | ✅ 225 contas: 69 atividades, 131 mercadorias, 20 domicílios | ✅ 3 exemplos didáticos + um regional | ✅ MCS de **16×18**: 4 setores, 3 fatores | ✅ **2 bens** (`BRD`, `MLK`) | milhares de contas (GTAP) |
| Backend | ✅ `scipy.optimize.root` sobre poucas incógnitas | ✅ Pyomo + IPOPT | ✅ `scipy` `root` | ✅ próprio | resolvedores dedicados |
| Linhas de Python | ~2.100 | ✅ 3.267 | ✅ 2.660 | ✅ 533 | n/a |
| **MCS: leitura, verificação de balanço, balanceamento** | ✅ longo/quadrado, porta dura, RAS que preserva sinais | ? | ✅ lê `SAM.xlsx` | ✅ tem `balance.py` | ✅ |
| Ninho de valor adicionado configurável (CES / CD / Leontief) | ✅ | ? | ✅ CES e Leontief | — | ✅ |
| Armington | ✅ | ✅ | ✅ | — | ✅ |
| CET nas exportações | ✅ (variante; o padrão fixa participações) | ? | ✅ | — | ✅ |
| **Bloco de margens de comércio e transporte** | ✅ | ? | — | — | ✅ |
| **Valor adicionado negativo tratado** | ✅ duas regras | ? | — | — | — |
| Demanda LES / Stone-Geary | ✅ com calibração por Frisch | ? | — | — | ✅ |
| **Atividades de conta própria** | ✅ detectadas e escaláveis | — | — | — | — |
| Split creditável/não creditável do tributo | ✅ | — | — | — | ✅ |
| **Fechamentos alternativos** | ✅ 5 combinações, cada uma com a sua variável de ajuste | ? | — | — | ✅ |
| **Walras como verificação e não equação** | ✅ mercado do numerário fora do sistema | ? | — | — | convenção do modelador |
| **Teste de reprodução do ano-base** | ✅ automático | 📄 "compara base e simulação" | — | — | convenção do modelador |
| **Teste de homogeneidade de grau zero** | ✅ automático | — | — | — | convenção do modelador |
| **Dinâmica recursiva** | ✅ com acumulação de capital e duas convenções de deflator | — | ✅ tem laço de iteração temporal | — | ✅ |
| **Bem-estar: VE e VC** | ✅ | — | — | — | convenção do modelador |
| Agregação por grupo imune à composição | ✅ | — | — | — | — |
| Licença | MIT | ? | CC0-1.0 | ? | proprietário/comercial |
| Estado declarado pelos autores | — | — | ✅ "muito preliminar… por favor não cite" | livro-texto | maduro |

### O que a tabela diz

Os três pacotes de EGC em Python que encontrei são **implementações do modelo-texto de Hosoe, Gawana
e Hashimoto (2010)** ou do `splcge` da biblioteca do GAMS — dois a quatro setores. O `PSLmodels/CGE`
diz de si mesmo, no README, que é "muito preliminar" e pede que não seja citado. Não há, até onde
apurei, um pacote de EGC em Python de escala aplicada.

Isso não faz do `cgekit` uma contribuição: faz dele **o preenchimento de uma lacuna que existe porque
quem faz EGC aplicado usa GAMS ou GEMPACK**, e com bons motivos — resolvedores dedicados,
comunidade, bases como a GTAP. A afirmação defensável é estreita: se você quer um EGC de escala
aplicada *em Python*, sem licença comercial e com as portarias contábeis embutidas, não achei
alternativa.

### Uma advertência

O `cgekit` foi validado contra uma réplica independente em JuMP/Ipopt do mesmo equilíbrio e contra o
pipeline que o originou, e reproduz o ano-base, a homogeneidade e Walras à precisão de máquina numa
MCS de 225 contas. Isso é evidência de que o que ele faz, faz certo. **Não** é evidência de que
cubra o que um GAMS cobre: não tem CET calibrado em série, nem múltiplas regiões, nem estimação de
elasticidades, nem a maturidade de uma plataforma com trinta anos de uso.

---

## Fontes

- [gEconpy](https://github.com/jessegrabowski/gEconpy) — instalado (2.2.0) e inspecionado
- [sequence-jacobian](https://github.com/shade-econ/sequence-jacobian) — instalado e inspecionado
- [dolo](https://github.com/EconForge/dolo.py) e [a sua documentação](https://www.econforge.org/dolo.py/)
- [pydsge](https://github.com/gboehl/pydsge) e [os seus docs](https://pydsge.readthedocs.io/en/latest/index.html)
- [econpizza](https://github.com/gboehl/econpizza)
- [PyMacLab](https://pythonhosted.org/pymaclab/) — documentação de janeiro de 2013
- [pycge](https://github.com/juanfung/pycge), [PSLmodels/CGE](https://github.com/PSLmodels/CGE),
  [python-cge](https://github.com/DavoudTaghawiNejad/python-cge) — clonados e lidos
- [Banco Mundial sobre EGC](https://www.worldbank.org/en/programs/economicpolicy-macro-modeling/cge) e
  [GEMPACK](https://en.wikipedia.org/wiki/Gempack)
