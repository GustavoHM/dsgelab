# Validation against Dynare

`dsgelab` is checked in three layers, and this folder holds the third.

| Layer | Against what | Where |
|---|---|---|
| Answers known on paper | closed-form RBC, Taylor principle, lognormal risk correction | `../tests/` |
| The model that motivated the package | `e50` of the paper, plus Dynare and MacroModelling.jl | `artigo-dinamico-gd/code/e71`, `e72` |
| **Third-party models** | **Dynare's own examples** | **this folder** |

The first two layers have the same author as the package. In this one the models come from
[dynare.org/resources](https://www.dynare.org/resources/) (the examples of Collard's tutorial and
Johannes Pfeifer's collection), and the reference solver is Dynare.

## Running everything

From the repository root:

```bash
bash validation/setup_octave_packages.sh             # Octave packages, at pinned versions
bash validation/run_all.sh                           # every comparison below
SKIP_JULIA=1 SKIP_MH=1 bash validation/run_all.sh    # without Julia and without the Metropolis
```

`run_all.sh` runs the whole external validation end to end. It downloads the `.mod` files, runs
Dynare and MacroModelling.jl, compares each result with `dsgelab`, and finishes with the Heutel
(2012) port. A step that fails does not stop the steps after it. The summary at the end lists the
failed steps, and the exit status is non-zero if any step failed. `SKIP_JULIA=1` skips the
MacroModelling.jl steps, and `SKIP_MH=1` skips the Metropolis–Hastings part of the estimation
comparison, which takes a few minutes. The script needs GNU Octave with the `statistics` and
`datatypes` packages, Dynare (`OCTAVE` and `DYNARE_MATLAB` override the search in the usual
places) and, for the calibration equations, Julia. The `compare_*.py` scripts run under the Python
on the `PATH`, or under `PYTHON`, which must have `dsgelab` installed.

`setup_octave_packages.sh` installs `statistics` 2.0.0 and `datatypes` 1.5.0, the versions with
which the published validation was run. Since `pkg install -forge` would fetch the newest release,
the script takes the versions, URLs and SHA-256 digests from the official index
(github.com/gnu-octave/packages) and refuses a file whose digest does not match.

The numbers in this file come from a run on Dynare 7.2 and Octave 11.3.0 on 12 October 2026, and
every comparison also passed on Dynare 7.1 with the same printed numbers apart from the Heutel
errors, which changed in the last digit.

`macromodelling/` is a Julia project. The `[compat]` section of its `Project.toml` pins
MacroModelling 0.2.1 and JSON3 1.14.3 on Julia 1.12, and its `CondaPkg.toml` restricts the Python
that PythonCall runs on to versions from 3.12 up to and excluding 3.15. `Manifest.toml` is listed
in `.gitignore` and is not in the repository, so `run_all.sh` resolves the remaining dependencies
when it instantiates the project before running `calibration.jl`.

`heutel/fetch.sh` downloads Heutel's MATLAB files from the public copy in Georgia State
University's institutional repository (scholarworks.gsu.edu), checks the SHA-256 of the archive
and unpacks it into `heutel/original/`. The files remain attributed to Garth Heutel and are not
versioned here. The port and its validation are described in [`../docs/heutel.md`](../docs/heutel.md).

## How to run

```bash
bash fetch.sh          # baixa os .mod (não versionados: são GPL, o pacote é MIT)
bash run_dynare.sh     # roda cada um no Octave + Dynare, exporta out/<nome>.mat
python compare.py      # confronta o dsgelab com o Dynare
```

This needs GNU Octave and Dynare. `run_dynare.sh` looks for Dynare's `matlab/` directory in the
usual places, and `DYNARE_MATLAB=/caminho` sets it explicitly.

## Result

```
  example1       e.e. 1.2e-14   IRFs 2.9e-14   momentos 2.6e-14   OK
  example2       e.e. 2.1e-05   IRFs 4.3e-06   momentos 3.3e-06   OK  (limitado pelo e.e. do Dynare, resíduo 1e-06)
  rbc_baseline   e.e. 0.0e+00   IRFs 2.3e-14   momentos 3.5e-14   OK
```

## The models

| File | What it exercises |
|---|---|
| `example1.mod` | Collard (2001). **Correlated** shocks and two exogenous processes that feed into each other, so that the transition of the states is not diagonal. |
| `example2.mod` | The same model in **logarithms**. The real allocation has to be the same, so the model tests the package's choice of coordinates, which carries proportional deviations for variables with a non-zero steady state and absolute deviations for the rest. |
| `RBC_baseline.mod` | Pfeifer, calibrated to US data. Fifteen variables, six of them **purely static** logarithmic identities, which give zero rows in `A` and a singular QZ pencil. Any implementation that inverts `A` fails on this model. |

## What the comparison measures, and why in that order

1. **Steady state**, variable by variable.
2. **Impulse responses**, all variables and all shocks, 40 periods.
3. **Theoretical standard deviations**, which test what the first two leave out. An IRF computed
   from an impulse to the state does not pass through the shock loading `Q`, whereas the moments
   do. A wrong `Q` survives the first two tests and fails the third.

Each `.mod` carries its own `stoch_simul` options (one applies the HP filter, another simulates 2000
periods). `run_dynare.sh` reruns `stoch_simul` with the same settings for all of them (order 1, 40
periods of IRFs, theoretical moments, no filter), so that differences in reporting options do not
enter the comparison of the models.

## Two conventions the comparison has to reconcile

**Dating of the stock.** Dynare's `k` is the END-of-period stock. In `dsgelab`, `k` is the stock
AVAILABLE in the period, which is Dynare's `k(-1)`. The two series differ by one period. The
comparison shifts exactly those predetermined variables whose row of `Q` is zero, which are the
ones that receive no innovation directly. Swapping the two datings gives a model that solves,
satisfies Blanchard-Kahn and responds to shocks, yet is a different model, and only the dynamic
comparison reveals the difference.

**Orthogonalisation.** Dynare reports IRFs to ORTHOGONALISED impulses, in which the response to
shock *i* is the response to column *i* of the Cholesky factor of the covariance, and not to an
impulse in *i* alone. With a diagonal covariance the two coincide. With the
`var e, u = phi*0.009^2` of `example1` they do not. `dsgelab` provides `irf_impulse` for this case.

## What the validation found

**In `example2`, Dynare's `steady` stops early.** Dynare's steady state leaves a residual of
`1.05e-06` in the model equations, and `dsgelab`'s leaves `1.0e-44`. Everything derived from the
steady state (linearisation, IRFs, moments) inherits that error, which accounts for the whole of
the `3e-06` discrepancy in the table. When two programs disagree, the number that should be trusted
is the one computed at the POINT that sets the equations to zero. `compare.py` therefore measures
the residual at both points and forms its verdict from them.

An independent check is available because `example1` and `example2` are the same economy in
different coordinates. Exponentiated, the `example2` steady state matches the levels of `example1`
to `1e-15`.

**An error in my port, which only the dynamics revealed.** In porting `RBC_baseline` I wrote
`log_k = log(k)` where the `.mod` says `log_k = log(k)` with Dynare's CURRENT `k` in mind, which is
the end-of-period stock and corresponds to `k(+1)` in the package's convention. The steady state
matched exactly, while the IRF of `log_k` lagged by one period. This case is the reason the
comparison covers trajectories and not only points.

## Optimal policy: the Ramsey planner

```bash
bash fetch.sh            # inclui mods/ramsey/, do conjunto de testes do Dynare 7.2
bash run_ramsey.sh       # ramsey_model + stoch_simul(order=1) no Octave + Dynare
python compare_ramsey.py
```

```
nk_linear         e.e. 0.0e+00   IRFs 5.3e-15   (6 séries; modelo com 8 variáveis, 4 do planejador)   OK
rbc_labortax      e.e. 5.0e-11   IRFs 4.9e-13   (8 séries; modelo com 11 variáveis, 6 do planejador)   OK
nk_ramsey_model   e.e. 2.7e-12   IRFs 2.1e-14   (8 séries; modelo com 13 variáveis, 8 do planejador)   OK
```

| Model | Origin | What it exercises |
|---|---|---|
| `ramsey/nk_linear.mod` | written here | The textbook New Keynesian model (Galí, 2015, ch. 5). It has an analytical answer, consisting of the targeting rule `π_t = −(λ/κ)(x_t − x_{t−1})` and the closed form of `x_t`, and `tests/test_optimal.py` checks both without Dynare. |
| `ramsey/rbc_labortax.mod` | written here | An RBC with a labour tax as the instrument. The equilibrium with `τ = 0` is efficient, and the planner keeps the tax at zero. It exercises the planner's condition for a **stock**, `k(+1)`, chosen at `t` and holding in expectation. |
| `mods/ramsey/nk_ramsey_model.mod` | Dynare's tests (`tests/optimal_policy`) | Nonlinear, with a Rotemberg Phillips curve and an Euler equation in `c(+1)` and `pai(+1)`. The derivatives of the constraints with respect to `t+1` depend on `t`, and the condition needs the lagged copies `lag_c` and `lag_r`. It runs **unchanged**. |

The comparison covers the steady state and the IRFs of every private variable **and of the
multipliers**. In `dsgelab`, `MULT_j` is the multiplier of the j-th equation that is not an
exogenous process. Dynare numbers one multiplier per equation, exogenous ones included, and gives
the same equation the same value with the same sign.

**Two pitfalls on Dynare's side.** `nk_ramsey_model.mod` calls `stoch_simul` without `order=`, and
Dynare's default is 2, so the IRFs it produces are second-order generalised impulse responses,
which differ from the first-order ones by `4e-05`. `run_ramsey.sh` sets `order=1`. Separately,
`oo_.steady_state` stores the `initval` of the `.mod` (`c = n = 0.9671684882`), which misses the
exact root `c = 0.875^{1/4}` in the seventh decimal place. The point at which Dynare actually
linearises is `oo_.dr.ys`, and the comparison is made against that point.

## Dynare's examples, read without porting: `from_mod`

`compare.py` checks ports written by hand. Here `dsgelab.from_mod` reads each `.mod` as it stands,
and the comparison is between what the two programs derive from the same file. Besides the three
models above, the set includes five models from the `examples/` folder of Dynare's repository (tag
6.4) and `fs2000_nonstationary.mod` from `tests/trend_var` (7.2), which writes the same model in
levels.

```bash
bash fetch.sh
bash run_examples.sh       # strip_mod.py reduz cada .mod ao modelo; stoch_simul(order=1) igual para todos
python compare_import.py
```

```
modelo         vars  aux    parâm.     e.e.     IRFs  momentos
RBC_baseline     15    5   3.6e-16  1.1e-13  4.4e-14   4.9e-14  OK
example1          6    5   0.0e+00  1.2e-14  1.7e-14   1.9e-14  OK
example2          6    5   0.0e+00  2.1e-05  4.3e-06   3.3e-06  OK  (e.e. do Dynare com resíduo 1e-06)
Gali_2015        14    7   0.0e+00  0.0e+00  1.6e-14   7.5e-15  OK
NK_baseline      28   16   1.8e-16  9.3e-14  5.9e-11   2.2e-11  OK
              parâmetros que o .mod deixa ao MATLAB, tomados do Dynare: gammma1, Rbar, Lambdax, LambdaYd
              ponto de partida do e.e. tomado do Dynare para 28 variável(is)
agtrend          15    6   0.0e+00  3.9e-08  2.1e-07   2.3e-07  OK  (e.e. do Dynare com resíduo 4e-07)
example3          6    5   0.0e+00  8.3e-15  1.4e-14   2.2e-14  OK
              ponto de partida do e.e. tomado do Dynare para 4 variável(is)
fs2000           14    8   3.2e-16  9.2e-15  3.4e-13   3.4e-13  OK
fs2000_nonstationary   11    6   0.0e+00  1.4e-08  1.5e-08   2.2e-08  OK  (e.e. do Dynare com resíduo 7e-08)
```

| Model | What the `.mod` reader has to handle |
|---|---|
| `fs2000` | `c(+2)`, shocks **inside** production and credit and not only in an AR(1), and a parameter named `del`, which is a reserved word in Python |
| `Gali_2015` | `model(linear)`, local `#` variables (one of them named `lambda`), the `steady_state(y)` operator, and a **price level** with a unit root |
| `agtrend` | the macro processor, with `@#define` and nested `@#if`/`@#else` |
| `example1`, `example3` | correlated shocks, with the constant `phi` assigned outside `parameters` |
| `NK_baseline` | 28 equations, indexation, price and wage rigidity, `predetermined_variables` |
| `RBC_baseline` | parameters computed inside `steady_state_model` |
| `fs2000_nonstationary` | `trend_var(growth_factor=)` with endogenous and stochastic growth factors, and `var(deflator=)` with composite deflators (`M(-1)/A`). Dynare and `dsgelab` each detrend the model independently, and the comparison is of the stationary model that results |

**What came from Dynare, as the output states.** `NK_baseline` computes four parameters and the
whole steady state in a MATLAB file (`NK_baseline_steadystate.m`), and `example3` solves for hours
in a MATLAB helper. A `.mod` reader does not execute MATLAB, so these parameters and these starting
points are taken from Dynare. The steady state itself is solved by `dsgelab` and compared. The
parameters that the `.mod` defines are read by `from_mod` and checked against Dynare's (column
`parâm.`).

**Unit root.** The price level of `Gali_2015` (`pi = p - p(-1)`) has a root of exactly 1. Dynare
counts it as stable (`qz_criterium = 1 + 1e-6`). By default `dsgelab` does not, and it refuses the
model with `BlanchardKahnError`. `solve(..., qz_criterium=1 + 1e-6)` reproduces Dynare's
convention, and `compare_import.py` uses that setting.

## Discretion, optimal simple rules and welfare

```bash
bash fetch.sh
bash run_policy.sh         # osr, discretionary_policy e evaluate_planner_objective no Dynare
python compare_policy.py   # --sem-dynare pula a avaliação da perda do Dynare no ótimo do dsgelab
```

```
OSR — osr_example.mod
  perda do dsgelab no ótimo do Dynare   0.4268335923233252
  perda reportada pelo Dynare           0.4268335923233251   desvio relativo 2.6e-16
  ótimo do dsgelab                      0.4266626273124760   menor ou igual que o do Dynare
     gammax0     dsgelab     0.424427   Dynare     0.445744
     gammac0     dsgelab     1.563685   Dynare     1.686970
     gamma_y_    dsgelab    16.666667   Dynare    16.660559
     gamma_inf_  dsgelab     3.977354   Dynare     3.563333
  perda do PRÓPRIO Dynare no ótimo do dsgelab 0.4266626273124778   desvio 4.3e-15

OSR sobre o bem-estar — nk_osr_planner.mod (planner_objective, ordem 2)
  perda do dsgelab no ótimo do Dynare   0.0225969550620490
  perda reportada pelo Dynare           0.0225969550620490   desvio relativo 1.5e-16
  ótimo do dsgelab                      0.0225969550603055   menor ou igual que o do Dynare
     phi_pi      dsgelab     1.071929   Dynare     1.072218
     phi_u       dsgelab    -0.301073   Dynare    -0.301586

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
  condicional   -70.778245895834  Dynare -70.778245895855   desvio relativo máx 3.5e-13

OSR: OK  OSR sobre o bem-estar: OK  discrição: OK  Ramsey do .mod: OK  bem-estar com capital: OK
```

`from_mod` reads the three Dynare files without translation. The declarations `planner_objective`,
`ramsey_model(...)`, `discretionary_policy(instruments=..., planner_discount=...)`, `osr_params`,
`optim_weights` and `osr_params_bounds` become `ModFile.planner`, `osr_params`, `optim_weights` and
`osr_bounds`.

**The `osr` optimum.** Evaluated at the optimum that Dynare reports, the `dsgelab` loss function
gives Dynare's value to 2.6e-16. The `dsgelab` optimum has a **lower** loss, 0.4266626 against
0.4268336, and three optimisers (BFGS, Nelder–Mead, Powell) reach it from the starting point of the
`.mod` and from Dynare's optimum. Dynare itself, evaluated at the `dsgelab` point
(`+osr/objective`), gives the same loss to 4e-15. Dynare's `csminwel` therefore stops early, along
an almost flat direction. At the optimum `gamma_y_ = 16.667 = 1/|σ|`, so the rule fully offsets
the demand shock, which enters the IS curve with coefficient `σ = −0.06` on the interest rate. The
third call to `osr` in the file uses `opt_algo=1`, which in Octave requires the `optim` package,
and `run_policy.sh` removes that call from the copy.

**Conditional welfare under discretion.** Outside Ramsey, Dynare's `evaluate_planner_objective`
takes `oo_.exo_simul(1,:)` as the first-period innovation whenever a simulation has been run, and
`dennis_1.mod` simulates 200 periods. Dynare's conditional figure therefore depends on a random
draw. `run_policy.sh` exports that draw, and `DiscretionSolution.welfare(shock=...)` reproduces it.
Without an initial shock the value is different, and that value is what `welfare()` returns by
default.

**The simple rule on welfare.** `ramsey/nk_osr_planner.mod`, written here, is `osr` with
`planner_objective`, so the loss is minus second-order unconditional welfare. At Dynare's optimum
the `dsgelab` loss function reproduces Dynare's value to 1.5e-16, and the `dsgelab` optimum has a
lower or equal loss. Bounds on the coefficients (`osr_params_bounds`) require, in Dynare, a
constrained optimiser that in Octave depends on extra packages, and the file does not use them.

**The accuracy floor under discretion.** The IRFs agree to 4e-08 because Dynare stops Dennis's
iteration at `sqrt(eps)`. `dsgelab` iterates to 1e-12 by default, and against Galí's closed form
(2015, ch. 5) its error stays below 1e-12 (`tests/test_policy.py`).

## Perfect foresight, with and without expectation errors

```bash
bash fetch.sh                # mods/foresight/, de tests/deterministic_simulations do Dynare 7.2
bash run_foresight.sh
python compare_foresight.py
```

```
rbc_det1                    condição inicial (histval)             desvio relativo máx 1.1e-14 (Output)  OK
rbc_det2                    choque no período 1                    desvio relativo máx 6.6e-15 (Capital)  OK
rbc_det3                    choques antecipados, períodos 4 a 8    desvio relativo máx 8.0e-14 (Output)  OK
pfwee                       erros de expectativa                   desvio relativo máx 2.8e-12 (c)  OK
pfwee_constant_sim_length   idem, horizonte constante              desvio relativo máx 3.0e-12 (c)  OK
```

`rbc_det1`–`3` validate `perfect_foresight` and `ModFile.deterministic` with an initial condition
set by `histval`, a shock in the first period and **anticipated** shocks (`shocks=`). `pfwee` and
`pfwee_constant_sim_length` validate `perfect_foresight_with_expectation_errors`. The information
table in `pfwee.csv` (news in periods 1 to 7, transitory shocks to the exogenous variable and
permanent changes in its terminal level) becomes the list of `News` through `ModFile.news`.

**Three settings the comparison had to match.**

1. **Dynare's tolerance.** The perfect-foresight solver stops at a residual of `1e-5`
   (`options_.dynatol`). With expectation errors it also recomputes the terminal steady state with
   `options_.simul.steady_tolf = eps^(1/3)`, an option copied from `solve_tolf` when the options are
   initialised, so changing `solve_tolf` afterwards leaves it unchanged. With the default
   tolerances the terminal `k` of `pfwee` misses the closed-form solution
   `((δ+β)/(aαx))^(1/(α−1))` by `4e-5`, and this accounts for the whole difference.
   `run_foresight.sh` asks for `1e-12` (and `1e-14` for the steady state) before every `setup` and
   every `solver`.
2. **The internal test of `pfwee.mod`.** The file ends with a manual replication and a test of
   exact equality between the two simulations, which fails under any other tolerance and aborts
   the script. The copy stops at `pfwee_simul`, which is what gets exported.
3. **News that is not news.** Some columns of the table only confirm what agents already expected
   (periods 4, 5 and 7), and Dynare does not replan at them. With a constant horizon this changes
   the length of the simulation, and `ModFile.news` discards those columns by the same rule.

**The exogenous variable as a level.** In Dynare `x` is a level with its own steady state (`x = 1`
in the `initval` of `pfwee`). `from_mod` carries it in the state `SHK_x`, with
`SHK_x(+1) = EXO_SS_x + x`, and a permanent change in the level is a change in the parameter
`EXO_SS_x`. The `x(+1)` in the Euler equation is the next value of that state.

## Estimation, smoothing, decomposition and forecasting

```bash
bash fetch.sh
bash run_estimation.sh          # fs2000 (calib, mode, mh), unitroot, condfc
python compare_estimation.py    # --sem-mh pula o Metropolis, que leva alguns minutos
```

Dynare's `fs2000.mod` is read by `from_mod` together with its priors and data. Two Dynare
conventions had to be reproduced. Without an initial value in `estimated_params`, the starting
point is the **prior mean**. The likelihood matches to the last digit only with
`options_.riccati_tol = 1e-16`, because by default Dynare switches to the steady-state gain of the
filter once the variance converges to 1e-6. In the diffuse filter, Dynare places the unit roots
**first** in the Schur form, with `P_inf` equal to the identity in that block. `dsgelab` does the
same, and the diffuse likelihood matches to 2.5e-12. Dynare's real-time decomposition leaves the
first-period shock in the initial condition, while the full decomposition does not, and both
conventions are reproduced.

## Occasionally binding constraints: OccBin

```bash
bash fetch.sh                # mods/occbin/, de tests/occbin do Dynare 7.2
bash run_occbin.sh           # escreve work_occbin/<nome>.mod e exporta out_occbin/<nome>.mat
python compare_occbin.py
```

```
borrcon      80 períodos  por partes 2.2e-16  linear 2.2e-16  regimes 30/30  iterações 6  OK
dynrbc      200 períodos  por partes 1.6e-15  linear 8.3e-16  regimes 169/169  iterações 99  OK
```

`borrcon.mod` has a borrowing constraint and surprise shocks in periods 10 and 30, with
`simul_curb_retrench`. `dynrbc.mod` has two constraints (irreversible and non-negative
investment), four regimes, and 52 surprise shocks in the impulse scenario (`option=1`). `from_mod`
reads both with the tagged equations, the `occbin_constraints` block, the `shocks(surprise)` and
the `occbin_setup`/`occbin_solver` options. Both the piecewise and the linear trajectories are
compared, in levels and in every period, together with the regime the agents expect in each period
(`oo_.occbin.simul.regime_history`). The working copy of `dynrbc` adds an `occbin_setup;` before
the solver, because without it Dynare's simulator does not receive the shock sequence.

## Extended path

```bash
bash fetch.sh                # mods/ep/, de tests/ep do Dynare 7.2
bash run_ep.sh               # leva uns dez minutos: o burnside roda sete variantes
python compare_ep.py
```

```
burnside.mod
  EP                                  101 períodos  desvio máx 7.2e-12 (relativo 5.7e-13)  OK
  EP estocástica, ordem 1             101 períodos  desvio máx 1.0e-11 (relativo 8.2e-13)  OK
    idem, híbrida                     101 períodos  desvio máx 1.1e-11 (relativo 8.3e-13)  OK
  EP estocástica, ordem 2             101 períodos  desvio máx 1.0e-11 (relativo 8.2e-13)  OK
    idem, árvore cheia                101 períodos  desvio máx 1.0e-11 (relativo 8.2e-13)  OK
    idem, híbrida                     101 períodos  desvio máx 1.0e-11 (relativo 8.2e-13)  OK
    idem, híbrida, árvore cheia       101 períodos  desvio máx 1.0e-11 (relativo 8.2e-13)  OK
linearmodel0.mod
  EP                                  201 períodos  desvio máx 2.0e-15 (relativo 3.8e-16)  OK
  solução de primeira ordem do Dynare  200 períodos  desvio máx 3.1e-14 (relativo 5.9e-15)  OK
```

The shocks are the ones Dynare drew, exported. The seven variants of `burnside.mod` differ from one
another by `2e-4` to `2e-1` in the tree price, and each matches its counterpart in Dynare to
`1e-11`. Two settings had to match. The first is the tolerance. Dynare stops at `1e-5` by default
and `run_ep.sh` asks for `1e-12`, so this tolerance is what limits the agreement. The second is the
Hessian in the hybrid variants. With the numerical Hessian the second-order risk correction differs
by `2e-7`, and with the exact one, `from_mod(..., derivatives="both")`, by `3e-10`. The variants
that ask for a fourth-order perturbation (`hybrid_order = 4`) are not compared.

## Trends and balanced growth

```bash
bash fetch.sh                # mods/bgp/, de tests/bgp do Dynare 7.2
bash run_bgp.sh              # avalia o bgpfun que o bgp.write do Dynare gera, em cinco pontos
python compare_bgp.py
```

```
o problema (bgp.write)
  solow   resíduos em 5 pontos contra bgpfun: desvio máx 0.0e+00  OK
  ramsey  resíduos em 5 pontos contra bgpfun: desvio máx 2.2e-16  OK
  nk      resíduos em 5 pontos contra bgpfun: desvio máx 0.0e+00  OK
a solução (asserções dos testes do Dynare)
  solow   300 partidas: K/Y 2.8e-12, g_Y 2.6e-14, g_K 2.3e-14, gE 1.1e-14  OK
  ramsey  fatores de crescimento contra 1.02: 1.3e-15  OK
  nk      níveis e fatores contra 1: 2.0e-15  OK
```

Dynare's tests solve the problem with MATLAB's `fsolve` from random points, and the Solow test is
skipped in Octave. For that reason the comparison covers the PROBLEM, meaning the residual that
`bgp.write` writes, point by point, together with the tests' assertions about the solution. The
level of a growing variable is not identified. A model in levels also has trivial paths (zero
population or zero capital), which `balanced_growth` discards before restarting from another
point. The detrending of a `.mod` with `trend_var` and `var(deflator=)` is validated in
`compare_import.py`, with `fs2000_nonstationary`.

## Declaring the model: diagnostics, closed form, homotopy, calibration

```bash
python compare_diagnostics.py     # roda o Dynare por conta própria
python compare_calibration.py     # lê macromodelling/calibration.json
```

`compare_diagnostics.py` has three parts. Its first part runs `diagnose` against Dynare's
`model_diagnostics` on a model broken on purpose (`diagnostics/walras.mod`, with Walras's law in
place of the Euler equation and a variable that appears only with a lead), and both programs
produce the same report. Next, `steady_state_model` serves as the steady state. In `fs2000` it
gives Dynare's point to `1e-16`, and the one in `diagnostics/bad_steady.mod`, wrong on purpose, is
refused by both programs. Homotopy (`diagnostics/homotopy.mod`) closes the script, on a model where
direct Newton fails and `homotopy_setup` reaches the closed form. Dynare reaches the same point
within its tolerance, `4e-6`, with the six digits it prints.

`compare_calibration.py` checks the calibration equations against MacroModelling.jl on one
RBC written in both programs, with `beta` pinned by `k/q = 2.5` (`k[ss] / q[ss] = 2.5 | β` in MacroModelling). `beta` and
the steady state agree to `4e-15`, and the closed form arbitrates between them. To regenerate the
`.json`, in a Julia project with MacroModelling and JSON3, run

```bash
julia --project=<projeto> macromodelling/calibration.jl < /dev/null
```

The `< /dev/null` is required. On first load, the PythonCall that MacroModelling uses waits for
something on standard input, and without it the process hangs indefinitely.
