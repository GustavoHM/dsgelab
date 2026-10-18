# Validation against Dynare

`dsgelab` is checked in three layers, and this folder holds the second.

| Layer | Against what | Where |
|---|---|---|
| 1. Answers known on paper | closed forms and answers known analytically: closed-form RBC, Taylor principle, lognormal risk coefficient | `../tests/` |
| **2. Models by other authors** | **Dynare's own examples and test files, one MacroModelling.jl calibration, and Heutel (2012)** | **this folder** |
| 3. An applied model from an unpublished paper by the package's author | Dynare and MacroModelling.jl | not public, so it cannot be rerun from this repository |

The first and third layers were set up by the package's author. In this one most models come from
[dynare.org/resources](https://www.dynare.org/resources/) (the examples of Collard's tutorial and
Johannes Pfeifer's collection), from Dynare's test suite and from Heutel's replication files, and
the reference solvers are Dynare and MacroModelling.jl. A few `.mod` files were written here for
the validation (`ramsey/nk_linear.mod`, `ramsey/rbc_labortax.mod`, `ramsey/nk_osr_planner.mod`,
`estimation/unitroot.mod` and the three models in `diagnostics/`). Some features have no
independent reference. The variance decomposition, pruning, announced parameter paths and the
Blanchard–Kahn check rest on the test suite, and the autocorrelations and the second-order
generalised impulse response have no test yet.

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

The numbers in this file come from a run on Dynare 7.2 and Octave 11.3.0 on 16 October 2026, which
printed the same numbers as an earlier run on 12 October. Every comparison also passed on Dynare 7.1 with the same printed numbers apart from the Heutel
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
bash fetch.sh          # downloads the .mod files (not versioned: they are GPL, the package is MIT)
bash run_dynare.sh     # runs each one in Octave + Dynare, exports out/<name>.mat
python compare.py      # compares dsgelab with Dynare
```

This needs GNU Octave and Dynare. `run_dynare.sh` looks for Dynare's `matlab/` directory in the
usual places, and `DYNARE_MATLAB=/path` sets it explicitly.

## Result

```
  example1       s.s. 1.2e-14   IRFs 2.9e-14   moments 2.6e-14   OK
  example2       s.s. 2.1e-05   IRFs 4.3e-06   moments 3.3e-06   OK  (limited by Dynare's s.s., residual 1e-06)
  rbc_baseline   s.s. 0.0e+00   IRFs 2.3e-14   moments 3.5e-14   OK
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

**Dating of the stock.** Dynare's `k` is the *end-of-period* stock. In `dsgelab`, `k` is the stock
*available* in the period, which is Dynare's `k(-1)`. The two series differ by one period. The
comparison shifts exactly those predetermined variables whose row of `Q` is zero, which are the
ones that receive no innovation directly. Swapping the two datings gives a model that solves,
satisfies Blanchard–Kahn and responds to shocks, yet is a different model, and only the dynamic
comparison reveals the difference.

**Orthogonalisation.** Dynare reports IRFs to *orthogonalised* impulses, in which the response to
shock *i* is the response to column *i* of the Cholesky factor of the covariance, and not to an
impulse in *i* alone. With a diagonal covariance the two coincide. With the
`var e, u = phi*0.009^2` of `example1` they do not. `dsgelab` provides `irf_impulse` for this case.

## What the validation found

**In `example2`, Dynare's `steady` stops at its default tolerance.** Dynare's steady state leaves a
residual of `1.05e-06` in the model equations, within the default `solve_tolf` of `eps^(1/3)`, about
`6e-06`, and `dsgelab`'s leaves `1.0e-44`. Everything derived from the steady state (linearisation,
IRFs, moments) inherits that error, which accounts for the discrepancies in the table, `2.1e-05` in
the steady state, `4.3e-06` in the IRFs and `3.3e-06` in the moments. When two programs disagree,
the residual of the equations at each program's point shows which point is closer to the solution.
`compare.py` therefore measures the residual at both points and forms its verdict from them.

An independent check is available because `example1` and `example2` are the same economy in
different coordinates. Exponentiated, the `example2` steady state matches the levels of `example1`
to `1e-15`.

**An error in my port, which only the dynamics revealed.** In porting `RBC_baseline` I copied the
`.mod`'s `log_k = log(k)` unchanged. In the `.mod`, `k` is Dynare's current `k`, the end-of-period
stock, which corresponds to `k(+1)` in the package's convention, so the port took the log of the
wrong period's stock. The steady state matched exactly, while the IRF of `log_k` lagged by one
period. This case is the reason the comparison covers trajectories as well as points.

## Optimal policy: the Ramsey planner

```bash
bash fetch.sh            # includes mods/ramsey/, from Dynare 7.2's test suite
bash run_ramsey.sh       # ramsey_model + stoch_simul(order=1) in Octave + Dynare
python compare_ramsey.py
```

```
nk_linear         s.s. 0.0e+00   IRFs 5.3e-15   (6 series; model with 8 variables, 4 from the planner)   OK
rbc_labortax      s.s. 5.0e-11   IRFs 4.9e-13   (8 series; model with 11 variables, 6 from the planner)   OK
nk_ramsey_model   s.s. 2.7e-12   IRFs 2.1e-14   (8 series; model with 13 variables, 8 from the planner)   OK
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
bash run_examples.sh       # strip_mod.py cuts each .mod down to the model; the same stoch_simul(order=1) for all
python compare_import.py
```

```
model          vars  aux    params     s.s.     IRFs   moments
RBC_baseline     15    5   3.6e-16  1.1e-13  4.4e-14   4.9e-14  OK
example1          6    5   0.0e+00  1.2e-14  1.7e-14   1.9e-14  OK
example2          6    5   0.0e+00  2.1e-05  4.3e-06   3.3e-06  OK  (Dynare's s.s. with residual 1e-06)
Gali_2015        14    7   0.0e+00  0.0e+00  1.6e-14   7.5e-15  OK
NK_baseline      28   16   1.8e-16  9.3e-14  5.9e-11   2.2e-11  OK
              parameters the .mod leaves to MATLAB, taken from Dynare: gammma1, Rbar, Lambdax, LambdaYd
              s.s. starting point taken from Dynare for 28 variable(s)
agtrend          15    6   0.0e+00  3.9e-08  2.1e-07   2.3e-07  OK  (Dynare's s.s. with residual 4e-07)
example3          6    5   0.0e+00  8.3e-15  1.4e-14   2.2e-14  OK
              s.s. starting point taken from Dynare for 4 variable(s)
fs2000           14    8   3.2e-16  9.2e-15  3.4e-13   3.4e-13  OK
fs2000_nonstationary   11    6   0.0e+00  1.4e-08  1.5e-08   2.2e-08  OK  (Dynare's s.s. with residual 7e-08)
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
`params`).

**Unit root.** The price level of `Gali_2015` (`pi = p - p(-1)`) has a root of exactly 1. Dynare
counts it as stable (`qz_criterium = 1 + 1e-6`). By default `dsgelab` does not, and it refuses the
model with `BlanchardKahnError`. `solve(..., qz_criterium=1 + 1e-6)` reproduces Dynare's
convention, and `compare_import.py` uses that setting.

## Discretion, optimal simple rules and welfare

```bash
bash fetch.sh
bash run_policy.sh         # osr, discretionary_policy and evaluate_planner_objective in Dynare
python compare_policy.py   # --no-dynare skips Dynare's evaluation of the loss at the dsgelab optimum
```

```
OSR — osr_example.mod
  dsgelab loss at Dynare's optimum     0.4268335923233252
  loss reported by Dynare              0.4268335923233251   relative deviation 2.6e-16
  dsgelab optimum                      0.4266626273124760   smaller or equal than Dynare's
     gammax0     dsgelab     0.424427   Dynare     0.445744
     gammac0     dsgelab     1.563685   Dynare     1.686970
     gamma_y_    dsgelab    16.666667   Dynare    16.660559
     gamma_inf_  dsgelab     3.977354   Dynare     3.563333
  Dynare's OWN loss at dsgelab's optimum 0.4266626273124778   deviation 4.3e-15

OSR on welfare — nk_osr_planner.mod (planner_objective, order 2)
  dsgelab loss at Dynare's optimum     0.0225969550620490
  loss reported by Dynare              0.0225969550620490   relative deviation 1.5e-16
  dsgelab optimum                      0.0225969550603055   smaller or equal than Dynare's
     phi_pi      dsgelab     1.071929   Dynare     1.072218
     phi_u       dsgelab    -0.301073   Dynare    -0.301586

Discretion — dennis_1.mod
  IRFs (5 variables x 3 shocks x 40)   max relative deviation 4.3e-08
  unconditional welfare    dsgelab 100.2259173048   Dynare 100.2259172521   5.3e-10
  conditional welfare      dsgelab 99.8889224747   Dynare 99.8889224138   6.1e-10
     (with the first innovation of Dynare's simulation, which is what it uses; without a shock: 99.0953548618)
  Dennis iterations: 18

Ramsey read from the .mod — nk_ramsey_model.mod
  9 series compared, multipliers included (same numbering as Dynare)
  steady state  5.9e-12     first-order IRFs  2.6e-14
  welfare    unconditional -25.213284815613  Dynare -25.213284815578
             conditional   -25.213284815613  Dynare -25.213284815580
             zero multipliers -25.213116382888  Dynare -25.213116382855   max relative deviation 1.4e-12

Welfare with capital — rbc_labortax.mod, order 2
  unconditional -70.761476342807  Dynare -70.761476342831
  conditional   -70.778245895834  Dynare -70.778245895855   max relative deviation 3.5e-13

OSR: OK  OSR on welfare: OK  discretion: OK  Ramsey from the .mod: OK  welfare with capital: OK
```

`from_mod` reads the three Dynare files without translation. The declarations `planner_objective`,
`ramsey_model(...)`, `discretionary_policy(instruments=..., planner_discount=...)`, `osr_params`,
`optim_weights` and `osr_params_bounds` become `ModFile.planner`, `osr_params`, `optim_weights` and
`osr_bounds`.

**The `osr` optimum.** Evaluated at the optimum that Dynare reports, the `dsgelab` loss function
gives Dynare's value to 2.6e-16. The `dsgelab` optimum has a **lower** loss, 0.4266626 against
0.4268336, and three optimisers (BFGS, Nelder–Mead, Powell) reach it from the starting point of the
`.mod` and from Dynare's optimum. Dynare itself, evaluated at the `dsgelab` point
(`+osr/objective`), gives the same loss to 4e-15. Dynare's `csminwel` therefore stops, at its
default tolerance, before the minimum, along an almost flat direction. At the optimum `gamma_y_ = 16.667 = 1/|σ|`, so the rule fully offsets
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
(2015, ch. 5) its largest error is 1.0e-12 (`tests/test_policy.py` requires less than 1e-10).

## Perfect foresight, with and without expectation errors

```bash
bash fetch.sh                # mods/foresight/, from Dynare 7.2's tests/deterministic_simulations
bash run_foresight.sh
python compare_foresight.py
```

```
rbc_det1                    initial condition (histval)            max relative deviation 1.1e-14 (Output)  OK
rbc_det2                    shock in period 1                      max relative deviation 6.6e-15 (Capital)  OK
rbc_det3                    anticipated shocks, periods 4 to 8     max relative deviation 8.0e-14 (Output)  OK
pfwee                       expectation errors                     max relative deviation 2.8e-12 (c)  OK
pfwee_constant_sim_length   idem, constant horizon                 max relative deviation 3.0e-12 (c)  OK
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
python compare_estimation.py    # --no-mh skips the Metropolis, which takes a few minutes
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
bash fetch.sh                # mods/occbin/, from Dynare 7.2's tests/occbin
bash run_occbin.sh           # writes work_occbin/<name>.mod and exports out_occbin/<name>.mat
python compare_occbin.py
```

```
borrcon      80 periods  piecewise 2.2e-16  linear 2.2e-16  regimes 30/30  iterations 6  OK
dynrbc      200 periods  piecewise 1.6e-15  linear 8.3e-16  regimes 169/169  iterations 99  OK
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
bash fetch.sh                # mods/ep/, from Dynare 7.2's tests/ep
bash run_ep.sh               # takes about ten minutes: burnside runs seven variants
python compare_ep.py
```

```
burnside.mod
  EP                                  101 periods  max deviation 7.2e-12 (relative 5.7e-13)  OK
  stochastic EP, order 1              101 periods  max deviation 1.0e-11 (relative 8.2e-13)  OK
    idem, hybrid                      101 periods  max deviation 1.1e-11 (relative 8.3e-13)  OK
  stochastic EP, order 2              101 periods  max deviation 1.0e-11 (relative 8.2e-13)  OK
    idem, full tree                   101 periods  max deviation 1.0e-11 (relative 8.2e-13)  OK
    idem, hybrid                      101 periods  max deviation 1.0e-11 (relative 8.2e-13)  OK
    idem, hybrid, full tree           101 periods  max deviation 1.0e-11 (relative 8.2e-13)  OK
linearmodel0.mod
  EP                                  201 periods  max deviation 2.0e-15 (relative 3.8e-16)  OK
  Dynare's first-order solution       200 periods  max deviation 3.1e-14 (relative 5.9e-15)  OK
```

The shocks are the ones Dynare drew, exported. The seven variants of `burnside.mod` differ from one
another by `2e-4` to `2e-1` in the tree price, and each matches its counterpart in Dynare to
`1.1e-11` or less. Two settings had to match. The first is the tolerance. Dynare stops at `1e-5` by default
and `run_ep.sh` asks for `1e-12`, so this tolerance is what limits the agreement. The second is the
Hessian in the hybrid variants. With the numerical Hessian the second-order risk correction differs
by `2e-7`, and with the exact one, `from_mod(..., derivatives="both")`, by `3e-10`. The variants
that ask for a fourth-order perturbation (`hybrid_order = 4`) are not compared.

## Trends and balanced growth

```bash
bash fetch.sh                # mods/bgp/, from Dynare 7.2's tests/bgp
bash run_bgp.sh              # evaluates, at five points, the bgpfun that Dynare's bgp.write generates
python compare_bgp.py
```

```
the problem (bgp.write)
  solow   residuals at 5 points against bgpfun: max deviation 0.0e+00  OK
  ramsey  residuals at 5 points against bgpfun: max deviation 2.2e-16  OK
  nk      residuals at 5 points against bgpfun: max deviation 0.0e+00  OK
the solution (assertions of Dynare's tests)
  solow   300 starts: K/Y 2.8e-12, g_Y 2.6e-14, g_K 2.3e-14, gE 1.1e-14  OK
  ramsey  growth factors against 1.02: 1.3e-15  OK
  nk      levels and factors against 1: 2.0e-15  OK
```

Dynare's tests solve the problem with MATLAB's `fsolve` from random points, and the Solow test is
skipped in Octave. For that reason the comparison covers the *problem*, meaning the residual that
`bgp.write` writes, point by point, together with the tests' assertions about the solution. The
level of a growing variable is not identified. A model in levels also has trivial paths (zero
population or zero capital), which `balanced_growth` discards before restarting from another
point. The detrending of a `.mod` with `trend_var` and `var(deflator=)` is validated in
`compare_import.py`, with `fs2000_nonstationary`.

## Local identification

```bash
bash fetch.sh                     # mods/identification/, from Dynare 7.2's tests/identification
bash run_identification.sh        # Dynare's identification at the prior mean, with advanced=1
python compare_identification.py
```

Three models are compared, each at the prior mean of its `estimated_params`. `kim2` (Kim, 2003)
comes from Dynare's tests: `phi` and `theta` only enter together and `dumpy` does not enter at all,
so no criterion has full rank, and the test runs with `tol_rank=1e-8`. In `as2007` (An and
Schorfheide, 2007), also from the tests, all 13 parameters are identified. `fs2000` comes from the
examples and estimates two shock standard deviations, which Dynare places first among its columns.
`run_identification.sh` cuts each file before its first `identification` or `estimation` command
and appends one with `advanced=1`, which adds the brute-force collinearity search. The numbers
below come from a run on 18 October 2026, with Dynare 7.2 and Octave 11.3.0.

| | kim2 | as2007 | fs2000 |
|---|---|---|---|
| moments, largest relative difference | 1.2e-13 | 3.4e-14 | 7.7e-13 |
| Jacobian of the moments, largest difference / column norm | 1.8e-08 | 2.7e-10 | 5.3e-09 |
| G of the spectrum, largest difference / largest entry | 2.4e-08 | 1.6e-12 | 1.4e-10 |
| size of Komunjer and Ng's matrix (minimal state) | 9 × 6 (1) | 45 × 31 (3) | 21 × 17 (2) |
| rank, sensitive, dependent and weak parameters, three criteria, two tolerances | same | same | same |
| multicollinearity, moments and spectrum, largest difference | not compared | 7.5e-12 | 5.5e-08 |
| collinearity search, largest cosine difference | 7.8e-09 (groups of one) | 6.2e-11 | 1.4e-09 |

Dynare differentiates analytically and `dsgelab` by central differences with a Richardson step,
at a step chosen for each parameter. In `fs2000`, `bet = 0.993` lies 0.007 from 1, where the
moments diverge, and a step of 0.1% of its value gave derivatives that were off by 1e-6; halving
the step until two Richardson estimates agree brings the difference with Dynare to 5e-9.

The verdicts are compared twice, with Dynare's tolerance (`"robust"`, or `1e-8` for kim2) and with
`dsgelab`'s default, `"auto"`, which also counts as zero a singular value below ten times the
estimated error of the derivatives. Both give Dynare's ranks and the same sets of sensitive,
dependent and weak parameters on all three models. On a model declared in Python, linearised by
finite differences, the derivatives carry an error near 1e-7. Two parameters that enter only
through their product then leave a singular value of that size, which the robust tolerance counts
as a rank, and `"auto"` exists for that case (`tests/test_identification.py`).

Komunjer and Ng's matrix is compared by its size, which gives the size of the minimal state. Its
entries, its condition number and the multicollinearity computed from it depend on the basis of
the minimal system, which each package chooses in its own way. The multicollinearity of the other
two criteria and the collinearity search for groups of more than one parameter are compared only
when the moments have full rank. In kim2 the columns of `phi` and `theta` are collinear to rounding
error, and a regression of `alph` on both measures that error: Dynare finds a cosine of 0.79 for
the pair and `dsgelab` keeps `theta` alone, at 0.72, because the second regressor only adds noise.

## Declaring the model: diagnostics, closed form, homotopy, calibration

```bash
python compare_diagnostics.py     # runs Dynare on its own
python compare_calibration.py     # reads macromodelling/calibration.json
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
the steady state agree to `4.4e-15`, and the closed form decides between them. To regenerate the
`.json`, in a Julia project with MacroModelling and JSON3, run

```bash
julia --project=<project> macromodelling/calibration.jl < /dev/null
```

The `< /dev/null` is required. On first load, the PythonCall that MacroModelling uses waits for
something on standard input, and without it the process hangs indefinitely.
