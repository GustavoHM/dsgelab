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

## Second- and third-order decision rules

```bash
bash run_decision_rule.sh
python compare_decision_rule.py
```

`example1` and `example2` compare `ghx`, `ghu`, `ghxx`, `ghuu`, `ghxu` and `ghs2` with Dynare 7.2.
The comparison uses the steady state exported by Dynare as the common point of linearisation. All
five tensors agree in the state-rule rows. `ghu` and `ghuu` also agree in the jump rows, with
absolute errors below `5e-14` in both models. The full risk constant `g_s2 = [h_s2; g_ss]`, with
`h_s2 = h_ss - h_uu:Sigma`, agrees in every row. The jump block `g_ku` is covered by a closed-form
RBC test in the unit suite. The comparison accounts for the extra exogenous transition in Dynare's
endogenous predetermined-state rows and for the different deviation scales of the two programs.

At third order, `ghuuu`, `ghxuu`, `ghxxu`, `ghxxx`, `ghxss` and `ghuss` agree in every row.
Both models use `derivatives="third"`. The largest absolute errors stay below `3e-13` in
`example1` and `2e-14` in `example2`. The `ghxss` errors are `1.44e-16` and `5.39e-17`;
the `ghuss` errors are `1.46e-16` and `5.35e-17`.
The exact RBC test also checks `h_uuu`, `h_kuu`,
`h_kku`, `h_kkk` and the composed jump blocks `g_uuu`, `g_kuu`, `g_kku` and `g_kkk`.
The sparse residual tensors have 161 and 216 nonzeros, below 1.4% of their 16,464 dense
entries. The Hessians have 46 and 56 nonzeros, below 4.8% of their 1,176 dense entries.
`solve_third_order(sparse=True)` gives the same solution and stays below `3e-13`
against Dynare on both models. With `direct_limit=0`, the matrix-free Sylvester solve uses
60 independent symmetric unknowns instead of 162 in these models. Its largest Dynare error
is `5.67e-13`, and its equation residual stays below `3.7e-14`.

`python compare_third_sparse.py` checks the canonical solver at larger scale against a closed-form
diagonal system. With 71 equations and 35 states it solves 551,670 unknowns in 1.9 seconds, with
a peak RSS of 175 MB on macOS. The residual and closed-form error are both `8.88e-16`. The known
term and residual use canonical coordinates, and each matvec expands one equation at a time with
an optimized contraction path. This benchmark covers the Sylvester core, not the unpublished
71-variable applied model or the dense decision-rule composition available on explicit request.
`python compare_third_sparse.py --full` runs the complete nonlinear pipeline with 71 variables,
35 states, 36 jumps and five shocks. Exact sparse third derivatives of the cubic state transitions
drive `K3`, all risk blocks and the iterative solution. The maximum residual is `3.05e-20`, elapsed
time is 135 seconds and peak RSS is 321 MB on macOS. This is a synthetic scale test, distinct from
the unpublished applied model.

The unit suite also checks conditional IRFs of the pruned rules at second and third order,
including the impact period and the nonlinear AR(1) recursion. `run_third_irf.sh` and
`compare_third_irf.py` compare 40 periods of the full third-order pruned rule on `example1` and
`example2`, with 5,000 replications and `drop=100`. The largest errors are `5.65e-4` in `example1`
and `1.50e-5` in `example2`, below the `1e-3` tolerance for Monte Carlo variation. Correlated shocks use the Cholesky factor.
Contracting `ghuuu` with the shock covariance does not recover `ghuss`.

`stochastic_steady_state` solves the fixed point of the full unpruned rule with `sigma=1`. Its
residual is zero on `example1`, `example2` and the linear NK model. In the logarithmic RBC, the
technology fixed point is zero in deviation (level 1), which is different from the positive
ergodic mean.

## Parameter Jacobians

```bash
bash run_derivatives.sh
python compare_derivatives.py
```

`example1` and `example2` compare `solution_jacobian` with Dynare's analytical `dghx` and `dghu`.
The composition covers `F P`, `F Q`, the extra exogenous transition of endogenous states and the
derivatives of the deviation scales. The comparison uses Dynare's steady state as the common point.
The largest errors are `3.49e-6` and `3.58e-6` in `example1` and `1.88e-5` and `6.96e-2` in
`example2`, below the tolerances of `0.1` absolute and `1%` relative. The largest relative error is
`0.512%`.

### Second-order Jacobians

```bash
bash run_second_derivatives.sh
python compare_second_derivatives.py
```

`second_order_jacobian` is compared with Dynare's analytical `dghxx`, `dghxu`, `dghuu` and
`dghs2` in every variable row, for model parameters, shock standard deviations and correlations.
The largest absolute errors are `6.22e-6` in `example1` and `1.56e-2` in `example2`; the largest
relative errors are `5.76e-8` and `0.552%`. The comparison uses the same tolerances as the
first-order case: `0.1` absolute and `1%` relative. Dynare 7.2 omits cross-covariance changes in
`dghs2` for `stderr` with correlated shocks. The package differentiates the full covariance; only
the comparator reproduces Dynare's omission to check that column.

### Third-order Jacobians

```bash
bash run_third_derivatives.sh
python compare_third_derivatives.py
```

`third_order_jacobian` is compared with `dghxxx`, `dghxxu`, `dghxuu` and `dghuuu` for model
parameters, shock standard deviations and correlations. `ghxss` and `ghuss` remain outside its
scope. The largest relative errors are `4.73e-8` in `example1` and `0.159%` in `example2`.
The tolerance is `0.1` absolute and `5%` relative for the parameter derivatives, which remain
numerical even when the model's third derivative is exact. The comparator uses
`sparse=True, direct_limit=0` for the solution and for every perturbed Jacobian point.

### Particle filter

```bash
bash run_particle.sh
python compare_particle.py
```

The scalar AR(1) comparison runs Dynare's `sequential_importance_particle_filter` with 100,000
particles and systematic resampling at first and second order. The likelihood differences are
`3.18e-2` and `5.22e-3`; the comparison allows `0.05` for Monte Carlo variation. The second-order
path checks the augmented pruned state and the `C z_{t-1} + D inov_t` observation timing.
The pruned third-order filter uses `[xf; xs; xt]` and the full rule, including `ghxss` and `ghuss`.
Its linear AR(1) likelihood is within `0.38` of Kalman; on the nonlinear RBC it differs from second order by
`0.0847`. Missing observations are skipped. `EstimationProblem(order=3)` exposes the same
likelihood to posterior mode and samplers. Third-order backward-information smoothing uses the
linear transition of `xf`, with a pseudoinverse for singular transition covariances. Its AR(1)
means agree with the Kalman smoother within `0.2` in the unit suite.

### Inversion filter

```bash
bash run_inversion.sh
python compare_inversion.py
```

The AR(1) comparison uses Dynare's `occbin.IVF_core` with a slack constraint. Recovered shocks,
the observation Jacobian, the total likelihood and every per-date contribution agree exactly
(tolerance `1e-8`). Unit tests also check the exact Gaussian density at first and pruned second
order, a nonlinear RBC path and missing observations.

`macromodelling/inversion.jl` and `compare_inversion_julia.py` also check a nonlinear RBC at
pruned second order. Recovered shocks differ from MacroModelling.jl by at most `0.0057` (tolerance
`0.01`). MacroModelling's likelihood uses `log|G|` once and omits `Sigma^-1`; dsgelab follows the
Dynare convention, so the likelihoods are not compared with a numerical adjustment.

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

## Central-bank models from the Macroeconomic Model Data Base

The Macroeconomic Model Data Base (MMB, [macromodelbase.com](https://www.macromodelbase.com))
collects models in Dynare, and its replication toolbox Rep-MMB
([github.com/AlexMeyer-Gohde/Rep-MMB](https://github.com/AlexMeyer-Gohde/Rep-MMB)) holds 148 of
them at the commit `fetch.sh` uses. The folder
`mmb/` reads thirteen of them with `from_mod`, as they are, and compares them with Dynare in the
same way as `compare_import.py`. The set covers models built or used at the Banco Central do
Brasil, the ECB, the Federal Reserve, the Banco Central de Chile, the Bank of England, the
Riksbank and the IMF, from 9 to 361 variables.

```bash
bash mmb/fetch.sh          # 13 .mod files at a fixed Rep-MMB commit, each checked by SHA-256
bash mmb/run_dynare.sh     # ../strip_mod.py and the same stoch_simul(order=1) for all
python mmb/compare.py
```

```
model         vars  aux    params     s.s.     IRFs   moments
BRA_SAMBA08     41   38   0.0e+00  0.0e+00  6.6e-13   3.8e-13  OK
CA_LS07         12   10   0.0e+00  0.0e+00  9.6e-15   1.1e-14  OK
CL_MS07        124   70   9.0e-14  0.0e+00  3.2e-10   2.6e-10  OK
EACZ_GEM03     239  100   3.9e-16  6.8e-13  1.1e-11   1.2e-08  OK
             s.s. starting point taken from Dynare for 2 variable(s)
EAUS_NAWM08    226  127   0.0e+00  4.8e-06  6.8e-05   7.0e-05  OK  (Dynare's s.s. with residual 5e-06)
EA_ALSV06        9   12   0.0e+00  0.0e+00  1.3e-13   1.5e-13  OK
EA_SWW14        50   35   1.1e-16  1.0e-37  5.6e-13   6.5e-13  OK
HK_FP13         47   41   0.0e+00  0.0e+00  3.9e-11   6.3e-09  OK
UK_SM11        123   64   0.0e+00  0.0e+00  2.6e-11   5.2e-11  OK
US_DNGS15       38   33   0.0e+00  0.0e+00  1.4e-12   5.1e-13  OK
US_FRB03       279  365   0.0e+00  0.0e+00  6.6e-10   1.8e-09  OK
US_FRB08       361  335   2.2e-16  0.0e+00  5.8e-10   1.7e-09  OK
             parameters the .mod leaves undefined, taken from Dynare: bpnca30
US_SW07         41   29   2.7e-16  2.3e-38  2.5e-13   5.3e-13  OK
             parameters the .mod leaves undefined, taken from Dynare: ccs, cinvs, crdpi
```

| Model | Origin |
|---|---|
| `BRA_SAMBA08` | SAMBA, Banco Central do Brasil |
| `EAUS_NAWM08` | New Area-Wide Model, ECB (Coenen, McAdam and Straub, 2008), euro area and United States |
| `EA_SWW14` | Smets, Warne and Wouters (2014), ECB |
| `US_SW07` | Smets and Wouters (2007) |
| `US_DNGS15` | Del Negro, Giannoni and Schorfheide (2015), Federal Reserve Bank of New York |
| `US_FRB03`, `US_FRB08` | FRB/US, Federal Reserve Board, linearised with rational expectations (2003 version of Levin, Wieland and Williams, and February 2008 version) |
| `CL_MS07` | Medina and Soto (2007), Banco Central de Chile |
| `UK_SM11` | Harrison, Thomas and de Weymarn (2008), Bank of England, in the version estimated by Barriel and Millard |
| `EA_ALSV06` | Adolfson, Laséen, Lindé and Villani, Riksbank, in the MMB's reduced version |
| `EACZ_GEM03` | Global Economy Model, IMF (Laxton and Pesenti, 2003), euro area and Czech Republic |
| `HK_FP13`, `CA_LS07` | Funke and Paetz (2013) for Hong Kong, Lubik and Schorfheide (2007) for Canada |

**What differs from `compare_import.py`.** The steady-state error is relative, and absolute for
values below 1. Several of these models have variables whose steady state is zero, and the
relative error of `compare_import.py` (denominator floored at 1e-8) would measure their rounding.
In `EACZ_GEM03` the output gaps are 7e-13 in `dsgelab` and 0 in Dynare, which the relative error
reports as 7e-05. In both scripts, a variable whose largest response is below 1e-9 of the
model's largest response is left out of the IRF column. Price and wage dispersion do not move at
first order, and in `EAUS_NAWM08` the wage dispersion `EA_SI` responds with 6e-11 in both
programs, against responses of order 1e4 in other variables. This rule leaves the output of
`compare_import.py` unchanged.

**What limits the comparison.** In `EAUS_NAWM08`, Dynare's steady state leaves a residual of
5e-06, and the one `dsgelab` finds leaves 9e-14. The IRFs and moments inherit that residual, as
in `example2`. `EACZ_GEM03` declares an innovation (`interest_`) that no equation uses. Dynare 7
refuses the file unless it runs with `nostrict`, which drops the innovation, and `run_dynare.sh`
passes that option for this model only. The parameters listed as taken from Dynare are declared
and never assigned in the `.mod`.

**What the set leaves out.** Eleven other Rep-MMB models from central banks and official
institutions were tried, and `from_mod` does not yet read or solve them. `G2_SIGMA08` (Federal
Reserve), `EA_AWM05` (ECB) and `CA_BMZ12` (Bank of Canada) declare a parameter twice, `ESREA_FIMOD12` stops on an expression the reader does not parse, and
`US_CMR14` uses `@#for`. `CA_ToTEM10` (Bank of Canada), `DEREA_GEAR16` (Bundesbank),
`EA_GNSS10` (Banca d'Italia), `EA_QUEST3` (European Commission) and `FI_AINO16` (Bank of
Finland) take parameters or the steady state from MATLAB files, and the steady state of
`GPM6_IMF13` does not converge from the file's starting point.

## Published-model files from DSGE_mod

Johannes Pfeifer's [DSGE_mod](https://github.com/JohannesPfeifer/DSGE_mod) collection supplies
tractable Dynare replication files for published models. Four are compared here at first order,
using Dynare 7.2 as the independent reference. The source is fixed at commit
`26c21f79e4c6a6b19b18e6330edc5b23489907a4`, and every downloaded file is checked by SHA-256.
The GPL source files are fetched rather than versioned in this MIT package.

```bash
bash dsge_mod/fetch.sh
bash dsge_mod/run_dynare.sh
python dsge_mod/compare.py
```

```
model                    vars  aux    params     s.s.     IRFs   moments
Aguiar_Gopinath_2007       21    7   0.0e+00  3.5e-16  1.1e-12   4.6e-13  OK
Ghironi_Melitz_2005        35    8   0.0e+00  3.1e-08  2.2e-07   1.8e-07  OK
Hansen_1985                 9    3   0.0e+00  3.1e-16  1.9e-14   1.5e-14  OK
Smets_Wouters_2007_45      40   27   0.0e+00  6.9e-15  9.8e-14   5.8e-13  OK
```

| Model | Source exercise and scope |
|---|---|
| `Aguiar_Gopinath_2007` | Aguiar and Gopinath (2007), Mexico benchmark and Table 4 shock calibration. The detrended system is compared; reconstruction of nonstationary levels and the paper's filtered-moment table are not. |
| `Ghironi_Melitz_2005` | Ghironi and Melitz (2005), financial autarky. Dynare's steady state leaves a residual of `2e-08`, which the IRF and moment differences inherit. Its MATLAB steady-state file supplies `fx` and `fx_`; those values and the initial guess are taken from Dynare. |
| `Hansen_1985` | Hansen (1985), the file's default indivisible-labor economy. The comparison uses unfiltered theoretical moments, not the paper's simulated HP-filtered Table 1. |
| `Smets_Wouters_2007_45` | Smets and Wouters (2007), at the calibration in the `.mod`. Three declared but unused parameters come from Dynare. The Bayesian estimation and the original paper's estimated results are not replicated. |

`prepare.py` makes two explicit source-specific working copies. For Aguiar–Gopinath it selects the
Mexico/Table 4 shock block and omits interleaved MATLAB plotting code; for Hansen it adds semicolons
to the macro-processor title assignments so the lightweight statement reader sees the following
`var` block separately. Neither changes equations, parameters or shock values. The other two files
are read directly.

These comparisons validate the supplied model systems against Dynare. They do not claim to reproduce
every table, figure, robustness exercise or estimation in the papers.

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
bash run_bgp.sh              # evaluates, at five points, the bgpfun that Dynare's bgp.write
                             # generates, and solves the models with their trends declared
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
trends declared in from_equations
  model   against        s.s.     IRFs    paths  variance
  solow   Dynare      4.0e-15  2.3e-14  1.1e-14   2.3e-14  OK
  solow   from_mod    4.4e-15  1.2e-14  1.3e-14   2.5e-14  OK
  ramsey  Dynare      5.7e-16        -        -         -  OK
  ramsey  from_mod    0.0e+00        -        -         -  OK
  fs2000  Dynare      3.4e-14  9.7e-14  8.2e-14   7.8e-14  OK
  fs2000  from_mod    1.4e-15  1.4e-13  1.0e-13   2.1e-14  OK
  ramsey  closed form of the detrended steady state: 5.4e-15  OK
```

Dynare's tests solve the problem with MATLAB's `fsolve` from random points, and the Solow test is
skipped in Octave. For that reason the comparison covers the *problem*, meaning the residual that
`bgp.write` writes, point by point, together with the tests' assertions about the solution. The
level of a growing variable is not identified. A model in levels also has trivial paths (zero
population or zero capital), which `balanced_growth` discards before restarting from another
point. The detrending of a `.mod` with `trend_var` and `var(deflator=)` is validated in
`compare_import.py`, with `fs2000_nonstationary`.

The third part checks the trends declared in `from_equations`. `run_bgp.sh` rewrites Solow and
Ramsey with their trends declared (`trend_var(growth_factor=EfficiencyGrowth) Efficiency;`,
`var(deflator=Efficiency*Population) Output PhysicalCapitalStock;`, and
`trend_var(growth_factor=1.02) x;` with `c` and `k` deflated by `x`), removes the trend equations they replace and saves them as
`out_bgp/<name>_trend.mod`. Solow gets shocks with standard deviations 0.01 and 0.005, which
the test file does not set. `fs2000_nonstationary.mod` is read unchanged up to its `steady`
command. Dynare solves each with `solve_tolf = 1e-14`, and for the two with shocks runs
`stoch_simul(order=1, irf=40, periods=0)` and simulates 200 periods of the shocks
`0.01 sin(0.7 t j + j)` with `simult_`. `compare_bgp.py` writes the three models as text equations
with `trend_vars=` and `deflators=`, starts the steady state from the `initval` block and
compares it with Dynare and with `from_mod` on the same `.mod`. The IRFs and the paths are
measured against each variable's largest response, and the variance matrix against its largest
entry. Equal paths under the same shocks mean equal decision rules, whatever the choice of
states. In `from_equations` the growth factors are states with the innovation dated `t+1`, so
the models have different states from `from_mod`'s, and fs2000's `del` is `delta`, because
`del` is a Python keyword and the parser refuses it.

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

## Method of moments

```bash
bash fetch.sh
bash run_mom.sh                  # first-order GMM and SMM, eight cases
python compare_mom.py
```

```
case            differences at Dynare's estimate (est.: own optimum; checked for SMM_e5)
  anscho_GMM_0    data 2.5e-16  model 1.4e-14  est. 0.0e+00  Q 2.4e-14  s.e. 2.1e-08  J 2.4e-14  OK
  anscho_SMM_0    data 2.5e-16  model 1.5e-14  est. 0.0e+00  Q 2.2e-14  s.e. 3.5e-09  J 2.2e-14  OK
  anscho_GMM_4    data 2.5e-16  model 1.1e-14  est. 9.2e-01  Q 1.9e-15  s.e. skip  J 1.9e-15  OK
  anscho_SMM_4    data 2.5e-16  model 9.6e-15  est. 8.1e-01  Q 1.2e-14  s.e. skip  J 1.2e-14  OK
  anscho_GMM_d1   data 2.0e-16  model 2.3e-14  est. 7.9e-01  Q 5.2e-14  s.e. skip  J 5.1e-14  OK
  anscho_SMM_d1   data 2.0e-16  model 2.4e-14  est. 6.5e-01  Q 1.3e-13  s.e. skip  J 1.3e-13  OK
  anscho_GMM_e5   data 2.0e-16  model 4.3e-14  est. 2.8e-02  Q 2.1e-13  s.e. skip  J 2.1e-13  OK
  anscho_SMM_e5   data 2.0e-16  model 1.1e-14  est. 2.3e-06  Q 2.3e-14  s.e. 4.9e-05  J 2.3e-14  OK
```

The model is An and Schorfheide (2007), from Dynare 7.2's `method_of_moments` tests. The cases
cover GMM and SMM at the starting point and after optimization, two datasets and eight or five
estimated parameters, including shock standard deviations and a shock correlation. SMM uses the
same shock series as Dynare. Data and model moments, the objective and J are compared at Dynare's
estimate, with Dynare's exported final weighting matrix. The estimate column is diagnostic:
`csminwel` and `scipy.optimize.least_squares` can stop at different local optima. In the
well-conditioned five-parameter SMM case, the complete estimates agree to `2.3e-6` and the
standard errors to `4.9e-5`.

For `c = 1 + 1/simulation_multiple`, the SMM covariance has a factor `c` and J a factor `1/c`.
Dynare 7.2 omits `c` from its standard errors and multiplies J by `c`. The comparison therefore
multiplies Dynare's standard errors by `sqrt(c)` and divides its J by `c^2`. Standard errors are
checked in the two starting-point cases and in the complete five-parameter SMM estimation.

### Pruned second-order moments

```bash
bash run_mom_order2.sh
python compare_mom_order2.py
```

`example1` and `example2` compare means, covariances and three autocovariance lags of every
endogenous variable with Dynare 7.2's `pruned_SS.pruned_state_space_system`. All differences are
below `1e-14`, and the comparison requires `1e-8`. The models read by `from_mod`, including their
auxiliary variables, differ by at most `2.93e-8` and `1.59e-6` in means and by `6.08e-9` and
`2.87e-8` in second moments, with a tolerance of `5e-5`.
Unit tests also check exact linear-model reductions and the lognormal AR(1), whose moments agree
to `3.2e-7` with the closed form, within the accuracy of a second-order approximation.

### IRF matching and external targets

```bash
bash run_irf.sh                  # Dynare's frequentist RBC example, initial point and estimation
python compare_irf.py
```

```
  rbc_irf_0  data 0.0e+00  model 3.4e-14  W 0.0e+00  Q 2.9e-16  H 2.9e-06  s.e. 0.0e+00  est. 0.0e+00  OK
  rbc_irf_5  data 0.0e+00  model 5.7e-15  W 0.0e+00  Q 1.7e-14  H 2.0e-05  s.e. 6.6e-05  est. 1.3e-05  OK
```

The example exercises sparse targets, vector expansion, the `varobs`/MATLAB ordering and diagonal
weights. The comparison checks data and model IRFs, W, Q and the Hessian of Q/2 at the same point;
the complete estimates differ by `1.3e-5` and the standard errors by `6.6e-5`. At the initial point
Dynare reports imaginary standard errors from an indefinite Hessian; `dsgelab` reports `NaN` instead.
The unit tests also cover cross-weights, correlated shocks, a non-diagonal external covariance and
local projections. On 120,000 simulated AR(1) observations, the local projections recover the
model's responses through horizon 10 to `0.012` in absolute error, with their joint HAC covariance.

## Filtered moments

```bash
bash fetch.sh
bash run_filtered.sh              # stoch_simul with hp_filter=1600, then bandpass_filter=[6 32]
python compare_filtered.py
```

```
  model         filter  variables  covariance  autocorr.  decomp. (pp)
  example1      hp              6     9.3e-15    5.9e-15        5.00 *  OK
  example1      bp              6     9.7e-15    7.6e-15        5.00 *  OK
  example2      hp              6     4.5e-06    2.0e-06       1.3e-04  OK  (limited by Dynare's s.s., residual 1e-06)
  example2      bp              6     4.5e-06    2.5e-06       1.3e-04  OK  (limited by Dynare's s.s., residual 1e-06)
  RBC_baseline  hp             15     2.3e-14    1.2e-14       1.4e-12  OK
  RBC_baseline  bp             15     1.9e-14    1.4e-14       1.4e-12  OK
  fs2000        hp             14     3.0e-14    6.2e-14       5.5e-12  OK
  fs2000        bp             14     3.0e-14    9.3e-14       5.2e-12  OK
  unitroot      hp              3     9.7e-16    4.2e-16       2.8e-14  OK
  unitroot      bp              3     1.4e-15    5.0e-16       2.8e-14  OK
```

The five models are the three of `run_dynare.sh`, `fs2000` and `estimation/unitroot.mod`, whose
price level has a unit root and no unfiltered variance. Dynare computes the filtered moments from
the spectral density on 512 frequencies and its inverse Fourier transform (Uhlig, 2001), and
`dsgelab` does the same on the same grid, so the comparison measures the solution and not the
quadrature. That quadrature is checked separately in `tests/test_filtered_moments.py`, against
integrals of the spectrum by `scipy.integrate.quad`. The comparison covers the covariance matrix,
the autocorrelations and cross-correlations at lags 1 to 5 and the variance decomposition. In
`example2` the difference is that of Dynare's steady state, as in the comparison above. The
decomposition of `example1`, whose shocks are correlated, differs by 5 percentage points because
the two programs attribute the covariance differently: Dynare orthogonalises the shocks by the
Cholesky factor, in the order of declaration, and `dsgelab` counts each shock with its own
variance. The numbers come from a run on 24 October 2026, with Dynare 7.2 and Octave 11.3.0.

## Calibration restrictions

```bash
bash fetch.sh
bash run_restrictions.sh          # Dynare's tests/gsa/ls2003a.mod, at the calibration and over the prior
python compare_restrictions.py
```

```
  ls2003_rel  21 IRF + 17 moment restrictions   calibration: values 8.1e-15, verdicts same   512 draws (486 solved, same draws): values 1.3e-12, verdicts differ in 0   OK
              shares: 80.5% y(1:4), e_ys, 67.9% the other IRF restrictions together, 2.1% all
  ls2003_abs  21 IRF + 17 moment restrictions   calibration: values 1.6e-15, verdicts same   512 draws (486 solved, same draws): values 4.1e-14, verdicts differ in 0   OK
              shares: 11.3% y(1:4), e_ys, 97.1% the other IRF restrictions together, 0.6% all
```

The model is Lubik and Schorfheide (2003), from Dynare's `tests/gsa/ls2003a.mod`, with its priors
and its `irf_calibration` and `moment_calibration` blocks. `restrictions/prepare.py` writes two
cases: `ls2003_rel` keeps the active `irf_calibration(relative_irf)` block, and `ls2003_abs` uses
the block the file leaves commented, in absolute units. Both restore the two commented lines of the
moment block, the variance of `y_obs` and its cross-correlations with `pie_obs` at leads and lags
`-4:4`, so that a variance and a range of lags are covered. The include is inlined and the `@#for`
loops are expanded, because `from_mod` reads neither, and `dsgelab` reads the same driver as
Dynare. At the calibration, the reference is `endogenous_prior_restrictions`. Over the prior, it is
the `map_calibration` step of `sensitivity(prior_range=0, Nsam=512)`, and `dsgelab` evaluates the
restrictions at the 512 parameter vectors Dynare drew. Both programs solve the same 486 draws. On
those draws the restriction values agree at every horizon and lag, each restriction's verdict
agrees on every draw, and the shares are those Dynare prints. A restriction over a range of
horizons or lags holds when any one of them is inside the bound. This is the logical OR described
in Dynare's manual, and `dsgelab` applies the same rule. The numbers come from a run on 30 October
2026, with Dynare 7.2 and Octave 11.3.0.

## Global sensitivity

```bash
bash fetch.sh
bash run_gsa.sh                   # Dynare's stability mapping of tests/gsa/ls2003a.mod, three samples
pip install SALib                 # 1.6.0 for the numbers below
python compare_gsa.py
```

```
Stability mapping against Dynare's sensitivity(stab=1), ls2003a
  ls2003_prior       94.9% stable   5.1% indeterminate   0.0% explosive
     classes against Dynare's solver (info): differ in 0; against its GSA lists: 0; with its rule by modulus: 0
     filtering on Dynare's groups (2 analyses, 2 the same as dsgelab's): Smirnov d 0.0e+00, p 2.2e-16; correlations 1.1e-16, p 1.8e-14
  ls2003_range       97.2% stable   2.8% indeterminate   0.0% explosive
     classes against Dynare's solver (info): differ in 0; against its GSA lists: 0; with its rule by modulus: 0
     filtering on Dynare's groups (2 analyses, 2 the same as dsgelab's): Smirnov d 0.0e+00, p 2.2e-16; correlations 1.1e-16, p 3.1e-14
  ls2003_explosive   79.0% stable   5.1% indeterminate  15.9% explosive
     classes against Dynare's solver (info): differ in 0; against its GSA lists: 37; with its rule by modulus: 8, all rank failures (info 5)
     filtering on Dynare's groups (3 analyses, 1 the same as dsgelab's): Smirnov d 0.0e+00, p 3.3e-16; correlations 7.6e-17, p 1.7e-13

Sobol indices against SALib
  Ishigami                     design 0.0e+00   indices 8.9e-16   (1 outputs, 20480 evaluations)
  ls2003, IRFs and moments     design 0.0e+00   indices 7.8e-16   (5 outputs, 1280 evaluations)
```

The model is Lubik and Schorfheide (2003), from Dynare's `tests/gsa/ls2003a.mod`, with its priors
and without its calibration blocks. `gsa/prepare.py` writes three drivers: 1024 draws from the
prior (`prior_range=0`), 1024 uniform draws over the prior's range (`prior_range=1`), and 1024
draws from the prior with the persistence of technology `rho_A` uniform on [0.5, 1.1], so that
some draws are explosive. `gsa/export_gsa.m` saves Dynare's draws and classes and runs Dynare's
`smirnov_test` and `corrcoef` on the groups its `stability_mapping.m` forms, and `dsgelab`
classifies the same draws with `ModFile.stability_map(points=, names=)`.

Every draw has the class that Dynare's solver gives it: `info` 0 is determinate, 3 explosive, 4
and 5 indeterminate. In the first two samples the toolbox's own lists agree too. In the third
they differ in 37 draws. The toolbox lists 36 of them as indeterminate while its solver reports
them explosive (`info` 3), and one as explosive while its solver reports a rank failure (`info`
5). `stability_mapping.m` labels a draw
indeterminate when entry `nspred + 1` of `sort(eigval)` is stable. When every eigenvalue of a
draw is real, `sort` orders them by signed value, so a large negative unstable root, or `-Inf`,
comes before the stable ones. Applied to the eigenvalues sorted by modulus, the same rule gives
`dsgelab`'s class at every draw except the eight rank failures (`info` 5), which no eigenvalue
count can see and which Dynare's solver, like `dsgelab`, calls indeterminate. The Smirnov
statistics and p-values and the correlations are computed by `dsgelab` on Dynare's own groups, so
that they are compared on the same draws in all three samples. Dynare runs the filtering only
on the structural parameters, and the comparison does the same.

The Sobol check follows the one in `cgelab`: SALib's `sample.sobol` with the same seed draws the
same points as `dsgelab.sobol`, and SALib's `analyze.sobol`, given `dsgelab`'s outputs in SALib's
row order, gives the same first-order and total indices. On the Ishigami function the indices are
also within 0.004 of the closed form. The second case varies `psi1`, `rho_R` and `tau` of the
same model inside the determinacy region and takes impulse responses to the monetary shock and
two moments. The numbers come from a run on 30 October 2026, with Dynare 7.2, Octave 11.3.0 and
SALib 1.6.0.

## Mixed-frequency data

```bash
bash fetch.sh
bash run_mixed.sh                 # fs2000 with output growth observed as its annual mean
python compare_mixed.py
```

```
fs2000 with output growth observed as its annual mean (48 of 192 quarters) and inflation every quarter
  likelihood at two points 8.0e-13   smoothed variables 1.3e-14   smoothed annual mean (Dynare's gy_obs_a) 3.4e-16
  analytic gradient against finite differences 1.0e-07 (relative)
  posterior kernel at the mode: dsgelab 935.771192   Dynare 935.771188   parameters within 0.003 posterior sd
```

Dynare has no aggregation option, so `mixed/prepare.py` writes what a Dynare user writes:
`fs2000.mod` with an auxiliary variable `gy_obs_a = (gy_obs + gy_obs(-1) + gy_obs(-2) +
gy_obs(-3))/4` and its steady state, observed in place of `gy_obs`, and a data file that runs
Dynare's `fs2000_data.m` and adds the four-quarter means of `gy_obs` at quarters 4, 8, ..., with
NaN in between. With `loglinear`, the linearised aggregate is the mean of the log deviations.
`dsgelab` reads the unchanged `fs2000.mod` and declares `aggregation={"gy_obs": ("mean", 4)}`
on the same numbers. The likelihood is compared at the prior means and at a point 5% away from
them in alternating directions, the smoothed variables and the smoothed aggregate at the first,
and the posterior mode found by each program. The gradient check is internal: the analytic
gradient through the stacked state against central differences of the likelihood. The numbers
come from a run on 30 October 2026, with Dynare 7.2 and Octave 11.3.0.

## Plots: the numbers Dynare's figures draw

```bash
bash fetch.sh
bash run_plots.sh                 # densities: draw_prior_density and posterior_moments
                                  # mcmc: fs2000 with mode_check, Metropolis and bayesian_irf (4 min)
                                  # identification: fs2000's identification with advanced=1
                                  # decomposition: real-time decompositions, conditional forecast
python compare_plots.py
```

```
prior densities of fs2000 (9 priors: beta, inv_gamma, normal): grid 4.3e-15, density 1.9e-12 (relative)
posterior density, 6037 draws (3037 repeated): 6.5e-16 (grid and density, relative to its peak)
posterior density, 5000 draws (0 repeated): 8.9e-16 (grid and density, relative to its peak)
Brooks and Gelman, 2 chains of 2500 draws, 101 windows: windows 0, univariate (27 panels) 3.5e-14, multivariate 7.8e-14 (relative)
trace_plot, moving average over 401 draws, alp and the log posterior: 4.1e-16 (relative)
mh_autocorrelation, alp, lags 0 to 30: 2.3e-15
mode_check, 9 parameters, 21 points each: grid 8.9e-16; log posterior and log-likelihood kernels 5.2e-11 (relative) for the 7 parameters with a bounded prior
  gam (normal prior, Dynare's grid runs from -10 to 10): 1.4e-11 relative where both solve the model, and 0 of 21 points that only one of them solves
  logmst (normal prior, Dynare's grid runs from -10 to 10): 1.9e-02 relative where both solve the model, and 10 of 21 points that only one of them solves
posterior_irf, 1200 draws, 4 variables, 2 shocks, 20 periods: responses 2.5e-10, mean 2.1e-10, median 2.1e-10, lower 2.3e-10, upper 2.0e-10, deciles 2.3e-10 (relative to the largest response); the statistics of Dynare's own responses 3.6e-17
posterior densities and modes of 'Priors and posteriors': 1.9e-15; smoothed shocks 2.6e-15 and observables 4.1e-16 at the posterior mean
identification strength, fs2000 at the prior mean, 9 parameters, on Dynare's 300 simulated observations: information matrix 1.2e-08 (relative to the root of its diagonal), strength 2.7e-07, strength_prior 4.4e-08, sensitivity 5.8e-09, sensitivity_prior 5.8e-09, moments_sensitivity 4.9e-09 (relative); the bars drawn 2.7e-07, the patterns 2.4e-09 and their singular values 5.2e-06 (relative, as Dynare prints them); identified parameters the same
real-time decompositions of fs2000 at the prior mean, 4 variables, 192 observations, forecast 4: realtime pool 3.4e-15, one-step conditional pool 3.4e-15, one-step forecast pool 3.2e-15; vintages 40, 80, 120 (realtime, forecast and 4-step conditional) 4.6e-15; what plot_shock_decomposition draws with realtime 1, 2 and 3, rolling and at vintage 80, 4.0e-15
annualised decompositions, type=yoy 1.9e-14 and type=aoa 4.8e-15; initial condition decomposition, 14 of Dynare's 15 columns (the other is an auxiliary variable named otherwise), 5.1e-15
conditional forecast with bands, gp_obs on a path for 4 periods by e_m, 12 periods, 2000 of Dynare's draws, 14 variables: mean 6.7e-15, bands 4.0e-15, unconditional mean 6.7e-15, unconditional bands 4.4e-15, controlled shock 8.0e-16 (the widest 80% band is 0.25)
```

The plots draw what the package computes. What they draw that it did not compute before is
compared here, in four parts.

The densities of `plot_posterior`: the nine priors of `fs2000.mod`, read by `from_mod`, against
`draw_prior_density`, grid and values; and the kernel estimate that `posterior_moments` returns,
on draws made in Octave, 3000 normal draws each repeated one to three times, as a Metropolis chain
repeats a draw when it rejects, so that Dynare's bandwidth correction for repeated draws is
exercised, and 5000 lognormal draws.

The rest comes from an estimation of `fs2000.mod` in Dynare, with `mode_check`, two chains of
2,500 draws and `bayesian_irf`, followed by `trace_plot` and `mh_autocorrelation_function` for
`alp`. While Dynare draws its figures, `plots/dyn_saveas.m` stands in for Dynare's own and records
the data of every line and bar in them instead of saving the file, and `plots/export_plots.m`
gathers them with the draws of both chains, burn-in included, the mode, and the parameters and
responses of the posterior IRFs. Each figure is then compared with the function that draws it here,
on the same inputs: `brooks_gelman` with the 27 panels of the univariate figures and the 3 of the
multivariate one, the moving average of `trace_plot` and `mh_autocorrelation` with theirs, the
posterior densities, modes, smoothed shocks and smoothed observables with those of the "Priors and
posteriors" and smoother figures, at the posterior mean that Dynare's smoother uses after
Metropolis. `mode_check` is compared with the slices Dynare saves, at Dynare's mode, and
`posterior_irf`, run on the 1,200 parameter vectors of Dynare's `PosteriorIRF`, with the responses
at each and with `oo_.PosteriorIRF.dsge`.

The grid of `mode_check` is Dynare's to rounding. For a parameter with a normal prior, whose
bounds are infinite, Dynare's rule (reproduced as written) sends the grid to -10 and 10. At those
values of `logmst`, the log of money growth, money grows by a factor of up to `e^10` a quarter,
and the two programs part: Dynare finds no solution above 1, where dsgelab finds one, and at -10
the two log posteriors, of order -2e7, differ by 2%. On the seven parameters with a bounded prior
and on `gam` the kernels agree within 6e-11. The responses of the posterior IRFs at each draw agree
within 3e-10 of the largest response, and their statistics within the same: computed on Dynare's own responses, they agree with Dynare's within 4e-17.
The numbers come from a run on 3 November 2026, with Dynare 7.2 and Octave 11.3.0.

The figures left out in 0.9 come from two more runs. The first is Dynare's `identification` of
`fs2000.mod` at the prior mean with `advanced=1`, which draws the identification strength, the
sensitivity bars, the collinearity patterns and the identification patterns. Dynare computes the
information matrix behind the strength on a sample of 300 observations that it simulates, so a copy
of its `dsge_likelihood.m`, put first on the path, keeps the sample it receives for the asymptotic
Hessian, and `plots/export_identification.m` saves it with Dynare's measures and figures.
`identification(strength=True)` on that sample gives the information matrix within 1.2e-8 of the
root of its diagonal, and the strength, the sensitivity component and the moments' sensitivity
within 2.7e-7, relative. The strength divides by the asymptotic standard deviation, the root of
the diagonal of the inverse information, whose smallest singular value is 4.6e-5 against a largest
of 7.6e6, and that ratio amplifies the difference between Dynare's analytic derivatives and the
differences of dsgelab. The bars of the strength figure agree within the same, the identification
patterns within 2.4e-9, and the singular values in their titles within the five digits Dynare
prints. The sensitivity figure is compared through the moments' sensitivity only: its bars divide
by the largest value, which is infinite for the two shock standard deviations, whose inverse gamma
priors have an infinite standard deviation, so Dynare's moments bars are all minus infinity.
`plot_identification_sensitivity` divides by the largest finite value. Its "Model" and "LRE model"
bars come from derivatives of Dynare's own reduced form and dynamic Jacobian, whose rows differ
from dsgelab's state space, and are not drawn.

The second run estimates nothing: `fs2000.mod` at the prior mean (`mode_compute=0`), with
`initial_condition_decomposition`, `shock_decomposition`, `realtime_shock_decomposition` with
`forecast=4` and `save_realtime=[40 80 120]`, and a `conditional_forecast` of 12 periods on 2,000
draws in which `gp_obs` follows a path for 4 periods and `e_m` is the controlled shock.
`plots/export_decomposition.m` saves the pools and vintages, what `plot_shock_decomposition`
returns (its `get_decomp_only`) with `type=yoy`, `type=aoa` and `realtime` 1, 2 and 3, rolling and
at vintage 80, and the conditional forecast with the normal draws behind it, drawn again in Octave
from the state of `randn` recorded just before the command. Everything agrees within 2e-14. The
initial condition decomposition is compared on 14 of Dynare's 15 columns: the fifteenth is
Dynare's auxiliary variable for `c(+1)`, which dsgelab calls `LEAD1_c`, and contributes only to
itself at the first date. These numbers come from a run on 6 November 2026, with the same versions.

## Out-of-sample forecast evaluation

```bash
bash fetch.sh
bash run_forecast_eval.sh         # Dynare's recursive estimation of fs2000, and R's dm.test
python compare_forecast_eval.py   # about 12 minutes, most of it the nine posterior modes
```

```
fs2000 — recursive estimation, samples ending at 176 to 184, 8-period forecasts
  at Dynare's modes: forecast means 2.6e-15, 90% bands 6.3e-09
  dsgelab's own modes: kernel minus Dynare's, worst +4.98e-06   parameters, worst 0.003 posterior sd   forecasts, worst 0.000 predictive sd
     gp_obs: RMSE relative to the random walk, h=1..8  2.25 2.04 2.42 3.39 2.60 2.46 2.43 2.99
     gy_obs: RMSE relative to the random walk, h=1..8  0.62 0.87 0.71 0.58 0.72 0.67 0.64 0.84

Diebold-Mariano against R's forecast::dm.test, 48 cases: statistic 3.1e-15, p-value 1.1e-15, same horizon used True
```

The recursive estimation is that of `fs2000.mod` from Dynare's examples, with model, priors and
data as they are, and the estimation line replaced by
`estimation(nobs=[176:184], mode_compute=4, mh_replic=0, forecast=8)`. Dynare estimates the
posterior mode on each of the nine samples, starting each sample after the first from the first
sample's mode, and forecasts the two observables eight quarters ahead with 90% bands.
`forecast_evaluation` runs twice on the same samples. With `estimate="fixed"` and Dynare's mode
for each sample, the forecast means and the bands are Dynare's. With `estimate="mode"`, `dsgelab`
finds its own modes, starting each sample from the previous one. The kernel at each of them is at
least as high as at Dynare's, the parameters lie within 0.003 posterior standard deviations of
Dynare's, and the forecasts within a thousandth of their predictive standard deviation. The
relative RMSEs are printed for reference: over these nine origins the model forecasts output
growth better than the random walk and inflation worse.

The Diebold–Mariano reference is R's `forecast::dm.test` (forecast 8.24.0 on R 4.5.1). Two error
series of 60 observations, moving averages of orders 2 and 1, are drawn in R with a fixed seed.
The 48 cases cross horizons 1 to 4, losses `|e|` and `e^2`, both variance estimators and the
three alternatives. None of them has a negative variance at the requested horizon, so the
fallback to horizon 1 that both programs make is checked only in the test suite. The numbers
come from a run on 30 October 2026, with
Dynare 7.2 and Octave 11.3.0. The BVAR benchmark (`bvar_lags=`) is checked in the next section.

## BVAR and DSGE-VAR

```bash
bash fetch.sh                     # Dynare 7.2's tests/bvar_a_la_sims and tests/dsge-var, by SHA-256
bash run_bvar.sh                  # about a minute
python compare_bvar.py            # about two minutes, most of it the four posterior modes
```

```
bvar_standalone: bvar_density 2, first_obs=20, train=10, flat=True
  prior and posterior, lags 1 to 2: df 0e+00, S 5.6e-17, XXi 5.4e-15, PhiHat 1.1e-14 (relative)
  log marginal density: 6198.9907 6207.4897   deviation 1.8e-12
  bvar_forecast 2, nobs=200, 2 periods, 1001 draws: RMSE over 782 observations 8.7e-18
  paths, draw by draw: with shocks 4.9e-17, without 2.8e-17 (largest value 0.05); mean with shocks 8.7e-18; explosive draws 0.000
  bvar_irf, SquareRoot: 1000 draws of 40 periods, responses draw by draw 1.9e-15, oo_.bvar.irf 1.1e-15 (relative to 0.0127)
  bvar_irf, Cholesky, as the upper factor L': 1000 draws of 40 periods, responses draw by draw 9.5e-16, oo_.bvar.irf 5.4e-16 (relative to 0.0127)
  bvar_irf, Cholesky, the columns of L: 1000 draws of 40 periods, responses draw by draw 2.3e-01, oo_.bvar.irf 7.1e-02 (relative to 0.0127)
bvar_and_dsge: bvar_density 8, first_obs=20, train=10, flat=False
  prior and posterior, lags 1 to 8: df 0e+00, S 5.6e-17, XXi 1.8e-14, PhiHat 1.4e-14 (relative)
  log marginal density: 6199.5260 6207.7011 6206.4335 6206.3822 6202.7006 6199.4387 6194.0685 6190.8352   deviation 0.0e+00
  bvar_forecast 8, nobs=200, 10 periods, 2001 draws: RMSE over 782 observations 1.6e-17
  paths, draw by draw: with shocks 2.1e-16, without 1.8e-16 (largest value 0.06); mean with shocks 7.8e-17; explosive draws 0.000
  forecast_evaluation's BVAR(2) benchmark, origins [100, 130, 160, 190], expanding and 60-observation windows, 1 to 4 steps: errors 1.6e-16 (largest error 0.0221)

dsgevar_forward_calibrated_lambda: 4 observables, 4 lags, T = 90 rows (Dynare's T 94)
  log posterior kernel at 4 points: 2.5e-12;  log prior 8.9e-16;  -inf at the same 0 point(s): True
  kernel at Dynare's mode -9.1e-13 from Dynare's; dsgelab's mode: kernel +3.05e-09 over Dynare's, parameters within 5.2e-05 posterior sd
  Laplace with Dynare's inverse Hessian 9.7e-13; with dsgelab's Hessian -1.6e-04
  with dsgelab's T: mode moves by up to 0.18 posterior sd
  bayesian_irf of the DSGE-VAR: 1000 parameter draws, 40 periods, responses draw by draw 4.9e-14, oo_.PosteriorIRF.bvardsge 3.8e-14 (relative to 2.021); VAR draws redrawn as explosive 0.0000

dsgevar_forward_estimated_lambda: 4 observables, 4 lags, T = 90 rows (Dynare's T 94)
  log posterior kernel at 7 points: 4.9e-12;  log prior 8.9e-16;  -inf at the same 1 point(s): True
  kernel at Dynare's mode -1.6e-12 from Dynare's; dsgelab's mode: kernel +2.20e-10 over Dynare's, parameters within 9.3e-06 posterior sd
  Laplace with Dynare's inverse Hessian 1.6e-12; with dsgelab's Hessian -7.3e-06
  with dsgelab's T: mode moves by up to 0.39 posterior sd, lambda 0.568 against 0.533 (lambda T = 51.1 against 50.1)
  bayesian_irf of the DSGE-VAR: 1000 parameter draws, 40 periods, responses draw by draw 1.1e-13, oo_.PosteriorIRF.bvardsge 5.7e-14 (relative to 2.031); VAR draws redrawn as explosive 0.0010

OK
```

The BVAR files are Dynare's `tests/bvar_a_la_sims/bvar_standalone.mod` and `bvar_and_dsge.mod`,
with their data (`bvar_sample.m`, 1001 observations of two series), options and hyperparameters.
`run_bvar.sh` runs the estimation of `bvar_and_dsge.mod` without Metropolis, since it is not
compared, and adds a `verbatim` block before `bvar_forecast` and before each `bvar_irf` that
resets Octave's generator and saves the normal draws the command is about to take. `bvar` rebuilds, for every lag length of `bvar_density`, the prior and the posterior that
Dynare's `bvar.toolbox` returns, and the log marginal density. `BVAR.forecast` takes the saved
draws through an object with the `standard_normal` method of a numpy `Generator`, and its paths
with and without shocks are Dynare's, draw by draw: Dynare's loop makes `bvar_replic + 1` draws,
so 1001 and 2001 here. The RMSE is that of the forecast at the posterior mean over the 782
observations after `nobs=200`. Without the saved draws, the forecast means agree only up to Monte
Carlo error.

`bvar_standalone.mod` ends with `bvar_irf(2,'Cholesky')` and `bvar_irf(2,'SquareRoot')`, 1000
draws of 40 periods each. A second `verbatim` block after each saves `oo_.bvar.irf` and the
sampled responses, and `BVAR.irf` takes the saved normal draws. Dynare 7.2's `bvar.irf` builds
the responses with the impact matrix `B` on the left of the lag matrices, so that its impulses are
the rows of `B`, and stores the response of variable `i` to impulse `j` under
`oo_.bvar.irf.Mean.(j).(i)`, where its code names the first field the variable and the second
the shock. With `SquareRoot`, `B`
is symmetric, the rows are the columns, and Dynare's numbers are those of `BVAR.irf` with the
fields read shock first. With `Cholesky`, the rows of the lower factor `L` are impulses with
covariance `L'L` rather than `Sigma`: the first variable's impulse moves only the first variable
on impact. `BVAR.irf` takes the columns of `L` by default, and with the upper factor `L'` as its
identification it gives Dynare's numbers. The responses to the columns differ from Dynare's by up
to 23% of the largest response in single draws, and the statistics by up to 7.1%.

The benchmark of `forecast_evaluation` is checked on `bvar_and_dsge.mod`, whose estimation sets
`first_obs=20` and whose model, two AR(1) processes for `dx` and `dy`, `from_mod` reads (with the
prior shapes put in lower case, as `NORMAL_PDF` is written there). `forecast_evaluation` runs with
the parameters fixed at their prior means and `bvar_lags=2`, at origins 100, 130, 160 and 190,
on an expanding window and on a rolling window of 60 observations. On each of the eight windows,
Octave sets `first_obs` and `nobs` to the same observations, calls `bvar.toolbox(2)` with Dynare's
default hyperparameters and no training sample, and runs the recursion that `bvar.forecast` uses
for its RMSE, the VAR at `PhiHat` from the last two observations, for four steps. The errors are
compared with `ForecastEvaluation.bvar_errors`.

The DSGE-VAR files are Dynare's `tests/dsge-var/dsgevar_forward_calibrated_lambda.mod`
(`dsge_var=0.8`) and `dsgevar_forward_estimated_lambda.mod` (`dsge_prior_weight` with a uniform
prior on [0, 1.9]), on the data their `simul_hybrid.mod` simulates in Octave: 500 periods of a
model with a hybrid Phillips curve, of which the tests use 90 from `first_obs=50`, with 4 lags and
a constant. `run_bvar.sh` keeps their models, priors, samples, optimiser (`mode_compute=4`) and
Metropolis (2000 draws) and removes `bayesian_irf`. After the mode, it evaluates Dynare's
`dsge_var_likelihood` at the prior mean, at the mode and at points perturbed from both by 5 and 10
per cent, and in the second file at the mode with `lambda` set to 0.3, 0.5, 1.5 and 0.2. Below
`(k + n) / T = 21 / 94 = 0.223` the prior is improper, and both programs give `-inf` at 0.2.
`from_mod` reads the files, and `ModFile.estimation_problem` the `dsge_var` options.

Dynare 7.2 builds the dataset with the 4 initial observations before `first_obs` and passes its
length, 94, as the `T` of Del Negro and Schorfheide's formula (A.2), while the sample moments
`Y'Y`, `Y'X` and `X'X` are sums over the 90 rows of the VAR. `dsgelab`'s `T` is 90. Under that
count the likelihood is the marginal density of the VAR under the normal-inverse-Wishart prior of
`lambda T` artificial observations, which `tests/test_dsge_var.py` checks against conjugate updating
one observation at a time, and its limit as `lambda` grows is the Gaussian likelihood of the VAR
the model implies. Under Dynare's count, neither holds. `compare_bvar.py` therefore compares the
two programs with `T` set to Dynare's, through a subclass of `EstimationProblem`, and then reports
how far `dsgelab`'s own `T` moves the mode. At the prior mean of the first file the two counts
give log posteriors of -373.37 (Dynare's) and -366.43.

For the DSGE-VAR's impulse responses, `run_bvar.sh` calls `PosteriorIRF` after the estimation,
as `bayesian_irf` would. It saves the parameter draws `get_posterior_subsample` takes (1000 of the
1000 Metropolis draws after the burn-in, with Octave's uniform generator restored afterwards) and
the normal draws, which the inverse-Wishart and matrix-normal draws of each VAR consume. At each
parameter draw, `dsge_var_irf` draws `(Phi, Sigma)` from the posterior of
`EstimationProblem.dsge_var_posterior`, redraws while the VAR is explosive, and rotates the
Cholesky factor of `Sigma` by the `Q` of the QR decomposition of the model's impact matrix, as
`PosteriorIRF_core1.m` does. Dynare draws `Sigma` with `floor((1 + lambda) T) - k` degrees of
freedom, where `dsgelab` takes the posterior's `(1 + lambda) T - k`, and the comparison uses
Dynare's through a subclass, with Dynare's `T`. The responses agree draw by draw, and so do the
mean, median, variance, 90% HPD interval and deciles of `oo_.PosteriorIRF.bvardsge`. In the second
file one VAR draw in a thousand was explosive and drawn again, in both programs. The numbers come
from a run on 6 November 2026, with Dynare 7.2 and Octave 11.3.0.

## Process discretisation: Tauchen and Rouwenhorst

```bash
pip install quantecon==0.13.1
python compare_markov.py          # about two minutes, most of it the 40-digit references
```

```
QuantEcon 0.13.1, 120 processes

tauchen against QuantEcon
  nodes 0.0e+00   transition 2.2e-16 (absolute)
tauchen against 40 digits: 29 distinct chains (the probabilities depend on rho and n only); 1 reducible in double precision for both packages, left out
  transition probabilities, relative:  dsgelab 1.1e-11   QuantEcon 1.8e+00, with probabilities rounded to zero in 13 chains
  stationary distribution:             dsgelab 1.1e-14   QuantEcon 2.9e-08, and 4 chains it finds reducible
rouwenhorst against QuantEcon
  nodes 0.0e+00   transition 0.0e+00   stationary 1.2e-16
rouwenhorst against the closed form
  binomial stationary distribution 2.5e-16   mean, variance and autocorrelations of the AR(1) 3.0e-14
invariance pi P = pi, every chain 1.1e-16
```

`tauchen` and `rouwenhorst` are compared with QuantEcon's functions of the same names on 120 AR(1)
processes: persistence -0.5, 0, 0.5, 0.9, 0.99 and 0.999, innovation standard deviations 0.007 and
1, chains of 2, 3, 7, 25 and 101 nodes, and means 0 and -1.3 (QuantEcon's `mu` is `(1 - rho)`
times the mean). The nodes and the transition matrices agree to rounding in every case, and so do
Rouwenhorst's stationary distributions.

QuantEcon is not the reference for Tauchen's small probabilities. It computes the upper tail as
`1 - cdf`, which keeps an absolute accuracy of 1e-16 but no relative accuracy below that, so
probabilities under 1e-16 are lost or rounded to zero. In the chains with two nodes this moves the
stationary distribution by 3e-8, and in four chains it leaves a probability at zero and makes the
chain reducible. `dsgelab` computes every interval from the side of the conditional mean it lies
on, with the normal survival function above it. The reference for those numbers is the same
construction in 40-digit arithmetic (mpmath), with the stationary distribution found by GTH
elimination and accepted only if it satisfies `pi P = pi` to 1e-30. Against it, the probabilities
agree to 1.1e-11 in relative terms and the stationary distributions to 1.1e-14. The largest
relative errors are in probabilities below 1e-200, where an interval ends some 35 standard
deviations from the conditional mean and the rounding of that end point alone moves the
probability by about `a^2 eps`. With persistence 0.999 and two nodes, the probability of leaving
an end node is below the smallest double, and both packages find the chain reducible.

For Rouwenhorst the closed form is the reference as well: the stationary distribution is binomial
with `n - 1` trials of probability one half, and the chain has the mean, the variance and the
autocorrelations of the AR(1) at every lag, which are checked at lags 1 and 4. The numbers come
from a run on 30 October 2026, with QuantEcon 0.13.1 on Python 3.14.

## Time iteration against dolo

```bash
bash dolo/setup.sh                            # Python 3.11 with dolo 0.4.9.20, and its models
dolo/.venv/bin/python dolo/run_dolo.py        # about 30 seconds
python compare_time_iteration.py              # about 25 seconds
```

```
1. dolo 0.4.9.20 (interpolation 2.2.7), tolerance 1e-11
 rbc (linear): dsgelab 362 iterations, 5.66 s; dolo 352 iterations, 7.49 s
  ok   rbc, linear: largest difference at the grid points          1.03e-11  (limit 1e-08)
  ok   rbc, linear: largest difference between them                1.00e-11  (limit 1e-08)
       Euler-equation errors at the cells' centres: max 5.98e-03, mean 1.60e-03
 rbc (cubic): dsgelab 329 iterations, 6.96 s; dolo 361 iterations, 9.30 s
  ok   rbc, cubic: largest difference at the grid points           9.52e-10  (limit 1e-08)
  ok   rbc, cubic: largest difference between them                 9.30e-10  (limit 1e-08)
       Euler-equation errors at the cells' centres: max 2.34e-03, mean 1.59e-04
 consumption_savings (linear): dsgelab 108 iterations, 0.34 s; dolo 135 iterations, 2.13 s
  ok   consumption_savings, linear: largest difference at the grid points  5.55e-11  (limit 1e-08)
  ok   consumption_savings, linear: largest difference between them  5.39e-11  (limit 1e-08)
       the borrowing limit binds at 24 of 100 grid points in dsgelab, 24 in dolo
       Euler-equation errors at the cells' centres: max 5.39e-03, mean 1.36e-04
 consumption_savings (cubic): dsgelab 107 iterations, 0.52 s; dolo 134 iterations, 3.05 s
  ok   consumption_savings, cubic: largest difference at the grid points  5.51e-11  (limit 1e-08)
  ok   consumption_savings, cubic: largest difference between them  5.36e-11  (limit 1e-08)
       the borrowing limit binds at 24 of 100 grid points in dsgelab, 24 in dolo
       Euler-equation errors at the cells' centres: max 6.13e-03, mean 1.27e-04
 rbc_mc (linear): dsgelab 351 iterations, 1.40 s; dolo 351 iterations, 4.46 s
  ok   rbc_mc, linear: largest difference at the grid points       5.73e-14  (limit 1e-08)
  ok   rbc_mc, linear: largest difference between them             5.57e-14  (limit 1e-08)
       Euler-equation errors at the cells' centres: max 1.11e-02, mean 1.53e-03
 rbc_mc (cubic): dsgelab 343 iterations, 1.39 s; dolo 343 iterations, 4.19 s
  ok   rbc_mc, cubic: largest difference at the grid points        5.51e-14  (limit 1e-08)
  ok   rbc_mc, cubic: largest difference between them              5.54e-14  (limit 1e-08)
       Euler-equation errors at the cells' centres: max 5.10e-03, mean 1.40e-04
2. stochastic growth with log utility and full depreciation, against the closed form
  linear: largest relative error of c with 10, 20, 40, 80 capital points: 1.37e-03, 3.35e-04, 8.35e-05, 2.08e-05
  ok   linear, 80 points: largest relative error of c              2.08e-05  (limit 3e-05)
  ok   linear: error ratio when the spacing halves, distance from 4  9.26e-02  (limit 5e-01)
  cubic: largest relative error of c with 10, 20, 40, 80 capital points: 4.97e-04, 1.14e-04, 2.74e-05, 6.72e-06
  ok   cubic, 80 points: largest relative error of c               6.72e-06  (limit 1e-05)
  ok   cubic: error ratio when the spacing halves, distance from 4  3.57e-01  (limit 5e-01)

all checks passed
```

dolo 0.4.9.20 installs only on Python 3.11, with pandas below 2.0, so `dolo/setup.sh` creates an
environment of its own with uv, with every version pinned in `dolo/requirements.txt`, and downloads
three of dolo's example models from its `v0.4.9.20` tag, checking their sha256 sums. `run_dolo.py`
solves them with dolo's `time_iteration` and writes the policies to
`out_time_iteration/dolo.json`, and `compare_time_iteration.py` solves the same models with
`dsgelab`, written as residual functions, on the same grids:

- `rbc.yaml`: states z and k, labour and investment as controls, labour bounded below by zero, the
  iid shock integrated by Gauss–Hermite with 5 nodes. The file asks for a 100 x 100 grid, and the
  comparison uses 20 x 20.
- `consumption_savings_iid.yaml`, as published: cash on hand on 100 points, the income shock on
  dolo's 5 equiprobable nodes, which `dsgelab` receives through `quadrature=`, and consumption
  between zero and cash on hand.
- `rbc_mc.yaml`, as published: capital on 50 points, productivity a two-state Markov chain, labour
  between 0.01 and 1 and investment between 0 and 1.

Both programs solve the same equations with the same interpolant, since dolo's cubic splines (from
the `interpolation` package) also have a zero second derivative at the ends and extrapolate
linearly. The complementarity convention is the same as well. dolo solves each step with at most
10 Newton steps to a residual of 1e-8 and offers no option to change either, and at that
tolerance its policy stops some 1e-7 from the fixed point (4e-7 on `rbc`, 3e-7 on `rbc_mc`).
`run_dolo.py` replaces the two solvers, for its own run only, with versions at a tolerance of
1e-12 and 50 steps. With that, the policies agree within 1e-9 at the grid points and at the
centres of the cells, relative to the largest control, and the iteration counts are close (the
same on `rbc_mc`). What remains on `rbc` is the stopping rule: both stop when the policy moves by
less than 1e-11, and with persistence 0.8 and a discount factor of 0.99 the iteration contracts
slowly, so a step of 1e-11 leaves the policy some 1e-9 from the fixed point. The Euler-equation
errors printed for each model are `dsgelab`'s, at the centres of the cells, where the
interpolation is least accurate, and they are largest where a bound starts to bind.

The closed form is the second reference: with log utility and full depreciation,
`c = (1 - alpha beta) a k^alpha`, whatever the process of productivity. The error at the centres
of the cells falls by a factor of four each time the spacing halves, with linear and with cubic
interpolation. A natural spline has a zero second derivative at the ends of the grid where the
policy's is not zero, so near the ends the cubic interpolant is also second order, with a constant
about three times smaller than the linear one's. The test suite checks the same closed form with a
product of two Markov chains, and the cubic interpolant against SciPy's natural `CubicSpline`. The
numbers come from a run on 30 October 2026, with dolo 0.4.9.20 on Python 3.11 and `dsgelab` on
Python 3.12.

## Improved time iteration and expectations iteration against dolo and the closed form

```bash
bash dolo/setup.sh                                 # as for time iteration
dolo/.venv/bin/python dolo/run_dolo_iti.py         # about 90 seconds
python compare_improved_time_iteration.py          # about 90 seconds
```

```
1. dolo 0.4.9.20's improved_time_iteration, tolerance 1e-11
 rbc (linear): dsgelab 0 + 9 Newton steps (0.90 s), time iteration 352; dolo 0 + 8 Newton steps, time iteration 352
  ok   rbc, linear: against dolo at the grid points                3.37e-10  (limit 1e-08)
  ok   rbc, linear: against dolo between them                      3.30e-10  (limit 1e-08)
  ok   rbc, linear: against time iteration                         4.37e-11  (limit 1e-09)
 rbc (cubic): dsgelab 0 + 9 Newton steps (2.85 s), time iteration 361; dolo 0 + 8 Newton steps, time iteration 361
  ok   rbc, cubic: against dolo at the grid points                 3.72e-10  (limit 1e-08)
  ok   rbc, cubic: against dolo between them                       3.68e-10  (limit 1e-08)
  ok   rbc, cubic: against time iteration                          2.67e-11  (limit 1e-09)
 consumption_savings (linear): dsgelab 40 + 4 Newton steps (0.07 s), time iteration 114; dolo 40 + 4 Newton steps, time iteration 114
  ok   consumption_savings, linear: against dolo at the grid points  3.46e-12  (limit 1e-08)
  ok   consumption_savings, linear: against dolo between them      3.30e-12  (limit 1e-08)
  ok   consumption_savings, linear: against time iteration         2.82e-12  (limit 1e-09)
 consumption_savings (cubic): dsgelab 40 + 4 Newton steps (0.19 s), time iteration 114; dolo 40 + 4 Newton steps, time iteration 114
  ok   consumption_savings, cubic: against dolo at the grid points  3.32e-12  (limit 1e-08)
  ok   consumption_savings, cubic: against dolo between them       3.17e-12  (limit 1e-08)
  ok   consumption_savings, cubic: against time iteration          2.85e-12  (limit 1e-09)
 rbc_mc (linear): dsgelab 0 + 10 Newton steps (0.21 s), time iteration 351; dolo 0 + 11 Newton steps, time iteration 351
  ok   rbc_mc, linear: against dolo at the grid points             1.96e-10  (limit 1e-08)
  ok   rbc_mc, linear: against dolo between them                   1.92e-10  (limit 1e-08)
  ok   rbc_mc, linear: against time iteration                      3.56e-11  (limit 1e-09)
 rbc_mc (cubic): dsgelab 0 + 10 Newton steps (0.87 s), time iteration 343; dolo 0 + 11 Newton steps, time iteration 343
  ok   rbc_mc, cubic: against dolo at the grid points              1.70e-10  (limit 1e-08)
  ok   rbc_mc, cubic: against dolo between them                    1.67e-10  (limit 1e-08)
  ok   rbc_mc, cubic: against time iteration                       3.37e-11  (limit 1e-09)
2. improved time iteration on stochastic growth, against the closed form
  linear: largest relative error of c with 10, 20, 40, 80 capital points: 1.37e-03, 3.35e-04, 8.35e-05, 2.08e-05
  linear: iterations, improved and time iteration: 4 and 17, 4 and 16, 4 and 17, 4 and 17; 1.02 s and 4.43 s in all
  ok   linear, 80 points: largest relative error of c              2.08e-05  (limit 3e-05)
  ok   linear: error ratio when the spacing halves, distance from 4  9.26e-02  (limit 5e-01)
  ok   linear: largest difference from time iteration              4.93e-12  (limit 1e-09)
  cubic: largest relative error of c with 10, 20, 40, 80 capital points: 4.97e-04, 1.14e-04, 2.74e-05, 6.72e-06
  cubic: iterations, improved and time iteration: 4 and 17, 4 and 17, 4 and 17, 4 and 17; 1.40 s and 4.38 s in all
  ok   cubic, 80 points: largest relative error of c               6.72e-06  (limit 1e-05)
  ok   cubic: error ratio when the spacing halves, distance from 4  3.57e-01  (limit 5e-01)
  ok   cubic: largest difference from time iteration               2.15e-12  (limit 1e-09)
3. expectations iteration on stochastic growth, against the closed form
  linear: largest relative error of c with 10, 20, 40, 80 capital points: 1.37e-03, 3.35e-04, 8.35e-05, 2.08e-05
  linear: iterations, expectations and time iteration: 28 and 19, 29 and 19, 29 and 19, 29 and 19; 2.24 s and 6.46 s in all
  ok   linear, 80 points: largest relative error of c              2.08e-05  (limit 3e-05)
  ok   linear: error ratio when the spacing halves, distance from 4  9.26e-02  (limit 5e-01)
  ok   linear: largest difference from time iteration              3.68e-12  (limit 1e-09)
  cubic: largest relative error of c with 10, 20, 40, 80 capital points: 4.97e-04, 1.14e-04, 2.74e-05, 6.72e-06
  cubic: iterations, expectations and time iteration: 26 and 19, 26 and 19, 26 and 19, 26 and 19; 2.43 s and 7.48 s in all
  ok   cubic, 80 points: largest relative error of c               6.72e-06  (limit 1e-05)
  ok   cubic: error ratio when the spacing halves, distance from 4  3.57e-01  (limit 5e-01)
  ok   cubic: largest difference from time iteration               4.69e-12  (limit 1e-09)
  without damping, the first steps move the policy by 6.3e-01, 8.9e-01, 1.5e+00, 2.0e+00, 3.7e+00

all checks passed
```

`run_dolo_iti.py` solves the three models of the time-iteration comparison, on the same grids, with
dolo 0.4.9.20's `improved_time_iteration`, and with its `time_iteration` from the same start for
the iteration counts, and writes `out_time_iteration/dolo_iti.json`. `compare_improved_time_iteration.py`
solves them with `dsgelab`'s `improved_time_iteration` from the same start. Both apply Newton's
method to the conditions of every grid point at once, with the derivative through next period's
policy. dolo solves each Newton step by a Neumann series in that derivative and `dsgelab` by
GMRES, both preconditioned by the blocks of the derivative with next period's policy held fixed.
dolo stops when the largest residual is below 1e-11, and `dsgelab` when a step moves no control by
more than 1e-11, so the policies agree within 3.7e-10 relative to the largest control, and
`dsgelab`'s agrees with its own time iteration run to 1e-12 within 4.4e-11. The Newton steps
number 4 to 10, where time iteration takes 114 to 361 iterations from the same start.

On `rbc` and `rbc_mc` both routines start from the controls of dolo's calibration, a constant
policy. On `consumption_savings`, as published, dolo's routine does not converge from the
calibration within its 1000 steps, and from `c = min(w, 1)` it takes 899 steps with linear
interpolation and does not converge with cubic. `run_dolo_iti.py` therefore starts it from its
own time iteration stopped at a tolerance of 1e-3, 40 iterations from `min(w, 1)`, and
`dsgelab` starts in the same way. dolo is not edited, and the starting policies are passed as
`dr0`. From `min(w, 1)` directly, `dsgelab`'s routine takes 9 steps on this grid.

The second part repeats the closed-form check of the time-iteration comparison with improved time
iteration, which takes 4 Newton steps on every grid where time iteration takes 16 or 17. The third
solves the same model with the expectation in the Euler equation declared as the jump `z`, by
`expectations_iteration`. dolo's documentation describes this iteration, as parameterized
expectations for its `fgh` models, and dolo 0.4.9.20 has no code for it, so the references are
the closed form and time iteration on the same grid. Both methods reproduce time iteration's
policy within 1e-11 and the closed form's errors to the digits printed. Expectations iteration
takes 26 to 29 iterations where time iteration takes 19, and a third of the time, because no
integral is evaluated inside the solve for the controls. Without damping it diverges, as its
docstring explains: a change in consumption moves the next update by about
`-(1 - alpha beta) / (alpha beta)`, -1.8, times as much. The numbers come from a run on
6 November 2026, with dolo 0.4.9.20 on Python 3.11 and `dsgelab` on Python 3.13.

## Value function iteration against QuantEcon and dolo

```bash
bash dolo/setup.sh                            # if not run already
dolo/.venv/bin/python dolo/run_dolo_vfi.py    # about 10 seconds
python compare_value_iteration.py             # about 25 seconds; needs quantecon 0.13.1
```

```
1. QuantEcon 0.13.1 DiscreteDP, policy iteration
 3 productivity nodes, 150 capital points: dsgelab 9 iterations, 0.26 s; QuantEcon 7 iterations, 0.05 s
  ok   3 x 150: largest difference in the value                    5.82e-12  (limit 2e-11)
       the policies are the same at every point
 7 productivity nodes, 200 capital points: dsgelab 11 iterations, 1.34 s; QuantEcon 8 iterations, 0.95 s
  ok   7 x 200: largest difference in the value                    2.24e-13  (limit 2e-11)
       the policies are the same at every point
2. dolo 0.4.9.20 (interpolation 2.2.7), value_iteration
 rbc_mc (cubic): dsgelab 16 iterations from zero, 7.89 s; dolo 3 iterations from its time-iteration policy, 4.14 s
  ok   rbc_mc: value, relative difference at the grid points       1.27e-11  (limit 1e-09)
  ok   rbc_mc: value, relative difference between them             1.27e-11  (limit 1e-09)
  ok   rbc_mc: controls, largest difference at the grid points     5.69e-07  (limit 1e-05)
  ok   rbc_mc: controls, largest difference between them           5.90e-07  (limit 1e-05)
       gradient of the objective at dsgelab's controls, interior points: max 7.0e-09
       gradient of the objective at dolo's controls, interior points: max 1.1e-05
3. stochastic growth with log utility and full depreciation, against the closed form
  linear: largest error of V at the grid points with 10, 20, 40, 80 capital points: 7.10e-03, 1.53e-03, 4.51e-04, 1.03e-04
  ok   linear, 80 points: largest error of V                       1.03e-04  (limit 2e-04)
  ok   linear: smallest error ratio when the spacing halves            3.40  (at least 3)
  cubic: largest error of V at the grid points with 10, 20, 40, 80 capital points: 1.55e-04, 1.12e-06, 6.11e-08, 3.19e-09
  ok   cubic, 80 points: largest error of V                        3.19e-09  (limit 1e-08)
  ok   cubic: smallest error ratio when the spacing halves            18.32  (at least 3)
4. three capital stocks on Smolyak grids, against the closed form
  level  nodes  iterations   V at nodes   V off the grid   policy    seconds
      1      7          11     1.58e-02         1.78e-02  5.29e-03       0.7
      2     25          11     2.70e-04         3.02e-04  1.50e-04       6.2
      3     69          11     6.41e-08         7.32e-08  8.89e-08      14.9
  ok   level 3: largest error of V at the nodes                    6.41e-08  (limit 2e-07)
  ok   level 3: largest error of V at 600 random points            7.32e-08  (limit 2e-07)
  ok   level 3: largest error of the saving shares                 8.89e-08  (limit 2e-07)
  ok   smallest error ratio of V from one level to the next           58.31  (at least 30)

all checks passed
```

QuantEcon's `DiscreteDP` takes a problem with finitely many states and actions. The comparison
builds it from the stochastic growth model with capital on a grid, next period's capital chosen
on the same grid and productivity a Rouwenhorst chain, and solves it by policy iteration, which is
exact. `dsgelab` solves the same problem with `choices=`: since next period's capital is always a
grid point, the linear interpolation of the value is exact there, and the two problems coincide.
The values then differ by less than `beta tol / (1 - beta)`, the bound that the stopping rule
gives, and the policies are the same at every point. QuantEcon 0.13.1 is installed in the
environment that runs the comparison, as for `compare_markov.py`.

dolo's `value_iteration` needs a model with a felicity, and among the three models `setup.sh`
downloads only `rbc_mc.yaml` declares one. As published, dolo 0.4.9.20 does not solve it.
`value_iteration` reads the bounds under names this version of dolo no longer uses, and once those
are aliased it starts from the calibrated controls, the same at every grid point. At low capital
they leave consumption negative, the evaluation of that policy diverges, and the run ends in NaN
after eight iterations. `run_dolo_vfi.py` aliases the bounds and starts from dolo's own
time-iteration policy for the same model. It also tightens the tolerances of the L-BFGS-B
maximization and of the policy evaluation, for which dolo exposes no option. The evaluation stops
when the value moves by less than 1e-8, which keeps the iteration from meeting a tighter
tolerance. With those changes dolo converges in three iterations. `dsgelab` starts from a zero value and needs 16,
each a maximization followed by 200 of Howard's steps.

The felicity in `rbc_mc.yaml` is `c^(1-sigma)/(1-sigma)` with `sigma = 5`, whose value for negative
consumption is also finite. dolo's maximizer starts near the solution and never reaches that
region. `dsgelab`'s searches the whole box of bounds and finds it, so the program in the comparison
sets the reward to `-inf` where consumption is not positive, which is the problem `rbc_mc.yaml`
describes. Both interpolate the value by cubic splines with a zero second derivative at the ends
and linear extrapolation. The values agree within 1.3e-11, relative to their size, at the grid
points and at the centres of the cells. The controls agree within 6e-7. The gradient of the
maximized objective, computed by central differences of the same objective at both policies, is
7e-9 at `dsgelab`'s and 1e-5 at dolo's. dolo's L-BFGS-B estimates the gradient by forward
differences, whose error is of that size, so the difference in the controls comes from dolo's
side.

The closed form is the third reference. With log utility and full depreciation the value is
`E + alpha/(1 - alpha beta) log k + H(z)`, with `H` the solution of a linear system in the chain's
transition matrix. At the grid points the linear interpolant's error falls by 3.4 to 4.6 each time
the spacing halves, and the cubic one's by 18 to 140. The test suite also checks the closed form
with a shock integrated by Gauss–Hermite on a two-dimensional grid, a closed form with labour and
two controls, which exercises the projected Newton method, and the discrete-choice problem against
an exact policy iteration written in the test. The numbers come from a run on 31 October 2026,
with dolo 0.4.9.20 on Python 3.11 and `dsgelab` on Python 3.12.

The fourth part puts the closed form on a `SmolyakGrid`. With three capital stocks,
`c + k_1' + k_2' + k_3' = z k_1^a_1 k_2^a_2 k_3^a_3`, and the shares of output saved in each stock as
the controls, the policy is `s_j = beta a_j` and the value is `sum_j b_j log k_j + h(z)`, with
`b_j = a_j / (1 - beta sum a)`. The three controls go through the projected Newton method. The
value is a polynomial of the capital stocks on the grid and a sum of logarithms in the closed
form, so its error falls with the level as the Chebyshev approximation of a logarithm does: by 58
from level 1 to 2 and by 4200 from level 2 to 3. Section 4 ran on 5 November 2026, on Python 3.13.

## The endogenous grid method against dolo, time iteration and the closed form

```bash
bash dolo/setup.sh                            # if not run already
dolo/.venv/bin/python dolo/run_dolo_egm.py    # about 10 seconds
python compare_egm.py                         # about 15 seconds
```

```
1. dolo 0.4.9.20 (interpolation 2.2.7), egm, tolerance 1e-12
 iid: dsgelab 125 iterations, 0.04 s; dolo 155 iterations, 1.33 s
  ok   iid: largest difference at 199 points of cash on hand       9.11e-13  (limit 1e-10)
       Euler errors between the endogenous points, relative to c: max 1.14e-03, mean 2.52e-05
 markov: dsgelab 310 iterations, 0.09 s; dolo 355 iterations, 0.85 s
  ok   markov: largest difference at 199 points of cash on hand    1.06e-11  (limit 1e-10)
       Euler errors between the endogenous points, relative to c: max 6.46e-03, mean 3.30e-05
2. time_iteration on the same model
  iid: on dolo's grids, 100 points of cash on hand up to 4 and 100 poststates up to 4: largest difference in c 3.36e-04, mean 4.82e-05
  ok   iid, dolo's grids: largest difference in c                  3.36e-04  (limit 5e-03)
  iid: both grids up to 8, with 100, 200, 400, 800, 1600 points, at cash on hand up to 4:
       largest difference in c 1.28e-03, 2.99e-04, 1.65e-04, 5.54e-05, 3.68e-05
       mean difference in c    1.19e-04, 3.93e-05, 1.34e-05, 1.32e-06, 4.25e-07
  ok   iid, up to 8, 1600 points: largest difference in c          3.68e-05  (limit 1e-04)
  ok   iid, up to 8, 1600 points: mean difference in c             4.25e-07  (limit 5e-06)
  markov: on dolo's grids, 100 points of cash on hand up to 4 and 100 poststates up to 4: largest difference in c 2.78e-03, mean 4.54e-04
  ok   markov, dolo's grids: largest difference in c               2.78e-03  (limit 5e-03)
  markov: both grids up to 8, with 100, 200, 400, 800, 1600 points, at cash on hand up to 4:
       largest difference in c 3.25e-03, 1.59e-03, 4.54e-04, 4.28e-04, 8.59e-05
       mean difference in c    4.38e-04, 9.18e-05, 8.14e-06, 1.26e-05, 1.94e-06
  ok   markov, up to 8, 1600 points: largest difference in c       8.59e-05  (limit 1e-04)
  ok   markov, up to 8, 1600 points: mean difference in c          1.94e-06  (limit 5e-06)
3. stochastic growth with log utility and full depreciation, against the closed form
  linear: largest relative error of c with 10, 20, 40, 80 poststates: 1.03e-02, 2.55e-03, 6.48e-04, 1.63e-04
         error ratios when the spacing halves: 4.02, 3.94, 3.99
  ok   linear, 80 points: largest relative error of c              1.63e-04  (limit 2e-04)
  ok   linear: error ratio when the spacing halves, distance from 4  6.01e-02  (limit 5e-01)
  cubic: largest relative error of c with 10, 20, 40, 80 poststates: 4.82e-03, 1.01e-03, 2.29e-04, 5.41e-05
         error ratios when the spacing halves: 4.76, 4.42, 4.23
  ok   cubic, 80 points: largest relative error of c               5.41e-05  (limit 1e-04)
  ok   cubic: error ratio when the spacing halves, distance from 4  7.56e-01  (limit 1e+00)

all checks passed
```

`consumption_savings_iid.yaml`, which `setup.sh` already downloads for the time-iteration
comparison, declares the `expectation`, `half_transition`, `direct_response_egm` and
`reverse_state` blocks that dolo's `egm` needs, and `compare_egm.py` writes the same four
functions as an `EGMProgram`, with `post_decision`, `a = w - c`, added. The second case replaces
the iid income of the file with log income an AR(1) with persistence 0.9, which dolo discretizes
as a three-state Rouwenhorst chain. dolo 0.4.9.20's `egm` runs on both as published. It prints a
debugging line at every step of its inner loop, which `run_dolo_egm.py` silences. It also returns
the policy resampled on the model's grid of cash on hand and interpolated there by cubic splines,
which between the grid's points is not the policy of the method, so the runner asks for 199
points, the 100 of the file and the centres between them, and reads the policy only there. Both
implementations interpolate linearly between the points of the endogenous grid, extrapolate
linearly and hold consumption between 0 and cash on hand, so they iterate on the same fixed point
and agree within 9e-13 with iid income and 1.1e-11 with the chain, relative to the largest
consumption. The counts of iterations differ because the two start from different policies, dolo
from its calibrated consumption, 0.9 at every point, and `dsgelab` from 0.9 times cash on hand,
and because dolo stops on the sum of the changes in the controls and in the endogenous grid and
`dsgelab` on the larger of the two.

Time iteration solves the same problem with a different discretisation, so the comparison with it
measures the two interpolation errors, not a shared fixed point. On a fixed grid of cash on hand,
the kink where the borrowing limit stops binding falls inside a cell, and the linear interpolant's
error there falls only with the spacing, not with its square. The endogenous grid has a point at
that kink, the one the lowest poststate, zero saving, leads to. On dolo's grids, with 100 points
up to 4, the two policies differ by up to 3.4e-4 with iid income and 2.8e-3 with the chain. With
the chain, next period's cash on hand exceeds 4 when income is high, time iteration extrapolates
its policy there, and the difference stays at that size as the grids are refined. With both grids
extended to 8 and compared up to 4, the mean difference falls from 1.2e-4 to 4.3e-7 with iid income
and from 4.4e-4 to 1.9e-6 with the chain, from 100 to 1600 points. The largest difference falls to
3.7e-5 and 8.6e-5, unevenly, because it depends on where the kink falls within time iteration's
cell.

The closed form is stochastic growth with log utility and full depreciation, written with next
period's capital as the poststate, `c = 1/E[beta alpha exp(z') k'^(alpha-1)/c']` and
`k = ((k' + c)/exp(z))^(1/alpha)`. Its policy, `c = (1 - alpha beta) exp(z) k^alpha`, is not
linear in `k`, and the relative error at the centres between the endogenous points falls by 3.9
to 4.0 each time the spacing halves with linear interpolation. With cubic interpolation the error
is smaller by a factor of 3 and falls by 4.2 to 4.8, as the square of the spacing, because the
natural spline's zero second derivative at the ends is not the policy's. The test suite also
checks a linear policy, consumption and saving with no income, which `egm` reproduces within
1e-11, the agreement with time iteration, the borrowing limit, the refinement from a coarse
solution, the simulation and the evaluation point by point. The numbers come from a run on
3 November 2026, with dolo 0.4.9.20 on Python 3.11 and `dsgelab` on Python 3.13.



## Sparse grids against Tasmanian and the closed form

```bash
bash tasmanian/setup.sh                             # Python 3.12, Tasmanian 8.2 built from source
tasmanian/.venv/bin/python tasmanian/run_tasmanian.py   # about a second
python compare_smolyak.py                           # about 3 minutes on an idle machine
```

```
1. Tasmanian 8.2, global grids of type 'level' on the clenshaw-curtis rule
  levels               box         points     nodes   weights
  [0]                  own              1   2.2e-16   0.0e+00
  [1]                  [-1, 1]          3   0.0e+00   1.6e-16
  [2]                  own              5   1.7e-16   1.9e-15
  [3]                  [-1, 1]          9   2.2e-16   7.8e-16
  [4]                  own             17   8.9e-16   5.3e-15
  [5]                  [-1, 1]         33   3.3e-16   6.8e-15
  [6]                  own             65   2.2e-16   2.2e-14
  [0, 0]               own              1   0.0e+00   0.0e+00
  [1, 1]               [-1, 1]          5   0.0e+00   2.2e-16
  [2, 2]               own             13   1.1e-16   1.4e-15
  [3, 3]               [-1, 1]         29   2.2e-16   1.3e-15
  [4, 4]               own             65   4.4e-16   2.1e-14
  [5, 5]               [-1, 1]        145   3.3e-16   2.0e-14
  [0, 0, 0]            own              1   0.0e+00   0.0e+00
  [1, 1, 1]            [-1, 1]          7   0.0e+00   3.3e-16
  [2, 2, 2]            own             25   2.2e-16   1.7e-15
  [3, 3, 3]            [-1, 1]         69   2.2e-16   2.2e-15
  [4, 4, 4]            own            177   8.9e-16   1.1e-14
  [0, 0, 0, 0]         own              1   5.6e-17   0.0e+00
  [1, 1, 1, 1]         [-1, 1]          9   0.0e+00   4.4e-16
  [2, 2, 2, 2]         own             41   3.3e-16   2.4e-15
  [3, 3, 3, 3]         [-1, 1]        137   2.2e-16   2.7e-15
  [4, 4, 4, 4]         own            401   4.4e-16   1.3e-14
  [0, 0, 0, 0, 0]      own              1   0.0e+00   0.0e+00
  [1, 1, 1, 1, 1]      [-1, 1]         11   0.0e+00   8.9e-16
  [2, 2, 2, 2, 2]      own             61   2.2e-16   2.7e-15
  [3, 3, 3, 3, 3]      [-1, 1]        241   2.2e-16   4.4e-15
  [4, 4, 4, 4, 4]      own            801   6.7e-16   1.7e-14
  [0, 0, 0, 0, 0, 0]   own              1   2.2e-16   0.0e+00
  [1, 1, 1, 1, 1, 1]   [-1, 1]         13   0.0e+00   1.0e-15
  [2, 2, 2, 2, 2, 2]   own             85   2.2e-16   2.5e-15
  [3, 3, 3, 3, 3, 3]   [-1, 1]        389   2.2e-16   8.4e-15
  [4, 4, 4, 4, 4, 4]   own           1457   4.4e-16   1.3e-14
  [3, 1]               own             19   4.4e-16   1.6e-15
  [1, 4]               own             35   1.1e-16   2.9e-15
  [3, 1, 2]            own             51   4.4e-16   3.0e-15
  [0, 2, 1, 3]         own             51   4.4e-16   2.3e-15
  [3, 2, 1, 1, 0]      own             97   3.3e-16   2.6e-15
  [3, 3, 2, 2, 1, 1]   own            329   3.9e-16   6.0e-15
  [4, 1, 1, 1, 1, 1]   own            627   4.4e-16   8.4e-15
  40 grids, 0 with a different number of points
  ok   largest difference in a node                                    8.88e-16  (limit 1e-14)
  ok   largest difference in an interpolation weight                   2.25e-14  (limit 1e-12)
2. stochastic growth with several capital stocks, log utility and full depreciation, against the closed form
   relative error of k_j' at 1000 points in the box; seconds in brackets
  4 states (3 capital stocks), isotropic
                levels points          projection      time iteration  difference
                     1      9   5.46e-03 [  1.6]   5.46e-03 [  5.8]     4.8e-11
                     2     41   1.23e-04 [  1.3]   1.23e-04 [  7.7]     4.8e-11
                     3    137   1.81e-06 [  0.8]   1.81e-06 [  8.5]     4.3e-11
                     4    401   2.09e-08 [  6.7]   2.11e-08 [ 33.5]     6.0e-10
  ok   4 states, level 4: largest relative error, projection           2.09e-08  (limit 3e-08)
  ok   4 states, level 4: largest relative error, time iteration       2.11e-08  (limit 3e-08)
  ok   4 states: smallest fall of the error per level, inverted        2.26e-02  (limit 0.05)
  ok   4 states: projection against time iteration                     5.97e-10  (limit 1e-08)
  5 states (4 capital stocks), isotropic
                levels points          projection      time iteration  difference
                     1     11   4.31e-03 [  1.6]   4.31e-03 [  4.7]     4.9e-11
                     2     61   1.11e-04 [  0.4]   1.11e-04 [  4.5]     4.8e-11
                     3    241   1.92e-06 [  1.4]   1.92e-06 [  7.6]     1.3e-10
                     4    801   2.51e-08 [ 17.4]   2.54e-08 [ 66.9]     7.7e-10
  ok   5 states, level 4: largest relative error, projection           2.51e-08  (limit 6e-08)
  ok   5 states, level 4: largest relative error, time iteration       2.54e-08  (limit 6e-08)
  ok   5 states: smallest fall of the error per level, inverted        2.57e-02  (limit 0.05)
  ok   5 states: projection against time iteration                     7.68e-10  (limit 1e-08)
  6 states (5 capital stocks), isotropic
                levels points          projection      time iteration  difference
                     1     13   4.36e-03 [  0.8]   4.36e-03 [  9.7]     1.5e-10
                     2     85   1.17e-04 [  1.0]   1.17e-04 [ 14.0]     1.4e-10
                     3    389   2.21e-06 [  7.6]   2.21e-06 [ 24.9]     1.3e-10
                     4   1457   3.38e-08 [155.3]   3.33e-08 [223.5]     2.3e-09
  ok   6 states, level 4: largest relative error, projection           3.38e-08  (limit 6e-08)
  ok   6 states, level 4: largest relative error, time iteration       3.33e-08  (limit 6e-08)
  ok   6 states: smallest fall of the error per level, inverted        2.68e-02  (limit 0.05)
  ok   6 states: projection against time iteration                     2.27e-09  (limit 1e-08)
  6 states, anisotropic: level 1 along a
                levels points          projection      time iteration  difference
    [3, 3, 3, 3, 3, 1]    363   2.21e-06 [  5.0]   2.21e-06 [ 12.9]     1.2e-10
    [4, 4, 4, 4, 4, 1]   1283   3.34e-08 [ 35.9]   3.33e-08 [ 90.9]     1.2e-10
    levels [3, 3, 3, 3, 3, 1]: 363 points against 389 at isotropic level 3
  ok   6 states, levels [3, 3, 3, 3, 3, 1]: error over isotropic level 3's  1.00e+00  (limit 1.05)
    levels [4, 4, 4, 4, 4, 1]: 1283 points against 1457 at isotropic level 4
  ok   6 states, levels [4, 4, 4, 4, 4, 1]: error over isotropic level 4's  9.90e-01  (limit 1.05)

all checks passed
```

Tasmanian is distributed on PyPI as source only, and its build calls `cmake` by name. `setup.sh`
installs CMake 4.4.4 and scikit-build 0.19.1 from PyPI into the environment, puts its `bin` first
on the `PATH` and builds Tasmanian 8.2 without build isolation, which would otherwise fetch the
build tools again, unpinned. On 3 November 2026 the build took a minute and a half on macOS 27
(arm64) with Apple clang 21.

A global grid of type `level` on Tasmanian's Clenshaw–Curtis rule has, along one axis, one node at
level 0 and `2^l + 1` at level `l`, the Chebyshev extrema, and takes the multi-indices whose levels
sum to at most the depth. That is JMMV's isotropic grid with Tasmanian's level `l` as JMMV's
`i - 1`. `liLevelLimits` caps the level along each axis, which gives JMMV's anisotropic grid.
Tasmanian builds the interpolant as Smolyak's combination of tensor Lagrange interpolants, and
`SmolyakGrid` by solving for the coefficients of JMMV's Chebyshev basis. On nested
nodes both spaces are the span of the same tensor products, so the interpolant is the same
polynomial and the weights agree to rounding, 2e-14 at most. The cases are 33 isotropic grids, in
one to six dimensions at levels 0 to 6 (up to 1457 points), and 7 anisotropic ones with a level 0
or 1 on some axes. The odd levels are on the canonical box `[-1, 1]^d` and the others on boxes of
their own, set with `setDomainTransform`. The test suite checks the interpolant without Tasmanian,
against a combination of tensor Lagrange interpolants written in the test.

The second part solves a planner's problem with `n` capital stocks, log utility and full
depreciation, `c + k_1' + ... + k_n' = a k_1^alpha_1 ... k_n^alpha_n` with `log a` an AR(1), whose
policy is `k_j' = beta alpha_j y` in any dimension. The shares `alpha_j` sum to 0.36, `beta` is
0.96 and the persistence 0.9. Each capital stock varies by 25 per cent on either side of its
steady state and productivity by 3 per cent, and the shock's standard deviation is 0.01, integrated
by five Gauss–Hermite nodes. The policy does not depend on the shock's variance, so the error
against the closed form is the approximation's alone. It falls by a factor of 37 to 87 per level,
and projection and time iteration, which solve the same collocation equations, agree within
2.3e-9. Time iteration stops when the policy moves by less than 1e-11, and the difference is
larger than that away from the nodes, most at level 4. The policy is linear in productivity. With
six states, level 1 along productivity removes 26 of 389 points at level 3 and 174 of 1457 at
level 4, and the error changes by less than 1 per cent. Both methods start from the first-order
policy, and time iteration then takes 13 to 15 iterations at levels 1 to 3. The numbers come from
a run on 5 November 2026, with Tasmanian 8.2 on Python 3.12 and `dsgelab` on Python 3.13, on a
machine shared with other jobs (load average about 60). The seconds in brackets are therefore
larger than on an idle machine: alone, on 3 November, the six-state grid of level 4 took 15 seconds by projection
and 31 by time iteration, and the anisotropic one 13 and 30.
## Heterogeneous agents against sequence-jacobian

```bash
bash sequence_jacobian/setup.sh                                  # if not run already
sequence_jacobian/.venv/bin/python sequence_jacobian/run_ssj.py  # about 60 seconds
python compare_heterogeneous.py                                  # about 50 seconds
```

```
sequence-jacobian 1.0.0 (numpy 1.26.4, numba 0.68.1), T = 300
1. Krusell-Smith (krusell_smith.ipynb), 7 income states by 500 asset points
  ok   grids and income chain (largest: e_grid)                      4.30e-10  (limit 1e-09)
  ok   steady state: unknowns (largest: beta)                        5.43e-12  (limit 1e-09)
  ok   steady state: policies (largest: Va)                          4.32e-10  (limit 1e-09)
  ok   steady state: distribution D, relative to its largest mass    5.08e-10  (limit 1e-07)
  ok   steady state: aggregates (largest: C)                         8.26e-11  (limit 1e-09)
       goods market (Walras), at asset market clearing: dsgelab -1.9e-13, sequence-jacobian 7.6e-11
  ok   household Jacobians, forward differences (largest: A/r)       2.27e-10  (limit 1e-07)
  ok   general-equilibrium Jacobians G (largest: K/Z)                6.48e-11  (limit 1e-07)
  ok   linear impulse responses to Z (largest: A)                    4.13e-11  (limit 1e-07)
  ok   nonlinear transition after Z (largest: K)                     1.84e-10  (limit 1e-08)
  ok   household Jacobians, central differences (largest: A/r)       2.24e-10  (limit 1e-07)
       seconds, dsgelab / sequence-jacobian: steady state 2.8 / 2.8, household Jacobians 0.3 / 2.0, G 0.4 / 2.6, nonlinear 1.3 / 7.8
2. One-asset HANK (hank.ipynb), 7 income states by 500 asset points
  ok   grids and income chain (largest: e_grid)                      4.30e-10  (limit 1e-09)
  ok   steady state: unknowns (largest: vphi)                        1.84e-10  (limit 1e-09)
  ok   steady state: policies (largest: Va)                          7.59e-10  (limit 1e-09)
  ok   steady state: distribution D, relative to its largest mass    2.11e-10  (limit 1e-07)
  ok   steady state: aggregates (largest: N)                         5.33e-11  (limit 1e-09)
       goods market (Walras), at asset and labour market clearing: dsgelab 1.2e-14, sequence-jacobian 1.1e-11
  ok   household Jacobians, forward differences (largest: A/r)       2.46e-10  (limit 1e-07)
  ok   general-equilibrium Jacobians G (largest: w/rstar)            1.63e-10  (limit 1e-07)
  ok   linear impulse responses to rstar (largest: w)                9.73e-11  (limit 1e-07)
  ok   nonlinear transition after rstar (largest: w)                 1.37e-10  (limit 1e-08)
  ok   household Jacobians, central differences (largest: A/r)       2.45e-10  (limit 1e-07)
       seconds, dsgelab / sequence-jacobian: steady state 5.5 / 6.6, household Jacobians 1.4 / 4.0, G 1.5 / 6.6, nonlinear 3.3 / 14.3
3. Labor search (labor_search.ipynb), 2 employment states by 5 productivity states by 100 asset points, with the Markov matrix moved by f and s
  ok   grids, income and the employment chain (largest: y)           1.78e-10  (limit 1e-09)
  ok   Markov matrix: Kronecker product of sequence-jacobian's two   0.00e+00  (limit 1e-09)
  ok   steady state: policies (largest: Va)                          3.61e-10  (limit 1e-09)
  ok   steady state: distribution D, relative to its largest mass    8.06e-10  (limit 1e-07)
  ok   steady state: aggregates (largest: B)                         4.14e-10  (limit 1e-09)
  ok   household Jacobians, forward differences (largest: A/r)       2.87e-10  (limit 1e-07)
  ok   household's linear response to s, the notebook's (largest: A)  2.19e-10  (limit 1e-07)
  ok   general-equilibrium Jacobians G (largest: r/s)                8.23e-11  (limit 1e-07)
  ok   linear impulse responses to s (largest: C)                    4.37e-11  (limit 1e-07)
  ok   nonlinear transition after s (largest: C)                     6.75e-11  (limit 1e-08)
  ok   household Jacobians, central differences (largest: A/r)       2.89e-10  (limit 1e-07)
  ok   distribution of the steady state with s = 0.12, relative to its largest mass  5.83e-10  (limit 1e-07)
  ok   transition from the distribution with s = 0.12 (largest: r)   5.99e-10  (limit 1e-08)
       seconds, dsgelab / sequence-jacobian: steady state 0.2 / 0.4, household Jacobians 0.8 / 2.1, G 0.9 / 3.0, nonlinear 2.0 / 7.0
4. dsgelab against itself
       Krusell-Smith household: fake news against direct, 10 columns, steps 1e-05, 1e-06, 1e-07: 1.4e-07, 2.7e-09, 3.2e-08
  ok   Krusell-Smith household: fake news against direct, step 1e-07  3.25e-08  (limit 1e-07)
       HANK household: fake news against direct, 10 columns, steps 1e-05, 1e-06, 1e-07: 1.5e-06, 6.7e-07, 4.1e-08
  ok   HANK household: fake news against direct, step 1e-07          4.08e-08  (limit 1e-07)
       Krusell-Smith, largest gap between the nonlinear and the linear response, for shocks of 1, 0.1, 0.01 per cent: 4.41e-05, 4.51e-07, 3.97e-09
  ok   Krusell-Smith: smallest ratio of the gaps, shock / 10            97.90  (at least 90)
       labor-search household (moving Markov): fake news against direct, 10 columns, steps 1e-03, 1e-04, 1e-05: 1.5e-05, 2.3e-09, 1.6e-09
  ok   labor-search household (moving Markov): fake news against direct, step 1e-05  1.64e-09  (limit 1e-07)
       labor search, largest gap between the nonlinear and the linear response, for shocks to s of 1, 0.1, 0.01 points: 1.01e-04, 1.02e-06, 1.02e-08
  ok   labor search: smallest ratio of the gaps, shock / 10             99.38  (at least 90)
  ok   labor search: path from an initial distribution against forward simulation  3.07e-13  (limit 1e-12)

all checks passed
```

The three models are sequence-jacobian's own, from `notebooks/krusell_smith.ipynb`,
`notebooks/hank.ipynb` and `notebooks/labor_search.ipynb` at its v1.0.0 tag, which `setup.sh`
downloads and checks against their sha256 sums. `run_ssj.py` copies the model code from the notebooks and checks that each piece
appears in them verbatim before running it. The household of the HANK model is
`hetblocks.hh_labor.hh`, part of the package. `compare_heterogeneous.py` writes the same models
for `dsgelab`, with the same calibration, unknowns and targets, the household's endogenous grid
method again in numpy, and the constrained households' labour supply solved by the same Newton
iteration on `log u'(c)`. Three settings differ from the notebooks. In both packages the
household's backward iteration stops when the policy moves by less than 1e-12, where
sequence-jacobian's default is 1e-8. sequence-jacobian's forward iteration stops at 1e-14,
where its default is 1e-10, and `dsgelab` finds the stationary distribution by a sparse solve. The
nonlinear transitions stop at 1e-11, where the notebooks use 1e-8. At the defaults the steady
states would differ by the tolerance, some 1e-7 in the policies, and hide the comparison.

The grids differ by 4.3e-10. sequence-jacobian computes the income chain's stationary distribution
by iterating to 1e-11, and `dsgelab` by GTH elimination, which is exact to rounding. The
normalisation of income to a mean of one carries that difference to every node. Everything else
agrees within 7.6e-10, relative to the largest value compared. The goods market, which neither
model targets, clears to 1.9e-13 and 1.2e-14 in `dsgelab`, against 7.6e-11 and 1.1e-11 in
sequence-jacobian, whose distribution stops at its tolerance.

sequence-jacobian differentiates one backward step by forward differences with step 1e-4, and
its Jacobians differ from the derivative by some 1e-5 of their size. The comparison of the
Jacobians, of `G` and of the impulse responses therefore runs `dsgelab` with the same
differences (`twosided=False`, `h=1e-4`), and a last line compares the central differences that
`dsgelab` uses by default with sequence-jacobian's own (`twosided=True`). Some responses are zero
in general equilibrium, such as assets in the HANK model, where the asset market clears at a fixed
supply of bonds. The differences in each group are therefore taken relative to the largest value
in the group.

Section 3 is the model whose Markov matrix moves along a path. `labor_search.ipynb` is
sequence-jacobian's tutorial on that feature: the household's employment status follows a
two-state chain whose matrix is a hetinput of the job-finding rate `f` and the separation rate
`s`, beside a productivity chain of 5 nodes, on 100 asset points. sequence-jacobian keeps the two
chains on two axes, and `dsgelab` on one chain of 10 nodes whose matrix is their Kronecker product,
which agrees exactly with the product of sequence-jacobian's two. The notebook stops at the
household, with its linear response to `s`. `run_ssj.py` adds the smallest general equilibrium,
which is not in the notebook and is written identically in both packages: a fixed supply of bonds
equal to the household's assets at the notebook's steady state, the interest rate as the unknown
and the bond market as the target, and a shock to `s` of 0.02 0.8^t. The closure has no lags, so
that a transition from another steady state's distribution, sequence-jacobian's `ss_initial` and
`dsgelab`'s `initial`, moves only the household. The steady state agrees within 8.1e-10, the
household Jacobians with respect to `f`, `s`, `r` and `w` within 2.9e-10, `G` within 8.3e-11, the
linear and nonlinear responses within 2.2e-10, and the return to the steady state from the
distribution with `s = 0.12` within 6.0e-10.

Section 4 checks the fake-news algorithm against brute force, the nonlinear path differentiated
for a shock at each of ten dates, both by central differences with the same step. The two are
different finite differences of the same derivative, and the policy has kinks where the borrowing
limit starts to bind and at every grid point. Their gap falls with the step, to 3e-8 and 4e-8 at
1e-7, below which rounding takes over. For the labor-search household, with respect to `f` and
`s`, the gap is 1.6e-9 at 1e-5 and rises to 1.7e-8 and 1.6e-7 at 1e-6 and 1e-7, where rounding
takes over sooner. In Krusell–Smith and in labor search, the gap between the nonlinear transition
and the linear response falls by at least 98 each time the shock falls by ten, as the square of
the shock. The household's path from the distribution with `s = 0.12`, at the steady state's
prices, agrees within 3.1e-13 with the steady state's transition matrix applied for 300 periods. The test suite checks the same on a grid of 3 by 40 points, together with Walras's law
along the path, the stationarity of the distribution, the simple blocks' exact Jacobians and a
solved block against the closed form of a forward-looking equation.

`dsgelab` has no numba, and its times depend on the load of the machine. The numbers come from a
run on 6 November 2026, with sequence-jacobian 1.0.0 on Python 3.11 and `dsgelab` on Python 3.13.
The differences of sections 1 and 2 are those of the first run, on 5 November.

## Markov regime switching: published solutions, statsmodels and the exact likelihood

The roadmap names RISE as the reference. RISE runs only in MATLAB. At commit
`9bdcd74cc72a2bf537ce3182f5883b8942b007b5` of `github.com/jmaih/RISE_toolbox` (27 September 2026)
the toolbox has 1652 `.p` files, MATLAB P-code, and the `.m` file next to each holds only its help
text. The `v2019` tag, kept for the earlier public version, is distributed the same way, with 1013
`.p` files. GNU Octave 11.3 does not execute P-code: `rise_startup` runs the comment-only `.m` file
and does nothing. The validation therefore uses the references that exist without MATLAB, the
solutions Foerster, Rubio-Ramírez, Waggoner and Zha print, closed forms, statsmodels' Hamilton
filter and the exact likelihood over every regime path. RISE's example folder holds the model files
of those papers (`frwz_nk.rs`, `rbcs.rs`, `simple.rs`), which fix the timing and the
parameterizations used here.

```bash
python compare_switching.py                   # about 20 seconds
```

```
1. Foerster et al. (2013), New Keynesian model: all solutions of the quadratic system
  psi(2) = 0.9: perturbed ['mu'], 402 starting points, 402 converged, 9 distinct solutions, 3 real
  ok   psi(2) = 0.9: number of solutions found minus nine              0.00e+00  (limit 0e+00)
       entries within a unit of the last printed digit: largest error 3.74e-04, 0.54 of that unit
  ok   psi(2) = 0.9: entries off by more than a unit of the last digit  0.00e+00  (limit 0e+00)
       real solutions, stability radius: 0.4532 (stable), 1.0160, 1.5354
  ok   psi(2) = 0.9: stable solutions found minus 1                    0.00e+00  (limit 0e+00)
  psi(2) = 0.7: perturbed ['mu'], 402 starting points, 402 converged, 9 distinct solutions, 3 real
  ok   psi(2) = 0.7: number of solutions found minus nine              0.00e+00  (limit 0e+00)
       entries within a unit of the last printed digit: largest error 1.94e-04, 0.55 of that unit
       solution (4): printed -0.7000+4.6327j, computed -0.666951+4.632663j
       solution (5): printed -0.7000-4.6327j, computed -0.666951-4.632663j
  ok   psi(2) = 0.7: entries off by more than a unit of the last digit  2.00e+00  (limit 2e+00)
       real solutions, stability radius: 0.4683 (stable), 0.9527 (stable), 1.5897
  ok   psi(2) = 0.7: stable solutions found minus 2                    0.00e+00  (limit 0e+00)

2. Foerster et al. (2016), Markov-switching RBC: printed first-order coefficients
  steady state c = 2.082588, k = 22.15038, z = 1.00706 (printed 2.08259, 22.1504, 1.007); perturbed ['mu']
  regime 1:
      0.04056   0.12648   0.00917   0.00005
      0.96920  -2.14061  -0.15521  -0.37201
      0.00000   0.10000   0.00725   0.01846
  regime 2:
      0.04056   0.00000   0.02687  -0.09681
      0.96920  -0.00000  -0.46499   0.92271
      0.00000   0.00000   0.02175  -0.04102
  ok   23 coefficients truncated to four decimals minus printed        0.00e+00  (limit 1e-12)
       constant of c in regime 1: computed 0.0000459, printed 4.9e-05
  ok   entries off by more than a unit of the last digit               1.00e+00  (limit 1e+00)
  complex Newton from 403 starting points: 4 solutions, 2 real, stability radii 0.9394, 1.0848, 2.6052, 2.6052
  ok   solutions found minus four (Groebner bases give four)           0.00e+00  (limit 0e+00)
  ok   stable solutions found minus one                                0.00e+00  (limit 0e+00)

3. Foerster et al. (2016), section 5: phi(s) pi_t + sigma(s) eps_t = E_t pi_{t+1}
  pi_t = -0.080000 eps_t in regime 1, -0.625000 in regime 2 (printed -0.08 and -0.625)
  ok   solution minus -sigma/phi                                       1.39e-17  (limit 1e-14)
  with the lag of inflation as a state, coefficients on it: (0.0000, 0.0000) radius 0.000; (-0.0000, 1.1294) radius 1.084; (1.2681, 0.9056) radius 1.539; (1.3158, -0.0000) radius 1.645
  ok   solutions found minus four                                      0.00e+00  (limit 0e+00)
  ok   stable solutions found minus one                                0.00e+00  (limit 0e+00)

4. Fisherian model with an AR(1) real rate: (diag(alpha) - rho T) c = 1
  c = 0.976532929599, 2.036336109008
  ok   c minus the closed form                                         4.44e-16  (limit 1e-14)

5. Identical regimes: the Klein solution of the constant-parameter model
  ok   rbc: F, P, Q minus Klein's, and constants                       1.78e-15  (limit 1e-13)
  ok   nk3: F, P, Q minus Klein's, and constants                       4.44e-16  (limit 1e-13)

all checks passed
```

Foerster et al. (2013, Atlanta Fed working paper 2013-1, section 6.2) solve a New Keynesian model in
which the drift of technology switches and enters the Euler equation at `t+1`, and the response to
inflation in the Taylor rule switches at `t`. The drift moves the steady state and is perturbed;
the response does not and keeps its value in each regime, which `switching_system` finds from the
residuals. They obtain every solution of the quadratic system with Gröbner bases and print the
nine, for `psi(2) = 0.9` and for `psi(2) = 0.7`. Newton's method in complex arithmetic, from the
Klein starting points and 400 random complex ones, finds the same nine in both cases, and the same
stable ones: solution (1) for the first, (1) and (3) for the second. Each printed entry is compared
in units of its last printed digit. The two that differ by more than one unit are the conjugate
pair (4)–(5) of the second table, whose real part is printed as -0.7000 and is -0.66695 here; the
other five entries of the same pair agree.

For the RBC of Foerster et al. (2016, section 6.1), the paper prints the coefficients of the stable
solution on `(k_{t-1}, z_{t-1}, eps_t, 1)` in its own timing, in which the predetermined variables
are dated `t` and depend on `eps_t`. `dsgelab`'s states are `(k_{t-1}, z_t)`, so the comparison
substitutes the law of motion of `z_t` into the solution. The printed values are truncated to four
decimals (0.018459 is printed as 0.0184, 0.12648 as 0.1264), and 23 of the 24 then agree exactly.
The constant of consumption in regime 1, printed to six decimals as 0.000049, is 0.0000459 here.
Complex Newton finds the four solutions the paper reports, one stable.

```bash
python compare_kim.py --prepare               # simulates the series, writes out_kim/input.json
bash statsmodels/setup.sh                     # if not run already
statsmodels/.venv/bin/python statsmodels/run_statsmodels.py
python compare_kim.py                         # about 2 minutes, most of it maximum likelihood
```

```
1. Hamilton's filter: statsmodels 0.14.4
  hamilton, 300 observations: log likelihood -213.4494648247, statsmodels -213.4494648247
  ok   hamilton: log likelihood minus statsmodels'                     2.84e-14  (limit 1e-09)
  ok   hamilton: filtered Pr(s=0) minus statsmodels'                   4.44e-16  (limit 1e-11)
  ok   hamilton: smoothed Pr(s=0) minus statsmodels'                   2.00e-15  (limit 1e-11)
       the smoothed probability puts the simulated regime above one half at 100.0% of the dates
  ok   hamilton: Kim minus the exact likelihood, 14 observations       8.88e-16  (limit 1e-11)
  maximum likelihood: dsgelab -207.12820351, statsmodels -207.12820351
       p[0->0]        0.980276     0.980276
       p[1->0]        0.095455     0.095455
       const[0]       2.142501     2.142501
       const[1]       6.134101     6.134104
       sigma2[0]      0.128807     0.128807
       sigma2[1]      0.751839     0.751836
  ok   hamilton: maximized log likelihood minus statsmodels'           8.29e-10  (limit 1e-06)
  ok   hamilton: implied reduced form minus statsmodels' estimate      2.57e-06  (limit 1e-04)
  ar, 299 observations after the first: log likelihood -473.0062781287, statsmodels -473.0062781296
  ok   ar: log likelihood minus statsmodels'                           9.44e-10  (limit 1e-08)
  ok   ar: filtered Pr(s=0) minus statsmodels'                         6.12e-11  (limit 1e-09)
  ok   ar: smoothed Pr(s=0) minus statsmodels'                         4.01e-11  (limit 1e-09)

2. Kim's filter against the exact likelihood, all regime paths enumerated
  ar, mu and phi switching, measurement s.d. 0.1: largest |Kim - exact| over 10 samples of 14 dates 2.12e-06 (|log likelihood| up to 30.5)
  ar, mu and phi switching, measurement s.d. 0.5: largest |Kim - exact| over 10 samples of 14 dates 7.63e-03 (|log likelihood| up to 28.8)
  ar, mu and phi switching, measurement s.d. 2.0: largest |Kim - exact| over 10 samples of 14 dates 1.84e-02 (|log likelihood| up to 30.7)
  ok   ar: largest |Kim - exact|                                       1.84e-02  (limit 5e-02)
  Foerster et al. (2013) NK, inflation observed: largest |Kim - exact| over 5 samples of 12 dates 8.62e-04 (|log likelihood| up to 61.1)
  ok   NK: largest |Kim - exact|                                       8.62e-04  (limit 5e-03)

all checks passed
```

Kim's filter collapses, at each date, the Gaussians that end in the same regime. It is exact in two
cases, which statsmodels 0.14.4 computes by Hamilton's filter. In the first, the state is i.i.d.,
so that observed inflation is `mu[s_t] + sigma[s_t] e_t`, statsmodels' `MarkovRegression`. In the
second, an AR(1) is observed without error around a switching mean, `MarkovAutoregression`
(Hamilton, 1989): given the regime, the data reveal the state. statsmodels conditions on the first
observation, and the comparison does too, through `init=` and a missing first value. The
smoothed probabilities come from the filter's joint probabilities of consecutive regimes, which
makes them exact in both cases. Smoothing the marginal probabilities alone, as Kim's (1994)
formula is often written, differs from statsmodels by up to 0.8 in the autoregression, because
there the observation at `t` depends on the regime at `t-1`. Maximum likelihood estimates the
Fisherian model's own parameters (the two inflation targets, the two responses to inflation and the
staying probabilities), which map one to one into statsmodels' reduced form, and reaches the same
maximum. The remaining differences, near 1e-9, come from the central differences behind the
solution's coefficients.

The exact likelihood is the mixture of the Kalman filters of all `2^T` regime paths, which
`enumerated_likelihood` computes for samples of 12 and 14 dates. Against it, the collapse costs
2e-6 in the log likelihood when the measurement error is small and up to 0.018 when it is large.
In the New Keynesian model of Foerster et al. (2013) with inflation observed, the error is 9e-4.
With output and the interest rate both observed, the regime is identified at almost every date and
the collapse is exact to 1e-13. The test suite also checks that identical regimes reproduce the
Kalman filter, Hamilton's filter written out in the test, the exact likelihood with missing data,
and posterior mode and Metropolis–Hastings on switching parameters and a transition probability.
The numbers come from a run on 5 November 2026, with `dsgelab` on Python 3.13 and statsmodels on
Python 3.12.
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
