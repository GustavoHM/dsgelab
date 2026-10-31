# Roadmap

*Versão em português: [ROADMAP.pt-BR.md](ROADMAP.pt-BR.md).*

Plan revised on 11 October 2026 and updated on 17 October 2026. The goal is for `dsgelab` to be
**as complete as possible**. **Version 1.0** covers what the reference tools offer (Dynare,
MacroModelling.jl, gEconpy, dolo, sequence-jacobian) and fills the gaps that applied models
exposed in the objects the package already has. The **public release** happens at **version 0.5**.

This plan replaces the one of 25 September, which left out full estimation, global methods,
heterogeneous agents, the declared analytic steady state and lags and leads in text equations. For
these items, the earlier plan pointed to the tools that already have them. On 11 October that
decision was reversed, and all of these items entered the plan. The arguments for the exclusion
became **design requirements** of the corresponding items, recorded next to each one.

The item-by-item comparison with the other tools is in [`docs/comparison.md`](docs/comparison.md).
`cgelab` moves through the same milestones in its own roadmap, so "0.3" means the 0.3 of each.

## Where we stand

Version 0.5.0, at `https://github.com/GustavoHM/dsgelab`, brings together milestones 0.2 to 0.4
and the publication work of milestone 0.5. It was released on 18 October 2026, on PyPI and on
Zenodo (DOI [10.5281/zenodo.23456475](https://doi.org/10.5281/zenodo.23456475)).
The steady state is found with positivity and an outer anchor, accepted in closed form after a
check, and computed with calibration equations and homotopy. The first-order solution is Klein/QZ
with Blanchard–Kahn enforced and, optionally, Dynare's unit-root criterion, and it gives impulse
responses, moments and variance decomposition. At second order the package computes the risk
correction, the ergodic mean in closed form, pruned simulation and the generalised impulse
response, and at third order the cubic block and the risk slope of the rules. Perfect foresight
takes an announced path of parameters and of innovations, and expectation errors. The package
also handles **occasionally binding constraints** (OccBin) and the **extended path**,
deterministic and stochastic. **Models in levels** are covered, with declared trends and the
balanced growth path, and so is Chebyshev projection. For data there are historical
decomposition and conditional forecasting, joint calibration with a rank diagnostic, the Kalman
filter and smoother, and Bayesian and maximum-likelihood estimation. Models can be written as
text equations, with exact derivatives up to the Hessian and lags and leads of any order. They
can also be built by deriving FOCs and assembling several agents, or read from a `.mod`, and
they are exported to Dynare and MacroModelling.jl. Optimal policy covers the Ramsey planner,
discretionary policy, optimal simple rules and welfare, and model diagnostics complete the set.

Validation against Dynare covers three models ported by hand, nine `.mod` files read without manual
conversion (five of them from Dynare's own `examples/` folder) and, from Dynare's test suite, the
tests for optimal policy, perfect foresight, estimation, OccBin, the extended path, balanced growth
and diagnostics. The calibration equations are validated against MacroModelling.jl. Details are in
[`validation/`](validation/README.md).

Version 0.7.0, released on 30 October 2026 on PyPI and on Zenodo (DOI
[10.5281/zenodo.23825088](https://doi.org/10.5281/zenodo.23825088)), brings milestones 0.6 and 0.7.
At third order the decision rule is complete in Dynare's form, with every cubic and risk block, the
stochastic steady state and sparse tensors for large models, and the solution has analytic
derivatives with respect to the parameters at all three orders. Estimation gains HMC and NUTS on the
gradient of the Kalman likelihood, and particle and inversion filters at second and third order. For
identification and data there are local identification, GMM, SMM and IRF matching, local
projections, filtered moments, calibration restrictions on IRFs and moments, global sensitivity
(stability mapping, Monte Carlo filtering and Sobol indices), out-of-sample forecast evaluation,
mixed-frequency data and plots. Each is validated against an independent reference: Dynare for most,
SALib for the Sobol indices, R's `forecast::dm.test` for the Diebold–Mariano test, and the
population result of Plagborg-Møller and Wolf (2021) for the local projections. The details are in
[`validation/`](validation/README.md).

Milestone 0.8 comes out in parts. Version 0.7.1, released on 30 October 2026 on PyPI and on Zenodo
(DOI [10.5281/zenodo.23827669](https://doi.org/10.5281/zenodo.23827669)), brings the first: the
discretisation of an AR(1) into a Markov chain by Tauchen's and Rouwenhorst's methods, validated
against QuantEcon, against the same probabilities in 40-digit arithmetic and against Rouwenhorst's
closed form. Version 0.7.2, released on 31 October 2026, adds time iteration on a Cartesian grid,
with bounds on the controls imposed by complementarity. It agrees with dolo's `time_iteration` on
three models from the dolo repository and with the closed form of stochastic growth.

## What counts as done

An item leaves the roadmap only when it meets the six conditions below. The package's standard
is that each feature is validated against an independent reference where one exists (another
implementation, a closed form or a published result), and a few features rest on internal tests
only. A new feature without that check would weaken the standard for the whole package.

1. Implemented with a public API in `__all__`.
2. **Validated against an independent reference**, which may be another implementation (Dynare
   and MacroModelling.jl so far, and dolo, sequence-jacobian and RISE for the items planned
   against them), a closed-form solution or a published result. The validation lives in a
   script that anyone can run.
3. Tested in the package's suite, with no dependence on external tools. Whatever needs Dynare or
   Julia lives in `validation/`.
4. Documented in the API reference, with a runnable example on the site, checked by
   `tools/check_site_examples.py`.
5. The matching row updated in `docs/comparison.md`.
6. An entry in [`CHANGELOG.md`](CHANGELOG.md).

## Milestones

### 0.2 — optimal policy, welfare and the gaps in the objects that already exist

- [x] **Ramsey under commitment**, timeless perspective, and its steady state
  (`ramsey_model`, `ramsey_steady_state`). Validated against the targeting rule and the closed form
  of Galí (2015, ch. 5) and against Dynare's `ramsey_model` on three models, including Dynare's
  `tests/optimal_policy/nk_ramsey_model.mod`, unchanged. On the three models the steady state
  agrees to 5e-11 and the IRFs of the variables and the multipliers to 5e-13.
- [x] **Optimal simple rules** (`osr`), on a quadratic loss in unconditional moments, with a warning
  when the optimum lies on the boundary. On Dynare's `osr_example.mod` the loss matches Dynare's to
  2.6e-16 at Dynare's optimum, and `dsgelab` finds an optimum with a lower loss, which Dynare itself
  confirms. The rules can also be optimised on welfare (`objective=`), as Dynare's `osr` with
  `planner_objective` does.
- [x] **Discretionary policy** (`discretionary_policy`), linear-quadratic, by Dennis's algorithm.
  Validated against Galí's closed form and against Dynare's `dennis_1.mod`.
- [x] **Welfare** (`welfare`, `consumption_equivalent`). The objective function is stored in the
  model. Welfare is computed at second order, conditional and unconditional, and for Ramsey with the
  lagged multipliers at zero. The proportional equivalent is taken in a named variable, without
  assuming log utility or that the variable is consumption. It matches Dynare's
  `evaluate_planner_objective` to 1.4e-12 for Ramsey and to 6.1e-10 under discretionary policy.
- [x] **Second-order generalised impulse responses** (`SecondOrderSolution.irf`), from pruned
  simulations with and without the shock, because the response depends on the state in which the
  shock arrives and on its sign.
- [x] **Sequence of innovations in perfect foresight** (`perfect_foresight(..., shocks=)`),
  validated against Dynare's `perfect_foresight_solver` on its `rbc_det1`–`3` to 1e-13.
- [x] **Perfect foresight with expectation errors** (`perfect_foresight_with_expectation_errors`),
  validated against Dynare's solver on `pfwee.mod` and `pfwee_constant_sim_length.mod` to 3e-12.
- [x] **Ramsey and discretionary policy from the `.mod`**. `from_mod` reads `planner_objective`,
  `ramsey_model`, `discretionary_policy`, `osr_params` and `optim_weights`, and Dynare's
  optimal-policy tests run without manual translation.

### 0.3 — estimation, forecasting and decomposition

Validated against Dynare on the `fs2000.mod` from its `examples/` folder, with its priors and data
(`validation/compare_estimation.py`).

- [x] Kalman filter on the linear solution, posterior mode and Metropolis–Hastings
  (`kalman_filter`, `posterior_mode`, `metropolis_hastings`). Against Dynare, the likelihood agrees
  to 6e-12 and the prior density to 4e-13, the mode has a kernel greater than or equal to Dynare's,
  and the Metropolis means lie within 3 Monte Carlo standard errors of Dynare's (largest 2.9).
- [x] **Historical decomposition** (`shock_decomposition`) and **conditional forecast**
  (`conditional_forecast`, with `controlled`), against Dynare's `shock_decomposition` (5e-15) and
  `conditional_forecast` (`tests/conditional_forecasts`, 1.6e-13).
- [x] **Observation equations** and **measurement error** (`state_space`, `EstimationProblem`),
  with observables in levels or in logs (`loglinear` with `logdata`), measurement standard
  deviations fixed or estimated (`"stderr y"`), and missing data.
- [x] **Kalman smoother** (`kalman_smoother`) and **exact diffuse initialisation** for unit roots,
  against Dynare's `diffuse_filter`, with the likelihood to 2.5e-12 and the smoother to 1e-15.
- [x] **Maximum likelihood** (`maximum_likelihood`), with standard errors from the Hessian.
- [x] **Full Bayesian estimation**, with Dynare's priors by mean and standard deviation (`prior`),
  SMC (`smc`), R-hat and effective sample size, and the prior predictive check with the share
  of draws with a determinate equilibrium (`prior_predictive`). Validated against closed forms
  (the normal conjugate case) and against Dynare's `estimation` on `fs2000.mod`.
- [x] **Marginal likelihood** by Laplace (the formula reproduces Dynare's number to 2e-13 with
  Dynare's covariance) and by modified harmonic mean, and `model_probabilities` for comparing
  models.
- [x] **Real-time historical decomposition** (`realtime_shock_decomposition`, 5e-15 against
  Dynare) and **forecasts with bands** (`forecast`, with state and shock uncertainty, against
  Dynare's `forecast`, and `posterior_forecast`, with parameter uncertainty).

### 0.4 — nonlinearities and the declaration of the model

Completed in version 0.4.0, on 11 October 2026.

- [x] **Import `.mod`** (`from_mod`), with macros, model-local variables, `steady_state()`, and
  leads and lags of any order. Validated on nine `.mod` files, read unchanged, against Dynare.
- [x] **Arbitrary lags and leads**, in the `.mod` and in `from_equations`, with every auxiliary
  variable listed in `auxiliary`. *On both paths, as the design requirement asks, what is
  predetermined is still declared in a single place. The lag of a jump variable is a new state with
  its own equation, and no auxiliary variable is created without a warning.*
- [x] **Occasionally binding constraints** by the piecewise-linear method of Guerrieri and
  Iacoviello, on the same Klein solver (`OccbinModel`, `occbin_simulate`). Checked against Dynare's
  `+occbin` with one and with two constraints (`borrcon.mod`, `dynrbc.mod`), to 2e-15, with the same
  regime histories.
- [x] **Extended path**, deterministic and stochastic (sparse and full trees, hybrid second-order
  correction), in `extended_path`. Checked against Dynare's `extended_path` in seven variants of
  `burnside.mod`, to 1.1e-11. *The hybrid correction of order higher than 2 is still missing, since
  it requires order-4 perturbation (0.7).*
- [x] **Declared analytic steady state**, including the `steady_state_model` of a `.mod`
  (`analytic_steady_state`, `ModFile.steady_state(analytic=True)`). *The design requirement is
  met, since* the analytic block is checked against the residual of the dynamic model before it is
  used, and refused if it does not zero it, as in Dynare.
- [x] **Calibration equations inside the model**, in the manner of MacroModelling.jl, matching
  MacroModelling to 4e-15.
- [x] **Several agents in one equilibrium** (`assemble`). The limits are those of `Problem`, and it
  is not a block language.
- [x] **Homotopy and continuation** for hard steady states, with an adaptive step, and the
  `homotopy_setup` of a `.mod`.
- [x] **Nonstationary models**. `trend_var`, `log_trend_var`, `var(deflator=)` and
  `var(log_deflator=)` are detrended as Dynare does, with the balanced-growth check.
  `balanced_growth` solves the `+bgp` problem, with residuals equal to Dynare's to 2.2e-16.
  *Declaring trends in `from_equations` is still missing, and for now the route is the `.mod`.*
- [x] **Model diagnostics** (`diagnose`), Dynare's `model_diagnostics`, with the same report.

### 0.5 — public release (released on 18 October 2026)

No new features. The items below are conditions for publishing.

**Prerequisites**

- [x] **Rename.** `dsgekit` was already in use on PyPI by another package, *"A Python toolkit for
  DSGE models"*. The package is now `dsgelab`, which was available on PyPI on 11 October 2026.
  The name still has to be reserved with a first upload to TestPyPI and to PyPI.
- [x] **One canonical repository.** `https://github.com/GustavoHM/dsgelab`. The copies of the
  research repository have been archived.
- [x] **Language.** The site has an English version at the root and a Portuguese version in
  `site/pt-br/`, like `cgelab`'s, and the examples run in both (12 October 2026). The code, the
  documentation (`docs/`, with the whitepaper PDF regenerated), `validation/README.md`,
  `CITATION.cff` and the messages that the scripts in `validation/` print are in English (13 to 16
  October). The roadmap exists in both languages, with `ROADMAP.md` in English and
  `ROADMAP.pt-BR.md` in Portuguese.
- [x] **Validation reproducible by third parties.** The published evidence, which consists of
  Dynare's `.mod` files, the closed forms and Heutel (2012), can be reproduced by anyone. As of
  12 October 2026, `validation/run_all.sh` runs everything at once. Versions are pinned (Octave
  packages by sha256 in `setup_octave_packages.sh`, MacroModelling.jl in
  `validation/macromodelling/`, `.mod` files at Dynare tags), and Heutel's files come from the
  public copy at Georgia State University (`validation/heutel/fetch.sh`), byte-identical in all
  six `.m` files. Everything passes on Dynare 7.2, Octave 11.3.0, statistics 2.0.0 and
  datatypes 1.5.0. On 16 October, the only comparison made on a model that is not public (the
  author's unpublished applied model, a DSGE with 71 variables) was marked on the site as not
  reproducible from the repository. The README and `validation/README.md` mark it the same way.
- [x] **Security review before the public release.**
- [x] **Author's review of every public text.** Nothing is published (the repository made public,
  the site, a GitHub release, PyPI, Zenodo) before the author has reviewed all the texts that go
  with it, namely the README, CHANGELOG, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, the issue
  and pull-request templates, `CITATION.cff`, the description in `pyproject.toml`, the site in both
  languages, `docs/` and `validation/README.md`. All 25 texts were approved on 18 October 2026,
  after a final review of their claims against the code.

**Publication engineering**

- [x] Continuous integration, with tests on Python 3.11 to 3.14, on Linux, macOS and Windows, and
  `check_site_examples.py`. *Written* (`tests.yml`): the suite runs against the installed wheel
  in 12 combinations, the sdist is tested from its own tree, and the site examples run. The
  first run on GitHub, on 13 October 2026, passed in all 17 jobs, Windows included. Python 3.10
  stopped being supported on 12 October 2026 because SciPy 1.15.3, the last version for it,
  does not load on macOS 27 (`__thread_bss`).
- [x] External validations as a scheduled CI job. *Written* (`validation.yml`, weekly and on
  demand). It runs on macOS, with Octave and Dynare from Homebrew, the pinned Octave packages, and
  Julia 1.12.7 with MacroModelling.jl. On Ubuntu the Dynare in the distribution's repositories is
  behind the current version, and compiling it every week would add its build time to each run.
  Homebrew's Dynare follows the current version, and so the job serves as an early warning of the
  risk described under Risks. The first run was
  triggered by hand on 16 October 2026, and its 25 steps passed in 28 minutes.
- [x] Installation tested from the wheel, with 148 tests against the installed wheel on Python
  3.14 on macOS (`DSGELAB_TEST_INSTALLED=1`), and the sdist with the files its own tests read.
- [x] `python -m build`, `twine check`, publication by trusted publishing, TestPyPI first
  (`release.yml`, triggered by a `v*` tag). The workflow checks the tag against the version,
  publishes to TestPyPI, runs the suite against what TestPyPI serves, and publishes to PyPI only
  after the required reviewer approves. Versions 0.5.0, 0.7.0 and 0.7.1 went out this way, the first
  on 18 October 2026 and the other two on 30 October.
- [x] Site on GitHub Pages, published from `site/` by `.github/workflows/pages.yml` on every
  change, with the links to `docs/` pointed at GitHub (`https://gustavohm.github.io/dsgelab/`).
- [x] DOI on Zenodo for each release, and `CITATION.cff` with ORCID and DOI. Version 0.7.1 is
  10.5281/zenodo.23827669 (0.7.0 is 10.5281/zenodo.23825088, 0.5.0 is 10.5281/zenodo.23456475), and
  the concept DOI 10.5281/zenodo.23456474, which resolves to the latest release, is in
  `CITATION.cff`.
- [x] `CONTRIBUTING.md`, code of conduct, issue and pull-request templates.
- [x] `py.typed` and type annotations on the public API, with no public function or method left
  unannotated and no new error reported by `mypy`. The lazy-import wrappers in `__init__.py`
  (`from_equations`, `assemble`, `to_dynare`...) accept `*args, **kwargs`, and the type checker
  does not check their arguments.
- [x] A known-limitations section in the README.

**Version policy from 0.5 on.** Versions follow semantic versioning. Before 1.0 a minor version
may still change the API, but every removal or incompatible change is preceded by a deprecation
warning that lasts at least one minor version. The public API is what is in `__all__`.

### 0.6 — identification, moments and data

- [x] **Local identification** (Iskrev; Komunjer and Ng; also Qu and Tkachenko, and the
  brute-force collinearity search), with `identification` and `ModFile.identification`.
  Validated against Dynare's `+identification` on three models (`kim2`, `as2007`, `fs2000`):
  the same verdicts in all three criteria and derivatives within 2e-8 (18 October 2026).
  Dynare's repetition over draws from the prior (`prior_mc`), its identification strength
  from the information matrix and its plots are left out.
- [x] **Global sensitivity**, covering the stability and determinacy region in the parameter space
  and Monte Carlo filtering. The reference is Dynare's `+gsa`. First-order and total Sobol
  indices for impulse responses and moments, validated against SALib. The distributions share
  their interface with `cgelab.sensitivity` (`Uniform`, `Normal`, `Triangular`), without either
  package importing the other. `stability_map` sorts draws into determinate, indeterminate,
  explosive and unsolved; `monte_carlo_filtering` (also `StabilityMap.filtering` and
  `RestrictionSample.filtering`) gives the Smirnov test of each parameter and the correlations
  inside each group; `sobol` and `solution_outputs` give the indices of IRFs and moments.
  Validated on Dynare 7.2's `tests/gsa/ls2003a.mod` over three samples of 1024 draws
  (30 October 2026): every draw has the class Dynare's solver gives it, and on Dynare's groups
  the Smirnov statistics agree exactly, their p-values within `3.3e-16` and the correlations'
  within `1.7e-13`. The sensitivity toolbox's own lists differ in 37 draws of the explosive
  sample: in 36 because it sorts the eigenvalues by signed value when they are all real, and in
  one rank failure that its eigenvalue rule cannot see. By modulus, its rule gives `dsgelab`'s
  classes everywhere but at the rank failures. Sobol's design and
  indices agree with SALib 1.6 within `8.9e-16` on the Ishigami function and on IRFs and moments
  of the same model. Dynare's reduced-form mapping (`redform`), RMSE filtering (`rmse`) and
  Morris screening are left out.
- [x] **Method of moments** (GMM, SMM and IRF matching). GMM and SMM work at first and
  pruned second order; IRF matching works at first order. Validated against Dynare 7.2's `+mom`
  in eight first-order GMM/SMM cases and two IRF-matching cases (24 October 2026). Second-order
  means and autocovariances agree on `example1` and `example2` to `1e-14`. The imported models,
  including their auxiliary variables, agree to `2.93e-8` and `1.59e-6` in means and to
  `6.08e-9` and `2.87e-8` in second moments.
  IRF matching is frequentist and accepts external targets, including local projections or
  event studies, with their joint covariance. `local_projection` supplies observed-shock IRFs
  and a joint HAC covariance; on simulated AR(1) data it recovers the model's IRFs in population
  (Plagborg-Møller and Wolf, 2021). Bayesian IRF matching and MATLAB transformation files are
  left out; Python transformations are accepted explicitly.
- [x] **Out-of-sample forecast evaluation**: recursive estimation over an expanding window,
  RMSE against a random walk, the Diebold–Mariano test, and the probability integral transform
  and coverage of the predictive density. The references are the forecast comparisons of Smets
  and Wouters (2007) and of Del Negro and Schorfheide (2013). The comparison with a BVAR waits
  on the BVAR of 0.9. `forecast_evaluation` re-estimates at each origin over an expanding or a
  rolling window, and `ForecastEvaluation` gives the RMSE, its ratio to the random walk's,
  `diebold_mariano` with the Harvey–Leybourne–Newbold correction, the PIT, the coverage and the
  log score. Validated against Dynare 7.2's recursive estimation of `fs2000` over nine samples
  (30 October 2026): at Dynare's modes the forecasts agree within `2.6e-15` and the 90% bands
  within `6.3e-9`. Re-estimated by `dsgelab`, every mode has a kernel at least as high as
  Dynare's and lies within `0.003` posterior standard deviations of it. The test agrees with R's
  `forecast::dm.test` within `3.1e-15` in 48 cases. The predictive density is conditional on the
  point estimate, as Dynare's after the mode; parameter uncertainty at every origin is left out.
- [x] **Calibration restrictions on IRFs and moments** (`irf_calibration`, `moment_calibration`),
  with `check_restrictions` at one point and `sample_restrictions` over draws from the prior, and
  the `.mod` blocks read by `from_mod`. Validated against Dynare 7.2 on its
  `tests/gsa/ls2003a.mod`, with and without `relative_irf` (30 October 2026): at the calibration
  and on the 512 prior draws of its sensitivity toolbox, the values agree within `1.3e-12`, the
  same draws solve, and every restriction has the same verdict on every draw. Like Dynare's, the
  restrictions work at first order. The Monte Carlo filtering of the parameters behind each
  restriction belongs to the global sensitivity item above.
- [x] **Filtered theoretical moments** (Hodrick–Prescott and band-pass), with `hp_filter` and
  `bandpass` in `moments`, `covariance` and `variance_decomposition`. Validated against Dynare's
  `stoch_simul` on five models, one with a unit root (24 October 2026).
- [x] **Mixed-frequency data**. `aggregation={name: ("mean" | "sum", every)}` in `state_space` and
  `EstimationProblem` stacks the lags of the state, so that an observable can be the mean or the
  sum of a variable over the last `every` periods, with `low_frequency` to place the observations.
  Filter, smoother, likelihood gradient, decompositions and forecasts work on the stacked state.
  Validated against Dynare 7.2 on `fs2000.mod` with output growth observed only as its annual
  mean, written in Dynare as the auxiliary variable a user would add (30 October 2026): the
  likelihood at two points agrees within `8.0e-13`, the smoothed variables within `1.3e-14`, and
  the posterior mode has a kernel at least as high as Dynare's, within `0.003` posterior standard
  deviations. Aggregates of growth rates of levels (the annual growth of a quarterly level) and
  the nonlinear filters are left out.
- [x] **Plots** of impulse responses, simulations, decompositions, forecasts and posterior
  distributions: `plot_irf`, `plot_paths`, `plot_decomposition`, `plot_forecast` and
  `plot_posterior`, returning matplotlib figures, with matplotlib as an optional dependency
  (`dsgelab[plot]`). The two densities the posterior plot draws are Dynare's:
  `prior_density` agrees with `draw_prior_density` on the nine priors of `fs2000` within
  `1.9e-12`, and `posterior_density` with the kernel estimate of `posterior_moments`, bandwidth
  corrected for repeated draws, within `8.9e-16` (30 October 2026). The test suite checks that
  each figure draws the numbers it is given. MCMC diagnostic plots and Dynare's identification
  and sensitivity plots are left out.

### 0.7 — complete third order and derivatives

- [x] **Perturbation core in Dynare's decision-rule form**, with the innovation as an
  argument (`g_u`, `g_uu`, `g_ku`). The risk slope (`g_ssk` and `h_ssk`) is already implemented.
  The second-order innovation blocks and the composed `DecisionRule` are validated against Dynare
  7.2 on `example1` and `example2` (24 October 2026). The cubic blocks `h_uuu`, `h_kuu` and
  `h_kku`, with their compositions, also agree with `ghuuu` and every row of `ghxuu`, `ghxxu` and
  `ghxxx`. The full second-order risk constant also agrees with `ghs2`. `ghxss` and `ghuss` agree
  in every row, with errors below `1.5e-16` on both models.
- [x] **Stochastic steady state** and analytic generalised IRFs. Conditional IRFs of the full
  pruned rule are implemented at third order, including `ghxss`, `ghuss` and the cubic innovation
  blocks, with unit tests for the recursion. Generalized IRFs use stochastic paths with `drop=100`
  and agree with Dynare on `example1` and `example2`. `stochastic_steady_state` solves the fixed
  point of the full unpruned rule and is distinct from the ergodic mean.
- [x] **Third order at the size of the applied model**, with sparse tensors. The symbolic frontend
  supplies exact second and third derivatives with `derivatives="third"`, stored in coordinate form.
  `solve_third_order(sparse=True)` also stores the cubic state coefficients by independent
  symmetric triple and contracts them directly in rule evaluation and risk terms. The iterative
  known term and residual use canonical coordinates, with one equation expanded per matvec and an
  optimized contraction path. A full nonlinear synthetic pipeline with 71 variables, 35 states and
  five shocks has a maximum residual of `3.05e-20` and a peak RSS of 321 MB. Explicit decision-rule
  composition still expands dense arrays. The unpublished applied model itself is not public.
- [x] **Derivatives of the solution with respect to the parameters**, by implicit and numerical
  differentiation, and **gradient-based estimation** (HMC/NUTS). `solution_jacobian` computes
  derivatives of `P`, `F`, `Q` and the steady state. General QZ solutions differentiate the implicit
  saddle-path conditions without another QZ decomposition at perturbed parameters; purely backward
  solutions use the inverse-system formulas. `second_order_jacobian` also computes `g_k`, `g_u`,
  `g_kk`, `g_uu`, `g_ku` and `g_s2`, for model parameters, shock standard deviations and correlations.
  `third_order_jacobian` adds `g_kkk`, `g_kku`, `g_kuu` and `g_uuu`. All are validated against Dynare's
  analytical derivatives; second-order relative errors stay below `0.552%`, and third-order errors
  stay below `0.159%`. Exact model derivatives reduce the cubic decision-rule errors below `3e-13`
  on `example1` and `2e-14` on `example2`. `log_likelihood_gradient` differentiates the first-order
  Kalman likelihood through its recursion, checked against finite differences for model parameters,
  shock standard deviations, correlations, measurement errors, missing data, diffuse starts, log
  transforms and models with jumps. `hmc` uses the posterior gradient with leapfrog integration,
  a mode-based mass matrix and dual-averaging step adaptation during burn-in. `nuts` adds
  multinomial trajectory sampling, the generalised U-turn criterion and divergence diagnostics.
  Both match the exact posterior of an AR(1) coefficient and a long Metropolis–Hastings run in the
  test suite. `posterior_mode` and `maximum_likelihood` use these
  gradients in BFGS.
- [x] **Nonlinear filters** for estimation at second and third order. The bootstrap particle
  filter works at first, pruned second and pruned third order, with systematic resampling, and
  matches Dynare's likelihood on the AR(1) reference within `0.006` (24 October 2026).
  `EstimationProblem(order=2)` exposes the same likelihood to posterior mode and samplers, with
  common random draws. The inversion filter also works at first and pruned second order and
  reproduces Dynare OccBin's shocks, Jacobian and likelihood exactly on the AR(1).
  `EstimationProblem(nonlinear_filter=)` selects either route. Recovered shocks also agree with
  MacroModelling.jl within `0.006` on a nonlinear RBC; its likelihood convention differs, so only
  the shocks are compared. `particle_smoother` works with `store=True` at all three orders.
  Full-rank first-order transitions use the Gaussian backward recursion; nonlinear and
  rank-deficient transitions use the stored particle ancestry, whose effective sample size can
  degenerate over long histories. Inversion smoothing returns the filter's paths without a
  backward pass: they are exact when the data identify every shock, and filtered otherwise. The
  pruned third-order particle filter uses the full rule, including `ghxss` and `ghuss`, and
  reproduces the Kalman likelihood on a linear model.
  `EstimationProblem(order=3)` exposes the likelihood and smoothing.

### 0.8 — global methods

Chebyshev collocation with up to three states already exists and reproduces the closed-form
solution of stochastic growth. Its shocks are i.i.d. and integrated by Gauss–Hermite, and its
policy has no bounds. The milestone is released in parts, one feature per patch version from
0.7.1 to 0.7.5, each published only after it meets the six conditions above. No release changes the
API of an earlier one. Version 0.8.0 closes the milestone, with no new feature of its own.

- [x] **0.7.1 — process discretisation.** `tauchen` and `rouwenhorst` turn an AR(1) into a
  `MarkovChain` with nodes, transition matrix, stationary distribution, moments and simulation.
  Against QuantEcon's `tauchen` and `rouwenhorst` on 120 processes, nodes and matrices agree to
  rounding. Rouwenhorst's chain has the binomial stationary distribution and the mean, variance and
  autocorrelations of the AR(1) (Kopecky and Suen, 2010). Tauchen's probabilities and stationary
  distributions agree with a 40-digit computation within 1.1e-11 and 1.1e-14; QuantEcon, which
  computes the upper tail as `1 - cdf`, is off by 3e-8 on the coarsest chains. The release also
  lengthens the wait in `release.yml` for TestPyPI to serve a new upload, which ran out on 0.7.0,
  and stops pip from reusing the index page it cached.
- [x] **0.7.2 — time iteration.** `time_iteration` solves the model's own residual, the one the
  perturbation uses, with its rows sorted into transitions and equilibrium conditions by what they
  contain at `t+1`. Exogenous states are given by a `MarkovChain` or kept on the grid with the
  shocks integrated by quadrature. `CartesianGrid` interpolates linearly or by a natural cubic
  spline. Bounds on the controls are imposed by complementarity (Fischer–Burmeister), for
  borrowing limits and the zero lower bound, and `TimeIterationSolution` evaluates and simulates
  the policy and reports Euler-equation errors. Against dolo 0.4.9.20's `time_iteration` on three
  of its own models (its RBC with an iid shock, its RBC with Markov productivity and bounds on both
  controls, and consumption and saving with a borrowing limit), on the same grids, the policies
  agree within 1e-9 with linear and with cubic interpolation, and the borrowing limit binds at the
  same grid points. dolo stops each step at a residual of 1e-8, which leaves its policy some 1e-7
  from the fixed point, so the comparison tightens that tolerance. On stochastic growth with log
  utility and full depreciation the error against the closed form falls as the square of the
  spacing, to 7e-6 with 80 capital points and cubic interpolation.
- [ ] **0.7.3 — value function iteration**, with Howard's improvement steps. It needs a
  declaration the model does not yet have: a reward, a discount factor and the transition of the
  endogenous states. Reference: the closed-form value function of stochastic growth with log
  utility and full depreciation, QuantEcon's `DiscreteDP` on the same discretised problem, and
  dolo's `value_iteration`.
- [ ] **0.7.4 — the endogenous grid method** (Carroll, 2006), for problems whose Euler equation
  inverts in closed form given the next period's policy, with the inverse supplied by the user.
  Reference: dolo's `egm` on the consumption–savings model, and the time-iteration solution of
  0.7.2 on the same grid.
- [ ] **0.7.5 — sparse grids.** Smolyak collocation (Judd, Maliar, Maliar and Valero, 2014),
  isotropic and anisotropic, so that projection and time iteration run with more than three
  states. Reference: Tasmanian's Smolyak nodes and interpolation weights, and the closed form of
  stochastic growth with several capital stocks, a planner's problem with log utility and full
  depreciation, whose policy is known in any dimension.
- [ ] **0.8.0** — the milestone complete: the comparison row and the README updated, and the
  release made in the same way as 0.7.0.

dolo 0.4.9.20 only installs on Python 3.11, with pandas below 2.0. On 30 October 2026, in such an
environment, its `time_iteration` solved stochastic growth with log utility and full depreciation
within 1.7e-6 of the closed form. The validation therefore runs dolo in an environment of its own,
created by a setup script with the versions pinned, as is done for Octave's packages. If dolo
stops installing, the fallback is Dolo.jl, through the Julia project already used for
MacroModelling.jl.

### 0.9 — the frontier

- [ ] **Markov regime switching** in solution and estimation. Validation against RISE.
- [ ] **Heterogeneous agents** by the sequence-space Jacobian method, validated against
  sequence-jacobian. *The design requirement, inherited from the earlier exclusion, is that* `vec`
  remains the way to write a finite number of groups, and that a distribution is a different
  algorithm, in a module of its own.
- [ ] **BVAR and DSGE-VAR**, to be checked against Dynare's `+bvar` and `dsge_var`.

### 1.0

Everything in the list above is covered, the API is frozen, and from then on versioning is strictly
semantic.

## Integration with `cgelab`

The one-period equilibrium, calibrated on a social accounting matrix, stays in `cgelab`. The
intertemporal residual, at the points where an agent's decisions depend on expectations of the
future, is solved here. The matrix does
not enter this package, and the dependence is optional, since only `cgelab`'s bridge module imports
`dsgelab` (`pip install "cgelab[dsge]"`). The bridge already exists in `cgelab.bridge`, as a Ramsey
CGE in which investment follows an Euler equation, with announced policy and first-order aggregate
uncertainty. The next extensions are in `cgelab`'s roadmap, in the section *Extensions that call
dsgelab*. The sites of the two packages link to each other.

## Risks

**Scope.** Several items are each the size of a whole package. Heterogeneous agents are the scope
of sequence-jacobian, global methods that of dolo, and regime switching that of RISE. Under this
roadmap's validation standard, each one is months of work. If one of them does not advance, the way
out is to publish first a validated bridge to the tool that already does it (as the Dynare exporter
does today for third order) and keep the item open until the package's own implementation passes
the same validation.

**Dependencies.** Estimation, automatic differentiation and global methods require large
dependencies. The core stays on numpy and scipy, and the rest comes in as optional extras, so that
users who only solve models do not need to install them.

**Validation that depends on external tools.** Almost all of the evidence comes from comparison
with Dynare and MacroModelling.jl. If one of these tools changes behaviour between versions, the
validations break without the package having changed. For that reason the `.mod` files come from
fixed Dynare tags (6.4 for the examples and 7.2 for the tests, with the files used identical to
those of 7.1), and the version used is recorded in the scripts.
