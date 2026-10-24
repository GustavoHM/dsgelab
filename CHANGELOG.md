# Changelog

All notable changes to `dsgelab`. Versions follow [semantic versioning](https://semver.org/);
before 1.0, a minor version may add to the API and a patch version only fixes bugs.

## Unreleased

Work toward milestone 0.6 (identification, moments and data).

### Added

- `identification`: local identification of the free parameters at one point, from the
  distribution of the observables, as Dynare's `identification` command computes it. Three
  criteria: the derivatives of the moments (Iskrev 2010), the Gram matrix of the derivatives
  of the spectrum (Qu and Tkachenko 2012) and the derivatives of a minimal state space
  (Komunjer and Ng 2011). Each reports its rank, the parameters that do not move it, the
  sets of linearly dependent parameters, the multicollinearity of each parameter and the
  weakly identified parameters and pairs. A brute-force search finds, for each parameter,
  the group of others whose effect on the moments best reproduces its own. The free
  parameters can include shock and measurement-error standard deviations (`"stderr e"`).
- `ModFile.identification`: the same for the `estimated_params` of a `.mod`, observing its
  `varobs`, at the prior mean (Dynare's default), the calibration or a given point.
- `tol_rank="auto"`, the default of `identification`: a singular value below ten times the
  estimated error of the derivatives counts as zero. The derivatives are central
  differences with a Richardson step, at a step chosen for each parameter, and two
  Richardson estimates measure their error. With Dynare's `"robust"` tolerance, two
  parameters that enter a model declared in Python only through their product came out
  identified, because the numerical linearisation leaves a singular value near 1e-7.
- `validation/compare_identification.py`: the three criteria against Dynare 7.2 on `kim2`
  and `as2007` from its tests and on `fs2000`. Ranks, sensitive, dependent and weak
  parameters are Dynare's under both tolerances, and the derivatives of the moments and the
  spectrum agree within 2e-8.
- Moments of filtered variables: `hp_filter=lambda` (the Hodrick-Prescott cycle) and
  `bandpass=(shortest, longest)` (the ideal band-pass filter) in `FirstOrderSolution.moments`,
  `covariance` and `variance_decomposition`, as Dynare's `stoch_simul` options of the same
  names. They are computed as Dynare computes them, from the spectral density on a grid of
  frequencies (`grid=512`) and its inverse Fourier transform, and exist with a unit root,
  which both filters remove.
- `moments(...)["autocov"]`: the autocovariance matrices `E[x_t x_{t-j}']`, filtered or not.
- `validation/compare_filtered.py`: filtered moments against Dynare 7.2 on five models,
  one with correlated shocks and one with a unit root, with both filters. Covariances and
  autocorrelations agree within 1e-13, except in example2, where Dynare's steady state
  leaves a residual of 1e-6 and the difference is 5e-6.

## 0.5.0 — 2026-10-18

The roadmap's milestone 0.5, the public release. No new solver features.

### Added

- `py.typed`: the package declares its type annotations (PEP 561).
- CI tests the built wheel on Linux, macOS and Windows with Python 3.11 to 3.14, and
  installs the sdist and tests it from its own tree. The site examples run in CI, and
  `twine check --strict` passes on both artefacts. `DSGELAB_TEST_INSTALLED=1` makes the
  suite import the installed package instead of `src/`.
- `validation/run_all.sh` runs the whole external validation and summarises which
  comparisons passed. `validation/setup_octave_packages.sh` installs the Octave packages
  at the versions the comparison used, checked by sha256, and `validation/macromodelling/`
  is a Julia project with MacroModelling.jl pinned. A weekly CI job runs all three.
- A release workflow that publishes by trusted publishing, to TestPyPI first, tests what
  TestPyPI serves, and only then publishes to PyPI.
- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, and issue and pull-request templates. The
  README has a section of known limitations.
- `.github/workflows/pages.yml` publishes the site to GitHub Pages, pointing the links to
  `docs/` at the files on GitHub. It is skipped while the repository is private.
- The site in English, with the Portuguese version under `site/pt-br/`. The examples of
  both versions run.
- `validation/heutel/fetch.sh` downloads Heutel's MATLAB files from the public copy at
  Georgia State University's repository and checks them; the six `.m` files are
  byte-identical to the ones the port was validated on.
- `dsgelab.examples.heutel.fetch()` downloads the same public copy, checks every file
  by SHA-256, and writes the translated algebra of the port to a cache directory
  (`~/.cache/dsgelab/heutel`, or `DSGELAB_HEUTEL_DIR`). Without it, `heutel.solve`
  raises `HeutelSourceMissing` naming the call to make, and the Heutel tests are
  skipped.

### Changed

- The validation scripts print their reports in English (`s.s.` for steady state,
  `moments`, `max relative deviation`, ...), with the same columns and numbers.
  `compare_estimation.py --no-mh` and `compare_policy.py --no-dynare` are the new flag
  names, and `--sem-mh` and `--sem-dynare` still work.
- The roadmap is in English (`ROADMAP.md`) and in Portuguese (`ROADMAP.pt-BR.md`).
- Examples, documentation, tests and validation scripts import the package as
  `import dsgelab as dsg`. The former alias `dk` came from the old name, dsgekit.
- The site links to cgelab, whose `cgelab.bridge` uses dsgelab for the intertemporal part of
  a model.
- The one comparison made on a model that is not public (the applied DSGE of an unpublished
  paper) is marked on the site as not reproducible from the repository.
- The translated algebra of the Heutel port (`dsgelab/examples/_heutel_algebra.py`) is
  no longer distributed, because it is a translation of Heutel's MATLAB files, which
  carry no licence. `heutel.fetch()` generates it on the user's machine, and
  `validation/heutel/translate.py` is gone.
- The calibration example on the site and in the user guide is replaced by one whose
  targets identify the free parameters (`target_jacobian` reports `identified=True`). The
  former example had a target that held identically, and `calibrate` returned different
  answers from different starting points.
- The validation figures quoted on the site match the current run of
  `validation/run_all.sh`. The README, `CONTRIBUTING.md`, the roadmap and the API
  reference are corrected where they described the checks, the validation layers or the
  API inaccurately.
- Python 3.11 or later is required. Python 3.10 is dropped, because SciPy 1.15.3, the
  last release for it, does not load on macOS 27.
- Error messages, the labels of derived first-order conditions ("FOC for control c",
  "Euler equation for state k", "constraint i: ...") and the header comments of files
  written by `to_dynare` and `to_macromodelling` are in English. The exported model text
  is unchanged.
- The external validation runs on Dynare 7.2, with Octave 11.3.0 and the Octave packages
  statistics 2.0.0 and datatypes 1.5.0. Every comparison prints the same numbers as on
  Dynare 7.1, and the Heutel errors move in their last digit.

### Fixed

- `SecondOrderSolution.simulate`, `mean_correction` and `ergodic_mean`, `ThirdOrderSolution.simulate`
  and `solve_third_order`, when given a covariance other than the one the solution was computed at,
  combined risk terms (`g_ss`, `h_ss`) of the old covariance with shocks or a state covariance of the
  new one. On the RBC with the standard deviation doubled, the mean correction of capital came out
  5.8e-05 instead of 2.4e-03. They now solve again at the covariance asked for, through the new
  `SecondOrderSolution.at_covariance`.
- The sdist now carries `validation/heutel/reference.json`, which its own test suite
  reads.
- `validation/compare_diagnostics.py` honours `OCTAVE` and `DYNARE_MATLAB` and looks in
  the same places as the other scripts, instead of only in Homebrew's.
- `modelo.mod` and `modelo.jl`, written by a site example, had been committed at the
  repository root; they are gone, and `tools/check_site_examples.py` now runs the
  examples in a temporary directory.
- Package metadata uses the SPDX license expression (`license = "MIT"`), which
  setuptools will require from 2027.

## 0.4.0 — 2026-10-11

The roadmap's milestone 0.4: nonlinearities and the declaration of the model. Each item is
validated against Dynare 7.1 on Dynare's own test files, against MacroModelling.jl, or
against a closed form.

### Added

- `OccbinModel`, `occbin_simulate`, `OccbinResult`: occasionally binding constraints by
  the piecewise-linear method of Guerrieri and Iacoviello (2015), as in Dynare's
  `+occbin`, with any number of constraints, time-varying rules found backwards, Dynare's
  regime update with `curb_retrench`, surprise shocks with re-planning, and the horizon
  extended while a constraint binds at its end. `from_mod` reads `bind=`/`relax=` equation tags,
  `occbin_constraints`, `shocks(surprise)` and the `occbin_setup`/`occbin_solver`
  options; `ModFile.occbin_simulate` runs them. Against Dynare's `borrcon.mod` (one
  constraint) and `dynrbc.mod` (two) to 2e-15, with identical regime histories.
- `extended_path`, `quadrature_nodes`, `ExtendedPath`: the extended path (Fair and
  Taylor, 1983) and the stochastic extended path (Adjemian and Juillard) of any order,
  on Dynare's sparse or full tree of Gauss–Hermite nodes, with the hybrid second-order
  risk correction, and `ModFile.extended_path`. Against Dynare's `burnside.mod` in seven
  variants to 1.1e-11, and `linearmodel0.mod` to 2e-15.
- Declared trends in `from_mod`: `trend_var`, `log_trend_var`, `var(deflator=)` and
  `var(log_deflator=)` are detrended as Dynare does, and an equation that still depends
  on the level of a trend is refused (Dynare's balanced-growth test, checked at several
  points). `fs2000_nonstationary.mod` from Dynare's tests joins `compare_import`.
- `balanced_growth`, `BalancedGrowth`, `ModFile.balanced_growth`: Dynare's `bgp.write`
  problem, solved by Levenberg–Marquardt with restarts away from degenerate paths. The
  residual matches Dynare's generated `bgpfun` to 2.2e-16 on its three tests.
- `diagnose`, Dynare's `model_diagnostics`. It reports variables absent at the current
  date, collinear relations of the static Jacobian with their variables and equations,
  redundant equations of the dynamic Jacobian, and unit roots, and gives the same report
  as Dynare on a model broken on purpose.
- `analytic_steady_state`, `SteadyStateError`, `from_equations(steady_state_model=)` and
  `ModFile.steady_state(analytic=True)`. A closed-form steady state is used only after it
  zeroes the model's residuals, and refused otherwise, as Dynare refuses it.
- Calibration equations inside the model, MacroModelling's `k[ss]/q[ss] = 2.5 | β`:
  `from_equations(calibration_equations=)` and `steady_state(calibrate=)`. Against
  MacroModelling.jl to 4e-15.
- `steady_state(homotopy=)` and `homotopy_setup` in `from_mod`: continuation from a
  parameter value where the model is easy, with adaptive steps.
- `assemble`: several agents' optimisation problems and the closing equations in one
  `Model`, refusing a variable chosen twice and renaming clashing multipliers.
- `from_equations` accepts lags and leads of any order, each carried by an auxiliary
  listed in `model.auxiliary`; a guess without the auxiliaries gets them from their
  variables.

### Changed

- `perfect_foresight(jacobian="auto")` is the default: the stacked Jacobian is assembled
  sparse from the model's exact derivatives when it has them (forward differences
  otherwise); ex-post states use them too.
- `klein` uses the real ordered QZ by default, the complex one as a fallback: SciPy 1.18's
  complex reordering loses up to 1e-8 on a pencil with a zero root and refuses one with
  several infinite roots.
- Moments and variance decompositions of a model with a unit root mask the
  nonstationary entries (`nan`) instead of failing; the Lyapunov equation is solved on
  the stationary block of an ordered Schur form.
- `from_mod` refuses a name declared twice.

## 0.3.0 — 2026-10-11

The first version that carries the roadmap's milestones 0.2 (optimal policy, welfare,
perfect foresight with news) and 0.3 (estimation, smoothing, decomposition,
forecasting), each item validated against Dynare 7.1 on Dynare's own files.

### Changed

- Renamed from `dsgekit` to `dsgelab`: `dsgekit` is taken on PyPI by an unrelated
  project. The import name, distribution name, CLI entry point and URLs move together.

### Added

- `state_space`, `kalman_filter`, `kalman_smoother`: the state space of a first-order
  solution, with observables in levels or logs (Dynare's `loglinear` with `logdata`),
  measurement errors and missing data; the filter starts stationary roots at their
  unconditional law and unit roots with the exact diffuse initialisation (Koopman and Durbin, 2000, unit
  roots first in the Schur form, as Dynare does). The likelihood matches Dynare's on
  `fs2000.mod` to 6e-12 and its `diffuse_filter` to 2.5e-12, and the smoother matches to
  1e-14.
- `EstimationProblem`, `posterior_mode`, `maximum_likelihood`, `metropolis_hastings`,
  `smc`, `prior_predictive`, `model_probabilities`, `posterior_forecast`, `rhat`,
  `effective_sample_size`: estimation, with Dynare's priors by mean and standard
  deviation (`prior`), the Laplace and modified-harmonic-mean marginal likelihoods,
  and `ModFile.estimation_problem` reading `estimated_params`, `varobs` and the options
  of `estimation(...)`. Validated on Dynare's `fs2000.mod`: prior density to 4e-13, a
  mode at least as high as Dynare's, Laplace by the same formula to 2e-13, and posterior
  means within 3 Monte Carlo standard errors of Dynare's (largest 2.9).
- `shock_decomposition`, `realtime_shock_decomposition`, `forecast`: against Dynare's
  commands of the same names to 5e-15 (forecast bands to 6e-9).
- `FirstOrderSolution.conditional_forecast(..., controlled=, levels=)`: Dynare's
  `conditional_forecast` with `controlled_varexo`, to 1.6e-13 on its own test file.
- `perfect_foresight_with_expectation_errors` and `News`: agents re-plan at each date of
  news from the states they have reached; constant or shrinking horizon. Validated
  against Dynare's `pfwee.mod` and `pfwee_constant_sim_length.mod` to 3e-12.
- `perfect_foresight(..., shocks=)` validated against Dynare's `rbc_det1`–`3`
  (anticipated shocks, initial conditions) to 1e-13.
- `from_mod`: exogenous variables carry their level (`EXO_SS_e`, from `initval`), may
  appear one period ahead, and the deterministic `shocks` block and `histval` are read;
  `ModFile.deterministic`, `ModFile.news`, `ModFile.initial_states` and
  `read_expectation_errors_csv` turn them into arguments for the solvers.
- `osr(..., objective=, discount=)`: an optimal simple rule on welfare (minus
  unconditional welfare at second order), Dynare's `osr` with `planner_objective`.
- `discretionary_policy` and `DiscretionSolution`: optimal policy under discretion
  (Dennis, 2007, as Dynare implements it), with IRFs, moments and welfare. Validated
  against Galí's closed form and Dynare's `tests/discretionary_policy/dennis_1.mod`.
- `osr` and `OSRResult`: optimal simple rules on a quadratic loss in unconditional
  moments, Dynare's `optim_weights` convention. On Dynare's `osr_example.mod` the loss
  matches Dynare's at its optimum to 2.6e-16, and the optimiser finds a lower one, which
  Dynare's own objective confirms.
- `welfare`, `welfare_model`, `consumption_equivalent`: second-order welfare,
  conditional and unconditional, and for Ramsey models with the lagged multipliers at
  zero. Matches Dynare's `evaluate_planner_objective` to 1.4e-12 for Ramsey and to
  6.1e-10 under discretion.
- `from_mod` reads `planner_objective`, `ramsey_model`, `discretionary_policy`,
  `osr_params`, `optim_weights` and `osr_params_bounds`; `ramsey_model` and
  `discretionary_policy` accept the resulting `ModFile`. `PrivateBlock` holds a
  private sector with instruments left open.
- `ramsey_model`, `ramsey_steady_state`, `ramsey_multipliers`: the Ramsey planner under
  commitment, timeless perspective, derived symbolically from the private-sector
  equations and a period objective. Validated against Galí's targeting rule and closed
  form, and against Dynare's `ramsey_model` on three models
  (`validation/compare_ramsey.py`).
- `from_mod`, `ModFile`: read a Dynare `.mod` file, with macros, model-local variables,
  `steady_state()`, and leads and lags of any order. Validated on eight `.mod` files read
  unchanged, five of them from Dynare's `examples/` (`validation/compare_import.py`).
- `solve(..., qz_criterium=)` and `klein(..., qz_criterium=)`: Dynare's convention that
  admits a unit root.
- `perfect_foresight(..., shocks=)`: a path of innovations, anticipated from the start.
- `SecondOrderSolution.irf`: the generalised impulse response of the pruned rule.
- `FirstOrderSolution.historical_decomposition` and `conditional_forecast`.
- `symbolic.parse_system` and `symbolic.model_from_expressions`: the symbolic front end
  split in two, so that derived equations reach the model builder without being printed
  and re-parsed.

## 0.2.0

- Third-order risk slopes, linear estimation (Kalman filter, posterior mode,
  random-walk Metropolis) and Chebyshev projection.
- Standalone package, separate from the research repository.

## 0.1.0

- Steady state, first-order (Klein/QZ) and second-order perturbation, perfect foresight,
  calibration on targets, symbolic front end, first-order-condition derivation,
  exporters to Dynare and MacroModelling.jl.
