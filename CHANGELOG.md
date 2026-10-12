# Changelog

All notable changes to `dsgelab`. Versions follow [semantic versioning](https://semver.org/);
before 1.0, a minor version may add to the API and a patch version only fixes.

## 0.4.0 — 2026-10-11

The roadmap's milestone 0.4: nonlinearities and the declaration of the model. Each item is
validated against Dynare 7.1 on Dynare's own test files, against MacroModelling.jl, or
against a closed form.

### Added

- `OccbinModel`, `occbin_simulate`, `OccbinResult`: occasionally binding constraints by
  the piecewise-linear method of Guerrieri and Iacoviello (2015), as Dynare's `+occbin`
  -- any number of constraints, time-varying rules found backwards, Dynare's regime
  update with `curb_retrench`, surprise shocks with re-planning, the horizon extended
  while a constraint binds at its end. `from_mod` reads `bind=`/`relax=` equation tags,
  `occbin_constraints`, `shocks(surprise)` and the `occbin_setup`/`occbin_solver`
  options; `ModFile.occbin_simulate` runs them. Against Dynare's `borrcon.mod` (one
  constraint) and `dynrbc.mod` (two) to 2e-15, with identical regime histories.
- `extended_path`, `quadrature_nodes`, `ExtendedPath`: the extended path (Fair and
  Taylor, 1983) and the stochastic extended path (Adjemian and Juillard) of any order,
  on Dynare's sparse or full tree of Gauss--Hermite nodes, with the hybrid second-order
  risk correction; `ModFile.extended_path`. Against Dynare's `burnside.mod` in seven
  variants to 1e-11, and `linearmodel0.mod` to 3e-13.
- Declared trends in `from_mod`: `trend_var`, `log_trend_var`, `var(deflator=)` and
  `var(log_deflator=)` are detrended as Dynare does, and an equation that still depends
  on the level of a trend is refused (Dynare's balanced-growth test, checked at several
  points). `fs2000_nonstationary.mod` from Dynare's tests joins `compare_import`.
- `balanced_growth`, `BalancedGrowth`, `ModFile.balanced_growth`: Dynare's `bgp.write`
  problem, solved by Levenberg--Marquardt with restarts away from degenerate paths;
  the residual matches Dynare's generated `bgpfun` exactly on its three tests.
- `diagnose`: Dynare's `model_diagnostics` -- variables absent at the current date,
  collinear relations of the static Jacobian with their variables and equations,
  redundant equations of the dynamic Jacobian, unit roots. Same report as Dynare on a
  model broken on purpose.
- `analytic_steady_state` and `SteadyStateError`; `from_equations(steady_state_model=)`;
  `ModFile.steady_state(analytic=True)`: a closed-form steady state is used only after it
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
  unconditional law and unit roots exactly diffuse (Koopman and Durbin, 2000, unit
  roots first in the Schur form, as Dynare). Likelihood against Dynare's on `fs2000.mod`
  to 6e-12 and against its `diffuse_filter` to 2e-12; smoother to 1e-15.
- `EstimationProblem`, `posterior_mode`, `maximum_likelihood`, `metropolis_hastings`,
  `smc`, `prior_predictive`, `model_probabilities`, `posterior_forecast`, `rhat`,
  `effective_sample_size`: estimation, with Dynare's priors by mean and standard
  deviation (`prior`), the Laplace and modified-harmonic-mean marginal likelihoods,
  and `ModFile.estimation_problem` reading `estimated_params`, `varobs` and the options
  of `estimation(...)`. Validated on Dynare's `fs2000.mod`: prior density to 4e-13, a
  mode at least as high as Dynare's, Laplace by the same formula to 2e-13, posterior
  means within Monte Carlo error.
- `shock_decomposition`, `realtime_shock_decomposition`, `forecast`: against Dynare's
  commands of the same names to 5e-15 (forecast bands to 6e-9).
- `FirstOrderSolution.conditional_forecast(..., controlled=, levels=)`: Dynare's
  `conditional_forecast` with `controlled_varexo`, to 1e-13 on its own test file.
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
  matches Dynare's at its optimum (2.5e-15) and the optimiser finds a lower one, which
  Dynare's own objective confirms.
- `welfare`, `welfare_model`, `consumption_equivalent`: second-order welfare,
  conditional and unconditional, and for Ramsey models with the lagged multipliers at
  zero. Matches Dynare's `evaluate_planner_objective` to 1e-12.
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
