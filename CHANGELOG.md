# Changelog

All notable changes to `dsgelab`. Versions follow [semantic versioning](https://semver.org/);
before 1.0, a minor version may add to the API and a patch version only fixes.

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
