# Changelog

All notable changes to `dsgelab`. Versions follow [semantic versioning](https://semver.org/);
before 1.0, a minor version may add to the API and a patch version only fixes.

## Unreleased

### Changed

- Renamed from `dsgekit` to `dsgelab`: `dsgekit` is taken on PyPI by an unrelated
  project. The import name, distribution name, CLI entry point and URLs move together.

### Added

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
