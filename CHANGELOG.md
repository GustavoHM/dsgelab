# Changelog

All notable changes to `dsgelab`. Versions follow [semantic versioning](https://semver.org/);
before 1.0, a minor version may add to the API and a patch version only fixes.

## Unreleased

### Changed

- Renamed from `dsgekit` to `dsgelab`: `dsgekit` is taken on PyPI by an unrelated
  project. The import name, distribution name, CLI entry point and URLs move together.

### Added

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
