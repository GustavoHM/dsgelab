# Changelog

All notable changes to `dsgelab`. Versions follow [semantic versioning](https://semver.org/);
before 1.0, a minor version may add to the API and a patch version only fixes bugs.

## Unreleased

Work toward milestone 0.6 (identification, moments and data) and milestone 0.7 (complete third
order and derivatives).

### Fixed

- Particle-filter states now use the observation date at first order. Smoothing
  retains filtered weights and predictive normalization, and uses terminal weights
  for the final mean and ESS. Nonlinear and rank-deficient transitions use stored
  particle ancestry, including the two-state, one-shock RBC. The result reports
  the smoothing method. Missing dates retain their actual effective sample size.
- Inversion and second-order SMM use the same state-shock product ordering as the
  pruned transition matrices, including models with multiple states and shocks.
- The diffuse likelihood gradient differentiates every term in the finite
  covariance update, including the observation's parameter-dependent loading.
- Inversion accepts rectangular observation Jacobians and partial missing data.
  Shocks are standardized by their covariance, and unresolved state uncertainty
  propagates between dates. The linear likelihood is exact. Nonlinear rectangular
  and missing-data cases use a local Gaussian approximation and report
  `approximate=True`.
- `maximum_likelihood` and `posterior_mode` no longer fail when the optimiser tries a point
  without a determinate solution and asks for the gradient there; the point gets the penalty and
  a flat gradient, and the line search backs off. It happened with priors whose support lets a
  persistence parameter pass 1.

### Added

- `SecondOrderSolution.h_uu` and `h_ku`: quadratic innovation and state-innovation blocks of the
  state rule. `decision_rule()` returns a `DecisionRule` for `[k'; u']`, with `g_k`, `g_u`,
  `g_kk`, `g_uu` and `g_ku` in Dynare's coefficient convention. `h_s2 = h_ss - h_uu:Sigma`
  and `DecisionRule.g_s2 = [h_s2; g_ss]` provide the full risk constant, validated against
  Dynare's `ghs2`. The state-rule tensors are validated against Dynare 7.2 on `example1` and
  `example2`.
- `ThirdOrderSolution.h_uuu`, `h_kuu` and `h_kku`, with `ThirdOrderSolution.decision_rule()`:
  cubic state-innovation blocks and their compositions `g_uuu`, `g_kuu` and `g_kku`, checked
  against the exact RBC, Dynare's `ghuuu` and the state rows of `ghxxu`. `ghxuu` agrees in every
  row, and the composed `g_kkk` also agrees with `ghxxx`. The risk-slope calculation reuses the
  ex-post second-order innovation blocks. `ThirdOrderSolution.g_xss` and `h_xss` add the risk-state
  terms, and `g_uss` and `h_uss` add the risk-innovation terms. `ghxss` and `ghuss` agree with
  Dynare in every row, with errors below `1.5e-16` on both examples.
- `SecondOrderSolution.irf` and `ThirdOrderSolution.irf`: conditional impulse responses of the
  pruned rules, starting from the deterministic steady state or a supplied state. The innovation
  enters in period 0. Third order uses the full Dynare recursion, with `ghxss`, `ghuss` and every
  cubic innovation block. Generalized IRFs average stochastic paths with `drop=100` and agree with
  Dynare on `example1` and `example2` within `1e-3`.
- `stochastic_steady_state`: fixed point of the full unpruned third-order decision rule with
  `sigma=1`, returned as a `SteadyState` with its residual and iteration count. The fixed point is
  distinct from the ergodic mean of the pruned system.
- `solution_jacobian`: derivatives of `P`, `F`, `Q` and the steady state. Purely backward models
  use implicit differentiation of `P = A^-1 B` and `Q = A^-1 C`. General QZ solutions differentiate
  the implicit saddle-path conditions, avoiding a QZ decomposition at every perturbed parameter.
  Central differences and Richardson extrapolation retain the best step triple. Validated against
  Dynare 7.2's analytical `dghx` and `dghu` on `example1` and `example2`, with relative errors
  below `2.4e-4`, and against closed forms with jumps and complex stable eigenvalue pairs.
- `log_likelihood_gradient`: recursive derivatives of the first-order Kalman
  likelihood. Uses `solution_jacobian` for model parameters and direct formulas for shock and
  measurement-error standard deviations and correlations. Supports `level` and `log`
  observations, `presample`, missing data and diffuse starts through the univariate filter.
- `log_posterior_gradient` and `Prior.score`: analytic prior derivatives for every supported
  distribution, added to the likelihood gradient.
- `hmc`: Hamiltonian Monte Carlo with leapfrog integration, Metropolis correction, a mass matrix
  from the posterior mode and `MCMCResult` diagnostics. During burn-in, a heuristic sets the first
  step and dual averaging (Hoffman and Gelman, 2014) tunes it toward the target acceptance. Each
  trajectory multiplies the step by a uniform draw on [0.8, 1.2]. `posterior_mode` and
  `maximum_likelihood` also use these gradients in BFGS.
- `nuts`: No-U-Turn sampling with multinomial choice over the trajectory, Betancourt's generalised
  U-turn criterion, the same step adaptation and `NUTSResult`. Reports divergences, maximum-depth
  hits and mean tree depth per chain after burn-in. Both samplers are checked against the exact
  posterior of an AR(1) coefficient, computed by quadrature, and a long Metropolis–Hastings run,
  with tolerances set by each chain's effective sample size.
- `second_order_jacobian`: derivatives of the full second-order decision rule, including `g_k`,
  `g_u`, `g_kk`, `g_uu`, `g_ku`, `g_s2` and the steady state. Uses central differences and
  Richardson extrapolation for model parameters, shock standard deviations and correlations.
  Validated against Dynare 7.2's analytical `dghxx`, `dghxu`, `dghuu` and `dghs2` in every
  variable row.
- `third_order_jacobian`: derivatives of `g_kkk`, `g_kku`, `g_kuu`, `g_uuu`, `g_s2` and the
  steady state with Richardson extrapolation, for model parameters, shock standard deviations
  and correlations. Validated against Dynare 7.2's analytical third-order derivatives.
- `derivatives="third"` in the symbolic frontend: exact third derivatives of the model
  residuals through `Model.third_fn`. `third_derivatives` uses them directly instead of
  differentiating the Hessian numerically. The default `"both"` remains unchanged. The cubic
  decision-rule comparisons against Dynare stay below `3e-13` on `example1` and `2e-14` on
  `example2`. `SparseSecondDerivatives` stores nonzero Hessian entries, and
  `SparseThirdDerivatives` stores nonzero cubic entries; both support
  contractions without densification. `solve_third_order(sparse=True)` also stores the cubic
  state coefficients by independent symmetric triple and contracts them in rule evaluation and
  risk terms. Explicit decision-rule composition expands dense arrays on demand.
  Direct and iterative cubic Sylvester solves use the same symmetric coordinate form, the latter
  with an `Abar` preconditioner. The iterative known term and residual use canonical coordinates,
  and each matvec expands one equation at a time with an optimized contraction path.
  `hessian=` reuses second derivatives alongside `third=`. A full nonlinear pipeline with 71
  variables, 35 states and five shocks validates the sparse route with a maximum residual of
  `3.1e-20` and a peak RSS of 321 MB. `third_order_jacobian` accepts `sparse` and `direct_limit`
  and applies them at every perturbed point.
- `particle_filter`: bootstrap likelihood at first, pruned second and pruned third order, with stationary
  initialization, systematic resampling by effective sample size and measurement-error covariance.
  Second order uses the augmented pruned state and the correct `C z_{t-1} + D inov_t` observation
  timing. Validated against Dynare 7.2's sequential importance particle filter on an AR(1).
  `EstimationProblem(order=2)` exposes the particle likelihood to posterior mode and samplers,
  with common random draws. `particle_smoother` adds smoothing of
  `particle_filter(..., store=True)`, with smoothed means and effective sample sizes at all three
  orders. Missing observations are skipped in the filter and smoother. The pruned third-order
  particle filter uses `[xf; xs; xt]` and the full rule, including `ghxss` and `ghuss`, in both
  the transition and the observation, and starts each particle from its own 1,000-period path.
  On a linear model with a jump observed under small measurement error, it reproduces the Kalman
  likelihood within Monte Carlo error.
  `EstimationProblem(order=3)` exposes the likelihood to posterior mode and samplers, and its
  `smoother` method returns third-order smoothed means.
- `inversion_filter`: minimum-norm shock inversion at first and pruned second order,
  with the observation-Jacobian correction and missing-data support. The linear
  likelihood is exact; nonlinear underdetermined cases use a local Gaussian approximation.
  `EstimationProblem(nonlinear_filter="inversion")` selects it instead of the particle filter.
  Validated against Dynare OccBin's `IVF_core` on an AR(1). `inversion_smoother` returns the
  inverted paths without a backward pass.
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
- `method_of_moments` and `ModFile.method_of_moments`: first-order GMM and SMM with
  identity, diagonal or Newey--West optimal weighting, iterated stages, standard errors and
  Hansen's J test. SMM uses fixed simulation draws and applies the `1 + 1/simulation_multiple`
  variance correction. `matched_moments` and `estimated_params` are read from the `.mod`.
  With `order=2`, GMM computes means and autocovariances from the pruned second-order ABCD
  system and SMM simulates the same system with fixed draws. The `.mod` option supplies the order.
- `irf_matching` and `ModFile.irf_matching`: frequentist first-order IRF matching with sparse
  targets, full or diagonal weights, correlated shocks and the inverse Hessian covariance.
  External targets carry their joint covariance; its inverse is the default efficient weight.
  The importer reads `matched_irfs` and `matched_irfs_weights`, including numeric ranges and
  vector expansion. MATLAB `irf_matching_file` scripts are not executed; a Python `transform`
  can be supplied explicitly.
- `local_projection`: observed-shock local projections with a joint Bartlett-HAC covariance
  across variables and horizons. `LocalProjectionResult.targets()` returns targets and their
  covariance in the order accepted by `irf_matching`.
- `validation/compare_mom.py` and `validation/compare_irf.py`: eight GMM/SMM cases and two
  IRF-matching cases against Dynare 7.2, at common parameter points and after estimation.
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
- `check_restrictions` and `sample_restrictions`: sign and interval restrictions on impulse
  responses at given horizons and on theoretical moments at given lags, as Dynare's
  `irf_calibration` and `moment_calibration` blocks state them. They are checked at one
  first-order solution and over draws from the prior, with the share of the solved draws that
  meets each restriction. The conventions follow Dynare: absolute deviations, a
  one-standard-deviation impulse (100 times a unit innovation with `relative_irf`), the variance
  at lag 0 for a variable with itself, correlations otherwise, and a logical OR over the horizons
  of one restriction. `RestrictionCheck` and `RestrictionSample` hold the values and verdicts.
- `ModFile.check_restrictions` and `ModFile.sample_restrictions`: the importer reads the
  `irf_calibration` and `moment_calibration` blocks, including `relative_irf` and `overwrite`, into
  `ModFile.irf_calibration` and `ModFile.moment_calibration`, and samples the priors of
  `estimated_params`.
- `validation/compare_restrictions.py`: both blocks of Dynare's `tests/gsa/ls2003a.mod` against
  Dynare 7.2, at the calibration and on the 512 prior draws of its sensitivity toolbox. The values
  agree within 1.3e-12, the same 486 draws solve, and each restriction's verdict agrees on every
  draw.
- `forecast_evaluation`: out-of-sample forecasts at a sequence of origins, re-estimating the
  posterior mode (or the likelihood, or keeping given parameters, or calling a given estimator)
  on an expanding or a rolling window, as Dynare's recursive estimation does. `ForecastEvaluation`
  holds the forecasts, their predictive covariance and the outcomes, and gives the RMSE by
  horizon, its ratio to the random walk's, the PIT, the coverage of the bands and the log
  predictive score, per observable or joint.
- `diebold_mariano`: the Diebold–Mariano test with the Harvey–Leybourne–Newbold correction, both
  variance estimators and the three alternatives, as R's `forecast::dm.test`; also
  `ForecastEvaluation.diebold_mariano` against the random walk or another forecast's errors.
- `validation/compare_forecast_eval.py`: Dynare 7.2's recursive estimation of `fs2000` over nine
  samples, with forecasts within 2.6e-15 at Dynare's modes and modes of equal or higher kernel
  when re-estimated, and 48 Diebold–Mariano tests against R's `forecast::dm.test` within 3.1e-15.
- `stability_map`: draws parameters from priors or from `Uniform`, `Normal` and `Triangular`
  (the interface of `cgelab.sensitivity`), on a scrambled Sobol sequence, a Latin hypercube or at
  random, solves the model at each draw and sorts the draws into determinate, indeterminate,
  explosive, without a steady state and without a solution, as Dynare's
  `dynare_sensitivity(stab=1)`. `ModFile.stability_map` takes the priors of `estimated_params`.
- `monte_carlo_filtering` and `smirnov_test`: which parameters separate the draws that show a
  behaviour from those that do not, by Dynare's Smirnov test, and the correlations between
  parameters inside each group. `StabilityMap.filtering` runs Dynare's three analyses
  (determinacy, indeterminacy, explosiveness), and `RestrictionSample.filtering` filters on the
  calibration restrictions, as Dynare's `map_calibration`.
- `sobol`: first-order and total Sobol indices on Saltelli's design, with bootstrap intervals,
  the design and estimators of `cgelab.sobol` and SALib; base points with a missing output are
  left out. `solution_outputs` makes the function from IRFs and moments of the first-order
  solution, in the units of `check_restrictions`.
- `Prior.ppf`: the quantile function of every prior shape.
- `validation/compare_gsa.py`: the stability mapping of Dynare 7.2 on `tests/gsa/ls2003a.mod`
  over three samples of 1024 draws, one with explosive draws, and the Sobol indices against
  SALib 1.6. Every draw has the class Dynare's solver gives it; the Smirnov tests and
  correlations on Dynare's groups agree within 1.7e-13, and SALib's design and indices within
  8.9e-16.
- `validation/mmb/`: thirteen models of the Macroeconomic Model Data Base, read by
  `from_mod` and compared with Dynare 7.2 at first order (steady state, IRFs and moments).
  They include SAMBA (Banco Central do Brasil), the ECB's NAWM, FRB/US in two versions
  (up to 361 variables), Smets and Wouters (2007) and the New York Fed model of Del Negro,
  Giannoni and Schorfheide (2015). All thirteen agree within 1e-04, and twelve within 2e-08. The
  `.mod` files come from a fixed Rep-MMB commit, checked by SHA-256, and are not versioned.
- `validation/dsge_mod/`: four published-model files from Johannes Pfeifer's DSGE_mod —
  Aguiar–Gopinath (2007), Ghironi–Melitz (2005), Hansen (1985) and Smets–Wouters (2007) — read
  by `from_mod` and compared with Dynare 7.2 at first order. The largest IRF deviation is
  2.2e-07, in Ghironi–Melitz, whose Dynare steady state leaves a residual of 2e-08. Sources
  are pinned by commit and SHA-256; the site documents the scope and limits of each comparison.

### Changed

- The site shows, under each example, what it prints and the value of each expression on a line
  of its own. `tools/check_site_examples.py --write` writes that output, and the check fails
  when the page no longer matches the code. Table cells wrap their text, so the comparison
  table no longer runs off the page.
- `validation/compare_import.py`: a variable whose largest response is below 1e-9 of the
  model's largest response is left out of the IRF comparison. Its printed results are
  unchanged.

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
