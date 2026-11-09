# Changelog

All notable changes to `dsgelab`. Versions follow [semantic versioning](https://semver.org/);
before 1.0, a minor version may add to the API and a patch version only fixes bugs. The exception
is the roadmap's milestone 0.8, planned in parts: versions 0.7.1 and 0.7.2 each added one of its
features, without changing the API of an earlier release. The rest of the milestone, planned as
0.7.3 to 0.7.5 and 0.8.0, was not published on its own and came out in 0.9.0.

## Unreleased

### Added

- `solve(..., check_steady=False)` linearises at a supplied point with nonzero residual, matching
  Dynare's `steady(nocheck)`. The default still refuses such a point.
- `from_mod` expands `@#for` over explicit arrays, accepts Dynare's lead/lag notation on parameters
  and repeated parameter declarations, and ignores `close all`, `clear all` and `clc`.
- The MMB validation adds SIGMA, the Area-Wide Model, FiMod and the six-region Global Projection
  Model, compared with Dynare on the steady state or declared linearisation point, all IRFs and
  theoretical standard deviations.

### Fixed

- Symbolic models print the constant pi as a number before lambdification, so an endogenous
  variable named `pi` no longer replaces the constant in `normpdf` and similar expressions.

## 1.0.0rc1 — 2026-11-08

Release candidate for 1.0, covering milestone 0.10 and the API audit before the freeze.

### Added

- `value_iteration` takes a `SmolyakGrid`, as `solve_projection` and `time_iteration` do, with the
  value and the policy interpolated by the grid's polynomial. `validation/compare_value_iteration.py`
  solves stochastic growth with three capital stocks and the saving shares as three controls on
  Smolyak grids of levels 1 to 3, against the closed-form value. At level 3, with 69 nodes, the
  value agrees within 6.4e-8 at the nodes and 7.3e-8 at 600 random points of the box, and the
  error falls by at least 58 from one level to the next.
- The figures left out in 0.9. `identification(..., strength=True)` computes the identification
  strength of Ratto and Iskrev (2011) as Dynare's `identification` does at first order, from the
  asymptotic information matrix of the Gaussian likelihood on the data or on a simulated sample
  (`IdentificationStrength`: the matrix, the strength and the sensitivity component relative to
  the parameter and to its prior standard deviation, the sensitivity of the moments and the
  identification patterns), drawn by `plot_identification_strength`,
  `plot_identification_sensitivity` and `plot_identification_patterns`.
  `realtime_shock_decomposition` takes `forecast` and `vintages` and returns Dynare's one-step
  conditional and forecast pools and, for each vintage, its decomposition, its forecast and its
  conditional decomposition, which `plot_decomposition` draws with `realtime` and `vintage`.
  `annualise_decomposition` and `plot_decomposition(frequency=...)` give Dynare's `type=yoy` and
  `type=aoa`, and `initial_condition_decomposition` Dynare's command of that name.
  `conditional_forecast_bands` draws the shocks left free, as Dynare's `conditional_forecast`, and
  returns a `ConditionalForecast` with its bands, `posterior_conditional_forecast` adds parameter
  draws, and `plot_conditional_forecast` draws the bands. `validation/compare_plots.py` compares
  them with Dynare's on `fs2000.mod` at the prior mean. On the sample Dynare simulates for its
  information matrix, the matrix agrees within 1.2e-8 of the root of its diagonal and the strength
  and sensitivity measures within 2.7e-7. The real-time, annualised and initial condition
  decompositions agree within 2e-14, and the conditional forecast and its bands, on Dynare's 2,000
  draws, within 7e-15. Dynare's `q2a` annualisation and its sensitivity bars for the reduced
  form and the dynamic model are not reproduced.
- `forecast_evaluation(bvar_lags=..., bvar_options=...)` adds a BVAR with the Sims–Zha prior as a
  second benchmark, re-estimated at each origin on the model's window. Its forecasts at the
  posterior mean are in `ForecastEvaluation.bvar_mean` and its errors in `bvar_errors`, and
  `rmse`, `relative_rmse` and `diebold_mariano` take `benchmark="bvar"`. On Dynare 7.2's
  `tests/bvar_a_la_sims/bvar_and_dsge.mod`, at four origins on an expanding and a rolling window,
  the errors 1 to 4 steps ahead agree within 1.6e-16 with those of Dynare's `bvar.toolbox` on the
  same windows (the largest error is 0.022).
- `BVAR.irf`, Dynare's `bvar_irf`: impulse responses over draws from the posterior, identified by
  the Cholesky factor, the symmetric square root or a callable, with the mean, median, variance,
  HPD interval, deciles and the bounds of `oo_.bvar.irf`. On `bvar_standalone.mod`, with the
  normal draws Octave took, the 1000 sampled responses agree with Dynare's within 1.9e-15 of the
  largest response, and the statistics within 1.1e-15. Dynare 7.2 takes the rows of the
  Cholesky factor `L` as the impulses, whose covariance is `L'L` rather than `Sigma`. `BVAR.irf`
  takes its columns, and reproduces Dynare's numbers with the upper factor as identification.
- `dsge_var_irf`, Dynare's `bayesian_irf` with `dsge_var`: the DSGE-VAR's impulse responses over
  the posterior, identified by the rotation of Del Negro and Schorfheide (2004) from the model's
  impact matrix. On `tests/dsge-var`, with the 1000 parameter draws and the normal draws Dynare
  took, the sampled responses agree within 1.1e-13 of the largest response and the statistics of
  `oo_.PosteriorIRF.bvardsge` within 5.7e-14. `plot_posterior_irf` draws the results of both.
- `improved_time_iteration`: Newton's method on the equilibrium conditions of every grid point at
  once, with next period's policy interpolated from the unknowns, as dolo's
  `improved_time_iteration`. It takes the declaration, the arguments and the result of
  `time_iteration`, on a `CartesianGrid` or a `SmolyakGrid`, with Markov or quadrature exogenous
  states and bounds by complementarity, and reaches the same fixed point. The derivative through
  next period's policy is applied as an operator, and each step is solved by GMRES, preconditioned
  by the blocks for each point. `validation/compare_improved_time_iteration.py` checks it against
  dolo 0.4.9.20's routine on dolo's RBC, Markov RBC and consumption–savings models, within 3.7e-10
  with linear and cubic interpolation, in 4 to 10 Newton steps where time iteration takes 114 to
  361 iterations. As published, dolo's routine does not converge on the consumption–savings model
  from its calibration, and `validation/dolo/run_dolo_iti.py` starts it from its own time
  iteration stopped at 1e-3, without editing dolo.
- `expectations_iteration`: iteration on the conditional expectations of the Euler equations,
  declared as jumps of the `Model` (`z - beta E[c'^(-gamma) R']`, with `c^(-gamma) - z` among the
  conditions without `t+1`), with the controls found from the expectations at each grid point
  without an integral, and `damping=0.5` on the update of the expectations. dolo documents this
  iteration, as parameterized expectations, and 0.4.9.20 does not implement it. On stochastic
  growth with log utility and full depreciation it agrees with `time_iteration` on the same grid
  within 5e-12 and with the closed form as time iteration does. Without damping it diverges there.
- A `HetBlock`'s Markov matrix may be an output of `hetinputs` that moves with the inputs along a
  path, in the steady state, the fake-news Jacobians, `G` and the nonlinear transition. `Pi_t`
  moves the distribution at `t` and the expectation the step takes at `t - 1`, as in
  sequence-jacobian 1.0.0. `HetBlock.path(..., Dbeg=...)` and
  `BlockModel.impulse_nonlinear(..., initial={name: Dbeg})` start a transition from a given
  distribution in place of the steady state's. `validation/compare_heterogeneous.py` adds the
  household of sequence-jacobian's `labor_search.ipynb`, whose employment chain moves with the
  job-finding and separation rates, closed by a fixed supply of bonds that the notebook does not
  have and that both packages write identically. The steady state agrees within 8.1e-10, the
  Jacobians with respect to the rates within 2.9e-10, `G` within 8.3e-11, the linear and
  nonlinear responses within 2.2e-10, and the return from the distribution of another steady
  state, sequence-jacobian's `ss_initial`, within 6.0e-10. The fake-news Jacobians agree with the
  brute-force ones within 1.6e-9, and a path from a given distribution agrees with the
  distribution moved period by period within 3.1e-13. The asset grid still stays fixed along a
  path, and a `NotImplementedError` now names only it.
- `HetBlock` accepts two endogenous states: `policy` and `grid` take two names, every individual
  array has two asset axes, and the distribution moves by the product of the two one-dimensional
  lotteries, as in sequence-jacobian's two-asset HANK. The steady state, the fake-news and direct
  Jacobians and the nonlinear transition work on the two grids. The validation reproduces
  sequence-jacobian 1.0.0's `two_asset.ipynb` on 3 by 50 by 70 points, with the steady state within
  9.9e-10, the forward household Jacobians within 1.3e-9 and the nonlinear transition within
  1.9e-8. The full linear approximations of `G` differ by 1.3e-2, while selected `K/Z` entries
  checked against sequence-jacobian's direct nonlinear transition agree within 1.4e-3. On a
  smaller grid, fake news agrees with direct differentiation within 1.3e-8.
- `solve_second_order_switching` and `solve_markov_switching(..., order=2)`: the second-order
  solution of a Markov-switching model, by the partition perturbation of Foerster,
  Rubio-Ramírez, Waggoner and Zha (2016, Proposition 2). Given the first-order solution, the laws
  of motion give the states' second derivatives in terms of the jumps', and the expected rows
  give three linear systems, in `(k, k)`, `(k, chi)` and `(chi, chi)`, one block per regime.
  `SecondOrderSwitchingSolution` has the blocks by regime, `decision_rule(i, j)` the rule for
  `[k'; u']` on `(k, eps', chi)` (`SwitchingDecisionRule`), pruned simulation and impulse
  responses along a regime path, and the conditional and ergodic means in closed form.
  `validation/compare_switching.py` reproduces the 96 second-order coefficients that Foerster et
  al. (2016) print for their RBC, all of them once rounded to four decimals (the largest
  difference is 0.48 of a unit of the fourth decimal). With identical regimes it agrees with
  `solve_second_order` within 7.9e-15, and on stochastic growth with log utility and full
  depreciation, with switching productivity, persistence and volatility, every coefficient agrees
  with the closed form within 1.3e-10.
- `from_equations(trend_vars=..., log_trend_vars=..., deflators=..., log_deflators=...)`
  declares the trends of a model written in levels, as `trend_var`, `log_trend_var`,
  `var(deflator=)` and `var(log_deflator=)` do in a `.mod`. The text equations are detrended as
  `from_mod` detrends a `.mod`, with Dynare's balanced-growth check, and the declarations are in
  `model.trends`. `validation/compare_bgp.py` writes Solow and Ramsey from Dynare 7.2's
  `tests/bgp`, with their trends declared, and `tests/trend_var/fs2000_nonstationary.mod` as
  text equations. Against Dynare, the steady state agrees within 3.4e-14, the IRFs within 9.7e-14
  of each variable's largest response, the paths under the same 200 shocks within 8.2e-14 and the
  variance matrix within 7.8e-14 of its largest entry. Against `from_mod` on the same `.mod` the
  largest difference is 1.4e-13.

### Changed

- `value_iteration`'s `interpolation` defaults to `None`, which is `"linear"` on a
  `CartesianGrid`, so that a `SmolyakGrid` can refuse a value. Calls that worked before give the
  same results.
- `max_iter` is the preferred iteration-limit keyword in `calibrate`, `perfect_foresight`,
  `extended_path`, `discretionary_policy`, `solve_markov_switching` and `msv_solutions`. `maxiter`
  remains accepted as a deprecated alias, with its previous default, and passing both names raises
  `ValueError`.
- In `from_equations`, the lead `x(+k)`, `k >= 2`, of a state starts from a new jump
  `LEAD0_x = x`, with `LEAD1_x = LEAD0_x(+1)`. Before, `LEAD1_x = x(+1)` had no jump at `t+1`, and
  with an exogenous state the solver refused the model as inconsistent. A model with the lead of
  an endogenous state has one more auxiliary and the same solution.

### Fixed

- The balanced-growth check of a `.mod` failed with a `NameError` on an equation that holds two
  trends after detrending, such as Solow with technology and population declared as trends.

## 0.9.0 — 2026-11-05

The rest of the roadmap's milestone 0.8 (global methods): value function iteration, the
endogenous grid method and Smolyak sparse grids. Milestone 0.9: Markov regime switching,
heterogeneous agents by the sequence-space Jacobian, and the BVAR and the DSGE-VAR. Also the rest
of Dynare's figures, at the author's request.

### Added

- `value_iteration`: value function iteration with Howard's (1960) improvement steps, on a
  `DynamicProgram`, a new declaration of the states, the controls, the reward, the transition, the
  discount factor and the bounds, which a `Model` does not hold. The value is kept on a
  `CartesianGrid`, one copy per node of a Markov chain, and interpolated linearly or by natural
  cubic splines. The shocks are integrated as in `time_iteration`, and the transition sees the next
  values of the Markov states, so that cash on hand can depend on next period's income. The
  maximization is a golden-section search with one control, a projected Newton method with
  finite-difference derivatives with several, and an exhaustive search with `choices=`, finite sets
  of values. `ValueIterationSolution` evaluates the value and the policy, simulates, and gives
  `bellman_errors`. `validation/compare_value_iteration.py` checks it against QuantEcon's
  `DiscreteDP` on the discretised growth model, within 6e-12 and with the same policy, against
  dolo 0.4.9.20's `value_iteration` on its Markov RBC, within 1.3e-11 of the value and 6e-7 of
  the controls, and against the closed-form value of stochastic growth. As published, dolo's
  `value_iteration` ends in NaN on that model, and `validation/dolo/run_dolo_vfi.py` makes four
  changes, listed there, without editing dolo.
- `egm`: the endogenous grid method (Carroll, 2006), on an `EGMProgram`, a new declaration of a
  problem with one endogenous state, its poststate, the expectation in the Euler equation, the
  direct response that inverts it for the controls, the reverse map from the poststate to the
  state, the half transition to next period's state and the bounds on the controls. These are
  dolo's `expectation`, `direct_response_egm`, `reverse_state` and `half_transition` blocks, with
  `post_decision`, the poststate given the state and the controls, added. The grid is on the
  poststate, a `CartesianGrid` with one axis, and the other states follow Markov chains, with the
  shocks integrated as in `time_iteration`. The policy is interpolated linearly or by natural cubic
  splines between the points of the endogenous grid, one per node of the chain, and held within
  the bounds, which gives a borrowing limit below the lowest point. `EGMSolution` evaluates and
  simulates the policy and gives `euler_errors`, relative to the controls.
  `validation/compare_egm.py` checks it against dolo 0.4.9.20's `egm` on its consumption–savings
  model, within 9e-13 with iid income and 1.1e-11 with a three-state Markov chain, against
  `time_iteration` on the same model, whose mean difference falls to 4e-7 and 2e-6 with 1600
  points, and against the closed form of stochastic growth, whose relative error falls as the
  square of the spacing, to 1.6e-4 with 80 points. `validation/dolo/run_dolo_egm.py` silences a
  debugging line in dolo's `egm` and reads its policy only at the points where dolo's resampled
  rule equals it, without editing dolo.
- `SmolyakGrid`: Smolyak sparse grids with the nodes and the basis of Judd, Maliar, Maliar and
  Valero (2014), nested Chebyshev extrema and disjoint sets of Chebyshev polynomials, isotropic or
  with one level per axis (JMMV's anisotropic grid). It evaluates the basis, the interpolant and
  the interpolation weights. `solve_projection(..., grid=SmolyakGrid(...))` collocates on it with
  any number of states, by Newton's method with a Jacobian assembled from the residual's
  derivatives at each node, and `time_iteration` takes it in place of a `CartesianGrid`.
  `validation/compare_smolyak.py` checks the nodes and the interpolation weights against Tasmanian
  8.2, built from source by `validation/tasmanian/setup.sh`, on 40 grids in one to six
  dimensions, within 9e-16 and 2e-14. On stochastic growth with three to five capital stocks, log
  utility and full depreciation, whose investment policy is a constant share of output in any
  dimension, the error falls by a factor of 37 or more per level, to 2e-8 with four states and
  3e-8 with six at level 4, and projection and time iteration agree within 2.3e-9.
- The rest of Dynare's figures, each from an object the package already returns: `plot_priors`
  (`plot_priors`), `trace_plot` (`trace_plot`, with its moving average),
  `plot_mh_autocorrelation` (`mh_autocorrelation_function`), `plot_brooks_gelman` (the univariate
  and multivariate diagnostics of `mcmc_diagnostics`), `plot_mode_check` (`mode_check`),
  `plot_smoothed_shocks` and `plot_smoothed_observables` (the smoother's "Smoothed shocks",
  "Smoothed observation errors" and "Historical and smoothed variables"), `plot_posterior_irf`
  (`bayesian_irf`), `plot_conditional_forecast`, `plot_occbin` and `plot_occbin_regimes`
  (`occbin_graph` and `occbin.plot_regimes`), `plot_identification` (the collinearity patterns of
  `identification`) and `plot_filtering` (`scatter_mcf` and `stability_mapping_univariate` of the
  sensitivity toolbox). `plot_decomposition` gains `groups`, `steady_state` and `detail`
  (`use_shock_groups`, `steadystate` and `detail_plot`), `plot_posterior` Dynare's axis limits,
  and `plot_paths` accepts a `TransitionPath`. The numbers these figures draw that the package did
  not compute before are computed as Dynare computes them, without matplotlib: `brooks_gelman`
  (returning a `BrooksGelman`), `mh_autocorrelation`, `mode_check`, on Dynare's grid, and
  `posterior_irf`, with the HPD interval and deciles of `posterior_moments`.
  `validation/compare_plots.py` estimates `fs2000.mod` in Dynare with `mode_check`, two chains
  and `bayesian_irf`, records the data of Dynare's own figures as they are drawn, and compares
  them on the same draws: the Brooks and Gelman statistics within 8e-14, the moving average and
  the autocorrelations within 3e-15, the `mode_check` grid within 9e-16 and its kernels within
  6e-11 where the prior is bounded, and the posterior IRFs on Dynare's 1,200 parameter draws
  within 3e-10 of the largest response. Dynare's identification-strength and sensitivity bars
  need the information matrix, which the package does not compute, and are left out, as are the
  annualised and vintage shock decompositions and the bands of conditional forecasts.
- `bvar` and `bvar_density`: a VAR with the prior of Sims and Zha (1998), as Dynare's
  `bvar_density` and `bvar_forecast` build it, from dummy observations with Dynare's
  hyperparameters (`tau`, `decay`, `lambda_`, `mu`, `omega`, `flat`, `train`) and options
  (`first_obs`, `presample`, `nobs`, `noconstant` as `constant=False`, `prefilter`). `BVAR` holds
  the normal-inverse-Wishart prior and posterior (`NormalInverseWishart`, with draws and the
  integrating constant), the log marginal density and the RMSE Dynare prints, and
  `BVAR.forecast` draws paths with and without shocks, with the normal draws in Dynare's order.
  `validation/compare_bvar.py` runs Dynare 7.2's `tests/bvar_a_la_sims` as they are, apart from
  the IRFs and the Metropolis of one of them: on lag lengths 1 to 2 and 1 to 8 the priors and
  posteriors agree within 1.8e-14, relative, and the log marginal densities within 1.8e-12.
  With the normal draws Octave took, the 1001 and 2001 forecast paths agree within 2.1e-16, draw
  by draw.
- The DSGE-VAR of Del Negro and Schorfheide (2004), as Dynare's `estimation(dsge_var=...)`:
  `EstimationProblem` takes `dsge_var` (the weight `lambda` of the model's prior, `inf`
  included), `dsge_varlag` and `dsge_var_constant`, or estimates the weight under the name
  `"dsge_prior_weight"`. The likelihood is that of the VAR with its coefficients and covariance
  integrated out, from the model's population moments at first order, and
  `dsge_var_prior` and `dsge_var_posterior` give the VAR's prior and posterior at given
  parameters. `ModFile.estimation_problem` reads `dsge_var`, `dsge_varlag` and `noconstant`.
  The suite checks the likelihood against conjugate updating, one observation at a time.
  `validation/compare_bvar.py` runs Dynare 7.2's `tests/dsge-var`, with `lambda = 0.8` and with
  `lambda` estimated, without Metropolis. Dynare counts the `dsge_varlag` initial observations
  in the likelihood's `T` (94 for 90 rows), and with that `T` the log posterior agrees within
  4.9e-12 at 11 points, and the mode `dsgelab` finds has a kernel at least as high as Dynare's,
  within 1e-4 posterior standard deviations of it. With `dsgelab`'s `T`, the number of rows,
  the mode moves by up to 0.18 and 0.39 posterior standard deviations, and the estimated
  `lambda` from 0.533 to 0.568.
- Heterogeneous agents by the sequence-space Jacobian method of Auclert, Bardóczy, Rognlie and
  Straub (2021), in a module of its own, `dsgelab.heterogeneous`; `Model` and `vec` are unchanged.
  A `HetBlock` holds a continuum of agents on a grid of one asset by the nodes of a Markov chain,
  with one step of their backward iteration written by the user, for example by the endogenous
  grid method with `interpolate_rows`, and the distribution moved by Young's (2010) lottery. The
  stationary distribution is found by one sparse solve. The block's Jacobians come from the
  fake-news algorithm, `T` backward steps per input, or, to check them, from the nonlinear path
  differentiated once for every date. `SimpleBlock` (and the decorator `simple`) holds equations
  in the aggregates with lags and leads, `x(-1)` and `x(+1)`, with Jacobians by the complex step,
  and `SolvedBlock` solves a model of blocks for some of its inputs. `BlockModel` joins the blocks
  by the names of their inputs and outputs, solves the steady state for unknowns that hit targets,
  and gives the Jacobians along the graph, the general-equilibrium Jacobians `G`, linear impulse
  responses and the nonlinear perfect-foresight transition by a quasi-Newton iteration. `log_grid`
  gives sequence-jacobian's asset grid. `validation/compare_heterogeneous.py` solves
  sequence-jacobian 1.0.0's own Krusell–Smith and one-asset HANK models, from its notebooks at the
  v1.0.0 tag, on 7 by 500 points: the steady states agree within 7.6e-10, the household
  Jacobians within 2.5e-10, `G` within 1.7e-10 and the linear and nonlinear impulse responses
  within 1.9e-10, relative to the largest value compared. The fake-news Jacobians agree with the
  direct ones within 4.1e-8, and Krusell–Smith's nonlinear transition approaches the linear
  response as the square of the shock. sequence-jacobian runs in an environment of its own,
  created by `validation/sequence_jacobian/setup.sh` with the versions pinned.

- `solve_markov_switching`: the first-order solution of a `Model` whose parameters switch with a
  Markov chain, from milestone 0.9. A switching parameter takes the value of the regime at `t` or,
  with `next_regime=`, at `t+1`. The approximation point is the steady state of the ergodic-mean
  parameters, and the parameters that move it are perturbed and enter as constants per regime,
  the partition principle of Foerster, Rubio-Ramírez, Waggoner and Zha (2016). The quadratic
  system of the minimum-state-variable solution (Farmer, Waggoner and Zha, 2011) is solved by
  Newton's method from the Klein solutions and from random points, in real or complex arithmetic
  (`msv_solutions`), and every distinct solution is kept, with its mean-square stability radius
  (`mean_square_stability`, Costa, Fragoso and Marques, 2005). `MarkovSwitchingSolution` gives
  impulse responses with the regime held, along a regime path or averaged over regimes,
  simulation of regimes and states, and exact unconditional and regime-conditional moments.
  `MarkovSwitchingError` reports when no stable solution is found, with what was found.
- `kim_filter`: the likelihood of a Markov-switching state space (`switching_state_space`) by
  Kim's (1994) filter, with filtered and smoothed regime probabilities, and
  `enumerated_likelihood`, the exact likelihood over every regime path on short samples.
  `MarkovSwitchingProblem` estimates switching parameters (`"psi[1]"`) and transition
  probabilities (`"p[0,1]"`) with `posterior_mode`, `maximum_likelihood`, `metropolis_hastings`
  and `smc`.
- RISE runs only in MATLAB, since its current release and its `v2019` tag are distributed as MATLAB
  P-code, which GNU Octave cannot execute, so the validation uses the references that exist without
  it. `validation/compare_switching.py` finds the nine solutions, three of them real, that Foerster
  et al. (2013) print for a New Keynesian model under two parameterizations, with the same one and
  two stable solutions. Of the 108 printed entries, 106 agree within a unit of the last printed
  digit, and the real part of the pair (4)–(5) in the second table is printed as -0.7000 where it is
  -0.66695 here. For the RBC of Foerster et al. (2016) it finds the four solutions they report, one
  of them stable, whose first-order coefficients agree in 23 of 24 entries once truncated to four
  decimals, as the paper prints them, and the constant of consumption in regime 1 is printed as
  0.000049 where it is 0.0000459 here. The script also checks closed forms, and the Klein solution
  when the regimes are identical. `validation/compare_kim.py` checks Kim's filter against
  statsmodels 0.14.4, in an environment of its own (`validation/statsmodels/setup.sh`). With an
  i.i.d. state it is Hamilton's filter, and the log likelihood, the filtered and the smoothed
  probabilities agree within 3e-14, 4e-16 and 2e-15; maximum likelihood in the model's parameters
  reaches statsmodels' maximum within 8e-10. In Hamilton's (1989) autoregression they agree within
  1e-9, 6e-11 and 4e-11. Against the exact likelihood over all regime paths, Kim's approximation
  errs by up to 0.018 on samples of 14 dates with large measurement errors, and by 9e-4 on the New
  Keynesian model with inflation observed.

### Changed

- `perfect_foresight`: when the model's residuals or derivatives raise, the exception keeps its
  type and message and carries a note with the dates whose equations failed, the Newton
  iteration, and the variables at both dates. A residual that solves an inner problem, such as
  cgelab's period equilibrium, used to fail without saying where on the path.
- `posterior_mode` and `maximum_likelihood` use the analytic gradient only for an
  `EstimationProblem`, so that they also accept a `MarkovSwitchingProblem`.
- `time_iteration` and `value_iteration` share the construction of the product Markov chain and of
  the shocks' quadrature. `time_iteration`'s results are unchanged.
- `time_iteration`'s `interpolation` defaults to `None`, which is `"linear"` on a `CartesianGrid`,
  so that a `SmolyakGrid` can refuse a value. `solve_projection`'s `bounds` defaults to `None`,
  for the call with `grid=`. Calls that worked before give the same results.

## 0.7.2 — 2026-10-31

The second part of the roadmap's milestone 0.8 (global methods): time iteration on a Cartesian
grid.

### Added

- `time_iteration`: a global solution by time iteration (Coleman, 1990) on the model's own
  residual, the one the perturbation solves. The rows are sorted into transitions and equilibrium
  conditions by what they contain at `t+1`. Exogenous states are given by a `MarkovChain`, whose
  nodes replace the state's law of motion, or are kept on the grid with the shocks integrated by
  Gauss–Hermite or by nodes and weights the user supplies. Bounds on the jumps, numbers or
  functions of today's states, are imposed by complementarity with the equilibrium condition in the
  same position, through the Fischer–Burmeister function, for borrowing limits and the zero lower
  bound. `TimeIterationSolution` evaluates and simulates the policy and gives the Euler-equation
  errors, as the natural residual of the complementarity. `CartesianGrid` holds the grid, with
  linear or natural cubic interpolation and linear extrapolation in both. The equations are called
  with every grid point at once when they accept arrays, and point by point otherwise.
  `validation/compare_time_iteration.py` solves three of dolo's own models (its RBC, its RBC with
  Markov productivity, and consumption and saving with a borrowing limit) on the same grids as dolo
  0.4.9.20 and agrees with its `time_iteration` within 1e-9, with linear and cubic interpolation.
  It also checks stochastic growth against its closed form, with errors that fall as the square of
  the spacing. dolo runs in an environment of its own, created by `validation/dolo/setup.sh` with
  the versions pinned.

## 0.7.1 — 2026-10-30

The first part of the roadmap's milestone 0.8 (global methods): the discretisation of a process
into a Markov chain.

### Added

- `tauchen` and `rouwenhorst`: an AR(1) as a Markov chain, by Tauchen's (1986) method and by
  Rouwenhorst's, with the recursion of Kopecky and Suen (2010). `MarkovChain` holds the nodes and
  the transition matrix, and gives the stationary distribution (GTH elimination), its mean,
  variance and autocorrelations, and simulated paths. Rouwenhorst's chain has the variance and the
  autocorrelations of the AR(1) exactly. Tauchen's probabilities are computed from the side of the
  conditional mean each interval lies on, so that small probabilities keep their relative accuracy
  in both tails. `validation/compare_markov.py` checks both against QuantEcon on 120 processes, to
  rounding, and Tauchen's probabilities and stationary distributions against a 40-digit
  computation, within 1.1e-11 and 1.1e-14. QuantEcon computes the upper tail as `1 - cdf`, and on
  the coarsest chains its stationary distribution is off by 3e-8.

### Fixed

- The release workflow waits up to 15 minutes, instead of three, for TestPyPI to list a new
  upload, and reads the index again on every try instead of reusing pip's cached page. On 0.7.0
  the wait ran out and the check had to be rerun by hand.
- The weekly validation job installs SALib and QuantEcon, the references of `compare_gsa.py` and
  `compare_markov.py`.

## 0.7.0 — 2026-10-30

The roadmap's milestones 0.6 (identification, moments and data) and 0.7 (complete third order and
derivatives), released together.

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
- `aggregation=` in `state_space` and `EstimationProblem`: observables seen at a lower frequency
  than the model's, as the mean or the sum of a variable over the last `every` periods, with the
  lags of the state stacked so that the filter, the smoother, the likelihood gradient, the shock
  decompositions and the forecasts work unchanged. `low_frequency` places the observations on the
  model's grid. `StateSpace` gains `aggregation` and `n_model_states`.
- `plot_irf`, `plot_paths`, `plot_decomposition`, `plot_forecast` and `plot_posterior`: matplotlib
  figures of impulse responses, paths, shock decompositions, forecasts with bands and posterior
  densities against the priors and the mode, with matplotlib as the optional `plot` extra.
  `prior_density` and `posterior_density` compute the two densities as Dynare's
  `draw_prior_density` and `posterior_moments` do.
- `validation/compare_mixed.py`: `fs2000.mod` with output growth observed only as its annual mean,
  against Dynare 7.2 with the equivalent auxiliary variable. Likelihood within 8.0e-13, smoothed
  variables within 1.3e-14, and a posterior mode with a kernel at least Dynare's.
- `validation/compare_plots.py`: the prior densities of `fs2000` against `draw_prior_density`
  within 1.9e-12, and the posterior kernel density against `posterior_moments` within 8.9e-16, on
  draws with and without repeats.
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
