# Comparison with other DSGE packages

This survey was carried out on 9 September 2026 and revised on 11 and 12 October 2026, after
versions 0.3 and 0.4 of `dsgelab`, which brought optimal policy, full Bayesian estimation, the
reading of `.mod` files, occasionally binding constraints, the extended path, declared trends and the
analytic steady state. Rows that changed in the October revisions are marked with ‡, and those from
9 September with †. The section [Gaps, also against tools in other
languages](#gaps-also-against-tools-in-other-languages) extends the comparison to Dynare
and MacroModelling.jl, and each gap there has a version in [`ROADMAP.md`](../ROADMAP.md).

The survey started from the question *was there already a package for DSGE and CGE in Python?*. On
the DSGE side such packages existed, and the table below compares five of them with `dsgelab`, with
Dynare as the reference. The CGE side has its own table, in the `cgelab` repository.

## How to read the tables

| Mark | Meaning |
|---|---|
| **✅** | verified by me, by installing the package and inspecting the API or the code. In the `dsgelab` column, implemented and tested in the package's suite, with the external validation cited in the cell when there is one |
| **○** | in the `dsgelab` column, implemented but not yet covered by a test in the package's suite |
| **📄** | stated in the official documentation; I did **not** verify it in the code |
| **❌** | tested here, and failed |
| **—** | absent as far as I checked |
| **?** | not checked |

I installed and inspected `gEconpy` (version 2.2.0) and `sequence-jacobian`. PyPI now has gEconpy
2.4.0, and versions after 2.2.0 were not re-checked. On the CGE side, I cloned and read the code of `pycge`, `PSLmodels/CGE` and `python-cge`. For `dolo`, `pydsge`, `econpizza` and `PyMacLab`
I read only the documentation, so a **📄** cell may be wrong without the table showing it.

---

# DSGE

`dsgelab` does not solve heterogeneous-agent models, which are the subject of **sequence-jacobian**
and **econpizza**. For global methods beyond polynomial projection, which in `dsgelab` goes up to
three states, **dolo** documents time, value and expectations iteration. **gEconpy** estimates
through PyMC and its gradient-based samplers. `dsgelab` has its own HMC and NUTS, driven by the
analytic gradient of the first-order Kalman likelihood, besides Metropolis–Hastings and SMC. At
second and third order it estimates with a particle filter, and at second order also with an
inversion filter. Dynare's reduced-form mapping and RMSE filtering still require **Dynare**, and
Markov regime switching requires **RISE**.

| Capability | dsgelab | gEconpy 2.2 | dolo | pydsge | econpizza | PyMacLab | Dynare (ref.) |
|---|---|---|---|---|---|---|---|
| **Model declaration** | residual function, text equations, optimisation problem † or Dynare `.mod` ‡ | ✅ `.GCN` file by optimisation blocks | 📄 YAML file | 📄 dolo's format | 📄 YAML | 📄 its own model file | `.mod` |
| **Derives the FOCs from the agent's problem** | ✅ † declared Lagrangian | ✅ | — | — | — | — | — |
| Numerical steady state | ✅ | ✅ | 📄 | 📄 | 📄 | 📄 | ✅ |
| Declared analytic steady state | ✅ ‡ checked against the model's residual before use, and against Dynare on `fs2000` at 1e-16 | ✅ `STEADY_STATE` block | ? | ? | ? | 📄 | ✅ |
| **1st-order perturbation** | ✅ Klein/QZ | ✅ | 📄 | 📄 | — | 📄 Klein | ✅ |
| **2nd-order perturbation + risk corrections** | ✅ | **—** | 📄 | — | — | 📄 Klein–Gomme | ✅ |
| 3rd-order perturbation | ✅ ‡ full decision rule in Dynare's form (`ghxxx`, `ghxxu`, `ghxuu`, `ghuuu`, `ghxss`, `ghuss`), against Dynare 7.2 on `example1` and `example2` to 3e-13, and the stochastic steady state. Sparse tensors, checked on a synthetic model with 71 variables | — | 📄 | — | — | — | ✅ |
| Global methods / projection | ✅ ‡ Chebyshev collocation with up to three states, which reproduces the closed-form solution of stochastic growth | — | 📄 time, value, expectations iteration | — | 📄 | — | — |
| Discretisation of an AR(1) (Tauchen, Rouwenhorst) | ✅ ‡ against QuantEcon on 120 processes, Tauchen's probabilities against a 40-digit computation and Rouwenhorst against its closed form | ? | ✅ `tauchen`, `rouwenhorst` | ? | ? | ? | ✅ Rouwenhorst options in the heterogeneous-agent module (`+heterogeneity`) |
| Heterogeneous agents (HANK) | — | — | — | — | 📄 | — | 📄 |
| Occasionally binding constraints | ✅ ‡ OccBin, against Dynare's `+occbin` at 2e-15, with one and with two constraints | — | 📄 | 📄 ZLB | 📄 | — | ✅ OccBin |
| Unit root accepted (`qz_criterium`) | ✅ ‡ optional; strict by default | ? | ? | ? | ? | ? | ✅ by default |
| **Nonlinear perfect foresight** | ✅ stacked-time Newton | ✅ `solve_perfect_foresight` | 📄 | ? | 📄 shooting | ? | ✅ |
| Announced parameter path | ✅ | ✅ `param_paths` | ? | ? | ? | ? | 📄 |
| Announced path of innovations | ✅ ‡ | ? | ? | ? | ? | ? | ✅ |
| Perfect foresight with expectation errors | ✅ ‡ against Dynare | ? | ? | ? | ? | ? | ✅ |
| IRFs | ✅ | ✅ | ? | 📄 | 📄 | 📄 | ✅ |
| Orthogonalised IRFs (Cholesky) | ✅ | ✅ `orthogonalize_shocks` | ? | ? | ? | ? | ✅ |
| Moments via Lyapunov | ✅ | ✅ `stationary_covariance_matrix` | ? | ? | — | 📄 | ✅ |
| Autocorrelations | ○ | ✅ `autocorrelation_matrix` | ? | ? | — | 📄 | ✅ |
| Variance decomposition | ✅ | ? | ? | ? | — | ? | ✅ |
| Simulation with pruning | ✅ | — (1st order only) | ? | ? | — | ? | ✅ |
| 2nd-order generalised impulse response | ○ ‡ | — | ? | ? | — | ? | ✅ |
| Historical decomposition and conditional forecast | ✅ ‡ on the linear solution | ? | — | ? | — | — | ✅ |
| Global sensitivity | ✅ ‡ stability mapping and Monte Carlo filtering against Dynare's `+gsa`, Sobol indices against SALib | ? | — | — | — | — | ✅ `+gsa`, with reduced-form mapping |
| Out-of-sample forecast evaluation | ✅ ‡ recursive and rolling re-estimation, RMSE against the random walk, Diebold–Mariano, PIT, coverage and log score; the recursive forecasts against Dynare on `fs2000`, the test against R's `forecast::dm.test` | ? | — | ? | — | — | ✅ recursive estimation (`nobs=[T1:T2]`) |
| **Optimal policy: Ramsey under commitment** | ✅ ‡ timeless, against Dynare, with IRFs to 5e-13 and the steady state to 5e-11 | — | — | — | — | — | ✅ |
| Optimal simple rules, discretion | ✅ ‡ against Dynare | — | — | — | — | — | ✅ |
| 2nd-order welfare, conditional and unconditional | ✅ ‡ against Dynare, to 1.4e-12 under Ramsey and 3.5e-13 in an RBC with capital (6.1e-10 under discretion) | — | — | — | — | — | ✅ |
| **Bayesian estimation** | ✅ ‡ mode, Laplace and Metropolis against Dynare on `fs2000`, SMC and maximum likelihood against the conjugate case, R-hat, ESS. HMC and NUTS at first order, with the likelihood gradient checked against finite differences and the samplers against the exact posterior of an AR(1) coefficient. Particle filter up to third order, against Dynare's likelihood on an AR(1) to 0.006, and inversion filter up to second order, against Dynare's OccBin filter on the same AR(1) | ✅ via PyMC | — | 📄 with ZLB | — | — 📄 states it has none | ✅ |
| Kalman filter / smoothing | ✅ ‡ filter with exact diffuse start and smoother, against Dynare (likelihood at 6e-12, smoother at 8.4e-15) | ✅ `statespace_from_gcn` | — | 📄 | — | — | ✅ |
| **Calibration on targets** | ✅ joint, with bounds | ✅ parameters calibrated to target ratios | ? | ? | ? | ? | 📄 |
| **Identification diagnostic for the calibration** | ✅ rank and null directions | ? | ? | ? | ? | ? | 📄 (model identification) |
| **Local identification of the model** | ✅ ‡ moments, spectrum and minimal system, against Dynare | ? | ? | ? | ? | ? | ✅ |
| Blanchard–Kahn | ✅ **enforced**, with margin | ✅ `check_bk_condition` | ? | ? | — | ? | ✅ reported |
| **Order-of-accuracy check** (`accuracy()`) | ✅ orders 2, 3 and 4 | ? | ? | ? | ? | ? | — |
| Export Dynare `.mod` | ✅ † round trip below 1e-13 | ✅ `make_mod_file` | 📄 | — | — | — | — |
| **Read Dynare `.mod`** | ✅ ‡ nine Dynare `.mod` files read without porting | — | — | — | — | — | n/a |
| Dependencies | numpy, scipy (sympy optional) | sympy, pytensor, PyMC, preliz | numpy, scipy, pandas (old pin) | dolo, numpy | **JAX** | numpy, scipy | — |
| Installs today (Python 3.14) | ✅ | ✅ | ❌ fails: pins a pandas that does not build | ? | ? | ❌ the sdist has neither `setup.py` nor `pyproject.toml` | n/a |
| Licence | MIT | GPL-3.0 | BSD-2 | ? | ? | ? | GPL |

### What the table says

**`gEconpy` is the package whose design comes closest to `dsgelab`.** Both derive the first-order
conditions from a declared problem, accept an analytic steady state, estimate by Bayesian methods and
export to Dynare. Of the 37 rows of the table, 10 mark gEconpy as absent (—) in a capability that
`dsgelab` has, among them second and third order, occasionally binding constraints and optimal
policy, and none records the reverse, while in another 7 the gEconpy capability was not checked (?).
The table leaves out one gEconpy feature that `dsgelab` does not offer, declaration by blocks,
which `assemble` covers only within the limits of `Problem`.

**`PyMacLab` solves the first order by Klein's method, as `dsgelab` does,** and the second by that of
Klein and Gomme, according to its documentation, which dates from January 2013. Among the packages in
the table, it is the only one that states it has no estimation. It and `dolo` did not install on
Python 3.14, and in its case the reason is that the 0.95.9 package on PyPI comes without a build
file, either `setup.py` or `pyproject.toml`.

**`dolo` documents methods absent from `dsgelab`,** such as time iteration and value iteration, as
well as perturbation up to third order. Installation failed because the package pins a version of
pandas that no longer builds, and I did not test Python versions earlier than 3.14.

**`econpizza` and `sequence-jacobian` take heterogeneous agents as their subject,** which the
`ROADMAP.md` plans for version 0.9 of `dsgelab`. For representative-agent models, the table records
for `econpizza`, from its documentation, perfect foresight, IRFs, occasionally binding constraints
and a global method.

**The first survey did not include a PyPI package called `dsgekit`,** *"A Python toolkit for
DSGE models"*, by Gustavo Vargas (0.1.0rc1). It was neither installed nor read, and for that reason it
is not in the table. Because of it, this package, previously called `dsgekit`, was renamed `dsgelab`
on 11 October 2026.

### What changed, and what is still missing

Since the survey of 9 September, `dsgelab` has come to derive FOCs from a declared Lagrangian, to
solve at third order, to export to Dynare and MacroModelling.jl, to estimate by Bayesian methods and
to solve by projection. When the Dynare exporter is checked, the package writes the `.mod`, Dynare
solves it, and the two solutions differ by less than 1e-13. For each of the three items below, the scope of the
feature is that of the test that validates it.

* **Third order, with the risk slope.** ``g_kkk`` and ``h_kkk`` are checked by the order of
  accuracy. ``h_ssk`` and ``g_ssk`` come from the curvature of the rule with respect to the
  innovation, contracted with ``Σ``. In the AR(1) in logs, ``h_ssk[a, a] = ρ σ²``, with positive
  sign, which is the cross term of ``(1+â)^ρ exp(σ²/2)``, and the capital row, whose rule is exact,
  comes out at zero. The object computed is the ``σ² k`` term of the decision rule. With version
  0.7, the rule takes the innovation as an argument in every monomial, in Dynare's form, and
  ``ghxxx``, ``ghxxu``, ``ghxuu``, ``ghuuu``, ``ghxss`` and ``ghuss`` agree with Dynare 7.2 on
  `example1` and `example2` to 3e-13. ``stochastic_steady_state`` returns the fixed point of the
  unpruned rule, which differs from the ergodic mean of the pruned system.
* **Bayesian estimation of the linear model.** The likelihood comes from the Kalman filter on the
  first-order solution, present since 0.2 together with the posterior mode and random-walk
  Metropolis–Hastings. Version 0.3 added the smoother, the exact diffuse start for unit roots,
  maximum likelihood, SMC and the marginal likelihood by Laplace and by modified harmonic mean.
  Against Dynare's `estimation` on `fs2000.mod`, the likelihood agrees to 6e-12, the prior density to
  4e-13 and the Laplace approximation, by the same formula, to 2e-13. Version 0.7 adds the analytic
  gradient of the Kalman likelihood, used by HMC and NUTS, and two nonlinear filters. The
  particle filter estimates up to third order and matches Dynare's likelihood on an AR(1) to 0.006.
  At first and second order, the inversion filter reproduces the shocks and the likelihood of
  Dynare's OccBin filter on that AR(1).
* **Projection by Chebyshev collocation.** The residual used in the perturbation is collocated on a
  tensor grid of up to three states. Rows without a future jump determine the next state for each
  shock, and the others are integrated by Gauss–Hermite. For stochastic growth with a
  closed-form solution, the policy reproduces ``c = (1-αβ) a k^α``. A sparse grid, value iteration
  and more than two shocks in the tensor quadrature are outside this solver.

Versions 0.3 and 0.4 also brought optimal policy, the reading of `.mod` files, occasionally binding
constraints, the extended path, declared trends, the analytic steady state and model diagnostics,
each validated as the table in the following section indicates. Version 0.6 adds local
identification, the method of moments, the moments of filtered variables, calibration
restrictions, out-of-sample forecast evaluation, global sensitivity, mixed-frequency data and
plots. Still absent are global methods beyond Chebyshev collocation, regime switching,
heterogeneous agents, BVAR, DSGE-VAR and the declaration of trends in `from_equations`. The README lists
them under *Known limitations*, and `ROADMAP.md` gives the version for each.

### How `dsgelab` differs from the other Python packages

`dsgelab` was not the first DSGE package in Python, since the PyMacLab documentation dates from 2013,
and the following section lists what Dynare offers that `dsgelab` does not yet. The points below are
those that set it apart from the Python packages in the table.

1. **From the agent's problem to the `.mod`, without manual porting.** The Lagrangian is declared,
   and the package derives the FOCs, solves the model and exports it to Dynare or to
   MacroModelling.jl, translating the timing of stock variables. The round trip through Dynare, below
   1e-13, checks this translation, which was the step where the manual port of `RBC_baseline` went
   wrong (see [`validation/README.md`](../validation/README.md)). Conversely,
   `from_mod` reads the `.mod` without porting.
2. **The same residual function serves perturbation, perfect foresight, the extended path and
   projection.** I did not audit the code of the other packages and so do not claim that they lack
   this. It was the concrete problem in the repository that motivated the package, where the
   perturbation and the transition rested on different equations.
3. **Second order, the third-order risk slope, OccBin and the extended path run in process,** without
   calling out to Octave and Dynare. Of the five other packages compared, gEconpy does not solve at
   second order, and dolo and PyMacLab, which document it, did not install on Python 3.14.
4. **The library enforces some checks and provides others as calls.** Blanchard–Kahn is enforced,
   and the solution is refused when the condition fails. The steady state is accepted on its
   residual, whatever the solver's success flag says. The order of accuracy (`accuracy()`), the
   identities that the model does not impose (`check_identity`) and the identification of the
   calibration by rank (`target_jacobian`) are separate calls, and `diagnose` reproduces the report
   of Dynare's `model_diagnostics`.
5. **Two required dependencies, numpy and scipy.** The sympy package is optional, and only the
   symbolic front end imports it (text equations, derivation of FOCs, reading of `.mod` files,
   exporters, Ramsey, discretionary policy, second-order welfare and OccBin). The package does not depend on PyMC, JAX or pytensor.
6. **The external validations are scripts in [`validation/`](../validation/README.md) that anyone
   can run.** `validation/run_all.sh` runs them all against Dynare 7.2 and MacroModelling.jl, with
   pinned versions, on `.mod` files taken from Dynare's own examples and tests and on a few written
   in this repository (`nk_linear.mod`, `rbc_labortax.mod` and `nk_osr_planner.mod` in
   `validation/ramsey/`, `validation/estimation/unitroot.mod` and the three models in
   `validation/diagnostics/`). Projection, which Dynare does not compute, is checked against a
   closed form in the test suite. Third order is checked there against closed forms and here
   against Dynare's decision rule on `example1` and `example2`.

A model written in `dsgelab` reaches Dynare through `to_dynare`, checked by round trip, when it needs
what `dsgelab` does not have yet, such as the reduced-form mapping of the sensitivity toolbox
(`redform`) or identification over draws from the prior (`prior_mc`). The `to_macromodelling` exporter writes the
same model for MacroModelling.jl, without a round-trip check.

---

# Gaps, also against tools in other languages

The table above compares `dsgelab` with Python packages, and this section extends the comparison to
what exists in any language. The list of items was checked against the subsystems and function files
of Dynare 7.1 (`lib/dynare/matlab/`) and against the list of exported functions of
MacroModelling.jl, in the versions installed at the time of the survey, and validation now runs on
Dynare 7.2. Since 11 October 2026 the goal has been to cover **all** of this list by 1.0, and the *Version*
column gives the [`ROADMAP.md`](../ROADMAP.md) milestone in which each item arrived or will arrive.

| Item | Who has it | Status in `dsgelab` | Version |
|---|---|---|---|
| Ramsey (commitment) | ✅ Dynare (`ramsey_model`) | **done**: `ramsey_model`, against Dynare on three models | 0.2 |
| Optimal simple rules and discretion | ✅ Dynare (`+osr`, `discretionary_policy`) | **done**: `osr`, `discretionary_policy`, against Dynare's tests | 0.2 |
| Second-order welfare and equivalent variation | ✅ Dynare (`evaluate_planner_objective`) | **done**: `welfare`, `consumption_equivalent` | 0.2 |
| Second-order generalised impulse response | ✅ MacroModelling (`get_girf`) | **done** | 0.2 |
| Sequences of innovations in perfect foresight | ✅ Dynare (`perfect_foresight_solver`) | **done**, against Dynare at 1e-13 | 0.2 |
| Expectation errors in perfect foresight | ✅ Dynare (`perfect_foresight_with_expectation_errors_solver`) | **done**, at 3e-12 | 0.2 |
| Kalman, mode, Metropolis | ✅ Dynare (`+estimate`), MacroModelling, gEconpy | **done**, against Dynare on `fs2000.mod` | 0.3 |
| Smoother, diffuse filter, maximum likelihood, SMC, diagnostics | ✅ Dynare | **done**, with smoother and diffuse filter against Dynare | 0.3 |
| Marginal likelihood and model comparison | ✅ Dynare | **done**: Laplace, modified harmonic mean, SMC | 0.3 |
| Historical decomposition and conditional forecast | ✅ Dynare (`shock_decomposition`, `+conditional_forecasts`), MacroModelling | **done**, including in real time, against Dynare | 0.3 |
| Out-of-sample forecast evaluation | ✅ Dynare (recursive estimation, `nobs=[T1:T2]`, `forecast`), R (`forecast::dm.test`) | **done**: `forecast_evaluation`, recursive or rolling, with RMSE against the random walk, Diebold–Mariano, PIT, coverage and log score; forecasts at Dynare's modes on `fs2000` within 3e-15, and the test as R's within 3e-15 | 0.6 |
| **Read `.mod`** | ✅ MacroModelling (`translate_mod_file`) | **done**: `from_mod`, nine Dynare `.mod` files | 0.4 |
| Arbitrary lags and leads | 📄 Dynare, 📄 MacroModelling | **done**, in the `.mod` and in `from_equations`, with every auxiliary listed | 0.4 |
| Occasionally binding constraints | ✅ Dynare (`+occbin`) | **done**: `occbin_simulate`, with one and two constraints at 2e-15 and regime histories identical to Dynare's | 0.4 |
| Extended path | ✅ Dynare (`extended_path`) | **done**, deterministic and stochastic (sparse and full tree, order-2 hybrid), within 1.1e-11 of Dynare | 0.4 |
| Declared analytic steady state | ✅ Dynare (`steady_state_model`), ✅ gEconpy | **done**, checked against the model before use and refused as Dynare refuses it | 0.4 |
| Calibration equations in the model | ✅ MacroModelling | **done**, against MacroModelling at 4.4e-15 | 0.4 |
| Homotopy | ✅ Dynare (`homotopy_setup`) | **done**, with adaptive step, and reaches the closed form where direct Newton fails | 0.4 |
| Declared trends and balanced growth | ✅ Dynare (`trend_var`, `deflator`, `+bgp`) | **done**: detrending with Dynare's check, and the `bgp.write` problem, with the same residuals that Dynare generates | 0.4 |
| Model diagnostics | ✅ Dynare (`model_diagnostics`) | **done**: `diagnose`, same report as Dynare on a deliberately broken model | 0.4 |
| Several agents in one equilibrium | 📄 gEconpy (blocks) | **done**: `assemble`, within the limits of `Problem` | 0.4 |
| Local identification | ✅ Dynare (`+identification`) | **done**: `identification`, the moments, spectrum and minimal-system criteria and the collinearity search, with the same verdicts as Dynare on three models and derivatives within 2e-8 | 0.6 |
| Global sensitivity | ✅ Dynare (`+gsa`), SALib | **done**: `stability_map`, `monte_carlo_filtering` and `sobol`, with the classes of Dynare's solver on 3 × 1024 draws of `ls2003a`, its Smirnov tests and correlations within 2e-13, and SALib's Sobol design and indices within 1e-15; reduced-form mapping and RMSE filtering missing | 0.6 |
| Method of moments and restrictions on IRFs and moments | ✅ Dynare (`+mom`, `irf_calibration`, `moment_calibration`) | **done**: GMM and SMM at first and pruned second order, frequentist IRF matching at first order, and calibration restrictions on IRFs and moments at first order, at a point and over the prior, with the same values and verdicts as Dynare on `ls2003a` | 0.6, 0.7 |
| Filtered moments | ✅ Dynare (`hp_filter`, `bandpass_filter`) | **done**: Hodrick–Prescott and band-pass, against Dynare on five models within 1e-13, a unit root included | 0.6 |
| Mixed frequency and plots | ✅ Dynare, gEconpy, MacroModelling | **done**: `aggregation=` in the state space and `EstimationProblem`, against Dynare's auxiliary-variable formulation on `fs2000` (likelihood within 8e-13); `plot_irf`, `plot_paths`, `plot_decomposition`, `plot_forecast` and `plot_posterior`, with Dynare's prior and posterior densities within 2e-12 | 0.6 |
| Third order in Dynare's decision-rule form and stochastic steady state | ✅ Dynare, MacroModelling | **done**, including all cubic blocks, risk terms and the stochastic fixed point | 0.7 |
| Sparse third order, derivatives with respect to parameters, nonlinear filters | 📄 Dynare (k-order), ✅ MacroModelling | **done**: canonical sparse third order validated on a full 71-variable nonlinear pipeline, implicit first-order QZ derivatives and nonlinear filters; explicit decision-rule composition uses dense arrays | 0.7 |
| Discretisation of processes (Tauchen, Rouwenhorst) | ✅ dolo, QuantEcon | **done**: `tauchen`, `rouwenhorst` and `MarkovChain`, against QuantEcon on 120 processes, with Tauchen's small probabilities and stationary distributions against a 40-digit computation (QuantEcon loses them in `1 - cdf`) | 0.7.1 |
| Global methods | 📄 dolo | Chebyshev collocation up to three states; time iteration, value function iteration, the endogenous grid method and sparse grids missing | 0.7.2–0.8 |
| Regime switching, heterogeneous agents, BVAR and DSGE-VAR | 📄 RISE, ✅ Dynare, sequence-jacobian | missing | 0.9 |

---

## Sources

- [gEconpy](https://github.com/jessegrabowski/gEconpy), installed (2.2.0) and inspected, with later versions not re-checked
- [sequence-jacobian](https://github.com/shade-econ/sequence-jacobian), installed and inspected
- [dolo](https://github.com/EconForge/dolo.py) and [its documentation](https://www.econforge.org/dolo.py/)
- [pydsge](https://github.com/gboehl/pydsge) and [its docs](https://pydsge.readthedocs.io/en/latest/index.html)
- [econpizza](https://github.com/gboehl/econpizza)
- [PyMacLab](https://pythonhosted.org/pymaclab/) (documentation from January 2013)
- [Dynare](https://www.dynare.org/) 7.2, installed (validation previously passed on 7.1, with the same numbers up to rounding noise); the validation `.mod` files come from tags 6.4 and 7.2
- [MacroModelling.jl](https://github.com/thorek1/MacroModelling.jl), installed
