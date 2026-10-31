# dsgelab

[![tests](https://github.com/GustavoHM/dsgelab/actions/workflows/tests.yml/badge.svg)](https://github.com/GustavoHM/dsgelab/actions/workflows/tests.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue)](https://github.com/GustavoHM/dsgelab/blob/main/LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23456474.svg)](https://doi.org/10.5281/zenodo.23456474)
[![site](https://img.shields.io/badge/site-gustavohm.github.io%2Fdsgelab-8a4b2a)](https://gustavohm.github.io/dsgelab/)

A general toolkit for solving DSGE models in Python. A model is declared once, as lists of
state, jump and shock names and one function that returns the equilibrium residuals. The
package computes the steady state, the first-order solution, impulse responses, moments,
second-order risk corrections, nonlinear perfect-foresight transitions and calibration on
targets from that declaration.

The package is built so that the stochastic block and the deterministic transition are the
*same model*. One way applied work fails is a perturbation solution built on one set of
equations beside a transition path solved by a hand-written fixed point on another set, similar
to the first but not identical. Here both call the same residual function, so the equations
behind them cannot differ.

```
pip install dsgelab
pip install "dsgelab[symbolic]"     # adds text equations with exact derivatives (sympy)
pip install "dsgelab[plot]"         # adds the plots (matplotlib)
```

Requires Python 3.11 or later, numpy and scipy. Pandas is not needed.

## Quick start

```python
import dsgelab as dsg
from dsgelab.examples import rbc

model, params, guess = rbc.build()
sol = dsg.solve(model, params, guess=guess)

print(sol.summary())
irf = sol.irf("eps_a", size=0.01, periods=40)      # per cent deviations
mom = sol.moments(cov={"eps_a": 0.007})            # std, corr, autocorr
```

A model of your own takes three lists and a function. The parameters can be any object with
attributes, here a dataclass.

```python
import numpy as np
from dataclasses import dataclass
from dsgelab import Model, solve

@dataclass
class P:
    alpha: float = 0.36
    beta:  float = 0.99
    delta: float = 1.0
    gamma: float = 1.0
    rho:   float = 0.95

def equations(nxt, cur, eps, p):
    y  = cur.a * cur.k ** p.alpha
    y1 = nxt.a * nxt.k ** p.alpha
    return [
        nxt.k - ((1 - p.delta) * cur.k + y - cur.c),                       # resource
        cur.c ** -p.gamma - p.beta * nxt.c ** -p.gamma                     # Euler
            * (p.alpha * y1 / nxt.k + 1 - p.delta),
        np.log(nxt.a) - p.rho * np.log(cur.a) - eps.eps_a,                 # technology
    ]

model = Model(states=["k", "a"], jumps=["c"], shocks=["eps_a"], equations=equations)
sol = solve(model, P(), guess={"k": 0.2, "a": 1.0, "c": 0.36})
```

Variables are addressed by *name*, in `cur` for period `t` and `nxt` for `t+1`. When
variables are read from a vector by position, swapping one index for another raises no error,
and the program solves a different model from the one intended. Reading each variable by name
rules that mistake out.

The [user guide](https://github.com/GustavoHM/dsgelab/blob/main/docs/guide.md) goes from the
declaration to second order, and the [site](https://gustavohm.github.io/dsgelab/) has runnable
examples in English and Portuguese.

## Terms

- A *state* (predetermined variable) has its value for period `t` fixed before the innovation
  of period `t` arrives, as capital and the technology level do. A *jump* (non-predetermined
  variable), such as consumption, moves within the period in response to that innovation.
- *Klein/QZ* is Klein's method for the linearised model, which orders the generalised Schur
  (QZ) decomposition of the matrix pair `(B, A)` of the canonical form below.
- The *risk correction* is the second-order term that shifts the decision rules in proportion
  to the variance of future shocks. At first order it is zero.
- *Pruning* drops, when the second-order rule is simulated, the terms of order higher than two
  that the recursion would otherwise accumulate, so that simulated paths do not diverge.

## What is in the box

| Capability | Entry point |
|---|---|
| Steady state, with positivity and outer anchors | `steady_state` |
| First-order solution (Klein/QZ, Blanchard–Kahn enforced) | `solve` |
| Parameter Jacobians of the first-, second- and third-order solutions and steady state | `solution_jacobian`, `second_order_jacobian`, `third_order_jacobian` |
| Impulse responses, simulation, moments, variance decomposition | `FirstOrderSolution` |
| Second-order perturbation, innovation tensors and risk corrections | `solve_second_order`, `SecondOrderSolution.decision_rule` |
| Third-order perturbation: full decision rule in Dynare's form, stochastic steady state, sparse tensors | `solve_third_order`, `ThirdOrderSolution.decision_rule`, `stochastic_steady_state` |
| Global solution by Chebyshev collocation, up to three states | `solve_projection` |
| Nonlinear perfect-foresight transitions, announced policy and shock paths | `perfect_foresight` |
| Perfect foresight with expectation errors (news, surprises) | `perfect_foresight_with_expectation_errors` |
| Ramsey optimal policy under commitment (timeless), validated against Dynare | `ramsey_model` |
| Optimal policy under discretion; optimal simple rules, on moments or on welfare | `discretionary_policy`, `osr` |
| Kalman filter (exact diffuse start) and smoother; measurement error; missing data | `state_space`, `kalman_filter`, `kalman_smoother` |
| Bootstrap particle filter and smoother up to pruned third order; inversion filter up to pruned second order | `particle_filter`, `particle_smoother`, `inversion_filter`, `EstimationProblem(order=2)` |
| Bayesian estimation: mode, Laplace, Metropolis-Hastings, SMC, R-hat, ESS, marginal likelihood; maximum likelihood | `EstimationProblem`, `posterior_mode`, `metropolis_hastings`, `smc` |
| Gradient-based sampling (HMC, NUTS) on the analytic gradient of the Kalman likelihood | `hmc`, `nuts`, `log_likelihood_gradient` |
| Local identification at a point (Iskrev; Komunjer and Ng; Qu and Tkachenko) and collinearity analysis, validated against Dynare's `identification` | `identification`, `ModFile.identification` |
| GMM and SMM at first and pruned second order; frequentist IRF matching at first order | `method_of_moments`, `irf_matching`, the `ModFile` methods |
| Observed-shock local projections with joint HAC covariance across variables and horizons | `local_projection` |
| Theoretical moments of HP- and band-pass-filtered variables, unit roots included | `FirstOrderSolution.moments(hp_filter=, bandpass=)` |
| Shock decomposition (also real-time), forecasts with bands, conditional forecasts | `shock_decomposition`, `forecast`, the method `FirstOrderSolution.conditional_forecast` |
| Out-of-sample forecast evaluation: recursive or rolling re-estimation, RMSE against the random walk, Diebold–Mariano, PIT, coverage | `forecast_evaluation`, `diebold_mariano` |
| Mixed-frequency observables (means or sums over several periods) | `state_space(aggregation=)`, `EstimationProblem(aggregation=)`, `low_frequency` |
| Calibration restrictions on IRFs and moments, at a point and over the prior | `check_restrictions`, `sample_restrictions` |
| Global sensitivity: stability mapping, Monte Carlo filtering, Sobol indices | `stability_map`, `monte_carlo_filtering`, `sobol` |
| Plots of impulse responses, paths, decompositions, forecasts and posteriors (matplotlib, optional) | `plot_irf`, `plot_paths`, `plot_decomposition`, `plot_forecast`, `plot_posterior` |
| Welfare at second order, conditional and unconditional; consumption equivalent | `welfare`, `consumption_equivalent` |
| Occasionally binding constraints (OccBin), any number of them, surprise shocks | `OccbinModel`, `occbin_simulate` |
| Extended path, deterministic and stochastic (sparse or full tree, hybrid correction) | `extended_path` |
| Models in levels: declared trends with Dynare's balanced-growth check; the growth path | `from_mod`, `balanced_growth` |
| Read a Dynare `.mod` file, macros and leads/lags included | `from_mod` |
| Export a model to Dynare and to MacroModelling.jl | `to_dynare`, `to_macromodelling` |
| Calibration on targets, with an identification diagnostic; calibration equations inside the model | `calibrate`, `target_jacobian`, `from_equations(calibration_equations=)` |
| Closed-form steady states, checked before use; homotopy for hard steady states | `analytic_steady_state`, `steady_state(homotopy=)` |
| Model diagnostics (Dynare's `model_diagnostics`) and independent checks | `diagnose`, `dsgelab.diagnostics` |
| Text equations with exact analytic derivatives, lags and leads of any order | `dsgelab.symbolic` |
| Several agents' optimisation problems in one equilibrium | `assemble` |

## Timing convention

`eps` is the innovation dated `t+1`, unknown at `t`, with `E_t eps_{t+1} = 0`. An AR(1) is
written as the residual

```python
def equations(nxt, cur, eps, p):
    return [nxt.a - p.rho * cur.a - eps.eps_a]
```

and the resulting law of motion is `k_{t+1} = P k_t + Q eps_{t+1}`. Treating the innovation as
known at `t` instead makes the model deterministic within the period, and every second-order
risk correction then comes out zero without a warning. Of the package's conventions, this is
the only one it cannot check in a user's model.

Two checks run on every model without being requested. `solve` raises `BlanchardKahnError`
when the Blanchard–Kahn condition fails, and the steady state is accepted only when it zeroes
the residuals. Three others take one call each by the user. `check_identity` tests an
identity such as Walras's law, `sol.accuracy()` reports the order of accuracy of a solution,
and `target_jacobian` checks whether calibration targets identify the free parameters, which
`calibrate` does not check by itself.

## Canonical form

```
A E_t x_{t+1} = B x_t + C eps_{t+1},   x_t = [k_t ; u_t]
u_t     = F k_t
k_{t+1} = P k_t + Q eps_{t+1}
```

with `k` the predetermined variables (states) and `u` the non-predetermined ones (jumps). The
Blanchard–Kahn condition requires the number of stable generalised eigenvalues of the matrix
pair `(B, A)` to equal the number of states. When it fails, the equilibrium is indeterminate
or does not exist, and `solve` raises `BlanchardKahnError` instead of returning `F` and `P`.

## Documentation

- [`docs/whitepaper.md`](https://github.com/GustavoHM/dsgelab/blob/main/docs/whitepaper.md), the mathematics the solvers implement
- [`docs/guide.md`](https://github.com/GustavoHM/dsgelab/blob/main/docs/guide.md), the user guide, from the declaration to second order
- [`docs/api.md`](https://github.com/GustavoHM/dsgelab/blob/main/docs/api.md), the API reference
- [`docs/design-notes.md`](https://github.com/GustavoHM/dsgelab/blob/main/docs/design-notes.md), the numerical decisions and the errors each one avoids
- [`docs/from-scripts.md`](https://github.com/GustavoHM/dsgelab/blob/main/docs/from-scripts.md), porting an existing research script
- [`docs/comparison.md`](https://github.com/GustavoHM/dsgelab/blob/main/docs/comparison.md), how this package compares with the existing ones, and what it lacks
- [`ROADMAP.md`](https://github.com/GustavoHM/dsgelab/blob/main/ROADMAP.md), what is still missing, and when it is planned
- [`CONTRIBUTING.md`](https://github.com/GustavoHM/dsgelab/blob/main/CONTRIBUTING.md), how to report a problem and what a contribution needs
- [The site](https://gustavohm.github.io/dsgelab/), with runnable examples in English and Portuguese (source in [`site/`](https://github.com/GustavoHM/dsgelab/tree/main/site))

Questions and bug reports go to the [issue tracker](https://github.com/GustavoHM/dsgelab/issues).

## Heutel environmental-policy example

[`docs/heutel.md`](https://github.com/GustavoHM/dsgelab/blob/main/docs/heutel.md) documents the
first-order port of Heutel's (2012) MATLAB replication scripts, including the optimal tax and
the tax and quota rules with lagged information. The port is checked against the original
algebra, run in Octave, and against impulse responses and moments solved independently in
Dynare.

```python
from dsgelab.examples import heutel
heutel.fetch()          # once per machine; needs an internet connection
sol = heutel.solve("tax")
irf = sol.irf("eps_a", size=0.01, periods=100)
```

The planner script's sensitivity calibration fails the admissible steady-state check, so the
Table 2 baseline has to be requested explicitly, with `heutel.solve("planner", heutel.Params())`.
Without it, `heutel.solve("planner")` raises `heutel.SteadyStateError`. The documentation lists
the differences between the supplied code and the published paper.

The port's scalar algebra is translated from Heutel's MATLAB files, which carry no licence, so
it is not distributed with the package. `heutel.fetch()` downloads the public copy from Georgia
State University's repository, checks each file against the SHA-256 of the files the port was
validated on, and writes the translation to `~/.cache/dsgelab/heutel` (or to
`DSGELAB_HEUTEL_DIR`).

## Verification

Each feature is validated against an independent reference where one exists, which may be
another implementation, a closed form or a published result. A few rest on internal tests
only, namely autocorrelations, variance decomposition, pruning, the second-order generalised
impulse response, announced parameter paths and the Blanchard–Kahn check, and the export to
MacroModelling.jl has no round trip. The checks come in three layers.

The first layer is the test suite (`pytest`), which compares the output with answers known
analytically. The full suite currently has 184 tests. Some need the `symbolic` extra (sympy)
or the Heutel files from `heutel.fetch()`; tests requiring unavailable optional files are
skipped. Among the closed forms it checks are the following.

- The RBC with log utility and full depreciation has the solution `P = [[α, 1], [0, ρ]]` and
  `F = [α, 1]`. Its innovation blocks give `h_uu = 1`, `h_ku = ρ`, `h_uuu = 1`, `h_kuu = ρ` and
  `h_kku = ρ(ρ − 1)`. The composed consumption rule gives `g_ku = ρ + α`, `g_uuu = 1`,
  `g_kuu = ρ + α` in the technology direction and `g_kuu = α²` in the capital direction. The
  two-state, one-innovation block gives `g_kku = (ρ + α)(ρ + α − 1)` in the technology direction
  and `g_kku = α²(α² − 1)` in the capital direction.
- The New Keynesian example is determinate under a Taylor rule with `φπ = 1.5` and
  indeterminate with `φπ = 0.8`, where `solve` raises `BlanchardKahnError`.
- On a lognormal AR(1), the second-order coefficient `h_ss` equals `σ²`, so the risk
  correction in the decision rule, `½·h_ss = σ²/2`, is the Jensen term.
- The perfect-foresight transition reproduces the closed-form path `k_{t+1} = αβ k_t^α`.
- On two independent AR(1) processes and one forward-looking equation, the shock loading is
  the identity. The textbook formula read off the QZ treats every equation as holding ex post
  and gives a dense loading here, which is why `klein` solves for the loading on the rows that
  hold ex post
  ([design notes](https://github.com/GustavoHM/dsgelab/blob/main/docs/design-notes.md#the-shock-loading-is-not-read-off-the-qz)).
- Halving the displacement in `accuracy()` divides the first-order residual by about four and
  the second-order residual by about eight.

The suite runs from a clone of the repository.

```
git clone https://github.com/GustavoHM/dsgelab && cd dsgelab
pip install -e ".[symbolic,test,plot]" && pytest
```

The second layer, in [`validation/`](https://github.com/GustavoHM/dsgelab/blob/main/validation/README.md),
runs Dynare's own example and test `.mod` files in Octave with Dynare, thirteen MMB models,
four published-model files from Johannes Pfeifer's DSGE_mod, and one calibration in
MacroModelling.jl. Each result is compared with `dsgelab`'s, for steady states, impulse
responses, second-order decision-rule tensors, parameter Jacobians, moments, optimal policy,
perfect foresight, estimation, OccBin, the extended path, balanced growth, identification,
diagnostics, GMM/SMM
and IRF matching. The same folder checks the port of Heutel (2012) against Heutel's own code.
`bash validation/run_all.sh` runs all of it, and CI runs it weekly.

The third layer is one comparison on an applied model from an unpublished paper by the author.
That model is not public, so the comparison cannot be rerun from the repository. Of the three
layers, the second is the one in which both the models and the reference software come from
other authors.

## Known limitations

This section lists what the package does not do yet. The
[roadmap](https://github.com/GustavoHM/dsgelab/blob/main/ROADMAP.md) says when each item is
planned.

- **The third-order decision rule is composed densely.** `solve_third_order(sparse=True)` stores
  the cubic coefficients by independent symmetric triple, but `decision_rule()` expands them into
  dense arrays, whose size grows with the cube of the number of variables and shocks.
- **Gradient-based sampling is first order.** HMC and NUTS use the analytic gradient of the
  Kalman likelihood, so they run on the first-order solution. At second and third order the
  particle likelihood goes to the posterior mode and to Metropolis–Hastings. The inversion filter
  stops at pruned second order.
- **Identification is local, at one point.** Dynare's repetition of the analysis over draws
  from the prior (`prior_mc`), its identification strength from the information matrix and its
  plots are not reproduced.
- **Global sensitivity covers stability mapping, Monte Carlo filtering and Sobol indices.**
  Dynare's reduced-form mapping (`redform`), its RMSE filtering against data (`rmse`), Morris
  screening and its plots are not reproduced.
- **IRF matching is first order.** GMM and SMM also accept `order=2`, using the pruned
  second-order rule. MATLAB `irf_matching_file` transformations are not executed; pass an
  explicit Python `transform=` instead.
- **Global methods are limited** to Chebyshev collocation with up to three states. There is no
  value or time iteration, endogenous grid, sparse grid, or Tauchen/Rouwenhorst discretisation.
- **No Markov regime switching, heterogeneous agents, BVAR or DSGE-VAR.**
- **Plots are basic.** Impulse responses, paths, decompositions, forecasts and posterior
  densities, with matplotlib as an optional dependency. MCMC diagnostics plots and Dynare's
  identification and sensitivity plots are not reproduced.
- **The extended path's hybrid correction stops at second order.**
- **Trends can be declared only in a `.mod` file.** `from_equations` does not accept them yet.
- **`from_mod` reads the model and ignores the script.** Commands such as `stoch_simul` and
  `estimation` are skipped, and their options are not applied. The dedicated routes
  `ModFile.identification`, `ModFile.method_of_moments` and `ModFile.irf_matching` read their
  respective blocks and accept explicit arguments. In the model block, the constructs the
  reader cannot translate (`@#include`, `@#for`, `varexo_det`, leads of a shock, `expectation`,
  `diff`, `adl`, external functions) are refused with an error that quotes the line containing
  them. Coverage is validated on the Dynare examples and tests listed in `validation/`.

## Related package

[cgelab](https://github.com/GustavoHM/cgelab), by the same author, solves computable general
equilibrium models. Its module `cgelab.bridge` uses dsgelab for the intertemporal part of a
model.

## Citing

If the package contributed to a published result, cite it as any other methodological input,
with the version, because a number that reproduces only under one version cannot be reproduced
without knowing which. [`CITATION.cff`](https://github.com/GustavoHM/dsgelab/blob/main/CITATION.cff)
holds the same metadata in the format GitHub reads.

Moreira, G. H. (2026). dsgelab: A general toolkit for solving DSGE models in Python (Version
0.7.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.23825088

```bibtex
@software{dsgelab_2026,
  author  = {Moreira, Gustavo Hoffmann},
  title   = {{dsgelab}: a general toolkit for solving DSGE models in Python},
  year    = {2026},
  version = {0.7.0},
  license = {MIT},
  doi     = {10.5281/zenodo.23825088},
  url     = {https://github.com/GustavoHM/dsgelab},
  note    = {Python package}
}
```

Each release is archived on Zenodo with a DOI of its own; the DOI above is that of version
0.7.0. The concept DOI [10.5281/zenodo.23456474](https://doi.org/10.5281/zenodo.23456474)
resolves to the latest release.
