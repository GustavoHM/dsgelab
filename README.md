# dsgelab

[![tests](https://github.com/GustavoHM/dsgelab/actions/workflows/tests.yml/badge.svg)](https://github.com/GustavoHM/dsgelab/actions/workflows/tests.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue)](https://github.com/GustavoHM/dsgelab/blob/main/LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23456474.svg)](https://doi.org/10.5281/zenodo.23456474)

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
| Impulse responses, simulation, moments, variance decomposition | `FirstOrderSolution` |
| Second-order perturbation and risk corrections | `solve_second_order` |
| Third-order perturbation, partial (the cubic block and the risk slope) | `solve_third_order` |
| Global solution by Chebyshev collocation, up to three states | `solve_projection` |
| Nonlinear perfect-foresight transitions, announced policy and shock paths | `perfect_foresight` |
| Perfect foresight with expectation errors (news, surprises) | `perfect_foresight_with_expectation_errors` |
| Ramsey optimal policy under commitment (timeless), validated against Dynare | `ramsey_model` |
| Optimal policy under discretion; optimal simple rules, on moments or on welfare | `discretionary_policy`, `osr` |
| Kalman filter (exact diffuse start) and smoother; measurement error; missing data | `state_space`, `kalman_filter`, `kalman_smoother` |
| Bayesian estimation: mode, Laplace, Metropolis-Hastings, SMC, R-hat, ESS, marginal likelihood; maximum likelihood | `EstimationProblem`, `posterior_mode`, `metropolis_hastings`, `smc` |
| Local identification at a point (Iskrev; Komunjer and Ng; Qu and Tkachenko) and collinearity analysis, validated against Dynare's `identification` | `identification`, `ModFile.identification` |
| Theoretical moments of HP- and band-pass-filtered variables, unit roots included | `FirstOrderSolution.moments(hp_filter=, bandpass=)` |
| Shock decomposition (also real-time), forecasts with bands, conditional forecasts | `shock_decomposition`, `forecast`, the method `FirstOrderSolution.conditional_forecast` |
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
analytically. It has 148 tests. Of these, 65 need the `symbolic` extra (sympy) and 15 the
Heutel files from `heutel.fetch()`, and without those files a run gives 137 passed and 11
skipped. Among the closed forms it checks are the following.

- The RBC with log utility and full depreciation has the solution `P = [[α, 1], [0, ρ]]` and
  `F = [α, 1]`.
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
pip install -e ".[symbolic,test]" && pytest
```

The second layer, in [`validation/`](https://github.com/GustavoHM/dsgelab/blob/main/validation/README.md),
runs Dynare's own example and test `.mod` files in Octave with Dynare, thirteen MMB models,
four published-model files from Johannes Pfeifer's DSGE_mod, and one calibration in
MacroModelling.jl. Each result is compared with `dsgelab`'s, for steady states, impulse
responses, moments, optimal policy, perfect foresight, estimation, OccBin, the extended path,
balanced growth, identification and diagnostics. The same folder checks the port of Heutel (2012) against
Heutel's own code. `bash validation/run_all.sh` runs all of it, and CI runs it weekly.

The third layer is one comparison on an applied model from an unpublished paper by the author.
That model is not public, so the comparison cannot be rerun from the repository. Of the three
layers, the second is the one in which both the models and the reference software come from
other authors.

## Known limitations

This section lists what the package does not do yet. The
[roadmap](https://github.com/GustavoHM/dsgelab/blob/main/ROADMAP.md) says when each item is
planned.

- **Third order is partial.** The cubic block and the risk slope (`g_ssk` and `h_ssk`) are
  implemented. Dynare's full decision-rule tensor, with the innovation in every monomial, and
  the stochastic steady state are planned for 0.7. The tensors are dense, so their size grows
  with the cube of the number of variables and shocks, and sparse tensors for large models are
  planned for the same version.
- **Estimation is linear.** The likelihood comes from the Kalman filter on the first-order
  solution. There is no particle or inversion filter, so no estimation at second or third
  order, and no gradient-based samplers (HMC/NUTS).
- **Identification is local, at one point.** Dynare's repetition of the analysis over draws
  from the prior (`prior_mc`), its identification strength from the information matrix and its
  plots are not reproduced.
- **No global sensitivity or method of moments** (Dynare's `+gsa`, `+mom`).
- **Global methods are limited** to Chebyshev collocation with up to three states. There is no
  value or time iteration, endogenous grid, sparse grid, or Tauchen/Rouwenhorst discretisation.
- **No Markov regime switching, heterogeneous agents, BVAR or DSGE-VAR.**
- **No plotting.** Results are arrays and dictionaries, and plotting is left to the user.
- **The extended path's hybrid correction stops at second order.**
- **Trends can be declared only in a `.mod` file.** `from_equations` does not accept them yet.
- **`from_mod` reads the model and ignores the script.** Commands such as `stoch_simul`,
  `estimation` or `identification` are skipped, and their options are not applied
  (`ModFile.identification` takes them as arguments). In the
  model block, the constructs the reader cannot translate (`@#include`, `@#for`, `varexo_det`,
  leads of a shock, `expectation`, `diff`, `adl`, external functions) are refused with an
  error that quotes the line containing them. Coverage is validated on the Dynare examples and
  tests listed in `validation/`.

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
0.5.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.23456475

```bibtex
@software{dsgelab_2026,
  author  = {Moreira, Gustavo Hoffmann},
  title   = {{dsgelab}: a general toolkit for solving DSGE models in Python},
  year    = {2026},
  version = {0.5.0},
  license = {MIT},
  doi     = {10.5281/zenodo.23456475},
  url     = {https://github.com/GustavoHM/dsgelab},
  note    = {Python package}
}
```

Each release is archived on Zenodo with a DOI of its own; the DOI above is that of version
0.5.0. The concept DOI [10.5281/zenodo.23456474](https://doi.org/10.5281/zenodo.23456474)
resolves to the latest release.
