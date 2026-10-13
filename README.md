# dsgelab

[![tests](https://github.com/GustavoHM/dsgelab/actions/workflows/tests.yml/badge.svg)](https://github.com/GustavoHM/dsgelab/actions/workflows/tests.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

A general toolkit for solving DSGE models in Python. You declare the model once —
state names, jump names, shock names, and one function returning the equilibrium
residuals — and everything else follows from that single declaration: the steady
state, the first-order solution, impulse responses, moments, second-order risk
corrections, nonlinear perfect-foresight transitions, and calibration on targets.

The design constraint behind it: the stochastic block and the deterministic
transition must be the *same model*. A common failure mode in applied work is a
perturbation solution built on one set of equations and a transition path solved
by a hand-written fixed point on another, similar but not identical, set. Here
both call the same residual function, so they cannot drift apart.

```
pip install "dsgelab @ git+https://github.com/GustavoHM/dsgelab"
pip install "dsgelab[symbolic] @ git+https://github.com/GustavoHM/dsgelab"
```

For development:

```
git clone https://github.com/GustavoHM/dsgelab && cd dsgelab
pip install -e ".[symbolic,test]" && pytest
```

Requires Python 3.11+, numpy and scipy. Pandas is not needed.

## Quick start

```python
import dsgelab as dk
from dsgelab.examples import rbc

model, params, guess = rbc.build()
sol = dk.solve(model, params, guess=guess)

print(sol.summary())
irf = sol.irf("eps_a", size=0.01, periods=40)      # per cent deviations
mom = sol.moments(cov={"eps_a": 0.007})            # std, corr, autocorr
```

Declaring a model of your own is three lists and a function:

```python
import numpy as np
from dsgelab import Model, solve

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
sol = solve(model, params, guess={"k": 0.2, "a": 1.0, "c": 0.36})
```

Variables are addressed by **name**, in `cur` for period `t` and `nxt` for `t+1`.
When variables are read from a vector by position, swapping one index for another
raises no error, and the program silently solves a different model from the one
intended. Reading each variable by name rules that mistake out.

## What is in the box

| Capability | Entry point |
|---|---|
| Steady state, with positivity and outer anchors | `steady_state` |
| First-order solution (Klein/QZ, Blanchard–Kahn enforced) | `solve` |
| Impulse responses, simulation, moments, variance decomposition | `FirstOrderSolution` |
| Second-order perturbation and risk corrections | `solve_second_order` |
| Nonlinear perfect-foresight transitions, announced policy and shock paths | `perfect_foresight` |
| Perfect foresight with expectation errors (news, surprises) | `perfect_foresight_with_expectation_errors` |
| Ramsey optimal policy under commitment (timeless), validated against Dynare | `ramsey_model` |
| Optimal policy under discretion; optimal simple rules, on moments or on welfare | `discretionary_policy`, `osr` |
| Kalman filter (exact diffuse start) and smoother; measurement error; missing data | `state_space`, `kalman_filter`, `kalman_smoother` |
| Bayesian estimation: mode, Laplace, Metropolis-Hastings, SMC, R-hat, ESS, marginal likelihood; maximum likelihood | `EstimationProblem`, `posterior_mode`, `metropolis_hastings`, `smc` |
| Shock decomposition (also real-time), forecasts with bands, conditional forecasts | `shock_decomposition`, `forecast`, `conditional_forecast` |
| Welfare at second order, conditional and unconditional; consumption equivalent | `welfare`, `consumption_equivalent` |
| Occasionally binding constraints (OccBin), any number of them, surprise shocks | `OccbinModel`, `occbin_simulate` |
| Extended path, deterministic and stochastic (sparse or full tree, hybrid correction) | `extended_path` |
| Models in levels: declared trends with Dynare's balanced-growth check; the growth path | `from_mod`, `balanced_growth` |
| Read a Dynare `.mod` file, macros and leads/lags included | `from_mod` |
| Calibration on targets, with an identification diagnostic; calibration equations inside the model | `calibrate`, `target_jacobian`, `from_equations(calibration_equations=)` |
| Closed-form steady states, checked before use; homotopy for hard steady states | `analytic_steady_state`, `steady_state(homotopy=)` |
| Model diagnostics (Dynare's `model_diagnostics`) and independent checks | `diagnose`, `dsgelab.diagnostics` |
| Text equations with exact analytic derivatives, lags and leads of any order | `dsgelab.symbolic` |
| Several agents' optimisation problems in one equilibrium | `assemble` |

## Timing convention

`eps` is the innovation dated `t+1`, unknown at `t`, with `E_t eps_{t+1} = 0`. An
AR(1) is written

```python
nxt.a - p.rho * cur.a - eps.eps_a
```

and the resulting law of motion is `k_{t+1} = P k_t + Q eps_{t+1}`. Writing it the
other way round — treating the innovation as known at `t` — makes the model
deterministic within the period and silently kills every risk correction at
second order. This is the one convention you must get right; everything else the
package can check for you.

## Canonical form

```
A E_t x_{t+1} = B x_t + C eps_{t+1},   x_t = [k_t ; u_t]
u_t     = F k_t
k_{t+1} = P k_t + Q eps_{t+1}
```

with `k` predetermined and `u` non-predetermined. The Blanchard–Kahn condition —
the number of stable generalised eigenvalues of the pencil `(B, A)` must equal
the number of states — is **enforced**, not reported: an indeterminate or
non-existent equilibrium raises `BlanchardKahnError` rather than returning a
matrix that means nothing.

## Documentation

- [`docs/whitepaper.md`](docs/whitepaper.md) — the mathematics the solvers implement
- [`docs/guide.md`](docs/guide.md) — the user guide, from declaration to second order
- [`docs/api.md`](docs/api.md) — API reference
- [`docs/design-notes.md`](docs/design-notes.md) — numerical decisions and the pitfalls they avoid
- [`docs/from-scripts.md`](docs/from-scripts.md) — porting an existing research script
- [`docs/comparison.md`](docs/comparison.md) — how this package compares with the ones that already exist, and where it does not win
- [`ROADMAP.md`](ROADMAP.md) — what is still missing, and when it is planned
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — how to report a problem and what a contribution needs
- [`site/`](site/) — a GitHub-Pages-ready page with runnable examples

## Heutel environmental-policy example

[`docs/heutel.md`](docs/heutel.md) documents the first-order port of Heutel's
MATLAB replication scripts, including optimal tax and tax/quota rules with
lagged information. The port is checked against original algebra in Octave
and independently solved Dynare impulse responses and moments.

```python
from dsgelab.examples import heutel
heutel.fetch()          # once per machine, see below
sol = heutel.solve("tax")
irf = sol.irf("eps_a", size=0.01, periods=100)
```

The planner source uses a sensitivity calibration that fails the admissible
steady-state check; the common baseline is an explicit choice:
`heutel.solve("planner", heutel.Params())`. See the documentation for differences
between the supplied code and the published paper.

The port's scalar algebra is translated from Heutel's MATLAB files, which carry no
licence, so it is not distributed with the package. `heutel.fetch()` downloads the
public copy from Georgia State University's repository, checks each file against the
SHA-256 of the files the port was validated on, and writes the translation to
`~/.cache/dsgelab/heutel` (or to `DSGELAB_HEUTEL_DIR`).

## Verification

Every result is checked against an independent reference, in two layers. The test suite
(`pytest`, 146 tests, numpy and scipy only) checks against answers known on paper rather
than against previous output:

- the RBC with log utility and full depreciation, whose solution is exactly
  `P = [[α, 1], [0, ρ]]` and `F = [α, 1]`;
- the New Keynesian Taylor principle, which must give determinacy above one and a
  continuum below it;
- the second-order risk correction on a lognormal AR(1), which must come out at
  exactly `σ²`;
- the perfect-foresight transition against the closed-form path
  `k_{t+1} = αβ k_t^α`;
- the shock loading on two independent AR(1) processes, which must be exactly
  the identity — a case where the textbook QZ formula is wrong, see
  [`docs/design-notes.md`](docs/design-notes.md#the-shock-loading-is-not-read-off-the-qz);
- the order of accuracy itself: halving the displacement must divide the
  first-order residual by four and the second-order residual by eight.

```
pip install -e ".[symbolic,test]" && pytest
```

The second layer, in [`validation/`](validation/README.md), runs Dynare's own `.mod` files
and test cases in Octave + Dynare, and a calibration in MacroModelling.jl, and compares
each with `dsgelab`: steady states, impulse responses, moments, optimal policy, perfect
foresight, estimation, OccBin, the extended path, balanced growth and diagnostics.
`bash validation/run_all.sh` runs all of it; CI runs it weekly.

## Known limitations

What the package does not do yet, so that nobody finds out halfway through a project.
The [roadmap](ROADMAP.md) says when each is planned.

- **Third order is partial.** The cubic block and the risk slope `g_σσk` are there; the
  full Dynare decision-rule tensor with the innovation as an argument, the stochastic
  steady state, and sparse tensors for large models are not. The tensors are dense: for
  the 71-variable model that motivated the package they do not fit in memory.
- **Estimation is linear.** The likelihood comes from the Kalman filter on the
  first-order solution. There is no particle or inversion filter, so no estimation at
  second or third order, and no gradient-based samplers (HMC/NUTS).
- **No identification analysis, global sensitivity, or method of moments** (Dynare's
  `+identification`, `+gsa`, `+mom`), and no HP- or band-pass-filtered theoretical moments.
- **Global methods are limited** to Chebyshev collocation with up to three states. No
  value or time iteration, endogenous grid, sparse grids, or Tauchen/Rouwenhorst.
- **No Markov regime switching, heterogeneous agents, BVAR or DSGE-VAR.**
- **No plotting.** Results are arrays and dictionaries; plotting is left to the user.
- **The extended path's hybrid correction stops at second order.**
- **Trends are declared only through a `.mod` file**, not yet in `from_equations`.
- **`from_mod` reads the model, not the script.** Commands such as `stoch_simul`,
  `estimation` or `identification` are skipped, not run, and their options are not
  applied. In the model itself, what the reader cannot translate is refused with the
  line that caused it: `@#include`, `@#for`, `varexo_det`, leads of a shock,
  `expectation`, `diff`, `adl`, external functions. Coverage is validated on the Dynare
  examples and tests listed in `validation/`.
- **Some of the validation material is in Portuguese.** The documentation, the code and
  the site are in English, with a Portuguese version of the site under `site/pt-br/`.
  The comparison scripts in `validation/` still print their reports in Portuguese, and
  the roadmap is written in Portuguese.

## Citing

If a published result went through this package, cite it — software is a methodological
input like any other, and the VERSION matters, since a number that only reproduces under
one version is not reproducible without it. [`CITATION.cff`](CITATION.cff) carries the
metadata in the format GitHub reads; fill in the DOI and ORCID before citing.

A repository URL is not an archivable reference. Connect the repository to Zenodo, cut a
release, and cite the DOI that comes out of it.
