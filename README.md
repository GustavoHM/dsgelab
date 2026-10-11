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

Requires Python 3.10+, numpy and scipy. Pandas is not needed.

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
An index slip in a hand-packed residual vector is silent and is the single most
expensive class of error in this kind of code; naming removes the class.

## What is in the box

| Capability | Entry point |
|---|---|
| Steady state, with positivity and outer anchors | `steady_state` |
| First-order solution (Klein/QZ, Blanchard–Kahn enforced) | `solve` |
| Impulse responses, simulation, moments, variance decomposition | `FirstOrderSolution` |
| Second-order perturbation and risk corrections | `solve_second_order` |
| Nonlinear perfect-foresight transitions, announced policy paths | `perfect_foresight` |
| Ramsey optimal policy under commitment (timeless), validated against Dynare | `ramsey_model` |
| Optimal policy under discretion; optimal simple rules | `discretionary_policy`, `osr` |
| Welfare at second order, conditional and unconditional; consumption equivalent | `welfare`, `consumption_equivalent` |
| Read a Dynare `.mod` file, macros and leads/lags included | `from_mod` |
| Calibration on targets, with an identification diagnostic | `calibrate`, `target_jacobian` |
| Independent checks: identities, slackness, BK margin, cross-solution | `dsgelab.diagnostics` |
| Text equations with exact analytic derivatives | `dsgelab.symbolic` |

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
- [`ROADMAP.md`](ROADMAP.md) — what is still missing, and what is left out on purpose
- [`site/`](site/) — a GitHub-Pages-ready page with runnable examples

## Heutel environmental-policy example

[`docs/heutel.md`](docs/heutel.md) documents the first-order port of Heutel's
MATLAB replication scripts, including optimal tax and tax/quota rules with
lagged information. The port is checked against original algebra in Octave
and independently solved Dynare impulse responses and moments.

```python
from dsgelab.examples import heutel
sol = heutel.solve("tax")
irf = sol.irf("eps_a", size=0.01, periods=100)
```

The planner source uses a sensitivity calibration that fails the admissible
steady-state check; the common baseline is an explicit choice:
`heutel.solve("planner", heutel.Params())`. See the documentation for differences
between the supplied code and the published paper.

[`docs/ai-memory.md`](docs/ai-memory.md) explains shared project context across
Codex, Claude Code, Grok, Kimi, Cursor, Gemini and Antigravity.

## Verification

`pytest` runs 36 tests, and they check against answers known on paper rather than
against previous output:

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

## Citing

If a published result went through this package, cite it — software is a methodological
input like any other, and the VERSION matters, since a number that only reproduces under
one version is not reproducible without it. [`CITATION.cff`](CITATION.cff) carries the
metadata in the format GitHub reads; fill in the DOI and ORCID before citing.

A repository URL is not an archivable reference. Connect the repository to Zenodo, cut a
release, and cite the DOI that comes out of it.
