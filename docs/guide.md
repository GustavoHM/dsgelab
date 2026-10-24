# dsgelab user guide

Unless a section says otherwise, the blocks continue from

```python
import dsgelab as dsg
from dsgelab.examples import rbc

model, params, guess = rbc.build()
```

and from the blocks before them, so `ss` (§2) and `sol` (§3) are available in
the later sections. `rbc` is the stochastic growth model with log
utility and full depreciation, with states `k` and `a`, the jump `c` and the shock
`eps_a`. Blocks that begin with the comment `# sketch` show the shape of code for
a model of your own and use names that `rbc` does not define.

1. [Declaring a model](#1-declaring-a-model)
2. [The steady state](#2-the-steady-state)
3. [First-order solution](#3-first-order-solution)
4. [Impulse responses, simulation, moments](#4-impulse-responses-simulation-moments)
5. [Second order and risk](#5-second-order-and-risk)
6. [Perfect-foresight transitions](#6-perfect-foresight-transitions)
7. [Calibration on targets](#7-calibration-on-targets)
8. [Diagnostics](#8-diagnostics)
9. [Text equations and exact derivatives](#9-text-equations-and-exact-derivatives)
10. [Multi-group and multi-sector models](#10-multi-group-and-multi-sector-models)

---

## 1. Declaring a model

A model is three name lists and one function. This section declares the model
that `rbc.build()` returns.

```python
import numpy as np
from dsgelab import Model

def equations(nxt, cur, eps, p):
    y  = cur.a * cur.k ** p.alpha
    y1 = nxt.a * nxt.k ** p.alpha
    return [
        nxt.k - ((1 - p.delta) * cur.k + y - cur.c),                  # resource constraint
        cur.c ** -p.gamma - p.beta * nxt.c ** -p.gamma                # Euler equation
            * (p.alpha * y1 / nxt.k + 1 - p.delta),
        np.log(nxt.a) - p.rho * np.log(cur.a) - eps.eps_a,            # technology
    ]

model = Model(
    states=["k", "a"],       # predetermined; exogenous processes belong here
    jumps=["c"],             # non-predetermined (forward-looking)
    shocks=["eps_a"],        # innovations, dated t+1
    equations=equations,     # f(nxt, cur, eps, params) -> one residual per variable
    name="rbc",
)
```

The number of residuals must equal `len(states) + len(jumps)`. `Model.residuals`
checks it at every evaluation, so a mismatch raises a `ValueError` at the first
one (usually inside `steady_state`), with the counts of residuals, states and
jumps.

### The equation function

`nxt` and `cur` are `VarView` objects over the variables at `t+1` and `t`. A
variable is read as `cur["k"]` or `cur.k`, and `in` tests membership.
`.vec("K")` gathers an indexed family `K0, K1, …` into an array (see §10). `eps`
is a `VarView` over the shocks.

`params` is whatever you pass to `solve`, whether a dataclass, a mapping or any
object with attributes. With a dataclass, `dataclasses.replace` produces each
point of a policy sweep without further code, and `set_params` uses it
internally.

### The optional fifth argument

If the equations need side information, they accept a fifth positional argument.

```python
def equations_with_ctx(nxt, cur, eps, params, ctx):
    ...
```

The signature is inspected once at construction, so both forms are accepted
without a flag. `ctx` is how a steady-state anchor, a per-period policy, or a
block of calibrated constants reaches the equations. See §2 and §6.

### Timing

`eps` is dated `t+1`, so an AR(1) is the residual
`nxt.a - p.rho * cur.a - eps.eps_a`. The package cannot detect a reversed timing
in a user's model. The model still solves, the impulse responses look plausible,
and every second-order risk correction is zero, with no error. The test suite
checks the convention on the log AR(1), where the risk coefficient `h_ss` must
equal `σ²` (§5).

---

## 2. The steady state

```python
ss = dsg.steady_state(model, guess, params)
ss["k"], ss.residual, ss.values
```

`guess` is a mapping of names to values, or an array in the model's variable
order. Convergence is judged on the residual, and the solver's success flag is
ignored, because `hybr` reports failure whenever its finite-difference Jacobian
stops improving, including when the starting point already is the solution. A
solve whose residual stays above `tol` raises `RuntimeError`.

### Positivity

```python
ss = dsg.steady_state(model, guess, params, positive=["k", "c", "a"])
```

Those coordinates are solved in logs, which keeps them positive at every
iterate. A clip would flatten the residual and stall the solver at zero. Use it
for stocks, prices and quantities, which a Newton step can otherwise take
negative, so that the next residual is a NaN or a complex number.

### Anchors: the outer loop

Some residuals need side information that is itself a function of the steady
state. The recurring cases are a portfolio adjustment cost centred on
steady-state wealth, and an exogenous level pinned to equilibrium demand.
Substituting the solution directly into that term makes it vanish, and the margin
the term was there to pin down becomes indeterminate. The term is therefore
carried in `ctx` and updated by an outer loop.

```python
# sketch: a model whose equations take ctx, with a demand variable D
def ctx_update(values, ctx):
    return {**ctx, "k_anchor": values["k"], "QR": renew_share * values["D"]}

ss = dsg.steady_state(my_model, my_guess, my_params, ctx=ctx0, ctx_update=ctx_update)
```

The loop stops when the context stops moving. `ss.ctx` carries the converged
context, and `solve` inherits it.

---

## 3. First-order solution

```python
sol = dsg.solve(model, params, guess=guess)         # solves the steady state too
sol = dsg.solve(model, params, steady=ss)           # or reuse one
print(sol.summary())                                # a one-screen report
```

`solve` differentiates the residuals at the steady state, assembles
`A E_t x_{t+1} = B x_t + C eps`, and calls the Klein/QZ solver, which raises
`BlanchardKahnError` when the Blanchard–Kahn condition fails. The result is a
`FirstOrderSolution`, whose main attributes are listed below.

| Attribute | Meaning |
|---|---|
| `sol.P` | state transition, `k_{t+1} = P k_t + Q eps_{t+1}` |
| `sol.F` | policy, `u_t = F k_t` |
| `sol.Q` | shock loading on the states |
| `sol.eigenvalues` | ordered generalised eigenvalue moduli |
| `sol.steady` | the `SteadyState` behind it |
| `sol.system` | the `A`, `B`, `C` matrices and the coordinate map |

### Coordinates

Everything is in the rescaled deviation

```
x = xbar + s * yhat,     s_i = |xbar_i| if nonzero, else 1
```

so `yhat` is a proportional deviation for variables with a nonzero steady state
and an absolute one for the rest. The change of variables is linear, so it adds
no curvature of its own to the second-order terms.

Whether a variable is proportional is decided from its steady-state level. A
scaling chosen by variable name would multiply a mean-zero variable by
`(1 + yhat)` around zero, and the shock would disappear without an error. The
`nk` example exists to test that branch.

Use `sol.to_levels(paths)` or `sol.coords.to_level(y)` to get back to levels.

---

## 4. Impulse responses, simulation, moments

```python
irf = sol.irf("eps_a", size=0.01, periods=40)          # per cent deviations
irf = sol.irf("eps_a", size=0.01, periods=40, levels=True)
irf = sol.irf("a", size=0.01, periods=40, kind="state")
```

Each call returns a dict from variable name to an array of length `periods`,
whose entry 0 is the impact date. `size` is the value of the innovation in the
units it has in the equations. Here `eps_a` is the innovation to `log a`, so
`size=0.01` raises `a` by 1 per cent on impact, and a one-standard-deviation
shock is `size=σ`. The deviations are in the coordinates of §3 and are
multiplied by 100 unless `percent=False`. `levels=True` returns levels.

`kind="shock"` applies an innovation, so the impact period is `k = Q e · size`.
`kind="state"` displaces the named state directly. That is how a change in the
level of an exogenous state (a hydrological shift, a one-off revaluation) is
represented without passing through its innovation.

```python
sim  = sol.simulate(periods=10_000, cov={"eps_a": 0.007}, seed=42, burn=200)
mom  = sol.moments(cov={"eps_a": 0.007}, lags=4)
dec  = sol.variance_decomposition(cov={"eps_a": 0.007})          # unconditional
dec4 = sol.variance_decomposition(cov={"eps_a": 0.007}, horizon=4)  # FEVD at h=4
```

`simulate` gives a dict of arrays. `moments` returns a dict with the keys
`names`, `std`, `corr`, `cov`, `autocorr` (by lag, then by name) and
`std_by_name`, and `variance_decomposition` one with `names`, `shocks`,
`variance` and `share`. The covariance may be a mapping of standard deviations, a
vector of standard deviations, or a full matrix. A shock switched off by setting
its variance to zero is handled, since the factorisation falls back to an
eigendecomposition where Cholesky refuses a singular matrix.

Moments come from the discrete Lyapunov equation, so they are exact for the
linear solution and carry no sampling error.

The moments of filtered series, the ones business-cycle tables report, take one
more argument. `hp_filter=1600` gives those of the Hodrick–Prescott cycle and
`bandpass=(6, 32)` those of cycles between 6 and 32 quarters, as Dynare's
`stoch_simul` computes them, from the spectral density. Both filters remove the
zero frequency, so a variable with a unit root, which has no unconditional
variance, has filtered moments.

```python
hp = sol.moments(cov={"eps_a": 0.007}, lags=4, hp_filter=1600)
bp = sol.moments(cov={"eps_a": 0.007}, bandpass=(6, 32))
```
`test_simulation_reproduces_the_analytic_moments` checks a long simulation against
them.

### Accuracy

```python
sol.accuracy()
# {'sizes': [...], 'residuals': [...], 'ratios': [...], 'implied_order': [2.00, 2.00], 'exact': False}
```

This feeds the decision rules back into the nonlinear residuals at shrinking
displacements. A correct first-order solution leaves a residual of order
`‖y‖²`, so halving the displacement divides the residual by roughly four. A
wrongly assembled system (a transposed block, a misplaced timing) leaves a
residual of order `‖y‖`, and the ratio comes out near two. The test reads the
ratio and disregards the size of any single residual.

For an exactly linear model the residuals sit at round-off, and the report says
`exact: True` instead of inferring an order from machine noise.

---

## 5. Second order and risk

```python
so = dsg.solve_second_order(sol, cov={"eps_a": 0.007})
so.g_kk, so.h_kk        # quadratic terms
so.g_ss, so.h_ss        # risk coefficients, which first order sets to zero
```

The decision rules are

```
u_t     = F k_t + ½ g_kk[k_t, k_t] + ½ g_ss
k_{t+1} = P k_t + ½ h_kk[k_t, k_t] + ½ h_ss + Q eps_{t+1}
```

A first-order simulation holds the mean of the states at the deterministic
steady state by construction, so every mean effect it reports comes from the
curvature of the static mapping alone, without the displacement of the decision
rules. When that displacement is large, as it can be in models with a convex
adjustment cost and an occasionally tight margin, a first-order simulation can
get the sign of a mean effect wrong as well as its size. Welfare rankings and
certainty-equivalent comparisons depend on that sign.

### The ergodic mean

```python
so.mean_correction()      # deviations from the deterministic steady state
so.ergodic_mean()         # levels
```

Both are computed in closed form from the pruned system. The displacement is of
order `σ²` while the simulated series has a standard deviation of order `σ`, so
separating them by averaging a simulation takes millions of periods in a
persistent model, and a shorter sample reports Monte Carlo noise with a plausible
sign.

### Simulation and pruning

```python
sim = so.simulate(periods=20_000, seed=3)              # pruned by default
```

Pruning (Kim, Kim, Schaumburg and Sims) keeps the quadratic term a function of
the first-order path only. Without it, the quadratic term is evaluated on a state
that already contains the quadratic terms of earlier periods, and a long sample
explodes. Iterating a quadratic map produces this whatever the numerical
settings, so an unpruned second-order sample cannot support a moment over a long
horizon.

### Checking it

```python
so.accuracy()['implied_order']    # ~3.0
```

Halving the displacement must divide the residual by about eight. A ratio near
four means the quadratic block is wrong and the solution is no better than first
order.

The test suite contains two closed-form checks that can be reproduced on a model
with an analogue. For `log a' = ρ log a + ε` the risk coefficient must equal
`h_ss = σ²`, and the correction in the rule, `½ h_ss = σ²/2`, is the Jensen
term. Where the true policy is exact in the model's own coordinates, the
corresponding risk term must be zero.

### Cost

The Sylvester equation is solved directly when `n_vars * n_states² ≤ direct_limit`
(6000 by default) and iteratively above it. The Hessian costs
`O((2n + n_shocks)²)` residual evaluations by finite differences. The symbolic
front end (§9) supplies the exact Hessian, evaluated from compiled expressions in
place of those residual evaluations.

---

## 6. Perfect-foresight transitions

The examples use a permanent rise in `beta` from 0.99 to 0.995, which raises the
capital stock.

```python
from dataclasses import replace

p_old, p_new = params, replace(params, beta=0.995)
ss0 = dsg.steady_state(model, guess, p_old)
ss1 = dsg.steady_state(model, guess, p_new)
T = 120

path = dsg.perfect_foresight(model, start=ss0, terminal=ss1, params=p_new, T=T)
path["k"]              # (T+1,) levels
path.levels            # (T+1, n_vars)
```

`perfect_foresight` solves the path by a stacked-time Newton iteration on the
same residual function that the perturbation linearises. The Jacobian is
block-bidiagonal and assembled sparse.

**Boundary conditions.** The `n_k` predetermined states at the start are
inherited from `start`, and the `n_u` forward jumps at the end sit at `terminal`.
Fixing the whole terminal vector instead overdetermines the system by `n_k`
equations, and the only symptom is a Newton iteration that stops converging.

### Announced paths

A *surprise* is a change in `terminal` with constant `params`. An *announced*
path is a sequence.

```python
schedule = [p_old] * 10 + [p_new] * (T + 1 - 10)
path = dsg.perfect_foresight(model, ss0, ss1, params=schedule, T=T)
```

`params` and `ctx` each accept a constant, a sequence of length `T+1`, or a
callable `f(t)`. Element `t` enters the equations that link date `t` to `t+1`, so
element `T` enters no equation and serves `path.derived` at the last date. A
statutory calendar known to agents from the outset is a sequence. The two paths
differ from the first date, because under a surprise the new parameters are in
force at date 0, while under an announcement the agents at date 0 respond to
parameters they expect to change later.

If the equations need *next period's* parameters (a policy that agents know will
change between `t` and `t+1`), carry them in `ctx`, since `ctx` is also a path.

### Starting elsewhere

```python
path = dsg.perfect_foresight(model, ss, ss, params=params, T=150,
                             initial_states=[0.5 * ss["k"], 1.0])
```

`initial_states` overrides the inherited states, for a transition that does not
begin at a steady state. It is an array of levels in the order of `model.states`,
here `k` then `a`. A mapping is not accepted.

### Along the path

```python
series = path.derived(lambda vals, p, ctx: {"y": vals["a"] * vals["k"] ** p.alpha})
```

`derived` evaluates a per-period function of the variables over the whole path.
Objects defined by identities (prices, shares, flows) are recovered this way
without recomputing them by hand at every date.

---

## 7. Calibration on targets

The example frees `beta` and `delta` and targets the capital-output ratio and the
investment share.

```python
def observables(ss, p):
    y = ss["a"] * ss["k"] ** p.alpha
    return {"ky": ss["k"] / y, "iy": p.delta * ss["k"] / y}   # capital-output, investment share

res = dsg.calibrate(
    model, params,
    free={"beta": 0.95, "delta": 0.5},
    targets={"ky": 10.0, "iy": 0.25},
    guess=guess,
    observables=observables,
    bounds={"beta": (0.5, 0.999), "delta": (0.001, 1.0)},
)
print(res.report())
res.params, res.steady, res.values
```

Targets are steady-state variables unless `observables` supplies them. Since
`iy = delta · ky`, the two targets give `delta = 0.025`, and the Euler equation
then gives `beta = 1/(alpha/ky + 1 − delta) = 0.98912`. The parameters and the steady state are solved as one system. Done by hand, the same
calibration becomes nested bisections, one per parameter, each of which treats
its target as set by its own parameter, although the targets are determined
jointly.

With `warm_start=True`, the default, each steady-state solve starts from the
previous one. The accepted solution is still judged on its own residual.

### Identification

`calibrate` does not check identification. `target_jacobian` does, from the
derivatives of the targets with respect to the free parameters.

```python
ident = dsg.target_jacobian(model, res.params, free=["beta", "delta"],
                            targets=["ky", "iy"], guess=guess,
                            observables=observables)
ident["identified"]        # True: two targets pin the two parameters

one = dsg.target_jacobian(model, params, free=["beta", "alpha"],
                          targets=["k"], guess=guess)
one["identified"]          # False: two parameters, one target
one["rank"], one["n_free"] # 1, 2
one["null_directions"]     # the direction no target sees
one["elasticity"]          # d log target / d log parameter
```

A calibration whose elasticity matrix is rank-deficient is not identified. Some
direction in parameter space leaves every target where it was, and the numbers
that come out of it depend on where the solver stops. Run this whenever a
calibration fails to converge, because tightening the solver tolerance does not
identify an unidentified system.

`target_jacobian` asks whether the steady-state targets pin down the parameters.
Before an estimation the question is a different one: whether the distribution of
the observed series does. `identification` answers it at one point, with the three
criteria of Dynare's `identification` command. These are the derivatives of the
moments (Iskrev), of the spectrum (Qu and Tkachenko) and of a minimal state space
(Komunjer and Ng).

```python
ident = dsg.identification(model, params, free=["rho", "stderr eps_a"],
                           observables=["c"], cov={"eps_a": 0.01}, guess=guess)
ident.identified              # every criterion has full rank
ident["moments"].dependent    # sets of parameters that move the moments together
print(ident.summary())
```

`ModFile.identification()` does the same for the `estimated_params` of a `.mod`,
observing its `varobs`, at the prior mean.

---

## 8. Diagnostics

The Blanchard–Kahn condition in `solve` and the residual test in `steady_state`
are the only checks that run by themselves. The user calls each of the functions
below.

```python
dsg.blanchard_kahn_report(sol)
```

Reports the margin between the largest stable and the smallest unstable root as
well as the pass. A model that satisfies Blanchard–Kahn with a root at 0.9999
changes classification under a parameter change small enough to be missed in a
sweep.

```python
def consumption_share(nxt, now, p, ctx):     # c = (1 - alpha*beta) y in this model
    return now["c"] - (1 - p.alpha * p.beta) * now["a"] * now["k"] ** p.alpha

path = dsg.perfect_foresight(model, ss, ss, params=params, T=150,
                             initial_states=[0.5 * ss["k"], 1.0])
dsg.check_identity(model, consumption_share, path.levels, params)
```

`check_identity` evaluates an identity the model does not impose. Its `points`
are `path.levels`, an array whose consecutive rows it pairs (a `TransitionPath`
itself is not accepted), or a sequence of `(x_next, x_now)` pairs. In `rbc` the
consumption share `c = (1 − αβ) y` follows from the equations without being one
of them. In a closed model the resource constraint follows from the budget
constraints, the firms' first-order conditions and the definitions of profit. It
is therefore not one of the equations, and its residual is an independent check
on the accounting. Imposing it as an equation would set that residual to zero by
construction and remove the check.

```python
dsg.check_slack(sol, {"investment_positive": lambda v: v["a"] * v["k"] ** params.alpha - v["c"]},
                size=0.05)
```

A `max(x, 0)` or a floor is not differentiable and cannot be perturbed, so
applied models replace it with its smooth interior branch. The substitution is
valid only while the constraint is slack, and a constraint that is slack at the
steady state can bind along a large impulse response. `check_slack` evaluates
each constraint, positive when slack, on the impulse responses in levels.

```python
from dataclasses import replace

dsg.steady_state_invariance(
    lambda rho: dsg.steady_state(model, guess, replace(params, rho=rho)),
    [0.0, 0.5, 0.9, 0.95], ["k", "c"])          # persistence does not move the steady state

model_s, params_s, guess_s = rbc.build_symbolic()                 # requires sympy
dsg.compare_irfs(sol, dsg.solve(model_s, params_s, guess=guess_s), "eps_a")
```

The first asserts, over a parameter grid, an invariance that is a theorem of the
model, such as output independent of a redistributive policy or a great ratio
pinned by the Euler equation. A failure shows that a change broke the block that
delivers it. The second compares two solutions' impulse responses, for checking a
port or a refactor. Comparing steady states is not enough, since two different
models can share a steady state and differ in their dynamics.

---

## 9. Text equations and exact derivatives

Requires sympy (`pip install "dsgelab[symbolic]"`).

```python
model = dsg.from_equations(
    [
        "k(+1) = (1-delta)*k + a*k**alpha - c",
        "c**(-gamma) = beta * c(+1)**(-gamma) * (alpha*a(+1)*k(+1)**(alpha-1) + 1 - delta)",
        "log(a(+1)) = rho*log(a) + eps_a",
    ],
    states=["k", "a"], jumps=["c"], shocks=["eps_a"],
    parameters=["alpha", "beta", "rho", "delta", "gamma"],
)
sol = dsg.solve(model, params, guess=guess)    # params and guess from rbc.build()
```

`x` is the variable at `t` and `x(+1)` at `t+1`. A shock name stands for the
innovation at `t+1`. Equations may use `=` or be expressions that must vanish.

The symbolic front end gives exact derivatives. The residual's Jacobian and
Hessian are differentiated symbolically and used in place of finite differences
in the first-order system, the second-order Hessian, the inner Hessian of the
third order, and the Newton steps of `perfect_foresight` and `extended_path`.
The steady-state root finder, the outer difference of the third-order tensor, and
`calibrate` and `target_jacobian` (derivatives with respect to the parameters)
still use finite differences. Exact derivatives matter at second order because a
second difference divides by the square of the step, so a finite-difference
Hessian carries roughly half the significant digits of the residual it came from,
and the risk corrections are among the small terms that lose those digits.

The parser applies two rules to names and dates.

- An undeclared name raises an error instead of becoming a free symbol, so a
  typo cannot turn into a new variable. For the same reason `beta`, `gamma`, `S`
  and other sympy objects are safe as parameter names, since the parser is given
  no global namespace to bind them to.
- A lag `x(-k)` is carried by the states `LAG1_x`, …, `LAGk_x`, and a lead
  `x(+k)` with `k >= 2` by the jumps `LEAD1_x`, …, `LEAD{k-1}_x`, where `LEADj_x`
  holds `x` at `t+j` (so `y(+2)` creates `LEAD1_y`). Each auxiliary has its own
  equation and is listed in `model.auxiliary`. What is predetermined is still
  declared in one place, since the lag of a jump becomes a new state and the jump
  keeps its classification. A lagged innovation is refused with an explanation.

`derivatives="jacobian"` skips the Hessian, which costs
`n_eq * (2n + n_shocks)²` expressions to build, and `"none"` skips both.

---

## 10. Multi-group and multi-sector models

Declare the family with an index suffix and gather it in the equations.

```python
# sketch: N groups; budget, euler, q_def, q_euler, revenue_neutral and shocks
# stand for the model's own residual blocks
states = [f"K{j}" for j in range(N)] + [f"KG{j}" for j in range(N)] + ["a", "z"]
jumps  = [f"C{j}" for j in range(N)] + [f"q{j}" for j in range(N)] + ["piB"]

def equations(nxt, cur, eps, p, ctx):
    K, KG = cur.vec("K", N), cur.vec("KG", N)
    C, q  = cur.vec("C", N), cur.vec("q", N)
    Kp    = nxt.vec("K", N)
    ...
    return np.concatenate([budget, euler, q_def, q_euler, [revenue_neutral], shocks])
```

`vec` reads the family in index order, so the equations are written on vectors
and the residual is assembled by `concatenate` in the same order as the variable
lists. Keeping the two orders together in one place prevents the layout from
drifting.

Constants that do not vary with the solution (group masses, calibrated shares, a
policy schedule) belong in `ctx` when they are arrays. The parameter dataclass
then holds only scalars, which `dataclasses.replace` can sweep.
