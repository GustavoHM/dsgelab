# dsgekit user guide

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

A model is three name lists and one function.

```python
from dsgekit import Model

model = Model(
    states=["k", "a"],       # predetermined; exogenous processes belong here
    jumps=["c"],             # non-predetermined (forward-looking)
    shocks=["eps_a"],        # innovations, dated t+1
    equations=equations,     # f(nxt, cur, eps, params) -> one residual per variable
    name="rbc",
)
```

The number of residuals must equal `len(states) + len(jumps)`; the constructor
checks it and says which is which when it does not.

### The equation function

```python
def equations(nxt, cur, eps, params):
    ...
    return [resid_1, resid_2, ...]
```

`nxt` and `cur` are `VarView` objects. Both `cur["k"]` and `cur.k` work; `in`
tests membership; `.vec("K")` gathers an indexed family `K0, K1, …` into an
array (see §10). `eps` is a `VarView` over the shocks.

`params` is whatever you pass to `solve` — a dataclass, a mapping, or any object
with attributes. Dataclasses are the natural choice: `dataclasses.replace` gives
you a policy sweep for free, and `set_params` uses it internally.

### The optional fifth argument

If the equations need side information, accept a fifth positional argument:

```python
def equations(nxt, cur, eps, params, ctx):
    ...
```

The signature is inspected once at construction, so both forms are accepted
without a flag. `ctx` is how a steady-state anchor, a per-period policy, or a
block of calibrated constants reaches the equations. See §2 and §6.

### Timing

`eps` is dated `t+1`. Write an AR(1) as

```python
nxt.a - p.rho * cur.a - eps.eps_a
```

Getting this backwards is the one mistake the package cannot detect for you: the
model still solves, the impulse responses still look plausible, and every
second-order risk correction is silently zero. §5 explains why.

---

## 2. The steady state

```python
ss = dk.steady_state(model, guess, params)
ss["k"], ss.residual, ss.values
```

`guess` is a mapping of names to values, or an array in the model's variable
order. Convergence is judged on the **residual**, never on the solver's success
flag: `hybr` reports failure whenever its finite-difference Jacobian stops
improving, and that includes the case where the starting point already *is* the
solution.

### Positivity

```python
ss = dk.steady_state(model, guess, params, positive=["k", "c", "a"])
```

Those coordinates are solved in logs. This imposes positivity exactly, rather
than by a clip that would flatten the residual and stall the solver at zero. Use
it for stocks, prices and quantities — a Newton step walks them negative and the
next residual is a NaN or a complex number.

### Anchors: the outer loop

Some residuals need side information that is itself a function of the steady
state. The recurring cases are a portfolio adjustment cost centred on
steady-state wealth, and an exogenous level pinned to equilibrium demand.
Substituting the solution directly into that term makes it vanish and leaves the
corresponding margin indeterminate — which is exactly what the term existed to
prevent. So it is carried in `ctx` and updated by an outer loop:

```python
def ctx_update(values, ctx):
    return {**ctx, "k_anchor": values["k"], "QR": renew_share * values["D"]}

ss = dk.steady_state(model, guess, params, ctx=ctx0, ctx_update=ctx_update)
```

The loop stops when the context stops moving. `ss.ctx` carries the converged
context, and `solve` inherits it automatically.

---

## 3. First-order solution

```python
sol = dk.solve(model, params, guess=guess)         # solves the steady state too
sol = dk.solve(model, params, steady=ss)           # or reuse one
```

`solve` differentiates the residuals at the steady state, assembles
`A E_t x_{t+1} = B x_t + C eps`, and calls the Klein/QZ solver with
Blanchard–Kahn enforced. What comes back:

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

so `yhat` is a **proportional** deviation for variables with a nonzero steady
state and an **absolute** one for the rest. The change of variables is linear,
which is what makes it safe at second order.

Testing which variables are proportional by their steady-state *level* rather
than by name matters. A scaling that went by name multiplies a mean-zero
variable by `(1 + yhat)` around zero and the shock disappears in silence. The
`nk` example exists to test that branch.

Use `sol.to_levels(paths)` or `sol.coords.to_level(y)` to get back to levels.

---

## 4. Impulse responses, simulation, moments

```python
irf = sol.irf("eps_a", size=0.01, periods=40)          # per cent deviations
irf = sol.irf("eps_a", size=0.01, periods=40, levels=True)
irf = sol.irf("a", size=0.01, periods=40, kind="state")
```

`kind="shock"` applies an innovation, so the impact period is `k = Q e size`.
`kind="state"` displaces the named state directly, which is what a counterfactual
on an exogenous **level** — a permanent hydrological shift, a one-off
revaluation — actually means.

```python
sim  = sol.simulate(periods=10_000, cov={"eps_a": 0.007}, seed=42, burn=200)
mom  = sol.moments(cov={"eps_a": 0.007}, lags=4)
dec  = sol.variance_decomposition(cov={"eps_a": 0.007})          # unconditional
dec4 = sol.variance_decomposition(cov={"eps_a": 0.007}, horizon=4)  # FEVD at h=4
```

The covariance may be a mapping of standard deviations, a vector of standard
deviations, or a full matrix. A shock switched off by setting its variance to
zero is handled (the factorisation falls back to an eigendecomposition, since
Cholesky refuses a singular matrix).

Moments come from the discrete Lyapunov equation, not from a simulation, so they
are exact for the linear solution. `test_simulation_reproduces_the_analytic_moments`
checks the two against each other.

### Accuracy

```python
sol.accuracy()
# {'sizes': [...], 'residuals': [...], 'ratios': [...], 'implied_order': [2.00, 2.00]}
```

This feeds the decision rules back into the **nonlinear** residuals at shrinking
displacements. A correct first-order solution leaves a residual of order
`‖y‖²`, so halving the displacement divides the residual by roughly four. A
wrongly assembled system — a transposed block, a misplaced timing — leaves a
residual of order `‖y‖` and the ratio comes out near two. The *ratio* is the
discriminating test; the size of any single residual is not.

For an exactly linear model the residuals sit at round-off and the report says
`exact: True` rather than inventing an order from machine noise.

---

## 5. Second order and risk

```python
so = dk.solve_second_order(sol, cov={"eps_a": 0.007})
so.g_kk, so.h_kk        # quadratic terms
so.g_ss, so.h_ss        # risk corrections -- what first order sets to zero
```

The decision rules are

```
u_t     = F k_t + ½ g_kk[k_t, k_t] + ½ g_ss
k_{t+1} = P k_t + ½ h_kk[k_t, k_t] + ½ h_ss + Q eps_{t+1}
```

**Why it matters.** A first-order simulation holds the mean of the states at the
deterministic steady state *by construction*. Every mean effect it reports is
therefore the curvature of the static mapping alone, with the displacement of the
decision rules missing. When that displacement is large — and in models with a
convex adjustment cost and an occasionally-tight margin it routinely is — a
first-order simulation can get the **sign** of a mean effect wrong, not merely
its size. Welfare rankings and certainty-equivalent comparisons are exactly the
objects that depend on it.

### The ergodic mean

```python
so.mean_correction()      # deviations from the deterministic steady state
so.ergodic_mean()         # levels
```

Computed in closed form from the pruned system, not by averaging a simulation.
The distinction is practical: the displacement is of order `σ²` while the
simulated series has standard deviation of order `σ`, so a sample long enough to
separate them by averaging runs to millions of periods in a persistent model, and
a shorter one reports Monte Carlo noise with a plausible sign.

### Simulation and pruning

```python
sim = so.simulate(periods=20_000, seed=3)              # pruned by default
```

Pruning (Kim–Kim–Schaumburg–Sims) keeps the quadratic term a function of the
first-order path only. Without it the quadratic feeds on itself and a long sample
explodes — not a numerical artefact to be tuned away but a property of iterating
a quadratic map, and the reason an unpruned second-order simulation cannot be
trusted at any horizon that makes moments meaningful.

### Checking it

```python
so.accuracy()['implied_order']    # ~3.0
```

Halving the displacement must divide the residual by about eight. A ratio near
four means the quadratic block is wrong and the solution is no better than first
order.

Two closed-form checks are in the test suite and worth reproducing on your own
model if you can find an analogue: for `log a' = ρ log a + ε` the risk term must
be exactly `h_ss = σ²`; and where the true policy is exact in the model's own
coordinates, the corresponding risk term must be exactly zero.

### Cost

The Sylvester equation is solved directly when `n_vars * n_states² ≤ direct_limit`
(6000 by default) and iteratively above it. The Hessian costs
`O((2n + n_shocks)²)` residual evaluations by finite differences; use the
symbolic front end (§9) to get it exactly and much faster.

---

## 6. Perfect-foresight transitions

```python
path = dk.perfect_foresight(model, start=ss0, terminal=ss1, params=p1, T=120)
path["k"]              # (T+1,) levels
path.levels            # (T+1, n_vars)
```

A stacked-time Newton on the same residual function the perturbation linearises.
The Jacobian is block-bidiagonal and assembled sparsely.

**Boundary conditions.** The `n_k` predetermined states at the start are
inherited from `start`; the `n_u` forward jumps at the end sit at `terminal`.
Fixing the whole terminal vector instead — the natural-looking mistake —
overdetermines the system by `n_k` equations, and the only symptom is a Newton
that stops converging.

### Announced paths

A *surprise* is a change in `terminal` with constant `params`. An *announced*
path is a sequence:

```python
schedule = [p_old] * 10 + [p_new] * (T + 1 - 10)
path = dk.perfect_foresight(model, ss0, ss1, params=schedule, T=T)
```

`params` and `ctx` each accept a constant, a sequence of length `T+1`, or a
callable `f(t)`. A statutory calendar known to agents from the outset is a
sequence; the difference from a surprise is visible on impact and is often the
whole point of the exercise.

If the equations need *next period's* parameters — a policy that agents know will
change between `t` and `t+1` — carry them in `ctx`, since `ctx` is also a path.

### Starting elsewhere

```python
path = dk.perfect_foresight(model, ss, ss, params=p, T=150,
                            initial_states=[0.5 * ss["k"], 1.0])
```

`initial_states` overrides the inherited states, for a transition that does not
begin at a steady state.

### Along the path

```python
series = path.derived(lambda vals, p, ctx: {"y": vals["a"] * vals["k"] ** p.alpha})
```

`derived` evaluates a per-period function of the variables over the whole path,
which is how objects defined by identities — prices, shares, flows — are
recovered without recomputing them by hand at every date.

---

## 7. Calibration on targets

```python
res = dk.calibrate(
    model, params,
    free={"beta": 0.95, "alpha": 0.30},
    targets={"ky": 0.35, "cy": 0.65},
    guess=guess,
    observables=lambda ss, p: {"ky": ss["k"] / (ss["a"] * ss["k"] ** p.alpha),
                               "cy": ss["c"] / (ss["a"] * ss["k"] ** p.alpha)},
    bounds={"alpha": (0.05, 0.9), "beta": (0.5, 0.999)},
)
print(res.report())
res.params, res.steady, res.values
```

Targets are steady-state variables unless `observables` supplies them. The whole
system is solved at once, which is the point: done by hand this becomes a nest of
bespoke bisections, one per parameter, each hiding the fact that the targets are
**jointly** determined.

Warm starting is on by default and is worth an order of magnitude in a sweep; the
accepted solution is still judged on its own residual.

### Identification

```python
ident = dk.target_jacobian(model, params, free=["beta", "alpha"],
                           targets=["k"], guess=guess)
ident["identified"]        # False
ident["rank"], ident["n_free"]
ident["null_directions"]   # the parameter directions no target sees
ident["elasticity"]        # d log target / d log parameter
```

A calibration whose elasticity matrix is rank-deficient is not identified: some
direction in parameter space leaves every target where it was, and the numbers
that come out of it are whichever point the solver happened to stop at. Run this
whenever a calibration "will not converge" — no solver tolerance fixes an
unidentified system.

---

## 8. Diagnostics

```python
dk.blanchard_kahn_report(sol)
```

Reports the **margin**, not only the pass. A model that satisfies Blanchard–Kahn
with a root at 0.9999 satisfies it in the same sense that a bridge holds with one
bolt: the classification flips under a parameter change small enough to be
invisible in a sweep.

```python
dk.check_identity(model, resource_constraint, path, params)
```

Evaluates an identity the model does **not** impose. The resource constraint of a
closed model is the canonical case: it follows from the budget constraints, the
firm's first-order conditions and the definitions of profit, so it is not one of
the equations — and that is what makes it evidence. A Walras residual is a leak
in the accounting; imposing it as an equation would hide the leak instead of
revealing it.

```python
dk.check_slack(sol, {"QF_positive": lambda v: v["QF"]}, size=3.0)
```

A `max(x, 0)` or a floor is not differentiable and cannot be perturbed, so
applied models replace it with its smooth interior branch. That substitution is
legitimate only while the constraint is slack — and slack at the steady state is
not slack along a large impulse response.

```python
dk.steady_state_invariance(build_fn, grid, ["Y", "K", "r"])
dk.compare_irfs(sol_a, sol_b, "eps_a")
```

The first asserts an invariance that is a *theorem* of the model over a parameter
grid — output independent of a redistributive policy, a great ratio pinned by the
Euler equation — which is the cheapest way to learn that a change broke the block
delivering it. The second compares two solutions' impulse responses, for checking
a port or a refactor: comparing steady states is not enough, since two different
models routinely share a steady state and differ entirely in their dynamics.

---

## 9. Text equations and exact derivatives

```python
from dsgekit.symbolic import from_equations

model = from_equations(
    [
        "k(+1) = (1-delta)*k + a*k**alpha - c",
        "c**(-gamma) = beta * c(+1)**(-gamma) * (alpha*a(+1)*k(+1)**(alpha-1) + 1 - delta)",
        "log(a(+1)) = rho*log(a) + eps_a",
    ],
    states=["k", "a"], jumps=["c"], shocks=["eps_a"],
    parameters=["alpha", "beta", "rho", "delta", "gamma"],
)
```

`x` is the variable at `t`, `x(+1)` at `t+1`, shock names are the innovation at
`t+1`. Equations may use `=` or be expressions that must vanish.

Beyond convenience, this gives **exact** derivatives: the residual, its Jacobian
and its Hessian are differentiated symbolically and used in place of finite
differences throughout. At second order that is not a nicety — a second
difference divides by the square of the step, so a finite-difference Hessian
carries roughly half the significant digits of the residual it came from, and the
risk corrections are precisely the small terms that loses.

Two deliberate restrictions:

- An undeclared name **raises** rather than becoming a free symbol. A typo must
  not quietly turn into a new variable. (This also means `beta`, `gamma`, `S`
  and other sympy objects are safe as parameter names — the parser is given no
  global namespace to bind them to.)
- A `(-1)` lag is refused with an explanation. Declare the lagged value as a
  state and give it its own transition equation, so that what is predetermined is
  stated in one place and cannot silently disagree with the equations.

`derivatives="jacobian"` skips the Hessian, which costs
`n_eq * (2n + n_shocks)²` expressions to build; `"none"` skips both.

---

## 10. Multi-group and multi-sector models

Declare the family with an index suffix and gather it in the equations:

```python
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
lists. Keep the two orders together in one place and the layout cannot drift.

Constants that do not vary with the solution — group masses, calibrated shares,
a policy schedule — belong in `ctx`, not in the parameter dataclass, when they
are arrays: it keeps `dataclasses.replace` usable for the scalar sweep.
