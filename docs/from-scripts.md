# Porting an existing research script

This note describes how to move a research script to `dsgelab` when the script
has a fixed-point loop for the steady state, a second loop for a policy variable,
a block of aggregates frozen at a reference point and, added later, a separate
linearised system for the stochastic exercises. It also lists what the move
provides.

The code blocks are sketches for the script's own model. Names such as
`statics`, `solve_ss`, `phi_grid`, `ctx0`, `renew_share` and the variables `QF`,
`QG`, `QR`, `D` and `base` stand for objects of that script, and `dsg` is
`import dsgelab as dsg`. For blocks that run as written, see the
[user guide](guide.md).

## What the move provides

**One model instead of two.** In this kind of code, the perturbation solution and
the deterministic transition can rest on different equations, similar, written by
the same person, and not identical. Once the equilibrium conditions are a single
residual function, the transition and the stochastic block cannot diverge,
because both call it.

**Objects that a fixed point does not give.** A fixed-point loop returns a steady
state. The same equations written declaratively also give generalised
eigenvalues, a determinacy verdict, impulse responses, moments, a variance
decomposition, second-order risk corrections and a nonlinear transition, all for
the model that produced the steady state.

**Checks with a reported number.** The order-of-accuracy test, the Blanchard–Kahn
margin, the Walras identity and the slackness of each smoothed corner each return
a number that can be reported. The Blanchard–Kahn condition runs on every solve,
and the other checks are one call each.

---

## Step 1. Fixed-point loops become equations

A loop like

```python
piB = 0.10
for _ in range(6000):
    benefit = piTE + (1 - phi) * piB
    ...
    piB_new = (RN0 + Gamma) / base
    piB = 0.5 * piB + 0.5 * piB_new
```

is a *variable* with an *equilibrium condition*. Declare it as a jump and write
the condition as a residual.

```python
jumps = [..., "piB"]

def equations(nxt, cur, eps, p, ctx):
    s = statics(cur, p, ctx)
    ...
    r[b] = s["piB"] * s["base"] - (s["RN0"] + s["Gamma"])
```

The static block that the loop recomputed on each pass becomes an ordinary helper
function of the current variables. Keep it, since `TransitionPath.derived` uses
it to recover the derived objects (prices, shares, flows) along a path, and the
constraints passed to `check_slack` read the corner quantities from it.

A quantity that was simultaneous with the loop but not itself iterated, because
under the old functional form it happened to be exogenous, must become a
variable too. Demand that was `D = ν Y` is one equation. Demand from a CES
aggregator is simultaneous with the price and needs the price to be endogenous.

## Step 2. Frozen reference blocks become live equations

Scripts often freeze a block at the reference steady state, holding output,
capital, the interest rate and the wage at `_reference` while the policy variable
moves. That is defensible when an invariance makes it exact, and the invariance
can be tested. Here `solve_ss(phi)` returns the `SteadyState` at the policy
`phi`.

```python
dsg.steady_state_invariance(lambda phi: solve_ss(phi), phi_grid, ["Y", "K", "r", "w"])
```

If the invariance holds, activating the block costs nothing and the test
documents why. If it does not, the frozen version was wrong and the test says so.
In both cases the result is recorded by a test instead of a comment in the code.

## Step 3. Corners become smooth branches plus a check

`max(D - QR - QG, 0)` and `max(base, 1e-9)` cannot be perturbed. Replace them
with the interior branch and verify slackness where it must hold.

```python
dsg.check_slack(sol, {
    "QF_positive": lambda v: v["QF"],
    "base_positive": lambda v: v["base"],
}, size=3.0, periods=80)
```

The check runs along the impulse responses, which return to the steady state,
because a constraint slack at a point can bind along a path. `size` is the
innovation in the units it has in the equations.

## Step 4. Anchors go in `ctx`

A portfolio adjustment cost centred on steady-state wealth, or an exogenous level
pinned to equilibrium demand, must not be replaced by the solution itself,
because the term would then vanish and the margin it pins would become
indeterminate. Carry it in `ctx` and let the outer loop settle it.

```python
def ctx_update(values, ctx):
    return {**ctx, "K_anchor": [values[f"K{j}"] for j in range(N)],
                   "QR": renew_share * statics(values, p, ctx)["D"]}

ss = dsg.steady_state(model, guess, params, ctx=ctx0, ctx_update=ctx_update,
                      positive=[f"K{j}" for j in range(N)])
```

`positive` takes model variables only. `QR` lives in `ctx`, so its sign is up to
`ctx_update`.

## Step 5. The calibration loops become one solve

Several nested bisections, one per parameter, each pinned on one target, hide the
fact that the targets are jointly determined. One call replaces them.

```python
def observables(ss, p):
    return {"mu": ss["QG"] / ss["D"],
            "elec_elasticity": elec_elasticity(ss, p),
            "damage": damage_share(ss, p)}

res = dsg.calibrate(model, params,
                    free={"kappaG": 8.0, "renew": 0.6, "d1": 0.01},
                    targets={"mu": 0.056, "elec_elasticity": 3.62, "damage": 0.003},
                    guess=guess,
                    observables=observables)
print(res.report())
```

`calibrate` does not check identification, so check it once with
`target_jacobian`.

```python
dsg.target_jacobian(model, params, free=["kappaG", "renew", "d1"],
                    targets=["mu", "elec_elasticity", "damage"], guess=guess,
                    observables=observables)["identified"]
```

## Step 6. The transition script disappears

A hand-written transition that solves one state's path with everything else
frozen is replaced by

```python
path = dsg.perfect_foresight(model, ss_old, ss_new, params=p_new, T=120)
```

in which every equilibrium condition holds in every period. If the policy
follows a statutory calendar known to agents, pass the schedule in place of a
constant, and the agents respond from date 0 to the announced path.

```python
from dataclasses import replace

schedule = [replace(p, phi=x) for x in phi_path]      # length T+1
path = dsg.perfect_foresight(model, ss_old, ss_new, params=schedule, T=T)
```

Element `t` of the schedule enters the equations that link date `t` to `t+1`, so
the solver reads elements 0 to `T-1`, and element `T` serves only `path.derived`
at the last date. The end of the path is tied to `ss_new`, whose jumps fix those
of date `T`, so `ss_new` should be the steady state under the policy that the
schedule has reached by then. An assertion that the last elements of `phi_path`
equal the policy behind `ss_new` keeps the two consistent.

## Step 7. Second order stops being an external dependency

If the mean effects were being produced by a port to another language or another
package, `solve_second_order` computes them in the same process, on the same
equations, with `mean_correction()` in closed form. An external port, if one
exists, can then serve as an independent check, by comparing its impulse
responses and its risk coefficients with the `h_ss` and `g_ss` that
`solve_second_order` produces.

---

## Order of work, and how to know each step is right

1. **Static block first.** Write `statics(values, params, ctx)` and check that it
   reproduces the script's reference numbers at the reference point.
2. **Residuals second.** Write `equations` and evaluate them at the script's own
   steady state. With `old` the script's solution as a mapping of names to
   values, `dsg.diagnose(model, params, old)["residual"]`, the largest absolute
   residual at that point, must be at machine zero. If the two codes encode the
   same equations, that residual is zero. The test evaluates every equation at
   the old point, so it covers equations whose effect a comparison of a few
   outputs could miss. (`steady_state(...).residual` does not serve here, since
   it is the residual at the point where the solver converged, wherever it
   started.)
3. **Solve.** `sol.accuracy()` must give an implied order near 2, and
   `blanchard_kahn_report` must show the condition satisfied, with its margin as
   the number to report.
4. **Identity.** `check_identity` on the resource constraint, along a transition
   (`path.levels`) as well as at the steady state.
5. **Invariances.** `steady_state_invariance` on whatever the old code froze.
6. **Then** the new objects: moments, transitions, second order.

Step 2 comes before step 6 because the residual at the old solution is what shows
that the new code encodes the same model as the script.
