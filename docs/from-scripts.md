# Porting an existing research script

Applied DSGE code tends to grow in a recognisable shape: a fixed-point loop for
the steady state, a second loop for a policy variable, a block of aggregates
frozen at a reference point, and — bolted on later — a separate linearised system
for the stochastic exercises. This note is about moving that to `dsgelab`, and
about what the move buys.

## What the move buys

**One model instead of two.** The most common structural problem in this kind of
code is that the perturbation solution and the deterministic transition rest on
different equations: similar, written by the same person, and not identical.
Once the equilibrium conditions are a single residual function, the transition
and the stochastic block cannot diverge, because both call it.

**Objects that do not exist otherwise.** A fixed point gives you a steady state
and nothing else. The same equations written declaratively give you generalised
eigenvalues, a determinacy verdict, impulse responses, moments, a
variance decomposition, second-order risk corrections and a nonlinear transition
— for the *same* model that produced the steady state.

**Checks you can quote.** The order-of-accuracy test, the Blanchard–Kahn margin,
the Walras identity, the slackness of every smoothed corner.

---

## Step 1 — fixed-point loops become equations

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
the condition as a residual:

```python
jumps = [..., "piB"]

def equations(nxt, cur, eps, p, cb):
    s = statics(cur, p, cb)
    ...
    r[b] = s["piB"] * s["base"] - (s["RN0"] + s["Gamma"])
```

The static block that the loop recomputed each pass becomes an ordinary helper
function of the current variables. Keep it: it is how the derived objects —
prices, shares, flows — are recovered along a path via `TransitionPath.derived`,
and how `check_slack` reads the corner quantities.

A quantity that was simultaneous with the loop but not itself iterated — because
under the old functional form it happened to be exogenous — must become a
variable too. Demand that was `D = ν Y` is one equation; demand from a CES
aggregator is simultaneous with the price and needs the price to be endogenous.

## Step 2 — frozen reference blocks become live equations

Scripts often freeze a block at the reference steady state: output, capital, the
interest rate and the wage held at `_reference` while the policy variable moves.
That is defensible when an invariance makes it exact, and it is worth *proving*
rather than assuming:

```python
dk.steady_state_invariance(lambda phi: solve_ss(phi), phi_grid, ["Y", "K", "r", "w"])
```

If the invariance holds, activating the block costs nothing and the test
documents why. If it does not, the frozen version was wrong and the test says so.
Either way the answer is evidence rather than a comment.

## Step 3 — corners become smooth branches plus a check

`max(D - QR - QG, 0)` and `max(base, 1e-9)` cannot be perturbed. Replace them
with the interior branch and verify slackness where it must hold:

```python
dk.check_slack(sol, {
    "QF_positive": lambda v: v["QF"],
    "base_positive": lambda v: v["base"],
}, size=3.0, periods=80)
```

At the steady state *and along the impulse responses*. Slack at a point is not
slack on a path, and that is the whole content of the check.

## Step 4 — anchors go in `ctx`

A portfolio adjustment cost centred on steady-state wealth, or an exogenous level
pinned to equilibrium demand, must not be substituted with the solution itself:
the term would vanish and the margin it pins would go indeterminate. Carry it in
`ctx` and let the outer loop settle it:

```python
def ctx_update(values, ctx):
    return {**ctx, "K_anchor": [values[f"K{j}"] for j in range(N)],
                   "QR": renew_share * statics(values, p, ctx)["D"]}

ss = dk.steady_state(model, guess, params, ctx=ctx0, ctx_update=ctx_update,
                     positive=[f"K{j}" for j in range(N)] + ["QR"])
```

## Step 5 — the calibration loops become one solve

Several nested bisections, one per parameter, each pinned on one target, hide the
fact that the targets are jointly determined. Replace them:

```python
res = dk.calibrate(model, params,
                   free={"kappaG": 8.0, "renew": 0.6, "d1": 0.01},
                   targets={"mu": 0.056, "elec_elasticity": 3.62, "damage": 0.003},
                   guess=guess,
                   observables=lambda ss, p: {"mu": ss["QG"] / ss["D"], ...})
print(res.report())
```

and check identification once:

```python
dk.target_jacobian(model, params, free=[...], targets=[...], guess=guess)["identified"]
```

## Step 6 — the transition script disappears

A hand-written transition that solves one state's path with everything else
frozen is replaced by

```python
path = dk.perfect_foresight(model, ss_old, ss_new, params=p_new, T=120)
```

in which **every** equilibrium condition holds in every period. If the policy
follows a statutory calendar known to agents, pass the schedule instead of a
constant and the announcement effect appears on impact:

```python
schedule = [replace(p, phi=x) for x in phi_path]      # length T+1
path = dk.perfect_foresight(model, ss_old, ss_new, params=schedule, T=T)
```

The last element of the schedule must be the terminal steady state's policy: the
terminal boundary condition points there, and a schedule ending elsewhere makes
the system inconsistent. Assert it rather than trusting it.

## Step 7 — second order stops being an external dependency

If the mean effects were being produced by a port to another language or another
package, `solve_second_order` computes them in the same process, on the same
equations, with `mean_correction()` in closed form. Keep the external port if you
have one: `compare_irfs` and a comparison of `h_ss`/`g_ss` make it an independent
check rather than a dependency.

---

## Order of work, and how to know each step is right

1. **Static block first.** Write `statics(values, params, ctx)` and check it
   reproduces the script's reference numbers at the reference point.
2. **Residuals second.** Write `equations` and verify
   `steady_state(...).residual` is at machine zero when handed the script's own
   steady state as a guess. If the equations are the same equations, the residual
   at the old solution is zero — this is the central regression test, and it is
   a stronger statement than any comparison of outputs.
3. **Solve.** `sol.accuracy()` must give an implied order near 2, and
   `blanchard_kahn_report` must pass with a margin you are willing to defend.
4. **Identity.** `check_identity` on the resource constraint, along the impulse
   responses, not only at the steady state.
5. **Invariances.** `steady_state_invariance` on whatever the old code froze.
6. **Then** the new objects: moments, transitions, second order.

Do not skip 2 to get to 6. The residual-at-the-old-solution test is what
distinguishes a port from a rewrite.
