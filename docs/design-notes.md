# Design notes

Why the package does what it does. Each entry is a decision that has a plausible
alternative, and each failure described below is one the alternative produces
without raising an error.

---

## Timing: the innovation is dated `t+1`

`f(x_{t+1}, x_t, eps)` treats `eps` as `eps_{t+1}`, unknown at `t`, with
`E_t eps_{t+1} = 0`. The law of motion is `k_{t+1} = P k_t + Q eps_{t+1}`.

The alternative dates the innovation at `t`, inside the information set. Then
`x_{t+1}` is a deterministic function of `(x_t, eps_t)`, the conditional
expectation in every Euler equation is degenerate, and no uncertainty is left
within the period. First order is unaffected, because certainty equivalence makes
the two timings coincide. At second order every risk correction comes out zero,
and the output looks like a model in which risk happens not to matter.

The library cannot detect this mistake in a model you declare. The package's own
test suite checks the convention on `log a' = ρ log a + ε`, where the
second-order coefficient `h_ss` must come out at `σ²` (the correction in the
decision rule is half of it, the Jensen term `σ²/2`), and under the wrong timing
it is zero. The test is
`test_risk_correction_on_a_lognormal_ar1_is_half_the_variance`.

---

## Real ordered QZ, with the complex one as fallback

`ordqz(..., output="real")`, falling back to `output="complex"` if the real
reordering refuses. A 2×2 diagonal block of the real form holds a complex
conjugate pair, and the two roots of a pair share their modulus, so the ordering
by modulus never splits a block between the stable and the unstable subspaces.
`klein` never reads a block entry by entry either. It inverts `Z11`, solves a
linear system in `T11` and multiplies by `S11`, each as a whole matrix. The real
form was not the first choice. Until October 2026 the complex one was the
default, for its 1×1 blocks. The default changed after the package's tests with
SciPy 1.18, in which LAPACK's complex reordering lost up to 1e-8 on a pencil with
a zero root and refused one with several infinite roots, while the real
reordering was exact on both (residual 1e-16 on the cases that failed).

## Blanchard–Kahn is enforced, and the margin is reported

Too few stable roots means no equilibrium, and too many means a continuum. In
either case `F` and `P` have no economic interpretation, so `klein` raises
`BlanchardKahnError` instead of returning them. `blanchard_kahn_report` reports
the *margin* between the largest stable and the smallest unstable root, because
a model that passes with a root at 0.9999 changes classification under a
parameter change small enough to be missed in a sweep, and the pass alone does
not show that.

## `A` is allowed to be singular

Static equations put zero rows in `A`, and nothing in the solver inverts it.
`F` and `P` come from the Schur vectors and the triangular blocks, and the shock
loading `Q` is solved on the rows that hold ex post (next section), so
`A^{-1} C` is never formed. An implementation that inverts `A` works on textbook
models and fails on any model with a static equation.

---

## The shock loading is not read off the QZ

The textbook formula for the innovation loading,

```
Q = Z11 T11^{-1} (Q^H C)_1
```

solves `A x_{t+1} = B x_t + C eps` as though *every* equation held exactly ex
post. Under this package's timing the innovation is dated `t+1`, so an equation
containing a next-period *jump* does not hold ex post. The jump responds to
`eps_{t+1}`, and `E_t` of that response is zero. Imposing such a row ex post adds
a false restriction, the resulting system for `Q` is inconsistent, and the QZ
returns a compromise between the true loading and the false restriction.

Three equations are enough to show it. With two independent AR(1) processes and
one Phillips curve, the loading on the two processes must be the identity, and
the textbook formula returns a dense matrix with off-diagonal entries of order
0.12. The eigenvalues and the transition matrix `P` are right, and the impulse
responses show no visible anomaly. Every moment, variance decomposition and
second-order risk correction computed from them is wrong, because all of them go
through `Q`.

The error vanishes on the standard textbook case. In an RBC with log utility and
full depreciation the Euler equation holds ex post, because the marginal product
of capital at `t+1` is independent of the technology shock at `t+1`, so every row
is an identity and the formula is correct. A test suite built on that model
alone does not detect the error.

`shock_loading`, which `klein` calls, therefore uses the rows that *do* hold ex
post, which are the rows containing no next-period jump. They are the transition
equation of each exogenous process, where the innovation enters, and the
accumulation identity of each endogenous state, where it does not. An
accumulation identity forces its row of `Q` to zero, since a capital stock chosen
at `t` cannot respond to a shock at `t+1`. There are enough such rows when every
predetermined variable has a law of motion, which is what being predetermined
means. When there are not, `shock_loading` raises `BlanchardKahnError` and states
how many of the states the rows span, instead of returning a minimum-norm answer.

The residual of that solve is stored as `shock_residual` and sits at round-off in
a well-posed model. Above `1e-8` times the larger of one and the largest
entry of `C`, `shock_loading` raises `BlanchardKahnError`, because a residual of that size
means some equation without a next-period jump nevertheless fails ex post. In
practice it is one whose only forward term is an exogenous state, such as
`q_t = beta f(a_{t+1})`. The fix is to introduce the forward term as a jump with
its own defining equation, which marks the expectational row as expectational.

## Coordinates: linear rescaling in place of log-linearisation

```
x = xbar + s * yhat,   s_i = |xbar_i| if nonzero, else 1
```

Two properties matter. First, the change of variables is *affine*, so the
second-order tensor transforms as a plain rescaling,
`H_hat[e,i,j] = H_level[e,i,j] s_i s_j`, and exact derivatives supplied in levels
can be converted with one multiplication. A genuine log-linearisation is
nonlinear and would not have this property.

Second, the branch is chosen by the steady-state *level*, and the variable's name
plays no part. A mean-zero shifter (a cost gap, or a wedge defined as a
deviation) has `xbar = 0`, and a rule that scaled by name would compute
`0 * (1 + yhat) = 0` and set the shock to zero without raising an error. The `nk`
example, whose every variable has a zero steady state, exists to exercise that
branch.

## Row equilibration

The rows of the pencil are scaled to unit maximum, and a row whose largest entry
is below 1e-12 is left unscaled. In an applied model the levels span many orders
of magnitude, with a capital stock of order 40 beside a mean-zero shifter, and
the unscaled pencil is ill-conditioned enough to move the eigenvalues that
Blanchard–Kahn counts. The column scaling and the row equilibration multiply the
pencil by diagonal matrices on the right and on the left, which makes them
equivalence transformations. They leave the solution unchanged and improve its
conditioning.

---

## Differencing steps

The Jacobian uses central differences at `h = 1e-6`. Forward differences are
first-order accurate and their truncation error enters the eigenvalues, which is
what the determinacy count reads.

Second differences use `h = 1e-4`. Round-off grows as `eps_machine / h²` while
truncation falls as `h²`, so the balance sits near `eps_machine^{1/4}`; the
`1e-6` used for the first derivative loses about six digits here.

The mixed partials use a six-point symmetric form that reuses cached single
displacements. The Hessian then costs `m² + m + 1` evaluations of the residual,
against `3m² − m + 1` if each pair recomputed its own displacements.

Exact derivatives from `dsgelab.symbolic` bypass all of this, and at second order
they remove the error of the second difference.

---

## Convergence is judged on the residual

The solver's success flag plays no part in accepting a solution.
`scipy.optimize.root` with `hybr` reports failure whenever its finite-difference
Jacobian stops improving, and that includes the case where the start point
already *is* the solution, with a residual of `5e-14` reported as a failure. It
also terminates early from a warm start that sits on the previous answer.
`steady_state` runs from the single start it is given and accepts on the maximum
residual alone, and `calibrate` solves the steady state that way at each
evaluation. `analytic_steady_state` solves nothing and accepts a closed form only
if its residual is below `tol`. Of these routines only `balanced_growth`
restarts, from a new random start, when a solution has a free level at zero.

---

## Stacked-time Newton: counting the boundary conditions

The boundary conditions are the `n_k` predetermined states at the start and the
`n_u` forward jumps at the end. Fixing the whole terminal vector overdetermines
the system by `n_k` equations. No error message is raised. Either the residual
vector is longer than the unknown vector, or the extra conditions are consistent
enough that Newton stops converging and the system looks like a hard problem.

The unknowns are the jumps at `t = 0` and the whole vector at `t = 1 … T`. The
Jacobian is block-bidiagonal and assembled sparsely.

With `jacobian="auto"`, the default, the Newton Jacobian uses the model's exact
derivatives when it has them (a model declared in text) and forward differences
otherwise. The accuracy of the converged path is set by `tol`, which the Jacobian does
not affect, so forward differences halve the cost of central ones without changing
the result. `jacobian="central"` is there for noisy residuals.

---

## Second order

### The Sylvester equation

The quadratic terms solve

```
Abar W + Bbar W (P kron P) = -K
```

with `W = [h_kk ; g_kk]` unfolded to `(n, n_k²)`, `Abar = [A1 | A3]`,
`Bbar = [0 | A2]`, `A1 = f1_k + f1_u F`, `A2 = f1_u`, `A3 = f2_u`, and
`K[e,i,j] = (Z' H_e Z)[i,j]` with `Z = [M; N; 0]`, `M = [P; F P]`, `N = [I; F]`.

It is solved directly, as `kron(Abar, I) + kron(Bbar, (P⊗P)')`, while
`n_vars * n_states²` does not exceed `direct_limit` (6000 by default). Above
that count a matrix-free LGMRES replaces the direct form, whose storage grows
with the square of the count.

### The risk terms

```
[A1 | A2 + A3] [h_ss ; g_ss] = -(A2 (g_kk : Q Sigma Q') + H : L Sigma L')
```

with `L = [Q; F Q; 0; I]` the derivative of the stacked argument with respect to
the perturbation parameter. This system is separate from the quadratic solve, and
the quadratic terms do not depend on `Sigma`, so a covariance sweep could reuse
`g_kk` and `h_kk`. `solve_second_order` does not exploit that. It solves the
Sylvester equation again on every call, and what a sweep can reuse is the Hessian,
passed as `hessian=`.

### Pruning

Pruning is the default. Without it, the quadratic term is evaluated on a state
that already contains the quadratic terms of earlier periods, and a long sample
explodes. Iterating a quadratic map produces this whatever the numerical
settings, so an unpruned second-order sample cannot support a moment over a long
horizon.

### The ergodic mean in closed form

`mean_correction` solves `E k^s = (I - P)^{-1} (½ h_kk : V_f + ½ h_ss)` instead
of averaging a simulation. The displacement is of order `σ²` and the series has
standard deviation of order `σ`, so separating them by averaging needs millions
of periods in a persistent model, and a shorter sample returns Monte Carlo noise
with a plausible sign and a plausible magnitude.

---

## Identities are checks and stay out of the equations

In a closed model the resource constraint *follows* from the budget constraints,
the firms' first-order conditions and the definitions of profit. It is therefore
not one of the equations, and its residual is an independent check on the
accounting. Imposing it as an equation would set that residual to zero by
construction and remove the check. `check_identity` computes that residual. The
user calls it, and `solve` does not.

The same logic governs `check_slack`. A `max(x, 0)` cannot be perturbed, so
applied models substitute the smooth interior branch. The substitution is valid
only while the constraint is slack, and a constraint slack at the steady state
need not be slack along a large impulse response, which is why the check runs
along the IRFs instead of at a single point.

---

## The order-of-accuracy test

`accuracy()`, which the user calls on a solution, feeds the decision rules back
into the nonlinear residuals at shrinking displacements and reports the implied
order. The test reads the ratio and disregards the size of any single residual,
because a wrongly assembled system can have a small residual at a small
displacement and still be first-order wrong. Halving the displacement must divide
the first-order residual by four and the second-order residual by eight.

For an exactly linear model the residuals sit at round-off, and the report says
`exact: True` instead of computing an order from the ratio of two numbers that are
both machine noise. Such a ratio carries no information about the order, and
reporting it would suggest an approximation error where the solution is exact.

---

## Parameters are copied, never mutated

`set_params` returns a copy. Calibration and policy sweeps evaluate the same
parameter object many times, and an in-place update makes every earlier result
refer to the latest values without any error. The symptom is a sweep whose
entries all equal the last one.
