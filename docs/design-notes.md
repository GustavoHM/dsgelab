# Design notes

Why the package does what it does. Each entry is a decision that has a plausible
alternative, and each failure described below is one the alternative produces
silently.

---

## Timing: the innovation is dated `t+1`

`f(x_{t+1}, x_t, eps)` treats `eps` as `eps_{t+1}`, unknown at `t`, with
`E_t eps_{t+1} = 0`. The law of motion is `k_{t+1} = P k_t + Q eps_{t+1}`.

The alternative — the innovation known at `t` — looks harmless and is not. If
`eps_t` is in the information set at `t`, then `x_{t+1}` is a deterministic
function of `(x_t, eps_t)`, the conditional expectation in every Euler equation
is degenerate, and there is no uncertainty left within the period. First order is
unaffected, because certainty equivalence makes the two coincide. Second order is
destroyed: every risk correction comes out zero, and the output looks like a
model in which risk happens not to matter.

This is the one convention the package cannot verify for you. The check is
`test_risk_correction_on_a_lognormal_ar1_is_half_the_variance`: on
`log a' = ρ log a + ε` the correction must be exactly `σ²`, and under the wrong
timing it is zero.

---

## Complex QZ, not real

`ordqz(..., output="complex")`. The real generalised Schur form produces 2×2
diagonal blocks whenever generalised eigenvalues come close in modulus, and those
blocks mix the stable and unstable invariant subspaces. That case is not exotic:
it is what several exogenous AR(1) processes with similar persistence look like,
which is to say most applied models. The complex form gives clean 1×1 blocks and
the partition is unambiguous.

## Blanchard–Kahn is enforced, and the margin is reported

Too few stable roots means no equilibrium; too many means a continuum. Either way
`F` and `P` are arithmetic without meaning, so `klein` raises rather than
returning them. `blanchard_kahn_report` then reports the *margin* between the
largest stable and the smallest unstable root, because a model that passes with a
root at 0.9999 will flip classification under a parameter change too small to see
in a sweep, and the pass alone does not say so.

## `A` is allowed to be singular

Static equations put zero rows in `A`. Nothing in the solver inverts it: the
shock loading is taken through the QZ transformation,
`Q = Z11 T11^{-1} (Q^H C)[:n_k]`, rather than through `A^{-1} C`. An
implementation that inverts `A` works on textbook models and fails on the first
applied one.

---

## The shock loading is not read off the QZ

The textbook formula for the innovation loading,

```
Q = Z11 T11^{-1} (Q^H C)_1
```

solves `A x_{t+1} = B x_t + C eps` as though **every** equation held exactly ex
post. Under this package's timing the innovation is dated `t+1`, so an equation
containing a next-period **jump** does not hold ex post: the jump responds to
`eps_{t+1}` and `E_t` of that response is zero. Imposing such a row ex post adds
a false restriction, the resulting system for `Q` is inconsistent, and the QZ
quietly returns a compromise between the true loading and the false restriction.

Three equations are enough to show it. Two independent AR(1) processes and one
Phillips curve: the loading on the two processes must be the identity, and the
textbook formula returns a dense matrix with off-diagonal entries of order 0.12.
Nothing about the output looks wrong — the eigenvalues are right, the transition
matrix `P` is right, the impulse responses are smooth and plausible — and every
moment, variance decomposition and second-order risk correction computed from
them is wrong, because all of them go through `Q`.

The reason the error survives casual testing is that it **vanishes** on the
standard textbook case. In an RBC with log utility and full depreciation the
Euler equation happens to hold ex post — the marginal product of capital at
`t+1` is independent of the technology shock at `t+1` — so every row is an
identity and the formula is correct. A test suite built on that model alone
reports a clean bill of health.

`shock_loading` therefore uses the rows that **do** hold ex post: the rows
containing no next-period jump. Those are exactly the transition equation of
each exogenous process, where the innovation enters, and the accumulation
identity of each endogenous state, where it does not — forcing that row of `Q`
to zero, which is right, since a capital stock chosen at `t` cannot respond to a
shock at `t+1`. There are enough such rows precisely when every predetermined
variable has a law of motion, which is what being predetermined means; when
there are not, the solver says which states are unspanned rather than returning
a minimum-norm answer.

The residual of that solve is reported as `shock_residual` and must be at
round-off. A nonzero value means some equation without a next-period jump
nevertheless fails ex post — in practice one whose only forward term is an
exogenous state, such as `q_t = beta f(a_{t+1})`. The fix is to introduce the
forward term as a jump with its own defining equation, which marks the
expectational row as expectational.

## Coordinates: linear rescaling, not log-linearisation

```
x = xbar + s * yhat,   s_i = |xbar_i| if nonzero, else 1
```

Two properties matter. First, the change of variables is **affine**, so the
second-order tensor transforms as a plain rescaling,
`H_hat[e,i,j] = H_level[e,i,j] s_i s_j`, and exact derivatives supplied in levels
can be converted with one multiplication. A genuine log-linearisation is
nonlinear and would not.

Second, the branch is chosen by the steady-state **level**, not by the variable's
name. A mean-zero shifter — a cost gap, a deviation-defined wedge — has
`xbar = 0`, and a rule that scaled by name would compute `0 * (1 + yhat) = 0` and
annihilate the shock without any error. The `nk` example, whose every variable
has a zero steady state, exists to hold that branch.

## Row equilibration

The rows of the pencil are scaled to unit maximum. In an applied model the levels
span many orders of magnitude — a capital stock of order 40 beside a mean-zero
shifter — and the unscaled pencil is ill-conditioned enough to move the very
eigenvalues that Blanchard–Kahn is counting. Both the column scaling and the row
equilibration are similarity operations on the pencil: neither changes the
solution, both change whether you can compute it.

---

## Differencing steps

Central differences for the Jacobian, at `h = 1e-6`. Forward differences are
first-order accurate and their truncation error enters the eigenvalues, which is
exactly what the determinacy count reads.

Second differences at `h = 1e-4`. Round-off grows as `eps_machine / h²` while
truncation falls as `h²`, so the balance sits near `eps_machine^{1/4}`; the
`1e-6` that is right for a first derivative loses roughly six digits here.

The mixed partials use a six-point symmetric form that reuses cached single
displacements, costing `2m² + 2m` evaluations rather than `4m²`.

Exact derivatives from `dsgekit.symbolic` bypass all of this and are strongly
preferred at second order.

---

## Convergence is judged on the residual

Never on the solver's success flag. `scipy.optimize.root` with `hybr` reports
failure whenever its finite-difference Jacobian stops improving, and that
includes the case where the start point already *is* the solution — a residual of
`5e-14` reported as a failure. It also terminates early from a warm start sitting
exactly on the previous answer. Every routine here takes the best residual over
several start points and accepts on the residual alone.

---

## Stacked-time Newton: counting the boundary conditions

`n_k` predetermined states at the start, `n_u` forward jumps at the end. Fixing
the whole terminal vector overdetermines the system by `n_k` equations. The
symptom is not an error message — the residual vector is simply longer than the
unknown vector, or, worse, the extra conditions are consistent enough that Newton
merely stops converging and looks like a hard problem.

The unknowns are the jumps at `t = 0` and the whole vector at `t = 1 … T`; the
Jacobian is block-bidiagonal and assembled sparsely.

Forward differences are the default for the Newton Jacobian. The accuracy of the
converged path is set by `tol`, not by the Jacobian, so the halved cost is free;
`jacobian="central"` is there for noisy residuals.

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

Solved directly, as `kron(Abar, I) + kron(Bbar, (P⊗P)')`, while
`n_vars * n_states²` stays under `direct_limit`; above it a matrix-free LGMRES
takes over, since the direct form costs the square of that count in entries.

### The risk terms

```
[A1 | A2 + A3] [h_ss ; g_ss] = -(A2 (g_kk : Q Sigma Q') + H : L Sigma L')
```

with `L = [Q; F Q; 0; I]` the derivative of the stacked argument with respect to
the perturbation parameter. Independent of the quadratic solve, so a covariance
sweep reuses `g_kk` and `h_kk`.

### Pruning

The default. An unpruned second-order simulation lets the quadratic term feed on
itself, and a long sample explodes. That is not a numerical artefact to be tuned
away; it is what iterating a quadratic map does, and it is the reason an unpruned
second-order sample cannot support a moment at any horizon that makes the moment
meaningful.

### The ergodic mean in closed form

`mean_correction` solves `E k^s = (I - P)^{-1} (½ h_kk : V_f + ½ h_ss)` rather
than averaging a simulation. The displacement is of order `σ²` and the series has
standard deviation of order `σ`, so separating them by averaging needs millions
of periods in a persistent model — and a shorter sample returns Monte Carlo noise
with a plausible sign and a plausible magnitude, which is worse than returning
nothing.

---

## Identities are checks, not equations

`check_identity` exists because the resource constraint of a closed model
*follows* from the budget constraints, the firm's first-order conditions and the
definitions of profit. It is therefore not one of the equations, and that is what
makes its residual evidence: a nonzero Walras gap is income leaking somewhere in
the accounting. Adding it as an equation would remove one degree of freedom
elsewhere and hide the leak rather than fix it.

The same logic governs `check_slack`. A `max(x, 0)` cannot be perturbed, so
applied models substitute the smooth interior branch. The substitution is
legitimate exactly while the constraint is slack — and slack at the steady state
is not slack along a large impulse response, which is why the check runs along
the IRFs and not at a point.

---

## The order-of-accuracy test

`accuracy()` feeds the decision rules back into the nonlinear residuals at
shrinking displacements and reports the implied order. This is the discriminating
test, and the size of any single residual is not: a wrongly assembled system can
have a small residual at a small displacement and still be first-order wrong.
Halving the displacement must divide the first-order residual by four and the
second-order residual by eight.

For an exactly linear model the residuals sit at round-off and the report says
`exact: True` instead of computing an order from the ratio of two machine-noise
numbers — which would come out at 1.0 and invite the reader to conclude the
solver is first-order accurate when in fact it was exact.

---

## Parameters are copied, never mutated

`set_params` returns a copy. Calibration and policy sweeps evaluate the same
parameter object many times, and an in-place update makes every earlier result
silently refer to the latest values. The bug surfaces as a sweep whose entries
are all equal to the last one.
