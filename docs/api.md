# dsgelab API reference

Everything listed here is importable from the top-level package
(`import dsgelab as dk`), except `from_equations`, which also lives in
`dsgelab.symbolic`.

---

## Declaring

### `Model(states, jumps, shocks, equations, name="model", jacobian_fn=None, hessian_fn=None)`

| Argument | Meaning |
|---|---|
| `states` | predetermined variable names, in vector order; exogenous processes belong here |
| `jumps` | non-predetermined variable names |
| `shocks` | innovation names, dated `t+1` |
| `equations` | `f(nxt, cur, eps, params)` or `f(nxt, cur, eps, params, ctx)`; the signature is inspected once |
| `jacobian_fn`, `hessian_fn` | optional exact derivatives in model LEVELS w.r.t. `w = [x_next; x_now; eps]`; set by `from_equations` |

Properties: `names`, `n_states`, `n_jumps`, `n_vars`, `n_shocks`, `index`.
Methods: `residuals(x_next, x_now, eps, params, ctx=None)`, `vector(mapping)`,
`dict(array)`, `position(name)`, `shock_position(name)`, `jacobian(...)`,
`hessian(...)`.

### `VarView`

Name-indexed view over one period's vector, handed to the equations.

| Member | Meaning |
|---|---|
| `v["k"]`, `v.k` | value of one variable |
| `"k" in v` | membership |
| `v.get(name, default)` | value or a default |
| `v.vec("K", n=None)` | gather `K0, K1, …` into an array, in index order |
| `v.to_dict()`, `v.array` | the whole period |

### `from_equations(equations, *, states, jumps, shocks, parameters, name="model", derivatives="both")`

Build a `Model` from text equations, with exact analytic derivatives. `x` is the
variable at `t`, `x(+1)` at `t+1`. `derivatives` is `"both"`, `"jacobian"` or
`"none"`. Requires sympy. Raises `SymbolicError` on an undeclared name, a `(-1)`
lag, a wrong equation count or a parse failure.

### `get_param(params, name)` / `set_params(params, updates)`

Read one parameter, or return a **copy** with several replaced. Works for a
dataclass, a mapping or a plain object. Copying rather than mutating is why a
sweep does not silently rewrite its own earlier results.

---

## Steady state

### `steady_state(model, guess, params, *, ctx=None, ctx_update=None, positive=(), method="hybr", tol=1e-11, outer_maxiter=60, outer_tol=1e-12, raise_on_fail=True) -> SteadyState`

| Argument | Meaning |
|---|---|
| `guess` | mapping of names to values, or an array in variable order |
| `ctx` | side information passed to the equations as their fifth argument |
| `ctx_update` | `f(values, ctx) -> ctx`, run after each inner solve; the outer loop stops when it settles |
| `positive` | names solved in logs, keeping them strictly positive |
| `tol` | accepted maximum absolute residual — the acceptance criterion, not the solver's flag |

### `SteadyState`

Fields `x`, `values`, `residual`, `ctx`, `outer_iterations`, `success`,
`message`. `ss["k"]` reads one value. `scale(floor=1e-8)` is the linearisation
column scaling; `is_proportional(floor)` says which variables carry proportional
rather than absolute deviations.

---

## First-order solution

### `solve(model, params, *, steady=None, guess=None, ctx=None, ctx_update=None, positive=(), h=1e-6, equilibrate=True, tol=1e-11, qz_criterium=None) -> FirstOrderSolution`

Pass either a solved `steady` or a `guess`. `h` is the differencing step;
`equilibrate` scales the rows of the pencil, which does not change the solution
and does change its conditioning.

### `FirstOrderSolution`

| Member | Meaning |
|---|---|
| `P`, `F`, `Q` | `k_{t+1} = P k_t + Q eps`, `u_t = F k_t` |
| `eigenvalues` | ordered generalised eigenvalue moduli |
| `steady`, `system`, `klein`, `coords`, `params`, `ctx` | the objects behind the solution |
| `state_names`, `jump_names` | ordered name lists |
| `irf(name, size=1.0, periods=40, *, kind="shock", levels=False, percent=True)` | impulse response; `kind="state"` displaces a state directly |
| `irf_impulse(impulse, periods=40, *, levels=False, percent=True)` | response to a whole innovation VECTOR — an orthogonalised (Cholesky) impulse, which is what standard packages report when shocks are correlated |
| `simulate(periods=1000, cov=None, *, seed=None, burn=100, levels=False)` | draw a path |
| `moments(cov=None, lags=1)` | `std`, `corr`, `cov`, `autocorr`, `std_by_name`, exact via Lyapunov |
| `covariance(cov=None)` | unconditional covariance of `[k; u]` |
| `variance_decomposition(cov=None, horizon=None)` | unconditional, or FEVD at a horizon |
| `state_path(k0, periods, shocks=None)` | propagate the states |
| `variables(k)` / `to_levels(paths)` | deviations of every variable / convert to levels |
| `accuracy(sizes=(0.02, 0.01, 0.005), seed=0)` | implied order of accuracy against the nonlinear system |
| `summary()` | one-screen report |

### `klein(A, B, n_k, C=None, tol=1e-9, qz_criterium=None) -> KleinSolution`

The QZ solver on its own, for a system you assembled yourself. Raises
`BlanchardKahnError` (with `.n_stable`, `.n_k`, `.eigenvalues`) when the stable
root count does not match `n_k`, or when `Z11` is singular. `A` may be singular,
which is what static equations produce; nothing inverts it.

`KleinSolution`: `F`, `P`, `Q`, `n_stable`, `n_k`, `eigenvalues`, `n_u`,
`shock_rows`, `shock_residual`, `companion()`.

### `shock_loading(A, C, F, n_k, tol=1e-9) -> (Q, residual, rows)`

The shock loading on its own. It is **not** read off the QZ transformation: the
textbook formula solves the system as though every equation held ex post, and
under this package's timing an equation containing a next-period jump does not.
See [`design-notes.md`](design-notes.md#the-shock-loading-is-not-read-off-the-qz).

### `first_order_system(model, ss, params, *, ctx="inherit", h=1e-6, equilibrate=True) -> LinearSystem`

The `A`, `B`, `C` matrices, the `Coordinates` map, the steady-state residual and
the row scaling. Uses `model.jacobian_fn` when available.

### `Coordinates`

`center`, `scale`; `to_level(yhat)`, `to_hat(x)`,
`Coordinates.from_steady_state(ss, floor=1e-8)`.

### `shock_covariance(shocks, cov) -> ndarray`

Coerces a mapping of standard deviations, a vector of standard deviations, or a
full matrix into a covariance matrix.

---

## Second order

### `solve_second_order(first, cov=None, *, h=1e-4, direct_limit=6000, hessian=None) -> SecondOrderSolution`

`cov` enters **only** the risk corrections, so a covariance sweep does not need
the Sylvester equation solved again. `hessian` reuses a precomputed tensor.
`direct_limit` is the unknown count above which the Sylvester solve goes
iterative.

### `SecondOrderSolution`

| Member | Meaning |
|---|---|
| `g_kk`, `h_kk` | quadratic terms, `(n_u, n_k, n_k)` and `(n_k, n_k, n_k)` |
| `g_ss`, `h_ss` | risk corrections; the rules add half of these |
| `sylvester_residual` | residual of the generalised Sylvester solve |
| `jumps(k, risk=True)` / `next_states(k, eps=None, risk=True)` | the decision rules |
| `simulate(periods=1000, cov=None, *, seed=None, burn=200, pruning=True, levels=False)` | pruned by default |
| `mean_correction(cov=None)` / `ergodic_mean(cov=None, levels=True)` | closed-form ergodic mean |
| `accuracy(sizes, seed=0)` | implied order, which must be ~3 |

### `second_derivatives(model, ss, params, *, ctx="inherit", h=1e-4, row_scale=None) -> ndarray`

The `(n_eq, m, m)` Hessian with `m = 2 n_vars + n_shocks`, in rescaled
coordinates. Uses `model.hessian_fn` when available.

---

## Transitions

### `perfect_foresight(model, start, terminal, *, params, T=120, ctx=None, initial_states=None, maxiter=50, tol=1e-10, damping=1.0, jacobian="forward", h=1e-7, guess=None, raise_on_fail=True) -> TransitionPath`

| Argument | Meaning |
|---|---|
| `start`, `terminal` | steady states supplying the initial states and the terminal jumps |
| `params`, `ctx` | a constant, a sequence of length `T+1`, or `f(t)` — a sequence is an ANNOUNCED path |
| `initial_states` | override the inherited states |
| `guess` | `(T+1, n_vars)` initial iterate, e.g. a neighbouring solution |
| `jacobian` | `"forward"` (default) or `"central"` |

### `TransitionPath`

`levels` `(T+1, n_vars)`, `path["k"]`, `values()`, `periods`, `converged`,
`iterations`, `residual`, `params_path`, `ctx_path`,
`derived(fn)` where `fn(values, params, ctx) -> mapping`.

### `path_of(value, periods)`

Normalises a constant, a sequence or a callable into a per-period list.

---

## Reading Dynare files

Requires `sympy`. The module docstring of `dsgelab.dynare` lists what is read and
what is refused.

### `from_mod(source, *, defines=None, derivatives="jacobian", name=None) -> ModFile`

`source` is a path to a `.mod` file or its text; `defines` are macro variables,
as `-D` on Dynare's command line. Every Dynare endogenous variable becomes a jump
with its Dynare dating, so its impulse response is Dynare's series with no shift.
Lags become states `LAG<k>_v`, leads beyond one become jumps `LEAD<k>_v`, and a
shock `e` becomes the state `SHK_e`, which is what lets it appear inside any
equation. `predetermined_variables` are re-dated as Dynare does.

### `ModFile`

| Attribute | Meaning |
|---|---|
| `model` | the `Model` |
| `params` | parameter values; one left to a MATLAB steady-state file is `nan` and listed in `unset_params` |
| `guess` | endogenous values from `steady_state_model` or `initval`; names a MATLAB helper computes are in `missing_guess` |
| `cov` | shock covariance in `model.shocks` order |
| `auxiliary` | every variable the translation added, `name -> (kind, of)` |
| `linear`, `steady_operators` | `model(linear)`; variables under `steady_state()` |

`steady_state(guess=None, params=None, **kwargs)` solves the steady state,
iterating on `steady_state()` terms and writing their values into `params`.
`full_guess(values)` fills the auxiliaries from values of the Dynare variables.

```python
mf = dk.from_mod("fs2000.mod")
ss = mf.steady_state()
sol = dk.solve(mf.model, mf.params, steady=ss, qz_criterium=1 + 1e-6)
```

A model with a unit root -- a price level, a random-walk technology -- needs
`qz_criterium=1 + 1e-6`, Dynare's default; `solve` is strict otherwise.

---

## Optimal policy

Requires `sympy`. See the module docstring of `dsgelab.optimal` for the derivation.

### `ramsey_model(equations, *, states, jumps, shocks, parameters, instruments, objective, discount, name="ramsey", derivatives="both", multiplier_prefix="MULT_", auxiliary="auto") -> Model`

The Ramsey planner under commitment, timeless perspective. `equations` are the
private-sector conditions in the syntax of `from_equations`, one fewer per
instrument than there are variables. Equations that carry a shock are exogenous
processes: they are kept and get no multiplier. The planner's first-order
conditions are derived symbolically and the result is an ordinary `Model`.

| Argument | Meaning |
|---|---|
| `instruments` | the variables left without an equation |
| `objective` | the period objective `U(x_t)`, MAXIMISED, in current-period variables |
| `discount` | a parameter name, an expression in the parameters, or a number |
| `auxiliary` | `"always"` routes every predetermined variable's condition through an auxiliary jump `FOC_<k>`; the solution is the same, the option is for verification |

Layout of the returned model: states are the private states, then `lag_MULT_i`,
then lagged copies `lag_<v>` that the conditions need; jumps are the private
jumps, then `MULT_i`, then auxiliaries `FOC_<k>`. `MULT_i` belongs to the i-th
non-exogenous equation, with the sign of a Lagrangian `U + MULT' c`; Dynare
gives the same equation the same value. `model.ramsey` records the layout.

### `ramsey_steady_state(model, private, params, **kwargs) -> SteadyState`

Fits the multipliers to a guess of the private allocation by
`ramsey_multipliers`, then solves the whole system with `steady_state`
(`kwargs` forwarded, e.g. `positive=`).

### `ramsey_multipliers(model, private, params) -> dict`

At a steady state the conditions are linear in the multipliers, so they are
fitted by least squares. The result is a full guess; `"_residual"` is the fit's
residual, and a large one says that `private` is not the planner's steady state.

```python
m = dk.ramsey_model(
    ["x - x(+1) + (1/sigma)*(i - pi(+1)) = 0",
     "pi = beta*pi(+1) + kappa*x + u",
     "u(+1) = rho*u + e"],
    states=["u"], jumps=["x", "pi", "i"], shocks=["e"],
    parameters=["beta", "sigma", "kappa", "lam", "rho"], instruments=["i"],
    objective="-(pi**2 + lam*x**2)/2", discount="beta")
ss = dk.ramsey_steady_state(m, dict(u=0, x=0, pi=0, i=0), params)
irf = dk.solve(m, params, steady=ss).irf("e", size=1.0)
```

---

## Calibration

### `calibrate(model, params, *, free, targets, guess, observables=None, bounds=None, ctx=None, ctx_update=None, positive=(), steady_tol=1e-11, tol=1e-10, maxiter=200, warm_start=True, raise_on_fail=True) -> CalibrationResult`

`free` maps parameter names to starting values; `targets` maps target names to
wanted values; `observables(ss, params) -> mapping` supplies targets that are
functions of the steady state rather than variables of it.

`CalibrationResult`: `params`, `values`, `steady`, `targets`, `achieved`,
`max_relative_error`, `success`, `iterations`, `message`, `report()`.

### `target_jacobian(model, params, *, free, targets, guess, observables=None, step=1e-5, ...) -> dict`

Returns `derivative`, `elasticity`, `singular_values`, `rank`, `n_free`,
`identified`, `null_directions`, `condition`, `base_targets`, `base_params`.

---

## Diagnostics

| Function | Answers |
|---|---|
| `blanchard_kahn_report(solution)` | how much room the determinacy classification has: `largest_stable_root`, `smallest_unstable_root`, `margin` |
| `check_identity(model, fn, points, params, ctx=None, tol=1e-8)` | does an identity the model does NOT impose hold? `fn(values_next, values_now, params, ctx) -> float` |
| `check_slack(solution, constraints, shocks=None, size=1.0, periods=60, tol=0.0)` | do smoothed inequality constraints stay slack along the impulse responses? |
| `compare_irfs(a, b, shock, periods=40, size=1.0, variables=None)` | do two solutions agree, variable by variable? |
| `steady_state_invariance(build, values, variables, tol=1e-9)` | does a claimed invariance hold over a parameter grid? `build(v) -> SteadyState` |

---

## Bundled examples

```python
from dsgelab.examples import rbc, nk

model, params, guess = rbc.build()             # closed-form case, delta = 1
model, params, guess = rbc.build_symbolic()    # same model as text equations
rbc.closed_form(params)                        # the exact P and F
rbc.steady_levels(params)                      # (k, c)

model, params, guess = nk.build()              # three-equation New Keynesian
```

`rbc` has a solution known on paper and is the reference case for the solver.
`nk` has a zero steady state throughout, which exercises the absolute-deviation
branch, and its determinacy region is the Taylor principle.

## Command line

```
dsgelab --example rbc --periods 12
```

Solves a bundled example, prints the summary, the accuracy check and the impulse
response — a smoke test of the installed package.
