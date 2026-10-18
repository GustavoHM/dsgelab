# dsgelab API reference

Everything listed here is importable from the top-level package
(`import dsgelab as dsg`). The four exceptions, `dsgelab.klein.shock_loading`,
`dsgelab.foresight.path_of`, `dsgelab.bgp.bgp_residual` and
`dsgelab.symbolic.SymbolicError`, are written with their module path. Import them with
`from dsgelab.klein import shock_loading` and so on. The attribute `dsgelab.klein`
is the function `klein`, which the package namespace places over the module of the
same name, so `dsgelab.klein.shock_loading` does not resolve. `from_equations` is
top-level and also lives in `dsgelab.symbolic`. `conditional_forecast` is a method
of `FirstOrderSolution`, and methods and fields are listed under their class.
Entries marked "Requires sympy" need the `symbolic` extra
(`pip install "dsgelab[symbolic]"`).

---

## Declaring

### `Model(states, jumps, shocks, equations, name="model", jacobian_fn=None, hessian_fn=None)`

| Argument | Meaning |
|---|---|
| `states` | predetermined variable names, in vector order; exogenous processes belong here |
| `jumps` | non-predetermined variable names |
| `shocks` | innovation names, dated `t+1` |
| `equations` | `f(nxt, cur, eps, params)` or `f(nxt, cur, eps, params, ctx)`; the signature is inspected once |
| `jacobian_fn`, `hessian_fn` | optional exact derivatives in model levels with respect to `w = [x_next; x_now; eps]`, set by `from_equations` |

Properties: `names`, `n_states`, `n_jumps`, `n_vars`, `n_shocks`, `index`.

| Method | Meaning |
|---|---|
| `residuals(x_next, x_now, eps, params, ctx=None)` | the residual vector; raises `ValueError` when the equations return a number of residuals other than `n_vars` |
| `vector(values)` | a mapping of names to values as an array in variable order |
| `dict(x)` | an array in variable order as a mapping of names to values |
| `position(name)`, `shock_position(name)` | index of a variable or of a shock |
| `jacobian(x_next, x_now, eps, params, ctx=None)`, `hessian(x_next, x_now, eps, params, ctx=None)` | the exact derivatives, or `None` when the model has none |

### `VarView`

Name-indexed view over one period's vector, handed to the equations.

| Member | Meaning |
|---|---|
| `v["k"]`, `v.k` | value of one variable |
| `"k" in v` | membership |
| `v.get(name, default=None)` | value or a default |
| `v.vec(prefix, n=None)` | gather `K0, K1, …` (prefix `"K"`) into an array, in index order |
| `v.to_dict()`, `v.array` | the whole period |

### `from_equations(equations, *, states, jumps, shocks, parameters, name="model", derivatives="both", steady_state_model=None, calibration_equations=None)`

Requires sympy. Builds a `Model` from text equations, with exact analytic
derivatives. `x` is the variable at `t` and `x(+1)` at `t+1`.

A lag `x(-k)` is carried by the states `LAG1_x`, …, `LAGk_x`, with
`LAG1_x(+1) = x` and `LAGj_x(+1) = LAG{j-1}_x`, so `LAGk_x` holds `x` at `t-k`.
A lead `x(+k)` with `k >= 2` is rewritten as `LEAD{k-1}_x(+1)` and carried by the
jumps `LEAD1_x`, …, `LEAD{k-1}_x`, with `LEAD1_x = x(+1)` and
`LEADj_x = LEAD{j-1}_x(+1)`. `LEADj_x` therefore holds `x` at `t+j`, and `y(+2)`
creates `LEAD1_y`. Each auxiliary has its own equation and is listed in
`model.auxiliary`. The lag of a jump becomes a new state, and the jump keeps its
classification.

`derivatives` is `"both"`, `"jacobian"` or `"none"`. `steady_state_model` is a
closed form given as assignments `"name = expr"` in order (Dynare's block), which
`steady_state(model, None, params)` uses after checking it.
`calibration_equations` is `{parameter: "steady-state condition"}`, the
counterpart of MacroModelling's `k[ss]/q[ss] = 2.5 | β`, solved together with the
steady state. Raises `dsgelab.symbolic.SymbolicError` (a `ValueError`) on an
undeclared name, a lagged innovation, a wrong equation count or a parse failure.

### `optimality_conditions(problem) -> dict`

Requires sympy. Derives the first-order conditions of one agent's optimisation
problem, a `dsgelab.blocks.Problem`. The result holds `equations` (the conditions
and the constraints, as text ready for `from_equations`), `jumps` (the controls
and multipliers the problem introduces), `states`, `multipliers` and `detail`,
which maps each equation to the control, state or constraint it comes from.

### `assemble(problems, *, equations=(), states=(), jumps=(), shocks=(), parameters=(), name="model", derivatives="both") -> Model`

Requires sympy. Joins several agents' optimisation problems
(`dsgelab.blocks.Problem`, whose first-order conditions `optimality_conditions`
derives) and the closing equations (market clearing, exogenous processes,
definitions) in one `Model`. A variable chosen by two agents is refused, and a
price that both take as given is declared `exogenous` to both. Default
multipliers that clash are prefixed with the agent's name (`household_lm_1`).
The equation count is checked, and the error message gives the count per block.
`model.assembly` records the origin of every equation. `Problem` sets the limits,
which are equality constraints and nothing dated `t+2`.

### `get_param(params, name)` / `set_params(params, updates)`

`get_param` reads one parameter and `set_params` returns a copy with several
replaced. Both work for a dataclass, a mapping or a plain object. Because the
argument is left unchanged, a sweep does not overwrite the parameter objects of
its earlier points.

---

## Steady state

### `steady_state(model, guess, params, *, ctx=None, ctx_update=None, positive=(), method="hybr", tol=1e-11, outer_maxiter=60, outer_tol=1e-12, raise_on_fail=True, calibrate=None, homotopy=None, homotopy_steps=10) -> SteadyState`

| Argument | Meaning |
|---|---|
| `guess` | mapping of names to values, or an array in variable order; `None` uses the model's `steady_state_model` |
| `ctx` | side information passed to the equations as their fifth argument |
| `ctx_update` | `f(values, ctx) -> ctx`, run after each inner solve; the outer loop stops when the context moves by less than `outer_tol`, or after `outer_maxiter` rounds |
| `positive` | names solved in logs, which keeps them strictly positive |
| `method` | the method of `scipy.optimize.root` |
| `tol` | accepted maximum absolute residual, which decides acceptance (the solver's own success flag is ignored) |
| `raise_on_fail` | with `True` a solve that misses `tol` raises `RuntimeError`; with `False` the `SteadyState` comes back with `success=False` |
| `calibrate` | `{parameter: "condition"}`, parameters solved jointly with the steady state (default `model.calibration_equations`) |
| `homotopy` | `{parameter: (start, end)}`, a continuation from a start where the model is easy to solve, in `homotopy_steps` steps, each warm-started from the last; a failed step is halved, and a successful one doubles the next |

With `guess=None` the model's `steady_state_model` is used, checked as by
`analytic_steady_state`.

### `analytic_steady_state(model, params, values, *, ctx=None, tol=1e-10) -> SteadyState`

A steady state given in closed form, `values` being a mapping or `f(params)`.
It is accepted only if it sets the dynamic residuals at `(y, y, 0)` to within
`tol` of zero. Otherwise `SteadyStateError` names the equations it violates.
Nothing is solved.

### `SteadyState`

Fields `x`, `values`, `residual`, `ctx`, `outer_iterations`, `success`,
`message`. `ss["k"]` reads one value. `scale(floor=1e-8)` is the linearisation
column scaling, and `is_proportional(floor=1e-8)` marks the variables carried in
proportional deviations (the others are in absolute deviations). After
calibration, `calibrated` holds the solved parameters and `params` the full set.
After a homotopy, `homotopy_path` holds the fractions of the way at which it
stopped.

### `SteadyStateError`

Raised by `analytic_steady_state`, and so by `steady_state` with `guess=None`,
when a closed-form steady state leaves a variable without a value or does not
solve the model. It subclasses `RuntimeError`. A numerical solve that misses its
tolerance raises `RuntimeError` itself, so `except RuntimeError` catches both.

---

## First-order solution

### `solve(model, params, *, steady=None, guess=None, ctx=None, ctx_update=None, positive=(), h=1e-6, equilibrate=True, tol=1e-11, qz_criterium=None) -> FirstOrderSolution`

Pass either a solved `steady` or a `guess`. `h` is the differencing step, used
when the model has no exact Jacobian. `equilibrate` scales the rows of the
pencil, which leaves the solution unchanged and changes its conditioning. A
model that fails the Blanchard–Kahn condition raises `BlanchardKahnError`.

### `FirstOrderSolution`

| Member | Meaning |
|---|---|
| `P`, `F`, `Q` | `k_{t+1} = P k_t + Q eps`, `u_t = F k_t` |
| `eigenvalues` | ordered generalised eigenvalue moduli |
| `steady`, `system`, `klein`, `coords`, `params`, `ctx` | the objects behind the solution |
| `state_names`, `jump_names` | ordered name lists |
| `irf(name, size=1.0, periods=40, *, kind="shock", levels=False, percent=True)` | impulse response, a dict from each variable name to an array of length `periods` whose entry 0 is the impact date. `size` is the value of the innovation in the units it has in the equations (pass the standard deviation for a one-standard-deviation shock). The deviations are proportional for variables with a nonzero steady state and absolute for the rest, times 100 with `percent`; `levels=True` returns levels. `kind="state"` displaces a state directly |
| `irf_impulse(impulse, periods=40, *, levels=False, percent=True)` | response to a whole innovation vector, such as an orthogonalised (Cholesky) impulse when shocks are correlated |
| `simulate(periods=1000, cov=None, *, seed=None, burn=100, levels=False)` | draw a path, a dict of arrays |
| `moments(cov=None, lags=1)` | a dict with `names`, `std`, `corr`, `cov`, `autocorr` (`{lag: {name: value}}`) and `std_by_name`, computed from the discrete Lyapunov equation |
| `covariance(cov=None)` | unconditional covariance of `[k; u]` |
| `variance_decomposition(cov=None, horizon=None)` | unconditional, or the forecast-error variance decomposition at a horizon |
| `state_path(k0, periods=40, shocks=None)` | propagate the states |
| `conditional_forecast(conditions, *, periods=None, initial=None, controlled=None, levels=False)` | mean forecast with variables on prescribed paths (`nan` leaves a date free), the `controlled` shocks chosen to hit them and the others at zero, as Dynare's `conditional_forecast` with `controlled_varexo` |
| `historical_decomposition(states)` | innovations behind a state path and the path each produces |
| `variables(k)` / `to_levels(paths)` | deviations of every variable / convert to levels |
| `accuracy(sizes=(0.02, 0.01, 0.005), seed=0)` | implied order of accuracy against the nonlinear system |
| `summary()` | a one-screen report, as a string |

### `klein(A, B, n_k, C=None, tol=1e-9, qz_criterium=None) -> KleinSolution`

The QZ solver on its own, for a system assembled by hand. Raises
`BlanchardKahnError` when the stable root count does not match `n_k`, or when
`Z11` is singular. `A` may be singular, as static equations make it, since
nothing inverts it.

`KleinSolution`: `F`, `P`, `Q`, `n_stable`, `n_k`, `eigenvalues`, `n_u`,
`shock_rows`, `shock_residual`, `companion()`.

### `BlanchardKahnError`

A `RuntimeError` raised by `klein`, and so by `solve`, with `.n_stable`, `.n_k`
and `.eigenvalues`. The check runs on every solve.

### `dsgelab.klein.shock_loading(A, C, F, n_k, tol=1e-9) -> (Q, residual, rows)`

The shock loading on its own; `klein` calls it to compute `Q`. It solves for the
loading on the rows that hold ex post. The textbook formula reads `Q` off the QZ
transformation and treats every equation as holding ex post, and under this
package's timing an equation containing a next-period jump holds only in
expectation. See
[`design-notes.md`](design-notes.md#the-shock-loading-is-not-read-off-the-qz).

### `first_order_system(model, ss, params, *, ctx="inherit", h=1e-6, equilibrate=True) -> LinearSystem`

The `A`, `B`, `C` matrices, the `Coordinates` map, the steady-state residual and
the row scaling. Uses `model.jacobian_fn` when available.

`LinearSystem`: `A`, `B`, `C`, `coords`, `steady_residual`, `row_scale`.

### `Coordinates`

`center`, `scale`; `to_level(yhat)`, `to_hat(x)`,
`Coordinates.from_steady_state(ss, floor=1e-8)`.

### `shock_covariance(shocks, cov) -> ndarray`

Coerces a mapping of standard deviations, a vector of standard deviations, or a
full matrix into a covariance matrix in the order of `shocks`.

---

## Second order

### `solve_second_order(first, cov=None, *, h=1e-4, direct_limit=6000, hessian=None) -> SecondOrderSolution`

`cov` enters only the risk corrections, so a covariance sweep does not solve the
Sylvester equation again. `hessian` reuses a precomputed tensor. `direct_limit`
is the unknown count above which the Sylvester solve goes iterative.

### `SecondOrderSolution`

| Member | Meaning |
|---|---|
| `g_kk`, `h_kk` | quadratic terms, `(n_u, n_k, n_k)` and `(n_k, n_k, n_k)` |
| `g_ss`, `h_ss` | risk coefficients; the rules add half of each |
| `sylvester_residual` | residual of the generalised Sylvester solve |
| `jumps(k, risk=True)` / `next_states(k, eps=None, risk=True)` | the decision rules |
| `simulate(periods=1000, cov=None, *, seed=None, burn=200, pruning=True, levels=False)` | pruned by default |
| `mean_correction(cov=None)` / `ergodic_mean(cov=None, levels=True)` | closed-form ergodic mean |
| `at_covariance(cov)` | this solution if `cov` is the covariance it was computed at, else a solution computed at `cov`; `simulate`, `mean_correction` and `ergodic_mean` go through it, because `g_ss` and `h_ss` depend on the covariance |
| `accuracy(sizes=(0.02, 0.01, 0.005), seed=0)` | implied order of accuracy, about 3 for a correct solution |

### `second_derivatives(model, ss, params, *, ctx="inherit", h=1e-4, row_scale=None, at=None) -> ndarray`

The `(n_eq, m, m)` Hessian with `m = 2 n_vars + n_shocks`, in rescaled
coordinates. Uses `model.hessian_fn` when available. `at` is the point
`[ŷ_{t+1}; ŷ_t; eps]`, in rescaled coordinates, at which it is evaluated (the
steady state by default).

---

## Third order

### `solve_third_order(second, cov=None, *, h=1e-3, h_hess=1e-4, direct_limit=6000, third=None) -> ThirdOrderSolution`

Adds to a second-order solution the cubic terms `g_kkk` and `h_kkk` and the risk
slope `g_ssk` and `h_ssk`, the coefficients on `σ² k`. The rules become

```
u_t     = F k + ½ g_kk[k,k] + ½ g_ss + ⅙ g_kkk[k,k,k] + ½ g_ssk k
k_{t+1} = P k + ½ h_kk[k,k] + ½ h_ss + ⅙ h_kkk[k,k,k] + ½ h_ssk k + Q eps
```

`cov` defaults to the covariance of `second`. `h` and `h_hess` are the outer and
inner steps of `third_derivatives`, and `third` reuses a precomputed tensor.
Dynare's full third-order tensor, with the innovation in every monomial, and the
stochastic steady state are not implemented; both are planned for 0.7.

### `ThirdOrderSolution`

| Member | Meaning |
|---|---|
| `second` | the `SecondOrderSolution` it extends |
| `g_kkk`, `h_kkk` | cubic terms, `(n_u, n_k, n_k, n_k)` and `(n_k, n_k, n_k, n_k)` |
| `g_ssk`, `h_ssk` | risk slope, `(n_u, n_k)` and `(n_k, n_k)` |
| `sylvester_residual`, `risk_slope_residual` | residuals of the two linear solves |
| `jumps(k, risk=True)` / `next_states(k, eps=None, risk=True)` | the decision rules |
| `simulate(periods=1000, cov=None, *, seed=None, burn=200, levels=False)` | simulation pruned in three layers (Andreasen, Fernández-Villaverde and Rubio-Ramírez) |
| `accuracy(sizes=(0.02, 0.01, 0.005), seed=0)` | implied order of the cubic block with the risk terms switched off, about 4 for a correct solution |

### `third_derivatives(model, ss, params, *, ctx="inherit", h=1e-3, h_hess=1e-4, row_scale=None) -> ndarray`

The `(n_eq, m, m, m)` tensor of third derivatives with `m = 2 n_vars + n_shocks`,
in rescaled coordinates. It is a central difference, with step `h`, of the
Hessian, which costs `2m` Hessian evaluations. The Hessian is the model's exact
one when it has `hessian_fn`, and a second difference with step `h_hess`
otherwise.

---

## Global solution

### `solve_projection(model, params, *, bounds, guess, order=5, cov=None, n_quad=7, positive=(), steady=None) -> Projection`

Chebyshev collocation. The jump policy is a tensor-product Chebyshev polynomial
of degree `order` in each predetermined state, on the box `bounds` (one
`(low, high)` pair per state, in levels and in state order, containing the steady
state), with `order + 1` nodes a side. At each node the rows of the steady-state
Jacobian with no next-period jump give the next state, and the other rows are
integrated with a Gauss–Hermite rule of `n_quad` nodes per shock and set to zero.
The first-order policy is the starting point. More than three states, or a model
with no jump, is refused.

### `Projection`

`jumps(k)` evaluates the policy at one state vector (levels, state order) or at a
row-stacked array of them and returns the jumps in levels. Fields
`coefficients` `(n_u, n_basis)`, `nodes` (in levels), `bounds`, `order`,
`residual` (the largest absolute residual at the nodes) and `success`.

---

## Transitions

### `perfect_foresight(model, start, terminal, *, params, T=120, ctx=None, initial_states=None, shocks=None, maxiter=50, tol=1e-10, damping=1.0, jacobian="auto", h=1e-7, guess=None, raise_on_fail=True) -> TransitionPath`

| Argument | Meaning |
|---|---|
| `start`, `terminal` | steady states supplying the initial states and the terminal jumps |
| `params`, `ctx` | a constant, a sequence of length `T+1` (an announced path), or `f(t)`; element `t` enters the equations linking date `t` to `t+1` |
| `initial_states` | levels of the initial states, an array in `model.states` order (a mapping is not accepted), overriding those of `start` |
| `guess` | `(T+1, n_vars)` initial iterate, e.g. a neighbouring solution |
| `shocks` | innovations, `(T, n_shocks)`: entry `t` links date `t` to `t+1`, all anticipated |
| `jacobian` | `"auto"` (the model's exact derivatives when it has them, else forward differences), `"exact"`, `"forward"` or `"central"`; the stacked Jacobian is assembled sparse |

A Newton iteration that does not converge in `maxiter` steps raises
`RuntimeError`, unless `raise_on_fail=False`.

### `perfect_foresight_with_expectation_errors(model, start, terminal, *, params, T, news, shocks=None, ctx=None, initial_states=None, constant_simulation_length=False, **kwargs) -> TransitionPath`

Perfect foresight in which agents learn, at the dates in `news`, that they were
wrong. At each date they take the states the realised path has brought them to,
form a new plan, and follow it until the next news. The path returned is the
realised one, spliced from the plans (`result.plans`, starting at
`result.plan_dates`). Every plan ends at date `T`, or, with
`constant_simulation_length`, runs `T` periods past its own date, as in Dynare's
`perfect_foresight_with_expectation_errors_solver` and its option of the same
name. `kwargs` go to `perfect_foresight`.

### `News(period, shocks=None, params=None, terminal=None, states=None)`

What arrives at date `period`: the innovations now expected (absolute clock, entry
`t` links `t` to `t+1`; entry `period-1` is the one just realised, so a surprise
at `period` is a revision of it), parameters from `period` on, the steady state now
expected, and levels of states revealed directly. `None` keeps the previous value.
The states at `period` are otherwise recomputed from the realised `y_{period-1}`
through the model's transition rows.

### `TransitionPath`

`levels` `(T+1, n_vars)`, `path["k"]`, `values()`, `periods`, `converged`,
`iterations`, `residual`, `params_path`, `ctx_path`,
`derived(fn)` where `fn(values, params, ctx) -> mapping`.

### `dsgelab.foresight.path_of(value, periods)`

Normalises a constant, a sequence or a callable into a list of length `periods`.
`perfect_foresight` applies it to `params` and `ctx` with `periods = T+1`. A
sequence of another length raises `ValueError`.

### `extended_path(model, steady, params, shocks=None, *, periods=None, cov=None, seed=None, horizon=200, order=0, nodes=3, tree="sparse", hybrid=None, initial_states=None, ctx=None, tol=1e-12, maxiter=50) -> ExtendedPath`

A stochastic simulation in which a nonlinear plan is solved every period (Fair
and Taylor, 1983). At each date the shock arrives, the states it leads to are
computed, the agents solve a problem that reaches the steady state `horizon`
periods later, and the first date of that plan is kept. `shocks` is
`(T, n_shocks)`, dates 1 to `T`. Without it, `periods` draws are taken from
`cov`, the covariance matrix of the shocks as an array in `model.shocks` order
(a mapping is not accepted, and `dsg.shock_covariance(model.shocks, sd)` turns a
mapping of standard deviations into the matrix). `initial_states` is an array in
state order.

With `order = p > 0` (Adjemian and Juillard's stochastic extended path), the
shocks of the next `p` periods are integrated over with `nodes` Gauss–Hermite
nodes per shock, on Dynare's `"sparse"` tree (only the branch with every shock at
zero keeps branching) or on the `"full"` one. In every child the transition rows
(those with no next-period jump) hold, and the expectational rows hold on
average. `hybrid` (a `SecondOrderSolution`, or `True`) adds the second-order risk
correction where a branch stops branching, as Dynare's `hybrid_order = 2`.

`ExtendedPath`: `levels` (row 0 the initial point, row `t` the outcome of the
`t`-th shock, Dynare's column `t+1`), `shocks`, `path["y"]`, `values()`,
`iterations`.

### `quadrature_nodes(cov, n=3) -> (nodes, weights)`

Tensor Gauss–Hermite nodes for the shocks with a variance, by Dynare's
construction (scaled by the upper Cholesky factor, first shock fastest, zero node
first). `cov` is a covariance matrix given as an array; a mapping is not
accepted.

---

## Occasionally binding constraints

Guerrieri and Iacoviello's (2015) piecewise-linear solution, as Dynare's `+occbin`.

### `OccbinModel(build, constraints)`

`build(binding)` returns the `Model` of the regime in which the constraints named in
the frozenset `binding` bind (the empty set is the reference regime, where the
steady state lies). `constraints` maps each name to `{"bind": (expr, op), "relax":
(expr, op) or None}` in current-period levels. A constraint binds where
`expr op 0`, and one without a relax condition is relaxed wherever it does not
bind. `OccbinModel.from_equations(equations, *, constraints, states, jumps, shocks,
parameters, name="model", **kwargs)` takes Dynare's tags,
`"[name='slot', bind='c', relax='d'] eq"`, and constraints as text,
`{"c": {"bind": "lam < 0", "relax": "b > m*y"}}`.

### `occbin_simulate(om, params, shocks, *, periods=100, maxit=30, curb_retrench=False, check_ahead=200, steady=None, guess=None, initial=None, qz_criterium=None) -> OccbinResult`

Surprise shocks `{shock: {date: value}}` (dates from 1). Each regime is linearised
at the reference steady state, with its residual there as a constant. Given the
dates at which each constraint binds, the path follows time-varying rules found
backwards from the last non-reference date. The guess of those dates is revised
with Dynare's rule (with `curb_retrench`, only the last date that should be
relaxed is released) until it reproduces itself. Agents re-plan at each surprise,
starting from the regimes they expected. A plan looks
`max(check_ahead, periods - last shock)` periods ahead, and further while a
constraint binds at its end.

`OccbinResult`: `piecewise` and `linear` paths in levels, `binding` (per
constraint, per date), `regimes` (the plan of each surprise date),
`expected_regimes(date)`, `iterations`. `OccbinError` is raised when no fixed
point is found or the guess cycles.

---

## Balanced growth

### `balanced_growth(model, params, *, guess=None, growth=None, fixed=None, ctx=None, tol=1e-12, seed=0, max_nfev=100000, tries=20) -> BalancedGrowth`

Dynare's `bgp.write` problem. `x_t = X g^t` must solve the model at two
consecutive dates, which gives `2n` equations in the levels and the growth
factors, solved by Levenberg–Marquardt from random starts around 1. The growth
factors are identified. The level of a growing variable is identified only
through its ratio to the variables that share its trend. A solution with a free
level at zero (the trivial path of a model in levels) is set aside and the search
restarts. `fixed` holds variables that do not grow at a given level.

`BalancedGrowth`: `levels`, `growth`, `residual`, `success`, `degenerate`,
`growing`, `ratio(a, b)` (refused when the two grow at different factors).

`dsgelab.bgp.bgp_residual(model, params, X, G, ctx=None)` is the residual itself,
the model evaluated at the two dates.

---

## Reading Dynare files

Requires sympy. The module docstring of `dsgelab.dynare` lists what is read and
what is refused.

### `from_mod(source, *, defines=None, derivatives="jacobian", name=None) -> ModFile`

`source` is a path to a `.mod` file or its text, and `defines` are macro
variables, as `-D` on Dynare's command line. Every Dynare endogenous variable
becomes a jump with its Dynare dating, so its impulse response is Dynare's series
with no shift. A lag `v(-k)` becomes the states `LAG1_v`, …, `LAG<k>_v`. A lead
`v(+k)` with `k >= 2` becomes `LEAD<k-1>_v(+1)`, carried by the jumps `LEAD1_v`,
…, `LEAD<k-1>_v`, so `y(+2)` creates `LEAD1_y`. A shock `e` becomes the state
`SHK_e`, which lets it appear inside any equation. `predetermined_variables` are
re-dated as Dynare does.

A model written in levels with declared trends (`trend_var(growth_factor=g) A;`,
`log_trend_var(log_growth_factor=g) A;`, `var(deflator=D) x;`,
`var(log_deflator=D) x;`) is detrended as Dynare does. `x(k)` becomes `x(k)*D(k)`
and `A(k)` its level times the growth since `t`. An equation that still depends
on the level of a trend other than through a common factor is refused; the test
is that the cross derivative of `log(lhs - rhs)` in the trend and any variable is
not zero, checked at several points. The variables of the result are the
detrended ones. Equations tagged `bind=`/`relax=` and the `occbin_constraints`
block build an `OccbinModel`, whose reference regime is `model`.

A construct that is not read raises `DynareImportError`, with the offending text.

### `ModFile`

| Attribute | Meaning |
|---|---|
| `model` | the `Model` |
| `params` | parameter values; one left to a MATLAB steady-state file is `nan` and listed in `unset_params` |
| `guess` | endogenous values from `steady_state_model` or `initval`; names a MATLAB helper computes are in `missing_guess` |
| `cov` | shock covariance in `model.shocks` order |
| `auxiliary` | every variable the translation added, `name -> (kind, of)` |
| `linear`, `steady_operators` | `model(linear)`; variables under `steady_state()` |
| `block` | the private block, a `PrivateBlock`; `model` is `None` when instruments are left without an equation |
| `planner` | `planner_objective`, `planner_discount`, `instruments` and the command (`"ramsey"` or `"discretion"`) |
| `osr_params`, `optim_weights`, `osr_bounds` | the `osr` problem, ready for `dsg.osr` |
| `det_shocks`, `histval` | the deterministic `shocks` block (`{e: {period: level}}`) and `histval` (`{(v, period): level}`) |
| `surprise` | `shocks(surprise)`, OccBin's unforeseen shocks, `{e: {period: value}}` |
| `occbin` | constraints, regime builder and `occbin_setup`/`occbin_solver` options |
| `homotopy` | `homotopy_setup`, `{parameter: (start, end)}` |
| `trends` | the declared `deflators` and `trend_vars` |
| `shock_states` | the state that carries each shock, for `shock_decomposition` |

A Dynare exogenous variable is a level with a steady state of its own. The state
`SHK_e` follows `SHK_e(+1) = EXO_SS_e + e`, and `EXO_SS_e` (from `initval`, zero by
default) is a parameter, so a permanent change in the level is a change in it.
`e(+1)` is read as the next value of that state.

| Method | Meaning |
|---|---|
| `steady_state(guess=None, params=None, *, analytic=False, **kwargs)` | solves the steady state, iterating on `steady_state()` terms and writing their values into `params`; with `analytic=True` the file's `steady_state_model` is taken as the steady state, checked and refused as Dynare refuses it |
| `full_guess(values=None)` | fills the auxiliaries from values of the Dynare variables |
| `deterministic(T, steady, *, det_shocks=None)` | `(shocks, initial_states)` for `perfect_foresight`, reproducing `perfect_foresight_setup(periods=T)`; date `t` is Dynare's period `t+1`, date `T` the terminal period, and `det_shocks` replaces the file's `shocks` block |
| `initial_states(steady, *, histval=None, exo_first=None)` | the first date's states, from `histval` (the file's by default) or else from `steady` |
| `news(table, T, params=None)` | `(news, start)` for `perfect_foresight_with_expectation_errors` from Dynare's information sets; a set that only confirms what was expected is skipped, as in Dynare |
| `occbin_model()`, `occbin_simulate(shocks=None, *, steady=None, **kwargs)` | OccBin with the file's surprise shocks and solver options |
| `extended_path(shocks=None, *, steady=None, **kwargs)` | `extended_path` with the file's covariance |
| `balanced_growth(params=None, **kwargs)` | `balanced_growth` with the exogenous variables held at their level |
| `estimation_problem(data, **kwargs)` | an `EstimationProblem` from `estimated_params`, `varobs` and the `estimation(...)` options; `kwargs` override them |

The example reads `fs2000.mod` from Dynare 6.4's examples, which
`validation/fetch.sh` downloads into `validation/mods/examples/`; it runs from the
repository root.

```python
mf = dsg.from_mod("validation/mods/examples/fs2000.mod")
ss = mf.steady_state()
sol = dsg.solve(mf.model, mf.params, steady=ss, qz_criterium=1 + 1e-6)
```

A model with a unit root, such as a price level or a random-walk technology,
needs `qz_criterium=1 + 1e-6`, Dynare's default. Without it `solve` counts a root
of modulus one as unstable.

### `read_expectation_errors_csv(path) -> dict`

Reads the CSV that `perfect_foresight_with_expectation_errors_setup(datafile=...)`
takes and returns the table that `ModFile.news` expects.

---

## Optimal policy

Requires sympy. See the module docstring of `dsgelab.optimal` for the derivation.

### `ramsey_model(equations, *, states=None, jumps=None, shocks=None, parameters=None, instruments=None, objective=None, discount=None, name="ramsey", derivatives="both", multiplier_prefix="MULT_", auxiliary="auto") -> Model`

The Ramsey planner under commitment, timeless perspective. `equations` are the
private-sector conditions in the syntax of `from_equations`, one fewer per
instrument than there are variables. Equations that carry a shock are exogenous
processes, which are kept and get no multiplier. The planner's first-order
conditions are derived symbolically and the result is an ordinary `Model`.

`equations` may also be a `ModFile` read from a `.mod` with `planner_objective`
and `ramsey_model(...)`, whose objective, discount and instruments are used
unless the arguments override them. Instruments are optional, as in Dynare.

| Argument | Meaning |
|---|---|
| `instruments` | the variables left without an equation |
| `objective` | the period objective `U(x_t)`, maximised, in current-period variables |
| `discount` | a parameter name, an expression in the parameters, or a number |
| `auxiliary` | `"always"` routes every predetermined variable's condition through an auxiliary jump `FOC_<k>`; the solution is the same, and the option is for verification |

In the returned model the states are the private states, then `lag_MULT_i`,
then the lagged copies `lag_<v>` that the conditions need. The jumps are the
private jumps, then `MULT_i`, then the auxiliaries `FOC_<k>`. `MULT_i` belongs to
the i-th non-exogenous equation, with the sign of a Lagrangian `U + MULT' c`, and
Dynare gives the same equation the same value. `model.ramsey` records the layout.
A malformed planner problem raises `RamseyError`.

### `ramsey_steady_state(model, private, params, **kwargs) -> SteadyState`

Fits the multipliers to a guess of the private allocation by
`ramsey_multipliers`, then solves the whole system with `steady_state`
(`kwargs` forwarded, e.g. `positive=`).

### `ramsey_multipliers(model, private, params) -> dict`

At a steady state the conditions are linear in the multipliers, so they are
fitted by least squares. The result is a full guess. `"_residual"` is the fit's
residual, and a large one says that `private` is not the planner's steady state.

```python
params = dict(beta=0.99, sigma=1.0, kappa=0.1, lam=0.05, rho=0.8)
m = dsg.ramsey_model(
    ["x - x(+1) + (1/sigma)*(i - pi(+1)) = 0",
     "pi = beta*pi(+1) + kappa*x + u",
     "u(+1) = rho*u + e"],
    states=["u"], jumps=["x", "pi", "i"], shocks=["e"],
    parameters=["beta", "sigma", "kappa", "lam", "rho"], instruments=["i"],
    objective="-(pi**2 + lam*x**2)/2", discount="beta")
ss = dsg.ramsey_steady_state(m, dict(u=0, x=0, pi=0, i=0), params)
irf = dsg.solve(m, params, steady=ss).irf("e", size=1.0)
```

### `discretionary_policy(equations, *, params, states=None, jumps=None, shocks=None, parameters=None, instruments=None, objective=None, discount=None, steady=None, cov=None, tol=1e-12, maxiter=3000, qz_criterium=1+1e-6) -> DiscretionSolution`

Optimal policy without commitment, in the linear-quadratic class, by Dennis's
(2007) algorithm as Dynare's `discretionary_policy` implements it. The private
block is stated as for `ramsey_model` (text, or a `ModFile` whose
`discretionary_policy` command names the instruments), and the objective is
maximised. A `.mod` file's `planner_objective` is a loss and is read with the
sign flipped. The equations must be linear and the objective quadratic, with no
linear term and no cross term between an instrument and the rest, and each of
these conditions is checked (`DiscretionError` when one fails). An equation with
an expected jump holds at `t`, a transition (whose only next-period terms are
states) at `t+1`, and a static one at `t`.

`DiscretionSolution`: `y_t = H y_{t-1} + G eps_t` over every private variable,
in absolute deviations, with Dynare's dating. Its methods are
`irf(name, size=1.0, periods=40, *, levels=False)`,
`irf_impulse(impulse, periods=40, *, levels=False)`, `covariance(cov=None)`,
`moments(cov=None)` and `welfare(cov=None, *, initial=None, shock=None)`, which
returns `{"unconditional", "conditional"}`. Dynare's conditional welfare uses the
first innovation of its own simulation when one was run, and `shock=` reproduces
it.

### `osr(model, params, *, rule_params, weights=None, objective=None, discount=None, cov=None, guess=None, steady=None, steady_fn=None, bounds=None, qz_criterium=None, method=None, options=None, penalty=1e8) -> OSRResult`

Optimal simple rule. It finds the coefficients `rule_params` that minimise
`sum_ij W_ij Cov(y_i, y_j)`, the first-order unconditional covariance in levels,
as Dynare's `osr` with `optim_weights`. `weights` is `{"y": 1, ("y", "pi"): 0.1}`
(a pair adds `w * Cov` once, as Dynare does) or a matrix over `model.names`. With
`objective=` and `discount=` instead of `weights=`, the loss is minus
unconditional welfare at second order, as Dynare's `osr` with
`planner_objective`.

The steady state is recomputed at each trial by `steady_fn(params)` or from
`guess`/`steady`. A trial without a determinate solution costs `penalty`. The
default optimiser is BFGS followed by a Nelder–Mead polish. `bounds={name: (lo,
hi)}` switches to L-BFGS-B and warns when a coefficient sits on a bound. A
malformed problem raises `OSRError`.

`OSRResult`: `params` (the optimal coefficients), `all_params`, `loss`,
`initial_loss`, `success`, `message`, `evaluations`, `solution`.

### `PrivateBlock`

The private sector before a policy closes it: sympy `expressions` in the naming of
`parse_system`, `states`, `jumps`, `shocks`, `parameters`, `labels`, and the
optional `instruments`, `objective`, `discount`. `ModFile.block` is one.

---

## Welfare

Requires sympy.

### `welfare(model, params, *, steady, cov, objective=None, discount=None, initial=None, name="WELFARE", qz_criterium=None) -> Welfare`

Appends the jump `WELFARE = U + beta * WELFARE(+1)` and solves to second order.
A Ramsey model supplies its own objective and discount. These are the numbers of
Dynare's `evaluate_planner_objective` at `order=2`. A model or objective that
cannot carry the calculation raises `WelfareError`.

`Welfare`: `unconditional` (the ergodic mean, in closed form), `conditional` (the
second-order rule at `initial`, the steady state by default), `deterministic`
(`U(steady)/(1-beta)`), `conditional_zero_multipliers` for a Ramsey model (the
lagged multipliers set to zero, which is the planner who commits at `t = 0`),
`discount`, `model` and `second_order`.

### `welfare_model(model, objective=None, discount=None, *, name="WELFARE") -> Model`

The augmented model, for using the welfare variable in IRFs or simulations.

### `consumption_equivalent(objective, variable, target, reference, *, model, params, steady, cov, discount, bracket=(-0.5, 0.5), **kwargs) -> float`

The permanent proportional change `lambda` in `variable`, applied inside the
objective, that brings unconditional welfare from `reference` to `target`. No log
utility and no variable called consumption is assumed. `kwargs` go to `welfare`.

---

## State space and Kalman filter

### `state_space(sol, observables, *, transforms=None, measurement=None, cov=None) -> StateSpace`

`k_{t+1} = T k_t + R eps_{t+1}`, `y_t = d + Z k_t + e_t` from a first-order
solution. Each observable is a `"level"` (default) or a `"log"`. A log observable
reproduces Dynare's `loglinear` with `logdata`; its constant `d` is the log of
the steady state, and the mapping is exact at first order in this package's
proportional coordinates. `measurement` gives the standard deviations of the
measurement errors (`{name: sd}`, a scalar, or a covariance). An observed growth
rate or a measured inflation rate is a variable of the model, defined by a model
equation, as in a `.mod`. Missing data are `nan`.

`StateSpace`: `T`, `R`, `Sigma`, `Z`, `d`, `H` (the measurement covariance),
`observables`, `states`, `shocks`, `transforms`, `solution`.

### `kalman_filter(ss, data, *, init="auto", univariate=None, tol=1e-12) -> FilterResult`

`ss` is a `StateSpace`. `init="auto"` starts stationary roots at their
unconditional distribution and unit roots diffuse, by the exact initialisation of
Koopman and Durbin (2000) with the unit-root block first in an ordered Schur
form, as Dynare's `diffuse_filter`. `"stationary"` and `"diffuse"` force one or
the other. The diffuse start and missing data both use the univariate
(observation by observation) filter, and `univariate=True` forces it. `tol` is
the variance below which a diffuse or prediction-error variance counts as zero.

`FilterResult`: `loglik`, `loglik_t` (per date), `a_pred`, `P_pred`, `a_filt`,
`P_filt` (predicted and filtered states and variances), `diffuse_periods`,
`univariate`.

### `kalman_smoother(ss, data=None, *, filtered=None, init="auto") -> SmootherResult`

Fixed-interval smoother (Durbin and Koopman, 2012), exact through the diffuse
periods. Pass the data, or a `FilterResult` as `filtered`.

`SmootherResult`: `states`, `shocks` (row `t` is the innovation that enters date
`t`), `measurement`, `state_variance` (`nan` in the diffuse periods), and
`variables()` in levels.

### `shock_decomposition(smoothed, *, variables=None, initial_shock_states=None) -> dict`

Each variable's smoothed path as the contribution of each shock plus the initial
condition, Dynare's definition. `initial_shock_states` (for a `.mod`,
`ModFile.shock_states`) counts the first value of a shock-carrying state as the
shock's.

### `realtime_shock_decomposition(space, data, *, init="auto", variables=None, initial_shock_states=None, first=2) -> dict`

The decomposition each date would have shown with the data up to it, as Dynare's
`realtime_shock_decomposition` pool, including its conventions (a first sample
of two observations, and the first period's shock left in the initial
condition).

### `forecast(space, data, horizon, *, init="auto", coverage=0.9, variables=None, include_measurement=False, logs=()) -> Forecast`

Point forecast from the last filtered state, and Gaussian bands from the
forecast-error variance (state uncertainty and future shocks).

`Forecast`: `mean`, `std`, `lower`, `upper`, `horizon`, `coverage`.

---

## Estimation

### `prior(shape, mean, sd, p3=None, p4=None) -> Prior`

Dynare's priors, set by mean and standard deviation. The shapes are `beta` (on
`[p3, p4]`), `gamma` (shifted by `p3`), `normal`, `inv_gamma` (type 1, for a
standard deviation, where `sd = inf` gives `nu = 2`), `inv_gamma2` and
`uniform`.

`Prior`: `shape`, `mean`, `sd`, `a` and `b` (the two parameters of the density),
`lower`, `upper`. `logpdf(x)` uses Dynare's normalising constants, and
`sample(rng, size=None)` draws from it.

### `EstimationProblem(model, data, params, observables, priors={}, cov=None, measurement=None, transforms=None, steady_fn=None, guess=None, init="auto", qz_criterium=None, first_obs=1, presample=0, free=None, bounds=None)`

What is estimated and against which data. Names follow Dynare's
`estimated_params`, with a parameter's own name, `"stderr e"` for the standard
deviation of a shock and `"stderr y"` for that of the measurement error on an
observable. `priors` map names to `Prior` or `(shape, mean, sd[, p3, p4])`.

| Method | Meaning |
|---|---|
| `log_likelihood(theta)`, `log_prior(theta)` | the two parts of the posterior |
| `log_posterior(theta)` | `-inf` where the model has no determinate solution |
| `solve(theta)` | `(FirstOrderSolution, StateSpace)` at `theta` |
| `filter(theta)`, `smoother(theta)` | a `FilterResult` and a `SmootherResult` |
| `start()` | the calibration, or the prior mean |

`ModFile.estimation_problem(data, **kwargs)` builds one from a `.mod`'s
`estimated_params`, `varobs` and `estimation(...)` options. With no initial value
Dynare starts from the prior mean, and so does `EstimationProblem`.

### `posterior_mode(problem, x0=None, *, method=None, options=None) -> ModeResult`

BFGS and a simplex polish in an unbounded reparametrisation of each prior's support.
`ModeResult`: `values`, `log_posterior`, `log_likelihood`, `hessian` (of minus the log
posterior, central differences), `covariance`, `std`, `laplace` (log marginal density).

### `maximum_likelihood(problem, x0=None, *, method=None, options=None) -> ModeResult`

The same on the likelihood, with standard errors from the inverse Hessian.

### `metropolis_hastings(problem, mode, *, draws=10000, chains=2, scale=0.8, init_scale=None, burn=0.5, seed=0) -> MCMCResult`

Random-walk Metropolis–Hastings with proposal `N(0, scale^2 * mode.covariance)`
(Dynare's `mh_jscale`), chains started over-dispersed (`mh_init_scale`), and the
first share `burn` of each chain dropped (`mh_drop`).

`MCMCResult`: `names`, `chains` `(n_chains, n_draws, n_params)`, `draws` (the
chains pooled), `log_posterior`, `acceptance`, `burn`. Its methods are `mean()`,
`quantiles(q=(0.05, 0.5, 0.95))`, `hpd(mass=0.9)`, `rhat()`, `ess()` and
`log_marginal_mhm(tau=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9))`
(Geweke's modified harmonic mean averaged over the truncations `tau`, as
Dynare's).

### `rhat(chains, names=None) -> dict`

Split R-hat (Gelman et al., 2013) of each parameter, from chains shaped
`(n_chains, n_draws, n_params)`. `MCMCResult.rhat()` calls it.

### `effective_sample_size(chains, names=None) -> dict`

Effective sample size of each parameter from the chains' autocorrelations,
truncated by Geyer's initial positive sequence. `MCMCResult.ess()` calls it.

### `smc(problem, *, particles=1000, stages=50, bend=2.0, mutation_steps=2, seed=0) -> SMCResult`

Likelihood-tempered SMC (Herbst and Schorfheide, 2014), from the prior. Its
`log_marginal` needs no mode and no harmonic mean.

`SMCResult`: `names`, `particles`, `weights`, `log_marginal`, `stages`,
`acceptance`.

### `prior_predictive(problem, draws=1000, *, seed=0, statistics=None) -> dict`

Shares of prior draws that are determinate, fail Blanchard–Kahn, or have no steady
state; `statistics(sol)` collected over the solvable draws.

### `model_probabilities(log_marginals, prior=None) -> dict`

Posterior model probabilities from log marginal likelihoods.

### `posterior_forecast(problem, draws, horizon, *, variables=None, n_draws=200, shocks=True, coverage=0.9, logs=(), seed=0) -> dict`

Forecast bands with parameter uncertainty (from MCMC or SMC draws) and shock
uncertainty, with keys `mean`, `lower`, `upper` and `paths`.

### `estimate(model, data, *, params, free, priors, guess, observables, measurement, cov=None, positive=(), bounds=None, draws=0, seed=0, proposal=None) -> EstimationResult`

The 0.2 interface, which remains alongside `EstimationProblem`. It maximises the
posterior over the parameters in `free` (name to starting value) and, with
`draws > 0`, runs a random-walk Metropolis–Hastings chain of that length. `data`
maps each variable in `observables` to a series in levels. `priors` maps names to
a `logpdf(value)` callable, `("normal", mean, sd)` or
`("invgamma", shape, scale)`, and `measurement` is the variance of the
observation error (a scalar, a vector or a matrix).

`EstimationResult`: `names`, `mode`, `log_posterior`, `draws`, `acceptance`,
`success`, `message`.

### `log_posterior(model, data, *, params, free, priors, guess, observables, measurement, cov=None, positive=()) -> float`

The 0.2 log posterior at the values in `free`, with the arguments of `estimate`;
`-inf` when the model has no saddle path.

### `log_likelihood(sol, Y, Z, R, Sigma) -> float`

The 0.2 Gaussian log likelihood of `Y`, `(T, n_obs)` in the solution's rescaled
deviations, under the state space of `sol` with observation matrix `Z`,
measurement covariance `R` and shock covariance `Sigma`. The filter starts at the
unconditional distribution of the states.

---

## Calibration

### `calibrate(model, params, *, free, targets, guess, observables=None, bounds=None, ctx=None, ctx_update=None, positive=(), steady_tol=1e-11, tol=1e-10, maxiter=200, warm_start=True, raise_on_fail=True) -> CalibrationResult`

`free` maps parameter names to starting values, and `targets` maps target names
to wanted values. `observables(ss, params) -> mapping` supplies targets that are
functions of the steady state and not variables of it. `calibrate` does not test
identification, which is what `target_jacobian` is for. A solve that misses
`tol` raises `RuntimeError` unless `raise_on_fail=False`.

`CalibrationResult`: `params`, `values`, `steady`, `targets`, `achieved`,
`max_relative_error`, `success`, `iterations`, `message`, `report()`.

### `target_jacobian(model, params, *, free, targets, guess, observables=None, step=1e-5, ctx=None, ctx_update=None, positive=(), steady_tol=1e-11) -> dict`

The derivatives of the targets with respect to the free parameters at `params`,
by finite differences with relative step `step`. Returns `derivative`,
`elasticity`, `singular_values`, `rank`, `n_free`, `identified` (`rank ==
n_free`), `null_directions` (the parameter directions that no target responds
to), `condition`, `base_targets` and `base_params`.

---

## Diagnostics

| Function | Answers |
|---|---|
| `blanchard_kahn_report(solution)` | how much room the determinacy classification has: `largest_stable_root`, `smallest_unstable_root`, `margin` |
| `check_identity(model, fn, points, params, ctx=None, tol=1e-8)` | does an identity that the model does not impose hold? `fn(values_next, values_now, params, ctx) -> float`; `points` is a `(T, n_vars)` array of levels, whose consecutive rows are paired (e.g. `path.levels`), or a sequence of `(x_next, x_now)` pairs |
| `check_slack(solution, constraints, shocks=None, size=1.0, periods=60, tol=0.0)` | do smoothed inequality constraints stay slack along the impulse responses? Each constraint is `fn(values) -> float`, positive when slack, evaluated on levels |
| `compare_irfs(a, b, shock, periods=40, size=1.0, variables=None)` | do two solutions agree, variable by variable? |
| `steady_state_invariance(build, values, variables, tol=1e-9)` | does a claimed invariance hold over a parameter grid? `build(v) -> SteadyState` |
| `diagnose(model, params, point, *, ctx=None, tol=1e-9)` | Dynare's `model_diagnostics`: the residual at the point (`residual`), variables absent at the current date, the rank of the static Jacobian with the collinear variables and equations of each relation, redundant equations of the dynamic Jacobian, a unit root, non-finite derivatives; `ok` and `messages` |

None of these runs by itself. The checks that run on every call are the
Blanchard–Kahn condition in `solve` and the residual test in `steady_state`.

---

## Exporting

Both writers need a model declared symbolically (by `from_equations`,
`assemble` or `ramsey_model`). A model declared by a callable raises
`dsgelab.export.ExportError`, a `RuntimeError`.

### `to_dynare(model, params, *, steady=None, cov=None, path=None, order=1, irf=40, extra="") -> str`

Writes the model as a Dynare `.mod` file and returns its text; with `path` it is
also written to disk. `steady` goes into an `initval` block, from which Dynare
runs `steady`. `cov` may have off-diagonal terms. `order` and `irf` are passed to
`stoch_simul`, and `extra` is inserted verbatim before it. `validation/roundtrip.py`
runs the output through Dynare and compares the results.

### `to_macromodelling(model, params, *, cov=None, path=None, name=None) -> str`

Writes the model as a MacroModelling.jl `@model`/`@parameters` file. Innovations
are standard normal in MacroModelling, so each standard deviation is written into
the equations as a parameter `sig_<shock>`. No round-trip test covers this
writer.

---

## Errors

| Class | Base | Raised when |
|---|---|---|
| `SteadyStateError` | `RuntimeError` | a closed-form steady state does not solve the model |
| `BlanchardKahnError` | `RuntimeError` | the stable root count does not match the number of states |
| `RamseyError` | `ValueError` | a planner problem is malformed or not representable |
| `DiscretionError` | `ValueError` | a discretionary problem is outside the linear-quadratic class, or the iteration fails |
| `OSRError` | `ValueError` | an optimal-simple-rule problem is malformed |
| `WelfareError` | `ValueError` | the objective or the model cannot carry a welfare calculation |
| `DynareImportError` | `ValueError` | a `.mod` construct is not read |
| `OccbinError` | `RuntimeError` | the regime iteration does not settle |
| `dsgelab.symbolic.SymbolicError` | `ValueError` | a text equation is malformed or uses an undeclared name |

A steady state, a transition or a calibration that misses its tolerance raises
`RuntimeError` itself.

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

`rbc` has a closed-form solution (log utility and full depreciation), against
which the tests check the solver. Every variable of `nk` has a zero steady state,
so the model exercises the absolute-deviation branch. It is determinate at the
default `phi_pi = 1.5` and indeterminate at `phi_pi = 0.8`. With `phi_x = 0.125`
the boundary lies near `phi_pi = 0.9875`, slightly below one.

## Command line

```
dsgelab --example rbc --periods 12
```

Solves a bundled example (`rbc` or `nk`) and prints the summary, the accuracy
check and the impulse response, which checks that the installed package runs.
