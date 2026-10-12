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

### `from_equations(equations, *, states, jumps, shocks, parameters, name="model", derivatives="both", steady_state_model=None, calibration_equations=None)`

Build a `Model` from text equations, with exact analytic derivatives. `x` is the
variable at `t`, `x(+1)` at `t+1`. A lag `x(-k)` or a lead `x(+k)` with `k >= 2`
is carried by an auxiliary variable -- `LAGk_x`, a state, or `LEADk_x`, a jump --
with its own equation, listed in `model.auxiliary`; the lag of a jump is a new
state, never a reclassification of the jump. `derivatives` is `"both"`,
`"jacobian"` or `"none"`. `steady_state_model` is a closed form, assignments
`"name = expr"` in order (Dynare's block), used by `steady_state(model, None,
params)` only after it is checked; `calibration_equations` is `{parameter:
"steady-state condition"}`, MacroModelling's `k[ss]/q[ss] = 2.5 | β`, solved with
the steady state. Requires sympy. Raises `SymbolicError` on an undeclared name, a
lagged innovation, a wrong equation count or a parse failure.

### `assemble(problems, *, equations=(), states=(), jumps=(), shocks=(), parameters=(), name="model", derivatives="both") -> Model`

Several agents' optimisation problems (`dsgelab.blocks.Problem`, whose first-order
conditions `optimality_conditions` derives) and the closing equations -- market
clearing, exogenous processes, definitions -- in one `Model`. A variable chosen by
two agents is refused (a price both take as given is `exogenous` to both); default
multipliers that clash are prefixed with the agent's name (`household_lm_1`); the
equation count is checked with a count per block in the message. The origin of
every equation is in `model.assembly`. The limits are those of `Problem`:
equality constraints, nothing dated `t+2`.

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
| `calibrate` | `{parameter: "condition"}`: parameters solved jointly with the steady state (default: `model.calibration_equations`) |
| `homotopy` | `{parameter: (start, end)}`: continuation from a start where the model is easy, in `homotopy_steps` steps, each warm-started from the last; a failed step is halved, a successful one doubles the next |

With `guess=None` the model's `steady_state_model` is used, checked as by
`analytic_steady_state`.

### `analytic_steady_state(model, params, values, *, ctx=None, tol=1e-10) -> SteadyState`

A steady state given in closed form (`values` a mapping or `f(params)`), USED
only if it zeroes the dynamic residuals at `(y, y, 0)` to `tol`; otherwise
`SteadyStateError` names the equations it violates. Nothing is solved.

### `SteadyState`

Fields `x`, `values`, `residual`, `ctx`, `outer_iterations`, `success`,
`message`. `ss["k"]` reads one value. `scale(floor=1e-8)` is the linearisation
column scaling; `is_proportional(floor)` says which variables carry proportional
rather than absolute deviations. After calibration, `calibrated` holds the solved
parameters and `params` the full set; after a homotopy, `homotopy_path` holds the
fractions of the way at which it stopped.

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
| `conditional_forecast(conditions, *, periods=None, initial=None, controlled=None, levels=False)` | mean forecast with variables on prescribed paths (`nan` leaves a date free), the `controlled` shocks chosen to hit them and the others at zero — Dynare's `conditional_forecast` with `controlled_varexo` |
| `historical_decomposition(states)` | innovations behind a state path and the path each produces |
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

### `perfect_foresight(model, start, terminal, *, params, T=120, ctx=None, initial_states=None, shocks=None, maxiter=50, tol=1e-10, damping=1.0, jacobian="auto", h=1e-7, guess=None, raise_on_fail=True) -> TransitionPath`

| Argument | Meaning |
|---|---|
| `start`, `terminal` | steady states supplying the initial states and the terminal jumps |
| `params`, `ctx` | a constant, a sequence of length `T+1`, or `f(t)` — a sequence is an ANNOUNCED path |
| `initial_states` | override the inherited states |
| `guess` | `(T+1, n_vars)` initial iterate, e.g. a neighbouring solution |
| `shocks` | innovations, `(T, n_shocks)`: entry `t` links date `t` to `t+1`, all anticipated |
| `jacobian` | `"auto"` (the model's exact derivatives when it has them, else forward differences), `"exact"`, `"forward"` or `"central"`; the stacked Jacobian is assembled sparse |

### `perfect_foresight_with_expectation_errors(model, start, terminal, *, params, T, news, shocks=None, ctx=None, initial_states=None, constant_simulation_length=False, **kwargs) -> TransitionPath`

Perfect foresight in which agents learn, at the dates in `news`, that they were
wrong. At each date they take the states the realised path has brought them to,
form a new plan, and follow it until the next news; the path returned is the
REALISED one, spliced from the plans (`result.plans`, starting at
`result.plan_dates`). Every plan ends at date `T`, or, with
`constant_simulation_length`, runs `T` periods past its own date -- Dynare's
`perfect_foresight_with_expectation_errors_solver` and its option of the same name.
`kwargs` go to `perfect_foresight`.

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

### `path_of(value, periods)`

Normalises a constant, a sequence or a callable into a per-period list.

### `extended_path(model, steady, params, shocks=None, *, periods=None, cov=None, seed=None, horizon=200, order=0, nodes=3, tree="sparse", hybrid=None, initial_states=None, ctx=None, tol=1e-12, maxiter=50) -> ExtendedPath`

A stochastic simulation by a nonlinear plan every period (Fair and Taylor, 1983):
the shock arrives, the states it leads to are computed, the agents solve a problem
that reaches the steady state `horizon` periods later, and the first date is kept.
`shocks` is `(T, n_shocks)`, date 1 to `T`, or `periods` draws from `cov`. With
`order = p > 0` (Adjemian and Juillard's stochastic extended path) the shocks of
the next `p` periods are integrated over with `nodes` Gauss--Hermite nodes per
shock, on Dynare's `"sparse"` tree (only the branch with every shock at zero keeps
branching) or the `"full"` one: transition rows (no next-period jump) hold in every
child, expectational rows on average. `hybrid` (a `SecondOrderSolution`, or `True`)
adds the second-order risk correction where a branch stops branching -- Dynare's
`hybrid_order = 2`. `ExtendedPath`: `levels` (row 0 the initial point, row `t` the
outcome of the `t`-th shock, Dynare's column `t+1`), `shocks`, `path["y"]`,
`values()`, `iterations`.

### `quadrature_nodes(cov, n=3) -> (nodes, weights)`

Tensor Gauss--Hermite nodes for the shocks with a variance, Dynare's construction
(scaled by the upper Cholesky factor, first shock fastest, zero node first).

---

## Occasionally binding constraints

Guerrieri and Iacoviello's (2015) piecewise-linear solution, as Dynare's `+occbin`.

### `OccbinModel(build, constraints)`

`build(binding)` returns the `Model` of the regime in which the constraints named in
the frozenset `binding` bind (the empty set is the reference regime, where the
steady state lies). `constraints` maps each name to `{"bind": (expr, op), "relax":
(expr, op) or None}` in current-period LEVELS: it binds where `expr op 0`; without a
relax condition it is relaxed wherever it does not bind.
`OccbinModel.from_equations(equations, *, constraints, states, jumps, shocks,
parameters)` takes Dynare's tags, `"[name='slot', bind='c', relax='d'] eq"`, and
constraints as text, `{"c": {"bind": "lam < 0", "relax": "b > m*y"}}`.

### `occbin_simulate(om, params, shocks, *, periods=100, maxit=30, curb_retrench=False, check_ahead=200, steady=None, guess=None, initial=None, qz_criterium=None) -> OccbinResult`

Surprise shocks `{shock: {date: value}}` (dates from 1). Each regime is linearised at
the reference steady state, with its residual there as a constant; given the dates
at which each constraint binds, the path follows time-varying rules found backwards
from the last non-reference date, and the guess is revised with Dynare's rule
(`curb_retrench`: release only the last date that should be relaxed) until it
reproduces itself. Agents re-plan at each surprise, starting from the regimes they
expected; a plan looks `max(check_ahead, periods - last shock)` periods ahead and
further while a constraint binds at its end. `OccbinResult`: `piecewise` and
`linear` paths in levels, `binding` (per constraint, per date), `regimes` (the plan
of each surprise date), `expected_regimes(date)`, `iterations`; `OccbinError` when
no fixed point is found or the guess cycles.

---

## Balanced growth

### `balanced_growth(model, params, *, guess=None, growth=None, fixed=None, ctx=None, tol=1e-12, seed=0, max_nfev=100000, tries=20) -> BalancedGrowth`

Dynare's `bgp.write` problem: `x_t = X g^t` must solve the model at two consecutive
dates, `2n` equations in the levels and the growth factors, solved by
Levenberg--Marquardt from random starts around 1. Growth factors are identified;
the level of a growing variable is not, only its ratio to those sharing its trend.
A solution with a free level at zero (the trivial path of a model in levels) is set
aside and the search restarts. `fixed` holds variables that do not grow at a level.
`BalancedGrowth`: `levels`, `growth`, `residual`, `success`, `degenerate`,
`growing`, `ratio(a, b)` (refused when the two grow at different factors).
`bgp_residual(model, params, X, G)` is the residual itself.

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

A model written in levels with declared trends -- `trend_var(growth_factor=g) A;`,
`log_trend_var(log_growth_factor=g) A;`, `var(deflator=D) x;`,
`var(log_deflator=D) x;` -- is detrended as Dynare does: `x(k)` becomes `x(k)*D(k)`
and `A(k)` its level times the growth since `t`; an equation that still depends on
the level of a trend other than through a common factor (the cross derivative of
`log(lhs - rhs)` in the trend and any variable is not zero, checked at several
points) is refused. The variables of the result are the detrended ones. Equations
tagged `bind=`/`relax=` and the `occbin_constraints` block build an `OccbinModel`;
`model` is the reference regime.

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
| `osr_params`, `optim_weights`, `osr_bounds` | the `osr` problem, ready for `dk.osr` |
| `det_shocks`, `histval` | the deterministic `shocks` block (`{e: {period: level}}`) and `histval` (`{(v, period): level}`) |
| `surprise` | `shocks(surprise)`, OccBin's unforeseen shocks, `{e: {period: value}}` |
| `occbin` | constraints, regime builder and `occbin_setup`/`occbin_solver` options |
| `homotopy` | `homotopy_setup`, `{parameter: (start, end)}` |
| `trends` | the declared `deflators` and `trend_vars` |

A Dynare exogenous variable is a LEVEL with a steady state of its own: the state
`SHK_e` follows `SHK_e(+1) = EXO_SS_e + e`, and `EXO_SS_e` (from `initval`, zero by
default) is a parameter, so a permanent change in the level is a change in it.
`e(+1)` is read as the next value of that state.

`deterministic(T, steady)` returns `(shocks, initial_states)` for `perfect_foresight`,
reproducing `perfect_foresight_setup(periods=T)`: date `t` is Dynare's period `t+1`,
date `T` the terminal period. `news(table, T)` returns `(news, start)` for
`perfect_foresight_with_expectation_errors` from Dynare's information sets
(`read_expectation_errors_csv(path)` reads the CSV that
`perfect_foresight_with_expectation_errors_setup(datafile=...)` takes); a set that
only confirms what was expected is skipped, as in Dynare. `initial_states(steady,
histval=None, exo_first=None)` builds the first date's states.

`steady_state(guess=None, params=None, *, analytic=False, **kwargs)` solves the
steady state, iterating on `steady_state()` terms and writing their values into
`params`; with `analytic=True` the file's `steady_state_model` IS the steady state,
checked and refused as Dynare refuses it. `full_guess(values)` fills the auxiliaries
from values of the Dynare variables. `occbin_model()` and `occbin_simulate(shocks=None,
**kwargs)` run OccBin with the file's surprise shocks and solver options;
`extended_path(shocks, **kwargs)` with its covariance; `balanced_growth(**kwargs)`
with the exogenous variables held at their level.

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

### `ramsey_model(equations, *, states=None, jumps=None, shocks=None, parameters=None, instruments=None, objective=None, discount=None, name="ramsey", derivatives="both", multiplier_prefix="MULT_", auxiliary="auto") -> Model`

`equations` may also be a `ModFile` read from a `.mod` with `planner_objective`
and `ramsey_model(...)`; its objective, discount and instruments are used, and
the arguments override them. Instruments are optional, as in Dynare.

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

### `discretionary_policy(equations, *, params, states=None, jumps=None, shocks=None, parameters=None, instruments=None, objective=None, discount=None, steady=None, cov=None, tol=1e-12, maxiter=3000, qz_criterium=1+1e-6) -> DiscretionSolution`

Optimal policy without commitment, in the linear-quadratic class, by Dennis's
(2007) algorithm as Dynare's `discretionary_policy` implements it. The private
block is stated as for `ramsey_model` -- text, or a `ModFile` whose
`discretionary_policy` command names the instruments --, and the objective is
MAXIMISED; a `.mod` file's `planner_objective` is a loss there and is read with
the sign flipped. The equations must be linear, the objective quadratic with no
linear term and no cross term between an instrument and the rest; each is
checked. An equation with an expected jump holds at `t`; a transition, whose
only next-period terms are states, at `t+1`; a static one at `t`.

`DiscretionSolution`: `y_t = H y_{t-1} + G eps_t` over every private variable,
in absolute deviations, Dynare dating. `irf(shock, size=1, periods=40,
levels=False)`, `irf_impulse(vector)`, `covariance(cov)`, `moments(cov)`,
`welfare(cov, initial=None, shock=None)` -> `{"unconditional", "conditional"}`.
Dynare's conditional welfare uses the first innovation of its own simulation when
one was run; `shock=` reproduces it.

### `osr(model, params, *, rule_params, weights=None, objective=None, discount=None, cov=None, guess=None, steady=None, steady_fn=None, bounds=None, qz_criterium=None, method=None, options=None, penalty=1e8) -> OSRResult`

With `objective=` and `discount=` instead of `weights=`, the loss is minus
unconditional welfare at second order -- Dynare's `osr` with `planner_objective`.

Optimal simple rule: the coefficients `rule_params` that minimise
`sum_ij W_ij Cov(y_i, y_j)`, the first-order unconditional covariance in levels --
Dynare's `osr` with `optim_weights`. `weights` is `{"y": 1, ("y", "pi"): 0.1}`
(a pair adds `w * Cov` ONCE, as Dynare does) or a matrix over `model.names`. The
steady state is recomputed at each trial by `steady_fn(params)` or from
`guess`/`steady`. A trial without a determinate solution costs `penalty`. BFGS
then a Nelder-Mead polish by default; `bounds={name: (lo, hi)}` switches to
L-BFGS-B and warns when a coefficient sits on a bound.

`OSRResult`: `params` (the optimal coefficients), `all_params`, `loss`,
`initial_loss`, `success`, `message`, `evaluations`, `solution`.

### `PrivateBlock`

The private sector before a policy closes it: sympy `expressions` in the naming of
`parse_system`, `states`, `jumps`, `shocks`, `parameters`, `labels`, and the
optional `instruments`, `objective`, `discount`. `ModFile.block` is one.

---

## Welfare

Requires `sympy`.

### `welfare(model, params, *, steady, cov, objective=None, discount=None, initial=None, name="WELFARE", qz_criterium=None) -> Welfare`

Appends the jump `WELFARE = U + beta * WELFARE(+1)`, solves to second order, and
returns `unconditional` (the ergodic mean, closed form), `conditional` (the
second-order rule at `initial`, the steady state by default), `deterministic`
(`U(steady)/(1-beta)`) and, for a Ramsey model, `conditional_zero_multipliers`
(lagged multipliers at zero: the planner who commits at `t = 0`). A Ramsey model
supplies its own objective and discount. These are the numbers of Dynare's
`evaluate_planner_objective` at `order=2`.

### `welfare_model(model, objective=None, discount=None, *, name="WELFARE") -> Model`

The augmented model, for using the welfare variable in IRFs or simulations.

### `consumption_equivalent(objective, variable, target, reference, *, model, params, steady, cov, discount, bracket=(-0.5, 0.5)) -> float`

The permanent proportional change `lambda` in `variable`, applied inside the
objective, that brings unconditional welfare from `reference` to `target`. No log
utility and no variable called consumption is assumed.

---

## State space and Kalman filter

### `state_space(sol, observables, *, transforms=None, measurement=None, cov=None) -> StateSpace`

`k_{t+1} = T k_t + R eps_{t+1}`, `y_t = d + Z k_t + e_t` from a first-order solution.
Each observable is a `"level"` (default) or a `"log"` (Dynare's `loglinear` with
`logdata`: `d` is the log of the steady state; exact at first order in this
package's proportional coordinates). `measurement` gives the standard deviations of
the measurement errors (`{name: sd}`, a scalar, or a covariance). Observation
equations are model equations: an observed growth rate or a measured inflation rate
is a variable of the model, as in a `.mod`. Missing data are `nan`.

### `kalman_filter(space, data, *, init="auto", univariate=None) -> FilterResult`

Log likelihood (`loglik`, and `loglik_t` per date), predicted and filtered states and
variances. `init="auto"` starts stationary roots at their unconditional distribution
and unit roots DIFFUSE, by the exact initialisation of Koopman and Durbin (2000) with
the unit-root block first in an ordered Schur form, as Dynare's `diffuse_filter`;
`"stationary"` and `"diffuse"` force one or the other. The diffuse start uses the
univariate (observation by observation) filter; so does missing data.

### `kalman_smoother(space, data=None, *, filtered=None, init="auto") -> SmootherResult`

Fixed-interval smoother (Durbin and Koopman, 2012), exact through the diffuse
periods: `states`, `shocks` (row `t` is the innovation INTO date `t`), `measurement`,
`state_variance` (nan in the diffuse periods); `variables()` in levels.

### `shock_decomposition(smoothed, *, variables=None, initial_shock_states=None) -> dict`

Each variable's smoothed path as the contribution of each shock plus the initial
condition, Dynare's definition. `initial_shock_states` (for a `.mod`:
`ModFile.shock_states`) counts the first value of a shock-carrying state as the shock's.

### `realtime_shock_decomposition(space, data, *, init="auto", variables=None, initial_shock_states=None, first=2) -> dict`

The decomposition each date would have shown with the data up to it -- Dynare's
`realtime_shock_decomposition` pool, including its conventions (first sample of two
observations; the first period's shock left in the initial condition).

### `forecast(space, data, horizon, *, init="auto", coverage=0.9, variables=None, include_measurement=False, logs=()) -> Forecast`

Point forecast from the last filtered state and Gaussian bands from the forecast-error
variance (state uncertainty and future shocks): `mean`, `std`, `lower`, `upper`.

---

## Estimation

### `prior(shape, mean, sd, p3=None, p4=None) -> Prior`

Dynare's priors by mean and standard deviation: `beta` (on `[p3, p4]`), `gamma`
(shifted by `p3`), `normal`, `inv_gamma` (type 1, for a standard deviation;
`sd = inf` gives `nu = 2`), `inv_gamma2`, `uniform`. `logpdf` uses Dynare's
normalising constants; `sample(rng, size)`.

### `EstimationProblem(model, data, params, observables, priors={}, cov=None, measurement=None, transforms=None, steady_fn=None, guess=None, init="auto", qz_criterium=None, first_obs=1, presample=0, free=None, bounds=None)`

What is estimated and against which data. Names follow Dynare's `estimated_params`:
a parameter, `"stderr e"` for a shock, `"stderr y"` for the measurement error on an
observable. `priors` map names to `Prior` or `(shape, mean, sd[, p3, p4])`. Methods:
`log_likelihood(theta)`, `log_prior(theta)`, `log_posterior(theta)` (`-inf` where the
model has no determinate solution), `solve(theta)`, `filter(theta)`,
`smoother(theta)`, `start()` (calibration, or the prior mean). `ModFile.estimation_problem(data)`
builds one from a `.mod`'s `estimated_params`, `varobs` and `estimation(...)` options;
with no initial value Dynare starts from the prior mean, and so does this.

### `posterior_mode(problem, x0=None, *, method=None, options=None) -> ModeResult`

BFGS and a simplex polish in an unbounded reparametrisation of each prior's support.
`ModeResult`: `values`, `log_posterior`, `log_likelihood`, `hessian` (of minus the log
posterior, central differences), `covariance`, `std`, `laplace` (log marginal density).

### `maximum_likelihood(problem, x0=None, ...) -> ModeResult`

The same on the likelihood; standard errors from the inverse Hessian.

### `metropolis_hastings(problem, mode, *, draws=10000, chains=2, scale=0.8, init_scale=None, burn=0.5, seed=0) -> MCMCResult`

Random-walk Metropolis-Hastings with proposal `N(0, scale^2 * mode.covariance)`
(Dynare's `mh_jscale`), chains started over-dispersed (`mh_init_scale`), `burn`
dropped (`mh_drop`). `MCMCResult`: `draws`, `mean()`, `quantiles()`, `hpd(mass)`,
`rhat()` (split R-hat), `ess()`, `log_marginal_mhm()` (Geweke's modified harmonic
mean, as Dynare's).

### `smc(problem, *, particles=1000, stages=50, bend=2.0, mutation_steps=2, seed=0) -> SMCResult`

Likelihood-tempered SMC (Herbst and Schorfheide, 2014), from the prior; its
`log_marginal` needs no mode and no harmonic mean.

### `prior_predictive(problem, draws=1000, *, seed=0, statistics=None) -> dict`

Shares of prior draws that are determinate, fail Blanchard-Kahn, or have no steady
state; `statistics(sol)` collected over the solvable draws.

### `model_probabilities(log_marginals, prior=None) -> dict`

Posterior model probabilities from log marginal likelihoods.

### `posterior_forecast(problem, draws, horizon, *, variables=None, n_draws=200, shocks=True, coverage=0.9, logs=(), seed=0) -> dict`

Forecast bands with parameter uncertainty (from MCMC or SMC draws) and shock
uncertainty: `mean`, `lower`, `upper`, `paths`.

`estimate`, `log_likelihood` and `log_posterior` (the 0.2 interface) remain.

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
| `diagnose(model, params, point, *, ctx=None, tol=1e-9)` | Dynare's `model_diagnostics`: variables absent at the current date, the rank of the static Jacobian with the collinear variables and equations of each relation, redundant equations of the dynamic Jacobian, a unit root, non-finite derivatives; `ok` and `messages` |

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
