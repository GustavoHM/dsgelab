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

`derivatives` is `"third"`, `"both"`, `"jacobian"` or `"none"`. The default
`"both"` builds the Jacobian and Hessian. `"third"` adds the exact third-derivative
tensor. `steady_state_model` is a
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
| `moments(cov=None, lags=1, *, hp_filter=None, bandpass=None, grid=512)` | a dict with `names`, `std`, `corr`, `cov`, `autocorr` (`{lag: {name: value}}`), `autocov` (`{lag: E[x_t x_{t-lag}']}`) and `std_by_name`, computed from the discrete Lyapunov equation. With `hp_filter=lambda` or `bandpass=(shortest, longest)`, the moments of the Hodrick–Prescott cycle or of the ideal band-pass filter, as Dynare's `stoch_simul` computes them: the spectral density on `grid` frequencies, times the squared gain of the filter, then the inverse Fourier transform. Both filters remove the zero frequency, so these moments exist with a unit root |
| `covariance(cov=None, *, hp_filter=None, bandpass=None, grid=512)` | unconditional covariance of `[k; u]`, filtered or not |
| `variance_decomposition(cov=None, horizon=None, *, hp_filter=None, bandpass=None, grid=512)` | unconditional, or the forecast-error variance decomposition at a horizon; with a filter, of the filtered variance. Each shock counts with its own variance, so with correlated shocks the shares differ from Dynare's, which orthogonalises them by Cholesky |
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

### `solution_jacobian(first, names, *, step=1e-3, levels=4, qz_criterium=None) -> SolutionJacobian`

Differentiates `P`, `F`, `Q` and the steady state. Purely backward models use
implicit differentiation: `dP = A^-1(dB-dA P)` and `dQ = A^-1(dC-dA Q)`.
General QZ solutions differentiate the implicit saddle-path conditions
`A[P; FP] = B[I; F]`, so perturbed parameters require a new steady state and
linearization but no new QZ decomposition. Central parameter differences use
Richardson extrapolation at halving step sizes, retaining the best triple.
`SolutionJacobian` contains `names`, `P`, `F`, `Q`, `steady` and `error`. The
arrays have a trailing parameter axis. `error` contains a second estimate as a
diagnostic.

### `second_order_jacobian(second, names, *, step=1e-3, levels=4) -> SecondOrderSolutionJacobian`

Differentiates the full second-order decision rule with the same central-difference
and Richardson procedure. `names` contains model parameters, shock standard
deviations (`"stderr e"`) and shock correlations (`"corr e1, e2"`). The result
contains `g_k`, `g_u`, `g_kk`, `g_uu`, `g_ku`, `g_s2`, `steady` and `error`, all
with a trailing parameter axis. The composition includes the predetermined-state
timing used by Dynare.

### `third_order_jacobian(third, names, *, step=1e-3, levels=4, sparse=False, direct_limit=6000) -> ThirdOrderSolutionJacobian`

Differentiates `g_kkk`, `g_kku`, `g_kuu`, `g_uuu`, `g_s2` and the steady state
with central differences and Richardson extrapolation. `names` contains model
parameters, shock standard deviations and correlations. `sparse` and
`direct_limit` select the storage and Sylvester route at every perturbed point.
The cubic composition
uses the same endogenous-state timing as the Dynare comparison. The risk-state
and risk-innovation terms `ghxss` and `ghuss` are not included.

### `log_likelihood_gradient(problem, theta) -> LikelihoodGradient`

Differentiates a first-order `EstimationProblem` likelihood through the Kalman
recursion. Model parameters use `solution_jacobian`; shock and measurement-error
standard deviations and correlations use direct formulas. `level` and `log`
observation transforms and `presample` are supported. Missing observations and
diffuse starts use the univariate recursion; the stable Schur block supplies
initial derivatives. Nonlinear filters raise `NotImplementedError`.
This is a recursive gradient, not automatic differentiation of the model solver.

`LikelihoodGradient` contains `loglik`, `loglik_t`, `gradient` and `gradient_t`.
The lower-level `kalman_likelihood_gradient(space, data, derivatives)` accepts a
mapping from names to dictionaries with `T`, `R`, `Sigma`, `Z`, `d` and `H`.

### `log_posterior_gradient(problem, theta) -> PosteriorGradient`

Adds the prior score to `log_likelihood_gradient`. `PosteriorGradient` contains
`logposterior`, `gradient`, `loglik` and `logprior`. An invalid support point
returns `logposterior = -inf` and a NaN gradient.

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
| `g_kk`, `h_kk` | quadratic state terms, `(n_u, n_k, n_k)` and `(n_k, n_k, n_k)` |
| `h_uu`, `h_ku` | quadratic innovation and state-innovation terms of the state rule |
| `g_ss`, `h_ss` | risk coefficients; the reduced rules add half of each |
| `h_s2` | the full state risk constant, `h_ss - h_uu:Sigma` |
| `sylvester_residual`, `innovation_residual` | residuals of the Sylvester and ex-post innovation solves |
| `decision_rule()` | a `DecisionRule` for `[k'; u']`, with `g_k`, `g_u`, `g_kk`, `g_uu`, `g_ku` and `g_s2 = [h_s2; g_ss]` in Dynare's coefficient convention |
| `jumps(k, risk=True)` / `next_states(k, eps=None, risk=True)` | the reduced decision rules used in simulation |
| `irf(name, size=0.01, periods=40, *, n_draws=0, seed=0, state=None, percent=False)` | conditional impulse response of the pruned reduced rule; the innovation enters in period 0 |
| `simulate(periods=1000, cov=None, *, seed=None, burn=200, pruning=True, levels=False)` | pruned by default |
| `mean_correction(cov=None)` / `ergodic_mean(cov=None, levels=True)` | closed-form ergodic mean |
| `at_covariance(cov)` | this solution if `cov` is the covariance it was computed at, else a solution computed at `cov`; `simulate`, `mean_correction` and `ergodic_mean` go through it, because `g_ss` and `h_ss` depend on the covariance |
| `accuracy(sizes=(0.02, 0.01, 0.005), seed=0)` | implied order of accuracy, about 3 for a correct solution |

### `second_derivatives(model, ss, params, *, ctx="inherit", h=1e-4, row_scale=None, at=None, sparse=False) -> ndarray | SparseSecondDerivatives`

The `(n_eq, m, m)` Hessian with `m = 2 n_vars + n_shocks`, in rescaled
coordinates. Uses `model.hessian_fn` when available. `at` is the point
`[ŷ_{t+1}; ŷ_t; eps]`, in rescaled coordinates, at which it is evaluated (the
steady state by default). With `sparse=True`, returns coordinate entries; the
numerical fallback still forms the dense Hessian first.

---

## Third order

### `solve_third_order(second, cov=None, *, h=1e-3, h_hess=1e-4, direct_limit=6000, third=None, hessian=None, sparse=False) -> ThirdOrderSolution`

Adds to a second-order solution the cubic terms `g_kkk`, `h_kkk`, `h_uuu`, `h_kuu` and `h_kku`,
and the risk slope `g_ssk` and `h_ssk`, the coefficients on `σ² k`. The reduced simulation rules become

```
u_t     = F k + ½ g_kk[k,k] + ½ g_ss + ⅙ g_kkk[k,k,k] + ½ g_ssk k
k_{t+1} = P k + ½ h_kk[k,k] + ½ h_ss + ⅙ h_kkk[k,k,k] + ½ h_ssk k + Q eps
```

`cov` defaults to the covariance of `second`. `h` and `h_hess` are the outer and
inner steps of `third_derivatives`. `third` and `hessian` reuse precomputed third
and second derivatives without recomputing them.
With `sparse=True`, the model derivatives and the cubic state coefficients
`g_kkk` and `h_kkk` stay in coordinate form. The coefficients store one entry
per equation and independent symmetric state triple. Contractions used by the
solver, risk terms, simulation and stochastic steady state do not densify them;
`to_dense()` and explicit `decision_rule()` composition do. Above `direct_limit`,
matrix-free LGMRES solves on the `n_k(n_k+1)(n_k+2)/6` independent symmetric
state triples. Its known term and residual use the same canonical coordinates,
and each matvec expands one equation at a time rather than the whole tensor.
The risk-state blocks `g_xss` and `h_xss` are converted to Dynare's `ghxss`,
and `g_uss` and `h_uss` form `ghuss`. All four agree with Dynare in validation.
Simulation and IRFs use the full pruned rule, including the risk and cubic innovation terms.
`stochastic_steady_state(third)` solves the fixed point of the full unpruned rule
with `sigma=1`, starting from the deterministic steady state by default.

### `ThirdOrderSolution`

| Member | Meaning |
|---|---|
| `second` | the `SecondOrderSolution` it extends |
| `g_kkk`, `h_kkk` | cubic state terms, `(n_u, n_k, n_k, n_k)` and `(n_k, n_k, n_k, n_k)` |
| `h_uuu` | cubic innovation term, `(n_k, n_eps, n_eps, n_eps)` |
| `h_kuu`, `h_kku` | cubic mixed terms, `(n_k, n_k, n_eps, n_eps)` and `(n_k, n_k, n_k, n_eps)` |
| `g_ssk`, `h_ssk` | risk slope, `(n_u, n_k)` and `(n_k, n_k)` |
| `g_xss`, `h_xss` | risk-state blocks, `(n_u, n_k)` and `(n_k, n_k)` |
| `g_uss`, `h_uss` | risk-innovation blocks, `(n_u, n_eps)` and `(n_k, n_eps)` |
| `sylvester_residual`, `risk_slope_residual`, `risk_state_residual`, `risk_shock_residual`, `innovation_residual` | residuals of the linear solves |
| `decision_rule()` | the second-order `DecisionRule`, plus `g_uuu`, `g_kuu`, `g_kku` and `g_kkk` |
| `jumps(k, eps=None, risk=True)` / `next_states(k, eps=None, risk=True)` | the full unpruned decision rules, including innovation curvature |
| `irf(name, size=0.01, periods=40, *, n_draws=0, seed=0, state=None, percent=False, drop=100)` | conditional response of the full pruned rule, or a generalized response over `n_draws` stochastic paths with the impulse after `drop` |
| `simulate(periods=1000, cov=None, *, seed=None, burn=200, levels=False)` | simulation pruned in three layers (Andreasen, Fernández-Villaverde and Rubio-Ramírez) |
| `accuracy(sizes=(0.02, 0.01, 0.005), seed=0)` | implied order of the cubic block with the risk terms switched off, about 4 for a correct solution |

### `third_derivatives(model, ss, params, *, ctx="inherit", h=1e-3, h_hess=1e-4, row_scale=None, sparse=False) -> ndarray | SparseThirdDerivatives`

The `(n_eq, m, m, m)` tensor of third derivatives with `m = 2 n_vars + n_shocks`,
in rescaled coordinates. Uses `model.third_fn` directly when available. Otherwise,
it is a central difference, with step `h`, of the Hessian, which costs `2m`
Hessian evaluations. The Hessian is the model's exact one when it has
`hessian_fn`, and a second difference with step `h_hess` otherwise. With
`sparse=True`, returns coordinate entries; the numerical fallback still forms
the dense tensor before extracting them.

### `SparseSecondDerivatives`

Coordinate storage for Hessians with `equations`, `indices`, `values`, `shape` and `nnz`.
`to_dense()` returns the full tensor. `contract(u, v)` evaluates
`H[e,a,b] u[a,i] v[b,j]` without densifying `H`. `scaled(scale, row_scale=None)`
applies the coordinate and equation scales. `from_dense(array)` converts an
existing Hessian.

### `SparseThirdDerivatives`

Coordinate storage with `equations`, `indices`, `values`, `shape` and `nnz`.
`to_dense()` returns the full tensor. `contract(u, v, w)` evaluates
`T[e,a,b,c] u[a,i] v[b,j] w[c,k]` without densifying `T`. `scaled(scale, row_scale=None)`
applies the coordinate and equation scales. `from_dense(array)` converts an
existing tensor.

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

### `tauchen(rho, sigma, n, *, mean=0.0, n_std=3.0) -> MarkovChain`

Tauchen's (1986) chain for the AR(1) `y' = (1 - rho) mean + rho y + sigma e`, `e ~ N(0, 1)`.
The `n` nodes are evenly spaced over `n_std` unconditional standard deviations
(`sigma / sqrt(1 - rho^2)`) on each side of `mean`, and each transition probability is the
normal probability of the interval between the midpoints around the next node, with the tails
going to the end nodes. Each interval is computed from the side of the conditional mean it lies
on, so that small probabilities in either tail keep their relative accuracy. The nodes and the
matrix are those of QuantEcon's `tauchen(n, rho, sigma, mu=(1 - rho) * mean, n_std=n_std)`.
With `rho` close to one and few nodes, the chain approximates the process poorly, and its
probabilities of leaving the end nodes can underflow to zero; `rouwenhorst` is the method for
that case. `rho` must lie strictly between -1 and 1, `sigma` must be positive, and `n` must be an
integer of at least 2.

### `rouwenhorst(rho, sigma, n, *, mean=0.0) -> MarkovChain`

Rouwenhorst's chain for the same AR(1), built by the recursion of Kopecky and Suen (2010) with
`p = q = (1 + rho) / 2`. The `n` nodes are evenly spaced over `sqrt(n - 1)` unconditional
standard deviations on each side of `mean`. The stationary distribution is binomial with `n - 1`
trials of probability one half, and the chain has the mean, the variance and the
autocorrelations of the AR(1) exactly, for every `n` and every `rho`.

### `MarkovChain(nodes, transition)`

A chain on the values `nodes` `(n,)`, with `transition[i, j]` `(n, n)` the probability of moving
from node `i` to node `j`; the rows must be nonnegative and sum to one. `stationary` is the
stationary distribution, computed by GTH elimination (Grassmann, Taksar and Heyman, 1985); a
reducible chain raises `ValueError`. `mean` and `variance` are the moments of the node values under
it, and `autocorrelation(lag=1)` their autocorrelation. `simulate(periods, *, start=None,
seed=None, indices=False)` draws a path of `periods` values, from node `start` or from the
stationary distribution, and returns node indices instead of values with `indices=True`.

### `time_iteration(model, params, *, grid, markov=None, cov=None, n_quad=5, quadrature=None, bounds=None, interpolation="linear", initial=None, guess=None, steady=None, positive=(), tol=1e-8, max_iter=1000, inner_tol=1e-12, inner_max_iter=50) -> TimeIterationSolution`

Time iteration (Coleman, 1990) on the model's own residual, the one the perturbation solves. Each
iteration takes the current policy as the next period's and solves the equilibrium conditions for
today's jumps at every grid point, by Newton's method point by point. It stops when no jump moves
by more than `tol` at any point.

The rows of the residual are sorted by what they contain at `t+1`, which is found by moving one
next-period variable at a time. A row with next-period states and no next-period jump is a
transition and gives the next states from today's states, jumps and shock. Every other row,
including a static one with nothing at `t+1`, is an equilibrium condition, integrated over the
next period and set to zero. The model needs as many transitions as states and as many
equilibrium conditions as jumps; otherwise the call raises `ValueError` and lists the rows. When
the transitions are affine in the next states, as in `k' - g(k, u, eps)`, the next states come
from one evaluation, and otherwise from a Newton solve.

| Argument | Meaning |
|---|---|
| `grid` | a `CartesianGrid` of the continuous states, or the mapping that builds one; every state without a Markov chain needs an axis |
| `markov` | `{state: MarkovChain}`, exogenous states given by chains whose nodes are the state's values in levels. Several chains are independent, and the product chain orders its nodes with the last chain fastest. The transition row of a Markov state (the row with that state at `t+1` and no endogenous state there) is left out |
| `cov`, `n_quad` | the shocks are integrated by a tensor Gauss–Hermite rule with `n_quad` nodes per shock, scaled by `cov` (`None` is the identity, as elsewhere in the package) |
| `quadrature` | `(nodes, weights)`, one row of shocks per node, in place of Gauss–Hermite. With a Markov chain and neither `cov` nor `quadrature`, the shocks are set to zero |
| `bounds` | `{jump: (lower, upper)}`, each a number, `None`, or `bound(states, params)` with `states` a name-indexed view of today's states (`s.w`, `s["w"]`) |
| `interpolation` | `"linear"`, or `"cubic"` for a natural cubic spline along each axis. Both extrapolate linearly beyond the grid |
| `initial` | the starting policy: a `TimeIterationSolution` (to refine on another grid), `initial(states, params)` returning the jumps in model order, or `{jump: number or function}`. With `None`, the first-order perturbation, which needs `guess` or `steady` |
| `tol`, `max_iter` | the stopping rule of the iteration, on the largest change of a jump |
| `inner_tol`, `inner_max_iter` | the Newton solve at each grid point, on the largest residual |

A bounded jump is paired with the equilibrium condition in its position, so that the `j`-th
equilibrium condition, in the order of the equations, goes with the `j`-th jump. The pair
satisfies the complementarity in the standard convention, which dolo also uses for its arbitrage
equations. The condition is zero between the bounds, nonnegative at the lower bound and
nonpositive at the upper one. A borrowing limit `c <= w` therefore goes with the Euler equation written as
`beta R E[(c'/c)^(-gamma)] - 1`, and a zero lower bound `i >= 0` with the policy rule written as
`i - rule`. The pair is solved through the Fischer–Burmeister function in its box form, and the
Newton steps keep a bounded jump within its bounds.

The equations are first called with every grid point at once, each variable a vector. If they fail
or disagree with a call at a single point (because they use `math.log` or `float()`, say), the
residual is evaluated point by point. The policy is the same either way, but the point-by-point
evaluation is slower.

### `TimeIterationSolution`

`controls(states)`, also `solution(states)`, gives the jumps at one state or at a row-stacked
array of states, in levels and in the model's state order, or at a mapping from state names to
values; a Markov state must take one of its chain's node values. `euler_errors(states=None)` gives
the equilibrium conditions, integrated, at `states`, by default at the centres of the grid's cells
and at every node of the chain. For a bounded jump the error is the natural residual
`mid(u - lower, F, u - upper)`, which is the condition itself between the bounds and only its
wrong-signed part at a bound. The errors are in the units the condition is written in.
`simulate(periods, *, start=None, seed=None, shocks=None)` returns `states`, `jumps` and `nodes`
(the chain's index), one row per period, with shocks drawn from the covariance or the quadrature
the solution was built with, or given in `shocks`. Between grid points the interpolated policy is
not held within the bounds. The policy at the grid points is in `values`
`(n_nodes, grid.n, n_jumps)`, and the bounds there in `lower` and `upper`. The other fields are
`grid`, `chain` (the product chain if there are several), `markov_states`, `interpolation`,
`iterations`, `converged`, `distance` (the last change) and `history` (every change).

### `CartesianGrid(axes)`

A tensor grid: `axes` maps each name to an increasing vector of nodes, or to `(low, high, n)` for
`n` evenly spaced ones. `points` lists every node, one row each, with the last axis varying
fastest; `midpoints` lists the centres of the cells; `shape`, `n` and `d` give the size.
`interpolate(values, points, kind="linear")` interpolates `values`, one row per grid point, at
`points`. The cubic interpolant is the natural spline along each axis, the same as SciPy's
`CubicSpline(bc_type="natural")` in one dimension, extended linearly beyond the ends. With a
policy whose second derivative at the ends of the grid is not zero, its error there falls as the
square of the spacing.

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
| `irf_calibration`, `moment_calibration`, `relative_irf` | the calibration blocks, as restriction tuples for `check_restrictions` |

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
| `identification(point="prior_mean", **kwargs)` | `identification` of the `estimated_params`, observing `varobs`, at the prior mean (Dynare's default), the calibration or a given point |
| `check_restrictions(point="calibration", *, qz_criterium=None, **kwargs)` | `check_restrictions` of the `irf_calibration` and `moment_calibration` blocks at the calibration, the prior mean or a given point |
| `sample_restrictions(draws=1000, **kwargs)` | `sample_restrictions` of the same blocks over the priors of `estimated_params`; `points=` and `names=` evaluate given draws |

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

### `state_space(sol, observables, *, transforms=None, measurement=None, cov=None, aggregation=None) -> StateSpace`

`k_{t+1} = T k_t + R eps_{t+1}`, `y_t = d + Z k_t + e_t` from a first-order
solution. Each observable is a `"level"` (default) or a `"log"`. A log observable
reproduces Dynare's `loglinear` with `logdata`; its constant `d` is the log of
the steady state, and the mapping is exact at first order in this package's
proportional coordinates. `measurement` gives the standard deviations of the
measurement errors (`{name: sd}`, a scalar, or a covariance). An observed growth
rate or a measured inflation rate is a variable of the model, defined by a model
equation, as in a `.mod`. Missing data are `nan`.

`aggregation` declares observables seen at a lower frequency than the model's,
`{name: (how, every)}`: the data are the `"mean"` or the `"sum"` of the variable
over the last `every` periods, in the units of its transform (a mean of logs for a
log observable; a sum of logs is refused), with `nan` in the periods between
observations. The state then stacks `every - 1` lags of the model's states, named
`"k(-1)"` and so on, as an auxiliary variable `(y + y(-1) + y(-2) + y(-3))/4` does in
a `.mod`. The filter, the smoother, the likelihood and its gradient, the shock
decompositions and the forecasts work on the stacked state, and the smoothed
variables are the model's. A stock observed at the end of each low-frequency
period needs no aggregation, only the missing values. `EstimationProblem` takes the
same `aggregation=`, at first order. Validated against Dynare on `fs2000.mod` with
output growth observed only as its annual mean, through the auxiliary variable a
Dynare user writes: the likelihood at two points within 8e-13, the smoothed
variables within 1.3e-14, and a posterior mode with a kernel at least Dynare's.

`StateSpace`: `T`, `R`, `Sigma`, `Z`, `d`, `H` (the measurement covariance),
`observables`, `states`, `shocks`, `transforms`, `solution`, `aggregation` and
`n_model_states` (the model's states, before the lags).

### `low_frequency(values, every, *, periods=None, offset=None) -> ndarray`

Places low-frequency observations on the model's grid, with `nan` in between:
`values[j]` lands in period `offset + j * every`, by default at the last period of
each low-frequency period (`offset = every - 1`), where an aggregate over the last
`every` periods is complete.

```python
import numpy as np
import dsgelab as dsg

ar = dsg.from_equations(["a(+1) = rho*a + e", "y = 1 + a"], states=["a"], jumps=["y"],
                        shocks=["e"], parameters=["rho"])
sol = dsg.solve(ar, dict(rho=0.8), guess=dict(a=0, y=1))
y = sol.to_levels(sol.simulate(periods=400, cov={"e": 0.1}, seed=1))["y"]
annual = dsg.low_frequency(y.reshape(-1, 4).mean(axis=1), 4)
pb = dsg.EstimationProblem(model=ar, data={"y": annual}, params=dict(rho=0.5), observables=["y"],
                           priors={"rho": ("beta", 0.5, 0.2), "stderr e": ("inv_gamma", 0.1, float("inf"))},
                           guess=dict(a=0, y=1), aggregation={"y": ("mean", 4)})
print(dsg.posterior_mode(pb).values)
```

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

### `particle_filter(solution, data, *, observables, transforms=None, measurement=None, cov=None, particles=1000, seed=0, resample=0.5, store=False) -> ParticleFilterResult`

Bootstrap particle filter at first, pruned second or pruned third order. Second-order particles
use the augmented state `[xf; xs; kron(xf,xf)]`; third-order particles use `[xf; xs; xt]`,
with observations evaluated before the state transition as `d + C z_{t-1} + D inov_t`.
Third order uses the full pruned rule, including `ghxss` and `ghuss`, for both the
transition and the observation, and starts each particle from its own 1,000-period path.
`measurement` is required and must have positive-definite covariance. `cov` sets the structural covariance at
first order; second order uses `solution.Sigma`. `resample` is the effective-sample-
size threshold as a fraction of `particles`, or `None` to disable systematic
resampling. Missing observations are skipped and contribute zero to the likelihood.
The result contains `loglik`, `loglik_t`, posterior `states`, final `weights`,
per-date `ess` and the particle count. With `store=True`, it also contains the
augmented state, weight and ancestry history required by `particle_smoother`.
`states[t]` refers to the state at the date of observation `t` at every order.

### `particle_smoother(filtered) -> ParticleSmootherResult`

Smoothing from `particle_filter(..., store=True)`, returning state means, effective
sample sizes and `method`. Full-rank first-order transitions use the Gaussian
backward recursion with filtered weights and predictive normalization
(`method="backward"`). Nonlinear and rank-deficient transitions use the stored
particle ancestry (`method="ancestral"`), preserving deterministic state relations
and the timing of observations and shocks. Ancestral smoothing can lose effective
sample size over long histories. The terminal mean and ESS use the terminal
filtered weights.

### `inversion_filter(solution, data, *, observables, transforms=None, cov=None, presample=0, tol=1e-9) -> InversionFilterResult`

Minimum-norm inversion in standardized shock coordinates, at first or pruned
second order, with the observation-Jacobian correction. Measurement error is not
supported, and there must be at least as many shocks as observables. First order
starts at zero and gives the exact Gaussian likelihood, propagating uncertainty
from unobserved shocks and missing observations. Second order starts at the pruned
system's mean and follows one local inverse. When observations are missing or fewer
than shocks, it uses a local Gaussian approximation to the unresolved uncertainty
and marginal density. The result flags this case with `approximate=True`.

The result also contains `loglik`, `loglik_t`, recovered `states` and `shocks`, and
the observation `jacobians`. Missing Jacobian rows are zero. A fully missing date
contributes zero to the likelihood while its state uncertainty propagates forward.

### `inversion_smoother(solution, data, *, observables, transforms=None, cov=None, presample=0, tol=1e-9) -> InversionFilterResult`

Returns the inversion filter's paths and approximation flag without a backward
pass. When observations leave shocks unidentified, these are filtered paths and
do not condition on later observations.

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
`lower`, `upper`. `logpdf(x)` uses Dynare's normalising constants,
`score(x)` differentiates it, and `sample(rng, size=None)` draws from it.

### `EstimationProblem(model, data, params, observables, priors={}, cov=None, measurement=None, transforms=None, steady_fn=None, guess=None, init="auto", qz_criterium=None, first_obs=1, presample=0, free=None, bounds=None, order=1, nonlinear_filter="particle", particles=1000, particle_seed=0, resample=0.5)`

What is estimated and against which data. Names follow Dynare's
`estimated_params`, with a parameter's own name, `"stderr e"` for the standard
deviation of a shock and `"stderr y"` for that of the measurement error on an
observable. `priors` map names to `Prior` or `(shape, mean, sd[, p3, p4])`.
With `order=2` or `order=3`, `nonlinear_filter="particle"` uses the bootstrap
filter with `particles`, `particle_seed` for common random draws and `resample` as
its ESS threshold. Third order requires the particle route.
`nonlinear_filter="inversion"` uses the exact shock inversion and does not allow
measurement error. Particle and inversion smoothing are available through `smoother`.

| Method | Meaning |
|---|---|
| `log_likelihood(theta)`, `log_prior(theta)` | the two parts of the posterior |
| `log_posterior(theta)` | `-inf` where the model has no determinate solution |
| `solve(theta)` | `(FirstOrderSolution, StateSpace)` at `theta` |
| `filter(theta)` | a `FilterResult`, `ParticleFilterResult` or `InversionFilterResult` |
| `smoother(theta)` | a `SmootherResult`, `ParticleSmootherResult` or `InversionFilterResult` |
| `start()` | the calibration, or the prior mean |

`ModFile.estimation_problem(data, **kwargs)` builds one from a `.mod`'s
`estimated_params`, `varobs` and `estimation(...)` options. With no initial value
Dynare starts from the prior mean, and so does `EstimationProblem`.

### `posterior_mode(problem, x0=None, *, method=None, options=None) -> ModeResult`

BFGS with `log_posterior_gradient` and a simplex polish in an unbounded
reparametrisation of each prior's support. `maximum_likelihood` uses
`log_likelihood_gradient` in the same way.
`ModeResult`: `values`, `log_posterior`, `log_likelihood`, `hessian` (of minus the log
posterior, central differences), `covariance`, `std`, `laplace` (log marginal density).

### `maximum_likelihood(problem, x0=None, *, method=None, options=None) -> ModeResult`

The same on the likelihood, with standard errors from the inverse Hessian.

### `metropolis_hastings(problem, mode, *, draws=10000, chains=2, scale=0.8, init_scale=None, burn=0.5, seed=0) -> MCMCResult`

Random-walk Metropolis–Hastings with proposal `N(0, scale^2 * mode.covariance)`
(Dynare's `mh_jscale`), chains started over-dispersed (`mh_init_scale`), and the
first share `burn` of each chain dropped (`mh_drop`).

### `hmc(problem, mode=None, *, draws=1000, chains=2, step_size=0.05, leapfrog=10, burn=0.5, target=0.8, adapt=True, init_scale=1.0, seed=0) -> MCMCResult`

Hamiltonian Monte Carlo for problems supported by `log_posterior_gradient`.
The mode's inverse covariance is the mass matrix when available. Leapfrog
integration and a Metropolis correction preserve the target density. With
`adapt=True`, `step_size` is a starting value: a heuristic finds a step with
acceptance near one half, and dual averaging (Hoffman and Gelman, 2014) moves it
toward `target` during `burn`, after which it stays fixed. Each trajectory
multiplies the step by a uniform draw on [0.8, 1.2]. `acceptance` is the mean
acceptance probability after burn-in. `init_scale` sets initial overdispersion.

### `nuts(problem, mode=None, *, draws=1000, chains=2, step_size=0.05, max_depth=8, burn=0.5, target=0.8, adapt=True, init_scale=1.0, seed=0) -> NUTSResult`

No-U-Turn sampling (Hoffman and Gelman, 2014; Betancourt, 2017). Each draw
doubles a trajectory in a random direction until the generalised U-turn
criterion holds or `max_depth` doublings are reached, and picks a point from it
with multinomial weights. The step adapts as in `hmc`. Points that cannot be
solved end the trajectory. `NUTSResult` extends `MCMCResult` with `divergences`,
`max_depth_hit` and `mean_depth`, all reported per chain after burn-in. Energy
errors above 1000 count as divergences.

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

### `forecast_evaluation(problem, origins, horizon, *, estimate="mode", theta=None, window=None, coverage=0.9, include_measurement=True, options=None) -> ForecastEvaluation`

Out-of-sample forecasts, as Dynare's recursive estimation makes them with
`nobs=[T1:T2]` and `forecast`. For each origin `n`, the sample is the first `n`
rows of the problem's data (after `first_obs`), or the last `window` rows before
the origin with `window`. On it the parameters are estimated, the sample is
filtered, and the observables are forecast `1..horizon` periods ahead. `estimate`
is `"mode"` (the posterior mode), `"ml"` (maximum likelihood), `"fixed"` (`theta`,
one vector or one row per origin) or a callable `estimate(problem_at_origin, x0)`
returning the parameters or a `ModeResult`. Each search starts from the previous
origin's estimate, and the first from `theta` or the problem's starting values.
The problem must be at first order.

The predictive density is the Kalman filter's Gaussian forecast at the estimate,
with mean `d + Z a_{T+h|T}` and variance `Z V_h Z' + H` (without `H` when
`include_measurement=False`). It is conditional on the point estimate, as
Dynare's forecast after the mode is; `posterior_forecast` adds parameter
uncertainty at one origin.

`ForecastEvaluation` has `variables` (the observables), `origins`, `horizon`, and
per observable `mean`, `std` and `actual` (arrays of origins by horizons, in the
units of the data, with `actual` NaN past the end of the sample) and `last` (the
observation at each origin). `covariance` holds the joint predictive covariances,
`theta` the parameters used at each origin (in the order of `names`), and, when
they were estimated, `modes` the `ModeResult` of each origin and `objective` the
posterior kernel or likelihood there. Pairs with a missing outcome are left out of
every statistic:

| Method | Returns |
|---|---|
| `errors`, `random_walk_errors` | `actual - mean`, and `actual - last` |
| `rmse(benchmark=False)` | RMSE by horizon, of the model or of the random walk |
| `relative_rmse()` | the model's RMSE over the random walk's |
| `diebold_mariano(variable, horizon, *, benchmark="random_walk", power=2, alternative="two.sided", varestimator="acf")` | the test of the `horizon`-step forecasts against the random walk or given errors |
| `pit()` | `Phi((actual - mean) / std)`, uniform when the density is right |
| `coverage_rate(coverage=None)` | share of outcomes inside the central band, by horizon |
| `log_score(joint=False)` | log predictive density, per observable or joint |

### `diebold_mariano(e1, e2, horizon=1, *, power=2, alternative="two.sided", varestimator="acf") -> DieboldMariano`

The Diebold–Mariano test of equal accuracy, from two series of forecast errors.
The loss differential is `|e1|^power - |e2|^power`. Its long-run variance sums the
autocovariances up to lag `horizon - 1`, with equal (`"acf"`) or Bartlett
(`"bartlett"`) weights. The statistic carries the Harvey–Leybourne–Newbold
correction and is compared with Student's t with `n - 1` degrees of freedom.
`alternative="less"` is that the first forecast is more accurate. When the
variance is not positive at `horizon > 1`, the test warns and uses horizon 1.
This is R's `forecast::dm.test`. `DieboldMariano` has `statistic`, `p_value`,
`horizon`, `power`, `alternative` and `n`.

Validated against Dynare 7.2's recursive estimation of `fs2000` over nine samples,
at Dynare's modes and with modes re-estimated by `dsgelab`, and against R's
`forecast::dm.test` in 48 cases (see `validation/README.md`).

```python
import dsgelab as dsg

ar = dsg.from_equations(["a(+1) = rho*a + e", "y = 1 + a"], states=["a"], jumps=["y"],
                        shocks=["e"], parameters=["rho"])
sol = dsg.solve(ar, dict(rho=0.8), guess=dict(a=0, y=1))
series = sol.to_levels(sol.simulate(periods=400, cov={"e": 0.1}, seed=1))["y"]
pb = dsg.EstimationProblem(
    model=ar, data={"y": series}, params=dict(rho=0.5), observables=["y"],
    priors={"rho": ("beta", 0.5, 0.2), "stderr e": ("inv_gamma", 0.1, float("inf"))},
    guess=dict(a=0, y=1))

ev = dsg.forecast_evaluation(pb, range(320, 400, 10), 4)
print(ev.relative_rmse()["y"], ev.coverage_rate()["y"])
print(ev.diebold_mariano("y", 1, alternative="less"))
```

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

## Identification

### `identification(model, params, *, free, observables, cov=None, measurement=None, transforms=None, guess=None, steady_fn=None, qz_criterium=None, lags=1, grid=5000, max_group=2, criteria=("moments", "spectrum", "minimal"), tol_rank="auto", tol_deriv=1e-8, tol_sv=1e-3, step=1e-2, levels=8) -> Identification`

Local identification of the parameters in `free` at one point, from the
distribution of the `observables`, as Dynare's `identification` command computes
it. `free` names parameters, shock standard deviations (`"stderr e"`) and
measurement-error standard deviations (`"stderr y"`), as in `EstimationProblem`. A
mapping also sets the point; a sequence takes the values in `params`, `cov` and
`measurement`. The observables must be stationary.

Three criteria are computed, each a rank condition on a matrix whose columns are
the free parameters:

| Criterion | Matrix | Full rank when |
|---|---|---|
| `"moments"` (Iskrev 2010) | derivatives of the means, the covariances and the autocovariances up to `lags` | no direction of the parameters leaves those moments unchanged |
| `"spectrum"` (Qu and Tkachenko 2012) | G, the Gram matrix of the derivatives of the mean and of the spectral density over `grid + 1` frequencies | no direction leaves the mean and the spectrum unchanged |
| `"minimal"` (Komunjer and Ng 2011) | the derivatives of a minimal state space representation, next to the similarity transforms of its state and of its shocks | no direction is absorbed by a change of basis; needs as many observables as shocks and no measurement error, as in Dynare |

The derivatives of the state space matrices are central differences with one
Richardson step. For each parameter the step is chosen among `levels` steps that
halve from `step` times its value, as the one where a second Richardson estimate,
from half the steps, agrees best; the difference between the two estimates
measures the error. With `tol_rank="auto"` a singular value below ten times that
error counts as zero (for G, its square, since G's eigenvalues are squared singular
values). `tol_rank="robust"` is Dynare's default, `max(shape) * eps(norm)`, which
assumes exact derivatives. A model declared in Python is linearised by finite
differences and its derivatives carry an error near `1e-10 / step`, so a robust
tolerance can report full rank for parameters that are not identified. The other
options are Dynare's, with the same names in its manual: `lags` is `ar`, `grid` is
`grid_nbr`, `max_group` is `max_dim_cova_group` (0 skips the search), and `tol_deriv`
and `tol_sv` keep their names.

`Identification`: `names`, `point`, `observables`, `lags`, `moments` (in Dynare's
order) and `moment_labels`, `checks` (also by index, `ident["moments"]`),
`skipped` (criteria not computed, with the reason), `minimal_states`,
`collinearity`, `identified` (every criterion computed has full rank), `summary()`.
`collinearity[k][name]` is `(group, cosine)`: the `k` other parameters whose
moment derivatives best reproduce those of `name`, as Dynare's brute-force search
finds them. The group is shorter than `k` when a larger one adds nothing.

`IdentificationCheck`: `criterion`, `jacobian` (before its rows are normalised),
`rank`, `required`, `identified`, `condition`, `tolerance`, `sensitive` (whether
each parameter moves the criterion at all) and `insensitive`, `dependent` (sets of
linearly dependent parameters, one per direction of the null space),
`multicollinearity` (the cosine between each parameter's column and its projection
on the others), `weak` and `weak_pairs`.

`ModFile.identification(point="prior_mean", **kwargs)` takes `free` from
`estimated_params` and the observables from `varobs`. `point` is `"prior_mean"`
(Dynare's default), `"calibration"` or a mapping.

```python
ar = dsg.from_equations(["a(+1) = rho*a + e", "y = 1 + a"], states=["a"], jumps=["y"],
                        shocks=["e"], parameters=["rho"])
ident = dsg.identification(ar, dict(rho=0.8), free=["rho", "stderr e"],
                           observables=["y"], cov={"e": 0.1}, guess=dict(a=0, y=1))
ident.identified          # True
print(ident.summary())
```

---

## Method of moments

### `method_of_moments(model, data, *, params, free, observables, moments, method="GMM", order=1, ...) -> MomentsResult`

Matches means, covariances and autocovariances declared in Dynare's
`matched_moments` syntax. GMM computes them analytically; SMM uses fixed simulation
draws. With `order=2`, both use the pruned second-order ABCD representation
`[xf; xs; kron(xf,xf)]`. GMM solves its covariance by a Lyapunov equation, and SMM
simulates it with the same draws at every parameter point. First order remains the
default. The importer reads `order` from the `.mod` options.

`MomentsResult` contains estimates, standard errors, the objective, matched moments,
weighting stages and Hansen's J test. `ModFile.method_of_moments` takes parameters
from `estimated_params`, moments from `matched_moments` and observables from `varobs`.

## Calibration restrictions

### `check_restrictions(solution, *, irf=(), moments=(), cov=None, relative_irf=False) -> RestrictionCheck`

Checks sign and interval restrictions on impulse responses and theoretical moments,
as Dynare's `irf_calibration` and `moment_calibration` blocks state them. An IRF
restriction is `(variable, shock, horizons, bound)`, with the impact period numbered
1. A moment restriction is `(variable1, variable2, lags, bound)`. At lag 0 it is the
variance when the two variables are the same and their correlation otherwise. At lag
`l` it is the correlation of `variable1` at `t` with `variable2` at `t + l`. Horizons
and lags are an integer, a sequence or a `"first:last"` string. A bound is `"+"`,
`"-"` or `(lower, upper)`, with strict inequalities.

The conventions are Dynare's. Values are absolute deviations from the steady state.
The impulse is one standard deviation of the shock, taken from the diagonal of its
covariance without a Cholesky factor, or 100 times a unit innovation with
`relative_irf=True`. A restriction over several horizons or lags holds when any one
of them is inside the bound, the logical OR of Dynare's manual. The list holds when
every restriction holds. To require every horizon, write one restriction per horizon.

`RestrictionCheck` has `restrictions` (each a `Restriction`, with `label`, `bound`
and `periods`), `values` (one array per restriction, over its periods), `inside`
(the same shape, booleans), `holds_each`, `holds`, `failed` (the labels of the
restrictions that fail) and `summary()`.

### `sample_restrictions(model, params, priors=None, *, irf=(), moments=(), cov=None, draws=1000, seed=0, points=None, names=None, relative_irf=False, guess=None, steady_fn=None, qz_criterium=None) -> RestrictionSample`

Evaluates the same restrictions over draws from the prior. This is the
`map_calibration` step of Dynare's sensitivity toolbox. `priors` takes the form used
by `EstimationProblem`: a parameter name or `"stderr e"` mapped to a `Prior` or to
`(shape, mean, sd[, p3, p4])`. `points` evaluates given rows instead of drawing,
with one column per name in `names`. Draws without a determinate solution keep
their status (`"blanchard_kahn"`, `"no_steady_state"` or `"other"`) and `nan`
values, and they are left out of the shares, as in Dynare. `RestrictionSample` has
`draws`, `status`, `solved`, `values` (an array of draws by periods per restriction),
`inside`, `holds_each` (draws by restrictions), `holds`, `shares`, `share(which=None)`
for the share of solved draws where the restrictions `which` all hold, and
`summary()`.

Validated against Dynare 7.2 on `tests/gsa/ls2003a.mod`. The comparison covers the
values at every horizon and lag, at the calibration and on Dynare's 512 prior draws,
the draws that solve, each restriction's verdict per draw and the shares (see
`validation/README.md`). The functions work at first order, as Dynare's do.

```python
import dsgelab as dsg
from dsgelab.examples import nk

model, params, guess = nk.build()
sol = dsg.solve(model, params, guess=guess)
cov = {"eps_a": 0.01, "eps_v": 0.0025}

irf = [("pi", "eps_v", "1:4", "-"),           # inflation falls in the first year after a monetary shock
       ("x", "eps_v", 1, (-0.01, -0.002))]    # the output gap falls by 0.2 to 1 point on impact
moments = [("x", "pi", 0, "+"),               # output gap and inflation move together
           ("pi", "pi", 1, (0.5, 0.88))]      # inflation is persistent, but not too much

check = dsg.check_restrictions(sol, irf=irf, moments=moments, cov=cov)
print(check.summary())

priors = {"phi_pi": ("gamma", 1.5, 0.5), "kappa": ("gamma", 0.1, 0.05)}
sample = dsg.sample_restrictions(model, params, priors, irf=irf, moments=moments, cov=cov,
                                 guess=guess, draws=500, seed=0)
print(sample.summary())
```

```
Calibration restrictions: 4 of 4 hold
  ok   irf    pi(1:4), eps_v                       -   -0.0006015, -0.0003008, -0.0001504, -7.519e-05
  ok   irf    x(1), eps_v            [-0.01, -0.002]   -0.003038
  ok   moment x, pi                                +   0.8141
  ok   moment pi, pi(1)                  [0.5, 0.88]   0.8754
Calibration restrictions over 500 draws, 429 solved
    share  kind   restriction                      bound
   100.0%  irf    pi(1:4), eps_v                       -
    98.4%  irf    x(1), eps_v            [-0.01, -0.002]
   100.0%  moment x, pi                                +
    51.0%  moment pi, pi(1)                  [0.5, 0.88]
    49.4%  all together
```

The 71 draws left out have no determinate equilibrium. Each of them has `phi_pi`
below 1, where the Taylor principle fails.

---

## Global sensitivity

### `Uniform(low, high)`, `Normal(mean, sd)`, `Triangular(low, mode, high)`

Distributions of a parameter, with `mean`, `sd`, `symmetric`, `draw(rng, n)` and
`ppf(u)`, the interface of `cgelab.sensitivity`'s. A `Prior`, or a prior tuple
`(shape, mean, sd[, p3, p4])`, works wherever a distribution does: `Prior.ppf` is
its quantile function.

### `stability_map(model, params, distributions=None, *, draws=2048, seed=0, sampling="sobol", points=None, names=None, cov=None, guess=None, steady_fn=None, qz_criterium=None, pvalue_ks=0.001) -> StabilityMap`

The stability mapping of Dynare's sensitivity toolbox (`dynare_sensitivity(stab=1)`).
It draws the parameters in `distributions` (a parameter name or `"stderr e"`
mapped to a distribution), on a scrambled Sobol sequence (`sampling="sobol"`, the
quasi-random design Dynare uses by default), a Latin hypercube (`"latin"`) or at
random, solves the model at each draw and records what happened. `points` and
`names` classify given draws instead. `StabilityMap` has `names`, `draws`, `status`
per draw, `shares`, `filtering(kind="determinate", columns=None)` and `summary()`.
A status is `"determinate"` (a unique stable solution), `"indeterminate"` (more
stable roots than predetermined variables, or Dynare's rank failure),
`"explosive"` (fewer), `"no_steady_state"` or `"no_solution"`: the verdicts of
Dynare's solver. `filtering` runs Dynare's three Monte Carlo filtering analyses:
the determinate draws against all the others, and the indeterminate or explosive
draws against the other solved draws. By default it tests every parameter but the
shock standard deviations, which cannot move the roots. `ModFile.stability_map`
draws from the priors of `estimated_params`.

### `monte_carlo_filtering(points, behaviour, names=None, *, nonbehaviour=None, pvalue_ks=0.001, title="behaviour") -> FilteringResult`

Which parameters separate the draws that show a behaviour from those that do not.
`behaviour` and `nonbehaviour` are masks or indices of rows of `points`; by default
the second group is every other draw. For each parameter, `smirnov_test` compares
its distribution in the two groups. `FilteringResult` has `names`, `d` and `p` (the
statistic and p-value of each parameter), `n_behaviour`, `n_nonbehaviour`,
`significant` (the parameters with `p < pvalue_ks`, the most significant first),
`correlations(group="behaviour", *, pvalue=1e-5, alpha2=0.0)` (pairs of parameters
correlated inside a group, as Dynare's `pvalue_corr` and `alpha2_stab` select them)
and `summary()`. `RestrictionSample.filtering(which=None)` filters the solved draws
of `sample_restrictions` on whether the restrictions `which` hold, as Dynare's
`map_calibration` does.

### `smirnov_test(x1, x2, alternative="two-sided") -> (d, p)`

The two-sample Smirnov test of Dynare's toolbox: the largest distance between the
empirical distribution functions, and the asymptotic p-value with Stephens'
small-sample factor. `alternative` is `"two-sided"`, `"less"` or `"greater"`.

### `sobol(run, parameters, *, n=1024, seed=20260724, bootstrap=200, conf_level=0.95) -> SobolResult`

First-order and total Sobol indices of `run` over independent parameters, on
Saltelli's design: `A`, `B` and every `AB_i`, from a scrambled Sobol sequence, at
`n (k + 2)` evaluations, with `n` a power of two. `run` maps `{name: value}` to a
number, a mapping or an array. The first-order index is Saltelli et al. (2010) and
the total index Jansen (1999), as in SALib and `cgelab.sobol`; the half-widths come
from a bootstrap over the base points. `SobolResult` has `names`, `outputs`, `A`,
`B`, `values` (each output on `A`, `B` and `AB`), `indices()` (per output, `S1`,
`S1_conf`, `ST`, `ST_conf` and the number `n` of base points used: a base point
where the output is missing anywhere is left out) and `summary()`.

### `solution_outputs(model, params, *, irf=(), moments=(), cov=None, guess=None, steady_fn=None, qz_criterium=None, relative_irf=False) -> callable`

A `run` for `sobol` from the first-order solution. `irf` lists `(variable, shock,
horizons)` and `moments` lists `(variable1, variable2, lags)`, with the units of
`check_restrictions`. Each horizon and lag is one output, named as a restriction
label (`"pi(4), eps_v"`, `"x, x"`); a point without a determinate solution gives
NaN. The names in `run.outputs`.

Validated on Dynare 7.2's `tests/gsa/ls2003a.mod`, over three samples of 1024
draws: each draw's class against Dynare's solver, and the Smirnov tests and
correlations on Dynare's groups. The Sobol design and indices are validated
against SALib 1.6 (see `validation/README.md`). Dynare's reduced-form mapping
(`redform`), RMSE filtering (`rmse`) and Morris screening are not reproduced.

```python
import dsgelab as dsg
from dsgelab.examples import nk

model, params, guess = nk.build()
cov = {"eps_a": 0.01, "eps_v": 0.0025}
priors = {"phi_pi": ("gamma", 1.5, 0.5), "kappa": ("gamma", 0.1, 0.05),
          "rho_v": dsg.Uniform(0.0, 0.9)}
stab = dsg.stability_map(model, params, priors, draws=512, guess=guess, cov=cov)
print(stab.summary())                                # phi_pi drives determinacy

run = dsg.solution_outputs(model, params, irf=[("pi", "eps_v", 4)],
                           moments=[("x", "x", 0)], cov=cov, guess=guess)
inside = {"phi_pi": dsg.Uniform(1.2, 2.5), "kappa": dsg.Uniform(0.05, 0.3),
          "rho_v": dsg.Uniform(0.0, 0.9)}
print(dsg.sobol(run, inside, n=256).summary())
```

## Plots

matplotlib is optional: `pip install "dsgelab[plot]"`. Each function returns a
matplotlib `Figure`, built without `pyplot`, so nothing global changes; save it
with `fig.savefig(...)` or let a notebook show it.

| Function | Draws |
|---|---|
| `plot_irf(responses, variables=None, *, columns=3, title=None, figsize=None)` | impulse responses, one panel per variable, impact numbered 1. `responses` is what `FirstOrderSolution.irf` returns, or `{label: responses}` to overlay shocks, models or parameter values |
| `plot_paths(paths, variables=None, *, columns=3, start=1, title=None, figsize=None)` | paths of variables (a simulation, smoothed variables, a transition), one panel each, or `{label: paths}` |
| `plot_decomposition(decomposition, variable, *, start=1, ax=None, title=None)` | a shock decomposition as stacked bars, positive contributions up and negative down, the initial condition as one more, and the total as a line |
| `plot_forecast(forecast, variables=None, *, history=None, columns=3, title=None, figsize=None)` | a `Forecast` or the dictionary of `posterior_forecast`, mean and band, after `history` when given |
| `plot_posterior(draws, *, names=None, priors=None, problem=None, mode=None, columns=3, title=None, figsize=None)` | for each parameter of an `MCMCResult` (or an array with `names`), the posterior density in black, the prior in grey and the mode dashed |

### `prior_density(prior, *, steps=200, truncation=1e-3) -> (x, f)`

A prior's density on a grid between the quantiles `truncation` and
`1 - truncation` (`10 truncation` for the inverse gammas), with the values above 10
dropped where the density peaks at the edge: Dynare's `draw_prior_density`.

### `posterior_density(draws, *, grid_points=512) -> (x, f)`

The kernel density of `posterior_moments` in Dynare: a Gaussian kernel with
Silverman's bandwidth, corrected for the draws a Metropolis chain repeats when it
rejects, by linear binning and a fast Fourier transform on `grid_points` points.

Both densities are validated against Dynare: `draw_prior_density` on the nine
priors of `fs2000.mod` within 2e-12, and `posterior_moments` on two sets of draws,
one with repeated values, within 9e-16 (see `validation/README.md`). The plots draw
what they are given; the test suite checks the numbers on each figure.

```python
import dsgelab as dsg
from dsgelab.examples import nk

model, params, guess = nk.build()
sol = dsg.solve(model, params, guess=guess)
fig = dsg.plot_irf({"monetary": sol.irf("eps_v"), "technology": sol.irf("eps_a")},
                   ["x", "pi", "i"])
fig.savefig("irf.png")
```

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
