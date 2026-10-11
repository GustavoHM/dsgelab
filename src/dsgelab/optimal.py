"""Optimal policy: Ramsey under commitment, discretion, and optimal simple rules.

The Ramsey planner under commitment, timeless perspective, comes first; then
:func:`discretionary_policy` and :func:`osr`, at the end of the module.

Optional: requires ``sympy`` (``pip install dsgelab[symbolic]``).

The input is the PRIVATE sector -- its equilibrium conditions, written in the
syntax of :func:`dsgelab.from_equations` -- with fewer equations than variables:
the instruments have no equation of their own. The planner picks the whole
allocation subject to those conditions, and its first-order conditions supply
the missing equations. What comes back is an ordinary :class:`~dsgelab.Model`,
so steady state, first- and second-order solutions, impulse responses and
moments work on it unchanged.

The mathematics
---------------
With ``c_i(x_{t+1}, x_t) = 0`` the private conditions that are not exogenous
processes and ``mu_i,t`` their multipliers, the planner maximises

    E_0 SUM_t beta^t [ U(x_t) + SUM_i mu_i,t c_i(x_{t+1}, x_t) ].

Because ``mu_t`` is known at ``t``, ``mu_t E_t c = E_t mu_t c`` and no
expectation is lost by moving the multiplier inside. Differentiating:

* a NON-PREDETERMINED variable ``v`` (a jump, an instrument) is chosen at ``t``,
  and appears in the period-``t`` condition as ``x_t`` and in the period-``t-1``
  condition as ``x_{t+1}``::

      dU/dv_t + mu_t' dc/dcur_v + beta^-1 mu_{t-1}' dc/dnxt_v |_(x_t, x_{t-1}) = 0

  The lagged multipliers become states, ``lag_MULT_i``, and so do lagged copies
  of any variable the shifted derivative still contains;

* a PREDETERMINED variable ``k`` -- in this package's dating, ``k_{t+1}`` is
  chosen at ``t`` -- gives the condition for ``k_{t+1}``, which holds in
  expectation at ``t``::

      mu_t' dc/dnxt_k + beta E_t[ dU/dk_{t+1} + mu_{t+1}' dc/dcur_k |_(x_{t+2}, x_{t+1}) ] = 0

  When the bracket depends on the current period only it is shifted one period
  forward in place. When it also depends on ``t+1`` -- which would put ``t+2``
  in a two-period model -- it is named as an auxiliary jump ``FOC_k``, defined
  one period back, and the condition reads ``beta * FOC_k(+1)``;

* EXOGENOUS processes, the equations that carry a shock, are not constraints the
  planner can trade against: they are kept as they are, and their variables get
  no first-order condition.

Timeless perspective: the lagged multipliers start at their steady-state value,
which in a perturbation solution is zero deviation. The impulse responses are
therefore those of a planner who has been committed since the infinite past --
the convention of Dynare's ``ramsey_model`` at order one.

What the multipliers mean
-------------------------
``MULT_i`` belongs to the i-th NON-EXOGENOUS equation, in the order given. The
sign is that of the Lagrangian above, with the objective MAXIMISED. A loss to be
minimised is passed with a minus sign; passing it without one leaves the
allocation unchanged in a linear-quadratic model and flips the multipliers,
which is a mistake the impulse responses of the real variables will not show.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .model import Model
from .steady import SteadyState, steady_state

__all__ = ["RamseyError", "ramsey_model", "ramsey_steady_state", "ramsey_multipliers",
           "PrivateBlock", "DiscretionError", "DiscretionSolution", "discretionary_policy",
           "OSRError", "OSRResult", "osr"]


class RamseyError(ValueError):
    """Raised for a planner problem that is malformed or not representable."""


def _symbols_of(sympy, expr, table):
    inv = {s: name for name, s in table.items()}
    return {inv[s] for s in expr.free_symbols if s in inv}


@dataclass
class PrivateBlock:
    """The private sector's conditions, before a policy closes the system.

    ``expressions`` use the naming of :func:`dsgelab.symbolic.parse_system`: ``v``
    for the variable at ``t``, ``v__next`` for ``t+1``, shocks and parameters as
    themselves. There are fewer of them than variables -- one fewer per
    instrument. A block comes from text equations or from a ``.mod`` file
    (:attr:`dsgelab.ModFile.block`), which also carries the file's
    ``planner_objective``, discount and instruments.
    """

    expressions: list
    states: list
    jumps: list
    shocks: list
    parameters: list
    labels: list
    instruments: list | None = None
    objective: object = None          # sympy, current-period symbols, MAXIMISED
    discount: object = None           # sympy, in the parameters
    name: str = "model"

    @property
    def variables(self) -> list:
        return list(self.states) + list(self.jumps)

    def tables(self, sympy):
        S = sympy.Symbol
        return ({v: S(v) for v in self.variables}, {v: S(f"{v}__next") for v in self.variables},
                {e: S(e) for e in self.shocks}, {p: S(p) for p in self.parameters})


def _private_block(equations, *, states=None, jumps=None, shocks=None, parameters=None,
                   instruments=None, objective=None, discount=None, flip_file_objective=False,
                   error=None) -> PrivateBlock:
    """Bring the three ways of stating a private sector to one :class:`PrivateBlock`."""
    from .symbolic import _namespace, _parse_one, _require_sympy, parse_system

    sympy = _require_sympy()
    err = error or RamseyError
    if hasattr(equations, "block") and getattr(equations, "block", None) is not None:
        mf = equations
        base = mf.block
        planner = getattr(mf, "planner", None) or {}
        obj = planner.get("objective")
        if obj is not None and flip_file_objective:
            obj = -obj
        block = PrivateBlock(list(base.expressions), list(base.states), list(base.jumps),
                             list(base.shocks), list(base.parameters), list(base.labels),
                             instruments=planner.get("instruments"), objective=obj,
                             discount=planner.get("discount"), name=base.name)
    elif isinstance(equations, PrivateBlock):
        block = equations
    else:
        if None in (states, jumps, shocks, parameters):
            raise err("text equations need states=, jumps=, shocks= and parameters=")
        states, jumps, shocks, parameters = (list(states), list(jumps), list(shocks),
                                             list(parameters))
        parsed = parse_system(equations, variables=states + jumps, shocks=shocks,
                              parameters=parameters)
        block = PrivateBlock(parsed["expressions"], states, jumps, shocks, parameters,
                             list(equations))

    ns = _namespace(sympy, block.variables, block.shocks, block.parameters)
    if instruments is not None:
        block.instruments = list(instruments)
    if objective is not None:
        block.objective = (_parse_one(sympy, objective, ns, "objective")
                           if isinstance(objective, str) else objective)
    if discount is not None:
        if isinstance(discount, str):
            block.discount = _parse_one(sympy, discount, ns, "discount")
        elif isinstance(discount, (int, float)):
            block.discount = sympy.Float(float(discount))
        else:
            block.discount = discount
    if block.objective is None:
        raise err("no objective: pass objective=, or a .mod file with planner_objective")
    if block.discount is None:
        raise err("no discount factor: pass discount=, or a .mod file with planner_discount")
    cur, nxt, eps, par = block.tables(sympy)
    if block.objective.free_symbols & (set(nxt.values()) | set(eps.values())):
        raise err("the objective must depend on current-period variables only")
    if block.discount.free_symbols - set(par.values()):
        raise err("the discount factor may depend on parameters only")
    n_missing = len(block.variables) - len(block.expressions)
    if block.instruments is not None:
        unknown = [v for v in block.instruments if v not in block.variables]
        if unknown:
            raise err(f"instruments not declared as states or jumps: {unknown}")
        if n_missing != len(block.instruments):
            raise err(f"{len(block.expressions)} private equations for "
                      f"{len(block.variables)} variables and {len(block.instruments)} "
                      f"instrument(s): expected {len(block.variables) - len(block.instruments)}")
    elif n_missing < 1:
        raise err(f"{len(block.expressions)} equations for {len(block.variables)} variables: "
                  f"a policy problem needs at least one variable without an equation")
    return block


def ramsey_model(equations, *, states: Sequence[str] | None = None,
                 jumps: Sequence[str] | None = None, shocks: Sequence[str] | None = None,
                 parameters: Sequence[str] | None = None,
                 instruments: Sequence[str] | None = None, objective=None, discount=None,
                 name: str = "ramsey", derivatives: str = "both",
                 multiplier_prefix: str = "MULT_", auxiliary: str = "auto") -> Model:
    """Build the Ramsey planner's equilibrium as a :class:`~dsgelab.Model`.

    Parameters
    ----------
    equations : sequence of str, ModFile or PrivateBlock
        The private-sector conditions, one fewer per instrument than there are
        variables. Equations that contain a shock are exogenous processes. A
        :class:`~dsgelab.ModFile` read from a ``.mod`` with ``planner_objective``
        and ``ramsey_model`` brings its own objective, discount and instruments,
        and the arguments below override them.
    states, jumps, shocks, parameters : sequence of str
        As in :func:`dsgelab.from_equations`; only for text equations.
    instruments : sequence of str, optional
        The variables left without an equation. They are not otherwise treated
        differently -- the planner chooses every endogenous variable -- but
        naming them catches a miscount. Dynare does not require them, and
        neither does a ``.mod`` read here.
    objective : str
        The period objective ``U(x_t)``, to be MAXIMISED, in current-period
        variables and parameters only.
    discount : str or float
        The planner's discount factor: a parameter name, an expression in the
        parameters, or a number.
    multiplier_prefix : str
        Names of the multipliers, ``MULT_1``, ``MULT_2``, ... by default, which
        is also what Dynare calls them.
    auxiliary : {"auto", "always"}
        ``"always"`` names the bracket of every predetermined variable as an
        auxiliary jump even where shifting it forward would do. The two
        formulations have the same solution; the option exists so that this can
        be verified.

    Returns
    -------
    Model
        States are the private states, then ``lag_MULT_i``, then the lagged
        copies ``lag_<v>`` the first-order conditions need; jumps are the
        private jumps, then the multipliers, then any auxiliary ``FOC_<k>``.
        ``model.ramsey`` records the layout.
    """
    from .symbolic import _require_sympy, model_from_expressions

    sympy = _require_sympy()
    if auxiliary not in ("auto", "always"):
        raise RamseyError(f"auxiliary must be 'auto' or 'always', got {auxiliary!r}")
    block = _private_block(equations, states=states, jumps=jumps, shocks=shocks,
                           parameters=parameters, instruments=instruments,
                           objective=objective, discount=discount)
    states, jumps = list(block.states), list(block.jumps)
    shocks, parameters = list(block.shocks), list(block.parameters)
    instruments = list(block.instruments or [])
    private = states + jumps
    exprs = block.expressions
    cur, nxt, eps, par = block.tables(sympy)
    shock_syms = set(eps.values())

    # --- exogenous processes: equations with a shock in them.
    exo_eq = [i for i, e in enumerate(exprs) if e.free_symbols & shock_syms]
    con_eq = [i for i in range(len(exprs)) if i not in exo_eq]
    exo_vars = set()
    for i in exo_eq:
        exo_vars |= _symbols_of(sympy, exprs[i], cur) | _symbols_of(sympy, exprs[i], nxt)
    bad = [v for v in exo_vars if v in jumps]
    if bad:
        raise RamseyError(
            f"variables {sorted(bad)} appear in an equation with a shock but are jumps. "
            f"Shocks belong in the transition of exogenous states, which the planner takes "
            f"as given.")
    if len(exo_vars) != len(exo_eq):
        raise RamseyError(
            f"{len(exo_eq)} equation(s) carry a shock but involve {len(exo_vars)} variable(s) "
            f"{sorted(exo_vars)}: an exogenous block must determine exactly its own variables")
    clash = [v for v in instruments if v in exo_vars]
    if clash:
        raise RamseyError(f"instruments {clash} are exogenous processes")
    endo = [v for v in private if v not in exo_vars]
    U, beta = block.objective, block.discount

    # --- names of what the planner adds.
    m = len(con_eq)
    taken = set(private) | set(shocks) | set(parameters)
    mult = [f"{multiplier_prefix}{j + 1}" for j in range(m)]
    lag_mult = [f"lag_{n}" for n in mult]
    for n in mult + lag_mult:
        if n in taken:
            raise RamseyError(f"the name {n!r} is already in use; pass multiplier_prefix=")
    S = sympy.Symbol
    mu = [S(n) for n in mult]
    mu_next = [S(f"{n}__next") for n in mult]
    lag_mu = [S(n) for n in lag_mult]
    lag_mu_next = [S(f"{n}__next") for n in lag_mult]
    C = [exprs[i] for i in con_eq]

    cur_to_nxt = {cur[v]: nxt[v] for v in private}
    cur_to_nxt.update({mu[j]: mu_next[j] for j in range(m)})

    lagged_needed: list[str] = []
    focs, aux_states, aux_jumps, aux_eqs = [], [], [], []

    for v in endo:
        dUv = sympy.diff(U, cur[v])
        d_cur = sum((mu[j] * sympy.diff(C[j], cur[v]) for j in range(m)), sympy.Integer(0))
        d_nxt = [sympy.diff(C[j], nxt[v]) for j in range(m)]
        if v in jumps:
            # shift dc/dnxt_v back one period: nxt -> cur, cur -> lag_<w>
            back = {}
            for w in _symbols_of(sympy, sum(d_nxt, sympy.Integer(0)), cur):
                back[cur[w]] = S(f"lag_{w}")
                if w not in lagged_needed:
                    lagged_needed.append(w)
            back.update({nxt[w]: cur[w] for w in private})
            lag_term = sum((lag_mu[j] * d_nxt[j].xreplace(back) for j in range(m)),
                           sympy.Integer(0))
            focs.append(dUv + d_cur + lag_term / beta)
        else:
            bracket = dUv + d_cur
            today = sum((mu[j] * d_nxt[j] for j in range(m)), sympy.Integer(0))
            if auxiliary == "always" or bracket.free_symbols & set(nxt.values()):
                z = f"FOC_{v}"
                if z in taken or z in aux_jumps:
                    raise RamseyError(f"the name {z!r} is already in use")
                aux_jumps.append(z)
                aux_eqs.append(S(z) - bracket)
                focs.append(today + beta * S(f"{z}__next"))
            else:
                focs.append(today + beta * bracket.xreplace(cur_to_nxt))

    lag_vars = [f"lag_{w}" for w in lagged_needed]
    for n in lag_vars:
        if n in taken or n in lag_mult:
            raise RamseyError(f"the name {n!r} is already in use")

    transitions = [lag_mu_next[j] - mu[j] for j in range(m)]
    transitions += [S(f"lag_{w}__next") - cur[w] for w in lagged_needed]

    new_states = states + lag_mult + lag_vars
    new_jumps = jumps + mult + aux_jumps
    all_exprs = list(exprs) + focs + aux_eqs + transitions
    labels = (list(block.labels) + [f"FOC[{v}]" for v in endo]
              + [f"definition of {z}" for z in aux_jumps]
              + [f"{n}(+1) = {n[4:]}" for n in lag_mult + lag_vars])

    model = model_from_expressions(all_exprs, states=new_states, jumps=new_jumps,
                                   shocks=shocks, parameters=parameters, name=name,
                                   derivatives=derivatives, source_equations=labels)
    model.ramsey = dict(
        private=private, private_states=states, private_jumps=jumps,
        instruments=instruments, exogenous=sorted(exo_vars), endogenous=endo,
        constraints=[block.labels[i] for i in con_eq], constraint_index=con_eq,
        multipliers=mult, n_private_equations=len(exprs),
        lagged_multipliers=lag_mult, lagged_variables=lag_vars, auxiliary=aux_jumps,
        objective=str(U), discount=str(beta),
        objective_expr=U, discount_expr=beta)
    return model


def _ramsey(model: Model) -> dict:
    info = getattr(model, "ramsey", None)
    if info is None:
        raise RamseyError("not a Ramsey model: build it with dsgelab.ramsey_model")
    return info


def ramsey_multipliers(model: Model, private, params) -> dict:
    """Multipliers, auxiliaries and lags consistent with a private steady state.

    At a steady state the first-order conditions are LINEAR in the multipliers
    and the auxiliaries, so given the private allocation they are solved by
    least squares. The residual of that fit says whether ``private`` is the
    planner's steady state at all: the private conditions alone are satisfied by
    a continuum of instrument settings, and only one of them leaves the
    first-order conditions solvable.

    Returns the full guess (a dict over every variable of ``model``) and stores
    the least-squares residual under the key ``"_residual"``.
    """
    info = _ramsey(model)
    values = {v: float(private[v]) for v in info["private"]}
    for lag in info["lagged_variables"]:
        values[lag] = values[lag[4:]]
    unknown = info["multipliers"] + info["auxiliary"]

    def full(theta):
        out = dict(values)
        out.update(dict(zip(unknown, theta)))
        for lm, mname in zip(info["lagged_multipliers"], info["multipliers"]):
            out[lm] = out[mname]
        return model.vector(out)

    n_private = info["n_private_equations"]
    rows = slice(n_private, n_private + len(info["endogenous"]) + len(info["auxiliary"]))
    zero = np.zeros(model.n_shocks)

    def r(theta):
        x = full(theta)
        return np.asarray(model.residuals(x, x, zero, params), dtype=float)[rows]

    k = len(unknown)
    r0 = r(np.zeros(k))
    J = np.column_stack([r(np.eye(k)[j]) - r0 for j in range(k)])   # exact: affine
    theta, *_ = np.linalg.lstsq(J, -r0, rcond=None)
    out = dict(values)
    out.update(dict(zip(unknown, theta)))
    for lm, mname in zip(info["lagged_multipliers"], info["multipliers"]):
        out[lm] = out[mname]
    out["_residual"] = float(np.max(np.abs(r(theta)))) if k else float(np.max(np.abs(r0)))
    return out


def ramsey_steady_state(model: Model, private, params, **kwargs) -> SteadyState:
    """The planner's steady state, started from a guess of the private allocation.

    The multipliers are first fitted by :func:`ramsey_multipliers`; the whole
    system is then solved by :func:`dsgelab.steady_state`, to which ``kwargs``
    are forwarded. A private guess that is already the planner's steady state is
    returned unchanged up to the solver's tolerance.
    """
    guess = ramsey_multipliers(model, private, params)
    guess.pop("_residual")
    return steady_state(model, guess, params, **kwargs)


# ======================================================================= discretion
class DiscretionError(ValueError):
    """Raised for a discretionary problem outside the linear-quadratic class."""


@dataclass
class DiscretionSolution:
    """Markov-perfect policy without commitment: ``y_t = H y_{t-1} + G eps_t``.

    ``y`` stacks every private variable, in absolute deviations from ``steady``.
    The dating is Dynare's: ``eps_t`` moves ``y_t`` on impact. A predetermined
    state of this package, ``k(+1) = ...``, enters as an equation one period
    later, so its response on impact is zero, as it is in :meth:`irf` of a
    first-order solution.
    """

    names: list
    instruments: list
    shocks: list
    H: np.ndarray
    G: np.ndarray
    steady: dict
    discount: float
    U: float                 # objective at the steady state
    U_y: np.ndarray          # gradient of the objective (zero: checked)
    U_yy: np.ndarray         # Hessian of the objective, maximised sign
    iterations: int
    Sigma: np.ndarray | None = None

    def _cov(self, cov):
        from .solution import shock_covariance
        if cov is None:
            if self.Sigma is None:
                raise ValueError("pass cov=: no shock covariance is attached")
            return self.Sigma
        return shock_covariance(self.shocks, cov)

    def irf(self, name: str, size: float = 1.0, periods: int = 40, *,
            levels: bool = False) -> dict:
        """Response of every variable to ``size`` units of the innovation ``name``."""
        e = np.zeros(len(self.shocks))
        e[self.shocks.index(name)] = size
        return self.irf_impulse(e, periods=periods, levels=levels)

    def irf_impulse(self, impulse, periods: int = 40, *, levels: bool = False) -> dict:
        y = self.G @ np.asarray(impulse, dtype=float)
        out = np.zeros((periods, len(self.names)))
        for t in range(periods):
            out[t] = y
            y = self.H @ y
        base = np.array([self.steady[n] for n in self.names]) if levels else 0.0
        out = out + base
        return {n: out[:, i] for i, n in enumerate(self.names)}

    def covariance(self, cov=None) -> np.ndarray:
        from scipy.linalg import solve_discrete_lyapunov
        Sigma = self._cov(cov)
        return solve_discrete_lyapunov(self.H, self.G @ Sigma @ self.G.T)

    def moments(self, cov=None) -> dict:
        V = self.covariance(cov)
        sd = np.sqrt(np.maximum(np.diag(V), 0.0))
        return dict(names=list(self.names), cov=V, std=sd,
                    std_by_name=dict(zip(self.names, sd)))

    def welfare(self, cov=None, *, initial=None, shock=None) -> dict:
        """Discounted objective, conditional and unconditional.

        In a linear-quadratic problem the objective is ``U + 1/2 y' U_yy y`` and
        the mean of ``y`` is zero, so both numbers are quadratic forms.
        CONDITIONAL welfare starts at ``y_{-1} = initial`` (the steady state by
        default) with the innovation ``shock`` in the first period (none by
        default): with ``y_0 = H y_{-1} + G shock``,

            W = U/(1-beta) + 1/2 y_0' P y_0 + 1/2 tr(U_yy M),

        ``P = U_yy + beta H' P H`` and ``M = beta H M H' + beta/(1-beta) G Sigma G'``.
        Dynare's ``evaluate_planner_objective`` takes the first period's
        innovation from ``oo_.exo_simul`` when a simulation has been run, which
        makes its conditional number depend on a random draw; pass the same
        ``shock`` to reproduce it.
        """
        from scipy.linalg import solve_discrete_lyapunov
        Sigma = self._cov(cov)
        b = self.discount
        S = self.G @ Sigma @ self.G.T
        V = solve_discrete_lyapunov(self.H, S)
        M = solve_discrete_lyapunov(np.sqrt(b) * self.H, b / (1 - b) * S)
        P = solve_discrete_lyapunov(np.sqrt(b) * self.H.T, self.U_yy)
        y_lag = np.zeros(len(self.names))
        if initial:
            for k, v in initial.items():
                y_lag[self.names.index(k)] = float(v) - self.steady[k]
        u = np.zeros(len(self.shocks)) if shock is None else np.asarray(shock, float)
        y0 = self.H @ y_lag + self.G @ u
        base = self.U / (1 - b)
        return dict(unconditional=base + 0.5 * float(np.sum(self.U_yy * V)) / (1 - b),
                    conditional=base + 0.5 * float(y0 @ P @ y0)
                    + 0.5 * float(np.sum(self.U_yy * M)))


def _sylvester_doubling(d, g, h, tol, maxit):
    v = d.copy()
    for _ in range(maxit):
        vadd = g @ v @ h
        v = v + vadd
        if np.linalg.norm(vadd, 1) <= tol * np.linalg.norm(v, 1):
            break
        g = g @ g
        h = h @ h
    return v


def _dennis(AAlag, AA0, AAlead, BB, bigw, instr_id, beta, maxit, tol, qz_criterium):
    """Dennis (2007), as Dynare's ``discretionary_policy_engine`` writes it."""
    eq_nbr, endo_nbr = AAlag.shape
    exo_nbr = BB.shape[1]
    y = [i for i in range(endo_nbr) if i not in instr_id]
    ni = len(instr_id)
    A0 = AA0[:, y]
    A1 = -AAlag[:, y]
    A2 = -AAlead[:, y]
    A3 = -AA0[:, instr_id]
    A4 = -AAlead[:, instr_id]
    A5 = -BB
    W = bigw[np.ix_(y, y)]
    Q = bigw[np.ix_(instr_id, instr_id)]
    A6 = -AAlag[:, instr_id]
    aux = np.any(A6 != 0, axis=0)
    na = int(aux.sum())
    m = eq_nbr + na
    A00 = np.zeros((m, m)); A00[:eq_nbr, :eq_nbr] = A0; A00[eq_nbr:, eq_nbr:] = np.eye(na)
    A11 = np.zeros((m, m)); A11[:eq_nbr, :eq_nbr] = A1; A11[:eq_nbr, eq_nbr:] = A6[:, aux]
    A22 = np.zeros((m, m)); A22[:eq_nbr, :eq_nbr] = A2
    A33 = np.zeros((m, ni)); A33[:eq_nbr] = A3; A33[eq_nbr:, aux] = np.eye(na)
    A44 = np.zeros((m, ni)); A44[:eq_nbr] = A4
    A55 = np.zeros((m, exo_nbr)); A55[:eq_nbr] = A5
    WW = np.zeros((m, m)); WW[:eq_nbr, :eq_nbr] = W

    H1 = np.zeros((m, m))
    F1 = np.zeros((ni, m))
    for it in range(1, maxit + 1):
        P = _sylvester_doubling(WW + beta * F1.T @ Q @ F1, beta * H1.T, H1, tol, maxit)
        if not np.all(np.isfinite(P)):
            raise DiscretionError("the value-function iteration produced non-finite numbers")
        D = A00 - A22 @ H1 - A44 @ F1
        Dinv = np.linalg.inv(D)
        A3DPD = A33.T @ Dinv.T @ P @ Dinv
        F1n = -np.linalg.solve(Q + A3DPD @ A33, A3DPD @ A11)
        H1n = Dinv @ (A11 + A33 @ F1n)
        gap = float(np.max(np.abs(np.vstack([H1n - H1, F1n - F1]))))
        H1, F1 = H1n, F1n
        if not np.isfinite(gap):
            raise DiscretionError("the iteration diverged (NaN)")
        if gap < tol:
            break
    else:
        raise DiscretionError(f"no convergence in {maxit} iterations")
    if np.max(np.abs(np.linalg.eigvals(H1))) > qz_criterium:
        raise DiscretionError("the discretionary equilibrium found is explosive")
    F2 = -np.linalg.solve(Q + A3DPD @ A33, A3DPD @ A55)
    H2 = Dinv @ (A55 + A33 @ F2)
    # back to the original variables: instruments, then the auxiliary lagged instruments
    n_aug = endo_nbr + na
    endo_augm = [i for i in range(n_aug) if i not in instr_id]
    H = np.zeros((n_aug, n_aug))
    G = np.zeros((n_aug, exo_nbr))
    H[np.ix_(endo_augm, endo_augm)] = H1
    H[np.ix_(instr_id, endo_augm)] = F1
    G[endo_augm] = H2
    G[instr_id] = F2
    aux_cols = [instr_id[k] for k in range(ni) if aux[k]]
    if na:
        H[:, aux_cols] = H[:, n_aug - na:]
    return H[:endo_nbr, :endo_nbr], G[:endo_nbr], it


def discretionary_policy(equations, *, params, states: Sequence[str] | None = None,
                         jumps: Sequence[str] | None = None,
                         shocks: Sequence[str] | None = None,
                         parameters: Sequence[str] | None = None,
                         instruments: Sequence[str] | None = None, objective=None,
                         discount=None, steady: Mapping[str, float] | None = None,
                         cov=None, tol: float = 1e-12, maxiter: int = 3000,
                         qz_criterium: float = 1 + 1e-6) -> DiscretionSolution:
    """Optimal policy under discretion, in the linear-quadratic class.

    The private block is stated as for :func:`ramsey_model` -- text equations,
    or a :class:`~dsgelab.ModFile` whose ``discretionary_policy`` command names
    the instruments --, and the objective is MAXIMISED. A ``.mod`` file's
    ``planner_objective`` is a LOSS under Dynare's ``discretionary_policy``,
    and is read with the sign flipped.

    The equations must be linear and the objective quadratic around ``steady``
    (all zeros by default), with no linear term and no cross term between an
    instrument and the other variables: the conditions under which Dennis's
    (2007) algorithm, which Dynare uses, is the discretionary equilibrium. Each
    is checked and refused with a message.

    ``tol`` stops the iteration on the policy rule; Dynare stops at
    ``sqrt(eps)``, about ``1.5e-8``, which is where a comparison with it levels off.

    Timing. Each equation is dated as Dynare would date it: a condition with a
    next-period jump holds in expectation at ``t``; a transition, whose only
    next-period terms are states (and shocks), holds at ``t+1``; a condition
    with no next-period term at all holds at ``t``.
    """
    from .symbolic import _require_sympy

    sympy = _require_sympy()
    block = _private_block(equations, states=states, jumps=jumps, shocks=shocks,
                           parameters=parameters, instruments=instruments,
                           objective=objective, discount=discount,
                           flip_file_objective=True, error=DiscretionError)
    if not block.instruments:
        raise DiscretionError("discretion needs the instruments named")
    names = block.variables
    n, ne = len(names), len(block.shocks)
    cur, nxt, eps, par = block.tables(sympy)
    exprs = block.expressions
    pvals = {par[p]: float(params[p] if isinstance(params, Mapping) else getattr(params, p))
             for p in block.parameters}
    yss = {v: float((steady or {}).get(v, 0.0)) for v in names}
    point = {**pvals, **{cur[v]: yss[v] for v in names}, **{nxt[v]: yss[v] for v in names},
             **{eps[e]: 0.0 for e in block.shocks}}

    res = np.array([float(e.xreplace(point)) for e in exprs])
    if np.max(np.abs(res), initial=0.0) > 1e-9:
        raise DiscretionError(f"the point given is not a steady state of the private block "
                              f"(largest residual {np.max(np.abs(res)):.2e}); pass steady=")
    Mx = sympy.Matrix(exprs)
    Jn = Mx.jacobian([nxt[v] for v in names])
    Jc = Mx.jacobian([cur[v] for v in names])
    Je = Mx.jacobian([eps[e] for e in block.shocks]) if ne else sympy.zeros(len(exprs), 0)
    every = set(cur.values()) | set(nxt.values()) | set(eps.values())
    for J in (Jn, Jc, Je):
        if any(x.free_symbols & every for x in J):
            raise DiscretionError("the private block is not linear; discretion here is "
                                  "linear-quadratic. Linearise the model first.")
    Jn, Jc = np.array(Jn.xreplace(point), float), np.array(Jc.xreplace(point), float)
    Je = np.array(Je.xreplace(point), float) if ne else np.zeros((len(exprs), 0))

    nxt_jumps = {nxt[v] for v in block.jumps}
    m = len(exprs)
    AAlag, AA0, AAlead, BB = (np.zeros((m, n)), np.zeros((m, n)), np.zeros((m, n)),
                              np.zeros((m, ne)))
    for i, e in enumerate(exprs):
        fs = e.free_symbols
        if fs & nxt_jumps:                         # expectational, dated t
            if fs & set(eps.values()):
                raise DiscretionError(f"equation {i + 1} has both an expected jump and a "
                                      f"shock; put the shock in an exogenous state")
            AAlead[i], AA0[i] = Jn[i], Jc[i]
        elif fs & set(nxt.values()) or fs & set(eps.values()):   # transition, dated t+1
            AA0[i], AAlag[i], BB[i] = Jn[i], Jc[i], Je[i]
        else:                                      # static, dated t
            AA0[i] = Jc[i]

    U = block.objective
    grad = [sympy.diff(U, cur[v]) for v in names]
    hess = sympy.Matrix([[sympy.diff(g, cur[w]) for w in names] for g in grad])
    if any(x.free_symbols & every for x in hess):
        raise DiscretionError("the objective is not quadratic")
    Uyy = np.array(hess.xreplace(pvals), float)
    Uy = np.array([float(g.xreplace(point)) for g in grad])
    if np.max(np.abs(Uy), initial=0.0) > 1e-10:
        bad = [v for v, g in zip(names, Uy) if abs(g) > 1e-10]
        raise DiscretionError(f"the objective has a linear term at the steady state in {bad}")
    instr_id = [names.index(v) for v in block.instruments]
    others = [i for i in range(n) if i not in instr_id]
    if np.max(np.abs(Uyy[np.ix_(instr_id, others)]), initial=0.0) > 1e-12:
        raise DiscretionError("the objective has a cross term between an instrument and "
                              "another variable, which Dennis's algorithm does not admit")
    beta = float(block.discount.xreplace(pvals))
    bigw = -Uyy              # a loss: Dennis minimises y' W y
    H, G, it = _dennis(AAlag, AA0, AAlead, BB, bigw, instr_id, beta, maxiter, tol, qz_criterium)
    from .solution import shock_covariance
    Sigma = shock_covariance(block.shocks, cov) if cov is not None else None
    return DiscretionSolution(names=names, instruments=list(block.instruments),
                              shocks=list(block.shocks), H=H, G=G, steady=yss, discount=beta,
                              U=float(U.xreplace(point)), U_y=Uy, U_yy=Uyy, iterations=it,
                              Sigma=Sigma)


# ============================================================ optimal simple rules
class OSRError(ValueError):
    """Raised for a malformed optimal-simple-rule problem."""


@dataclass
class OSRResult:
    """The coefficients that minimise the loss, and what the loss is there."""

    params: dict              # the optimal rule coefficients
    all_params: object        # the full parameter set at the optimum
    loss: float
    initial_loss: float
    success: bool
    message: str
    evaluations: int
    solution: object          # FirstOrderSolution at the optimum


def _weight_matrix(names, weights):
    if isinstance(weights, Mapping):
        W = np.zeros((len(names), len(names)))
        for key, w in weights.items():
            a, b = (key, key) if isinstance(key, str) else key
            for v in (a, b):
                if v not in names:
                    raise OSRError(f"weighted variable {v!r} is not in the model")
            # One entry, as Dynare's optim_weights stores it: `y, inflation 0.1`
            # adds 0.1 * Cov(y, inflation) to the loss, not twice that.
            W[names.index(a), names.index(b)] += float(w)
        return W
    W = np.asarray(weights, float)
    if W.shape != (len(names), len(names)):
        raise OSRError(f"weight matrix must be {len(names)}x{len(names)}")
    return W


def osr(model: Model, params, *, rule_params: Sequence[str], weights=None, objective=None,
        discount=None, cov=None, guess=None, steady: SteadyState | None = None,
        steady_fn=None, bounds=None, qz_criterium: float | None = None,
        method: str | None = None, options: Mapping | None = None,
        penalty: float = 1e8) -> OSRResult:
    """Optimal simple rule: the rule coefficients that minimise a quadratic loss.

    The loss is ``sum_ij W_ij Cov(y_i, y_j)``, the unconditional covariance at
    first order, in levels -- Dynare's ``osr`` with ``optim_weights``. Given
    ``objective`` and ``discount`` instead, the loss is minus unconditional
    welfare at second order (:func:`dsgelab.welfare`) -- Dynare's ``osr`` with
    ``planner_objective``. A parameter vector at which the model has no
    determinate solution gets the loss ``penalty``, as in Dynare.

    Parameters
    ----------
    model : Model
        The model closed by the rule; ``rule_params`` are among its parameters.
    weights : mapping or array
        ``{"y": 1, ("y", "pi"): 0.1}`` -- a pair adds ``w * Cov(y, pi)`` ONCE,
        as Dynare's ``optim_weights`` does --, or a matrix over ``model.names``.
    guess, steady, steady_fn
        How the steady state is found at each trial: ``steady_fn(params)``, or
        :func:`dsgelab.steady_state` from ``guess`` (or from ``steady``), warm
        started from the previous trial.
    bounds : mapping, optional
        ``{name: (low, high)}``; switches the optimiser to L-BFGS-B.
    """
    from scipy.optimize import minimize

    from .model import get_param, set_params
    from .solve import solve
    from .solution import shock_covariance

    rule_params = list(rule_params)
    if not rule_params:
        raise OSRError("no rule parameters to optimise")
    if (weights is None) == (objective is None):
        raise OSRError("pass either weights= (a quadratic loss in moments) or objective= "
                       "and discount= (minus unconditional welfare), not both")
    Sigma = shock_covariance(model.shocks, cov)
    if weights is not None:
        W = _weight_matrix(model.names, weights)
    else:
        from .welfare import welfare as _welfare, welfare_model
        if discount is None:
            raise OSRError("objective= needs discount=")
        aug = welfare_model(model, objective, discount)
    x0 = np.array([float(get_param(params, p)) for p in rule_params])
    if steady is None and guess is None and steady_fn is None:
        raise OSRError("pass steady=, guess= or steady_fn=")
    last = {"ss": steady, "sol": None, "n": 0}

    def evaluate(x):
        last["n"] += 1
        p = set_params(params, dict(zip(rule_params, map(float, x))))
        try:
            if steady_fn is not None:
                ss = steady_fn(p)
            else:
                start = last["ss"].values if last["ss"] is not None else guess
                ss = steady_state(model, start, p)
            sol = solve(model, p, steady=ss, qz_criterium=qz_criterium)
            if weights is not None:
                V = sol.covariance(Sigma)
                s = sol.coords.scale
                loss = float(np.sum(W * (V * np.outer(s, s))))
            else:
                loss = -_welfare(aug, p, steady=ss, cov=Sigma,
                                 qz_criterium=qz_criterium).unconditional
        except Exception:
            return penalty, None, None
        if not np.isfinite(loss):
            return penalty, None, None
        last["ss"] = ss
        return loss, sol, p

    initial = evaluate(x0)[0]
    kw = dict(options or {})
    f = lambda x: evaluate(x)[0]
    if bounds:
        b = [tuple(bounds.get(p, (None, None))) for p in rule_params]
        res = minimize(f, x0, method=method or "L-BFGS-B", bounds=b,
                       options={"ftol": 1e-15, "gtol": 1e-10, **kw})
    else:
        res = minimize(f, x0, method=method or "BFGS", options={"gtol": 1e-10, **kw})
        if method is None:
            # A quadratic loss in a rule coefficient is often flat in one direction,
            # and BFGS on finite differences stops there with a precision-loss
            # message. A simplex polish from that point costs a few hundred
            # evaluations and settles it.
            pol = minimize(f, res.x, method="Nelder-Mead",
                           options={"xatol": 1e-10, "fatol": 1e-16, "maxiter": 20000,
                                    "maxfev": 20000})
            if pol.fun <= res.fun:
                res = pol
    loss, sol, p = evaluate(res.x)
    if bounds:
        for name, v in zip(rule_params, res.x):
            lo, hi = bounds.get(name, (None, None))
            if (lo is not None and abs(v - lo) < 1e-8) or (hi is not None and abs(v - hi) < 1e-8):
                import warnings
                warnings.warn(f"osr: {name} sits on its bound; the optimum may lie outside")
    return OSRResult(params=dict(zip(rule_params, map(float, res.x))), all_params=p,
                     loss=loss, initial_loss=initial, success=bool(res.success),
                     message=str(res.message), evaluations=last["n"], solution=sol)
