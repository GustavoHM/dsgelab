"""Optimal policy: the Ramsey planner under commitment, timeless perspective.

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

from typing import Mapping, Sequence

import numpy as np

from .model import Model
from .steady import SteadyState, steady_state

__all__ = ["RamseyError", "ramsey_model", "ramsey_steady_state", "ramsey_multipliers"]


class RamseyError(ValueError):
    """Raised for a planner problem that is malformed or not representable."""


def _symbols_of(sympy, expr, table):
    inv = {s: name for name, s in table.items()}
    return {inv[s] for s in expr.free_symbols if s in inv}


def ramsey_model(equations: Sequence[str], *, states: Sequence[str], jumps: Sequence[str],
                 shocks: Sequence[str], parameters: Sequence[str],
                 instruments: Sequence[str], objective: str, discount,
                 name: str = "ramsey", derivatives: str = "both",
                 multiplier_prefix: str = "MULT_", auxiliary: str = "auto") -> Model:
    """Build the Ramsey planner's equilibrium as a :class:`~dsgelab.Model`.

    Parameters
    ----------
    equations : sequence of str
        The private-sector conditions, one fewer per instrument than there are
        variables. Equations that contain a shock are exogenous processes.
    states, jumps, shocks, parameters : sequence of str
        As in :func:`dsgelab.from_equations`. The instruments are among them,
        usually among the jumps.
    instruments : sequence of str
        The variables left without an equation. They are not otherwise treated
        differently -- the planner chooses every endogenous variable -- but
        their number is what makes the private block square once the
        first-order conditions are added, and naming them catches a miscount.
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
    from .symbolic import _require_sympy, model_from_expressions, parse_system

    sympy = _require_sympy()
    if auxiliary not in ("auto", "always"):
        raise RamseyError(f"auxiliary must be 'auto' or 'always', got {auxiliary!r}")
    states, jumps = list(states), list(jumps)
    shocks, parameters = list(shocks), list(parameters)
    instruments = list(instruments)
    private = states + jumps

    unknown = [v for v in instruments if v not in private]
    if unknown:
        raise RamseyError(f"instruments not declared as states or jumps: {unknown}")
    if len(equations) != len(private) - len(instruments):
        raise RamseyError(
            f"{len(equations)} private equations for {len(private)} variables and "
            f"{len(instruments)} instrument(s): expected {len(private) - len(instruments)}")

    parsed = parse_system(equations, variables=private, shocks=shocks, parameters=parameters)
    exprs, cur, nxt, eps, par = (parsed[k] for k in ("expressions", "cur", "nxt", "eps", "par"))
    ns = parsed["namespace"]
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

    # --- objective and discount factor.
    from .symbolic import _parse_one
    U = _parse_one(sympy, objective, ns, "objective")
    if U.free_symbols & (set(nxt.values()) | shock_syms):
        raise RamseyError("the objective must depend on current-period variables only")
    if isinstance(discount, str):
        beta = _parse_one(sympy, discount, ns, "discount")
        if beta.free_symbols - set(par.values()):
            raise RamseyError("the discount factor may depend on parameters only")
    else:
        beta = sympy.Float(float(discount))

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
    labels = (list(equations) + [f"FOC[{v}]" for v in endo]
              + [f"definition of {z}" for z in aux_jumps]
              + [f"{n}(+1) = {n[4:]}" for n in lag_mult + lag_vars])

    model = model_from_expressions(all_exprs, states=new_states, jumps=new_jumps,
                                   shocks=shocks, parameters=parameters, name=name,
                                   derivatives=derivatives, source_equations=labels)
    model.ramsey = dict(
        private=private, private_states=states, private_jumps=jumps,
        instruments=instruments, exogenous=sorted(exo_vars), endogenous=endo,
        constraints=[equations[i] for i in con_eq], constraint_index=con_eq,
        multipliers=mult,
        lagged_multipliers=lag_mult, lagged_variables=lag_vars, auxiliary=aux_jumps,
        objective=objective, discount=discount,
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

    n_private = len(info["private"]) - len(info["instruments"])
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
