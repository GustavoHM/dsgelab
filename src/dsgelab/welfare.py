"""Welfare at second order: the discounted objective, conditional and unconditional.

Optional: requires ``sympy`` (``pip install dsgelab[symbolic]``).

The objective travels with the model. :func:`welfare` appends one jump,

    WELFARE_t = U(x_t) + beta * E_t WELFARE_{t+1},

and solves the augmented model to second order. Welfare is then an ordinary
variable of the solution, which settles two things at once. CONDITIONAL welfare
is the second-order rule for ``WELFARE`` evaluated at the initial state -- the
steady state by default --, and so carries the risk correction ``1/2 g_ss``
that a deterministic evaluation misses. UNCONDITIONAL welfare is its ergodic
mean, in the closed form of the pruned system. These are the two numbers
Dynare's ``evaluate_planner_objective`` reports at ``order=2``.

For a Ramsey model the objective and discount come from :func:`ramsey_model`,
and a second conditional value starts the lagged multipliers at zero instead of
at their steady state: the planner who commits at ``t = 0`` rather than the
timeless one.

Welfare as consumption: :func:`consumption_equivalent` finds the permanent
proportional change in a named variable that makes two welfare numbers equal.
It does not assume a log utility or a variable called consumption: the change
is applied to the variable the caller names, inside the objective the model
already has.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

from .model import Covariance, Model, Params, get_param
from .steady import SteadyState, steady_state

if TYPE_CHECKING:
    import sympy

__all__ = ["WelfareError", "Welfare", "welfare", "welfare_model", "consumption_equivalent"]


class WelfareError(ValueError):
    """Raised when the objective or the model cannot carry a welfare calculation."""


@dataclass
class Welfare:
    """Welfare numbers, in the units of the objective."""

    unconditional: float
    conditional: float
    deterministic: float                  # U(steady) / (1 - beta)
    conditional_zero_multipliers: float | None
    discount: float
    model: Model
    second_order: object

    def __repr__(self) -> str:
        z = ("" if self.conditional_zero_multipliers is None
             else f", conditional_zero_multipliers={self.conditional_zero_multipliers:.10g}")
        return (f"Welfare(unconditional={self.unconditional:.10g}, "
                f"conditional={self.conditional:.10g}{z}, "
                f"deterministic={self.deterministic:.10g})")


def _objective(model: Model, objective, discount):
    from .symbolic import _namespace, _parse_one, _require_sympy

    sympy = _require_sympy()
    sym = getattr(model, "symbolic", None)
    if sym is None:
        raise WelfareError("welfare needs a model built symbolically (from_equations, "
                           "from_mod, ramsey_model)")
    info = getattr(model, "ramsey", None)
    ns = _namespace(sympy, model.names, model.shocks, sym["parameters"])
    if objective is None:
        if info is None:
            raise WelfareError("pass objective=: only a Ramsey model carries its own")
        U = info["objective_expr"]
    else:
        U = _parse_one(sympy, objective, ns, "objective") if isinstance(objective, str) \
            else objective
    if discount is None:
        if info is None:
            raise WelfareError("pass discount=")
        beta = info["discount_expr"]
    elif isinstance(discount, str):
        beta = _parse_one(sympy, discount, ns, "discount")
    elif isinstance(discount, (int, float)):
        beta = sympy.Float(float(discount))
    else:
        beta = discount
    nxt = {sympy.Symbol(f"{v}__next") for v in model.names}
    if U.free_symbols & (nxt | {sympy.Symbol(e) for e in model.shocks}):
        raise WelfareError("the objective must depend on current-period variables only")
    return sympy, U, beta


def welfare_model(model: Model, objective: str | sympy.Expr | None = None,
                  discount: str | float | sympy.Expr | None = None, *,
                  name: str = "WELFARE") -> Model:
    """``model`` plus the jump ``name`` with ``name = U + beta * name(+1)``."""
    from .symbolic import model_from_expressions

    sympy, U, beta = _objective(model, objective, discount)
    if name in model.names:
        raise WelfareError(f"the model already has a variable called {name!r}")
    sym = model.symbolic
    S = sympy.Symbol
    exprs = list(sym["expressions"]) + [S(name) - U - beta * S(f"{name}__next")]
    labels = list(sym["equations"]) + [f"{name} = U + beta*{name}(+1)"]
    aug = model_from_expressions(exprs, states=model.states, jumps=list(model.jumps) + [name],
                                 shocks=model.shocks, parameters=sym["parameters"],
                                 name=f"{model.name}+welfare", derivatives="both",
                                 source_equations=labels)
    if getattr(model, "ramsey", None) is not None:
        aug.ramsey = dict(model.ramsey)
    aug.welfare = dict(name=name, objective=U, discount=beta)
    return aug


def welfare(model: Model, params: Params, *, steady: SteadyState, cov: Covariance,
            objective: str | sympy.Expr | None = None,
            discount: str | float | sympy.Expr | None = None,
            initial: Mapping[str, float] | None = None,
            name: str = "WELFARE", qz_criterium: float | None = None) -> Welfare:
    """Conditional and unconditional welfare, at second order.

    Parameters
    ----------
    model : Model
        A symbolic model. A Ramsey model supplies its own objective and
        discount; any other needs ``objective=`` (MAXIMISED, current-period
        variables) and ``discount=``.
    steady : SteadyState
        The model's steady state; the welfare variable is added to it.
    cov : covariance of the shocks, as for :func:`dsgelab.solve_second_order`.
    initial : mapping, optional
        Levels of states at which conditional welfare is evaluated; the steady
        state for every state not listed.
    """
    from .secondorder import solve_second_order
    from .solve import solve
    from .symbolic import _require_sympy

    pre = getattr(model, "welfare", None)
    if pre is not None and objective is None and discount is None:
        # already augmented by welfare_model: reused as is, which is what makes a loop
        # over parameters (osr on welfare) cheap -- the symbolic work is done once
        sympy = _require_sympy()
        aug, name = model, pre["name"]
        U, beta = pre["objective"], pre["discount"]
        base_names = [v for v in model.names if v != name]
    else:
        sympy, U, beta = _objective(model, objective, discount)
        aug = welfare_model(model, U, beta, name=name)
        base_names = list(model.names)
    pnames = aug.symbolic["parameters"]
    pvals = {sympy.Symbol(p): float(get_param(params, p)) for p in pnames}
    point = {sympy.Symbol(v): float(steady.values[v]) for v in base_names}
    b = float(beta.xreplace(pvals))
    if not 0 < b < 1:
        raise WelfareError(f"the discount factor must lie in (0, 1); it is {b}")
    u_ss = float(U.xreplace({**pvals, **point}))
    values = {v: steady.values[v] for v in base_names}
    values[name] = u_ss / (1 - b)
    ss = steady_state(aug, values, params)
    first = solve(aug, params, steady=ss, qz_criterium=qz_criterium)
    so = solve_second_order(first, cov)
    coords = first.coords
    j = aug.jumps.index(name)
    pos = aug.position(name)

    def at(k_levels):
        k_hat = np.array([(k_levels.get(s, ss.values[s]) - coords.center[aug.position(s)])
                          / coords.scale[aug.position(s)] for s in aug.states])
        return float(coords.center[pos] + coords.scale[pos] * so.jumps(k_hat)[j])

    start = dict(initial or {})
    conditional = at(start)
    zero = None
    info = getattr(aug, "ramsey", None)
    if info is not None:
        zero = at({**start, **{lm: 0.0 for lm in info["lagged_multipliers"]}})
    unconditional = float(so.ergodic_mean(cov)[name])
    return Welfare(unconditional=unconditional, conditional=conditional,
                   deterministic=u_ss / (1 - b), conditional_zero_multipliers=zero,
                   discount=b, model=aug, second_order=so)


def consumption_equivalent(objective: str, variable: str, target: float, reference: float,
                           *, model: Model, params: Params, steady: SteadyState,
                           cov: Covariance, discount: str | float | sympy.Expr,
                           bracket: tuple[float, float] = (-0.5, 0.5), **kwargs: Any) -> float:
    """The permanent proportional change in ``variable`` that equates two welfare levels.

    Solves ``W(lambda) = target`` where ``W(lambda)`` is the unconditional
    welfare of ``reference``'s economy with ``variable`` replaced by
    ``(1 + lambda) * variable`` inside the objective. ``reference`` is that
    economy's own unconditional welfare and serves as a check: ``lambda = 0``
    must reproduce it. Positive ``lambda`` means the reference economy needs
    more of ``variable`` to reach ``target``.
    """
    from scipy.optimize import brentq

    from .symbolic import _require_sympy

    sympy = _require_sympy()
    x = sympy.Symbol(variable)
    lam = sympy.Symbol("__lambda__")

    def W(lmb):
        from .symbolic import _namespace, _parse_one
        ns = _namespace(sympy, model.names, model.shocks, model.symbolic["parameters"])
        U = _parse_one(sympy, objective, ns, "objective").xreplace({x: (1 + lam) * x})
        U = U.xreplace({lam: lmb})
        return welfare(model, params, steady=steady, cov=cov, objective=U,
                       discount=discount, **kwargs).unconditional

    base = W(0.0)
    if abs(base - reference) > 1e-8 * max(1.0, abs(reference)):
        raise WelfareError(f"lambda = 0 gives {base}, not the reference {reference}")
    return float(brentq(lambda l: W(l) - target, *bracket, xtol=1e-14))
