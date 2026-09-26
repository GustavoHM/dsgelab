"""Declare a model as equations in text, and get exact derivatives with it.

Optional: requires ``sympy`` (``pip install dsgekit[symbolic]``).

Syntax
------
``x`` is the variable at ``t``; ``x(+1)`` is the same variable at ``t+1``; shock
names stand for the innovation dated ``t+1``. Equations may be written with an
``=`` or as an expression that must vanish. A lag is expressed the way it is in
any state-space model -- by declaring the lagged value as a state -- rather than
by a ``(-1)`` operator, so that what is predetermined is stated once, in the
variable lists, and cannot silently disagree with the equations.

What this buys beyond convenience: the residual, its Jacobian and its Hessian
are all differentiated EXACTLY. At second order that is not a nicety. A second
difference divides by the square of the step, so a finite-difference Hessian
carries roughly half the significant digits of the residual it came from, and
the risk corrections are precisely the small terms that loses.
"""

from __future__ import annotations

import re
from typing import Sequence

import numpy as np

from .model import Model, get_param

__all__ = ["from_equations", "SymbolicError"]

_LEAD = re.compile(r"\b([A-Za-z_]\w*)\s*\(\s*\+?\s*1\s*\)")
_LAG = re.compile(r"\b([A-Za-z_]\w*)\s*\(\s*-\s*1\s*\)")


class SymbolicError(ValueError):
    """Raised for a malformed equation, an unknown name or a missing dependency."""


def _require_sympy():
    try:
        import sympy  # noqa: F401
    except ImportError as exc:      # pragma: no cover - environment-dependent
        raise SymbolicError(
            "the symbolic front end needs sympy: pip install 'dsgekit[symbolic]'") from exc
    import sympy
    return sympy


def from_equations(equations: Sequence[str], *, states: Sequence[str],
                   jumps: Sequence[str], shocks: Sequence[str],
                   parameters: Sequence[str], name: str = "model",
                   derivatives: str = "both") -> Model:
    """Build a :class:`~dsgekit.model.Model` from text equations.

    Parameters
    ----------
    equations : sequence of str
        One per variable. ``"lhs = rhs"`` is read as ``lhs - rhs``.
    states, jumps, shocks, parameters : sequence of str
        The declared names. Anything appearing in an equation and not declared
        raises rather than becoming a free symbol -- a typo must not quietly
        turn into a new variable.
    derivatives : {"both", "jacobian", "none"}
        Which exact derivatives to build. The Hessian costs
        ``n_eq * (2n + n_shocks)^2`` expressions to differentiate and lambdify,
        so it is worth skipping for a large model that will only be solved to
        first order.
    """
    sympy = _require_sympy()
    from sympy.parsing.sympy_parser import parse_expr, standard_transformations

    states, jumps = list(states), list(jumps)
    shocks, parameters = list(shocks), list(parameters)
    var_names = states + jumps
    n = len(var_names)
    if len(equations) != n:
        raise SymbolicError(
            f"{len(equations)} equations for {n} variables "
            f"({len(states)} states + {len(jumps)} jumps)")

    clash = set(var_names) & set(parameters)
    if clash:
        raise SymbolicError(f"names used as both variable and parameter: {sorted(clash)}")

    cur = {v: sympy.Symbol(v) for v in var_names}
    nxt = {v: sympy.Symbol(f"{v}__next") for v in var_names}
    eps = {s: sympy.Symbol(s) for s in shocks}
    par = {p: sympy.Symbol(p) for p in parameters}

    allowed = {}
    allowed.update({v: cur[v] for v in var_names})
    allowed.update({f"{v}__next": nxt[v] for v in var_names})
    allowed.update(eps)
    allowed.update(par)
    functions = {"log": sympy.log, "exp": sympy.exp, "sqrt": sympy.sqrt,
                 "Abs": sympy.Abs, "abs": sympy.Abs, "sin": sympy.sin,
                 "cos": sympy.cos, "tan": sympy.tan, "Min": sympy.Min,
                 "Max": sympy.Max, "min": sympy.Min, "max": sympy.Max}
    local_dict = dict(allowed)
    local_dict.update(functions)
    known = set(allowed.values())
    # The parser's own transformations emit these constructors, so they must
    # resolve; everything else is deliberately absent from the global namespace.
    # An undeclared name therefore becomes a free Symbol rather than silently
    # binding to a sympy function of the same name -- `gamma`, `beta` and `S`
    # are ordinary parameter names in economics and all three are sympy objects.
    global_dict = {"Symbol": sympy.Symbol, "Integer": sympy.Integer,
                   "Float": sympy.Float, "Rational": sympy.Rational,
                   "factorial": sympy.factorial}

    exprs = []
    for i, raw in enumerate(equations):
        text = _LAG.sub(r"\1__LAG_NOT_SUPPORTED", str(raw))
        if "__LAG_NOT_SUPPORTED" in text:
            raise SymbolicError(
                f"equation {i}: a (-1) lag is not accepted. Declare the lagged value as a "
                f"state and add its own transition equation, so that what is predetermined "
                f"is stated in one place.")
        text = _LEAD.sub(r"\1__next", text)
        if text.count("=") > 1:
            raise SymbolicError(f"equation {i}: more than one '='")
        if "=" in text:
            lhs, rhs = text.split("=")
            text = f"({lhs}) - ({rhs})"
        try:
            expr = parse_expr(text, local_dict=local_dict, global_dict=global_dict,
                              transformations=standard_transformations)
        except Exception as exc:
            raise SymbolicError(f"equation {i}: cannot parse {raw!r} ({exc})") from exc
        unknown = {s for s in expr.free_symbols if s not in known}
        if unknown:
            raise SymbolicError(
                f"equation {i}: undeclared name(s) {sorted(str(u) for u in unknown)} in "
                f"{raw!r}. Declare them as variables, shocks or parameters.")
        exprs.append(expr)

    w_syms = ([nxt[v] for v in var_names] + [cur[v] for v in var_names]
              + [eps[s] for s in shocks])
    all_syms = w_syms + [par[p] for p in parameters]
    m = len(w_syms)

    f_res = sympy.lambdify(all_syms, sympy.Matrix(exprs), modules="numpy")

    def _args(x_next, x_now, e, params):
        return (list(np.asarray(x_next, dtype=float))
                + list(np.asarray(x_now, dtype=float))
                + list(np.asarray(e, dtype=float))
                + [float(get_param(params, p)) for p in parameters])

    def residual(nv, cv, ev, params):
        vals = ([nv[v] for v in var_names] + [cv[v] for v in var_names]
                + [ev[s] for s in shocks]
                + [float(get_param(params, p)) for p in parameters])
        return np.asarray(f_res(*vals), dtype=float).ravel()

    jac_fn = hess_fn = None
    if derivatives in ("jacobian", "both"):
        Jm = sympy.Matrix(exprs).jacobian(sympy.Matrix(w_syms))
        f_jac = sympy.lambdify(all_syms, Jm, modules="numpy")

        def jac_fn(x_next, x_now, e, params, ctx=None):
            return np.asarray(f_jac(*_args(x_next, x_now, e, params)),
                              dtype=float).reshape(n, m)

    if derivatives == "both":
        blocks = [sympy.hessian(expr, w_syms) for expr in exprs]
        f_hess = sympy.lambdify(all_syms, blocks, modules="numpy")

        def hess_fn(x_next, x_now, e, params, ctx=None):
            raw = f_hess(*_args(x_next, x_now, e, params))
            return np.array([np.asarray(b, dtype=float).reshape(m, m) for b in raw])

    elif derivatives not in ("jacobian", "none"):
        raise SymbolicError(f"derivatives must be 'both', 'jacobian' or 'none', "
                            f"got {derivatives!r}")

    model = Model(states=states, jumps=jumps, shocks=shocks, equations=residual,
                  name=name, jacobian_fn=jac_fn, hessian_fn=hess_fn)
    model.symbolic = dict(equations=list(equations), expressions=exprs,
                          parameters=parameters, symbols=dict(cur=cur, nxt=nxt, eps=eps))
    return model
