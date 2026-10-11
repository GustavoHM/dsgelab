"""Declare a model as equations in text, and get exact derivatives with it.

Optional: requires ``sympy`` (``pip install dsgelab[symbolic]``).

Syntax
------
``x`` is the variable at ``t``; ``x(+1)`` is the same variable at ``t+1``; shock
names stand for the innovation dated ``t+1``. Equations may be written with an
``=`` or as an expression that must vanish. :func:`from_equations` also accepts
lags ``x(-k)`` and leads ``x(+k)``, carried by auxiliary variables it lists in
``model.auxiliary``; what is predetermined is still stated once, in the variable
lists.

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

__all__ = ["from_equations", "parse_system", "model_from_expressions", "SymbolicError"]

_LEAD = re.compile(r"\b([A-Za-z_]\w*)\s*\(\s*\+?\s*1\s*\)")
_LAG = re.compile(r"\b([A-Za-z_]\w*)\s*\(\s*-\s*1\s*\)")


class SymbolicError(ValueError):
    """Raised for a malformed equation, an unknown name or a missing dependency."""


def _require_sympy():
    try:
        import sympy  # noqa: F401
    except ImportError as exc:      # pragma: no cover - environment-dependent
        raise SymbolicError(
            "the symbolic front end needs sympy: pip install 'dsgelab[symbolic]'") from exc
    import sympy
    return sympy


def _namespace(sympy, var_names, shocks, parameters):
    """Symbols and parser dictionaries for a set of declared names."""
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
    # The parser's own transformations emit these constructors, so they must
    # resolve; everything else is deliberately absent from the global namespace.
    # An undeclared name therefore becomes a free Symbol rather than silently
    # binding to a sympy function of the same name -- `gamma`, `beta` and `S`
    # are ordinary parameter names in economics and all three are sympy objects.
    global_dict = {"Symbol": sympy.Symbol, "Integer": sympy.Integer,
                   "Float": sympy.Float, "Rational": sympy.Rational,
                   "factorial": sympy.factorial}
    return dict(cur=cur, nxt=nxt, eps=eps, par=par, local=local_dict,
                glob=global_dict, known=set(allowed.values()))


def _parse_one(sympy, raw, ns, label):
    """Parse one equation (or expression) into a sympy expression."""
    from sympy.parsing.sympy_parser import parse_expr, standard_transformations
    text = _LAG.sub(r"\1__LAG_NOT_SUPPORTED", str(raw))
    if "__LAG_NOT_SUPPORTED" in text:
        raise SymbolicError(
            f"{label}: a (-1) lag is not accepted. Declare the lagged value as a "
            f"state and add its own transition equation, so that what is predetermined "
            f"is stated in one place.")
    text = _LEAD.sub(r"\1__next", text)
    if text.count("=") > 1:
        raise SymbolicError(f"{label}: more than one '='")
    if "=" in text:
        lhs, rhs = text.split("=")
        text = f"({lhs}) - ({rhs})"
    try:
        expr = parse_expr(text, local_dict=ns["local"], global_dict=ns["glob"],
                          transformations=standard_transformations)
    except Exception as exc:
        raise SymbolicError(f"{label}: cannot parse {raw!r} ({exc})") from exc
    unknown = {s for s in expr.free_symbols if s not in ns["known"]}
    if unknown:
        raise SymbolicError(
            f"{label}: undeclared name(s) {sorted(str(u) for u in unknown)} in "
            f"{raw!r}. Declare them as variables, shocks or parameters.")
    return expr


def parse_system(equations: Sequence[str], *, variables: Sequence[str],
                 shocks: Sequence[str], parameters: Sequence[str]) -> dict:
    """Parse text equations WITHOUT requiring one equation per variable.

    The building block shared by :func:`from_equations` and by the optimal-policy
    module, whose private-sector block has fewer equations than variables -- the
    policy instruments are left free for the planner to choose.

    Returns a dict with the sympy ``expressions`` and the symbol tables
    ``cur``, ``nxt``, ``eps`` and ``par``.
    """
    sympy = _require_sympy()
    variables, shocks, parameters = list(variables), list(shocks), list(parameters)
    clash = set(variables) & set(parameters)
    if clash:
        raise SymbolicError(f"names used as both variable and parameter: {sorted(clash)}")
    ns = _namespace(sympy, variables, shocks, parameters)
    exprs = [_parse_one(sympy, raw, ns, f"equation {i}") for i, raw in enumerate(equations)]
    return dict(expressions=exprs, cur=ns["cur"], nxt=ns["nxt"], eps=ns["eps"],
                par=ns["par"], namespace=ns)


def model_from_expressions(exprs, *, states: Sequence[str], jumps: Sequence[str],
                           shocks: Sequence[str], parameters: Sequence[str],
                           name: str = "model", derivatives: str = "both",
                           source_equations: Sequence[str] | None = None) -> Model:
    """Build a :class:`~dsgelab.model.Model` from sympy expressions already in hand.

    The expressions must use the naming convention of :func:`parse_system`:
    ``v`` for the variable at ``t``, ``v__next`` for ``t+1``, shock and parameter
    names as themselves. This is what lets code that DERIVES equations -- first-
    order conditions, a planner's problem -- hand them over without printing and
    re-parsing them.
    """
    sympy = _require_sympy()
    states, jumps = list(states), list(jumps)
    shocks, parameters = list(shocks), list(parameters)
    var_names = states + jumps
    n = len(var_names)
    if len(exprs) != n:
        raise SymbolicError(
            f"{len(exprs)} equations for {n} variables "
            f"({len(states)} states + {len(jumps)} jumps)")
    cur = {v: sympy.Symbol(v) for v in var_names}
    nxt = {v: sympy.Symbol(f"{v}__next") for v in var_names}
    eps = {s: sympy.Symbol(s) for s in shocks}
    par = {p: sympy.Symbol(p) for p in parameters}
    known = set(cur.values()) | set(nxt.values()) | set(eps.values()) | set(par.values())
    for i, e in enumerate(exprs):
        stray = {s for s in e.free_symbols if s not in known}
        if stray:
            raise SymbolicError(f"equation {i}: symbols outside the declared layout: "
                                f"{sorted(str(x) for x in stray)}")
    equations = (list(source_equations) if source_equations is not None
                 else [str(e) for e in exprs])

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
    model.symbolic = dict(equations=list(equations), expressions=list(exprs),
                          parameters=parameters, symbols=dict(cur=cur, nxt=nxt, eps=eps))
    return model


_DATED = re.compile(r"\b([A-Za-z_]\w*)\s*\(\s*([+-]?)\s*(\d+)\s*\)")


def _expand_dating(equations, states, jumps, shocks):
    """Rewrite lags ``x(-k)`` and leads ``x(+k)``, ``k >= 2``, through auxiliaries.

    ``x(-k)`` becomes the state ``LAGk_x`` with ``LAG1_x(+1) = x`` and
    ``LAGj_x(+1) = LAG{j-1}_x``; ``x(+k)`` becomes ``LEAD{k-1}_x(+1)`` with
    ``LEAD1_x = x(+1)`` and ``LEADj_x = LEAD{j-1}_x(+1)``. Returns the new
    equations, states, jumps and ``{auxiliary: (kind, of)}``.
    """
    variables = set(states) | set(jumps)
    max_lag, max_lead = {}, {}

    def sub(m):
        name, sign, k = m.group(1), m.group(2), int(m.group(3))
        if name in shocks:
            if sign == "-" or k > 1:
                raise SymbolicError(f"{m.group(0)}: an innovation is dated t+1 and takes no "
                                    f"lead or lag; carry it in a state")
            return m.group(0)
        if name not in variables:
            return m.group(0)
        if sign == "-":
            if k == 0:
                return name
            max_lag[name] = max(max_lag.get(name, 0), k)
            return f"LAG{k}_{name}"
        if k <= 1:
            return name if k == 0 else f"{name}(+1)"
        max_lead[name] = max(max_lead.get(name, 0), k - 1)
        return f"LEAD{k - 1}_{name}(+1)"

    out = [_DATED.sub(sub, str(e)) for e in equations]
    states, jumps = list(states), list(jumps)
    aux = {}
    for v in states + [j for j in jumps]:
        for k in range(1, max_lag.get(v, 0) + 1):
            n = f"LAG{k}_{v}"
            prev = v if k == 1 else f"LAG{k - 1}_{v}"
            states.append(n)
            aux[n] = ("lag", v)
            out.append(f"{n}(+1) = {prev}")
        for k in range(1, max_lead.get(v, 0) + 1):
            n = f"LEAD{k}_{v}"
            prev = v if k == 1 else f"LEAD{k - 1}_{v}"
            jumps.append(n)
            aux[n] = ("lead", v)
            out.append(f"{n} = {prev}(+1)")
    clash = [a for a in aux if a in variables]
    if clash:
        raise SymbolicError(f"auxiliary names collide with declared variables: {clash}")
    return out, states, jumps, aux


def from_equations(equations: Sequence[str], *, states: Sequence[str],
                   jumps: Sequence[str], shocks: Sequence[str],
                   parameters: Sequence[str], name: str = "model",
                   derivatives: str = "both",
                   steady_state_model: Sequence[str] | None = None) -> Model:
    """Build a :class:`~dsgelab.model.Model` from text equations.

    Parameters
    ----------
    equations : sequence of str
        One per declared variable. ``"lhs = rhs"`` is read as ``lhs - rhs``.
        ``x(+1)`` is the next period. A lag ``x(-k)`` or a lead ``x(+k)`` with
        ``k >= 2`` is accepted and carried by an auxiliary variable --
        ``LAGk_x``, a state, or ``LEADk_x``, a jump -- each with its own
        equation, listed in ``model.auxiliary``. What is predetermined is still
        declared in one place: the lag of a jump is a new state with its own
        transition, never a reclassification of the jump.
    states, jumps, shocks, parameters : sequence of str
        The declared names. Anything appearing in an equation and not declared
        raises rather than becoming a free symbol -- a typo must not quietly
        turn into a new variable.
    derivatives : {"both", "jacobian", "none"}
        Which exact derivatives to build. The Hessian costs
        ``n_eq * (2n + n_shocks)^2`` expressions to differentiate and lambdify,
        so it is worth skipping for a large model that will only be solved to
        first order.
    steady_state_model : sequence of str, optional
        The steady state in closed form, as assignments ``"name = expr"``
        evaluated in order -- Dynare's ``steady_state_model`` block. Intermediate
        names are allowed; every variable must end up with a value. It is
        attached as ``model.steady_state_model`` and used by
        ``dsgelab.steady_state(model, None, params)``, which CHECKS it against
        the equations and refuses it if it does not solve them.
    """
    states, jumps = list(states), list(jumps)
    n = len(states) + len(jumps)
    if len(equations) != n:
        raise SymbolicError(
            f"{len(equations)} equations for {n} variables "
            f"({len(states)} states + {len(jumps)} jumps)")
    eqs, states_x, jumps_x, aux = _expand_dating(equations, states, jumps, list(shocks))
    parsed = parse_system(eqs, variables=states_x + jumps_x, shocks=shocks,
                          parameters=parameters)
    model = model_from_expressions(parsed["expressions"], states=states_x, jumps=jumps_x,
                                   shocks=shocks, parameters=parameters, name=name,
                                   derivatives=derivatives, source_equations=eqs)
    model.auxiliary = aux
    if steady_state_model is not None:
        model.steady_state_model = _closed_form(list(steady_state_model), parameters)
    return model


def _closed_form(assignments, parameters):
    """``params -> {name: value}`` from assignments evaluated in order."""
    sympy = _require_sympy()
    from sympy.parsing.sympy_parser import parse_expr, standard_transformations
    funcs = {"log": sympy.log, "exp": sympy.exp, "sqrt": sympy.sqrt, "Abs": sympy.Abs,
             "abs": sympy.Abs, "min": sympy.Min, "max": sympy.Max}
    parsed = []
    known = set(parameters)
    for k, line in enumerate(assignments):
        if "=" not in line:
            raise SymbolicError(f"steady_state_model line {k + 1}: expected 'name = expr'")
        lhs, rhs = line.split("=", 1)
        lhs = lhs.strip()
        local = {n: sympy.Symbol(n) for n in known}
        local.update(funcs)
        expr = parse_expr(rhs.replace("^", "**"), local_dict=local,
                          global_dict={"Symbol": sympy.Symbol, "Integer": sympy.Integer,
                                       "Float": sympy.Float, "Rational": sympy.Rational},
                          transformations=standard_transformations)
        unknown = sorted(str(sym) for sym in expr.free_symbols if str(sym) not in known)
        if unknown:
            raise SymbolicError(f"steady_state_model line {k + 1}: {unknown} used before "
                                f"being assigned")
        parsed.append((lhs, expr))
        known.add(lhs)

    def evaluate(params):
        env = {sympy.Symbol(p): float(get_param(params, p)) for p in parameters}
        out = {}
        for lhs, expr in parsed:
            val = float(expr.xreplace(env))
            env[sympy.Symbol(lhs)] = val
            out[lhs] = val
        return out

    return evaluate
