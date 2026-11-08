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
are differentiated EXACTLY, with an opt-in exact third derivative. At second order that is not a nicety. A second
difference divides by the square of the step, so a finite-difference Hessian
carries roughly half the significant digits of the residual it came from, and
the risk corrections are precisely the small terms that loses.
"""

from __future__ import annotations

import re
from itertools import combinations_with_replacement, permutations
from typing import Mapping, Sequence

import numpy as np

from .model import Model, get_param
from .tensors import SparseSecondDerivatives, SparseThirdDerivatives

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

    jac_fn = hess_fn = third_fn = None
    if derivatives in ("jacobian", "both", "third"):
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

    if derivatives == "third":
        hessian_equations = []
        hessian_indices = []
        hessian_blocks = []
        for equation, expr in enumerate(exprs):
            for i, j in combinations_with_replacement(range(m), 2):
                derivative = expr.diff(w_syms[i], w_syms[j])
                if derivative == 0:
                    continue
                for index in sorted(set(permutations((i, j)))):
                    hessian_equations.append(equation)
                    hessian_indices.append(index)
                    hessian_blocks.append(derivative)
        hessian_equation_array: np.ndarray = np.asarray(hessian_equations, dtype=np.int64)
        hessian_index_array: np.ndarray = np.asarray(hessian_indices, dtype=np.int64).reshape(-1, 2)
        f_hess = sympy.lambdify(all_syms, hessian_blocks, modules="numpy")

        def hess_fn(x_next, x_now, e, params, ctx=None):
            values = np.asarray(f_hess(*_args(x_next, x_now, e, params)),
                                dtype=float).reshape(-1)
            return SparseSecondDerivatives(hessian_equation_array, hessian_index_array,
                                           values, (n, m, m))

    if derivatives == "third":
        equation_indices = []
        derivative_indices = []
        blocks = []
        for equation, expr in enumerate(exprs):
            for i, j, k in combinations_with_replacement(range(m), 3):
                derivative = expr.diff(w_syms[i], w_syms[j], w_syms[k])
                if derivative == 0:
                    continue
                for index in sorted(set(permutations((i, j, k)))):
                    equation_indices.append(equation)
                    derivative_indices.append(index)
                    blocks.append(derivative)
        equation_index_array: np.ndarray = np.asarray(equation_indices, dtype=np.int64)
        derivative_index_array: np.ndarray = np.asarray(
            derivative_indices, dtype=np.int64).reshape(-1, 3)
        f_third = sympy.lambdify(all_syms, blocks, modules="numpy")

        def third_fn(x_next, x_now, e, params, ctx=None):
            values = np.asarray(f_third(*_args(x_next, x_now, e, params)),
                                dtype=float).reshape(-1)
            return SparseThirdDerivatives(equation_index_array, derivative_index_array, values,
                                          (n, m, m, m))

    elif derivatives not in ("both", "jacobian", "none"):
        raise SymbolicError(f"derivatives must be 'third', 'both', 'jacobian' or 'none', "
                            f"got {derivatives!r}")

    model = Model(states=states, jumps=jumps, shocks=shocks, equations=residual,
                  name=name, jacobian_fn=jac_fn, hessian_fn=hess_fn, third_fn=third_fn)
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
        copy = v in states and max_lead.get(v, 0) > 0
        if copy:
            jumps.append(f"LEAD0_{v}")
            aux[f"LEAD0_{v}"] = ("lead", v)
            out.append(f"LEAD0_{v} = {v}")
        for k in range(1, max_lead.get(v, 0) + 1):
            n = f"LEAD{k}_{v}"
            prev = (f"LEAD0_{v}" if copy else v) if k == 1 else f"LEAD{k - 1}_{v}"
            jumps.append(n)
            aux[n] = ("lead", v)
            out.append(f"{n} = {prev}(+1)")
    clash = [a for a in aux if a in variables]
    if clash:
        raise SymbolicError(f"auxiliary names collide with declared variables: {clash}")
    return out, states, jumps, aux


_TOKEN = re.compile(r"\b([A-Za-z_]\w*)(?:\s*\(\s*([+-]?)\s*(\d+)\s*\))?")


def _dated(name, off):
    return name if off == 0 else f"{name}({off:+d})"


def _on_dated(text, names, fn):
    """Replace each ``name`` or ``name(k)`` of ``names`` in ``text`` by ``fn(name, k)``."""
    def sub(m):
        if m.group(1) not in names:
            return m.group(0)
        off = int(m.group(3) or 0)
        return fn(m.group(1), -off if m.group(2) == "-" else off)
    return _TOKEN.sub(sub, str(text))


def _names_in(text):
    return {m.group(1) for m in _TOKEN.finditer(str(text))}


def _trend_declarations(variables, shocks, parameters, trend_vars, log_trend_vars,
                        deflators, log_deflators):
    """Check the declared trends; return ``{trend: (kind, g)}`` and ``{var: (kind, D)}``."""
    trends, defl = {}, {}
    for kind, given in (("mul", trend_vars), ("log", log_trend_vars)):
        for a, g in dict(given or {}).items():
            if a in trends:
                raise SymbolicError(f"trend variable {a!r} declared twice")
            if a in variables or a in shocks or a in parameters:
                raise SymbolicError(f"trend variable {a!r} is also declared as a variable, "
                                    f"a shock or a parameter")
            trends[a] = (kind, str(g))
    for kind, given in (("mul", deflators), ("log", log_deflators)):
        for v, d in dict(given or {}).items():
            if v in defl:
                raise SymbolicError(f"deflator declared twice for {v!r}")
            if v not in variables:
                raise SymbolicError(f"deflator declared for {v!r}, which is not a state or "
                                    f"a jump")
            defl[v] = (kind, str(d))
    for v, (_, d) in defl.items():
        bad = _names_in(d) & (set(shocks) | set(defl))
        if bad:
            raise SymbolicError(f"the deflator of {v} may not contain the shock or deflated "
                                f"variable(s) {sorted(bad)}")
    for a, (_, g) in trends.items():
        bad = _names_in(g) & (set(shocks) | set(defl) | set(trends))
        if bad:
            raise SymbolicError(f"the growth factor of {a} may not contain the shock or "
                                f"trended name(s) {sorted(bad)}")
    return trends, defl


def _detrend(equations, variables, trends, defl):
    """Dynare's detrending, on the text of the equations.

    Each ``x(k)`` of a deflated variable becomes ``x(k) * D(k)`` (``x(k) + D(k)``
    with a log deflator). Each ``A(k)`` of a trend then becomes its level at ``t``
    times the growth since, ``A(k) = A g(+1) ... g(+k)`` and
    ``A(-k) = A / (g g(-1) ... g(-k+1))``, with sums for a log trend. The level is
    the name ``__TREND__A``, which :func:`from_equations` sets to 1 (0 for a log
    trend) once the balanced-growth check has passed.
    """
    dated = set(variables) | set(trends)

    def shift(text, k):
        return _on_dated(text, dated, lambda n, o: _dated(n, o + k))

    def deflate(name, off):
        kind, d = defl[name]
        return f"({_dated(name, off)}{'*' if kind == 'mul' else '+'}({shift(d, off)}))"

    def trend_at(a, off):
        kind, g = trends[a]
        up, down = ("*", "/") if kind == "mul" else ("+", "-")
        out = f"__TREND__{a}"
        for j in range(1, off + 1):
            out += f"{up}({shift(g, j)})"
        for j in range(0, -off):
            out += f"{down}({shift(g, -j)})"
        return f"({out})"

    return [_on_dated(_on_dated(e, set(defl), deflate), set(trends), trend_at)
            for e in equations]


def from_equations(equations: Sequence[str], *, states: Sequence[str],
                   jumps: Sequence[str], shocks: Sequence[str],
                   parameters: Sequence[str], name: str = "model",
                   derivatives: str = "both",
                   steady_state_model: Sequence[str] | None = None,
                   calibration_equations: dict | None = None,
                   trend_vars: Mapping[str, str] | None = None,
                   log_trend_vars: Mapping[str, str] | None = None,
                   deflators: Mapping[str, str] | None = None,
                   log_deflators: Mapping[str, str] | None = None) -> Model:
    """Build a :class:`~dsgelab.model.Model` from text equations.

    Parameters
    ----------
    equations : sequence of str
        One per declared variable. ``"lhs = rhs"`` is read as ``lhs - rhs``.
        ``x(+1)`` is the next period. A lag ``x(-k)`` or a lead ``x(+k)`` with
        ``k >= 2`` is accepted and carried by auxiliary variables, each with its
        own equation and listed in ``model.auxiliary``: a lag ``x(-k)`` by the
        states ``LAG1_x`` ... ``LAGk_x``, a lead ``x(+k)`` by the jumps
        ``LEAD1_x`` ... ``LEAD{k-1}_x``, so that ``y(+2)`` becomes ``LEAD1_y(+1)``. What is predetermined is still
        declared in one place: the lag of a jump is a new state with its own
        transition, never a reclassification of the jump.
    states, jumps, shocks, parameters : sequence of str
        The declared names. Anything appearing in an equation and not declared
        raises rather than becoming a free symbol -- a typo must not quietly
        turn into a new variable.
    derivatives : {"third", "both", "jacobian", "none"}
        Which exact derivatives to build. ``"third"`` includes the Jacobian,
        Hessian and third-derivative tensor. The Hessian costs
        ``n_eq * (2n + n_shocks)^2`` expressions to differentiate and lambdify,
        and the third derivative costs ``n_eq * (2n + n_shocks)^3``. The default
        ``"both"`` does not build that tensor.
    steady_state_model : sequence of str, optional
        The steady state in closed form, as assignments ``"name = expr"``
        evaluated in order -- Dynare's ``steady_state_model`` block. Intermediate
        names are allowed; every variable must end up with a value. It is
        attached as ``model.steady_state_model`` and used by
        ``dsgelab.steady_state(model, None, params)``, which CHECKS it against
        the equations and refuses it if it does not solve them.
    calibration_equations : mapping, optional
        ``{parameter: "steady-state condition"}``, attached as
        ``model.calibration_equations`` and solved with the steady state by
        :func:`dsgelab.steady_state` (see its ``calibrate`` argument).
    trend_vars, log_trend_vars : mapping, optional
        ``{trend: "growth factor"}``, the ``trend_var(growth_factor=...)`` and
        ``log_trend_var(log_growth_factor=...)`` of a ``.mod``. A trend is not a
        state or a jump and has no equation: ``A(+1) = g(+1) A`` (``A(+1) = A +
        g(+1)`` for a log trend) is implied. The growth factor may contain
        variables and parameters, but no shock and nothing trended.
    deflators, log_deflators : mapping, optional
        ``{variable: "deflator"}``, the ``var(deflator=...)`` and
        ``var(log_deflator=...)`` of a ``.mod``. The equations are written in
        levels, and the declared variable is the detrended one: every ``x(k)`` is
        read as ``x(k) * D(k)`` (``x(k) + D(k)``), and each trend as its level
        times the growth since ``t``. Each equation must then depend on the
        level of a trend only through a common factor, which is checked at
        random points, as Dynare checks it, and refused with a
        :class:`SymbolicError` otherwise. The model returned is the detrended
        one, with the declarations in ``model.trends``, as in ``ModFile.trends``.
        The parameters have no values at this point, so an equation balanced
        only for some values of its parameters is refused.
    """
    states, jumps = list(states), list(jumps)
    n = len(states) + len(jumps)
    if len(equations) != n:
        raise SymbolicError(
            f"{len(equations)} equations for {n} variables "
            f"({len(states)} states + {len(jumps)} jumps)")
    trends, defl = _trend_declarations(states + jumps, list(shocks), list(parameters),
                                       trend_vars, log_trend_vars, deflators, log_deflators)
    written = list(equations)
    if trends or defl:
        equations = _detrend(written, states + jumps, trends, defl)
    levels = [f"__TREND__{a}" for a in trends]
    eqs, states_x, jumps_x, aux = _expand_dating(equations, states, jumps, list(shocks))
    parsed = parse_system(eqs, variables=states_x + jumps_x, shocks=shocks,
                          parameters=list(parameters) + levels)
    exprs = parsed["expressions"]
    if trends:
        from .bgp import _check_balanced
        sympy = _require_sympy()
        level = {a: parsed["par"][f"__TREND__{a}"] for a in trends}
        start = {a: 1 if kind == "mul" else 0 for a, (kind, _) in trends.items()}
        # the parameters have no values here: one draw, held at every point of the check
        rng = np.random.default_rng(0)
        draw = {p: float(rng.uniform(0.6, 1.4)) for p in parameters}
        for k, e in enumerate(exprs[:n]):
            _check_balanced(sympy, e, level, trends, written[k], k, draw, 1e-8,
                            error=SymbolicError)
        exprs = [e.xreplace({level[a]: sympy.Integer(start[a]) for a in trends})
                 for e in exprs]
        for a in trends:
            eqs = [re.sub(rf"\b__TREND__{a}\b", str(start[a]), e) for e in eqs]
    model = model_from_expressions(exprs, states=states_x, jumps=jumps_x,
                                   shocks=shocks, parameters=parameters, name=name,
                                   derivatives=derivatives, source_equations=eqs)
    model.auxiliary = aux
    model.trends = dict(deflators=dict(defl), trend_vars=dict(trends))
    if steady_state_model is not None:
        model.steady_state_model = _closed_form(list(steady_state_model), parameters)
    if calibration_equations:
        unknown = [p for p in calibration_equations if p not in parameters]
        if unknown:
            raise SymbolicError(f"calibration_equations for undeclared parameters {unknown}")
        model.calibration_equations = dict(calibration_equations)
    return model


def calibration_residual(model: Model, targets: dict, params):
    """``g(x, params)``: the residuals of steady-state conditions, one per target."""
    sympy = _require_sympy()
    from sympy.parsing.sympy_parser import parse_expr, standard_transformations
    pnames = list((getattr(model, "symbolic", None) or {}).get("parameters", [])) or (
        list(params) if isinstance(params, dict) else [])
    local = {v: sympy.Symbol(v) for v in model.names}
    local.update({q: sympy.Symbol(q) for q in pnames})
    local.update({"log": sympy.log, "exp": sympy.exp, "sqrt": sympy.sqrt})
    glob = {"Symbol": sympy.Symbol, "Integer": sympy.Integer, "Float": sympy.Float,
            "Rational": sympy.Rational}
    exprs = []
    for par, text in targets.items():
        t = text.replace("^", "**")
        if t.count("=") != 1:
            raise SymbolicError(f"calibration condition for {par}: expected one '='")
        lhs, rhs = t.split("=")
        e = parse_expr(f"({lhs}) - ({rhs})", local_dict=local, global_dict=glob,
                       transformations=standard_transformations)
        stray = sorted(str(x) for x in e.free_symbols if str(x) not in local)
        if stray:
            raise SymbolicError(f"calibration condition for {par}: undeclared {stray}")
        exprs.append(e)
    syms = [sympy.Symbol(v) for v in model.names] + [sympy.Symbol(q) for q in pnames]
    f = sympy.lambdify(syms, exprs, modules="numpy")

    def g(x, p):
        return np.asarray(f(*list(np.asarray(x, float)),
                            *[float(get_param(p, q)) for q in pnames]), float).ravel()

    return g


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
