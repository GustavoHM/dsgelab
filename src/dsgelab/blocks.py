"""Derive the first-order conditions from an agent's problem.

Optional: requires ``sympy`` (``pip install dsgelab[symbolic]``).

What it does, and what it does not
----------------------------------
It takes a declared recursive optimisation problem (the period objective, the choice variables,
the states and the constraints) and returns the FOCs as EQUATIONS IN TEXT, ready for
:func:`dsgelab.symbolic.from_equations`. The gain is not in typing: the algebra is no longer done
by hand. Deriving a Lagrangian with four constraints and three states by hand is where sign errors
and forgotten terms come in, and they do not show up in the steady state.

What it does NOT do is choose the problem for you. Transversality conditions, inequality
constraints, complementarity and non-standard envelopes remain the modeller's job: the package
derives what you declared, and the declaration is the modelling.

The mathematics
---------------
With ``y`` the controls, ``k`` the states and ``h_j`` the constraints written as ``= 0``, the
period Lagrangian is

    L_t = u(y_t, k_t) + SUM_j lm_j,t * h_j(y_t, k_t, k_{t+1})

and the conditions are

    dL_t/dy_t = 0                                    for each control
    dL_t/dk_{t+1} + beta * E_t[ dL_{t+1}/dk_{t+1} ] = 0    for each state

The second is the Euler equation, and its second term is the derivative of the NEXT period's
Lagrangian with respect to the state it inherits, obtained by shifting ``dL_t/dk_t`` one period
ahead. If that shift produces a term in ``t+2``, the problem cannot be represented in two periods
and the function refuses, naming the state that caused it, instead of returning wrong equations.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Mapping, Sequence

if TYPE_CHECKING:
    from .model import Model

__all__ = ["FOCError", "optimality_conditions", "Problem", "assemble"]

_LEAD = re.compile(r"\b([A-Za-z_]\w*)\s*\(\s*\+?\s*1\s*\)")


class FOCError(ValueError):
    """Raised for a malformed problem, an undeclared name or a two-period-ahead term."""


def _require_sympy():
    try:
        import sympy  # noqa: F401
    except ImportError as exc:      # pragma: no cover
        raise FOCError("deriving FOCs requires sympy: "
                       "pip install 'dsgelab[symbolic]'") from exc
    import sympy
    return sympy


def _parse(text, local_dict, global_dict, label):
    from sympy.parsing.sympy_parser import parse_expr, standard_transformations
    t = _LEAD.sub(r"\1__next", str(text))
    if t.count("=") > 1:
        raise FOCError(f"{label}: more than one '='")
    if "=" in t:
        lhs, rhs = t.split("=")
        t = f"({lhs}) - ({rhs})"
    try:
        return parse_expr(t, local_dict=local_dict, global_dict=global_dict,
                          transformations=standard_transformations)
    except Exception as exc:
        raise FOCError(f"{label}: cannot parse {text!r} ({exc})") from exc


class Problem:
    """A declared recursive optimisation problem.

    Parameters
    ----------
    objective : str
        The PERIOD return, ``u(y_t, k_t)``. The discount factor enters through ``discount``, not
        here.
    controls : sequence of str
        Variables chosen at ``t`` that are not states.
    states : sequence of str
        Predetermined variables. The value chosen at ``t`` is ``k(+1)``.
    constraints : sequence of str
        Equality constraints, ``"lhs = rhs"`` or an expression that equals zero. Each one gets a
        multiplier.
    discount : str
        Name of the discount parameter.
    parameters : sequence of str
        Every name that is neither a variable nor a multiplier. An undeclared name raises an error
        instead of becoming a free symbol: a typo cannot turn into a new parameter.
    exogenous : sequence of str, optional
        Variables that enter the problem but the agent does NOT choose, such as productivity or
        prices taken as given. They produce no FOC.
    multipliers : sequence of str, optional
        Names of the multipliers, in the order of the constraints. When absent they are called
        ``lm_1``, ``lm_2``…
    """

    def __init__(self, objective: str, *, controls: Sequence[str], states: Sequence[str] = (),
                 constraints: Sequence[str] = (), discount: str = "beta",
                 parameters: Sequence[str] = (), exogenous: Sequence[str] = (),
                 multipliers: Sequence[str] | None = None, name: str = "agent"):
        self.objective = objective
        self.controls = list(controls)
        self.states = list(states)
        self.constraints = list(constraints)
        self.discount = discount
        self.parameters = list(parameters)
        self.exogenous = list(exogenous)
        self.name = name
        n = len(self.constraints)
        self.multipliers = (list(multipliers) if multipliers is not None
                            else [f"lm_{i + 1}" for i in range(n)])
        if len(self.multipliers) != n:
            raise FOCError(f"{n} constraints and {len(self.multipliers)} multipliers")
        all_names = self.controls + self.states + self.exogenous + self.multipliers
        dup = {v for v in all_names if all_names.count(v) > 1}
        if dup:
            raise FOCError(f"names repeated among variables and multipliers: {sorted(dup)}")

    def __repr__(self) -> str:
        return (f"Problem({self.name!r}, {len(self.controls)} controls, "
                f"{len(self.states)} states, {len(self.constraints)} constraints)")


def optimality_conditions(problem: Problem) -> dict:
    """Derive the FOCs. Returns text equations, ready for ``from_equations``.

    Returns
    -------
    dict with
        ``equations``   the FOCs plus the constraints, as strings;
        ``jumps``       controls and multipliers, the non-predetermined variables that the
                        problem introduces;
        ``states``      the states, in the declared order;
        ``multipliers`` the names used;
        ``detail``      a map from each equation to its origin (which control, which state, which
                        constraint), so that the result can be audited rather than being an opaque
                        block.
    """
    sympy = _require_sympy()
    p = problem

    variables = p.controls + p.states + p.exogenous + p.multipliers
    cur = {v: sympy.Symbol(v) for v in variables}
    nxt = {v: sympy.Symbol(f"{v}__next") for v in variables}
    nxt2 = {v: sympy.Symbol(f"{v}__next2") for v in variables}
    par = {q: sympy.Symbol(q) for q in list(p.parameters) + [p.discount]}

    local = {}
    local.update(cur)
    local.update({f"{v}__next": nxt[v] for v in variables})
    local.update(par)
    known = set(cur.values()) | set(nxt.values()) | set(par.values())
    functions = {"log": sympy.log, "exp": sympy.exp, "sqrt": sympy.sqrt, "Abs": sympy.Abs,
                 "sin": sympy.sin, "cos": sympy.cos, "Min": sympy.Min, "Max": sympy.Max}
    local_dict = dict(local); local_dict.update(functions)
    global_dict = {"Symbol": sympy.Symbol, "Integer": sympy.Integer,
                   "Float": sympy.Float, "Rational": sympy.Rational,
                   "factorial": sympy.factorial}

    def parse_checked(text, label):
        e = _parse(text, local_dict, global_dict, label)
        unknown = {s for s in e.free_symbols if s not in known}
        if unknown:
            raise FOCError(
                f"{label}: undeclared name(s) {sorted(str(s) for s in unknown)}. "
                f"Declare them in controls, states, exogenous or parameters.")
        return e

    u = parse_checked(p.objective, "objective")
    hs = [parse_checked(c, f"constraint {i + 1}") for i, c in enumerate(p.constraints)]

    # period Lagrangian
    L = u + sum(cur[m] * h for m, h in zip(p.multipliers, hs))

    def lead(expr):
        """Shift the expression one period ahead: x -> x(+1), x(+1) -> x(+2)."""
        return expr.xreplace({nxt[v]: nxt2[v] for v in variables}) \
                   .xreplace({cur[v]: nxt[v] for v in variables})

    printer = _plain_printer(sympy)
    equations, detail, jump_order = [], {}, []

    # (1) FOC for each control
    for c in p.controls:
        # No `simplify`: it sometimes puts the expression over a denominator that is a VARIABLE,
        # and a residual divided by k(+1) is equivalent only while k(+1) stays away from zero.
        # The raw derivative is exactly the FOC.
        foc = sympy.expand(sympy.diff(L, cur[c]))
        txt = printer(foc)
        equations.append(txt)
        detail[txt] = f"FOC for control {c}"

    # (2) Euler equation for each state
    for k in p.states:
        now = sympy.diff(L, nxt[k])                    # effect of choosing k(+1) today
        later = lead(sympy.diff(L, cur[k]))            # what k(+1) yields tomorrow
        forbidden = [v for v in variables if nxt2[v] in later.free_symbols]
        if forbidden:
            raise FOCError(
                f"the Euler equation for state {k!r} requires terms in t+2 ({sorted(forbidden)}): "
                f"the problem cannot be represented in two periods. Introduce the variable at t+1 "
                f"as a control with its own definition.")
        euler = sympy.expand(now + par[p.discount] * later)
        txt = printer(euler)
        equations.append(txt)
        detail[txt] = f"Euler equation for state {k}"

    # (3) the constraints, which are also equations of the model
    for i, (c, h) in enumerate(zip(p.constraints, hs), start=1):
        txt = printer(h)
        equations.append(txt)
        detail[txt] = f"constraint {i}: {c}"

    jump_order = list(p.controls) + list(p.multipliers)
    return dict(equations=equations, jumps=jump_order, states=list(p.states),
                multipliers=list(p.multipliers), detail=detail)


def _plain_printer(sympy):
    """Print in the syntax of ``from_equations``: ``x__next`` becomes ``x(+1)`` again."""
    from sympy.printing.str import StrPrinter
    pr = StrPrinter(settings={"order": "none"})

    def do(expr):
        s = pr.doprint(expr)
        return re.sub(r"\b([A-Za-z_]\w*)__next\b", r"\1(+1)", s)

    return do


def assemble(problems: Sequence[Problem], *, equations: Sequence[str] = (),
             states: Sequence[str] = (), jumps: Sequence[str] = (),
             shocks: Sequence[str] = (), parameters: Sequence[str] = (),
             name: str = "model", derivatives: str = "both") -> Model:
    """Combine several optimisation problems and the closing equations into one ``Model``.

    Each problem contributes its FOCs, its constraints and its multipliers
    (:func:`optimality_conditions`). The closure -- market clearing, exogenous
    processes, definitions -- is the modeller's text, in ``equations``, and the
    variables that only the closure introduces (prices, aggregate shocks) are
    declared in ``states`` and ``jumps``. This is not a block language: the limits
    are those of :class:`Problem` (equality constraints, nothing in ``t+2``).

    What is checked before assembling: a variable chosen by two agents is refused
    (a price that both take as given is ``exogenous`` to both); multipliers whose
    default name is repeated across agents get the agent's name as a prefix
    (``household_lm_1``); and the number of equations must equal the number of
    variables, with the count per block in the message when it does not. The
    origin of each equation is kept in ``model.assembly``.
    """
    from .symbolic import from_equations

    chosen: dict = {}
    for p in problems:
        for v in p.controls + p.states:
            if v in chosen:
                raise FOCError(f"{v!r} is chosen by two agents, {chosen[v]!r} and {p.name!r}: "
                               f"one of them must take it as given (exogenous)")
            chosen[v] = p.name
    names = [p.name for p in problems]
    if len(set(names)) != len(names):
        raise FOCError(f"two problems share a name: {names}")
    all_mult = [m for p in problems for m in p.multipliers]
    clash = {m for m in all_mult if all_mult.count(m) > 1}
    renamed = {}
    eqs, st, jp, origin = [], [], [], []
    for p in problems:
        if clash & set(p.multipliers):
            new = [f"{p.name}_{m}" if m in clash else m for m in p.multipliers]
            renamed.update({f"{p.name}_{m}": (p.name, m) for m in p.multipliers if m in clash})
            p = Problem(p.objective, controls=p.controls, states=p.states,
                        constraints=p.constraints, discount=p.discount,
                        parameters=p.parameters, exogenous=p.exogenous, multipliers=new,
                        name=p.name)
        foc = optimality_conditions(p)
        for e in foc["equations"]:
            eqs.append(e)
            origin.append((p.name, foc["detail"].get(e, "")))
        st += [s for s in foc["states"] if s not in st]
        jp += [j for j in foc["jumps"] if j not in jp]
    for e in equations:
        eqs.append(e)
        origin.append(("closure", e))
    st += [s for s in states if s not in st]
    jp += [j for j in jumps if j not in jp and j not in st]
    n = len(st) + len(jp)
    if len(eqs) != n:
        counts = {nm: sum(1 for o in origin if o[0] == nm) for nm in names + ["closure"]}
        raise FOCError(f"{len(eqs)} equations for {n} variables ({len(st)} states, "
                       f"{len(jp)} jumps); equations by block: {counts}")
    params = sorted(set(parameters) | {q for p in problems for q in p.parameters}
                    | {p.discount for p in problems})
    model = from_equations(eqs, states=st, jumps=jp, shocks=list(shocks), parameters=params,
                           name=name, derivatives=derivatives)
    model.assembly = dict(origin=origin, renamed_multipliers=renamed, agents=names)
    return model
