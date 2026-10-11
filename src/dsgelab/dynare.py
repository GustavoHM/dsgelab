"""Read a Dynare ``.mod`` file into a :class:`~dsgelab.Model`.

Optional: requires ``sympy`` (``pip install dsgelab[symbolic]``).

What is read
------------
The declarations (``var``, ``varexo``, ``parameters``, ``predetermined_variables``),
the parameter assignments, the ``model`` block -- with model-local ``#``
variables, equation tags, ``model(linear)``, leads and lags of any order and the
``steady_state(x)`` operator --, ``steady_state_model``, ``initval`` and
``shocks``. The macro processor is covered for ``@#define``, ``@#if`` /
``@#ifdef`` / ``@#ifndef`` / ``@#else`` / ``@#endif`` and ``@{...}``
substitution. Every other statement -- ``stoch_simul``, ``estimation``,
``verbatim`` MATLAB, plots -- is skipped: this is an importer of MODELS, not an
interpreter of Dynare scripts. What cannot be read is refused with the line
that caused it, never approximated: ``@#include``, ``@#for``, ``varexo_det``,
leads of a shock, ``expectation``, ``diff``, ``adl``, external functions.

The translation
---------------
Dynare dates every variable at the period in which it is DETERMINED and lets an
equation reach any lead or lag. This package's form is two-period:
``f(x_{t+1}, x_t, eps_{t+1}) = 0``. The bridge is the textbook one, and every
variable it adds is listed in :attr:`ModFile.auxiliary` rather than created in
silence:

* every Dynare endogenous variable becomes a jump, and keeps its Dynare dating,
  so that its impulse response IS Dynare's series, with no shift;
* a lag ``v(-k)`` becomes a state ``LAG<k>_v``, with ``LAG1_v(+1) = v`` and
  ``LAG<k>_v(+1) = LAG<k-1>_v``;
* a lead ``v(+k)``, ``k >= 2``, becomes ``LEAD<k-1>_v(+1)``, with
  ``LEAD1_v = v(+1)`` and ``LEAD<k>_v = LEAD<k-1>_v(+1)``;
* a shock ``e``, dated ``t`` in Dynare, becomes the state ``SHK_e`` with
  ``SHK_e(+1) = e``, where ``e`` is now this package's innovation dated ``t+1``.
  This is what lets a shock appear anywhere, inside the production function as
  well as in an AR(1);
* ``predetermined_variables`` are re-dated as Dynare does: ``k(+1)`` becomes
  ``k``, ``k`` becomes ``k(-1)``.

The representation is not the smallest -- a hand-written model would make the
capital stock a state and fold the AR(1) into its shock -- but it is the one in
which the import can be checked line by line against the ``.mod``.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .model import Model
from .steady import SteadyState, steady_state

__all__ = ["DynareImportError", "ModFile", "from_mod"]


class DynareImportError(ValueError):
    """Raised for a ``.mod`` construct that is not read, with the offending text."""


# --------------------------------------------------------------------- text
def _strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    out = []
    for line in text.splitlines():
        if line.lstrip().startswith("@#"):
            out.append(line.split("//")[0])
            continue
        # a % or // starts a comment; neither occurs inside a model expression
        line = re.split(r"//|%", line, maxsplit=1)[0]
        out.append(line)
    return "\n".join(out)


def _macro_value(expr: str, env: dict):
    expr = expr.strip()
    py = (expr.replace("&&", " and ").replace("||", " or ")
          .replace("!=", " __NE__ ").replace("!", " not ").replace(" __NE__ ", "!="))
    py = re.sub(r"\btrue\b", "True", py)
    py = re.sub(r"\bfalse\b", "False", py)
    try:
        return eval(py, {"__builtins__": {}}, dict(env))      # names are macro values only
    except Exception as exc:
        raise DynareImportError(f"macro expression not understood: {expr!r} ({exc})") from exc


def _macro(text: str, defines: Mapping | None) -> str:
    env = dict(defines or {})
    fixed = set(env)             # names given by the caller win over @#define
    out, stack = [], []          # stack of (taking_this_branch, any_branch_taken)

    def active():
        return all(s[0] for s in stack)

    for line in text.splitlines():
        s = line.strip()
        if s.startswith("@#"):
            body = s[2:].strip()
            word = body.split(None, 1)[0] if body else ""
            rest = body[len(word):].strip()
            if word == "define":
                if active():
                    name, _, val = rest.partition("=")
                    if name.strip() not in fixed:
                        env[name.strip()] = _macro_value(val, env)
            elif word in ("if", "ifdef", "ifndef"):
                if word == "if":
                    cond = bool(_macro_value(rest, env)) if active() else False
                elif word == "ifdef":
                    cond = rest in env
                else:
                    cond = rest not in env
                stack.append([cond, cond])
            elif word == "elseif":
                if not stack:
                    raise DynareImportError("@#elseif without @#if")
                take = (not stack[-1][1]) and bool(_macro_value(rest, env))
                stack[-1] = [take, stack[-1][1] or take]
            elif word == "else":
                if not stack:
                    raise DynareImportError("@#else without @#if")
                stack[-1] = [not stack[-1][1], True]
            elif word == "endif":
                if not stack:
                    raise DynareImportError("@#endif without @#if")
                stack.pop()
            elif word == "echo" or word == "":
                pass
            else:
                raise DynareImportError(f"macro directive not supported: {s!r}")
            continue
        if not active():
            continue
        if "@{" in line:
            line = re.sub(r"@\{([^}]*)\}", lambda m: str(_macro_value(m.group(1), env)), line)
        out.append(line)
    if stack:
        raise DynareImportError("@#if without @#endif")
    return "\n".join(out)


def _statements(text: str) -> list[str]:
    return [s.strip() for s in text.split(";") if s.strip()]


_BLOCKS = {"model", "initval", "endval", "histval", "steady_state_model", "shocks", "mshocks",
           "estimated_params", "estimated_params_init", "estimated_params_bounds",
           "estimated_params_remove", "observation_trends", "optim_weights", "homotopy_setup",
           "conditional_forecast_paths", "irf_calibration", "moment_calibration", "verbatim",
           "filter_initial_state", "matched_moments", "deterministic_trends",
           "occbin_constraints", "model_replace", "model_remove", "perfect_foresight_controlled_paths",
           "shock_groups", "init2shocks", "generate_irfs", "epilogue", "pac_target_info",
           "heteroskedastic_shocks", "steady_state_model", "planner_objective_block"}


def _head(stmt: str) -> tuple[str, str]:
    m = re.match(r"([A-Za-z_]\w*)\s*(\(([^)]*)\))?\s*$", stmt)
    if m:
        return m.group(1), (m.group(3) or "")
    return "", ""


def _drop_groups(text: str) -> str:
    """Remove parenthesised attribute lists, with nesting and quoted strings inside."""
    out, depth, quote = [], 0, None
    for ch in text:
        if quote:
            if ch == quote:
                quote = None
            continue
        if depth and ch in "'\"":
            quote = ch
            continue
        if ch == "(":
            depth += 1
            continue
        if ch == ")" and depth:
            depth -= 1
            if not depth:
                out.append(" ")
            continue
        if not depth:
            out.append(ch)
    return "".join(out)


def _declared(stmt: str, keyword: str) -> list[str]:
    body = stmt[len(keyword):]
    body = re.sub(r"\$[^$]*\$", " ", body)              # TeX names
    body = _drop_groups(body)                           # var(log), (long_name='...')
    return [n for n in re.split(r"[\s,]+", body) if n]


# ------------------------------------------------------------- expressions
_FUNCS = {"log", "ln", "log10", "exp", "sqrt", "abs", "sign", "min", "max", "sin", "cos",
          "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh", "erf", "erfc", "normcdf",
          "normpdf", "cbrt", "steady_state", "STEADY_STATE"}
_REFUSED = {"expectation", "EXPECTATION", "diff", "adl", "var_expectation",
            "pac_expectation", "pac_target_nonstationary", "sum"}


def _sympy_functions(sympy):
    def normcdf(x, m=0, s=1):
        return (1 + sympy.erf((x - m) / (s * sympy.sqrt(2)))) / 2

    def normpdf(x, m=0, s=1):
        return sympy.exp(-((x - m) / s) ** 2 / 2) / (s * sympy.sqrt(2 * sympy.pi))

    return {"log": sympy.log, "ln": sympy.log, "log10": lambda x: sympy.log(x, 10),
            "exp": sympy.exp, "sqrt": sympy.sqrt, "abs": sympy.Abs, "sign": sympy.sign,
            "min": sympy.Min, "max": sympy.Max, "sin": sympy.sin, "cos": sympy.cos,
            "tan": sympy.tan, "asin": sympy.asin, "acos": sympy.acos, "atan": sympy.atan,
            "sinh": sympy.sinh, "cosh": sympy.cosh, "tanh": sympy.tanh, "erf": sympy.erf,
            "erfc": sympy.erfc, "normcdf": normcdf, "normpdf": normpdf, "cbrt": sympy.cbrt}


class _Parser:
    """Turns Dynare expression text into sympy, one placeholder per (name, offset)."""

    def __init__(self, sympy, endo, exo, params, locals_=None):
        self.sympy = sympy
        self.endo, self.exo, self.params = set(endo), set(exo), list(params)
        self.param_set = set(params)
        self.locals = dict(locals_ or {})       # model-local name -> sympy expr
        self.refs: dict[str, tuple[str, int]] = {}
        self.steady_refs: set[str] = set()
        self.funcs = _sympy_functions(sympy)

    def parse(self, text: str, label: str):
        sympy = self.sympy
        from sympy.parsing.sympy_parser import parse_expr, standard_transformations
        for bad in _REFUSED:
            if re.search(rf"\b{bad}\s*\(", text):
                raise DynareImportError(f"{label}: {bad}(...) is not read: {text!r}")
        mapping: dict[str, object] = {}

        def ph(name, off):
            key = f"__R{len(mapping)}__"
            mapping[key] = (name, off)
            return key

        def ss_sub(m):
            name = m.group(1)
            if name not in self.endo:
                raise DynareImportError(f"{label}: steady_state() of a non-endogenous {name!r}")
            self.steady_refs.add(name)
            key = f"__S{len(mapping)}__"
            mapping[key] = ("__steady__", name)
            return key

        t = text.replace("^", "**")
        t = re.sub(r"\b(?:steady_state|STEADY_STATE)\s*\(\s*([A-Za-z_]\w*)\s*\)", ss_sub, t)

        def ref(m):
            name, off = m.group(1), m.group(2)
            if name in _FUNCS:
                return m.group(0)
            if name in self.endo or name in self.exo:
                o = int(off.replace(" ", "")) if off is not None else 0
                return ph(name, o)
            if name in self.param_set and off is None:
                key = f"__P{len(mapping)}__"
                mapping[key] = ("__param__", name)
                return key
            if name in self.locals and off is None:
                key = f"__L{len(mapping)}__"
                mapping[key] = ("__local__", name)
                return key
            if off is not None:
                if name in self.locals:
                    raise DynareImportError(f"{label}: a lead or lag of the model-local "
                                            f"variable {name!r} is not read")
                return m.group(0)
            return m.group(0)

        t = re.sub(r"\b([A-Za-z_]\w*)\b(?:\s*\(\s*([+-]?\s*\d+)\s*\))?", ref, t)
        local = {}
        for key, val in mapping.items():
            if val[0] == "__steady__":
                local[key] = sympy.Symbol(f"STEADY_STATE_{val[1]}")
            elif val[0] == "__param__":
                local[key] = sympy.Symbol(val[1])
            elif val[0] == "__local__":
                local[key] = self.locals[val[1]]
            else:
                local[key] = sympy.Symbol(f"__D__{val[0]}__{val[1]}")
                self.refs[str(local[key])] = val
        local.update(self.funcs)
        glob = {"Symbol": sympy.Symbol, "Integer": sympy.Integer, "Float": sympy.Float,
                "Rational": sympy.Rational, "pi": sympy.pi, "inf": sympy.oo, "Inf": sympy.oo}
        try:
            expr = parse_expr(t, local_dict=local, global_dict=glob,
                              transformations=standard_transformations)
        except Exception as exc:
            raise DynareImportError(f"{label}: cannot parse {text!r} ({exc})") from exc
        known = {str(v) for v in local.values() if isinstance(v, sympy.Symbol)}
        for e in self.locals.values():
            known |= {str(s) for s in e.free_symbols}
        unknown = sorted(str(s) for s in expr.free_symbols if str(s) not in known)
        if unknown:
            raise DynareImportError(f"{label}: undeclared name(s) {unknown} in {text!r}")
        return expr


def _numeric(sympy, text: str, env: Mapping[str, float], funcs):
    from sympy.parsing.sympy_parser import parse_expr, standard_transformations
    t = text.replace("^", "**")
    vals = {k: v for k, v in env.items()
            if v is not None and not (isinstance(v, float) and math.isnan(v))}
    local = {}

    def protect(m):
        nm = m.group(0)
        if nm in vals:
            key = f"__V{len(local)}__"
            local[key] = sympy.Float(vals[nm])
            return key
        return nm

    t = re.sub(r"\b[A-Za-z_]\w*\b", protect, t)
    local.update(funcs)
    glob = {"Symbol": sympy.Symbol, "Integer": sympy.Integer, "Float": sympy.Float,
            "Rational": sympy.Rational, "pi": sympy.pi, "inf": sympy.oo, "Inf": sympy.oo}
    expr = parse_expr(t, local_dict=local, global_dict=glob,
                      transformations=standard_transformations)
    if expr.free_symbols:
        raise ValueError(f"unresolved {sorted(map(str, expr.free_symbols))}")
    return float(expr)


# -------------------------------------------------------------------- result
@dataclass
class ModFile:
    """A ``.mod`` file read into this package's form.

    Attributes
    ----------
    model : Model
    params : dict
        Parameter values from the file's assignments. A parameter the file
        leaves to a MATLAB steady-state file stays ``nan`` and is listed in
        :attr:`unset_params`.
    guess : dict
        Endogenous values from ``steady_state_model``, or from ``initval``.
        Names the file could not evaluate (a MATLAB helper function) are absent.
    cov : ndarray
        Shock covariance, in ``model.shocks`` order.
    endogenous, exogenous : list of str
        The Dynare names, in declaration order.
    auxiliary : dict
        Every variable the translation added, with what it stands for.
    """

    model: Model
    params: dict
    guess: dict
    cov: np.ndarray
    endogenous: list
    exogenous: list
    auxiliary: dict
    linear: bool = False
    steady_operators: list = field(default_factory=list)
    unset_params: list = field(default_factory=list)
    missing_guess: list = field(default_factory=list)
    source: str = ""

    def full_guess(self, values: Mapping[str, float] | None = None) -> dict:
        """A value for every model variable, given values for the Dynare ones."""
        base = dict(self.guess)
        if values:
            base.update({k: float(v) for k, v in values.items() if k in self.endogenous})
        out = {}
        for v in self.endogenous:
            out[v] = float(base.get(v, 0.0))
        for name, (kind, of) in self.auxiliary.items():
            out[name] = 0.0 if kind == "shock" or of not in out else out[of]
        return out

    def steady_state(self, guess: Mapping[str, float] | None = None, params=None,
                     **kwargs) -> SteadyState:
        """Solve the steady state, iterating on ``steady_state(x)`` terms if any.

        ``guess`` overrides :attr:`guess` for the Dynare names it lists. When the
        model has ``steady_state(x)`` terms, their values are written into
        ``params`` (:attr:`params` by default), which is what the dynamic
        solution must then be given.
        """
        params = self.params if params is None else params
        x0 = self.full_guess(guess)
        if not self.steady_operators:
            return steady_state(self.model, x0, params, **kwargs)
        for _ in range(50):
            for v in self.steady_operators:
                params[f"STEADY_STATE_{v}"] = x0[v]
            ss = steady_state(self.model, x0, params, **kwargs)
            gap = max(abs(ss.values[v] - x0[v]) for v in self.steady_operators)
            x0 = dict(ss.values)
            if gap < 1e-13:
                break
        return ss

    def name_of(self, v: str) -> str:
        """The Dynare name behind a model variable, or the variable itself."""
        kind, of = self.auxiliary.get(v, (None, v))
        return of


# ------------------------------------------------------------------- import
def from_mod(source, *, defines: Mapping | None = None, derivatives: str = "jacobian",
             name: str | None = None) -> ModFile:
    """Read a Dynare ``.mod`` file.

    Parameters
    ----------
    source : str or path
        A path to a ``.mod`` file, or its text.
    defines : mapping, optional
        Macro variables, as ``-D`` on Dynare's command line; they override
        ``@#define`` in the file.
    derivatives : {"jacobian", "both", "none"}
        Exact derivatives to build. ``"jacobian"`` by default: the Hessian of
        an imported model with many leads and lags is expensive to build, and
        only second order needs it.
    """
    from .symbolic import _require_sympy, model_from_expressions

    sympy = _require_sympy()
    p = Path(str(source))
    if "\n" not in str(source) and p.suffix == ".mod" and p.exists():
        text, label = p.read_text(), p.stem
    else:
        text, label = str(source), "mod"
    name = name or label

    if re.search(r"^\s*@#\s*(include|includepath|for)\b", text, flags=re.M):
        raise DynareImportError("@#include, @#includepath and @#for are not read")
    src = _macro(_strip_comments(text), defines)

    stmts = _statements(src)
    endo, exo, params, predet = [], [], [], []
    param_text: list[tuple[str, str]] = []
    model_stmts, model_opts = None, ""
    ssm, initval, shocks = [], [], []
    i = 0
    while i < len(stmts):
        st = stmts[i]
        word = re.match(r"[A-Za-z_]\w*", st)
        word = word.group(0) if word else ""
        head, opts = _head(st)
        if head in _BLOCKS and head == word:
            j = i + 1
            while j < len(stmts) and stmts[j] != "end":
                j += 1
            if j == len(stmts):
                raise DynareImportError(f"block {head!r} has no 'end;'")
            body = stmts[i + 1:j]
            if head == "model":
                if model_stmts is not None:
                    raise DynareImportError("more than one model block")
                model_stmts, model_opts = body, opts
            elif head == "steady_state_model":
                ssm = body
            elif head == "initval":
                initval = body
            elif head == "shocks":
                shocks += body
            i = j + 1
            continue
        if word == "var" and not st.startswith("var_"):
            endo += _declared(st, "var")
        elif word == "varexo":
            exo += _declared(st, "varexo")
        elif word == "varexo_det":
            raise DynareImportError("varexo_det is not read")
        elif word == "parameters":
            params += _declared(st, "parameters")
        elif word == "predetermined_variables":
            predet += _declared(st, "predetermined_variables")
        else:
            m = re.match(r"([A-Za-z_]\w*)\s*=(?!=)(.*)$", st, flags=re.S)
            if m and m.group(1) not in endo and m.group(1) not in exo:
                param_text.append((m.group(1), m.group(2).strip()))
        i += 1

    if model_stmts is None:
        raise DynareImportError("no model block")
    clash = (set(endo) & set(exo)) | (set(endo) & set(params)) | (set(exo) & set(params))
    if clash:
        raise DynareImportError(f"names declared twice: {sorted(clash)}")
    linear = "linear" in model_opts

    funcs = _sympy_functions(sympy)

    # --- parameters, in order of assignment
    # Dynare passes an assignment to an undeclared name through as MATLAB; the
    # examples use it for a constant read in the shocks block (example1's phi).
    pvals: dict[str, float] = {p_: float("nan") for p_ in params}
    matlab: dict[str, float] = {}
    for pname, expr in param_text:
        try:
            val = _numeric(sympy, expr, {**matlab, **pvals}, funcs)
        except Exception:
            val = float("nan")
        (pvals if pname in pvals else matlab)[pname] = val

    # --- model equations
    parser = _Parser(sympy, endo, exo, params)
    eqs = []
    for k, st in enumerate(model_stmts):
        st = re.sub(r"^\s*\[[^\]]*\]\s*", "", st)
        if not st:
            continue
        if st.startswith("#"):
            nm, _, rhs = st[1:].partition("=")
            parser.locals[nm.strip()] = parser.parse(rhs, f"model-local {nm.strip()}")
            continue
        if st.count("=") > 1 and "==" not in st:
            raise DynareImportError(f"equation {len(eqs) + 1}: more than one '=': {st!r}")
        if "=" in st and "==" not in st:
            lhs, rhs = st.split("=")
            text_eq = f"({lhs}) - ({rhs})"
        else:
            text_eq = st
        eqs.append((st, parser.parse(text_eq, f"equation {len(eqs) + 1}")))
    if len(eqs) != len(endo):
        raise DynareImportError(f"{len(eqs)} equations for {len(endo)} endogenous variables")

    # --- dating: predetermined_variables, then the two-period translation
    S = sympy.Symbol
    refs = parser.refs
    shift = {n: (1 if n in predet else 0) for n in endo}
    for v in predet:
        if v not in endo:
            raise DynareImportError(f"predetermined variable {v!r} is not declared with var")
    max_lag, max_lead, shock_lag = {}, {}, {}

    def dated(sym_name):
        nm, off = refs[sym_name]
        if nm in exo:
            if off > 0:
                raise DynareImportError(f"a lead of the shock {nm!r} is not read")
            shock_lag[nm] = max(shock_lag.get(nm, 0), -off)
            return nm, off
        off -= shift[nm]
        if off < 0:
            max_lag[nm] = max(max_lag.get(nm, 0), -off)
        if off > 1:
            max_lead[nm] = max(max_lead.get(nm, 0), off - 1)
        return nm, off

    exprs, labels = [], []
    for raw, e in eqs:
        sub = {}
        for s_ in e.free_symbols:
            sname = str(s_)
            if sname not in refs:
                continue
            nm, off = dated(sname)
            if nm in exo:
                base = f"SHK_{nm}" if off == 0 else f"LAG{-off}_SHK_{nm}"
                sub[s_] = S(base)
            elif off == 0:
                sub[s_] = S(nm)
            elif off == 1:
                sub[s_] = S(f"{nm}__next")
            elif off > 1:
                sub[s_] = S(f"LEAD{off - 1}_{nm}__next")
            else:
                sub[s_] = S(f"LAG{-off}_{nm}")
        exprs.append(e.xreplace(sub))
        labels.append(raw)

    auxiliary: dict[str, tuple[str, str]] = {}
    states, jumps = [], list(endo)
    for v in endo:
        for k in range(1, max_lag.get(v, 0) + 1):
            n_ = f"LAG{k}_{v}"
            states.append(n_)
            auxiliary[n_] = ("lag", v)
            prev = v if k == 1 else f"LAG{k - 1}_{v}"
            exprs.append(S(f"{n_}__next") - S(prev))
            labels.append(f"{n_}(+1) = {prev}")
        for k in range(1, max_lead.get(v, 0) + 1):
            n_ = f"LEAD{k}_{v}"
            jumps.append(n_)
            auxiliary[n_] = ("lead", v)
            prev = v if k == 1 else f"LEAD{k - 1}_{v}"
            exprs.append(S(n_) - S(f"{prev}__next"))
            labels.append(f"{n_} = {prev}(+1)")
    used_shocks = [e_ for e_ in exo if any(refs[r][0] == e_ for r in refs)]
    for e_ in exo:
        n_ = f"SHK_{e_}"
        states.append(n_)
        auxiliary[n_] = ("shock", e_)
        exprs.append(S(f"{n_}__next") - S(e_))
        labels.append(f"{n_}(+1) = {e_}")
        for k in range(1, shock_lag.get(e_, 0) + 1):
            m_ = f"LAG{k}_SHK_{e_}"
            states.append(m_)
            auxiliary[m_] = ("shock", e_)
            prev = n_ if k == 1 else f"LAG{k - 1}_SHK_{e_}"
            exprs.append(S(f"{m_}__next") - S(prev))
            labels.append(f"{m_}(+1) = {prev}")
    taken = set(endo) | set(exo) | set(params)
    clash = [a for a in auxiliary if a in taken]
    if clash:
        raise DynareImportError(f"auxiliary names collide with declared ones: {clash}")
    del used_shocks

    steady_params = [f"STEADY_STATE_{v}" for v in sorted(parser.steady_refs)]
    all_params = list(params) + steady_params
    model = model_from_expressions(exprs, states=states, jumps=jumps, shocks=list(exo),
                                   parameters=all_params, name=name,
                                   derivatives=derivatives, source_equations=labels)

    # --- steady state guess: steady_state_model first, then initval
    env = {**matlab, **pvals}
    guess: dict[str, float] = {}
    for body in (initval, ssm):
        local_env = dict(env)
        local_env.update(guess)
        for st in body:
            m = re.match(r"([A-Za-z_]\w*)\s*=(?!=)(.*)$", st, flags=re.S)
            if not m:
                continue
            nm, rhs = m.group(1), m.group(2)
            if re.match(r"\s*\[", nm):
                continue
            try:
                val = _numeric(sympy, rhs, local_env, funcs)
            except Exception:
                continue            # a MATLAB helper: left to the caller
            local_env[nm] = val
            if nm in endo:
                guess[nm] = val
            elif nm in pvals and body is ssm:
                pvals[nm] = val     # Dynare lets steady_state_model set parameters
    missing = [v for v in endo if v not in guess]

    # --- shocks
    cov = np.zeros((len(exo), len(exo)))
    cur = None
    env_all = {**env, **guess}
    for st in shocks:
        m = re.match(r"var\s+([A-Za-z_]\w*)\s*$", st)
        if m:
            cur = m.group(1)
            continue
        m = re.match(r"stderr\s+(.*)$", st, flags=re.S)
        if m and cur is not None:
            sd = _numeric(sympy, m.group(1), env_all, funcs)
            i_ = exo.index(cur)
            cov[i_, i_] = sd ** 2
            continue
        m = re.match(r"var\s+([A-Za-z_]\w*)\s*=(.*)$", st, flags=re.S)
        if m:
            i_ = exo.index(m.group(1))
            cov[i_, i_] = _numeric(sympy, m.group(2), env_all, funcs)
            continue
        m = re.match(r"(var|corr)\s+([A-Za-z_]\w*)\s*,\s*([A-Za-z_]\w*)\s*=(.*)$", st, flags=re.S)
        if m:
            i_, j_ = exo.index(m.group(2)), exo.index(m.group(3))
            val = _numeric(sympy, m.group(4), env_all, funcs)
            if m.group(1) == "corr":
                val *= math.sqrt(cov[i_, i_] * cov[j_, j_])
            cov[i_, j_] = cov[j_, i_] = val
            continue
        # periods/values: deterministic shocks, not read here

    return ModFile(model=model, params={**pvals, **{p_: float("nan") for p_ in steady_params}},
                   guess=guess, cov=cov, endogenous=list(endo), exogenous=list(exo),
                   auxiliary=auxiliary, linear=linear,
                   steady_operators=sorted(parser.steady_refs),
                   unset_params=[p_ for p_ in params if math.isnan(pvals[p_])],
                   missing_guess=missing, source=label)

