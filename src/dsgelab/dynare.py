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

__all__ = ["DynareImportError", "ModFile", "from_mod", "read_expectation_errors_csv"]


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


_BLOCKS = {"osr_params_bounds", "model", "initval", "endval", "histval", "steady_state_model", "shocks", "mshocks",
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


def _options(text: str) -> dict:
    """``(a=1, instruments=(i,r), flag)`` -> ``{"a": "1", "instruments": "(i,r)", "flag": ""}``."""
    text = text.strip()
    if not text.startswith("("):
        return {}
    depth, end = 0, len(text)
    for k, ch in enumerate(text):
        depth += ch == "("
        depth -= ch == ")"
        if depth == 0:
            end = k
            break
    body, out, cur, depth = text[1:end], {}, "", 0
    for ch in body + ",":
        if ch == "," and depth == 0:
            key, _, val = cur.partition("=")
            if key.strip():
                out[key.strip()] = val.strip()
            cur = ""
            continue
        depth += ch == "("
        depth -= ch == ")"
        cur += ch
    return out


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
    block: object = None
    planner: dict | None = None
    osr_params: list = field(default_factory=list)
    optim_weights: dict = field(default_factory=dict)
    osr_bounds: dict = field(default_factory=dict)
    histval: dict = field(default_factory=dict)
    det_shocks: dict = field(default_factory=dict)
    histval_unread: list = field(default_factory=list)
    estimated: list = field(default_factory=list)
    varobs: list = field(default_factory=list)
    estimation_options: dict = field(default_factory=dict)
    program: dict = field(default_factory=dict, repr=False)

    def full_guess(self, values: Mapping[str, float] | None = None) -> dict:
        """A value for every model variable, given values for the Dynare ones."""
        base = dict(self.guess)
        if values:
            base.update({k: float(v) for k, v in values.items() if k in self.endogenous})
        out = {}
        for v in self.endogenous:
            out[v] = float(base.get(v, 0.0))
        for name, (kind, of) in self.auxiliary.items():
            if kind == "shock":
                out[name] = float(self.params.get(f"EXO_SS_{of}", 0.0))
            else:
                out[name] = out.get(of, 0.0)
        return out

    def initial_states(self, steady: SteadyState, *, histval: Mapping | None = None,
                       exo_first: Mapping[str, float] | None = None) -> np.ndarray:
        """States at the first date of a transition, from Dynare's ``histval``.

        The first date of a transition here is Dynare's period 1. ``LAG<k>_v``
        there is Dynare's ``v`` in period ``1-k``, taken from ``histval`` (this
        file's by default, keyed ``(name, period)``) or else from ``steady``;
        ``SHK_e`` is the exogenous level in period 1, ``exo_first[e]`` or else its
        steady state.
        """
        hv = dict(self.histval if histval is None else histval)
        out = []
        for name in self.model.states:
            kind, of = self.auxiliary.get(name, (None, name))
            if kind == "lag":
                k = int(name[3:name.index("_")])
                out.append(float(hv.get((of, 1 - k), steady.values[of])))
            elif kind == "shock" and name.startswith("SHK_"):
                out.append(float((exo_first or {}).get(of, steady.values[name])))
            else:
                out.append(float(steady.values[name]))
        return np.array(out)

    def deterministic(self, T: int, steady: SteadyState, *,
                      det_shocks: Mapping | None = None) -> tuple:
        """Dynare's ``perfect_foresight_setup(periods=T)`` as arguments for
        :func:`dsgelab.perfect_foresight`.

        Returns ``(shocks, initial_states)``. ``det_shocks`` (this file's
        ``shocks`` block by default) maps each exogenous variable to
        ``{period: level}``, Dynare's periods counted from 1; the path is known
        from the start. ``histval`` sets the initial states. Date ``t`` of the
        transition is Dynare's period ``t+1``, and date ``T`` is Dynare's
        terminal period ``T+1``.
        """
        ds = dict(self.det_shocks if det_shocks is None else det_shocks)

        def level(e, period):
            return float(ds.get(e, {}).get(period, self.params[f"EXO_SS_{e}"]))

        shocks = np.zeros((T, len(self.model.shocks)))
        for e in self.exogenous:
            j = self.model.shocks.index(e)
            for t in range(T):
                if t + 2 <= T:
                    shocks[t, j] = level(e, t + 2) - self.params[f"EXO_SS_{e}"]
        k0 = self.initial_states(steady, exo_first={e: level(e, 1) for e in self.exogenous})
        return shocks, k0

    def news(self, table: Mapping[int, Mapping], T: int, params=None) -> tuple:
        """Dynare's information sets as :class:`~dsgelab.foresight.News`.

        ``table`` maps each period in which information arrives (Dynare's
        ``learnt_in``, from 1) to ``{"paths": {e: {period: level}}, "terminal":
        {e: level}}``: the levels of the exogenous variables agents now expect,
        and the steady-state levels they now expect to converge to -- one column
        of the CSV that ``perfect_foresight_with_expectation_errors_setup`` reads
        (:func:`read_expectation_errors_csv`). A period with no level stated is
        expected at the terminal level.

        An information set that only confirms what agents already expected is
        not news and starts no new plan, as in Dynare -- which matters under
        ``constant_simulation_length``, where every plan extends the horizon.

        Returns ``(news, start)`` for
        :func:`dsgelab.perfect_foresight_with_expectation_errors`, ``start`` being
        the steady state at the file's own exogenous levels.
        """
        from .foresight import News
        from .model import set_params

        base = dict(self.params if params is None else params)
        start = self.steady_state(params=base)
        out = []
        expected = None                      # (path from the news date on, terminal) held so far
        for learnt in sorted(table):
            entry = table[learnt]
            term = {e: float(entry.get("terminal", {}).get(e, base[f"EXO_SS_{e}"]))
                    for e in self.exogenous}
            seen = {e: {t: float(entry.get("paths", {}).get(e, {}).get(t, term[e]))
                        for t in range(learnt, T + 1)} for e in self.exogenous}
            if expected is not None:
                held_path, held_term = expected
                if held_term == term and all(seen[e][t] == held_path[e].get(t, held_term[e])
                                             for e in self.exogenous for t in seen[e]):
                    continue                 # nothing new: Dynare does not re-plan either
            expected = (seen, term)
            p_k = set_params(base, {f"EXO_SS_{e}": v for e, v in term.items()})
            terminal = self.steady_state(guess={**start.values}, params=p_k)
            paths = entry.get("paths", {})

            def level(e, period):
                return float(paths.get(e, {}).get(period, term[e]))

            shocks = np.zeros((T, len(self.model.shocks)))
            for e in self.exogenous:
                j = self.model.shocks.index(e)
                for t in range(learnt - 1, T):        # entry t sets SHK_e at t+1 = period t+2
                    shocks[t, j] = (level(e, t + 2) - term[e]) if t + 2 <= T else 0.0
            states = {f"SHK_{e}": level(e, learnt) for e in self.exogenous}
            out.append(News(period=learnt - 1, shocks=shocks, params=p_k, terminal=terminal,
                            states=states))
        return out, start

    @property
    def shock_states(self) -> dict:
        """``{"SHK_e": "e"}``: the states that carry each Dynare shock's own value.

        For :func:`dsgelab.shock_decomposition`, so that the shock of Dynare's
        first period is counted as a shock and not as an initial condition.
        """
        return {f"SHK_{e}": e for e in self.exogenous}

    def estimation_problem(self, data, **kwargs):
        """An :class:`~dsgelab.EstimationProblem` from this file's estimation blocks.

        ``estimated_params`` gives the priors (and ``"stderr e"`` names for shock
        and measurement standard deviations), ``varobs`` the observables, and the
        options of ``estimation(...)`` the rest: ``loglinear`` with ``logdata``
        makes every observable a log, ``first_obs`` and ``presample`` carry over,
        ``diffuse_filter`` makes the start diffuse. ``data`` maps each observable
        to its series, as Dynare's datafile would; ``kwargs`` override anything.
        """
        from .estimation import EstimationProblem

        if not self.estimated:
            raise DynareImportError("this .mod has no estimated_params block")
        if not self.varobs:
            raise DynareImportError("this .mod has no varobs")
        opts = self.estimation_options
        loglin = "loglinear" in opts
        if loglin and "logdata" not in opts:
            raise DynareImportError("loglinear without logdata: the data would need logging "
                                    "first; log them and pass transforms='log'")
        priors = {e["name"]: e["prior"] for e in self.estimated if e["prior"] is not None}
        free = [e["name"] for e in self.estimated]
        sd = {e: float(np.sqrt(self.cov[i, i])) for i, e in enumerate(self.exogenous)}
        params = dict(self.params)
        meas = {}
        for e in self.estimated:
            # Dynare: with no initial value in estimated_params the starting point is the
            # PRIOR MEAN, not the calibration
            start = e["init"] if np.isfinite(e["init"]) else (
                e["prior"].mean if e["prior"] is not None else np.nan)
            if not np.isfinite(start):
                continue
            if e["kind"] == "param":
                params[e["target"]] = start
            elif e["kind"] == "stderr" and e["role"] == "shock":
                sd[e["target"]] = start
            elif e["kind"] == "stderr":
                meas[e["target"]] = start

        def steady_fn(p):
            return self.steady_state(params=p)

        kw = dict(model=self.model, data=data, params=params, observables=list(self.varobs),
                  priors=priors, free=free, cov=sd, measurement=meas or None,
                  transforms="log" if loglin else "level",
                  steady_fn=steady_fn,
                  init="diffuse" if "diffuse_filter" in opts else "auto",
                  first_obs=int(float(opts.get("first_obs", 1))),
                  presample=int(float(opts.get("presample", 0))))
        bounds = {e["name"]: (e["lower"], e["upper"]) for e in self.estimated
                  if e["prior"] is None and np.isfinite(e["lower"])}
        if bounds:
            kw["bounds"] = bounds
        kw.update(kwargs)
        return EstimationProblem(**kw)

    def steady_state(self, guess: Mapping[str, float] | None = None, params=None,
                     **kwargs) -> SteadyState:
        """Solve the steady state, iterating on ``steady_state(x)`` terms if any.

        ``guess`` overrides :attr:`guess` for the Dynare names it lists. When the
        model has ``steady_state(x)`` terms, their values are written into
        ``params`` (:attr:`params` by default), which is what the dynamic
        solution must then be given.
        """
        params = self.params if params is None else params
        if self.model is None:
            raise DynareImportError(
                "this .mod leaves instruments without an equation: pass it to "
                "dsgelab.ramsey_model or dsgelab.discretionary_policy first")
        if guess is None and self.program and params is not self.params:
            # the file's steady_state_model / initval, evaluated at THESE parameters:
            # an exact start where the file gives closed forms, and the parameters it
            # sets inside steady_state_model, recomputed (Dynare does the same)
            from .symbolic import _require_sympy
            sympy = _require_sympy()
            g, _, upd = _run_program(sympy, self.program, dict(params), _sympy_functions(sympy))
            if isinstance(params, dict):
                params.update(upd)
            base = self.full_guess()
            base.update({k: v for k, v in g.items()})
            x0 = self.full_guess(base)
            for name, (kind, of) in self.auxiliary.items():
                if kind == "shock":
                    x0[name] = float(params.get(f"EXO_SS_{of}", 0.0))
        else:
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
    # verbatim blocks are MATLAB, whose own `end` would end the block early
    src = re.sub(r"^\s*verbatim\s*;.*?^\s*end\s*;", " ", src, flags=re.S | re.M)

    stmts = _statements(src)
    endo, exo, params, predet = [], [], [], []
    param_text: list[tuple[str, str]] = []
    model_stmts, model_opts = None, ""
    ssm, initval, shocks, histval_body = [], [], [], []
    planner_text, policy_cmd, policy_opts = None, None, {}
    osr_params_list, weights_body, bounds_body = [], [], []
    estim_body, varobs, estim_opts = [], [], {}
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
            elif head == "histval":
                histval_body = body
            elif head == "initval":
                initval = body
            elif head == "shocks":
                shocks += body
            elif head == "optim_weights":
                weights_body = body
            elif head == "osr_params_bounds":
                bounds_body = body
            elif head == "estimated_params":
                estim_body = body
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
        elif word == "planner_objective":
            planner_text = st[len(word):].strip()
        elif word in ("ramsey_model", "ramsey_policy", "discretionary_policy"):
            policy_cmd = "discretion" if word == "discretionary_policy" else "ramsey"
            policy_opts = _options(st[len(word):])
        elif word == "osr_params":
            osr_params_list += _declared(st, "osr_params")
        elif word == "varobs":
            varobs += _declared(st, "varobs")
        elif word == "estimation":
            estim_opts = _options(st[len(word):])
        elif word == "osr":
            osr_opts = _options(st[len(word):])
            if "planner_discount" in osr_opts:
                policy_opts.setdefault("planner_discount", osr_opts["planner_discount"])
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
    planner_present = planner_text is not None or policy_cmd is not None
    if len(eqs) != len(endo) and not (planner_present and len(eqs) < len(endo)):
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
            if off > 1:
                raise DynareImportError(f"a lead of more than one period of the exogenous "
                                        f"{nm!r} is not read")
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
                # e(+1) is the next value of the state that carries the level of e
                base = (f"SHK_{nm}__next" if off == 1 else f"SHK_{nm}" if off == 0
                        else f"LAG{-off}_SHK_{nm}")
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
        exprs.append(S(f"{n_}__next") - S(f"EXO_SS_{e_}") - S(e_))
        labels.append(f"{n_}(+1) = EXO_SS_{e_} + {e_}")
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
    # A Dynare exogenous variable is a LEVEL with a steady state of its own (zero unless
    # initval says otherwise); this package's innovation has mean zero. The level is the
    # state SHK_e, with SHK_e(+1) = EXO_SS_e + e, and a permanent change in it is a
    # change in the parameter EXO_SS_e.
    exo_params = [f"EXO_SS_{e_}" for e_ in exo]
    clash = [q for q in exo_params if q in params]
    if clash:
        raise DynareImportError(f"parameter names reserved for exogenous levels: {clash}")
    all_params = list(params) + steady_params + exo_params
    model = None
    if len(exprs) == len(states) + len(jumps):
        model = model_from_expressions(exprs, states=states, jumps=jumps, shocks=list(exo),
                                       parameters=all_params, name=name,
                                       derivatives=derivatives, source_equations=labels)
    from .optimal import PrivateBlock
    block = PrivateBlock(list(exprs), list(states), list(jumps), list(exo), list(all_params),
                         list(labels), name=name)

    # --- the planner: objective (current-dated endogenous variables), discount, instruments
    planner = None
    if planner_present:
        planner = dict(kind=policy_cmd, objective=None, discount=None, instruments=None,
                       text=planner_text)
        if planner_text is not None:
            ob = parser.parse(planner_text, "planner_objective")
            sub = {}
            for s_ in ob.free_symbols:
                if str(s_) in refs:
                    nm, off = refs[str(s_)]
                    if nm in exo or off != 0:
                        raise DynareImportError("planner_objective may only contain "
                                                "current-period endogenous variables")
                    sub[s_] = S(nm)
            planner["objective"] = ob.xreplace(sub)
        disc = policy_opts.get("planner_discount", "1")
        planner["discount"] = _Parser(sympy, [], [], params).parse(disc, "planner_discount")
        if "instruments" in policy_opts:
            planner["instruments"] = _declared(policy_opts["instruments"].strip("()"), "")

    # --- steady state guess: steady_state_model first, then initval
    program = dict(initval=list(initval), ssm=list(ssm), matlab=dict(matlab), endo=list(endo),
                   exo=list(exo), params=list(params))
    guess, exo_level, ssm_params = _run_program(sympy, program, pvals, funcs)
    pvals.update(ssm_params)
    env = {**matlab, **pvals, **{e_: 0.0 for e_ in exo}}
    missing = [v for v in endo if v not in guess]
    for e_ in exo:
        pvals[f"EXO_SS_{e_}"] = exo_level.get(e_, 0.0)
    histval, histval_unread = {}, []
    for st in histval_body:
        m = re.match(r"([A-Za-z_]\w*)\s*\(\s*([+-]?\d+)\s*\)\s*=(.*)$", st, flags=re.S)
        if not m:
            raise DynareImportError(f"histval: cannot read {st!r}")
        try:
            histval[(m.group(1), int(m.group(2)))] = _numeric(
                sympy, m.group(3), {**env, **guess, **pvals}, funcs)
        except Exception:
            histval_unread.append(st)     # a MATLAB variable of the script: left to the caller

    # --- shocks
    cov = np.zeros((len(exo), len(exo)))
    cur = None
    det_periods: list = []
    det_shocks: dict = {}
    env_all = {**env, **pvals, **guess}
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
        m = re.match(r"periods\s+(.*)$", st, flags=re.S)
        if m and cur is not None:
            det_periods = []
            for part in re.split(r"[\s,]+", m.group(1).strip()):
                if ":" in part:
                    a, b_ = part.split(":")
                    det_periods.append(list(range(int(a), int(b_) + 1)))
                elif part:
                    det_periods.append([int(part)])
            continue
        m = re.match(r"values\s+(.*)$", st, flags=re.S)
        if m and cur is not None:
            vals = [_numeric(sympy, v, env_all, funcs)
                    for v in re.split(r"\s*,\s*|\s+(?![^()]*\))", m.group(1).strip()) if v]
            if len(vals) == 1:
                vals = vals * len(det_periods)
            if len(vals) != len(det_periods):
                raise DynareImportError(f"shocks: {len(det_periods)} periods and {len(vals)} values for {cur}")
            for group, v in zip(det_periods, vals):
                for t in group:
                    det_shocks.setdefault(cur, {})[t] = v
            continue

    estimated = [_estimated_entry(sympy, st, env_all, funcs, endo, exo) for st in estim_body]

    weights = {}
    for st in weights_body:
        m = re.match(r"([A-Za-z_]\w*)\s*(?:,\s*([A-Za-z_]\w*))?\s+(.+)$", st, flags=re.S)
        if not m:
            raise DynareImportError(f"optim_weights: cannot read {st!r}")
        a, b_, w = m.group(1), m.group(2), _numeric(sympy, m.group(3), env_all, funcs)
        weights[a if b_ is None else (a, b_)] = w
    bounds = {}
    for st in bounds_body:
        parts = [x.strip() for x in st.split(",")]
        if len(parts) != 3:
            raise DynareImportError(f"osr_params_bounds: cannot read {st!r}")
        bounds[parts[0]] = (_numeric(sympy, parts[1], env_all, funcs),
                            _numeric(sympy, parts[2], env_all, funcs))

    return ModFile(model=model, block=block, planner=planner, osr_params=osr_params_list,
                   histval=histval, histval_unread=histval_unread, det_shocks=det_shocks,
                   estimated=estimated, varobs=varobs, estimation_options=estim_opts,
                   program=program,
                   optim_weights=weights, osr_bounds=bounds, params={**pvals, **{p_: float("nan") for p_ in steady_params}},
                   guess=guess, cov=cov, endogenous=list(endo), exogenous=list(exo),
                   auxiliary=auxiliary, linear=linear,
                   steady_operators=sorted(parser.steady_refs),
                   unset_params=[p_ for p_ in params if math.isnan(pvals[p_])],
                   missing_guess=missing, source=label)



def read_expectation_errors_csv(path) -> dict:
    """Read the CSV of ``perfect_foresight_with_expectation_errors_setup(datafile=...)``.

    The file has one column per (exogenous variable, period of information), a
    row per real period and a last row ``Terminal``. Returns the ``table`` that
    :meth:`ModFile.news` takes.
    """
    import csv

    rows = list(csv.reader(open(path)))
    names = rows[0][1:]
    learnt = [int(float(x)) for x in rows[1][1:]]
    table: dict = {}
    for row in rows[2:]:
        label = row[0].strip()
        for e, l, cell in zip(names, learnt, row[1:]):
            if cell.strip() == "":
                continue
            entry = table.setdefault(l, {"paths": {}, "terminal": {}})
            if label.lower().startswith("terminal"):
                entry["terminal"][e] = float(cell)
            else:
                period = int(re.search(r"(\d+)", label).group(1))
                entry["paths"].setdefault(e, {})[period] = float(cell)
    return table


def _run_program(sympy, program: dict, pvals: Mapping, funcs) -> tuple:
    """Evaluate initval, then steady_state_model, at the parameter values ``pvals``.

    Returns ``(guess, exogenous levels, parameters set inside steady_state_model)``.
    A statement that calls something this reader cannot evaluate (a MATLAB
    helper) is skipped, and the variables it would set are left to the caller.
    """
    endo, exo = program["endo"], program["exo"]
    env = {**program["matlab"], **dict(pvals), **{e_: 0.0 for e_ in exo}}
    guess: dict = {}
    exo_level: dict = {}
    updates: dict = {}
    for which in ("initval", "ssm"):
        local_env = dict(env)
        local_env.update(updates)
        local_env.update(guess)
        local_env.update(exo_level)
        for st in program[which]:
            m = re.match(r"([A-Za-z_]\w*)\s*=(?!=)(.*)$", st, flags=re.S)
            if not m:
                continue
            nm, rhs = m.group(1), m.group(2)
            try:
                val = _numeric(sympy, rhs, local_env, funcs)
            except Exception:
                continue
            local_env[nm] = val
            if nm in endo:
                guess[nm] = val
            elif nm in exo:
                exo_level[nm] = val
            elif nm in program["params"] and which == "ssm":
                updates[nm] = val
    return guess, exo_level, updates


_PRIOR_SHAPES = {"beta_pdf", "gamma_pdf", "normal_pdf", "inv_gamma_pdf", "inv_gamma1_pdf",
                 "inv_gamma2_pdf", "uniform_pdf", "weibull_pdf"}


def _split_top(text: str) -> list:
    out, cur, depth = [], "", 0
    for ch in text:
        if ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
            continue
        depth += ch == "("
        depth -= ch == ")"
        cur += ch
    out.append(cur.strip())
    return out


def _estimated_entry(sympy, st: str, env, funcs, endo, exo) -> dict:
    """One line of ``estimated_params``."""
    from .priors import prior

    st = st.strip()
    kind, target = "param", None
    if st.startswith("stderr "):
        kind = "stderr"
        st = st[len("stderr "):]
    elif st.startswith("corr "):
        kind = "corr"
        st = st[len("corr "):]
    parts = _split_top(st)
    if kind == "corr":
        target = (parts[0], parts[1])
        parts = parts[2:]
        name = f"corr {target[0]}, {target[1]}"
    else:
        target = parts[0]
        parts = parts[1:]
        name = target if kind == "param" else f"stderr {target}"

    def num(x):
        x = x.strip()
        if x.lower() == "nan" or x == "":
            return float("nan")
        return _numeric(sympy, x, env, funcs)

    shape_at = next((i for i, x in enumerate(parts) if x.strip() in _PRIOR_SHAPES), None)
    pre = parts if shape_at is None else parts[:shape_at]
    init = lower = upper = float("nan")
    if len(pre) >= 1:
        init = num(pre[0])
    if len(pre) >= 3:
        lower, upper = num(pre[1]), num(pre[2])
    pr = None
    if shape_at is not None:
        shape = parts[shape_at].strip()
        rest = [num(x) for x in parts[shape_at + 1:]]
        mean, sd = rest[0], rest[1]
        p3 = rest[2] if len(rest) > 2 else None
        p4 = rest[3] if len(rest) > 3 else None
        p3 = None if p3 is not None and np.isnan(p3) else p3
        p4 = None if p4 is not None and np.isnan(p4) else p4
        pr = prior(shape, mean, sd, p3, p4)
    if kind == "stderr":
        role = "shock" if target in exo else ("measurement" if target in endo else None)
        if role is None:
            raise DynareImportError(f"estimated_params: stderr of unknown {target!r}")
    else:
        role = kind
    return dict(name=name, kind=kind, role=role, target=target, init=init, lower=lower,
                upper=upper, prior=pr)
