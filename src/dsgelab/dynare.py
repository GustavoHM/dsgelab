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
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike

from .model import Model
from .steady import SteadyState, steady_state

if TYPE_CHECKING:
    from .bgp import BalancedGrowth
    from .estimation import EstimationProblem
    from .extended_path import ExtendedPath
    from .foresight import News
    from .identification import Identification
    from .irf_matching import IRFMatchingResult
    from .moment_estimation import MomentsResult
    from .restrictions import RestrictionCheck, RestrictionSample
    from .kalman import Data
    from .occbin import OccbinModel, OccbinResult

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


_BLOCKS = {"occbin_constraints", "homotopy_setup", "osr_params_bounds", "model", "initval", "endval", "histval", "steady_state_model", "shocks", "mshocks",
           "estimated_params", "estimated_params_init", "estimated_params_bounds",
           "estimated_params_remove", "observation_trends", "optim_weights", "homotopy_setup",
           "conditional_forecast_paths", "irf_calibration", "moment_calibration", "verbatim",
           "filter_initial_state", "matched_moments", "matched_irfs", "matched_irfs_weights",
           "deterministic_trends",
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


def _numeric_vector(sympy, text: str, env: Mapping, funcs) -> list[float]:
    """Read the numeric vectors used by matched_irfs, including MATLAB ranges."""
    t = text.strip()
    if t.startswith("(") and t.endswith(")"):
        t = t[1:-1].strip()
    if t in env and not np.isscalar(env[t]):
        return [float(v) for v in np.asarray(env[t], dtype=float).ravel()]
    bracketed = t.startswith("[") and t.endswith("]")
    if bracketed:
        t = t[1:-1].strip()
    parts, cur, depth = [], "", 0
    for ch in t + (" " if bracketed or "," not in t else ","):
        depth += ch == "("
        depth -= ch == ")"
        if depth == 0 and (ch == "," or (ch.isspace() and (bracketed or "," not in t))):
            if cur.strip():
                parts.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if not parts:
        parts = [t]
    out: list[float] = []
    for part in parts:
        bare = part[1:-1].strip() if part.startswith("(") and part.endswith(")") else part
        if bare in env and not np.isscalar(env[bare]):
            out.extend(float(v) for v in np.asarray(env[bare], dtype=float).ravel())
            continue
        bits = part.split(":")
        if len(bits) in (2, 3):
            nums = [_numeric(sympy, b, env, funcs) for b in bits]
            start, stop = nums[0], nums[-1]
            step = nums[1] if len(bits) == 3 else 1.0
            if step == 0:
                raise ValueError("a MATLAB range cannot have zero step")
            count = max(0, int(math.floor((stop - start) / step)) + 1)
            out.extend(start + i * step for i in range(count))
        else:
            out.append(_numeric(sympy, part, env, funcs))
    return out


def _read_irf_blocks(body, weight_body, sympy, env, funcs):
    """Translate matched_irfs and matched_irfs_weights into numeric dictionaries."""
    entries, cur = [], None
    for st in body:
        keyword, _, rest = st.partition(" ")
        keyword, rest = keyword.strip(), rest.strip()
        if keyword == "var":
            cur = dict(variable=rest, shock=None, periods=None, values=None, weights=[1.0])
            entries.append(cur)
        elif cur is None or keyword not in ("varexo", "periods", "values", "weights"):
            raise DynareImportError(f"matched_irfs: cannot read {st!r}")
        elif keyword == "varexo":
            cur["shock"] = rest
        else:
            cur[keyword] = _numeric_vector(sympy, rest, env, funcs)
    for e in entries:
        if e["shock"] is None or e["periods"] is None or e["values"] is None:
            raise DynareImportError("matched_irfs: every var needs varexo, periods and values")
        for key in ("values", "weights"):
            # Dynare's preprocessor pads a shorter vector with its last entry;
            # matched_irfs_blocks only sees the expanded vector.
            if 0 < len(e[key]) < len(e["periods"]):
                e[key] += [e[key][-1]] * (len(e["periods"]) - len(e[key]))
            if len(e[key]) != len(e["periods"]):
                raise DynareImportError(f"matched_irfs: periods and {key} have different lengths")
    crosses, cur = [], {}
    for st in weight_body:
        keyword, _, rest = st.partition(" ")
        keyword, rest = keyword.strip(), rest.strip()
        if keyword not in ("var", "varexo", "periods", "weights"):
            raise DynareImportError(f"matched_irfs_weights: cannot read {st!r}")
        if keyword == "weights":
            cur["weights"] = _numeric_vector(sympy, rest, env, funcs)
            crosses.append(cur)
            cur = {}
            continue
        count = cur.get("_counts", {}).get(keyword, 0) + 1
        cur.setdefault("_counts", {})[keyword] = count
        if count > 2:
            raise DynareImportError("matched_irfs_weights: only two IRFs per entry")
        value = rest if keyword in ("var", "varexo") else _numeric_vector(sympy, rest, env, funcs)
        cur[f"{keyword}{count}"] = value
    if cur:
        raise DynareImportError("matched_irfs_weights: entry without weights")
    for e in crosses:
        if len(e["periods1"]) != len(e["periods2"]):
            raise DynareImportError("matched_irfs_weights: horizons have different lengths")
        if len(e["weights"]) == 1 and len(e["periods1"]) > 1:
            e["weights"] *= len(e["periods1"])
        if len(e["weights"]) != len(e["periods1"]):
            raise DynareImportError("matched_irfs_weights: horizons and weights differ")
        e.pop("_counts", None)
    return entries, crosses


_CALIB_LINE = re.compile(r"^(?P<head>.*?),\s*(?P<bound>[+-]|\[(?P<interval>.*)\])\s*$", re.S)
_CALIB_NAME = re.compile(r"^([A-Za-z_]\w*)\s*(?:\(\s*([+-]?\d+)\s*(?::\s*([+-]?\d+)\s*)?\))?$")


def _read_calibration_blocks(irf_body, moment_body, sympy, env, funcs, endo, exo):
    """Translate irf_calibration and moment_calibration into restriction tuples."""
    def dated(text, block):
        m = _CALIB_NAME.match(text.strip())
        if not m:
            raise DynareImportError(f"{block}: cannot read {text!r}")
        if m.group(2) is None:
            return m.group(1), None
        a = int(m.group(2))
        b = int(m.group(3)) if m.group(3) is not None else a
        return m.group(1), list(range(a, b + 1))

    def line(st, block):
        m = _CALIB_LINE.match(st)
        if not m:
            raise DynareImportError(f"{block}: cannot read {st!r}")
        head = [x.strip() for x in m.group("head").split(",")]
        if m.group("interval") is not None:
            parts = [x.strip() for x in m.group("interval").split(",")]
            if len(parts) != 2:
                raise DynareImportError(f"{block}: an interval is [lower, upper] in {st!r}")
            bound = tuple(_numeric(sympy, x, env, funcs) for x in parts)
        else:
            bound = m.group("bound")
        if len(head) != 2:
            raise DynareImportError(f"{block}: cannot read {st!r}")
        return head, bound

    irf, moments = [], []
    for st in irf_body:
        (first, shock), bound = line(st, "irf_calibration")
        v, periods = dated(first, "irf_calibration")
        if periods is None or v not in endo or shock not in exo:
            raise DynareImportError(f"irf_calibration: write 'variable(horizons), shock, "
                                    f"bound' with declared names, not {st!r}")
        irf.append((v, shock, periods, bound))
    for st in moment_body:
        (first, second), bound = line(st, "moment_calibration")
        v1, lead = dated(first, "moment_calibration")
        v2, lags = dated(second, "moment_calibration")
        if lead is not None or v1 not in endo or v2 not in endo:
            raise DynareImportError(f"moment_calibration: write 'variable, variable(lags), "
                                    f"bound' with declared names, not {st!r}")
        moments.append((v1, v2, lags or [0], bound))
    return irf, moments


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
    homotopy: dict = field(default_factory=dict)
    occbin: dict | None = field(default=None, repr=False)
    surprise: dict = field(default_factory=dict)
    trends: dict = field(default_factory=dict)
    varobs: list = field(default_factory=list)
    estimation_options: dict = field(default_factory=dict)
    matched_moments: list = field(default_factory=list)
    matched_irfs: list = field(default_factory=list)
    matched_irfs_weights: list = field(default_factory=list)
    mom_options: dict = field(default_factory=dict)
    irf_calibration: list = field(default_factory=list)
    moment_calibration: list = field(default_factory=list)
    relative_irf: bool = False
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

    def news(self, table: Mapping[int, Mapping], T: int,
             params: Mapping[str, float] | None = None) -> tuple[list[News], SteadyState]:
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

    def estimation_problem(self, data: Data, **kwargs: Any) -> EstimationProblem:
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

        kw: dict[str, Any]
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

    def identification(self, point: str | Mapping[str, float] = "prior_mean",
                       **kwargs: Any) -> Identification:
        """:func:`dsgelab.identification` of the ``estimated_params``, observing ``varobs``.

        ``point`` is where identification is checked: ``"prior_mean"`` (Dynare's
        default), ``"calibration"`` (the values the file assigns), or a mapping from
        the estimated names to values, which may leave some out. The shocks'
        covariance, the measurement errors and the steady state come from the file,
        as in :meth:`estimation_problem`. ``kwargs`` go to
        :func:`~dsgelab.identification` and override anything.
        """
        from .identification import identification

        if not self.estimated:
            raise DynareImportError("this .mod has no estimated_params block")
        if not self.varobs:
            raise DynareImportError("this .mod has no varobs")
        sd = {e: float(np.sqrt(self.cov[i, i])) for i, e in enumerate(self.exogenous)}
        params = dict(self.params)
        meas: dict[str, float] = {}
        values = {}
        for e in self.estimated:
            if isinstance(point, Mapping) and e["name"] in point:
                v = float(point[e["name"]])
            elif point == "prior_mean" or isinstance(point, Mapping):
                v = e["prior"].mean if e["prior"] is not None else (
                    e["init"] if np.isfinite(e["init"]) else np.nan)
            elif point == "calibration":
                if e["kind"] == "param":
                    v = params.get(e["target"], np.nan)
                elif e["role"] == "shock":
                    v = sd[e["target"]]
                else:
                    v = np.nan
            else:
                raise ValueError(f"point must be 'prior_mean', 'calibration' or a mapping, "
                                 f"not {point!r}")
            if not np.isfinite(v):
                raise ValueError(f"no value for {e['name']!r} at point={point!r}")
            values[e["name"]] = v
            if e["kind"] == "param":
                params[e["target"]] = v
            elif e["role"] == "shock":
                sd[e["target"]] = v
            else:
                meas[e["target"]] = v
        kw: dict[str, Any] = dict(free=values, observables=list(self.varobs), cov=sd,
                                  measurement=meas or None,
                                  transforms="log" if "loglinear" in self.estimation_options
                                  else "level",
                                  steady_fn=lambda p: self.steady_state(params=p))
        kw.update(kwargs)
        return identification(self.model, params, **kw)

    def method_of_moments(self, data: Data, method: str = "GMM", *,
                          moments: Sequence[str] | None = None, **kwargs: Any) -> MomentsResult:
        """:func:`dsgelab.method_of_moments` of the ``estimated_params``, on ``varobs``.

        The estimated names, their starting values and their bounds come from
        ``estimated_params`` (a missing starting value is the calibration, as with
        ``estimated_params_init(use_calibration)``), the moments from the
        ``matched_moments`` block, and the steady state from the file. ``kwargs``
        go to :func:`~dsgelab.method_of_moments` and override anything.
        """
        from .moment_estimation import method_of_moments

        if not self.estimated:
            raise DynareImportError("this .mod has no estimated_params block")
        if not self.varobs:
            raise DynareImportError("this .mod has no varobs")
        moments = list(moments if moments is not None else self.matched_moments)
        if not moments:
            raise DynareImportError("no matched_moments block: pass moments=")
        sd = {e: float(np.sqrt(self.cov[i, i])) for i, e in enumerate(self.exogenous)}
        S = np.array(self.cov, dtype=float)
        start, bounds = {}, {}
        for e in self.estimated:
            if e["kind"] == "param":
                cal = self.params.get(e["target"], np.nan)
            elif e["kind"] == "stderr":
                cal = sd.get(e["target"], np.nan)
            else:
                i, j = (self.exogenous.index(x) for x in e["target"])
                cal = S[i, j] / np.sqrt(S[i, i] * S[j, j]) if S[i, i] * S[j, j] > 0 else 0.0
            start[e["name"]] = e["init"] if np.isfinite(e["init"]) else cal
            lo = e["lower"] if np.isfinite(e["lower"]) else -np.inf
            hi = e["upper"] if np.isfinite(e["upper"]) else np.inf
            bounds[e["name"]] = (lo, hi)
        cov = np.zeros((len(self.model.shocks),) * 2)
        order = [self.model.shocks.index(e) for e in self.exogenous]
        cov[np.ix_(order, order)] = S
        perturbation_order = int(float(self.mom_options.get("order", "1") or "1"))
        kw: dict[str, Any] = dict(params=dict(self.params), free=start, bounds=bounds,
                                  observables=list(self.varobs), moments=moments,
                                  method=method, cov=cov, order=perturbation_order,
                                  steady_fn=lambda p: self.steady_state(params=p))
        kw.update(kwargs)
        return method_of_moments(self.model, data, **kw)

    def irf_matching(self, *, targets: Sequence[Sequence] | None = None,
                     **kwargs: Any) -> IRFMatchingResult:
        """:func:`dsgelab.irf_matching` from this file's estimation and IRF blocks.

        The frequentist first-order command is supported. ``kwargs`` override the
        imported targets, weighting and options; external targets can therefore
        replace the block without changing the model.
        """
        from .irf_matching import _targets, irf_matching

        if not self.estimated:
            raise DynareImportError("this .mod has no estimated_params block")
        if any(e["prior"] is not None for e in self.estimated):
            raise DynareImportError("Bayesian IRF matching is not supported; use a frequentist block")
        if self.mom_options.get("order", "1") not in ("", "1"):
            raise DynareImportError("IRF matching currently supports order=1 only")
        if self.mom_options.get("irf_matching_file") and "transform" not in kwargs:
            raise DynareImportError("a MATLAB irf_matching_file is not executed; pass transform=")
        sd = {e: float(np.sqrt(self.cov[i, i])) for i, e in enumerate(self.exogenous)}
        S = np.array(self.cov, dtype=float)
        start, bounds = {}, {}
        for e in self.estimated:
            if e["kind"] == "param":
                cal = self.params.get(e["target"], np.nan)
            elif e["kind"] == "stderr":
                cal = sd.get(e["target"], np.nan)
            else:
                i, j = (self.exogenous.index(x) for x in e["target"])
                cal = S[i, j] / np.sqrt(S[i, i] * S[j, j]) if S[i, i] * S[j, j] > 0 else 0.0
            start[e["name"]] = e["init"] if np.isfinite(e["init"]) else cal
            bounds[e["name"]] = (e["lower"] if np.isfinite(e["lower"]) else -np.inf,
                                 e["upper"] if np.isfinite(e["upper"]) else np.inf)
        cov = np.zeros((len(self.model.shocks),) * 2)
        order = [self.model.shocks.index(e) for e in self.exogenous]
        cov[np.ix_(order, order)] = S
        if targets is None:
            rows = []
            for entry in self.matched_irfs:
                rows += [(entry["variable"], entry["shock"], h, v, w)
                         for h, v, w in zip(entry["periods"], entry["values"], entry["weights"])]
            targets = rows
        if not targets:
            raise DynareImportError("no matched_irfs block: pass targets=")
        canonical, _, diagonal, _ = _targets(targets, self.model, self.varobs)
        W = np.diag(diagonal)
        where = {(v, e, h): i for i, (v, e, h, _) in enumerate(canonical)}
        for cross in self.matched_irfs_weights:
            for h1, h2, w in zip(cross["periods1"], cross["periods2"], cross["weights"]):
                key1 = (cross["var1"], cross["varexo1"], int(h1))
                key2 = (cross["var2"], cross["varexo2"], int(h2))
                if key1 not in where or key2 not in where:
                    raise DynareImportError("a matched_irfs_weights entry names no declared target")
                i, j = where[key1], where[key2]
                W[i, j] = W[j, i] = w
        kw: dict[str, Any] = dict(params=dict(self.params), free=start, bounds=bounds,
                                  observables=list(self.varobs), cov=cov, weighting=W,
                                  relative_irf=("relative_irf" in self.mom_options and
                                                self.mom_options["relative_irf"] in ("", "1", "true")),
                                  steady_fn=lambda p: self.steady_state(params=p))
        kw.update(kwargs)
        return irf_matching(self.model, targets, **kw)

    def _shock_cov(self) -> np.ndarray:
        """The file's shock covariance, in ``model.shocks`` order."""
        cov = np.zeros((len(self.model.shocks),) * 2)
        order = [self.model.shocks.index(e) for e in self.exogenous]
        cov[np.ix_(order, order)] = np.asarray(self.cov, dtype=float)
        return cov

    def _restriction_kwargs(self, kwargs: Mapping[str, Any]) -> dict:
        kw: dict[str, Any] = dict(irf=list(self.irf_calibration),
                                  moments=list(self.moment_calibration),
                                  relative_irf=self.relative_irf)
        kw.update(kwargs)
        if not kw["irf"] and not kw["moments"]:
            raise DynareImportError("no irf_calibration or moment_calibration block: "
                                    "pass irf= or moments=")
        return kw

    def check_restrictions(self, point: str | Mapping[str, float] = "calibration", *,
                           qz_criterium: float | None = None,
                           **kwargs: Any) -> RestrictionCheck:
        """:func:`dsgelab.check_restrictions` of this file's calibration blocks.

        The restrictions come from ``irf_calibration`` and ``moment_calibration``,
        with ``relative_irf`` if the block sets it. ``point`` is ``"calibration"``
        (the values the file assigns, the default), ``"prior_mean"``, or a mapping
        from the ``estimated_params`` names to values, which may leave some out.
        ``kwargs`` go to :func:`~dsgelab.check_restrictions` and override anything.
        """
        from .estimation import unpack_point
        from .restrictions import check_restrictions
        from .solve import solve

        kw = self._restriction_kwargs(kwargs)
        values: dict[str, float] = {}
        if point == "prior_mean":
            values = {e["name"]: e["prior"].mean for e in self.estimated
                      if e["prior"] is not None}
        elif isinstance(point, Mapping):
            values = {k: float(v) for k, v in point.items()}
        elif point != "calibration":
            raise ValueError(f"point must be 'calibration', 'prior_mean' or a mapping, "
                             f"not {point!r}")
        p, S, _ = unpack_point(self.model, dict(self.params), self._shock_cov(), {},
                               list(values), list(values.values()))
        sol = solve(self.model, p, steady=self.steady_state(params=p),
                    qz_criterium=qz_criterium)
        kw.setdefault("cov", S)
        return check_restrictions(sol, **kw)

    def sample_restrictions(self, draws: int = 1000, **kwargs: Any) -> RestrictionSample:
        """:func:`dsgelab.sample_restrictions` of this file's calibration blocks.

        The draws come from the priors of ``estimated_params``, and the parameters
        it leaves out keep the file's values, as in Dynare's sensitivity analysis.
        ``kwargs`` go to :func:`~dsgelab.sample_restrictions` and override
        anything; ``points=`` with ``names=`` evaluates given draws instead.
        """
        from .restrictions import sample_restrictions

        kw = self._restriction_kwargs(kwargs)
        priors = {e["name"]: e["prior"] for e in self.estimated if e["prior"] is not None}
        if not priors and kw.get("points") is None:
            raise DynareImportError("no priors in estimated_params: pass points= and names=")
        kw.setdefault("cov", self._shock_cov())
        kw.setdefault("steady_fn", lambda p: self.steady_state(params=p))
        return sample_restrictions(self.model, dict(self.params), priors, draws=draws, **kw)

    def steady_state(self, guess: Mapping[str, float] | None = None,
                     params: dict[str, float] | None = None, *,
                     analytic: bool = False, **kwargs: Any) -> SteadyState:
        """Solve the steady state, iterating on ``steady_state(x)`` terms if any.

        With ``analytic=True`` the file's ``steady_state_model`` (or ``initval``)
        IS the steady state: it is evaluated, checked against the model's
        equations by :func:`dsgelab.analytic_steady_state`, and refused if it
        leaves a variable out or does not solve them -- Dynare's behaviour.

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
        if analytic:
            # the file's steady_state_model AS the steady state: no Newton step, and
            # refused unless it zeroes the dynamic model's residuals
            from .steady import SteadyStateError, analytic_steady_state
            from .symbolic import _require_sympy
            sympy = _require_sympy()
            g, exo_lv, upd = _run_program(sympy, self.program, dict(params),
                                          _sympy_functions(sympy))
            if isinstance(params, dict):
                params.update(upd)
            missing = [v for v in self.endogenous if v not in g]
            if missing:
                raise SteadyStateError(
                    f"steady_state_model / initval give no closed form for {missing} "
                    f"(a MATLAB helper, or not declared): use analytic=False to solve for them")
            vals = self.full_guess(g)
            for name, (kind, of) in self.auxiliary.items():
                if kind == "shock":
                    vals[name] = float(params.get(f"EXO_SS_{of}", 0.0))
            return analytic_steady_state(self.model, params, vals,
                                         tol=kwargs.get("tol", 1e-10))
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

    def extended_path(self, shocks: ArrayLike | None = None, *,
                      steady: SteadyState | None = None, **kwargs: Any) -> ExtendedPath:
        """:func:`dsgelab.extended_path` with this file's covariance of the shocks."""
        from .extended_path import extended_path
        steady = self.steady_state() if steady is None else steady
        kwargs.setdefault("cov", self.cov)
        return extended_path(self.model, steady, self.params, shocks, **kwargs)

    def balanced_growth(self, params: Mapping[str, float] | None = None,
                        **kwargs: Any) -> BalancedGrowth:
        """:func:`dsgelab.balanced_growth`, with the exogenous variables held at their level.

        Dynare's ``bgp.write`` problem for this file: the states that carry the
        exogenous variables do not grow and sit at ``EXO_SS``.
        """
        from .bgp import balanced_growth
        params = self.params if params is None else params
        fixed = {name: float(params.get(f"EXO_SS_{of}", 0.0))
                 for name, (kind, of) in self.auxiliary.items() if kind == "shock"}
        fixed.update(kwargs.pop("fixed", {}) or {})
        return balanced_growth(self.model, params, fixed=fixed, **kwargs)

    def occbin_model(self) -> OccbinModel:
        """The :class:`~dsgelab.occbin.OccbinModel` of the file's tagged equations."""
        if self.occbin is None:
            raise DynareImportError("this .mod has no occbin_constraints")
        from .occbin import OccbinModel
        om = OccbinModel(build=self.occbin["build"], constraints=self.occbin["constraints"])
        om._models[frozenset()] = self.model
        return om

    def occbin_simulate(self, shocks: Mapping[str, Mapping[int, float]] | None = None, *,
                        steady: SteadyState | None = None, **kwargs: Any) -> OccbinResult:
        """:func:`dsgelab.occbin_simulate` with the file's surprise shocks and options.

        ``occbin_setup(simul_periods=)`` and ``occbin_solver(simul_maxit=,
        simul_curb_retrench, simul_check_ahead_periods=)`` set the defaults.
        """
        from .occbin import occbin_simulate
        o = (self.occbin or {}).get("options", {})
        kw = {}
        if "simul_periods" in o:
            kw["periods"] = int(o["simul_periods"])
        if "simul_maxit" in o:
            kw["maxit"] = int(o["simul_maxit"])
        if "simul_check_ahead_periods" in o:
            kw["check_ahead"] = int(o["simul_check_ahead_periods"])
        if "simul_curb_retrench" in o:
            kw["curb_retrench"] = o["simul_curb_retrench"] in ("", "1", "true")
        kw.update(kwargs)
        steady = self.steady_state() if steady is None else steady
        return occbin_simulate(self.occbin_model(), self.params,
                               self.surprise if shocks is None else shocks,
                               steady=steady, **kw)

    def name_of(self, v: str) -> str:
        """The Dynare name behind a model variable, or the variable itself."""
        kind, of = self.auxiliary.get(v, (None, v))
        return of


# ------------------------------------------------------------------- import
def from_mod(source: str | os.PathLike[str], *, defines: Mapping | None = None,
             derivatives: str = "jacobian",
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
    estim_body, varobs, estim_opts, mom_opts = [], [], {}, {}
    matched_body: list[str] = []
    irf_body: list[str] = []
    irf_weights_body: list[str] = []
    irf_calib_body: list[str] = []
    moment_calib_body: list[str] = []
    relative_irf = False
    homotopy_body = []
    deflators: dict = {}
    trend_vars: dict = {}
    occbin_body = []
    surprise_body = []
    occbin_opts: dict = {}
    i = 0
    while i < len(stmts):
        st = stmts[i]
        word_match = re.match(r"[A-Za-z_]\w*", st)
        word = word_match.group(0) if word_match else ""
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
            elif head == "shocks" and "surprise" in _csv(opts):
                surprise_body += body       # shocks(surprise): OccBin's unforeseen shocks
            elif head == "shocks":
                shocks += body
            elif head == "optim_weights":
                weights_body = body
            elif head == "osr_params_bounds":
                bounds_body = body
            elif head == "estimated_params":
                estim_body = body
            elif head == "matched_moments":
                matched_body = [" ".join(b.split()) for b in body]
            elif head == "matched_irfs":
                irf_body = [] if "overwrite" in _csv(opts) else irf_body
                irf_body += [" ".join(b.split()) for b in body]
            elif head == "matched_irfs_weights":
                irf_weights_body = [] if "overwrite" in _csv(opts) else irf_weights_body
                irf_weights_body += [" ".join(b.split()) for b in body]
            elif head == "irf_calibration":
                irf_calib_body = [] if "overwrite" in _csv(opts) else irf_calib_body
                irf_calib_body += [" ".join(b.split()) for b in body]
                relative_irf = relative_irf or "relative_irf" in _csv(opts)
            elif head == "moment_calibration":
                moment_calib_body = [] if "overwrite" in _csv(opts) else moment_calib_body
                moment_calib_body += [" ".join(b.split()) for b in body]
            elif head == "homotopy_setup":
                homotopy_body = body
            elif head == "occbin_constraints":
                occbin_body = body
            i = j + 1
            continue
        if word == "var" and not st.startswith("var_"):
            names_ = _declared(st, "var")
            vo = _options(st[len(word):])
            for key in ("deflator", "log_deflator"):
                if key in vo:
                    for n_ in names_:
                        deflators[n_] = ("log" if key == "log_deflator" else "mul", vo[key])
            endo += names_
        elif word in ("trend_var", "log_trend_var"):
            vo = _options(st[len(word):])
            key = "log_growth_factor" if word == "log_trend_var" else "growth_factor"
            if key not in vo:
                raise DynareImportError(f"{word} needs {key}=: {st!r}")
            for n_ in _declared(st, word):
                trend_vars[n_] = ("log" if word == "log_trend_var" else "mul", vo[key])
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
        elif word == "method_of_moments":
            mom_opts = _options(st[len(word):])
        elif word in ("occbin_setup", "occbin_solver"):
            occbin_opts.update(_options(st[len(word):]))
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
    for kind_, names_ in (("var", endo), ("varexo", exo), ("parameters", params)):
        dup = sorted({n_ for n_ in names_ if names_.count(n_) > 1})
        if dup:
            raise DynareImportError(f"names declared twice with {kind_}: {dup}")
    clash = (set(endo) & set(exo)) | (set(endo) & set(params)) | (set(exo) & set(params))
    if clash:
        raise DynareImportError(f"names declared twice: {sorted(clash)}")
    linear = "linear" in model_opts

    funcs = _sympy_functions(sympy)

    # --- parameters, in order of assignment
    # Dynare passes an assignment to an undeclared name through as MATLAB; the
    # examples use it for a constant read in the shocks block (example1's phi).
    pvals: dict[str, float] = {p_: float("nan") for p_ in params}
    matlab: dict[str, Any] = {}
    for pname, expr in param_text:
        try:
            val = _numeric(sympy, expr, {**matlab, **pvals}, funcs)
        except Exception:
            try:
                val = _numeric_vector(sympy, expr, {**matlab, **pvals}, funcs)
            except Exception:
                val = float("nan")
        (pvals if pname in pvals else matlab)[pname] = val

    # --- model equations (an OccBin equation has variants, tagged bind=/relax=)
    parser = _Parser(sympy, list(endo) + list(trend_vars), exo, params)
    eqs, meta = [], []
    for k, st in enumerate(model_stmts):
        tagm = re.match(r"^\s*\[([^\]]*)\]\s*", st)
        tags = _tags(tagm.group(1)) if tagm else {}
        st = st[tagm.end():] if tagm else st
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
        if "bind" in tags or "relax" in tags:
            if "name" not in tags:
                raise DynareImportError(f"an OccBin variant needs a name tag: {st!r}")
            meta.append((tags["name"], frozenset(_csv(tags.get("bind", ""))),
                         frozenset(_csv(tags.get("relax", "")))))
        else:
            meta.append(None)
    slot_names = []
    for m_ in meta:
        if m_ is not None and m_[0] not in slot_names:
            slot_names.append(m_[0])
    if deflators or trend_vars:
        eqs = _detrend(sympy, parser, eqs, endo, exo, deflators, trend_vars, pvals)
    n_equations = sum(1 for m_ in meta if m_ is None) + len(slot_names)
    planner_present = planner_text is not None or policy_cmd is not None
    if n_equations != len(endo) and not (planner_present and n_equations < len(endo)):
        raise DynareImportError(f"{n_equations} equations for {len(endo)} endogenous variables")

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

    translated = []
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
        translated.append((raw, e.xreplace(sub)))

    # one slot per equation; an OccBin slot holds its variants, and the model below is
    # the REFERENCE regime -- every constraint slack -- in which the steady state lives
    slots, seen = [], {}
    for (raw, e), m_ in zip(translated, meta):
        if m_ is None:
            slots.append((raw, e))
        else:
            name_, bind_, relax_ = m_
            if name_ not in seen:
                seen[name_] = len(slots)
                slots.append([])
            slots[seen[name_]].append((bind_, relax_, raw, e))
    exprs, labels = [], []
    for sl in slots:
        if isinstance(sl, tuple):
            labels.append(sl[0])
            exprs.append(sl[1])
        else:
            raw, e = _pick_variant(sl, frozenset())
            labels.append(raw)
            exprs.append(e)
    n_dynare_eqs = len(exprs)
    occbin_constraints = _occbin_constraints(occbin_body, parser, refs, exo, S)

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
    det_shocks: dict = {}
    env_all = {**env, **pvals, **guess}
    _read_shocks(shocks, cov, det_shocks, sympy, env_all, funcs, exo)

    estimated = [_estimated_entry(sympy, st, env_all, funcs, endo, exo) for st in estim_body]
    surprise: dict = {}
    _read_shocks(surprise_body, None, surprise, sympy, env_all, funcs, exo)
    occbin = None
    if occbin_body or any(not isinstance(sl, tuple) for sl in slots):
        aux_rows, aux_labels = list(exprs[n_dynare_eqs:]), list(labels[n_dynare_eqs:])

        def build(binding, _slots=slots):
            rows, labs = [], []
            for sl in _slots:
                raw, e = sl if isinstance(sl, tuple) else _pick_variant(sl, frozenset(binding))
                rows.append(e)
                labs.append(raw)
            tag = ",".join(sorted(binding))
            return model_from_expressions(rows + aux_rows, states=states, jumps=jumps,
                                          shocks=list(exo), parameters=all_params,
                                          name=f"{name}[{tag}]" if tag else name,
                                          derivatives=derivatives,
                                          source_equations=labs + aux_labels)

        names_used = {n for sl in slots if not isinstance(sl, tuple)
                      for b, r, _, _ in sl for n in b | r}
        unknown = names_used - set(occbin_constraints)
        if unknown:
            raise DynareImportError(f"equation tags name constraints that occbin_constraints "
                                    f"does not define: {sorted(unknown)}")
        occbin = dict(constraints=occbin_constraints, build=build, options=occbin_opts)
    homotopy = {}
    for st in homotopy_body:
        parts = [x.strip() for x in _split_top(st)]
        if parts[0] not in params:
            raise DynareImportError(f"homotopy_setup: {parts[0]!r} is not a parameter "
                                    f"(exogenous steady states are not read here)")
        if len(parts) == 3:
            homotopy[parts[0]] = (_numeric(sympy, parts[1], env_all, funcs),
                                  _numeric(sympy, parts[2], env_all, funcs))
        elif len(parts) == 2:
            homotopy[parts[0]] = (pvals[parts[0]], _numeric(sympy, parts[1], env_all, funcs))
        else:
            raise DynareImportError(f"homotopy_setup: cannot read {st!r}")

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

    matched_irfs, matched_irfs_weights = _read_irf_blocks(
        irf_body, irf_weights_body, sympy, env_all, funcs)
    irf_calibration, moment_calibration = _read_calibration_blocks(
        irf_calib_body, moment_calib_body, sympy, env_all, funcs, endo, exo)
    return ModFile(model=model, block=block, planner=planner, osr_params=osr_params_list,
                   histval=histval, histval_unread=histval_unread, det_shocks=det_shocks,
                   estimated=estimated, varobs=varobs, estimation_options=estim_opts,
                   matched_moments=matched_body, matched_irfs=matched_irfs,
                   matched_irfs_weights=matched_irfs_weights, mom_options=mom_opts,
                   irf_calibration=irf_calibration, moment_calibration=moment_calibration,
                   relative_irf=relative_irf,
                   homotopy=homotopy, occbin=occbin, surprise=surprise,
                   trends=dict(deflators=dict(deflators), trend_vars=dict(trend_vars)),
                   program=program,
                   optim_weights=weights, osr_bounds=bounds, params={**pvals, **{p_: float("nan") for p_ in steady_params}},
                   guess=guess, cov=cov, endogenous=list(endo), exogenous=list(exo),
                   auxiliary=auxiliary, linear=linear,
                   steady_operators=sorted(parser.steady_refs),
                   unset_params=[p_ for p_ in params if math.isnan(pvals[p_])],
                   missing_guess=missing, source=label)



def read_expectation_errors_csv(path: str | os.PathLike[str]) -> dict:
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


def _tags(text: str) -> dict:
    """``name = 'x', bind = 'c1,c2'`` -> ``{"name": "x", "bind": "c1,c2"}``."""
    out = {}
    for m in re.finditer(r"(\w+)\s*=\s*(['\"])(.*?)\2", text):
        out[m.group(1)] = m.group(3)
    return out


def _csv(text: str) -> list:
    return [x.strip() for x in text.split(",") if x.strip()]


def _pick_variant(variants, binding: frozenset):
    """The variant of an OccBin equation that applies when ``binding`` constraints bind."""
    hits = [(raw, e) for b, r, raw, e in variants if b <= binding and not (r & binding)]
    if len(hits) != 1:
        raise DynareImportError(
            f"{len(hits)} variants of an OccBin equation apply when {sorted(binding) or 'no'} "
            f"constraint(s) bind: {[v[2] for v in variants]}")
    return hits[0]


def _occbin_constraints(body, parser, refs, exo, S) -> dict:
    """``occbin_constraints`` -> ``{name: {"bind": (expr, op), "relax": (expr, op) | None}}``."""
    out, cur = {}, None
    for st in body:
        st = st.strip()
        m = re.match(r"name\s+['\"](\w+)['\"]\s*$", st)
        if m:
            cur = m.group(1)
            out[cur] = {"bind": None, "relax": None}
            continue
        m = re.match(r"(bind|relax|error_bind|error_relax)\s+(.*)$", st, flags=re.S)
        if not m or cur is None:
            raise DynareImportError(f"occbin_constraints: cannot read {st!r}")
        kind, text = m.group(1), m.group(2).strip()
        if kind.startswith("error_"):
            continue
        cm = re.match(r"(.*?)(<=|>=|<|>)(.*)$", text, flags=re.S)
        if not cm:
            raise DynareImportError(f"occbin_constraints: {kind} {text!r} is not a comparison")
        lhs, op, rhs = cm.group(1), cm.group(2), cm.group(3)
        e = parser.parse(f"({lhs}) - ({rhs})", f"occbin {kind} {cur}")
        sub = {}
        for s_ in e.free_symbols:
            if str(s_) in refs:
                nm, off = refs[str(s_)]
                if off != 0 or nm in exo:
                    raise DynareImportError("occbin_constraints may only use current-period "
                                            "endogenous variables")
                sub[s_] = S(nm)
        out[cur][kind] = (e.xreplace(sub), op)
    return out


def _read_shocks(body, cov, det, sympy, env_all, funcs, exo) -> None:
    """A ``shocks`` block: variances into ``cov`` (if given), ``periods``/``values`` into ``det``."""
    cur = None
    det_periods: list = []
    for st in body:
        m = re.match(r"var\s+([A-Za-z_]\w*)\s*$", st)
        if m:
            cur = m.group(1)
            if cur not in exo:
                raise DynareImportError(f"shocks: {cur!r} is not an exogenous variable")
            continue
        m = re.match(r"stderr\s+(.*)$", st, flags=re.S)
        if m and cur is not None and cov is not None:
            sd = _numeric(sympy, m.group(1), env_all, funcs)
            i_ = exo.index(cur)
            cov[i_, i_] = sd ** 2
            continue
        m = re.match(r"var\s+([A-Za-z_]\w*)\s*=(.*)$", st, flags=re.S)
        if m and cov is not None:
            i_ = exo.index(m.group(1))
            cov[i_, i_] = _numeric(sympy, m.group(2), env_all, funcs)
            continue
        m = re.match(r"(var|corr)\s+([A-Za-z_]\w*)\s*,\s*([A-Za-z_]\w*)\s*=(.*)$", st, flags=re.S)
        if m and cov is not None:
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
                    det.setdefault(cur, {})[t] = v
            continue


def _detrend(sympy, parser, eqs, endo, exo, deflators, trend_vars, pvals, tol=1e-8):
    """Dynare's detrending: a deflated variable is its stationary part times its deflator.

    ``var(deflator=D) x`` makes ``x`` the detrended variable: every ``x(k)`` of the
    model becomes ``x(k) * D(k)`` (``x(k) + D(k)`` with ``log_deflator``). A trend
    variable ``A`` declared by ``trend_var(growth_factor=g)`` then becomes its level
    at ``t`` times the growth since: ``A(k) = A g(+1) ... g(+k)``, and
    ``A(-k) = A / (g g(-1) ... g(-k+1))``; with ``log_trend_var`` the growth is a sum.
    Each equation must then depend on the level of each trend only through a factor
    -- the balanced-growth condition, Dynare's test: the cross derivative of
    ``log(lhs - rhs)`` in the trend and in any variable is zero. It is checked at
    several points, and the level is then set to 1 (0 for a log trend).
    """
    S = sympy.Symbol
    for n_ in deflators:
        if n_ not in endo:
            raise DynareImportError(f"deflator declared for {n_!r}, which is not endogenous")
    level = {a: S(f"__TREND__{a}") for a in trend_vars}

    def dated(name, off):
        sym = S(f"__D__{name}__{off}")
        parser.refs[str(sym)] = (name, off)
        return sym

    def shift(expr, k):
        sub = {}
        for s_ in expr.free_symbols:
            if str(s_) in parser.refs:
                nm, off = parser.refs[str(s_)]
                if nm in exo:
                    raise DynareImportError("a deflator or growth factor may not contain "
                                            "an exogenous variable")
                sub[s_] = dated(nm, off + k)
        return expr.xreplace(sub)

    defl = {n_: (kind, parser.parse(text, f"deflator of {n_}")) for n_, (kind, text)
            in deflators.items()}
    growth = {a: (kind, parser.parse(text, f"growth factor of {a}")) for a, (kind, text)
              in trend_vars.items()}
    for a, (_, g) in growth.items():
        bad = {parser.refs[str(s_)][0] for s_ in g.free_symbols if str(s_) in parser.refs} \
            & (set(trend_vars) | set(deflators))
        if bad:
            raise DynareImportError(f"the growth factor of {a} depends on trended {sorted(bad)}")

    def trend_at(a, off):
        kind, g = growth[a]
        if kind == "mul":
            out = level[a]
            for j in range(1, off + 1):
                out = out * shift(g, j)
            for j in range(0, -off):
                out = out / shift(g, -j)
        else:
            out = level[a]
            for j in range(1, off + 1):
                out = out + shift(g, j)
            for j in range(0, -off):
                out = out - shift(g, -j)
        return out

    out = []
    for k, (raw, e) in enumerate(eqs):
        sub = {}
        for s_ in e.free_symbols:
            ref = parser.refs.get(str(s_))
            if ref and ref[0] in defl:
                kind, d = defl[ref[0]]
                sub[s_] = s_ * shift(d, ref[1]) if kind == "mul" else s_ + shift(d, ref[1])
        e = e.xreplace(sub)
        sub = {}
        for s_ in e.free_symbols:
            ref = parser.refs.get(str(s_))
            if ref and ref[0] in trend_vars:
                sub[s_] = trend_at(*ref)
        e = e.xreplace(sub)
        _check_balanced(sympy, e, level, trend_vars, raw, k, pvals, tol)
        e = e.xreplace({level[a]: (1 if trend_vars[a][0] == "mul" else 0) for a in level})
        out.append((raw, e))
    return out


def _check_balanced(sympy, e, level, trend_vars, raw, k, pvals, tol):
    """d/dy [d log(e)/dT] = 0 for every trend level T: checked at random points."""
    import numpy as np
    present = [a for a in level if level[a] in e.free_symbols]
    if not present:
        return
    others = sorted((s_ for s_ in e.free_symbols if s_ not in level.values()), key=str)
    rng = np.random.default_rng(k)
    for a in present:
        T = level[a]
        dlog = sympy.diff(e, T)
        f = sympy.lambdify([T] + others, [e, dlog], "numpy")
        t0 = 1.0 if trend_vars[a][0] == "mul" else 0.0
        ratios = []
        for _ in range(6):
            vals = []
            for s_ in others:
                nm = str(s_)
                v = pvals.get(nm)
                vals.append(float(v) if v is not None and np.isfinite(v)
                            else float(rng.uniform(0.6, 1.4)))
            for t_ in (t0, t0 + 0.37):
                fe, fd = (complex(x) for x in f(t_, *vals))
                if abs(fe) > 1e-8 and np.isfinite(abs(fe)) and np.isfinite(abs(fd)):
                    ratios.append((t_, fd / fe))
        for t_ in {r[0] for r in ratios}:
            rs = [r[1] for r in ratios if r[0] == t_]
            if len(rs) > 1 and max(abs(r - rs[0]) for r in rs) > tol * max(1.0, abs(rs[0])):
                raise DynareImportError(
                    f"equation {k + 1} ({raw!r}) is not compatible with balanced growth: "
                    f"after detrending it still depends on the level of the trend {a!r} "
                    f"other than through a common factor")
