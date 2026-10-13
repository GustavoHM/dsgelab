"""Export a model declared in dsgelab to Dynare and to MacroModelling.jl.

Works only for models built with :func:`dsgelab.symbolic.from_equations`: the equations have to
exist as EXPRESSIONS to be printed again in another syntax, and a Python residual function cannot
be decompiled. A model declared by callable raises :class:`ExportError` saying so.

Translating the dating
----------------------
This is the only part that is not mechanical, and it is where a port made by hand goes wrong.
dsgelab has TWO kinds of state, and they are exported in different ways:

* **Predetermined stocks**: capital and the like, whose row of the shock loading ``Q`` is zero,
  that is, which receive no innovation directly. In dsgelab ``cur["k"]`` is the stock AVAILABLE at
  ``t``. In Dynare this is declared with ``predetermined_variables`` and ``k`` is written without a
  lag; ``nxt["k"]`` becomes ``k(+1)``. In MacroModelling, ``k[-1]`` and ``k[0]``.
* **Exogenous processes**: those that receive an innovation. They follow Dynare's STANDARD dating,
  in which the innovation is realised at ``t`` and moves the variable at ``t``, so the equation is
  written one period EARLIER: ``nxt.a = rho*cur.a + eps`` becomes ``a = rho*a(-1) + eps_a``. It is
  the same equation at a different date, and it is what leaves the innovation without a lag, as
  Dynare requires.

Jumps follow the standard dating: ``cur["c"]`` becomes ``c`` and ``nxt["c"]`` becomes ``c(+1)``.

This is the convention of the port that already exists in this repository, which agrees with the
Python to 1.3e-05 percentage points. Writing the exogenous processes with a lead instead of
shifting them gives a model that solves and whose risk corrections vanish silently.

And the standard deviations
---------------------------
In Dynare they go in the ``shocks`` block; in MacroModelling the innovations are STANDARD normal
and the standard deviation enters IN THE EQUATION. Forgetting this in the second case gives unit
variances and absurd moments, which is why the exporter multiplies by ``sig_<shock>`` in the
equation and declares the parameter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .model import Covariance, Model, Params, get_param
from .steady import SteadyState

__all__ = ["ExportError", "to_dynare", "to_macromodelling"]


class ExportError(RuntimeError):
    """Raised when a model cannot be exported (declared by callable, or missing pieces)."""


def _symbolic(model: Model) -> dict:
    sym = getattr(model, "symbolic", None)
    if sym is None:
        raise ExportError(
            f"{model.name!r} was declared with a Python function, and a function cannot be "
            f"decompiled into equations. Declare it with dsgelab.symbolic.from_equations to "
            f"export it.")
    return sym


def _classify(model: Model, sym: dict) -> tuple[list, list]:
    """Split the states into PREDETERMINED stocks and EXOGENOUS processes.

    A state is exogenous when it appears in an equation that carries an innovation; the others are
    stocks, whose row of the shock loading is zero because a decision taken at ``t`` does not
    respond to a shock at ``t+1``. The classification comes from the EQUATIONS and not from a
    solution, so that the exporter works without solving the model.
    """
    eps_syms = set(sym["symbols"]["eps"].values())
    cur, nxt = sym["symbols"]["cur"], sym["symbols"]["nxt"]
    exogenous, with_shock = set(), []
    for expr in sym["expressions"]:
        free = expr.free_symbols
        if not (free & eps_syms):
            continue
        with_shock.append(expr)
        for v in model.states:
            if cur[v] in free or nxt[v] in free:
                exogenous.add(v)
    predetermined = [v for v in model.states if v not in exogenous]
    return predetermined, sorted(exogenous, key=model.states.index)


def _shifted_ok(model: Model, sym: dict, expr) -> bool:
    """Can the equation be rewritten one period earlier without inventing lags of jumps?

    Shifting an equation back is legitimate, since what holds at every ``t`` holds at ``t-1``, but
    if it contains a JUMP at ``t`` that jump becomes ``c(-1)``, and Dynare responds by creating an
    auxiliary variable and changing the count of states. A pure exogenous process does not have
    this problem, and the shift exists for those.
    """
    cur = sym["symbols"]["cur"]
    free = expr.free_symbols
    return not any(cur[v] in free for v in model.jumps)


def _renamer(model: Model, sym: dict, style: str, *, predetermined, shift: bool):
    """Replace dsgelab's symbols with the names of the target language."""
    import sympy
    cur, nxt = sym["symbols"]["cur"], sym["symbols"]["nxt"]
    pre = set(predetermined)
    sub = {}
    for v in model.names:
        if style == "dynare":
            if shift:                      # equation rewritten one period earlier
                now, ahead = (f"{v}(-2)", f"{v}(-1)") if v in pre else (f"{v}(-1)", v)
            else:
                now, ahead = (v, f"{v}(+1)") if v in pre else (v, f"{v}(+1)")
        else:                              # macromodelling
            if shift:
                now, ahead = (f"{v}[-2]", f"{v}[-1]") if v in pre else (f"{v}[-1]", f"{v}[0]")
            else:
                now, ahead = (f"{v}[-1]", f"{v}[0]") if v in pre else (f"{v}[0]", f"{v}[1]")
        sub[cur[v]] = sympy.Symbol(now)
        sub[nxt[v]] = sympy.Symbol(ahead)
    for s in model.shocks:
        target = s if style == "dynare" else f"sig_{s} * {s}[x]"
        sub[sym["symbols"]["eps"][s]] = sympy.Symbol(target)
    return sub


def _printer():
    """A printer that writes powers as ``^`` — both target languages use it, Python does not."""
    from sympy.printing.str import StrPrinter

    class _P(StrPrinter):
        def _print_Pow(self, expr, rational=False):
            base = self.parenthesize(expr.base, 60)
            exp = self.parenthesize(expr.exp, 60)
            return f"{base}^{exp}"

    return _P(settings={"order": "none"})


def _equations(model: Model, sym: dict, style: str) -> tuple[list[str], list, list]:
    """Equations in the target syntax, plus the lists of predetermined and exogenous states."""
    printer = _printer()
    predetermined, exogenous = _classify(model, sym)
    eps_syms = set(sym["symbols"]["eps"].values())
    sub_normal = _renamer(model, sym, style, predetermined=predetermined, shift=False)
    sub_shift = _renamer(model, sym, style, predetermined=predetermined, shift=True)
    out = []
    for expr in sym["expressions"]:
        has_shock = bool(expr.free_symbols & eps_syms)
        if has_shock and not _shifted_ok(model, sym, expr):
            raise ExportError(
                "an equation with an innovation also contains a jump dated t, and shifting it one "
                "period back would create a lagged jump that changes the count of states in "
                "Dynare. Isolate the exogenous process in its own equation.")
        out.append(printer.doprint(expr.xreplace(sub_shift if has_shock else sub_normal)))
    return out, predetermined, exogenous


def _shock_sd(model: Model, cov) -> np.ndarray:
    from .solution import shock_covariance
    S = shock_covariance(model.shocks, cov)
    return np.sqrt(np.maximum(np.diag(S), 0.0)), S


# ---------------------------------------------------------------------------
# Dynare
# ---------------------------------------------------------------------------

def to_dynare(model: Model, params: Params, *, steady: SteadyState | None = None,
              cov: Covariance = None,
              path: str | Path | None = None, order: int = 1, irf: int = 40,
              extra: str = "") -> str:
    """Write the model as a Dynare ``.mod`` file and return its text.

    Parameters
    ----------
    steady : SteadyState, optional
        Written into an ``initval`` block. Dynare then runs ``steady`` from it. Supplying one
        matters: Dynare's own solver stops at a looser tolerance than this package's, and starting
        it at an already-solved point removes that as a source of disagreement.
    cov : mapping, array or None
        Shock covariance. Off-diagonal terms are written as ``var e, u = ...``.
    order, irf :
        Passed to ``stoch_simul``.
    extra : str
        Appended verbatim before ``stoch_simul``, for options this exporter does not model.
    """
    sym = _symbolic(model)
    eqs, predetermined, exogenous = _equations(model, sym, "dynare")
    pars = list(sym["parameters"])

    lines = [f"// {model.name} — GENERATED by dsgelab.to_dynare. Do not edit by hand.",
             "//",
             "// Dating. Predetermined stocks go in `predetermined_variables`: the variable",
             "// written without a lag is the stock AVAILABLE at t, and X(+1) is the one decided",
             "// at t. Exogenous processes follow Dynare's STANDARD dating (the innovation is",
             "// realised at t and moves the variable at t), so their equation is written one",
             "// period earlier.",
             f"//   predetermined : {', '.join(predetermined) or '(none)'}",
             f"//   exogenous     : {', '.join(exogenous) or '(none)'}",
             "",
             f"var {', '.join(model.names)};",
             f"varexo {', '.join(model.shocks)};",
             ""]
    if predetermined:
        lines += [f"predetermined_variables {' '.join(predetermined)};", ""]
    lines.append(f"parameters {', '.join(pars)};")
    for p in pars:
        lines.append(f"{p} = {float(get_param(params, p))!r};")
    lines += ["", "model;"]
    lines += [f"    {e} = 0;" for e in eqs]
    lines += ["end;", ""]

    if steady is not None:
        lines.append("initval;")
        lines += [f"    {v} = {float(steady[v])!r};" for v in model.names]
        lines += [f"    {s} = 0;" for s in model.shocks]
        lines += ["end;", "", "steady;", "check;", ""]

    sd, S = _shock_sd(model, cov)
    lines.append("shocks;")
    for i, s in enumerate(model.shocks):
        lines.append(f"    var {s} = {float(S[i, i])!r};")
    for i, si in enumerate(model.shocks):
        for j, sj in enumerate(model.shocks):
            if j > i and abs(S[i, j]) > 0:
                lines.append(f"    var {si}, {sj} = {float(S[i, j])!r};")
    lines += ["end;", ""]
    if extra:
        lines += [extra, ""]
    lines.append(f"stoch_simul(order={order}, irf={irf}, nograph);")

    text = "\n".join(lines) + "\n"
    if path is not None:
        Path(path).write_text(text, encoding="utf-8")
    return text


# ---------------------------------------------------------------------------
# MacroModelling.jl
# ---------------------------------------------------------------------------

def to_macromodelling(model: Model, params: Params, *, cov: Covariance = None,
                      path: str | Path | None = None,
                      name: str | None = None) -> str:
    """Write the model as a MacroModelling.jl ``@model``/``@parameters`` file.

    The innovations are standard normal in MacroModelling, so each standard deviation is written
    into the equation as ``sig_<shock>`` and declared as a parameter. Leaving it out is the mistake
    that produces unit variances and moments off by orders of magnitude.
    """
    sym = _symbolic(model)
    eqs, predetermined, exogenous = _equations(model, sym, "macromodelling")
    pars = list(sym["parameters"])
    sd, _ = _shock_sd(model, cov)
    tag = name or model.name.replace("-", "_").replace(" ", "_")

    lines = [f"# {tag} — GENERATED by dsgelab.to_macromodelling. Do not edit by hand.",
             "# Dating: a predetermined stock available at t is `x[-1]`; a jump at t is",
             "# `x[0]`. The innovations are STANDARD normal here, so the standard deviation enters",
             "# IN THE EQUATION.",
             f"#   predetermined : {', '.join(predetermined) or '(none)'}",
             f"#   exogenous     : {', '.join(exogenous) or '(none)'}",
             "",
             "using MacroModelling",
             "",
             f"@model {tag} begin"]
    lines += [f"    {e} = 0" for e in eqs]
    lines += ["end", "", f"@parameters {tag} begin"]
    for p in pars:
        lines.append(f"    {p} = {float(get_param(params, p))!r}")
    for i, s in enumerate(model.shocks):
        lines.append(f"    sig_{s} = {float(sd[i])!r}")
    lines += ["end", ""]

    text = "\n".join(lines) + "\n"
    if path is not None:
        Path(path).write_text(text, encoding="utf-8")
    return text
