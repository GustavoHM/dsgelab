"""Exportar um modelo declarado em dsgekit para Dynare e para MacroModelling.jl.

Só funciona para modelos construídos com :func:`dsgekit.symbolic.from_equations`: as equações
precisam existir como EXPRESSÕES para serem reimpressas noutra sintaxe, e uma função Python de
resíduos não pode ser descompilada. Um modelo declarado por callable levanta
:class:`ExportError` dizendo isso.

A tradução da datação
---------------------
É a única parte que não é mecânica, e é onde um porte feito à mão erra. O dsgekit tem DOIS tipos de
estado, e eles se exportam de maneiras diferentes:

* **Estoques predeterminados** — capital e afins, cuja linha do carregamento de choque ``Q`` é nula,
  isto é, que não recebem inovação diretamente. No dsgekit ``cur["k"]`` é o estoque DISPONÍVEL em
  ``t``. No Dynare isso se declara com ``predetermined_variables`` e escreve-se ``k`` sem defasagem;
  ``nxt["k"]`` vira ``k(+1)``. No MacroModelling, ``k[-1]`` e ``k[0]``.
* **Processos exógenos** — os que recebem inovação. Seguem o dating PADRÃO do Dynare, em que a
  inovação se realiza em ``t`` e move a variável em ``t``, e por isso a equação é escrita um período
  ATRÁS: ``nxt.a = rho*cur.a + eps`` vira ``a = rho*a(-1) + eps_a``. É a mesma equação numa data
  diferente, e é o que põe a inovação sem defasagem, como o Dynare exige.

Saltos seguem o dating padrão: ``cur["c"]`` vira ``c`` e ``nxt["c"]`` vira ``c(+1)``.

Esta é a convenção do porte que já existe neste repositório e que confere com o Python a 1,3e-05
pontos percentuais. Escrever os processos exógenos com lead em vez de os deslocar dá um modelo que
resolve e cujas correções de risco somem em silêncio.

E os desvios-padrão
-------------------
No Dynare vão no bloco ``shocks``; no MacroModelling as inovações são normais PADRÃO e o
desvio-padrão entra NA EQUAÇÃO. Esquecer isso no segundo caso dá variâncias unitárias e momentos
absurdos, e é por isso que o exportador multiplica ``sig_<choque>`` na equação e declara o parâmetro.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .model import Model, get_param
from .steady import SteadyState

__all__ = ["ExportError", "to_dynare", "to_macromodelling"]


class ExportError(RuntimeError):
    """Raised when a model cannot be exported (declared by callable, or missing pieces)."""


def _symbolic(model: Model) -> dict:
    sym = getattr(model, "symbolic", None)
    if sym is None:
        raise ExportError(
            f"{model.name!r} was declared with a Python function, and a function cannot be "
            f"decompiled into equations. Declare it with dsgekit.symbolic.from_equations to "
            f"export it.")
    return sym


def _classify(model: Model, sym: dict) -> tuple[list, list]:
    """Separa os estados em estoques PREDETERMINADOS e processos EXÓGENOS.

    Um estado é exógeno quando aparece numa equação que carrega uma inovação; os demais são
    estoques, cuja linha do carregamento de choque é nula porque uma decisão tomada em ``t`` não
    responde a um choque de ``t+1``. A classificação vem das EQUAÇÕES e não de uma solução, para
    que o exportador funcione sem resolver o modelo.
    """
    eps_syms = set(sym["symbols"]["eps"].values())
    cur, nxt = sym["symbols"]["cur"], sym["symbols"]["nxt"]
    exogenos, com_choque = set(), []
    for expr in sym["expressions"]:
        livres = expr.free_symbols
        if not (livres & eps_syms):
            continue
        com_choque.append(expr)
        for v in model.states:
            if cur[v] in livres or nxt[v] in livres:
                exogenos.add(v)
    predeterminados = [v for v in model.states if v not in exogenos]
    return predeterminados, sorted(exogenos, key=model.states.index)


def _shifted_ok(model: Model, sym: dict, expr) -> bool:
    """A equação pode ser reescrita um período atrás sem inventar defasagens de saltos?

    Deslocar uma equação para trás é legítimo — o que vale em todo ``t`` vale em ``t-1`` —, mas se
    ela contiver um SALTO em ``t`` esse salto vira ``c(-1)``, e o Dynare responde criando uma
    variável auxiliar e mudando a contagem de estados. Um processo exógeno puro não tem esse
    problema, e é para eles que o deslocamento existe.
    """
    cur = sym["symbols"]["cur"]
    livres = expr.free_symbols
    return not any(cur[v] in livres for v in model.jumps)


def _renamer(model: Model, sym: dict, style: str, *, predeterminados, shift: bool):
    """Substituição dos símbolos do dsgekit pelos nomes da linguagem de destino."""
    import sympy
    cur, nxt = sym["symbols"]["cur"], sym["symbols"]["nxt"]
    pre = set(predeterminados)
    sub = {}
    for v in model.names:
        if style == "dynare":
            if shift:                      # equação reescrita um período atrás
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
    """Equações na sintaxe de destino, mais as listas de estados predeterminados e exógenos."""
    printer = _printer()
    predeterminados, exogenos = _classify(model, sym)
    eps_syms = set(sym["symbols"]["eps"].values())
    sub_normal = _renamer(model, sym, style, predeterminados=predeterminados, shift=False)
    sub_shift = _renamer(model, sym, style, predeterminados=predeterminados, shift=True)
    out = []
    for expr in sym["expressions"]:
        tem_choque = bool(expr.free_symbols & eps_syms)
        if tem_choque and not _shifted_ok(model, sym, expr):
            raise ExportError(
                "uma equação com inovação também contém um salto datado em t, e deslocá-la um "
                "período atrás criaria uma defasagem de salto que muda a contagem de estados no "
                "Dynare. Isole o processo exógeno na sua própria equação.")
        out.append(printer.doprint(expr.xreplace(sub_shift if tem_choque else sub_normal)))
    return out, predeterminados, exogenos


def _shock_sd(model: Model, cov) -> np.ndarray:
    from .solution import shock_covariance
    S = shock_covariance(model.shocks, cov)
    return np.sqrt(np.maximum(np.diag(S), 0.0)), S


# ---------------------------------------------------------------------------
# Dynare
# ---------------------------------------------------------------------------

def to_dynare(model: Model, params, *, steady: SteadyState | None = None, cov=None,
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
    eqs, predeterminados, exogenos = _equations(model, sym, "dynare")
    pars = list(sym["parameters"])

    lines = [f"// {model.name} — GERADO por dsgekit.to_dynare. Não editar à mão.",
             "//",
             "// Datação. Estoques predeterminados entram em `predetermined_variables`: a variável",
             "// escrita sem defasagem é o estoque DISPONÍVEL em t, e X(+1) é o decidido em t.",
             "// Processos exógenos seguem o dating PADRÃO do Dynare — a inovação realiza-se em t e",
             "// move a variável em t —, e por isso a sua equação vai escrita um período atrás.",
             f"//   predeterminados : {', '.join(predeterminados) or '(nenhum)'}",
             f"//   exógenos        : {', '.join(exogenos) or '(nenhum)'}",
             "",
             f"var {', '.join(model.names)};",
             f"varexo {', '.join(model.shocks)};",
             ""]
    if predeterminados:
        lines += [f"predetermined_variables {' '.join(predeterminados)};", ""]
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

def to_macromodelling(model: Model, params, *, cov=None, path: str | Path | None = None,
                      name: str | None = None) -> str:
    """Write the model as a MacroModelling.jl ``@model``/``@parameters`` file.

    The innovations are standard normal in MacroModelling, so each standard deviation is written
    into the equation as ``sig_<shock>`` and declared as a parameter. Leaving it out is the mistake
    that produces unit variances and moments off by orders of magnitude.
    """
    sym = _symbolic(model)
    eqs, predeterminados, exogenos = _equations(model, sym, "macromodelling")
    pars = list(sym["parameters"])
    sd, _ = _shock_sd(model, cov)
    tag = name or model.name.replace("-", "_").replace(" ", "_")

    lines = [f"# {tag} — GERADO por dsgekit.to_macromodelling. Não editar à mão.",
             "# Datação: um estoque predeterminado disponível em t é `x[-1]`; um salto em t é",
             "# `x[0]`. As inovações são normais PADRÃO aqui, logo o desvio-padrão entra NA EQUAÇÃO.",
             f"#   predeterminados : {', '.join(predeterminados) or '(nenhum)'}",
             f"#   exógenos        : {', '.join(exogenos) or '(nenhum)'}",
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
