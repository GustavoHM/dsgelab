"""Derivar as condições de primeira ordem a partir do problema do agente.

Opcional: requer ``sympy`` (``pip install dsgelab[symbolic]``).

O que faz, e o que não faz
--------------------------
Recebe um problema de otimização recursivo declarado — função-objetivo do período, variáveis de
escolha, estados e restrições — e devolve as CPOs como EQUAÇÕES EM TEXTO, prontas para
:func:`dsgelab.symbolic.from_equations`. O que se ganha não é digitação: é que a álgebra deixa de
ser feita à mão. Derivar um lagrangiano com quatro restrições e três estados à mão é onde entram os
erros de sinal e os termos esquecidos, e eles não aparecem no estado estacionário.

O que NÃO faz: escolher o problema por você. Condições de transversalidade, restrições de
desigualdade, complementaridade e envelopes não padrão continuam sendo do modelador — o pacote
deriva o que você declarou, e a declaração é a modelagem.

A matemática
------------
Com ``y`` os controles, ``k`` os estados e ``h_j`` as restrições escritas como ``= 0``, o
lagrangiano do período é

    L_t = u(y_t, k_t) + SUM_j lm_j,t * h_j(y_t, k_t, k_{t+1})

e as condições são

    dL_t/dy_t = 0                                    para cada controle
    dL_t/dk_{t+1} + beta * E_t[ dL_{t+1}/dk_{t+1} ] = 0    para cada estado

A segunda é a Euler, e o segundo termo é a derivada do lagrangiano do período SEGUINTE em relação ao
estado que ele herda — obtida deslocando ``dL_t/dk_t`` um período à frente. Se esse deslocamento
produzir um termo em ``t+2``, o problema não é representável em dois períodos e a função recusa,
dizendo qual estado o causou, em vez de devolver equações erradas.
"""

from __future__ import annotations

import re
from typing import Mapping, Sequence

__all__ = ["FOCError", "optimality_conditions", "Problem"]

_LEAD = re.compile(r"\b([A-Za-z_]\w*)\s*\(\s*\+?\s*1\s*\)")


class FOCError(ValueError):
    """Raised for a malformed problem, an undeclared name or a two-period-ahead term."""


def _require_sympy():
    try:
        import sympy  # noqa: F401
    except ImportError as exc:      # pragma: no cover
        raise FOCError("a derivação de CPOs precisa de sympy: "
                       "pip install 'dsgelab[symbolic]'") from exc
    import sympy
    return sympy


def _parse(text, local_dict, global_dict, rotulo):
    from sympy.parsing.sympy_parser import parse_expr, standard_transformations
    t = _LEAD.sub(r"\1__next", str(text))
    if t.count("=") > 1:
        raise FOCError(f"{rotulo}: mais de um '='")
    if "=" in t:
        lhs, rhs = t.split("=")
        t = f"({lhs}) - ({rhs})"
    try:
        return parse_expr(t, local_dict=local_dict, global_dict=global_dict,
                          transformations=standard_transformations)
    except Exception as exc:
        raise FOCError(f"{rotulo}: não consigo analisar {text!r} ({exc})") from exc


class Problem:
    """Um problema de otimização recursivo, declarado.

    Parameters
    ----------
    objective : str
        Retorno do PERÍODO, ``u(y_t, k_t)``. O fator de desconto entra por ``discount`` e não aqui.
    controls : sequence of str
        Variáveis escolhidas em ``t`` que não são estados.
    states : sequence of str
        Variáveis predeterminadas. O valor escolhido em ``t`` é ``k(+1)``.
    constraints : sequence of str
        Restrições de igualdade, ``"lhs = rhs"`` ou expressão que se anula. Cada uma ganha um
        multiplicador.
    discount : str
        Nome do parâmetro de desconto.
    parameters : sequence of str
        Todo nome que não é variável nem multiplicador. Um nome não declarado levanta erro em vez
        de virar símbolo livre: um erro de digitação não pode virar um parâmetro novo.
    exogenous : sequence of str, optional
        Variáveis que entram no problema e o agente NÃO escolhe — produtividade, preços tomados
        como dados. Não geram CPO.
    multipliers : sequence of str, optional
        Nomes dos multiplicadores, na ordem das restrições. Ausentes, chamam-se ``lm_1``, ``lm_2``…
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
            raise FOCError(f"{n} restrições e {len(self.multipliers)} multiplicadores")
        todos = self.controls + self.states + self.exogenous + self.multipliers
        dup = {v for v in todos if todos.count(v) > 1}
        if dup:
            raise FOCError(f"nomes repetidos entre variáveis e multiplicadores: {sorted(dup)}")

    def __repr__(self) -> str:
        return (f"Problem({self.name!r}, {len(self.controls)} controles, "
                f"{len(self.states)} estados, {len(self.constraints)} restrições)")


def optimality_conditions(problem: Problem) -> dict:
    """Deriva as CPOs. Devolve equações em texto, prontas para ``from_equations``.

    Returns
    -------
    dict com
        ``equations``   as CPOs mais as restrições, como strings;
        ``jumps``       controles e multiplicadores — as variáveis não predeterminadas que o
                        problema introduz;
        ``states``      os estados, na ordem declarada;
        ``multipliers`` os nomes usados;
        ``detail``      um mapa de cada equação à sua origem (qual controle, qual estado, qual
                        restrição), para que o resultado seja auditável e não um bloco opaco.
    """
    sympy = _require_sympy()
    p = problem

    variaveis = p.controls + p.states + p.exogenous + p.multipliers
    cur = {v: sympy.Symbol(v) for v in variaveis}
    nxt = {v: sympy.Symbol(f"{v}__next") for v in variaveis}
    nxt2 = {v: sympy.Symbol(f"{v}__next2") for v in variaveis}
    par = {q: sympy.Symbol(q) for q in list(p.parameters) + [p.discount]}

    local = {}
    local.update(cur)
    local.update({f"{v}__next": nxt[v] for v in variaveis})
    local.update(par)
    conhecidos = set(cur.values()) | set(nxt.values()) | set(par.values())
    funcoes = {"log": sympy.log, "exp": sympy.exp, "sqrt": sympy.sqrt, "Abs": sympy.Abs,
               "sin": sympy.sin, "cos": sympy.cos, "Min": sympy.Min, "Max": sympy.Max}
    local_dict = dict(local); local_dict.update(funcoes)
    global_dict = {"Symbol": sympy.Symbol, "Integer": sympy.Integer,
                   "Float": sympy.Float, "Rational": sympy.Rational,
                   "factorial": sympy.factorial}

    def analisa(texto, rotulo):
        e = _parse(texto, local_dict, global_dict, rotulo)
        desconhecidos = {s for s in e.free_symbols if s not in conhecidos}
        if desconhecidos:
            raise FOCError(
                f"{rotulo}: nome(s) não declarado(s) {sorted(str(s) for s in desconhecidos)}. "
                f"Declare-os em controls, states, exogenous ou parameters.")
        return e

    u = analisa(p.objective, "objetivo")
    hs = [analisa(c, f"restrição {i + 1}") for i, c in enumerate(p.constraints)]

    # lagrangiano do período
    L = u + sum(cur[m] * h for m, h in zip(p.multipliers, hs))

    def avanca(expr):
        """Desloca a expressão um período à frente: x -> x(+1), x(+1) -> x(+2)."""
        return expr.xreplace({nxt[v]: nxt2[v] for v in variaveis}) \
                   .xreplace({cur[v]: nxt[v] for v in variaveis})

    printer = _plain_printer(sympy)
    equacoes, detalhe, ordem_jumps = [], {}, []

    # (1) CPO de cada controle
    for c in p.controls:
        # Sem `simplify`: ele às vezes põe a expressão sobre um denominador que é uma VARIÁVEL,
        # e um resíduo dividido por k(+1) só é equivalente enquanto k(+1) não se aproxima de zero.
        # A derivada crua é exatamente a CPO.
        foc = sympy.expand(sympy.diff(L, cur[c]))
        txt = printer(foc)
        equacoes.append(txt)
        detalhe[txt] = f"CPO do controle {c}"

    # (2) Euler de cada estado
    for k in p.states:
        agora = sympy.diff(L, nxt[k])                  # efeito de escolher k(+1) hoje
        depois = avanca(sympy.diff(L, cur[k]))         # o que k(+1) rende amanhã
        proibidos = [v for v in variaveis if nxt2[v] in depois.free_symbols]
        if proibidos:
            raise FOCError(
                f"a Euler do estado {k!r} exige termos em t+2 ({sorted(proibidos)}): o problema "
                f"não é representável em dois períodos. Introduza a variável em t+1 como um "
                f"controle com a sua própria definição.")
        euler = sympy.expand(agora + par[p.discount] * depois)
        txt = printer(euler)
        equacoes.append(txt)
        detalhe[txt] = f"Euler do estado {k}"

    # (3) as restrições, que também são equações do modelo
    for i, (c, h) in enumerate(zip(p.constraints, hs), start=1):
        txt = printer(h)
        equacoes.append(txt)
        detalhe[txt] = f"restrição {i}: {c}"

    ordem_jumps = list(p.controls) + list(p.multipliers)
    return dict(equations=equacoes, jumps=ordem_jumps, states=list(p.states),
                multipliers=list(p.multipliers), detail=detalhe)


def _plain_printer(sympy):
    """Imprime na sintaxe de ``from_equations``: ``x__next`` volta a ser ``x(+1)``."""
    from sympy.printing.str import StrPrinter
    pr = StrPrinter(settings={"order": "none"})

    def do(expr):
        s = pr.doprint(expr)
        return re.sub(r"\b([A-Za-z_]\w*)__next\b", r"\1(+1)", s)

    return do
