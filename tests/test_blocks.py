"""Derivação das condições de primeira ordem a partir do problema do agente."""

import numpy as np
import pytest

import dsgekit as dk

pytest.importorskip("sympy")
from dsgekit.blocks import FOCError, Problem, optimality_conditions  # noqa: E402
from dsgekit.examples import rbc  # noqa: E402
from dsgekit.symbolic import from_equations  # noqa: E402


def _problema_rbc():
    return Problem(
        "log(c)",
        controls=["c", "i"], states=["k"], exogenous=["a"],
        constraints=["c + i = a*k**alpha", "k(+1) = (1-delta)*k + i"],
        discount="beta", parameters=["alpha", "beta", "delta", "rho"], name="rbc")


def test_deriva_a_euler_do_rbc_e_resolve_no_mesmo_ponto():
    """O teste que importa não é a forma das equações: é a SOLUÇÃO.

    O bloco derivado tem quatro saltos (dois controles e dois multiplicadores) onde o modelo
    escrito à mão tem um. Se a derivação estiver certa, a matriz de transição dos estados que os
    dois produzem é a mesma — e para este RBC ela é conhecida no papel, `[[alpha, 1], [0, rho]]`.
    """
    foc = optimality_conditions(_problema_rbc())
    eqs = foc["equations"] + ["log(a(+1)) = rho*log(a) + eps_a"]
    model = from_equations(eqs, states=["k", "a"], jumps=foc["jumps"], shocks=["eps_a"],
                           parameters=["alpha", "beta", "delta", "rho"], name="rbc-foc")

    p = dict(alpha=0.36, beta=0.99, delta=1.0, rho=0.95)
    k, c = rbc.steady_levels(rbc.Params())
    guess = dict(k=k, a=1.0, c=c, i=k, lm_1=-1 / c, lm_2=-1 / c)
    sol = dk.solve(model, p, guess=guess)

    ik, ia = model.states.index("k"), model.states.index("a")
    assert np.allclose(sol.P[np.ix_([ik, ia], [ik, ia])],
                       [[p["alpha"], 1.0], [0.0, p["rho"]]], atol=1e-8)


def test_a_cpo_do_consumo_e_a_utilidade_marginal_menos_o_multiplicador():
    foc = optimality_conditions(_problema_rbc())
    assert any("1/c" in e and "lm_1" in e for e in foc["equations"])
    assert foc["multipliers"] == ["lm_1", "lm_2"]
    assert foc["jumps"] == ["c", "i", "lm_1", "lm_2"]


def test_cada_equacao_diz_de_onde_veio():
    """O resultado tem de ser auditável, e não um bloco opaco de álgebra."""
    foc = optimality_conditions(_problema_rbc())
    origens = set(foc["detail"].values())
    assert "CPO do controle c" in origens
    assert "Euler do estado k" in origens
    assert any(o.startswith("restrição 1") for o in origens)


def test_nome_nao_declarado_e_recusado():
    with pytest.raises(FOCError, match="não declarado"):
        optimality_conditions(Problem("log(c)", controls=["c"],
                                      constraints=["c = w*n"], parameters=["beta"]))


def test_problema_que_exigiria_t_mais_2_e_recusado():
    """Um custo de ajustamento sobre a VARIAÇÃO do estoque põe k(+2) na Euler.

    A função recusa dizendo qual estado o causou, em vez de devolver equações que o resto do
    pacote não sabe representar.
    """
    p = Problem("log(c)", controls=["c"], states=["k"],
                constraints=["c + k(+1) - k = -phi*(k(+1) - k)**2"],
                discount="beta", parameters=["beta", "phi"])
    with pytest.raises(FOCError, match="t\\+2"):
        optimality_conditions(p)
