"""Terceira ordem: o bloco cúbico contra o sistema não linear."""

import numpy as np
import pytest

import dsgekit as dk
from dsgekit.examples import rbc

pytest.importorskip("sympy")


def _solucoes(sigma=0.007):
    model, params, guess = rbc.build_symbolic()
    ss = dk.steady_state(model, guess, params)
    first = dk.solve(model, params, steady=ss)
    second = dk.solve_second_order(first, cov={"eps_a": sigma})
    third = dk.solve_third_order(second, cov={"eps_a": sigma})
    return model, params, first, second, third


def test_as_tres_ordens_sobem_um_grau_cada():
    """A verificação decisiva: 2, 3 e 4 na ordem implícita de acurácia.

    Dividir o deslocamento pela metade tem de dividir o resíduo por quatro em primeira ordem, por
    oito em segunda e por dezesseis em terceira. Um coeficiente cúbico errado não sobrevive ao
    terceiro degrau — o resíduo continua caindo por oito, e a razão o denuncia.
    """
    _, _, first, second, third = _solucoes()
    assert all(1.9 < o < 2.1 for o in first.accuracy()["implied_order"])
    assert all(2.85 < o < 3.15 for o in second.accuracy()["implied_order"])
    assert all(3.8 < o < 4.2 for o in third.accuracy()["implied_order"])


def test_sylvester_de_terceira_ordem_resolve_exatamente():
    _, _, _, _, third = _solucoes()
    assert third.sylvester_residual < 1e-10


def test_os_termos_cubicos_sao_simetricos_nos_tres_indices():
    """Simetria exata na teoria; impô-la é o que tira o ruído da diferenciação da equação."""
    _, _, _, _, third = _solucoes()
    for W in (third.g_kkk, third.h_kkk):
        assert np.max(np.abs(W - np.transpose(W, (0, 2, 1, 3)))) < 1e-12
        assert np.max(np.abs(W - np.transpose(W, (0, 1, 3, 2)))) < 1e-12


def test_simulacao_podada_de_terceira_ordem_nao_explode():
    """Um mapa cúbico iterado diverge mais depressa que o quadrático; a poda é o que o segura."""
    _, _, _, _, third = _solucoes(sigma=0.02)
    sim = third.simulate(periods=20_000, seed=3)
    assert np.all(np.isfinite(sim["k"]))
    assert np.max(np.abs(sim["k"])) < 1.0


def test_modelo_linear_tem_terceira_ordem_desprezivel():
    """No NK, escrito em desvios logarítmicos, não há o que a terceira ordem acrescente.

    A tolerância é 1e-6 e não zero por uma razão que vale registrar: este modelo é declarado por
    uma FUNÇÃO, logo a hessiana vem de diferenças finitas e traz ruído da ordem de 1e-10; a
    terceira derivada diferencia esse objeto de novo, e o ruído sobe para 1e-8. Os termos de
    segunda ordem, que vêm da hessiana diretamente, saem em zero exato. Declarado pelo front end
    simbólico, com hessiana analítica, o cúbico também sairia em zero.
    """
    from dsgekit.examples import nk
    model, params, guess = nk.build()
    first = dk.solve(model, params, guess=guess)
    second = dk.solve_second_order(first, cov={"eps_a": 0.01, "eps_v": 0.0025})
    third = dk.solve_third_order(second, cov={"eps_a": 0.01, "eps_v": 0.0025})
    assert np.max(np.abs(second.g_kk)) == 0.0        # hessiana exata: zero exato
    assert np.max(np.abs(third.g_kkk)) < 1e-6        # ruído de diferenciar a hessiana
    assert np.max(np.abs(third.h_kkk)) < 1e-6
