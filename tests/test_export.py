"""Exportadores para Dynare e MacroModelling.jl."""

import numpy as np
import pytest

import dsgekit as dk
from dsgekit.examples import rbc

pytest.importorskip("sympy")
from dsgekit.export import ExportError, to_dynare, to_macromodelling  # noqa: E402


def test_modelo_declarado_por_funcao_e_recusado():
    """Uma função Python não pode ser descompilada em equações, e o erro diz isso."""
    model, params, guess = rbc.build()
    with pytest.raises(ExportError, match="from_equations"):
        to_dynare(model, params)


def test_dynare_separa_estoques_predeterminados_de_processos_exogenos():
    """A distinção é o que faz o porte funcionar, e o cabeçalho a registra.

    `k` é estoque: entra em `predetermined_variables` e é escrito sem defasagem. `a` é processo
    exógeno: segue o dating padrão, e a sua equação vai um período atrás para que a inovação
    apareça sem defasagem, como o Dynare exige.
    """
    model, params, guess = rbc.build_symbolic()
    ss = dk.steady_state(model, guess, params)
    texto = to_dynare(model, params, steady=ss, cov={"eps_a": 0.007})
    assert "predetermined_variables k;" in texto
    assert "predeterminados : k" in texto
    assert "exógenos        : a" in texto
    assert "rho*log(a(-1))" in texto          # equação do AR deslocada
    assert "k(+1)" in texto                    # estoque decidido em t
    assert "var eps_a = 4.9" in texto


def test_dynare_escreve_covariancia_cruzada():
    model, params, guess = rbc.build_symbolic()
    cov = np.array([[0.009 ** 2]])
    texto = to_dynare(model, params, cov=cov)
    assert "var eps_a =" in texto
    assert "stoch_simul(order=1" in texto


def test_macromodelling_poe_o_desvio_padrao_na_equacao():
    """Nas convenções do MacroModelling as inovações são normais PADRÃO.

    Esquecer de multiplicar sig_* na equação dá variâncias unitárias e momentos fora de escala por
    ordens de grandeza — o erro que este teste existe para impedir.
    """
    model, params, guess = rbc.build_symbolic()
    texto = to_macromodelling(model, params, cov={"eps_a": 0.007})
    assert "sig_eps_a * eps_a[x]" in texto
    assert "sig_eps_a = 0.007" in texto
    assert "k[-1]" in texto and "c[1]" in texto
    assert "@model" in texto and "@parameters" in texto


def test_potencias_saem_com_circunflexo():
    """Python escreve `**`; as duas linguagens de destino, `^`."""
    model, params, guess = rbc.build_symbolic()
    texto = to_dynare(model, params)
    assert "**" not in texto
    assert "^" in texto
