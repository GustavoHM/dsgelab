"""Exporters to Dynare and MacroModelling.jl."""

import numpy as np
import pytest

import dsgelab as dsg
from dsgelab.examples import rbc

pytest.importorskip("sympy")
from dsgelab.export import ExportError, to_dynare, to_macromodelling  # noqa: E402


def test_model_declared_by_function_is_refused():
    """A Python function cannot be decompiled into equations, and the error says so."""
    model, params, guess = rbc.build()
    with pytest.raises(ExportError, match="from_equations"):
        to_dynare(model, params)


def test_dynare_separates_predetermined_stocks_from_exogenous_processes():
    """The port works because of this distinction, and the header records it.

    `k` is a stock. It goes into `predetermined_variables` and is written without a lag. `a` is an
    exogenous process. It follows the default timing, and its equation moves one period back so
    that the innovation appears without a lag, as Dynare requires.
    """
    model, params, guess = rbc.build_symbolic()
    ss = dsg.steady_state(model, guess, params)
    texto = to_dynare(model, params, steady=ss, cov={"eps_a": 0.007})
    assert "predetermined_variables k;" in texto
    assert "predetermined : k" in texto
    assert "exogenous     : a" in texto
    assert "rho*log(a(-1))" in texto          # AR equation shifted
    assert "k(+1)" in texto                    # stock decided in t
    assert "var eps_a = 4.9" in texto


def test_dynare_escreve_covariancia_cruzada():
    model, params, guess = rbc.build_symbolic()
    cov = np.array([[0.009 ** 2]])
    texto = to_dynare(model, params, cov=cov)
    assert "var eps_a =" in texto
    assert "stoch_simul(order=1" in texto


def test_macromodelling_poe_o_desvio_padrao_na_equacao():
    """In MacroModelling's conventions the innovations are STANDARD normals.

    Forgetting to multiply by sig_* in the equation gives unit variances and moments off by orders
    of magnitude, the error this test exists to prevent.
    """
    model, params, guess = rbc.build_symbolic()
    texto = to_macromodelling(model, params, cov={"eps_a": 0.007})
    assert "sig_eps_a * eps_a[x]" in texto
    assert "sig_eps_a = 0.007" in texto
    assert "k[-1]" in texto and "c[1]" in texto
    assert "@model" in texto and "@parameters" in texto


def test_powers_are_written_with_caret():
    """Python writes `**`, and the two target languages write `^`."""
    model, params, guess = rbc.build_symbolic()
    texto = to_dynare(model, params)
    assert "**" not in texto
    assert "^" in texto
