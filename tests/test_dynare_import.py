"""Reading Dynare .mod files.

The comparison against Dynare itself, on Dynare's own examples, is
validation/compare_import.py. These tests need neither Dynare nor Octave.
"""

import numpy as np
import pytest

pytest.importorskip("sympy")

import dsgelab as dk  # noqa: E402
from dsgelab.dynare import DynareImportError, from_mod  # noqa: E402
from dsgelab.examples import rbc  # noqa: E402
from dsgelab.export import to_dynare  # noqa: E402


def test_round_trip_through_the_exporter():
    """to_dynare writes a .mod; from_mod reads it back; the dynamics must not move.

    The exporter writes the stock k as a predetermined variable. Read back, `k` is
    Dynare's series -- the stock decided at t -- which is this package's k(+1).
    """
    model, params, guess = rbc.build_symbolic()
    ss = dk.steady_state(model, guess, params)
    sol = dk.solve(model, params, steady=ss)
    text = to_dynare(model, params, steady=ss, cov={"eps_a": 0.007})

    mf = from_mod(text)
    ss2 = mf.steady_state()
    sol2 = dk.solve(mf.model, mf.params, steady=ss2)
    for v in ("k", "c", "a"):
        assert abs(ss2.values[v] - ss.values[v]) < 1e-12

    a = sol.irf("eps_a", size=0.007, periods=30, levels=True)
    b = sol2.irf("eps_a", size=0.007, periods=30, levels=True)
    assert np.max(np.abs(a["c"] - b["c"])) < 1e-12
    assert np.max(np.abs(a["a"] - b["a"])) < 1e-12
    assert np.max(np.abs(a["k"][1:] - b["k"][:-1])) < 1e-12


MOD = r"""
// comments of both kinds, % and /* */
@#define habit = 1
var y c k a (long_name='technology (log, AR(1))') i;
varexo e $\varepsilon$;
parameters alpha beta delta rho h;
alpha = 0.33; beta = 0.99; delta = 0.025; rho = 0.9;
@#if habit == 1
h = 0;            % the habit term below is switched off by value, not by macro
@#else
h = 0.5;
@#endif
model;
# mpk = alpha*exp(a(+1))*k^(alpha-1);
[name='Euler']
1/c = beta/c(+1)*(mpk + 1 - delta);
y = exp(a)*k(-1)^alpha;
k = (1-delta)*k(-1) + i;
y = c + i;
a = rho*a(-1) + e;
end;
initval;
k = ((1/beta - 1 + delta)/alpha)^(1/(alpha-1));
y = k^alpha; i = delta*k; c = y - i; a = 0;
end;
shocks;
var e; stderr 0.01;
end;
stoch_simul(order=1, irf=20);
"""


def test_features_and_the_neoclassical_closed_form():
    mf = from_mod(MOD)
    assert mf.endogenous == ["y", "c", "k", "a", "i"]
    assert mf.params["h"] == 0.0                         # the macro took the first branch
    assert mf.cov[0, 0] == pytest.approx(1e-4)
    assert set(mf.auxiliary) == {"LAG1_k", "LAG1_a", "SHK_e"}
    ss = mf.steady_state()
    k = ((1 / 0.99 - 1 + 0.025) / 0.33) ** (1 / (0.33 - 1))
    assert ss.values["k"] == pytest.approx(k, rel=1e-12)
    sol = dk.solve(mf.model, mf.params, steady=ss)
    irf = sol.irf("e", size=0.01, periods=10, levels=True)
    assert irf["a"][0] == pytest.approx(0.01, rel=1e-12)  # Dynare dating: a_0 = e_0
    assert irf["a"][1] == pytest.approx(0.009, rel=1e-12)


def test_defines_override_the_file():
    mf = from_mod(MOD, defines={"habit": 0})
    assert mf.params["h"] == 0.5


def test_leads_beyond_one_and_shocks_inside_equations():
    text = """
    var x z;
    varexo u;
    parameters rho;
    rho = 0.5;
    model;
    x = 0.5*x(+2) + exp(u)*z - 1;
    z = rho*z(-1) + u;
    end;
    initval; x = -2; z = 0; end;
    """
    mf = from_mod(text)
    assert "LEAD1_x" in mf.auxiliary and mf.auxiliary["LEAD1_x"] == ("lead", "x")
    ss = mf.steady_state()
    assert ss.values["x"] == pytest.approx(-2.0)
    sol = dk.solve(mf.model, mf.params, steady=ss)
    irf = sol.irf("u", size=0.1, periods=40, levels=True, percent=False)
    # x_t = 0.5 E_t x_{t+2} + exp(u_t) z_t - 1 with z_t = 0.5 z_{t-1} + u_t: at first order
    # z_t = 0.1 * 0.5^t and x_t - x* = sum_j 0.5^j z_{t+2j} = z_t / (1 - 0.5 * 0.25)
    z = 0.1 * 0.5 ** np.arange(40)
    assert np.max(np.abs((irf["x"] + 2.0) - z / (1 - 0.125))) < 1e-12


def test_steady_state_operator_and_unit_root():
    text = """
    var pi p y yhat;
    varexo e;
    parameters beta kappa;
    beta = 0.99; kappa = 0.1;
    model(linear);
    pi = beta*pi(+1) + kappa*yhat;
    yhat = y - steady_state(y);
    y = 0.5*y(-1) + e;
    pi = p - p(-1);
    end;
    shocks; var e = 1; end;
    """
    mf = from_mod(text)
    assert mf.linear and mf.steady_operators == ["y"]
    ss = mf.steady_state()
    with pytest.raises(dk.BlanchardKahnError):
        dk.solve(mf.model, mf.params, steady=ss)          # the price level has a unit root
    sol = dk.solve(mf.model, mf.params, steady=ss, qz_criterium=1 + 1e-6)
    irf = sol.irf("e", size=1.0, periods=30, percent=False)
    assert np.max(np.abs(np.cumsum(irf["pi"]) - irf["p"])) < 1e-12


@pytest.mark.parametrize("text, match", [
    ("var x; varexo e; model; x = x(+1) + e(+2); end;", "more than one period"),
    ("var x; varexo e; model; x = expectation(-1)(x) + e; end;", "expectation"),
    ("var x y; varexo e; model; x = e; end;", "1 equations for 2"),
    ("@#include \"other.mod\"\nvar x; varexo e; model; x = e; end;", "include"),
    ("var x; varexo e; model; x = q + e; end;", "undeclared"),
])
def test_what_is_not_read_is_refused(text, match):
    with pytest.raises(DynareImportError, match=match):
        from_mod(text)
