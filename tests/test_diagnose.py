"""dsgelab.diagnose: Dynare's model_diagnostics, checked without Dynare.

validation/compare_diagnostics.py runs Dynare's model_diagnostics on the same broken model.
"""

import pytest

pytest.importorskip("sympy")

import dsgelab as dk  # noqa: E402

PARAMS = dict(alpha=0.33, delta=0.025, rho=0.9)


def test_walras_redundancy_and_an_absent_variable_are_found():
    # The Euler equation is missing and the resource constraint appears twice in
    # disguise; q enters only one period ahead.
    m = dk.from_equations(["y = exp(a)*k**alpha", "y = c + i", "k(+1) = (1-delta)*k + i",
                           "c + i = exp(a)*k**alpha", "a(+1) = rho*a + e", "q(+1) = a"],
                          states=["k", "a"], jumps=["y", "c", "i", "q"], shocks=["e"],
                          parameters=list(PARAMS))
    k = 10.0
    y = k ** 0.33
    point = dict(k=k, a=0.0, y=y, c=y - 0.025 * k, i=0.025 * k, q=0.0)
    r = dk.diagnose(m, PARAMS, point)
    assert not r["ok"]
    assert r["absent_current"] == ["q"] and r["absent"] == []
    assert r["relations"] == 1
    assert sorted(r["collinear_variables"][0]) == ["c", "i", "k", "y"]
    assert [i for i, _ in r["collinear_equations"][0]] == [1, 2, 4]
    assert r["redundant_equations"] == 1


def test_a_healthy_model_is_clean():
    m = dk.from_equations(["1/c = beta/c(+1)*(alpha*exp(a(+1))*k(+1)**(alpha-1) + 1 - delta)",
                           "c + k(+1) = exp(a)*k**alpha + (1-delta)*k", "a(+1) = rho*a + e"],
                          states=["k", "a"], jumps=["c"], shocks=["e"],
                          parameters=["alpha", "beta", "delta", "rho"])
    p = dict(alpha=0.33, beta=0.99, delta=0.025, rho=0.9)
    kss = ((1 / 0.99 - 1 + 0.025) / 0.33) ** (1 / (0.33 - 1))
    ss = dk.steady_state(m, dict(k=kss, a=0.0, c=kss ** 0.33 - 0.025 * kss), p)
    r = dk.diagnose(m, p, ss)
    assert r["ok"] and r["relations"] == 0 and r["redundant_equations"] == 0
    assert r["residual"] < 1e-12


def test_a_unit_root_is_named_as_the_cause():
    m = dk.from_equations(["p(+1) = p + pi(+1)", "pi(+1) = 0.5*pi + e"],
                          states=["p", "pi"], jumps=[], shocks=["e"], parameters=[])
    r = dk.diagnose(m, {}, dict(p=0.0, pi=0.0))
    assert r["relations"] == 1 and r["unit_root"]


def test_nonfinite_point_is_reported():
    m = dk.from_equations(["y = log(x)", "x(+1) = 0.5*x + e"], states=["x"], jumps=["y"],
                          shocks=["e"], parameters=[])
    r = dk.diagnose(m, {}, dict(x=0.0, y=0.0))
    assert not r["ok"]


RBC = ["1/c = beta/c(+1)*(alpha*exp(a(+1))*k(+1)**(alpha-1) + 1 - delta)",
       "c + k(+1) = exp(a)*k**alpha + (1-delta)*k", "a(+1) = rho*a + e"]
RBC_P = dict(alpha=0.33, beta=0.99, delta=0.025, rho=0.9)


def test_closed_form_steady_state_is_checked_and_used():
    m = dk.from_equations(RBC, states=["k", "a"], jumps=["c"], shocks=["e"],
                          parameters=list(RBC_P), steady_state_model=[
                              "a = 0", "kss = ((1/beta - 1 + delta)/alpha)^(1/(alpha-1))",
                              "k = kss", "c = k^alpha - delta*k"])
    ss = dk.steady_state(m, None, RBC_P)
    assert ss.message.startswith("closed form") and ss.residual < 1e-12
    solved = dk.steady_state(m, dict(k=30.0, a=0.0, c=2.0), RBC_P)
    assert abs(ss.values["k"] - solved.values["k"]) < 1e-10


def test_a_wrong_closed_form_is_refused():
    m = dk.from_equations(RBC, states=["k", "a"], jumps=["c"], shocks=["e"],
                          parameters=list(RBC_P), steady_state_model=[
                              "a = 0", "k = (1/beta/alpha)^(1/(alpha-1))", "c = k^alpha - delta*k"])
    with pytest.raises(dk.SteadyStateError, match="equation 1"):
        dk.steady_state(m, None, RBC_P)
    with pytest.raises(dk.SteadyStateError, match="without a value"):
        dk.analytic_steady_state(m, RBC_P, {"k": 1.0})
