"""The Ramsey planner against answers known on paper.

The comparison with Dynare's ``ramsey_model`` lives in ``validation/compare_ramsey.py``;
these tests need neither Dynare nor Octave.
"""

import numpy as np
import pytest

pytest.importorskip("sympy")

import dsgelab as dk
from dsgelab.optimal import (RamseyError, ramsey_model, ramsey_multipliers,
                             ramsey_steady_state)

NK = ["x - x(+1) + (1/sigma)*(i - pi(+1)) = 0",
      "pi = beta*pi(+1) + kappa*x + u",
      "u(+1) = rho*u + e"]
NK_PARAMS = dict(beta=0.99, sigma=1.0, kappa=0.1, lam=0.25, rho=0.5)


def _nk(**kw):
    return ramsey_model(NK, states=["u"], jumps=["x", "pi", "i"], shocks=["e"],
                        parameters=["beta", "sigma", "kappa", "lam", "rho"],
                        instruments=["i"], objective="-(pi**2 + lam*x**2)/2",
                        discount="beta", **kw)


def _nk_irf(periods=30):
    m = _nk()
    ss = ramsey_steady_state(m, dict(u=0, x=0, pi=0, i=0), NK_PARAMS)
    return dk.solve(m, NK_PARAMS, steady=ss).irf("e", size=1.0, periods=periods,
                                                  percent=False)


def test_nk_commitment_satisfies_the_targeting_rule():
    # Gali (2015, ch. 5): pi_t = -(lam/kappa)(x_t - x_{t-1}), with x_{-1} = 0 (timeless).
    irf = _nk_irf()
    x, pi = irf["x"], irf["pi"]
    x_lag = np.r_[0.0, x[:-1]]
    p = NK_PARAMS
    assert np.max(np.abs(pi + p["lam"] / p["kappa"] * (x - x_lag))) < 1e-12


def test_nk_commitment_matches_the_closed_form():
    # x_t = delta x_{t-1} - kappa delta / (lam (1 - delta beta rho)) u_t
    irf = _nk_irf()
    b, k, l, r = (NK_PARAMS[n] for n in ("beta", "kappa", "lam", "rho"))
    a = l / (l * (1 + b) + k ** 2)
    delta = (1 - np.sqrt(1 - 4 * b * a ** 2)) / (2 * a * b)
    x_cf, prev = np.zeros_like(irf["x"]), 0.0
    for t, u in enumerate(irf["u"]):
        x_cf[t] = prev = delta * prev - k * delta / (l * (1 - delta * b * r)) * u
    assert np.max(np.abs(irf["x"] - x_cf)) < 1e-12


def test_layout_and_metadata():
    m = _nk()
    assert m.states == ["u", "lag_MULT_1", "lag_MULT_2"]
    assert m.jumps == ["x", "pi", "i", "MULT_1", "MULT_2"]
    assert m.ramsey["exogenous"] == ["u"]
    assert m.ramsey["constraint_index"] == [0, 1]
    assert m.ramsey["lagged_variables"] == [] and m.ramsey["auxiliary"] == []


RBC = ["1/c = beta*(1/c(+1))*(alpha*exp(a(+1))*k(+1)**(alpha-1)*n(+1)**(1-alpha) + 1 - delta)",
       "chi*n**phi*c = (1-tau)*(1-alpha)*exp(a)*k**alpha*n**(-alpha)",
       "c + k(+1) = exp(a)*k**alpha*n**(1-alpha) + (1-delta)*k",
       "a(+1) = rho*a + e"]
RBC_PARAMS = dict(beta=0.99, alpha=0.33, delta=0.025, rho=0.95, chi=8.0, phi=1.0)
RBC_GUESS = dict(n=0.33, k=9.4, c=0.76, tau=0.0, a=0.0)


def _rbc(**kw):
    return ramsey_model(RBC, states=["k", "a"], jumps=["c", "n", "tau"], shocks=["e"],
                        parameters=list(RBC_PARAMS), instruments=["tau"],
                        objective="log(c) - chi*n**(1+phi)/(1+phi)", discount="beta", **kw)


def test_efficient_economy_keeps_the_tax_at_zero():
    # With tau = 0 the competitive equilibrium is the social planner's, so the Ramsey
    # planner never uses the instrument, and the multipliers on the household's own
    # conditions are zero: they do not bind.
    m = _rbc()
    ss = ramsey_steady_state(m, RBC_GUESS, RBC_PARAMS, positive=["c", "n", "k"])
    assert abs(ss.values["tau"]) < 1e-12
    assert abs(ss.values["MULT_1"]) < 1e-12 and abs(ss.values["MULT_2"]) < 1e-12
    irf = dk.solve(m, RBC_PARAMS, steady=ss).irf("e", size=0.01, periods=30, levels=True)
    assert np.max(np.abs(irf["tau"])) < 1e-12
    assert np.max(np.abs(irf["MULT_1"])) < 1e-12


def test_auxiliary_formulation_has_the_same_solution():
    # A predetermined variable's condition is either shifted forward in place or routed
    # through an auxiliary jump FOC_k. The two are the same equilibrium.
    out = []
    for aux in ("auto", "always"):
        m = _rbc(auxiliary=aux)
        ss = ramsey_steady_state(m, RBC_GUESS, RBC_PARAMS, positive=["c", "n", "k"])
        irf = dk.solve(m, RBC_PARAMS, steady=ss).irf("e", size=0.01, periods=30, levels=True)
        out.append((m, irf))
    assert out[0][0].ramsey["auxiliary"] == []
    assert out[1][0].ramsey["auxiliary"] == ["FOC_k"]
    for v in ("c", "n", "k", "tau", "MULT_3"):
        assert np.max(np.abs(out[0][1][v] - out[1][1][v])) < 1e-10


def test_nonlinear_constraints_get_lagged_copies():
    # The Euler's derivative with respect to c(+1) depends on c and r at t, so the
    # condition for c carries lagged copies of both.
    m = ramsey_model(
        ["a(+1) = rho*a + u",
         "1/c = beta*(1/c(+1))*(r/pai(+1))",
         "omega*pai*(pai-1) = beta*omega*(c/c(+1))*pai(+1)*(pai(+1)-1)"
         " + epsilon*exp(a)*n*(c/exp(a)*phi*n**gamma - (epsilon-1)/epsilon)",
         "exp(a)*n = c + (omega/2)*(pai-1)**2"],
        states=["a"], jumps=["pai", "c", "n", "r"], shocks=["u"],
        parameters=["beta", "rho", "epsilon", "omega", "phi", "gamma"], instruments=["r"],
        objective="log(c) - phi*n**(1+gamma)/(1+gamma)", discount=0.99)
    assert set(m.ramsey["lagged_variables"]) == {"lag_c", "lag_r"}
    params = dict(beta=0.99, gamma=3.0, omega=17.0, epsilon=8.0, phi=1.0, rho=0.95)
    ss = ramsey_steady_state(m, dict(a=0, pai=1, r=1 / 0.99, c=0.97, n=0.97), params)
    assert abs(ss.values["c"] - 0.875 ** 0.25) < 1e-10      # c^4 = (epsilon-1)/epsilon
    assert abs(ss.values["pai"] - 1.0) < 1e-12


def test_multipliers_fit_reports_a_wrong_steady_state():
    m = _nk()
    good = ramsey_multipliers(m, dict(u=0, x=0, pi=0, i=0), NK_PARAMS)
    assert good["_residual"] < 1e-14
    bad = ramsey_multipliers(m, dict(u=0, x=0, pi=0.5, i=0), NK_PARAMS)
    assert bad["_residual"] > 1e-3


def test_miscounted_instruments_are_refused():
    with pytest.raises(RamseyError, match="expected"):
        ramsey_model(NK, states=["u"], jumps=["x", "pi", "i"], shocks=["e"],
                     parameters=["beta", "sigma", "kappa", "lam", "rho"],
                     instruments=["i", "x"], objective="-pi**2", discount="beta")


def test_shock_in_a_jump_equation_is_refused():
    with pytest.raises(RamseyError, match="jumps"):
        ramsey_model(["x = x(+1) - (i - pi(+1)) + e", "pi = 0.99*pi(+1) + 0.1*x"],
                     states=[], jumps=["x", "pi", "i"], shocks=["e"], parameters=[],
                     instruments=["i"], objective="-pi**2", discount=0.99)


def test_objective_with_a_lead_is_refused():
    with pytest.raises(RamseyError, match="current-period"):
        ramsey_model(NK, states=["u"], jumps=["x", "pi", "i"], shocks=["e"],
                     parameters=["beta", "sigma", "kappa", "lam", "rho"],
                     instruments=["i"], objective="-pi(+1)**2", discount="beta")
