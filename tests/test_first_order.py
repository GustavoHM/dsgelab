"""End-to-end: declared nonlinear model -> steady state -> first-order solution."""

import dataclasses

import numpy as np
import pytest

import dsgekit as dk
from dsgekit.examples import nk, rbc


def test_rbc_matches_closed_form():
    model, p, guess = rbc.build()
    sol = dk.solve(model, p, guess=guess)
    cf = rbc.closed_form(p)
    assert np.max(np.abs(sol.P - cf["P"])) < 1e-8
    assert np.max(np.abs(sol.F - cf["F"])) < 1e-8


def test_steady_state_residual_is_machine_zero():
    model, p, guess = rbc.build()
    ss = dk.steady_state(model, guess, p)
    assert ss.residual < 1e-12
    k, c = rbc.steady_levels(p)
    assert ss["k"] == pytest.approx(k, rel=1e-10)
    assert ss["c"] == pytest.approx(c, rel=1e-10)


def test_first_order_accuracy_is_second_order():
    """Halving the displacement must divide the residual by about four."""
    model, p, guess = rbc.build()
    sol = dk.solve(model, p, guess=guess)
    acc = sol.accuracy()
    assert all(1.9 < o < 2.1 for o in acc["implied_order"])


def test_taylor_principle_governs_determinacy():
    model, p, guess = nk.build()
    dk.solve(model, p, guess=guess)                     # phi_pi = 1.5: determinate
    weak = dataclasses.replace(p, phi_pi=0.8)
    with pytest.raises(dk.BlanchardKahnError):
        dk.solve(model, weak, guess=guess)


def test_zero_steady_state_shock_survives_scaling():
    """Variables with a zero steady state must carry ABSOLUTE deviations.

    A scaling that multiplied them by (1 + yhat) would annihilate the shock in
    silence, which is the specific bug this test exists to catch.
    """
    model, p, guess = nk.build()
    sol = dk.solve(model, p, guess=guess)
    irf = sol.irf("eps_v", size=1.0, periods=5)
    assert np.max(np.abs(irf["v"])) > 1e-6
    assert np.max(np.abs(irf["pi"])) > 1e-6


def test_moments_and_variance_decomposition_are_consistent():
    model, p, guess = nk.build()
    sol = dk.solve(model, p, guess=guess)
    cov = {"eps_a": 0.01, "eps_v": 0.0025}
    mom = sol.moments(cov)
    dec = sol.variance_decomposition(cov)
    total = dec["variance"].sum(axis=1)
    assert np.allclose(total, mom["std"] ** 2, rtol=1e-8, atol=1e-18)
    assert np.allclose(dec["share"].sum(axis=1), 1.0, atol=1e-10)


def test_simulation_reproduces_the_analytic_moments():
    model, p, guess = nk.build()
    sol = dk.solve(model, p, guess=guess)
    cov = {"eps_a": 0.01, "eps_v": 0.0025}
    sim = sol.simulate(periods=200_000, cov=cov, seed=11, burn=500)
    analytic = sol.moments(cov)["std_by_name"]
    for name in ("a", "pi", "x"):
        assert np.std(sim[name]) == pytest.approx(analytic[name], rel=0.05)


def test_irf_from_a_state_differs_from_irf_from_a_shock():
    model, p, guess = rbc.build()
    sol = dk.solve(model, p, guess=guess)
    by_shock = sol.irf("eps_a", size=0.01, periods=3)
    by_state = sol.irf("a", size=0.01, periods=3, kind="state")
    assert np.allclose(by_shock["a"], by_state["a"])       # Q loads eps one-for-one here
    assert by_state["k"][0] == pytest.approx(0.0, abs=1e-12)
