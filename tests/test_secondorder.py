"""Second-order perturbation, checked against results that are known analytically."""

import numpy as np
import pytest

import dsgekit as dk
from dsgekit.examples import rbc


def test_quadratic_terms_give_third_order_accuracy():
    """Halving the displacement must divide the residual by about eight."""
    model, p, guess = rbc.build()
    sol = dk.solve(model, p, guess=guess)
    so = dk.solve_second_order(sol, cov={"eps_a": 0.007})
    acc = so.accuracy()
    assert all(2.85 < o < 3.15 for o in acc["implied_order"])
    assert so.sylvester_residual < 1e-10


def test_risk_correction_on_a_lognormal_ar1_is_half_the_variance():
    """log a' = rho log a + eps implies E_t a' = a^rho * exp(sigma^2/2).

    In proportional deviations that is a drift of exactly sigma^2/2, and the
    rule adds h_ss/2, so h_ss must come out at sigma^2. A closed-form check on
    the risk term itself, which is the piece first order sets to zero.
    """
    model, p, guess = rbc.build()
    sigma = 0.007
    sol = dk.solve(model, p, guess=guess)
    so = dk.solve_second_order(sol, cov={"eps_a": sigma})
    i_a = model.states.index("a")
    assert so.h_ss[i_a] == pytest.approx(sigma ** 2, rel=1e-6)


def test_exact_policy_leaves_no_risk_correction_in_capital():
    """k' = alpha*beta*a*k^alpha holds state by state, so k carries no sigma^2 term.

    Consumption likewise. If either came out nonzero the risk block would be
    manufacturing a correction the model does not have.
    """
    model, p, guess = rbc.build()
    sol = dk.solve(model, p, guess=guess)
    so = dk.solve_second_order(sol, cov={"eps_a": 0.007})
    i_k = model.states.index("k")
    assert abs(so.h_ss[i_k]) < 1e-8
    assert abs(so.g_ss[model.jumps.index("c")]) < 1e-8


def test_ergodic_mean_matches_the_lognormal_mean():
    model, p, guess = rbc.build()
    sigma = 0.007
    sol = dk.solve(model, p, guess=guess)
    so = dk.solve_second_order(sol, cov={"eps_a": sigma})
    theory = np.exp(sigma ** 2 / (1 - p.rho ** 2) / 2) - 1.0
    assert so.mean_correction()["a"] == pytest.approx(theory, rel=1e-3)


def test_pruned_simulation_stays_bounded():
    """An unpruned quadratic map explodes; pruning is what makes moments meaningful."""
    model, p, guess = rbc.build()
    sol = dk.solve(model, p, guess=guess)
    so = dk.solve_second_order(sol, cov={"eps_a": 0.02})
    sim = so.simulate(periods=20_000, seed=3, pruning=True)
    assert np.all(np.isfinite(sim["k"]))
    assert np.max(np.abs(sim["k"])) < 1.0
