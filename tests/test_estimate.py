"""Kalman likelihood and a posterior mode, against an AR(1) whose answer is known."""

import numpy as np
import pytest

import dsgekit as dk
from dsgekit import Model


def _ar1(rho=0.8):
    def equations(nxt, cur, eps, p):
        return [nxt.y - p["rho"] * cur.y - eps.e]

    model = Model(states=["y"], jumps=[], shocks=["e"], equations=equations, name="ar1")
    params = {"rho": rho}
    sol = dk.solve(model, params, guess={"y": 0.0})
    return model, params, sol


def test_kalman_matches_the_conditional_ar1_likelihood():
    """For y' = ρ y + ε observed without a latent state, the filter is the AR(1) likelihood."""
    rho, sigma = 0.8, 0.1
    _, _, sol = _ar1(rho)
    rng = np.random.default_rng(0)
    T = 500
    eps = rng.normal(0.0, sigma, size=T)
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = rho * y[t - 1] + eps[t]
    # Conditional Gaussian likelihood, dropping the stationary first observation.
    innov = y[1:] - rho * y[:-1]
    hand = -0.5 * ((T - 1) * np.log(2 * np.pi * sigma ** 2) + np.sum(innov ** 2) / sigma ** 2)
    Y = y.reshape(-1, 1)
    got = dk.log_likelihood(sol, Y, np.array([[1.0]]), np.array([[1e-16]]), np.array([[sigma ** 2]]))
    # The filter also prices the first observation from the stationary law, so the
    # two differ by that one term and must agree on the rest to a fraction of a nat.
    assert got == pytest.approx(hand, rel=0, abs=3.0)


def test_posterior_mode_of_rho_on_a_simulated_ar1():
    model, params, sol = _ar1(0.8)
    rng = np.random.default_rng(1)
    sim = sol.simulate(periods=800, cov={"e": 0.1}, seed=1)
    levels = sol.to_levels(sim)
    noise = rng.normal(0.0, 1e-4, size=levels["y"].shape)
    data = {"y": levels["y"] + noise}
    result = dk.estimate(
        model, data, params=params, free={"rho": 0.5},
        priors={"rho": ("normal", 0.7, 0.2)},
        guess={"y": 0.0}, observables=["y"], measurement=1e-8,
        cov={"e": 0.1}, bounds={"rho": (0.05, 0.98)},
        draws=150, seed=2, proposal={"rho": 0.01},
    )
    assert result.success
    assert result.mode["rho"] == pytest.approx(0.8, abs=0.03)
    assert 0.05 < result.acceptance < 0.95
    assert result.draws["rho"][75:].mean() == pytest.approx(0.8, abs=0.05)
