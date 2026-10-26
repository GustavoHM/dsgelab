"""Inversion filtering: exact Gaussian density and pruned nonlinear paths."""

import numpy as np
import pytest
from scipy.stats import norm

import dsgelab as dsg


def _ar1():
    def eqs(nxt, cur, eps, p):
        return [nxt.y - p["rho"] * cur.y - eps.e]
    model = dsg.Model(states=["y"], jumps=[], shocks=["e"], equations=eqs, name="ar1")
    rng = np.random.default_rng(4)
    y = np.zeros(8)
    for t in range(1, 8):
        y[t] = 0.5 * y[t - 1] + 0.2 * rng.standard_normal()
    expected = norm.logpdf(y[0], scale=0.2)
    for t in range(1, 8):
        expected += norm.logpdf(y[t] - 0.5 * y[t - 1], scale=0.2)
    return model, y, expected


def test_inversion_filter_matches_exact_gaussian_density_at_both_orders():
    model, data, expected = _ar1()
    first = dsg.solve(model, {"rho": 0.5}, guess={"y": 0.0})
    second = dsg.solve_second_order(first, cov={"e": 0.2})
    one = dsg.inversion_filter(first, {"y": data}, observables=["y"], cov={"e": 0.2})
    two = dsg.inversion_filter(second, {"y": data}, observables=["y"])
    assert one.loglik == pytest.approx(expected, abs=1e-10)
    assert two.loglik == pytest.approx(expected, abs=1e-10)
    actual = np.r_[data[0], data[1:] - 0.5 * data[:-1]]
    np.testing.assert_allclose(one.shocks[:, 0], actual, atol=1e-12)
    np.testing.assert_allclose(two.shocks[:, 0], actual, atol=1e-12)


def test_pruned_inversion_handles_nonlinear_model_and_missing_data():
    pytest.importorskip("sympy")
    from dsgelab.examples import rbc
    model, params, guess = rbc.build_symbolic()
    steady = dsg.steady_state(model, guess, params)
    second = dsg.solve_second_order(dsg.solve(model, params, steady=steady),
                                    cov={"eps_a": 0.007})
    simulated = second.simulate(periods=12, burn=100, levels=True, seed=2)
    data = {"a": simulated["a"].copy()}
    result = dsg.inversion_filter(second, data, observables=["a"])
    assert np.isfinite(result.loglik)
    assert result.states.shape == (12, 2)
    assert result.jacobians.shape == (12, 1, 1)
    data["a"][3] = np.nan
    missing = dsg.inversion_filter(second, data, observables=["a"])
    assert missing.loglik_t[3] == 0.0
    assert missing.shocks[3, 0] == 0.0
    assert np.isfinite(missing.loglik)
    with pytest.raises(ValueError, match="at least as many shocks"):
        dsg.inversion_filter(second, {"a": simulated["a"], "c": simulated["c"]},
                             observables=["a", "c"])
