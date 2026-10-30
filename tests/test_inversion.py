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


def test_two_shock_inversion_preserves_state_shock_interactions():
    pytest.importorskip("sympy")
    model = dsg.from_equations(
        ["x(+1)=0.4*x+e1+0.5*x*e2", "z(+1)=0.3*z+e2"],
        states=["x", "z"], jumps=[], shocks=["e1", "e2"], parameters=[])
    second = dsg.solve_second_order(dsg.solve(model, {}, guess={"x": 0.0, "z": 0.0}),
                                    cov={"e1": 0.2, "e2": 0.3})
    # Pruned recursion: xf'=.4*xf+e1, xs'=.4*xs+.5*xf*e2, z'=.3*z+e2.
    shocks = np.array([[0.1, 0.2], [0.3, 0.4], [-0.2, 0.1]])
    data = np.array([[0.1, 0.2], [0.36, 0.46], [-0.039, 0.238]])
    result = dsg.inversion_filter(second, data, observables=["x", "z"])
    np.testing.assert_allclose(result.shocks, shocks, atol=1e-9)
    np.testing.assert_allclose(result.states, data, atol=1e-9)
    assert not result.approximate


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("partial_missing", [False, True])
def test_rectangular_and_missing_inversions_match_joint_gaussian_density(order, partial_missing):
    from scipy.stats import multivariate_normal
    pytest.importorskip("sympy")
    model = dsg.from_equations(
        ["x(+1)=0.5*x+0.2*z+e1", "z(+1)=0.4*z+e2"],
        states=["x", "z"], jumps=[], shocks=["e1", "e2"], parameters=[])
    first = dsg.solve(model, {}, guess={"x": 0.0, "z": 0.0})
    Sigma = np.array([[0.04, 0.015], [0.015, 0.09]])
    solution = first if order == 1 else dsg.solve_second_order(first, cov=Sigma)
    data = np.array([[0.1, 0.2], [0.7, np.nan], [np.nan, np.nan],
                     [-0.2, 0.1], [0.3, -0.1]])
    observables = ["x", "z"] if partial_missing else ["x"]
    if not partial_missing:
        data = data[:, :1]
    nobs, periods = len(observables), len(data)
    # Joint law from the independent structural innovations, starting at zero.
    loading = np.zeros((periods * nobs, periods * 2))
    for t in range(periods):
        for s in range(t + 1):
            loading[t * nobs:(t + 1) * nobs, s * 2:(s + 1) * 2] = np.linalg.matrix_power(
                first.P, t - s)[:nobs] @ first.Q
    joint_cov = loading @ np.kron(np.eye(periods), Sigma) @ loading.T
    observed = ~np.isnan(data.ravel())
    expected = multivariate_normal.logpdf(data.ravel()[observed],
                                          cov=joint_cov[np.ix_(observed, observed)])
    result = dsg.inversion_filter(solution, data, observables=observables, cov=Sigma)
    assert result.loglik == pytest.approx(expected, abs=1e-9)
    assert result.loglik_t[2] == 0.0
    assert np.all(result.jacobians[np.isnan(data)] == 0.0)
    assert result.approximate == (order == 2)
    for t in range(periods):
        prefix = data[:t + 1].ravel()
        seen = np.flatnonzero(~np.isnan(prefix))
        rows = slice(t * nobs, (t + 1) * nobs)
        conditional = joint_cov[rows][:, seen] @ np.linalg.solve(
            joint_cov[np.ix_(seen, seen)], prefix[seen])
        np.testing.assert_allclose(result.states[t, :nobs], conditional, atol=1e-9)


def test_rectangular_inversion_whitens_unequal_correlated_shocks():
    model = dsg.Model(states=["x"], jumps=[], shocks=["e1", "e2"],
                      equations=lambda nxt, cur, eps, p: [nxt.x - eps.e1 - eps.e2])
    first = dsg.solve(model, {}, guess={"x": 0.0})
    Sigma = np.array([[0.04, 0.01], [0.01, 0.09]])
    result = dsg.inversion_filter(first, [0.3], observables=["x"], cov=Sigma)
    expected = Sigma @ np.ones(2) * 0.3 / Sigma.sum()
    np.testing.assert_allclose(result.shocks[0], expected, atol=1e-12)
    assert result.loglik == pytest.approx(norm.logpdf(0.3, scale=np.sqrt(Sigma.sum())))


def test_nonlinear_rectangular_inversion_finds_the_minimum_norm_root():
    pytest.importorskip("sympy")
    model = dsg.from_equations(
        ["x(+1)=e1+e2+0.1*(e1+e2)**2"], states=["x"], jumps=[],
        shocks=["e1", "e2"], parameters=[])
    Sigma = np.array([[0.04, 0.01], [0.01, 0.09]])
    second = dsg.solve_second_order(dsg.solve(model, {}, guess={"x": 0.0}), cov=Sigma)
    # The small inverse root is e1+e2=.3. Its standardized minimum norm is known.
    result = dsg.inversion_filter(second, [0.309], observables=["x"])
    expected = Sigma @ np.ones(2) * 0.3 / Sigma.sum()
    np.testing.assert_allclose(result.shocks[0], expected, atol=1e-8)
    assert result.states[0, 0] == pytest.approx(0.309, abs=1e-9)
    assert result.loglik == pytest.approx(
        norm.logpdf(0.3, scale=np.sqrt(Sigma.sum())) - np.log(1.06), abs=1e-8)
    assert result.approximate
