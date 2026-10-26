"""Gradients of the first-order Kalman likelihood."""

import numpy as np
import pytest

import dsgelab as dsg


def _finite_difference(problem, theta, h=1e-5):
    return np.array([
        (problem.log_likelihood(theta + h * np.eye(len(theta))[i])
         - problem.log_likelihood(theta - h * np.eye(len(theta))[i])) / (2 * h)
        for i in range(len(theta))])


def test_likelihood_gradient_matches_finite_differences():
    def eqs(nxt, cur, eps, p):
        return [nxt.y - p["rho"] * cur.y - eps.e]

    model = dsg.Model(states=["y"], jumps=[], shocks=["e"], equations=eqs, name="ar1")
    rng = np.random.default_rng(4)
    y = np.zeros(30)
    for t in range(1, 30):
        y[t] = 0.5 * y[t - 1] + 0.2 * rng.standard_normal()
    problem = dsg.EstimationProblem(
        model, {"y": y}, params={"rho": 0.5}, observables=["y"],
        priors={"rho": ("normal", 0.5, 0.1), "stderr e": ("gamma", 0.2, 0.05),
                "stderr y": ("gamma", 0.1, 0.02)},
        cov={"e": 0.2}, measurement={"y": 0.1}, guess={"y": 0.0})
    theta = problem.start()
    result = dsg.log_likelihood_gradient(problem, theta)
    assert result.loglik == pytest.approx(problem.log_likelihood(theta), abs=1e-12)
    np.testing.assert_allclose(result.gradient, _finite_difference(problem, theta),
                               rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(result.gradient_t.sum(axis=0), result.gradient)


def test_likelihood_gradient_through_general_saddle_path():
    def eqs(nxt, cur, eps, p):
        return [nxt.x - p["rho"] * cur.x - p["phi"] * eps.e,
                p["a"] * nxt.y - cur.y + p["b"] * cur.x]

    model = dsg.Model(states=["x"], jumps=["y"], shocks=["e"], equations=eqs)
    params = {"rho": 0.7, "phi": 0.2, "a": 0.5, "b": 0.3}
    first = dsg.solve(model, params, guess={"x": 0.0, "y": 0.0})
    data = first.simulate(periods=60, cov={"e": 0.2}, seed=3)
    problem = dsg.EstimationProblem(
        model, {"y": data["y"]}, params=params, observables=["y"],
        priors={"rho": ("normal", 0.7, 0.1), "phi": ("normal", 0.2, 0.05),
                "a": ("normal", 0.5, 0.1), "b": ("normal", 0.3, 0.1),
                "stderr e": ("gamma", 0.2, 0.05), "stderr y": ("gamma", 0.1, 0.02)},
        cov={"e": 0.2}, measurement={"y": 0.1}, guess={"x": 0.0, "y": 0.0})
    theta = problem.start()
    result = dsg.log_likelihood_gradient(problem, theta)
    assert result.loglik == pytest.approx(problem.log_likelihood(theta), abs=1e-12)
    np.testing.assert_allclose(result.gradient, _finite_difference(problem, theta),
                               rtol=2e-4, atol=2e-4)


def test_likelihood_gradient_log_transform_and_presample():
    def eqs(nxt, cur, eps, p):
        return [nxt.y - p["rho"] * cur.y - (1 - p["rho"]) * p["mu"] - eps.e]

    model = dsg.Model(states=["y"], jumps=[], shocks=["e"], equations=eqs, name="ar1-level")
    rng = np.random.default_rng(8)
    y = np.full(30, 2.0)
    for t in range(1, 30):
        y[t] = 0.5 * y[t - 1] + 1 + 0.2 * rng.standard_normal()
    problem = dsg.EstimationProblem(
        model, {"y": np.log(y)}, params={"rho": 0.5, "mu": 2.0}, observables=["y"],
        priors={"rho": ("normal", 0.5, 0.1), "mu": ("normal", 2.0, 0.2),
                "stderr e": ("gamma", 0.2, 0.05), "stderr y": ("gamma", 0.1, 0.02)},
        cov={"e": 0.2}, measurement={"y": 0.1}, guess={"y": 2.0},
        transforms="log", presample=2)
    theta = problem.start()
    result = dsg.log_likelihood_gradient(problem, theta)
    assert result.loglik == pytest.approx(problem.log_likelihood(theta), abs=1e-12)
    np.testing.assert_allclose(result.gradient, _finite_difference(problem, theta),
                               rtol=2e-5, atol=3e-5)
    np.testing.assert_allclose(result.gradient_t[2:].sum(axis=0), result.gradient)


def test_likelihood_gradient_correlation_missing_and_diffuse():
    def eqs(nxt, cur, eps, p):
        return [nxt.x - 0.5 * cur.x - eps.e1,
                nxt.z - 0.4 * cur.z - eps.e2]

    model = dsg.Model(states=["x", "z"], jumps=[], shocks=["e1", "e2"],
                      equations=eqs, name="two-ar1")
    rng = np.random.default_rng(12)
    eps = rng.standard_normal((30, 2)) @ np.array([[0.2, 0.0], [0.05, 0.15]]).T
    y = np.zeros((30, 2))
    for t in range(1, 30):
        y[t, 0] = 0.5 * y[t - 1, 0] + eps[t, 0]
        y[t, 1] = 0.4 * y[t - 1, 1] + eps[t, 1]
    problem = dsg.EstimationProblem(
        model, {"x": y[:, 0], "z": y[:, 1]}, params={},
        observables=["x", "z"],
        priors={"stderr e1": ("gamma", 0.2, 0.05),
                "stderr e2": ("gamma", 0.15, 0.03),
                "corr e1, e2": ("normal", 0.2, 0.1)},
        cov=np.array([[0.04, 0.01], [0.01, 0.0225]]), guess={"x": 0.0, "z": 0.0})
    theta = problem.start()
    result = dsg.log_likelihood_gradient(problem, theta)
    np.testing.assert_allclose(result.gradient, _finite_difference(problem, theta),
                               rtol=2e-5, atol=2e-5)

    missing = y.copy()
    missing[3, 0] = np.nan
    problem.data = {"x": missing[:, 0], "z": missing[:, 1]}
    problem.Y = problem._data_matrix()
    missing_result = dsg.log_likelihood_gradient(problem, theta)
    assert missing_result.loglik == pytest.approx(problem.log_likelihood(theta), abs=1e-12)
    np.testing.assert_allclose(missing_result.gradient, _finite_difference(problem, theta),
                               rtol=2e-5, atol=2e-5)

    def random_walk(nxt, cur, eps, p):
        return [nxt.x - p["rho"] * cur.x - eps.e1]

    unit = dsg.Model(states=["x"], jumps=[], shocks=["e1"], equations=random_walk)
    diffuse = dsg.EstimationProblem(
        unit, {"x": np.arange(8.0)}, params={"rho": 1.01}, observables=["x"],
        priors={"rho": ("normal", 1.01, 0.1), "stderr e1": ("gamma", 0.2, 0.05)},
        cov={"e1": 0.2}, guess={"x": 0.0}, init="diffuse", qz_criterium=1.1)
    diffuse_result = dsg.log_likelihood_gradient(diffuse, diffuse.start())
    assert diffuse_result.loglik == pytest.approx(diffuse.log_likelihood(diffuse.start()),
                                                 abs=1e-12)
    np.testing.assert_allclose(diffuse_result.gradient, _finite_difference(diffuse, diffuse.start()),
                               rtol=2e-4, atol=2e-4)
