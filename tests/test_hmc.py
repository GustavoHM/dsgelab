"""Prior scores and Hamiltonian Monte Carlo."""

import numpy as np
import pytest

import dsgelab as dsg


def _problem():
    def eqs(nxt, cur, eps, p):
        return [nxt.y - p["rho"] * cur.y - eps.e]

    model = dsg.Model(states=["y"], jumps=[], shocks=["e"], equations=eqs, name="ar1")
    rng = np.random.default_rng(4)
    y = np.zeros(25)
    for t in range(1, 25):
        y[t] = 0.5 * y[t - 1] + 0.2 * rng.standard_normal()
    return dsg.EstimationProblem(
        model, {"y": y}, params={"rho": 0.5}, observables=["y"],
        priors={"rho": ("normal", 0.5, 0.2)}, cov={"e": 0.2}, guess={"y": 0.0})


def test_prior_scores_match_finite_differences():
    cases = [
        (dsg.prior("normal", 0.2, 0.5), 0.3),
        (dsg.prior("beta", 0.4, 0.2), 0.4),
        (dsg.prior("gamma", 1.0, 0.4), 1.2),
        (dsg.prior("inv_gamma", 1.0, 0.5), 1.1),
        (dsg.prior("inv_gamma2", 1.0, 0.5), 1.3),
        (dsg.prior("uniform", 0.3, 0.2), 0.3),
    ]
    h = 1e-6
    for distribution, x in cases:
        expected = (distribution.logpdf(x + h) - distribution.logpdf(x - h)) / (2 * h)
        assert distribution.score(x) == pytest.approx(expected, rel=1e-5, abs=1e-5)
    assert np.isnan(dsg.prior("gamma", 1.0, 0.2).score(-1.0))


def test_log_posterior_gradient_matches_finite_difference():
    problem = _problem()
    theta = problem.start()
    result = dsg.log_posterior_gradient(problem, theta)
    h = 1e-5
    expected = (problem.log_posterior(theta + h) - problem.log_posterior(theta - h)) / (2 * h)
    assert result.logposterior == pytest.approx(problem.log_posterior(theta), abs=1e-12)
    np.testing.assert_allclose(result.gradient, [expected], rtol=1e-7, atol=1e-7)


def test_hmc_returns_finite_posterior_chains():
    problem = _problem()
    mode = dsg.posterior_mode(problem)
    result = dsg.hmc(problem, mode, draws=50, chains=2, burn=0.2,
                     step_size=0.03, leapfrog=5, seed=2)
    assert isinstance(result, dsg.MCMCResult)
    assert result.chains.shape == (2, 40, 1)
    assert result.log_posterior.shape == (2, 40)
    assert result.burn == 10
    assert np.all(np.isfinite(result.draws))
    assert np.all(np.isfinite(result.log_posterior))
    assert np.all((result.acceptance > 0.4) & (result.acceptance <= 1.0))
    expected = np.array([problem.log_posterior(x) for x in result.draws])
    np.testing.assert_allclose(result.log_posterior.reshape(-1), expected, rtol=1e-10, atol=1e-10)


def test_hmc_argument_errors():
    problem = _problem()
    with pytest.raises(ValueError, match="step_size"):
        dsg.hmc(problem, step_size=0.0)
    with pytest.raises(ValueError, match="leapfrog"):
        dsg.hmc(problem, leapfrog=0)
    with pytest.raises(ValueError, match="burn"):
        dsg.hmc(problem, burn=1.0)
