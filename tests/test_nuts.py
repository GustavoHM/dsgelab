"""No-U-Turn sampling and its diagnostics."""

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


def test_nuts_returns_chains_and_depth_diagnostics():
    problem = _problem()
    result = dsg.nuts(problem, dsg.posterior_mode(problem), draws=25, chains=2,
                      burn=0.2, step_size=0.03, max_depth=4, seed=2)
    assert isinstance(result, dsg.NUTSResult)
    assert isinstance(result, dsg.MCMCResult)
    assert result.chains.shape == (2, 20, 1)
    assert result.log_posterior.shape == (2, 20)
    assert np.all(np.isfinite(result.draws))
    assert np.all(np.isfinite(result.log_posterior))
    assert np.all(result.divergences == 0)
    assert np.all(result.mean_depth > 0)
    assert np.all(result.mean_depth <= 4)
    expected = np.array([problem.log_posterior(x) for x in result.draws])
    np.testing.assert_allclose(result.log_posterior.reshape(-1), expected,
                               rtol=1e-10, atol=1e-10)


def test_nuts_reports_divergences_for_an_unstable_step():
    problem = _problem()
    result = dsg.nuts(problem, draws=5, chains=1, burn=0.0,
                      step_size=1e6, max_depth=2, adapt=False, seed=3)
    assert result.divergences[0] > 0
    assert np.all(np.isfinite(result.log_posterior))


def test_nuts_argument_errors():
    problem = _problem()
    with pytest.raises(ValueError, match="step_size"):
        dsg.nuts(problem, step_size=0.0)
    with pytest.raises(ValueError, match="max_depth"):
        dsg.nuts(problem, max_depth=0)
    with pytest.raises(ValueError, match="burn"):
        dsg.nuts(problem, burn=1.0)
