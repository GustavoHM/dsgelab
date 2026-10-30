"""HMC and NUTS against the exact posterior and a long Metropolis–Hastings run.

The posterior of one AR(1) coefficient is computed by quadrature, so the samplers
are checked against the distribution itself, and against random-walk Metropolis,
which needs no gradient. A sampler that targets the wrong distribution, or mixes
too slowly to explore it, moves the mean or the standard deviation outside the
Monte Carlo tolerance.
"""

import numpy as np
import pytest

import dsgelab as dsg


@pytest.fixture(scope="module")
def reference():
    def eqs(nxt, cur, eps, p):
        return [nxt.y - p["rho"] * cur.y - eps.e]

    model = dsg.Model(states=["y"], jumps=[], shocks=["e"], equations=eqs, name="ar1")
    rng = np.random.default_rng(4)
    y = np.zeros(25)
    for t in range(1, 25):
        y[t] = 0.5 * y[t - 1] + 0.2 * rng.standard_normal()
    problem = dsg.EstimationProblem(
        model, {"y": y}, params={"rho": 0.5}, observables=["y"],
        priors={"rho": ("normal", 0.5, 0.2)}, cov={"e": 0.2}, guess={"y": 0.0})
    mode = dsg.posterior_mode(problem)
    sd = mode.std["rho"]
    grid = np.linspace(mode.x[0] - 8 * sd, mode.x[0] + 8 * sd, 801)
    lp = np.array([problem.log_posterior(np.array([x])) for x in grid])
    w = np.exp(lp - lp.max())
    w /= w.sum()
    mean = float(w @ grid)
    exact = dict(mean=mean, sd=float(np.sqrt(w @ (grid - mean) ** 2)))
    mh = dsg.metropolis_hastings(problem, mode, draws=5000, chains=2, seed=1)
    return problem, mode, exact, mh


def _agrees(result, exact, minimum_ess=80):
    """Mean and standard deviation within 4.5 Monte Carlo standard errors of the exact ones.

    The errors use the chains' own effective sample size, which must be large enough
    that a sampler cannot pass by mixing so slowly that its tolerance becomes wide.
    """
    ess = result.ess()["rho"]
    assert ess > minimum_ess
    se_mean = exact["sd"] / np.sqrt(ess)
    se_sd = exact["sd"] / np.sqrt(2 * ess)
    assert result.draws.mean() == pytest.approx(exact["mean"], abs=4.5 * se_mean)
    assert result.draws.std() == pytest.approx(exact["sd"], abs=4.5 * se_sd)


def _matches_metropolis(result, mh):
    """The same comparison against the long Metropolis run, with both errors combined."""
    ess, ess_mh = result.ess()["rho"], mh.ess()["rho"]
    sd = mh.draws.std()
    se = sd * np.sqrt(1 / ess + 1 / ess_mh)
    assert result.draws.mean() == pytest.approx(mh.draws.mean(), abs=4.5 * se)
    assert result.draws.std() == pytest.approx(sd, abs=4.5 * se / np.sqrt(2))


def test_long_metropolis_run_matches_the_exact_posterior(reference):
    _, _, exact, mh = reference
    _agrees(mh, exact, minimum_ess=250)


def test_hmc_matches_the_exact_posterior_and_metropolis(reference):
    problem, mode, exact, mh = reference
    result = dsg.hmc(problem, mode, draws=300, chains=2, seed=1)
    _agrees(result, exact)
    _matches_metropolis(result, mh)
    assert result.rhat()["rho"] < 1.05
    assert np.all((result.acceptance > 0.65) & (result.acceptance < 0.99))


def test_nuts_matches_the_exact_posterior_and_metropolis(reference):
    problem, mode, exact, mh = reference
    result = dsg.nuts(problem, mode, draws=400, chains=2, seed=1)
    _agrees(result, exact)
    _matches_metropolis(result, mh)
    assert result.rhat()["rho"] < 1.05
    assert np.all(result.divergences == 0)
    assert np.all((result.acceptance > 0.65) & (result.acceptance < 0.99))


def test_nuts_without_adaptation_still_targets_the_posterior(reference):
    # a fixed step isolates the tree: the backward doubling and the multinomial
    # choice must keep the posterior invariant whatever the step
    problem, mode, exact, _ = reference
    result = dsg.nuts(problem, mode, draws=300, chains=2, burn=0.1, step_size=0.8,
                      adapt=False, seed=2)
    _agrees(result, exact)
    assert result.mean_depth.min() >= 1
