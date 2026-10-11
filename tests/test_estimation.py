"""Filter, smoother, priors and samplers, against answers known in closed form.

The comparison with Dynare (likelihood, prior, smoother, mode, Laplace, MCMC, shock
decompositions, forecasts, the diffuse filter) is validation/compare_estimation.py.
"""

import math

import numpy as np
import pytest
from scipy import stats

pytest.importorskip("sympy")

import dsgelab as dk  # noqa: E402
from dsgelab.estimation import (EstimationProblem, maximum_likelihood,  # noqa: E402
                                metropolis_hastings, model_probabilities, posterior_mode,
                                prior_predictive, smc)
from dsgelab.kalman import (forecast, kalman_filter, kalman_smoother,  # noqa: E402
                            realtime_shock_decomposition, shock_decomposition, state_space)
from dsgelab.priors import prior  # noqa: E402

# y_t = mu + sigma u_t, u white noise: every density here is known in closed form.
IID = dk.from_equations(["u(+1) = e", "y = mu + sigma*u"], states=["u"], jumps=["y"],
                        shocks=["e"], parameters=["mu", "sigma"])


def _iid_data(T=60, mu=0.3, sigma=0.5, seed=0):
    return {"y": mu + sigma * np.random.default_rng(seed).standard_normal(T)}


def _iid_problem(data, m0=0.0, s0=1.0, sigma=0.5):
    return EstimationProblem(model=IID, data=data, params=dict(mu=0.0, sigma=sigma),
                             observables=["y"], priors={"mu": ("normal", m0, s0)},
                             guess=dict(u=0.0, y=0.0), cov={"e": 1.0})


def _exact_log_marginal(y, m0, s0, sigma):
    T = y.size
    C = sigma ** 2 * np.eye(T) + s0 ** 2 * np.ones((T, T))
    return float(stats.multivariate_normal(np.full(T, m0), C).logpdf(y))


def test_likelihood_of_iid_draws():
    data = _iid_data()
    pb = _iid_problem(data)
    ll = pb.log_likelihood([0.3])
    assert ll == pytest.approx(stats.norm(0.3, 0.5).logpdf(data["y"]).sum(), rel=1e-12)


def test_univariate_and_multivariate_filters_agree():
    m = dk.from_equations(["a(+1) = 0.9*a + e1", "b(+1) = 0.5*b + 0.3*a + e2",
                           "y1 = a + b", "y2 = a - 2*b"],
                          states=["a", "b"], jumps=["y1", "y2"], shocks=["e1", "e2"],
                          parameters=[])
    sol = dk.solve(m, {}, guess=dict(a=0, b=0, y1=0, y2=0))
    sim = sol.to_levels(sol.simulate(periods=80, cov={"e1": 0.1, "e2": 0.2}, seed=1))
    data = {"y1": sim["y1"], "y2": sim["y2"]}
    sp = state_space(sol, ["y1", "y2"], cov={"e1": 0.1, "e2": 0.2}, measurement=0.01)
    a = kalman_filter(sp, data, univariate=False)
    b = kalman_filter(sp, data, univariate=True)
    assert a.loglik == pytest.approx(b.loglik, rel=1e-12)
    # a missing value is skipped; the remaining likelihood stays finite and smaller
    data["y2"] = data["y2"].copy(); data["y2"][10] = np.nan
    c = kalman_filter(sp, data)
    assert c.univariate and np.isfinite(c.loglik)


def test_smoother_reproduces_the_data_and_the_law_of_motion():
    m = dk.from_equations(["a(+1) = 0.9*a + e", "y = 2 + a"], states=["a"], jumps=["y"],
                          shocks=["e"], parameters=[])
    sol = dk.solve(m, {}, guess=dict(a=0, y=2))
    y = sol.to_levels(sol.simulate(periods=50, cov={"e": 0.1}, seed=2))["y"]
    sp = state_space(sol, ["y"], cov={"e": 0.1})
    sm = kalman_smoother(sp, {"y": y})
    assert np.max(np.abs(sm.variables()["y"] - y)) < 1e-12
    k, eps = sm.states, sm.shocks
    for t in range(1, len(y)):
        assert np.allclose(k[t], sol.P @ k[t - 1] + sol.Q @ eps[t], atol=1e-12)


def test_exact_diffuse_start_is_the_limit_of_a_vague_one():
    # Local level: x(+1) = x + e, y = x + n. A huge but finite initial variance kappa
    # gives log L(kappa) = log L_diffuse - 1/2 log(kappa) + o(1).
    m = dk.from_equations(["x(+1) = x + e", "y = x"], states=["x"], jumps=["y"],
                          shocks=["e"], parameters=[])
    sol = dk.solve(m, {}, guess=dict(x=0, y=0), qz_criterium=1 + 1e-6)
    rng = np.random.default_rng(3)
    y = np.cumsum(0.1 * rng.standard_normal(40)) + 0.05 * rng.standard_normal(40)
    sp = state_space(sol, ["y"], cov={"e": 0.1}, measurement=0.05)
    exact = kalman_filter(sp, {"y": y}, init="diffuse")
    assert exact.diffuse_periods == 1
    from dsgelab.kalman import _univariate
    kappa = 1e6     # large enough for O(1/kappa), small enough not to lose digits
    vague = _univariate(sp, sp.data_matrix({"y": y}), np.zeros(1), np.array([[kappa]]),
                        np.zeros((1, 1)), 1e-12)
    assert vague.loglik + 0.5 * math.log(kappa) == pytest.approx(exact.loglik, abs=1e-5)


def test_priors_have_the_moments_they_are_given():
    rng = np.random.default_rng(0)
    for shape, mean, sd in [("beta", 0.356, 0.02), ("gamma", 2.0, 0.5), ("normal", 0.1, 0.2),
                            ("inv_gamma", 0.1, 0.05), ("uniform", 0.5, 0.2)]:
        p = prior(shape, mean, sd)
        draws = p.sample(rng, 400000)
        assert draws.mean() == pytest.approx(mean, rel=2e-2)
        if shape != "inv_gamma":
            assert draws.std() == pytest.approx(sd, rel=2e-2)
    # densities integrate to one
    from scipy.integrate import quad
    for p in (prior("beta", 0.5, 0.1), prior("gamma", 2, 0.5), prior("inv_gamma", 0.1, 0.05)):
        hi = p.upper if np.isfinite(p.upper) else 50
        assert quad(lambda x: math.exp(p.logpdf(x)), p.lower + 1e-12, hi, limit=200)[0] \
            == pytest.approx(1.0, abs=1e-5)


def test_mode_laplace_and_maximum_likelihood_in_the_conjugate_case():
    data = _iid_data()
    y = data["y"]
    m0, s0, sigma = 0.0, 1.0, 0.5
    pb = _iid_problem(data, m0, s0, sigma)
    mode = posterior_mode(pb)
    T = y.size
    post_var = 1 / (1 / s0 ** 2 + T / sigma ** 2)
    post_mean = post_var * (m0 / s0 ** 2 + y.sum() / sigma ** 2)
    assert mode.values["mu"] == pytest.approx(post_mean, abs=1e-6)
    assert mode.std["mu"] == pytest.approx(math.sqrt(post_var), rel=1e-3)
    # with a Gaussian posterior the Laplace approximation is exact
    assert mode.laplace == pytest.approx(_exact_log_marginal(y, m0, s0, sigma), abs=1e-4)
    ml = maximum_likelihood(pb)
    assert ml.values["mu"] == pytest.approx(y.mean(), abs=1e-6)
    assert ml.std["mu"] == pytest.approx(sigma / math.sqrt(T), rel=1e-3)


def test_metropolis_and_smc_recover_the_posterior_and_the_marginal():
    data = _iid_data(T=40)
    y = data["y"]
    pb = _iid_problem(data)
    exact = _exact_log_marginal(y, 0.0, 1.0, 0.5)
    mode = posterior_mode(pb)
    mh = metropolis_hastings(pb, mode, draws=4000, chains=4, seed=1)
    post_mean = mode.values["mu"]
    assert mh.mean()["mu"] == pytest.approx(post_mean, abs=0.02)
    assert max(mh.rhat().values()) < 1.02
    assert min(mh.ess().values()) > 500
    assert mh.log_marginal_mhm() == pytest.approx(exact, abs=0.05)
    lo, hi = mh.hpd(0.9)["mu"]
    assert lo < post_mean < hi
    s = smc(pb, particles=600, stages=20, seed=2)
    assert s.log_marginal == pytest.approx(exact, abs=0.15)
    assert s.mean()["mu"] == pytest.approx(post_mean, abs=0.02)


def test_prior_predictive_counts_indeterminacy():
    # Taylor principle: determinate iff phi > 1. Under phi ~ U(0, 2) half the draws are.
    m = dk.from_equations(["pi = pi(+1)/phi + u", "u(+1) = 0.5*u + e", "i = phi*pi"],
                          states=["u"], jumps=["pi", "i"], shocks=["e"], parameters=["phi"])
    pb = EstimationProblem(model=m, data={"pi": np.zeros(5)}, params=dict(phi=1.5),
                           observables=["pi"], priors={"phi": ("uniform", 1.0, 2 / math.sqrt(12))},
                           guess=dict(u=0, pi=0, i=0))
    out = prior_predictive(pb, draws=400, seed=3)
    assert out["determinate"] == pytest.approx(0.5, abs=0.08)
    assert out["determinate"] + out["blanchard_kahn"] + out["no_steady_state"] \
        + out["other"] == pytest.approx(1.0)


def test_model_probabilities():
    p = model_probabilities({"A": -100.0, "B": -100.0 + math.log(3)})
    assert p["B"] == pytest.approx(0.75)


def test_decompositions_and_forecast_are_consistent():
    m = dk.from_equations(["a(+1) = 0.9*a + e1", "b(+1) = 0.5*b + e2", "y = 1 + a + b"],
                          states=["a", "b"], jumps=["y"], shocks=["e1", "e2"], parameters=[])
    sol = dk.solve(m, {}, guess=dict(a=0, b=0, y=1))
    y = sol.to_levels(sol.simulate(periods=30, cov={"e1": 0.1, "e2": 0.1}, seed=4))["y"]
    sp = state_space(sol, ["y"], cov={"e1": 0.1, "e2": 0.1}, measurement=0.01)
    sm = kalman_smoother(sp, {"y": y})
    dec = shock_decomposition(sm, variables=["y", "a"])
    for v in ("y", "a"):
        parts = dec["initial"][v] + dec["shocks"]["e1"][v] + dec["shocks"]["e2"][v]
        assert np.allclose(parts, dec["total"][v], atol=1e-14)
    rt = realtime_shock_decomposition(sp, {"y": y}, variables=["y"])
    assert rt["total"]["y"][-1] == pytest.approx(dec["total"]["y"][-1], abs=1e-12)
    f = forecast(sp, {"y": y}, 12, variables=["y"])
    fr = kalman_filter(sp, {"y": y})
    a = fr.a_filt[-1]
    for h in range(12):
        a = sol.P @ a
        assert f.mean["y"][h] == pytest.approx(1 + a.sum(), abs=1e-12)
    assert np.all(np.diff(f.std["y"]) >= -1e-12)      # uncertainty grows with the horizon


def test_posterior_forecast_bands_widen_with_parameter_uncertainty():
    from dsgelab.estimation import posterior_forecast
    data = _iid_data(T=40)
    pb = _iid_problem(data)
    mode = posterior_mode(pb)
    mh = metropolis_hastings(pb, mode, draws=1000, chains=2, seed=5)
    only_params = posterior_forecast(pb, mh, 4, variables=["y"], shocks=False, n_draws=300)
    full = posterior_forecast(pb, mh, 4, variables=["y"], n_draws=300)
    # without shocks the spread is the posterior spread of mu; with them it is wider
    width_p = only_params["upper"]["y"] - only_params["lower"]["y"]
    width_f = full["upper"]["y"] - full["lower"]["y"]
    assert np.all(width_f > width_p)
    assert only_params["mean"]["y"][0] == pytest.approx(mh.mean()["mu"], abs=0.01)
