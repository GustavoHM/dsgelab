"""Kim's filter, the exact likelihood by enumeration, and estimation with switching parameters."""

import math

import numpy as np
import pytest
from scipy.stats import norm

import dsgelab as dsg
from dsgelab.examples import rbc

T2 = np.array([[0.95, 0.05], [0.10, 0.90]])


def fisher():
    """pistar + alpha (pi_t - pistar) = r_t + E_t pi_{t+1}, r_t i.i.d.: pi = mu(s) + sd(s) e."""
    model = dsg.Model(states=["r"], jumps=["pi"], shocks=["e"],
                      equations=lambda nxt, cur, eps, p: [
                          nxt.r - p["sig"] * eps.e,
                          p["pistar"] + p["alpha"] * (cur.pi - p["pistar"]) - cur.r - nxt.pi])
    params = dict(pistar=2.0, alpha=1.5, sig=1.0)
    switching = {"pistar": (2.0, 8.0), "alpha": (2.5, 1.2)}
    return model, params, switching


def ar_model():
    return dsg.Model(states=["z"], jumps=["y"], shocks=["e"],
                     equations=lambda nxt, cur, eps, p: [
                         nxt.z - p["phi"] * cur.z - p["sig"] * eps.e,
                         cur.y - p["mu"] - cur.z])


def hamilton(y, mu, sd, T):
    """Hamilton's (1989) filter and Kim's smoother for y = mu[s] + sd[s] e, written out."""
    pi = np.linalg.solve(np.vstack([(T.T - np.eye(2))[:1], np.ones(2)]), [0.0, 1.0])
    pred, filt, ll = [], [], 0.0
    prob = pi
    for t, yt in enumerate(y):
        if t:
            prob = filt[-1] @ T
        pred.append(prob)
        dens = norm.pdf(yt, mu, sd)
        ll += math.log(prob @ dens)
        filt.append(prob * dens / (prob @ dens))
    smooth = [filt[-1]]
    for t in range(len(y) - 2, -1, -1):
        smooth.insert(0, filt[t] * (T @ (smooth[0] / pred[t + 1])))
    return ll, np.array(filt), np.array(smooth)


def test_with_identical_regimes_kim_is_the_kalman_filter():
    model, params, guess = rbc.build()
    sol = dsg.solve(model, params, guess=guess)
    data = {"c": sol.simulate(60, cov={"eps_a": 0.01}, seed=3, levels=True)["c"]}
    kf = dsg.kalman_filter(dsg.state_space(sol, ["c"], measurement={"c": 0.001},
                                           cov={"eps_a": 0.01}), data)
    ms = dsg.solve_markov_switching(model, params, {"rho": (0.95, 0.95)}, T2, guess=guess)
    space = dsg.switching_state_space(ms, ["c"], measurement={"c": 0.001}, cov={"eps_a": 0.01})
    kim = dsg.kim_filter(space, data)
    assert kim.loglik == pytest.approx(kf.loglik, abs=1e-9)
    np.testing.assert_allclose(kim.states, kf.a_filt, atol=1e-10)
    np.testing.assert_allclose(kim.filtered, np.broadcast_to(ms.ergodic, kim.filtered.shape),
                               atol=1e-12)


def test_without_state_dynamics_kim_is_hamiltons_filter():
    model, params, switching = fisher()
    ms = dsg.solve_markov_switching(model, params, switching, T2, guess=dict(r=0.0, pi=4.0))
    space = dsg.switching_state_space(ms, ["pi"])
    y = ms.simulate(80, seed=7, levels=True)["pi"]
    kim = dsg.kim_filter(space, y)
    sd = np.sqrt(space.Z[:, 0, 0] ** 2)       # r has unit variance
    ll, filt, smooth = hamilton(y, space.d[:, 0], sd, T2)
    assert kim.loglik == pytest.approx(ll, abs=1e-10)
    np.testing.assert_allclose(kim.filtered, filt, atol=1e-12)
    np.testing.assert_allclose(kim.smoothed, smooth, atol=1e-12)
    np.testing.assert_allclose(space.d[:, 0], [ms.to_levels({"pi": ms.f[s]})["pi"][0]
                                               for s in (0, 1)])
    short = y[:10].copy()
    short[3] = np.nan
    assert dsg.kim_filter(space, short).loglik == pytest.approx(
        dsg.enumerated_likelihood(space, short), abs=1e-11)


def test_an_exactly_observed_autoregression_is_filtered_exactly():
    ms = dsg.solve_markov_switching(ar_model(), dict(mu=1.0, phi=0.6, sig=0.8),
                                    {"mu": (2.0, -1.0)}, [[0.9, 0.1], [0.2, 0.8]],
                                    guess=dict(z=0.0, y=1.0))
    space = dsg.switching_state_space(ms, ["y"])
    y = ms.simulate(12, seed=2, levels=True)["y"]
    init = (ms.ergodic, np.array([[y[0] - 2.0], [y[0] + 1.0]]), np.zeros((2, 1, 1)))
    data = y.copy()
    data[0] = np.nan
    kim = dsg.kim_filter(space, data, init=init)
    assert kim.loglik == pytest.approx(dsg.enumerated_likelihood(space, data, init=init),
                                       abs=1e-10)
    assert abs(kim.loglik_t[0]) < 1e-14          # no data at the first date


def test_kims_collapse_is_an_approximation_with_measurement_error():
    ms = dsg.solve_markov_switching(ar_model(), dict(mu=1.0, phi=0.6, sig=0.8),
                                    {"mu": (2.0, -1.0), "phi": (0.9, 0.3)},
                                    [[0.9, 0.1], [0.2, 0.8]], guess=dict(z=0.0, y=1.0))
    space = dsg.switching_state_space(ms, ["y"], measurement={"y": 0.5})
    y = ms.simulate(10, seed=4, levels=True)["y"]
    error = abs(dsg.kim_filter(space, y).loglik - dsg.enumerated_likelihood(space, y))
    assert 1e-8 < error < 5e-2
    with pytest.raises(ValueError, match="max_paths"):
        dsg.enumerated_likelihood(space, np.zeros(30))


def test_problem_names_switching_parameters_and_transition_probabilities():
    model, params, switching = fisher()
    y = np.zeros(5)
    problem = dsg.MarkovSwitchingProblem(model, {"pi": y}, params, ["pi"], switching, T2,
                                         free=["alpha[1]", "p[0,0]", "stderr e"],
                                         guess=dict(r=0.0, pi=4.0))
    np.testing.assert_allclose(problem.start(), [1.2, 0.95, 1.0])
    p, S, meas, sw, T = problem.unpack([1.4, 0.8, 2.0])
    assert sw["alpha"][1] == 1.4 and sw["alpha"][0] == 2.5
    np.testing.assert_allclose(T, [[0.8, 0.2], [0.1, 0.9]])
    assert S[0, 0] == pytest.approx(4.0)
    lo, hi = problem._support()
    assert (lo[1], hi[1]) == (0.0, 1.0)
    with pytest.raises(KeyError):
        dsg.MarkovSwitchingProblem(model, {"pi": y}, params, ["pi"], switching, T2,
                                   free=["beta[0]"])
    with pytest.raises(ValueError, match="every entry"):
        dsg.MarkovSwitchingProblem(model, {"pi": y}, params, ["pi"], switching, T2,
                                   free=["p[0,0]", "p[0,1]"])


def test_posterior_mode_and_metropolis_hastings_with_switching_parameters():
    model, params, switching = fisher()
    ms = dsg.solve_markov_switching(model, params, switching, T2, guess=dict(r=0.0, pi=4.0))
    y = ms.simulate(150, seed=11, levels=True)["pi"]
    problem = dsg.MarkovSwitchingProblem(
        model, {"pi": y}, params, ["pi"], switching, T2, guess=dict(r=0.0, pi=4.0),
        priors={"pistar[1]": ("normal", 6.0, 3.0), "p[1,1]": ("beta", 0.8, 0.1)})
    mode = dsg.posterior_mode(problem)
    assert mode.success
    assert mode.values["pistar[1]"] == pytest.approx(8.0, abs=1.0)
    assert 0.7 < mode.values["p[1,1]"] < 0.99
    assert problem.log_posterior(mode.x) >= problem.log_posterior([8.0, 0.9]) - 1e-6
    mcmc = dsg.metropolis_hastings(problem, mode, draws=150, chains=1, seed=0)
    assert 0.1 < mcmc.acceptance[0] < 0.9
    assert np.all(np.isfinite(mcmc.log_posterior))
    fr = problem.filter(mode.x)
    assert fr.smoothed.shape == (150, 2)
    np.testing.assert_allclose(fr.smoothed.sum(axis=1), 1.0)
    # A transition outside [0, 1] or a parameter with no stable solution has likelihood -inf.
    assert problem.log_likelihood([8.0, 1.3]) == -np.inf
