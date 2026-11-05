"""Markov-switching first-order solution: closed forms, Klein, stability, IRFs and moments."""

import numpy as np
import pytest

import dsgelab as dsg
from dsgelab.examples import nk, rbc

T2 = np.array([[0.9, 0.1], [0.25, 0.75]])


def fisher_model():
    """alpha(s) pi_t = r_t + E_t pi_{t+1}, with r_{t+1} = rho r_t + e_{t+1}."""
    return dsg.Model(states=["r"], jumps=["pi"], shocks=["e"],
                     equations=lambda nxt, cur, eps, p: [
                         nxt.r - p["rho"] * cur.r - eps.e,
                         p["alpha"] * cur.pi - cur.r - nxt.pi])


def ar_model():
    """y_t = mu(s) + z_t, z_{t+1} = phi z_t + sig e_{t+1}."""
    return dsg.Model(states=["z"], jumps=["y"], shocks=["e"],
                     equations=lambda nxt, cur, eps, p: [
                         nxt.z - p["phi"] * cur.z - p["sig"] * eps.e,
                         cur.y - p["mu"] - cur.z])


@pytest.mark.parametrize("build, name", [(rbc.build, "rho"), (nk.build, "rho_a")])
def test_identical_regimes_give_the_klein_solution(build, name):
    model, params, guess = build()
    sol = dsg.solve(model, params, guess=guess)
    v = getattr(params, name)
    ms = dsg.solve_markov_switching(model, params, {name: (v, v)}, T2, guess=guess)
    assert ms.perturbed == []
    np.testing.assert_allclose(ms.F, np.broadcast_to(sol.F, ms.F.shape), atol=1e-13)
    np.testing.assert_allclose(ms.P, np.broadcast_to(sol.P, ms.P.shape), atol=1e-13)
    np.testing.assert_allclose(ms.Q, np.broadcast_to(sol.Q, ms.Q.shape), atol=1e-13)
    assert np.max(np.abs(ms.f)) == 0.0 and np.max(np.abs(ms.p)) == 0.0
    irf = sol.irf(model.shocks[0], periods=12)
    for regime in (0, 1):
        got = ms.irf(model.shocks[0], periods=12, regime=regime)
        for n in model.names:
            np.testing.assert_allclose(got[n], irf[n], atol=1e-12)


def test_fisherian_model_has_the_closed_form():
    alpha, rho = np.array([1.8, 1.1]), 0.7
    ms = dsg.solve_markov_switching(fisher_model(), dict(alpha=1.0, rho=rho), {"alpha": alpha},
                                    T2, guess=dict(r=0.0, pi=0.0))
    c = np.linalg.solve(np.diag(alpha) - rho * T2, np.ones(2))
    np.testing.assert_allclose(ms.F[:, 0, 0], c, rtol=1e-13)
    assert ms.solution.stable and ms.solution.radius == pytest.approx(rho ** 2)


def test_frwz_simple_model_is_solved_exactly():
    # Foerster et al. (2016), section 5: pi_t = -sigma(s)/phi(s) eps_t.
    model = dsg.Model(states=["e"], jumps=["pi"], shocks=["eps"],
                      equations=lambda nxt, cur, eps, p: [
                          nxt.e - eps.eps, p["phi"] * cur.pi + p["sigma"] * cur.e - nxt.pi])
    T = [[0.95, 0.05], [0.15, 0.85]]
    ms = dsg.solve_markov_switching(model, dict(phi=1.0, sigma=0.1),
                                    {"phi": (1.25, 0.96), "sigma": (0.1, 0.6)}, T,
                                    guess=dict(e=0.0, pi=0.0))
    np.testing.assert_allclose(ms.F[:, 0, 0], [-0.08, -0.625], rtol=1e-13)
    # With the lag as a state the quadratic system has four solutions, one of them stable.
    lagged = dsg.Model(states=["pl"], jumps=["pi"], shocks=["eps"],
                       equations=lambda nxt, cur, eps, p: [nxt.pl - cur.pi,
                                                           p["phi"] * cur.pi - nxt.pi])
    system, _, _ = dsg.switching_system(lagged, dict(phi=1.0), {"phi": (1.25, 0.96)}, T,
                                        guess=dict(pl=0.0, pi=0.0))
    found, tried, converged = dsg.msv_solutions(system, starts=60, seed=0)
    assert len(found) == 4 and sum(s.stable for s in found) == 1
    assert tried == 62 and converged >= 4
    a1, a2 = (np.linalg.solve(np.array(T), [1.25, 0.96]))
    assert any(np.allclose(s.F[:, 0, 0], [a1, a2]) for s in found)


def test_mean_square_stability_of_a_scalar_chain():
    a = np.array([1.1, 0.5])
    T = np.array([[0.8, 0.2], [0.4, 0.6]])
    op = np.array([[T[0, 0] * a[0] ** 2, T[1, 0] * a[1] ** 2],
                   [T[0, 1] * a[0] ** 2, T[1, 1] * a[1] ** 2]])
    P = np.array([[[[a[0]]], [[a[0]]]], [[[a[1]]], [[a[1]]]]])
    expected = np.max(np.abs(np.linalg.eigvals(op)))
    assert dsg.mean_square_stability(P, T) == pytest.approx(expected, rel=1e-14)
    # A matrix that depends on the next regime only, Costa, Fragoso and Marques' form.
    Pn = np.array([[[a[0]]], [[a[1]]]])
    op = np.array([[T[0, 0] * a[0] ** 2, T[1, 0] * a[0] ** 2],
                   [T[0, 1] * a[1] ** 2, T[1, 1] * a[1] ** 2]])
    assert dsg.mean_square_stability(Pn, T) == pytest.approx(np.max(np.abs(np.linalg.eigvals(op))))
    with pytest.raises(ValueError):
        dsg.mean_square_stability(np.zeros((3, 1, 1)), T)


def test_perturbed_parameters_are_those_that_move_the_steady_state():
    ms = dsg.solve_markov_switching(ar_model(), dict(mu=1.0, phi=0.6, sig=0.5),
                                    {"mu": (2.0, -1.0), "phi": (0.8, 0.3)}, T2,
                                    guess=dict(z=0.0, y=1.0))
    assert ms.perturbed == ["mu"]
    pi = ms.ergodic
    assert ms.steady["y"] == pytest.approx(pi @ [2.0, -1.0])
    # y = mu(s) + z exactly: the constants put each regime on its mean.
    levels = ms.to_levels({"y": ms.f[:, 0]})["y"]
    np.testing.assert_allclose(levels, [2.0, -1.0], rtol=1e-9)
    with pytest.raises(dsg.MarkovSwitchingError, match="does not solve the model"):
        dsg.solve_markov_switching(ar_model(), dict(mu=1.0, phi=0.6, sig=0.5),
                                   {"mu": (2.0, -1.0)}, T2, guess=dict(z=0.0, y=1.0),
                                   perturbed=())


def test_moments_have_the_closed_form():
    mu, phi, sig = np.array([2.0, -1.0]), 0.6, 0.5
    ms = dsg.solve_markov_switching(ar_model(), dict(mu=1.0, phi=phi, sig=sig), {"mu": mu}, T2,
                                    guess=dict(z=0.0, y=1.0))
    pi = ms.ergodic
    m = ms.moments(cov={"e": 1.0})
    scale = ms.coords.scale[ms.model.position("y")]
    var_y = sig ** 2 / (1 - phi ** 2) + pi[0] * pi[1] * (mu[0] - mu[1]) ** 2
    assert m["std_by_name"]["y"] * scale == pytest.approx(np.sqrt(var_y), rel=1e-9)
    assert m["mean"][ms.model.position("y")] == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(m["conditional_mean"][:, 1] * scale + ms.steady["y"], mu,
                               rtol=1e-9)
    sim = ms.simulate(100000, seed=1)
    assert np.std(sim["y"]) == pytest.approx(m["std_by_name"]["y"], rel=0.03)
    assert np.mean(sim["regime"] == 0) == pytest.approx(pi[0], abs=0.03)


def test_expected_impulse_response_averages_over_regime_paths():
    ms = dsg.solve_markov_switching(ar_model(), dict(mu=1.0, phi=0.6, sig=0.5),
                                    {"mu": (2.0, -1.0), "phi": (0.9, 0.2)}, T2,
                                    guess=dict(z=0.0, y=1.0))
    expected = ms.irf("e", periods=6, regime=0, expected=True)["z"]
    chain = dsg.MarkovChain([0.0, 1.0], T2)
    draws = []
    for seed in range(4000):
        path = chain.simulate(6, start=0, seed=seed, indices=True)
        draws.append(ms.irf("e", periods=6, regime=path)["z"])
    np.testing.assert_allclose(np.mean(draws, axis=0), expected, rtol=0.03, atol=0.3)
    # phi is current-dated: from regime 0 the second period is phi(0) times the first.
    assert expected[1] == pytest.approx(0.9 * expected[0])


def test_multiple_stable_solutions_are_reported():
    # Foerster et al. (2013), psi(2) = 0.7: two of the three real solutions are stable.
    def equations(nxt, cur, eps, p):
        k = p["kappa"]

        def adj(x):
            return 1 - 0.5 * k * (x.PI - 1) ** 2
        return [nxt.rl - cur.R, nxt.e - eps.eps_r,
                1 - p["beta"] * adj(cur) * cur.Y / (adj(nxt) * nxt.Y) / np.exp(p["mu"])
                * cur.R / nxt.PI,
                (1 - p["eta"]) + p["eta"] * adj(cur) * cur.Y
                + p["beta"] * k * adj(cur) / adj(nxt) * (nxt.PI - 1) * nxt.PI
                - k * (cur.PI - 1) * cur.PI,
                (cur.rl / p["Rss"]) ** p["rho"] * cur.PI ** ((1 - p["rho"]) * p["psi"])
                * np.exp(p["sigma"] * cur.e) - cur.R / p["Rss"]]
    model = dsg.Model(states=["rl", "e"], jumps=["PI", "Y", "R"], shocks=["eps_r"],
                      equations=equations)
    beta = 0.9976
    params = dict(beta=beta, kappa=161.0, eta=10.0, rho=0.8, sigma=0.0025, mu=0.005, psi=1.0,
                  Rss=np.exp(0.005) / beta)
    Rss = params["Rss"]
    ms = dsg.solve_markov_switching(model, params, {"mu": (0.0075, 0.0025), "psi": (3.1, 0.7)},
                                    [[0.9, 0.1], [0.1, 0.9]], next_regime=("mu",), starts=60,
                                    guess=dict(rl=Rss, e=0.0, PI=1.0, Y=0.9, R=Rss))
    assert ms.perturbed == ["mu"]
    assert len(ms.solutions) == 3 and len(ms.stable_solutions) == 2
    assert ms.F[0][2, 0] == pytest.approx(0.59067, abs=1e-5)
    assert "2 mean-square stable" in ms.summary()


def test_no_stable_solution_raises_with_what_was_found():
    model = dsg.Model(states=["k"], jumps=["x"], shocks=["e"],
                      equations=lambda nxt, cur, eps, p: [nxt.k - p["a"] * cur.k - eps.e,
                                                          cur.x - cur.k])
    with pytest.raises(dsg.MarkovSwitchingError, match="mean-square stable") as err:
        dsg.solve_markov_switching(model, dict(a=1.0), {"a": (1.2, 0.95)}, [[0.9, 0.1], [0.1, 0.9]],
                                   guess=dict(k=0.0, x=0.0))
    assert len(err.value.solutions) == 1 and err.value.solutions[0].radius > 1


def test_a_forward_state_without_a_jump_is_refused():
    model = dsg.Model(states=["a"], jumps=["q"], shocks=["e"],
                      equations=lambda nxt, cur, eps, p: [nxt.a - p["rho"] * cur.a - eps.e,
                                                          cur.q - 0.9 * nxt.a])
    with pytest.raises(dsg.MarkovSwitchingError, match="law of motion"):
        dsg.solve_markov_switching(model, dict(rho=0.5), {"rho": (0.5, 0.6)}, T2,
                                   guess=dict(a=0.0, q=0.0))


def test_simulation_follows_a_given_regime_path():
    ms = dsg.solve_markov_switching(ar_model(), dict(mu=1.0, phi=0.6, sig=0.5),
                                    {"mu": (2.0, -1.0)}, T2, guess=dict(z=0.0, y=1.0))
    path = np.r_[np.zeros(30, int), np.ones(30, int)]
    sim = ms.simulate(50, seed=3, burn=10, regimes=path, levels=True)
    np.testing.assert_array_equal(sim["regime"], path[10:])
    np.testing.assert_allclose(sim["y"] - sim["z"], np.where(path[10:] == 0, 2.0, -1.0),
                               rtol=1e-9)
