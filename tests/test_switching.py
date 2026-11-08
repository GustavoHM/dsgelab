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


# ------------------------------------------------------------------------------ second order
def brock_mirman():
    """Log utility and full depreciation: k' = alpha beta y and c = (1 - alpha beta) y exactly."""
    return dsg.Model(states=["k", "z"], jumps=["c", "y"], shocks=["e"],
                     equations=lambda nxt, cur, eps, p: [
                         nxt.k - (cur.y - cur.c),
                         nxt.z - np.exp(p["rho"] * np.log(cur.z) + p["sigma"] * eps.e),
                         1 / cur.c - p["beta"] * p["alpha"] * nxt.y / (nxt.k * nxt.c),
                         cur.y - p["A"] * cur.z * cur.k ** p["alpha"]])


def solve_brock_mirman(order=2):
    a, beta = 0.33, 0.96
    A, rho, sigma = np.array([1.05, 0.95]), np.array([0.9, 0.5]), np.array([0.01, 0.03])
    Abar = float(np.array([T2[1, 0], T2[0, 1]]) @ A / (T2[0, 1] + T2[1, 0]))
    k = (a * beta * Abar) ** (1 / (1 - a))
    ms = dsg.solve_markov_switching(
        brock_mirman(), dict(alpha=a, beta=beta, A=1.0, rho=0.9, sigma=0.01),
        {"A": A, "rho": rho, "sigma": sigma}, T2, next_regime=("rho", "sigma"), order=order,
        guess=dict(k=k, z=1.0, c=(1 - a * beta) * Abar * k ** a, y=Abar * k ** a))
    return ms, a, A / Abar - 1, rho, sigma


def test_second_order_with_identical_regimes_is_solve_second_order():
    model, params, guess = rbc.build()
    s2 = dsg.solve_second_order(dsg.solve(model, params, guess=guess))
    ms = dsg.solve_markov_switching(model, params, {"rho": (params.rho, params.rho)}, T2,
                                    guess=guess, order=2)
    assert isinstance(ms, dsg.SecondOrderSwitchingSolution) and ms.residual < 1e-14
    for i in range(2):
        np.testing.assert_allclose(ms.g_kk[i], s2.g_kk, atol=1e-13)
        np.testing.assert_allclose(ms.g_ss[i], s2.g_ss, atol=1e-13)
        for j in range(2):
            np.testing.assert_allclose(ms.h_kk[i, j], s2.h_kk, atol=1e-13)
            np.testing.assert_allclose(ms.h_uu[i, j], s2.h_uu, atol=1e-13)
            np.testing.assert_allclose(ms.h_ku[i, j], s2.h_ku, atol=1e-13)
            np.testing.assert_allclose(ms.h_ss[i, j], s2.h_s2, atol=1e-13)
    assert max(np.max(np.abs(ms.g_ks)), np.max(np.abs(ms.h_ks)), np.max(np.abs(ms.h_us))) == 0
    rule, mine = s2.decision_rule(), ms.decision_rule(1, 0)
    for name in ("g_k", "g_u", "g_kk", "g_uu", "g_ku"):
        np.testing.assert_allclose(getattr(mine, name), getattr(rule, name), atol=1e-13)
    mean, got = s2.mean_correction(), ms.ergodic_mean(levels=False)
    for n in model.names:
        assert got[n] == pytest.approx(mean[n], abs=1e-12)


def test_second_order_has_the_closed_form_of_log_utility_and_full_depreciation():
    ms, a, d, rho, sigma = solve_brock_mirman()
    assert ms.first.perturbed == ["A"]
    Gk = np.array([[a * (a - 1), a], [a, 0.0]])
    for i in range(2):
        Gs = np.array([a * d[i], d[i]])
        for r in range(2):                                    # c and y
            np.testing.assert_allclose(ms.g_kk[i, r], Gk, atol=1e-6)
            np.testing.assert_allclose(ms.g_ks[i, r], Gs, atol=1e-6)
        np.testing.assert_allclose(ms.g_ss[i], 0.0, atol=1e-6)
        for j in range(2):
            np.testing.assert_allclose(ms.h_kk[i, j, 0], Gk, atol=1e-6)
            np.testing.assert_allclose(ms.h_ks[i, j, 0], Gs, atol=1e-6)
            np.testing.assert_allclose(ms.h_kk[i, j, 1], np.diag([0, rho[j] * (rho[j] - 1)]),
                                       atol=1e-6)
            np.testing.assert_allclose(ms.h_ku[i, j, 1, :, 0], [0, rho[j] * sigma[j]], atol=1e-6)
            assert ms.h_uu[i, j, 1, 0, 0] == pytest.approx(sigma[j] ** 2, abs=1e-6)
            np.testing.assert_allclose(ms.h_ss[i, j], 0.0, atol=1e-6)
    # The rule for c' composes the two: log(1 + c') = log(1 + chi d') + rho' log(1 + z)
    # + sigma' eps + alpha log(1 + chi d) + alpha log(1 + z) + alpha^2 log(1 + k).
    rule = ms.decision_rule(0, 1)
    assert rule.g_kk[2, 0, 0] == pytest.approx(a ** 2 * (a ** 2 - 1), abs=1e-6)
    assert rule.g_ku[2, 0, 0] == pytest.approx(a ** 2 * sigma[1], abs=1e-6)
    assert rule.g_ks[2, 0] == pytest.approx(a ** 2 * (d[1] + a * d[0]), abs=1e-6)


def test_frwz_simple_model_has_no_second_order_terms():
    # Foerster et al. (2016), Proposition 3: the first-order solution is exact.
    model = dsg.Model(states=["e"], jumps=["pi"], shocks=["eps"],
                      equations=lambda nxt, cur, eps, p: [
                          nxt.e - eps.eps, p["phi"] * cur.pi + p["sigma"] * cur.e - nxt.pi])
    ms = dsg.solve_markov_switching(model, dict(phi=1.0, sigma=0.1),
                                    {"phi": (1.25, 0.96), "sigma": (0.1, 0.6)},
                                    [[0.95, 0.05], [0.15, 0.85]], guess=dict(e=0.0, pi=0.0),
                                    starts=20, order=2)
    for name in ("g_kk", "g_ks", "g_ss", "h_kk", "h_ku", "h_uu", "h_ks", "h_us", "h_ss"):
        assert np.max(np.abs(getattr(ms, name))) < 1e-10, name


def test_order_must_be_one_or_two():
    with pytest.raises(ValueError, match="order"):
        dsg.solve_markov_switching(fisher_model(), dict(alpha=1.0, rho=0.7), {"alpha": (1.8, 1.1)},
                                   T2, guess=dict(r=0.0, pi=0.0), order=3)


def test_second_order_conditional_mean_matches_a_simulation():
    ms, *_ = solve_brock_mirman()
    cond = ms.conditional_mean()
    sim = ms.simulate(20000, seed=1, burn=200)
    sd = np.std(sim["c"])
    for s in (0, 1):
        assert abs(np.mean(sim["c"][sim["regime"] == s]) - cond[s, 2]) < 0.1 * sd
    first = ms.first.simulate(20000, seed=1, burn=200)
    np.testing.assert_array_equal(first["regime"], sim["regime"])
    # The second-order terms move the conditional means away from the first-order ones.
    gap = np.abs(cond - ms.first.moments()["conditional_mean"])
    assert np.max(gap) > 1e-5


def test_second_order_irf_with_a_linear_model_is_the_first_order_irf():
    ms = dsg.solve_markov_switching(fisher_model(), dict(alpha=1.0, rho=0.7),
                                    {"alpha": (1.8, 1.1)}, T2, guess=dict(r=0.0, pi=0.0), order=2)
    path = [0, 0, 1, 1, 0, 1]
    got, want = ms.irf("e", periods=6, regime=path), ms.first.irf("e", periods=6, regime=path)
    for n in ("r", "pi"):
        np.testing.assert_allclose(got[n], want[n], atol=1e-12)
    np.testing.assert_allclose(ms.irf("e", periods=6, regime=path, pruning=False)["pi"],
                               want["pi"], atol=1e-12)


def test_second_order_irf_is_asymmetric_and_depends_on_the_regime():
    ms, *_ = solve_brock_mirman()
    up = ms.irf("e", size=1.0, periods=8, regime=0)
    down = ms.irf("e", size=-1.0, periods=8, regime=0)
    other = ms.irf("e", size=1.0, periods=8, regime=1)
    assert np.max(np.abs(up["c"] + down["c"])) > 1e-6
    assert np.max(np.abs(up["c"] - other["c"])) > 1e-3
    # z' = z^rho exp(sigma eps): its response at impact is exact at second order up to eps^3.
    assert up["z"][0] / 100 == pytest.approx(0.01 + 0.5 * 0.01 ** 2, rel=1e-6)


def test_max_iter_alias_in_solve_markov_switching():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        new = dsg.solve_markov_switching(fisher_model(), dict(alpha=1.0, rho=0.7),
                                         {"alpha": (1.8, 1.1)}, T2, guess=dict(r=0.0, pi=0.0),
                                         max_iter=100)
    with pytest.warns(DeprecationWarning, match="maxiter is deprecated"):
        old = dsg.solve_markov_switching(fisher_model(), dict(alpha=1.0, rho=0.7),
                                         {"alpha": (1.8, 1.1)}, T2, guess=dict(r=0.0, pi=0.0),
                                         maxiter=100)
    np.testing.assert_allclose(new.F, old.F)


def test_max_iter_alias_in_msv_solutions():
    import warnings

    system, _, _ = dsg.switching_system(fisher_model(), dict(alpha=1.0, rho=0.7),
                                        {"alpha": (1.8, 1.1)}, T2, guess=dict(r=0.0, pi=0.0))
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        new, _, _ = dsg.msv_solutions(system, max_iter=100)
    with pytest.warns(DeprecationWarning, match="maxiter is deprecated"):
        old, _, _ = dsg.msv_solutions(system, maxiter=100)
    np.testing.assert_allclose(new[0].F, old[0].F)
