"""Bootstrap particle filtering, with linear-model and argument checks."""

import numpy as np
import pytest

import dsgelab as dsg


def _ar1_data():
    def eqs(nxt, cur, eps, p):
        return [nxt.y - p["rho"] * cur.y - eps.e]
    model = dsg.Model(states=["y"], jumps=[], shocks=["e"], equations=eqs, name="ar1")
    rng = np.random.default_rng(3)
    y = np.zeros(20)
    for t in range(1, 20):
        y[t] = 0.5 * y[t - 1] + 0.2 * rng.standard_normal()
    return model, y + 0.2 * rng.standard_normal(20)


def test_particle_filter_matches_kalman_on_linear_model():
    model, data = _ar1_data()
    first = dsg.solve(model, {"rho": 0.5}, guess={"y": 0.0})
    second = dsg.solve_second_order(first, cov={"e": 0.2})
    reference = dsg.kalman_filter(
        dsg.state_space(first, ["y"], measurement={"y": 0.2}, cov={"e": 0.2}), {"y": data})
    one = dsg.particle_filter(first, {"y": data}, observables=["y"],
                              measurement={"y": 0.2}, cov={"e": 0.2},
                              particles=50_000, seed=4)
    two = dsg.particle_filter(second, {"y": data}, observables=["y"],
                              measurement={"y": 0.2}, particles=50_000, seed=4)
    assert one.loglik == pytest.approx(reference.loglik, abs=0.25)
    assert two.loglik == pytest.approx(reference.loglik, abs=0.25)
    assert two.states.shape == (20, 50_000, 1)
    assert np.all((two.ess >= 1.0) & (two.ess <= 50_000.0))
    assert np.allclose(two.weights.sum(), 1.0)


def test_particle_smoother_matches_kalman_smoother_on_linear_model():
    model, data = _ar1_data()
    first = dsg.solve(model, {"rho": 0.5}, guess={"y": 0.0})
    second = dsg.solve_second_order(first, cov={"e": 0.2})
    reference = dsg.kalman_smoother(
        dsg.state_space(first, ["y"], measurement={"y": 0.2}, cov={"e": 0.2}), {"y": data})
    for solution in (first, second):
        filtered = dsg.particle_filter(solution, {"y": data}, observables=["y"],
                                       measurement={"y": 0.2}, cov={} if solution is second else {"e": 0.2},
                                       particles=2_000, seed=4, store=True)
        smoothed = dsg.particle_smoother(filtered)
        assert np.max(np.abs(reference.states[:, 0] - smoothed.mean[:, 0])) < 0.2
        assert np.all((smoothed.ess >= 1.0) & (smoothed.ess <= 2_000.0))
    with pytest.raises(ValueError, match="store=True"):
        dsg.particle_smoother(dsg.ParticleFilterResult(
            loglik=0.0, loglik_t=np.zeros(1), states=np.zeros((1, 1, 1)),
            weights=np.ones(1), ess=np.ones(1), particles=1))


def test_particle_filter_and_smoother_accept_missing_observations():
    model, data = _ar1_data()
    data = data.copy()
    data[3] = np.nan
    data[7] = np.nan
    first = dsg.solve(model, {"rho": 0.5}, guess={"y": 0.0})
    second = dsg.solve_second_order(first, cov={"e": 0.2})
    for solution in (first, second):
        filtered = dsg.particle_filter(solution, {"y": data}, observables=["y"],
                                       measurement={"y": 0.2}, cov={} if solution is second else {"e": 0.2},
                                       particles=2_000, seed=4, store=True)
        assert np.isfinite(filtered.loglik)
        assert filtered.loglik_t[[3, 7]] == pytest.approx([0.0, 0.0])
        smoothed = dsg.particle_smoother(filtered)
        assert smoothed.mean.shape == (20, 1)
        assert np.all(np.isfinite(smoothed.mean))


def test_particle_filter_argument_errors():
    model, data = _ar1_data()
    first = dsg.solve(model, {"rho": 0.5}, guess={"y": 0.0})
    with pytest.raises(ValueError, match="positive definite"):
        dsg.particle_filter(first, {"y": data}, observables=["y"], measurement={"y": 0.0})
    with pytest.raises(ValueError, match="particles must be positive"):
        dsg.particle_filter(first, {"y": data}, observables=["y"],
                            measurement={"y": 0.2}, particles=0)
    with pytest.raises(ValueError, match="between 0 and 1"):
        dsg.particle_filter(first, {"y": data}, observables=["y"],
                            measurement={"y": 0.2}, resample=1.5)


def test_third_order_particle_filter_matches_linear_kalman():
    model, data = _ar1_data()
    first = dsg.solve(model, {"rho": 0.5}, guess={"y": 0.0})
    second = dsg.solve_second_order(first, cov={"e": 0.2})
    third = dsg.solve_third_order(second)
    reference = dsg.kalman_filter(
        dsg.state_space(first, ["y"], measurement={"y": 0.5}, cov={"e": 0.2}), {"y": data})
    result = dsg.particle_filter(third, {"y": data}, observables=["y"],
                                 measurement={"y": 0.5}, particles=10_000, seed=4)
    assert result.loglik == pytest.approx(reference.loglik, abs=0.6)
    assert result.states.shape == (20, 10_000, 1)
    assert result.history is None
    filtered = dsg.particle_filter(third, {"y": data}, observables=["y"],
                                   measurement={"y": 0.5}, particles=2_000,
                                   seed=4, store=True)
    smoothed = dsg.particle_smoother(filtered)
    reference_smooth = dsg.kalman_smoother(
        dsg.state_space(first, ["y"], measurement={"y": 0.5}, cov={"e": 0.2}), {"y": data})
    assert np.max(np.abs(reference_smooth.states[:, 0] - smoothed.mean[:, 0])) < 0.2
    assert smoothed.mean.shape == (20, 1)


def test_third_order_particle_filter_matches_kalman_for_a_jump():
    # a small measurement error makes the likelihood depend on the current shock, which
    # the observation equation must carry at every order
    pytest.importorskip("sympy")
    model = dsg.from_equations(["a(+1) = rho*a + e", "y = 1 + a"], states=["a"], jumps=["y"],
                               shocks=["e"], parameters=["rho"])
    first = dsg.solve(model, {"rho": 0.75}, guess={"a": 0.0, "y": 1.0})
    data = first.to_levels(first.simulate(periods=60, cov={"e": 0.01}, seed=2))["y"]
    reference = dsg.kalman_filter(
        dsg.state_space(first, ["y"], measurement={"y": 0.05}, cov={"e": 0.01}), {"y": data})
    third = dsg.solve_third_order(dsg.solve_second_order(first, cov={"e": 0.01}))
    for solution in (third.second, third):
        result = dsg.particle_filter(solution, {"y": data}, observables=["y"],
                                     measurement={"y": 0.05}, particles=20_000, seed=1)
        assert result.loglik == pytest.approx(reference.loglik, abs=0.5)


def test_third_order_particle_filter_nonlinear_and_missing_data():
    from dsgelab.examples import rbc
    model, params, guess = rbc.build_symbolic()
    first = dsg.solve(model, params, guess=guess)
    second = dsg.solve_second_order(first, cov={"eps_a": 0.04})
    third = dsg.solve_third_order(second)
    data = third.simulate(periods=5, seed=4, cov={"eps_a": 0.2})["c"]
    second_result = dsg.particle_filter(second, data, observables=["c"],
                                        measurement=0.5, particles=3_000, seed=3)
    third_result = dsg.particle_filter(third, data, observables=["c"],
                                       measurement=0.5, particles=3_000, seed=3)
    assert third_result.loglik == pytest.approx(second_result.loglik, abs=0.5)
    assert np.all(np.isfinite(third_result.states))
    data[2] = np.nan
    filtered = dsg.particle_filter(third, data, observables=["c"],
                                   measurement=0.5, particles=100, seed=3, store=True)
    assert filtered.loglik_t[2] == 0.0
    assert filtered.ess[2] == pytest.approx(1 / np.sum(filtered.history["weights"][2] ** 2))
    stored = dsg.particle_filter(third, data, observables=["c"], measurement=0.5,
                                 particles=500, seed=3, store=True)
    smoothed = dsg.particle_smoother(stored)
    assert smoothed.mean.shape == (5, 2)
    assert np.all(np.isfinite(smoothed.mean))


def test_first_order_filtered_states_use_the_observation_date():
    model, _ = _ar1_data()
    first = dsg.solve(model, {"rho": 0.0}, guess={"y": 0.0})
    filtered = dsg.particle_filter(first, [1.0], observables=["y"], cov={"e": 1.0},
                                   measurement={"y": 0.01}, particles=100_000, seed=3)
    # Independent N(0, 1) prior and N(0, .01**2) measurement error.
    assert filtered.weights @ filtered.states[0, :, 0] == pytest.approx(1 / 1.0001, abs=0.01)


def test_smoothing_independent_dates_preserves_filter_and_terminal_weights():
    z = [np.array([[0.0], [1.0]]), np.array([[0.0], [10.0]])]
    weights = [np.array([0.3, 0.7]), np.array([0.9, 0.1])]
    filtered = dsg.ParticleFilterResult(
        loglik=0.0, loglik_t=np.zeros(2), states=np.stack(z), weights=weights[-1],
        ess=np.ones(2), particles=2,
        history=dict(A=np.zeros((1, 1)), B=np.ones((1, 1)), c=np.zeros(1),
                     Vinov=np.ones((1, 1)), nk=1, second=False, z=z, weights=weights))
    smoothed = dsg.particle_smoother(filtered)
    np.testing.assert_allclose(smoothed.mean[:, 0], [0.7, 1.0])
    np.testing.assert_allclose(smoothed.ess, [1 / 0.58, 1 / 0.82])
    assert smoothed.method == "backward"


def test_rank_deficient_smoothing_preserves_the_deterministic_state_relation():
    model = dsg.Model(
        states=["x", "lag"], jumps=[], shocks=["e"],
        equations=lambda nxt, cur, eps, p: [nxt.x - 0.5 * cur.x - eps.e,
                                           nxt.lag - cur.x])
    first = dsg.solve(model, {}, guess={"x": 0.0, "lag": 0.0})
    filtered = dsg.particle_filter(first, [0.1, 0.2, -0.1, 0.3], observables=["x"],
                                   cov={"e": 0.2}, measurement=0.1,
                                   particles=3000, seed=2, store=True)
    smoothed = dsg.particle_smoother(filtered)
    assert smoothed.method == "ancestral"
    np.testing.assert_allclose(smoothed.mean[1:, 1], smoothed.mean[:-1, 0], atol=1e-14)
    np.testing.assert_allclose(smoothed.mean[-1], filtered.weights @ filtered.states[-1])


@pytest.mark.parametrize("order", [2, 3])
def test_nonlinear_rbc_smoothing_accepts_more_states_than_shocks(order):
    from dsgelab.examples import rbc
    model, params, guess = rbc.build_symbolic()
    second = dsg.solve_second_order(dsg.solve(model, params, guess=guess), cov={"eps_a": 0.007})
    solution = second if order == 2 else dsg.solve_third_order(second)
    data = second.simulate(periods=3, burn=20, levels=True, seed=2)
    filtered = dsg.particle_filter(solution, data, observables=["c"], measurement=0.01,
                                   particles=500, seed=2, store=True)
    smoothed = dsg.particle_smoother(filtered)
    assert smoothed.method == "ancestral"
    assert np.isfinite(smoothed.mean).all()
    np.testing.assert_allclose(smoothed.mean[-1], filtered.weights @ filtered.states[-1])
    assert smoothed.ess[-1] == pytest.approx(1 / np.sum(filtered.weights ** 2))


@pytest.mark.parametrize("order", [2, 3])
def test_ancestral_smoothing_without_resampling_matches_independent_posteriors(order):
    model, _ = _ar1_data()
    first = dsg.solve(model, {"rho": 0.0}, guess={"y": 0.0})
    second = dsg.solve_second_order(first, cov={"e": 1.0})
    solution = second if order == 2 else dsg.solve_third_order(second)
    data = np.array([0.2, 0.8, -0.5])
    filtered = dsg.particle_filter(solution, data, observables=["y"], measurement=1.0,
                                   particles=20_000, resample=None, seed=3, store=True)
    smoothed = dsg.particle_smoother(filtered)
    # Independent unit-variance prior and measurement error give mean y/2.
    np.testing.assert_allclose(smoothed.mean[:, 0], data / 2, atol=0.04)
