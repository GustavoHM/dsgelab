"""Out-of-sample forecast evaluation, against answers known in closed form.

The comparison with Dynare's recursive estimation and with R's forecast::dm.test is
validation/compare_forecast_eval.py.
"""

import dataclasses
import math

import numpy as np
import pytest
from scipy import stats

import dsgelab as dsg
from dsgelab import diebold_mariano, forecast_evaluation
from dsgelab.estimation import EstimationProblem, maximum_likelihood


def _ar1_model():
    def eqs(nxt, cur, eps, p):
        return [nxt.y - p["rho"] * cur.y - eps.e]
    return dsg.Model(states=["y"], jumps=[], shocks=["e"], equations=eqs, name="ar1")


def _ar1_data(T, rho=0.8, sigma=0.5, seed=0):
    rng = np.random.default_rng(seed)
    y = np.zeros(T)
    y[0] = sigma / math.sqrt(1 - rho ** 2) * rng.standard_normal()
    for t in range(1, T):
        y[t] = rho * y[t - 1] + sigma * rng.standard_normal()
    return y


def _ar1_problem(y, sigma=0.5):
    return EstimationProblem(_ar1_model(), {"y": y}, params={"rho": 0.8}, observables=["y"],
                             priors={"rho": ("normal", 0.7, 0.2)}, cov={"e": sigma},
                             guess={"y": 0.0}, init="stationary")


def test_ar1_forecasts_are_the_closed_form():
    rho, sigma = 0.8, 0.5
    y = _ar1_data(60, rho, sigma)
    ev = forecast_evaluation(_ar1_problem(y, sigma), [40, 45, 50], 6, estimate="fixed",
                             theta=[rho])
    h = np.arange(1, 7)
    for o, n in enumerate(ev.origins):
        np.testing.assert_allclose(ev.mean["y"][o], rho ** h * y[n - 1], rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(ev.std["y"][o],
                                   sigma * np.sqrt((1 - rho ** (2 * h)) / (1 - rho ** 2)),
                                   rtol=1e-12)
        np.testing.assert_array_equal(ev.actual["y"][o], y[n:n + 6])
        assert ev.last["y"][o] == y[n - 1]
    # past the end of the sample the outcome is missing, and drops out of every statistic
    late = forecast_evaluation(_ar1_problem(y, sigma), [57], 6, estimate="fixed", theta=[rho])
    assert np.isfinite(late.actual["y"][0, :3]).all() and np.isnan(late.actual["y"][0, 3:]).all()
    assert np.isnan(late.rmse()["y"][3:]).all()
    # the statistics follow from the errors
    e = ev.errors["y"]
    np.testing.assert_allclose(ev.rmse()["y"], np.sqrt((e ** 2).mean(axis=0)), rtol=1e-14)
    r = ev.random_walk_errors["y"]
    np.testing.assert_allclose(ev.relative_rmse()["y"],
                               np.sqrt((e ** 2).mean(axis=0) / (r ** 2).mean(axis=0)), rtol=1e-14)
    np.testing.assert_allclose(ev.pit()["y"], stats.norm.cdf(e / ev.std["y"]), rtol=1e-14)
    np.testing.assert_allclose(ev.log_score()["y"],
                               stats.norm.logpdf(y[ev.origins[:, None] + h - 1],
                                                 ev.mean["y"], ev.std["y"]), rtol=1e-12)
    np.testing.assert_allclose(ev.log_score(joint=True), ev.log_score()["y"], rtol=1e-12)


def test_the_predictive_density_is_calibrated_at_the_true_parameters():
    # one-step forecasts from the true model: the PIT is uniform and 90% bands cover 90%
    y = _ar1_data(1600, seed=1)
    ev = forecast_evaluation(_ar1_problem(y), range(100, 1600), 1, estimate="fixed", theta=[0.8])
    pit = ev.pit()["y"][:, 0]
    assert stats.kstest(pit, "uniform").pvalue > 0.01
    assert ev.coverage_rate()["y"][0] == pytest.approx(0.9, abs=0.025)
    assert ev.coverage_rate(0.5)["y"][0] == pytest.approx(0.5, abs=0.04)
    # and the model beats the random walk on a persistent but stationary series
    assert ev.relative_rmse()["y"][0] < 1
    assert ev.diebold_mariano("y", 1, alternative="less").p_value < 0.05


def test_recursive_and_rolling_windows_reestimate_on_their_own_samples():
    y = _ar1_data(80, seed=2)
    pb = _ar1_problem(y)
    rec = forecast_evaluation(pb, [50, 60], 2, estimate="ml")
    roll = forecast_evaluation(pb, [50, 60], 2, estimate="ml", window=30)
    for o, n in enumerate([50, 60]):
        own = maximum_likelihood(dataclasses.replace(pb, data=y[:n]))
        assert rec.theta[o, 0] == pytest.approx(own.x[0], abs=1e-5)
        win = maximum_likelihood(dataclasses.replace(pb, data=y[n - 30:n]))
        assert roll.theta[o, 0] == pytest.approx(win.x[0], abs=1e-5)
        assert rec.modes[o].log_posterior == rec.objective[o]
    assert not np.isclose(rec.theta[1, 0], roll.theta[1, 0], atol=1e-3)
    # a callable estimator sees the window and the previous estimate
    seen = []

    def half(sub, x0):
        seen.append((sub.Y.shape[0], None if x0 is None else float(x0[0])))
        return [0.5]
    forecast_evaluation(pb, [50, 60], 1, estimate=half, theta=[0.3])
    assert seen == [(50, 0.3), (60, 0.5)]


def test_diebold_mariano_is_the_hln_statistic():
    rng = np.random.default_rng(5)
    e1, e2 = rng.standard_normal(40), 1.3 * rng.standard_normal(40)
    d = e1 ** 2 - e2 ** 2
    n = d.size
    g0 = np.mean((d - d.mean()) ** 2)
    g1 = np.sum((d[1:] - d.mean()) * (d[:-1] - d.mean())) / n
    k = math.sqrt((n + 1 - 4 + 2 / n) / n)
    stat = d.mean() / math.sqrt((g0 + 2 * g1) / n) * k
    t = diebold_mariano(e1, e2, 2)
    assert t.statistic == pytest.approx(stat, rel=1e-13)
    assert t.p_value == pytest.approx(2 * stats.t.cdf(-abs(stat), n - 1), rel=1e-12)
    assert diebold_mariano(e2, e1, 2).statistic == pytest.approx(-stat, rel=1e-13)
    bart = diebold_mariano(e1, e2, 2, varestimator="bartlett")
    assert bart.statistic == pytest.approx(d.mean() / math.sqrt((g0 + g1) / n) * k, rel=1e-13)
    absolute = diebold_mariano(e1, e2, power=1, alternative="greater")
    assert absolute.power == 1 and absolute.alternative == "greater"
    # missing errors drop out in pairs
    with_nan = diebold_mariano(np.append(e1, np.nan), np.append(e2, 0.3), 2)
    assert with_nan.statistic == t.statistic and with_nan.n == n


def test_diebold_mariano_falls_back_to_horizon_one():
    # a loss differential alternating in sign has a negative first autocovariance
    d = np.tile([1.0, -1.0], 10) + 1e-3 * np.arange(20)
    e1, e2 = np.sqrt(np.maximum(d, 0)), np.sqrt(np.maximum(-d, 0))
    with pytest.warns(UserWarning, match="horizon 1"):
        t = diebold_mariano(e1, e2, 2)
    assert t.horizon == 1


def test_bad_inputs_are_refused():
    y = _ar1_data(30)
    pb = _ar1_problem(y)
    with pytest.raises(ValueError, match="needs theta"):
        forecast_evaluation(pb, [20], 2, estimate="fixed")
    with pytest.raises(ValueError, match="origins"):
        forecast_evaluation(pb, [31], 2, estimate="fixed", theta=[0.8])
    with pytest.raises(ValueError, match="origins"):
        forecast_evaluation(pb, [10], 2, estimate="fixed", theta=[0.8], window=20)
    with pytest.raises(ValueError, match="one row"):
        forecast_evaluation(pb, [20, 25], 2, estimate="fixed", theta=np.ones((3, 1)))
    with pytest.raises(ValueError, match="first-order"):
        forecast_evaluation(dataclasses.replace(pb, order=2, init="auto"), [20], 2,
                            estimate="fixed", theta=[0.8])
    with pytest.raises(ValueError, match="alternative"):
        diebold_mariano([1.0, 2.0], [1.0, 1.0], alternative="both")
