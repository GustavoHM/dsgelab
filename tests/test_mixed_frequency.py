"""Mixed-frequency data: temporal aggregation in the state space, against closed forms.

The comparison with Dynare (fs2000 with output growth observed as an annual mean) is
validation/compare_mixed.py.
"""

import math

import numpy as np
import pytest
from scipy import stats

import dsgelab as dsg
from dsgelab.derivatives import log_likelihood_gradient
from dsgelab.estimation import EstimationProblem
from dsgelab.kalman import kalman_filter, kalman_smoother, state_space

RHO, SD = 0.7, 0.4


def _model():
    def eqs(nxt, cur, eps, p):
        return [nxt.y - p["rho"] * cur.y - eps.e, cur.w - 1.0 - cur.y]
    return dsg.Model(states=["y"], jumps=["w"], shocks=["e"], equations=eqs, name="ar1"), \
        {"y": 0.0, "w": 1.0}


def _quarterly(T, seed=0):
    rng = np.random.default_rng(seed)
    y = np.zeros(T)
    y[0] = SD / math.sqrt(1 - RHO ** 2) * rng.standard_normal()
    for t in range(1, T):
        y[t] = RHO * y[t - 1] + SD * rng.standard_normal()
    return y


def _exact(y_obs, every, T, how="mean"):
    """The Gaussian log likelihood of the aggregates, from the AR(1)'s autocovariances."""
    lags = np.abs(np.subtract.outer(np.arange(T), np.arange(T)))
    C = SD ** 2 / (1 - RHO ** 2) * RHO ** lags
    rows = [t for t in range(T) if np.isfinite(y_obs[t])]
    A = np.zeros((len(rows), T))
    w = 1.0 / every if how == "mean" else 1.0
    for r, t in enumerate(rows):
        A[r, t - every + 1:t + 1] = w
    mean = np.full(len(rows), 1.0 if how == "mean" else float(every))
    return stats.multivariate_normal(mean, A @ C @ A.T).logpdf(y_obs[rows])


def test_low_frequency_places_the_aggregates():
    np.testing.assert_array_equal(dsg.low_frequency([5.0, 6.0], 3),
                                  [np.nan, np.nan, 5.0, np.nan, np.nan, 6.0])
    out = dsg.low_frequency([1.0, 2.0, 3.0], 4, periods=10, offset=1)
    assert out.size == 10 and out[1] == 1.0 and out[5] == 2.0 and out[9] == 3.0
    assert np.isnan(out[[0, 2, 3, 4, 6, 7, 8]]).all()


def test_the_likelihood_of_annual_means_is_the_exact_one():
    model, guess = _model()
    sol = dsg.solve(model, {"rho": RHO}, guess=guess)
    T = 48
    y = 1.0 + _quarterly(T)
    annual = dsg.low_frequency(y.reshape(-1, 4).mean(axis=1), 4)
    for how, data in (("mean", annual), ("sum", 4 * annual)):
        sp = state_space(sol, ["w"], cov={"e": SD}, aggregation={"w": (how, 4)})
        assert sp.n_states == 4 and sp.states == ["y", "y(-1)", "y(-2)", "y(-3)"]
        fr = kalman_filter(sp, {"w": data}, init="stationary")
        assert fr.loglik == pytest.approx(_exact(data, 4, T, how), rel=1e-11)
    # a stock seen once a year needs no aggregation, only the missing quarters
    stock = np.where(np.arange(T) % 4 == 3, y, np.nan)
    sp = state_space(sol, ["w"], cov={"e": SD})
    fr = kalman_filter(sp, {"w": stock}, init="stationary")
    rows = np.isfinite(stock)
    lags = np.abs(np.subtract.outer(np.arange(T), np.arange(T)))[np.ix_(rows, rows)]
    exact = stats.multivariate_normal(np.ones(rows.sum()),
                                      SD ** 2 / (1 - RHO ** 2) * RHO ** lags).logpdf(stock[rows])
    assert fr.loglik == pytest.approx(exact, abs=1e-9)


def test_estimation_smoothing_and_forecasts_run_on_the_stacked_state():
    model, guess = _model()
    y = 1.0 + _quarterly(120, seed=1)
    quarterly = y.copy()
    data = {"w": dsg.low_frequency(y.reshape(-1, 4).mean(axis=1), 4), "y": quarterly - 1.0}
    pb = EstimationProblem(model, data, params={"rho": 0.5}, observables=["w", "y"],
                           priors={"rho": ("beta", 0.5, 0.2), "stderr e": ("inv_gamma", 0.3, 1.0),
                                   "stderr y": ("inv_gamma", 0.05, 1.0)},
                           guess=guess, aggregation={"w": ("mean", 4)})
    theta = np.array([0.6, 0.35, 0.05])
    g = log_likelihood_gradient(pb, theta)
    h = 1e-6
    fd = [(pb.log_likelihood(theta + h * e) - pb.log_likelihood(theta - h * e)) / (2 * h)
          for e in np.eye(3)]
    np.testing.assert_allclose(g.gradient, fd, rtol=1e-6)
    assert g.loglik == pytest.approx(pb.log_likelihood(theta), rel=1e-12)
    # the smoother's variables are the model's, and the shock decomposition adds up
    sol, space = pb.solve(theta)
    sm = kalman_smoother(space, pb.Y)
    assert sm.states.shape == (120, 4) and sm.variables()["w"].shape == (120,)
    dec = dsg.shock_decomposition(sm, variables=["y"])
    np.testing.assert_allclose(dec["total"]["y"], dec["initial"]["y"] + dec["shocks"]["e"]["y"],
                               atol=1e-12)
    f = dsg.forecast(space, pb.Y, 3, variables=["y"])
    np.testing.assert_allclose(f.mean["y"], sm.states[-1, 0] * 0.6 ** np.arange(1, 4) * 1.0,
                               atol=1e-10)
    ev = dsg.forecast_evaluation(pb, [100], 4, estimate="fixed", theta=theta)
    assert ev.mean["w"].shape == (1, 4) and np.isfinite(ev.std["w"]).all()


def test_aggregation_is_refused_where_it_does_not_apply():
    model, guess = _model()
    sol = dsg.solve(model, {"rho": RHO}, guess=guess)
    with pytest.raises(ValueError, match="not observed"):
        state_space(sol, ["w"], aggregation={"y": ("mean", 4)})
    with pytest.raises(ValueError, match="'mean' or 'sum'"):
        state_space(sol, ["w"], aggregation={"w": ("median", 4)})
    with pytest.raises(ValueError, match="sum of logs"):
        state_space(sol, ["w"], transforms="log", aggregation={"w": ("sum", 4)})
    with pytest.raises(ValueError, match="order=1"):
        EstimationProblem(model, {"w": np.ones(8)}, params={"rho": RHO}, observables=["w"],
                          priors={"rho": ("beta", 0.5, 0.2)}, guess=guess, order=2,
                          aggregation={"w": ("mean", 4)})
