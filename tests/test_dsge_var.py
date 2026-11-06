"""The DSGE-VAR (Del Negro and Schorfheide, 2004), against closed forms.

The comparison with Dynare's estimation(dsge_var=...) is validation/compare_bvar.py.
"""

import math

import numpy as np
import pytest
from scipy import stats

import dsgelab as dsg
from dsgelab.estimation import EstimationProblem, posterior_mode


def _model():
    """Two AR(1) states, the second driven by the first, observed around means 1 and 2."""
    def eqs(nxt, cur, eps, p):
        return [nxt.a - p["rho"] * cur.a - eps.e1,
                nxt.b - p["phi"] * cur.a - 0.5 * cur.b - eps.e2,
                cur.y - 1.0 - cur.a,
                cur.z - 2.0 - cur.b]
    return dsg.Model(states=["a", "b"], jumps=["y", "z"], shocks=["e1", "e2"], equations=eqs,
                     name="two_ar")


def _data(T=120, seed=0):
    rng = np.random.default_rng(seed)
    a = b = 0.0
    out = np.empty((T, 2))
    for t in range(T):
        a, b = 0.7 * a + 0.3 * rng.standard_normal(), 0.4 * a + 0.5 * b + 0.2 * rng.standard_normal()
        out[t] = 1.0 + a, 2.0 + b
    return {"y": out[:, 0], "z": out[:, 1]}


def _problem(data, **kw):
    base = dict(model=_model(), data=data, params=dict(rho=0.6, phi=0.3), observables=["y", "z"],
                priors={"rho": ("beta", 0.6, 0.2), "phi": ("normal", 0.3, 0.2)},
                cov={"e1": 0.3, "e2": 0.2}, guess=dict(a=0, b=0, y=1, z=2), first_obs=3,
                dsge_varlag=2, dsge_var=0.8)
    base.update(kw)
    return EstimationProblem(**base)


def _sequential(prior, Y, lags):
    """log p(Y) by conjugate updating, one observation at a time, and the final posterior."""
    Phi, XXi, S, df = prior.Phi, prior.XXi, prior.S, prior.df
    n = S.shape[0]
    P = np.linalg.inv(XXi)
    ll = 0.0
    for t in range(lags, len(Y)):
        x = np.append(Y[t - lags:t][::-1].ravel(), 1.0)
        nu = df - n + 1
        scale = S * (1 + x @ np.linalg.solve(P, x)) / nu
        ll += stats.multivariate_t(loc=x @ Phi, shape=scale, df=nu).logpdf(Y[t])
        Pn = P + np.outer(x, x)
        Phin = np.linalg.solve(Pn, P @ Phi + np.outer(x, Y[t]))
        S = S + np.outer(Y[t], Y[t]) + Phi.T @ P @ Phi - Phin.T @ Pn @ Phin
        P, Phi, df = Pn, Phin, df + 1
    return ll, Phi, S, df


def test_likelihood_is_the_marginal_of_the_var_under_the_dsge_prior():
    """Formula (A.2) is p(Y) of the VAR with the NIW prior of lambda T artificial observations,
    which conjugate updating computes one observation at a time, with T the VAR's rows."""
    data = _data()
    pb = _problem(data)
    theta = np.array([0.65, 0.35])
    Y = np.column_stack([data["y"], data["z"]])         # first_obs = 3: rows 0 and 1 start it
    prior = pb.dsge_var_prior(theta)
    ll, Phi, S, df = _sequential(prior, Y, 2)
    assert pb._dsge_var_nobs() == len(Y) - 2
    assert pb.log_likelihood(theta) == pytest.approx(ll, abs=1e-8)
    post = pb.dsge_var_posterior(theta)
    np.testing.assert_allclose(post.Phi, Phi, atol=1e-10)
    np.testing.assert_allclose(post.S, S, rtol=1e-9)
    assert post.df == pytest.approx(df)
    # the prior's mean is the VAR the model implies: E[y_t | y_{t-1}, y_{t-2}] from the solution
    k = prior.Phi.shape[0]
    assert prior.df == pytest.approx(0.8 * pb._dsge_var_nobs() - k)
    A1 = prior.Phi[:2].T
    np.testing.assert_allclose(A1, [[0.65, 0.0], [0.35, 0.5]], atol=1e-10)
    np.testing.assert_allclose(prior.Phi[2:4], 0.0, atol=1e-10)
    np.testing.assert_allclose(prior.Phi[-1], (np.eye(2) - A1) @ [1.0, 2.0], atol=1e-10)
    np.testing.assert_allclose(prior.S / (0.8 * pb._dsge_var_nobs()), np.diag([0.09, 0.04]),
                               atol=1e-10)


def test_infinite_weight_is_the_limit():
    data = _data()
    theta = np.array([0.65, 0.35])
    limit = _problem(data, dsge_var=math.inf).log_likelihood(theta)
    gaps = [abs(_problem(data, dsge_var=w).log_likelihood(theta) - limit)
            for w in (1e2, 1e3, 1e4)]
    assert gaps[0] > gaps[1] > gaps[2] and gaps[2] < 0.05
    # with an infinite weight, the Gaussian likelihood of the VAR at Phi*, Sigma*, which is the
    # likelihood of the model itself, conditional on its first two observations
    pb = _problem(data, dsge_var=1.0)
    prior = pb.dsge_var_prior(theta)
    Phi, Sigma = prior.Phi, prior.S / (1.0 * pb._dsge_var_nobs())
    Y = np.column_stack([data["y"], data["z"]])
    direct = sum(stats.multivariate_normal(np.append(Y[t - 2:t][::-1].ravel(), 1.0) @ Phi,
                                           Sigma).logpdf(Y[t]) for t in range(2, len(Y)))
    assert limit == pytest.approx(direct, abs=1e-8)


def test_estimated_weight():
    data = _data()
    fixed = _problem(data)
    est = _problem(data, dsge_var=None,
                   priors={"rho": ("beta", 0.6, 0.2), "phi": ("normal", 0.3, 0.2),
                           "dsge_prior_weight": ("uniform", math.nan, math.nan, 0.0, 2.0)})
    assert est.is_dsge_var and est.names[-1] == "dsge_prior_weight"
    theta = np.array([0.65, 0.35])
    assert est.log_likelihood(np.append(theta, 0.8)) == pytest.approx(fixed.log_likelihood(theta),
                                                                       abs=1e-12)
    assert est.log_posterior(np.append(theta, 0.8)) == pytest.approx(
        fixed.log_posterior(theta) + math.log(0.5), abs=1e-12)
    assert est.start()[-1] == pytest.approx(1.0)       # the prior mean, as in Dynare
    # below (k + n) / T the prior is improper
    k, n, T = 5, 2, est._dsge_var_nobs()
    assert est.log_likelihood(np.append(theta, 0.99 * (k + n) / T)) == -math.inf
    assert np.isfinite(est.log_likelihood(np.append(theta, 1.01 * (k + n) / T)))
    mode = posterior_mode(est)
    assert mode.success and 0.0 < mode.values["dsge_prior_weight"] < 2.0


def test_errors():
    data = _data()
    with pytest.raises(ValueError, match="first_obs must be at least 3"):
        _problem(data, first_obs=2)
    with pytest.raises(ValueError, match="measurement"):
        _problem(data, measurement={"y": 0.1})
    with pytest.raises(ValueError, match="presample"):
        _problem(data, presample=4)
    with pytest.raises(ValueError, match="positive"):
        _problem(data, dsge_var=0.0)
    with pytest.raises(ValueError, match="DSGE-VAR"):
        dsg.posterior_forecast(_problem(data), np.array([[0.6, 0.3]]), 4)
    assert not _problem(data, dsge_var=None).is_dsge_var


def test_irf_with_a_heavy_prior_is_the_model_irf():
    # the observables follow a VAR(1) with innovations (e1, e2), so the VAR(2) the model implies is
    # exact, Sigma* = A0 A0', and as lambda grows the DSGE-VAR's responses become the model's
    data = _data()
    pb = _problem(data, dsge_var=1e5)
    theta = np.array([0.7, 0.4])
    out = dsg.dsge_var_irf(pb, np.tile(theta, (40, 1)), periods=8, n_draws=None, seed=2)
    assert out["draws"] == 40 and out["explosive"] == 0.0
    A = np.array([[0.7, 0.0], [0.4, 0.5]])
    resp = np.diag([0.3, 0.2])
    for h in range(8):
        for j, e in enumerate(["e1", "e2"]):
            for i, v in enumerate(["y", "z"]):
                assert out["mean"][e][v][h] == pytest.approx(resp[i, j], abs=2e-3)
        resp = A @ resp
    assert np.all(out["variance"]["e1"]["y"] < 1e-5)


def test_irf_draws_follow_the_posterior_and_the_rotation():
    data = _data()
    pb = _problem(data)
    X = np.array([[0.65, 0.35], [0.55, 0.25], [0.7, 0.3]])
    out = dsg.dsge_var_irf(pb, X, periods=6, n_draws=None, seed=np.random.default_rng(5))
    rng = np.random.default_rng(5)
    for d, theta in enumerate(X):
        Phi, Sigma = pb.dsge_var_posterior(theta).draw(1, seed=rng)
        _, space = pb.solve(theta)
        A0 = space.Z @ space.R @ np.diag(np.sqrt(np.diag(space.Sigma)))
        B = np.array([[out["responses"][e][v][d, 0] for e in space.shocks] for v in ["y", "z"]])
        # the impact matrix factors Sigma, and is A0 rotated by the same orthogonal matrix
        np.testing.assert_allclose(B @ B.T, Sigma[0], rtol=1e-10)
        Q = np.linalg.solve(np.linalg.cholesky(Sigma[0]), B)
        np.testing.assert_allclose(Q @ Q.T, np.eye(2), atol=1e-12)
        Rr = np.linalg.solve(Q.T, A0.T)
        assert abs(Rr[1, 0]) < 1e-12 and np.all(np.diag(Rr) > 0)
        A1 = Phi[0][:2].T
        np.testing.assert_allclose([out["responses"][e]["y"][d, 1] for e in space.shocks],
                                   (A1 @ B)[0], rtol=1e-10)
    st = out["responses"]["e1"]["z"]
    np.testing.assert_allclose(out["variance"]["e1"]["z"], st.var(axis=0, ddof=1), rtol=1e-12)
    np.testing.assert_allclose(out["median"]["e1"]["z"], np.median(st, axis=0), rtol=1e-12)
    # a subsample of n_draws, and the refusals
    assert dsg.dsge_var_irf(pb, np.tile(X, (3, 1)), periods=3, n_draws=4)["draws"] == 4
    with pytest.raises(ValueError, match="dsge_var"):
        dsg.dsge_var_irf(_problem(data, dsge_var=None), X, periods=3)
