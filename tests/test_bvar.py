"""The BVAR with the Sims–Zha prior, against closed forms.

The comparison with Dynare's bvar_density and bvar_forecast is validation/compare_bvar.py.
"""

import math

import numpy as np
import pytest
from scipy import stats

import dsgelab as dsg
from dsgelab import NormalInverseWishart, bvar, bvar_density


def _var_data(T=160, seed=0):
    """A stationary VAR(2) in two variables, with means 1 and -0.5."""
    rng = np.random.default_rng(seed)
    A1 = np.array([[0.5, 0.1], [0.2, 0.3]])
    A2 = np.array([[0.2, 0.0], [-0.1, 0.1]])
    mean = np.array([1.0, -0.5])
    c = (np.eye(2) - A1 - A2) @ mean
    L = np.linalg.cholesky(np.array([[0.04, 0.01], [0.01, 0.02]]))
    y = np.tile(mean, (T + 50, 1))
    for t in range(2, T + 50):
        y[t] = c + A1 @ y[t - 1] + A2 @ y[t - 2] + L @ rng.standard_normal(2)
    return {"x": y[50:, 0], "z": y[50:, 1]}


def _regressors(Y, lags):
    return np.array([np.append(Y[t - lags:t][::-1].ravel(), 1.0) for t in range(lags, len(Y))])


def _predictive(niw, x, y):
    """log p(y | x) under the NIW: a multivariate t (the conjugate VAR's predictive)."""
    n = niw.S.shape[0]
    nu = niw.df - n + 1
    scale = niw.S * (1 + x @ niw.XXi @ x) / nu
    return stats.multivariate_t(loc=x @ niw.Phi, shape=scale, df=nu).logpdf(y)


def test_without_dummies_the_posterior_is_least_squares():
    data = _var_data()
    Y = np.column_stack([data["x"], data["z"]])
    # tau = 0 and no other dummy: the prior adds nothing, and the posterior is OLS on the
    # training sample and the rest
    b = bvar(data, 2, first_obs=13, train=10, tau=0.0, omega=0, lambda_=0.0, mu=0.0)
    X, y = _regressors(Y, 2), Y[2:]
    Phi = np.linalg.lstsq(X, y, rcond=None)[0]
    u = y - X @ Phi
    np.testing.assert_allclose(b.posterior.Phi, Phi, atol=1e-12)
    np.testing.assert_allclose(b.posterior.S, u.T @ u, rtol=1e-10)
    np.testing.assert_allclose(b.posterior.XXi, np.linalg.inv(X.T @ X), rtol=1e-8)
    # the n * lags Minnesota rows, zero with tau = 0, still count in df, as in Dynare
    assert b.posterior.df == len(y) + 2 * 2 - X.shape[1]
    np.testing.assert_allclose(b.posterior.Sigma, u.T @ u / (b.posterior.df - 3), rtol=1e-10)
    np.testing.assert_allclose(b.lag_matrices[0], Phi[:2].T, atol=1e-12)
    np.testing.assert_allclose(b.intercept, Phi[-1], atol=1e-12)


def test_marginal_density_is_the_product_of_the_predictive_densities():
    """log p(y_1..y_m) is the sum of the one-step multivariate-t predictives, and the posterior
    after m observations is the conjugate update of the one after m - 1."""
    data = _var_data(80)
    Y = np.column_stack([data["x"], data["z"]])
    kw = dict(first_obs=14, train=8, tau=3.0, decay=0.5, lambda_=5.0, mu=2.0)
    fits = [bvar(data, 2, nobs=k, **kw) for k in range(0, 12)]
    assert fits[0].log_marginal == 0.0 and fits[5].nobs == 5
    np.testing.assert_allclose(fits[0].posterior.Phi, fits[0].prior.Phi)
    for k in range(1, 12):
        t = kw["first_obs"] + k - 2                     # 0-based row of the new observation
        x = np.append(Y[t - 2:t][::-1].ravel(), 1.0)
        prev, cur = fits[k - 1].posterior, fits[k].posterior
        np.testing.assert_allclose(fits[k].log_marginal - fits[k - 1].log_marginal,
                                   _predictive(prev, x, Y[t]), rtol=0, atol=1e-8)
        P = np.linalg.inv(prev.XXi)
        Pn = P + np.outer(x, x)
        np.testing.assert_allclose(cur.Phi, np.linalg.solve(Pn, P @ prev.Phi + np.outer(x, Y[t])),
                                   atol=1e-9)
        assert cur.df == prev.df + 1


def test_a_tight_prior_is_the_random_walk():
    data = _var_data()
    b = bvar(data, 2, first_obs=13, train=10, tau=1e6, lambda_=0.0, mu=0.0)
    expected = np.zeros((5, 2))
    expected[0, 0] = expected[1, 1] = 1.0
    np.testing.assert_allclose(b.posterior.Phi[:4], expected[:4], atol=1e-6)


def test_forecasts():
    data = _var_data()
    b = bvar(data, 2, first_obs=3, nobs=150)
    assert b.realized.shape == (8, 2)
    f = b.forecast(6, 400, seed=3)
    # the point forecast iterates the VAR at the posterior mean
    hist = b.initial.copy()
    for t in range(6):
        y = sum(b.lag_matrices[l] @ hist[-1 - l] for l in range(2)) + b.intercept
        np.testing.assert_allclose([f.point["x"][t], f.point["z"][t]], y, rtol=1e-12)
        hist = np.vstack([hist[1:], y])
    assert f.paths.shape == (400, 6, 2) and f.explosive == 0.0
    # one step ahead the paths without shocks are linear in the coefficients: their mean is the
    # point forecast, up to Monte Carlo error
    sd = f.paths_no_shock[:, 0, 0].std() / math.sqrt(400)
    assert abs(f.mean(shocks=False)["x"][0] - f.point["x"][0]) < 4 * sd
    lo, hi = f.bands()
    assert np.all(lo["x"] < f.median()["x"]) and np.all(f.median()["x"] < hi["x"])
    again = b.forecast(6, 400, seed=3)
    np.testing.assert_array_equal(again.paths, f.paths)
    rm = b.rmse()
    path = b.forecast(8, draws=0).point
    assert rm["x"] == pytest.approx(np.sqrt(np.mean((path["x"] - b.realized[:, 0]) ** 2)))


def test_draws_have_the_posterior_moments():
    niw = NormalInverseWishart(Phi=np.array([[0.5, 0.1], [0.0, 0.4], [1.0, -1.0]]),
                               XXi=np.diag([0.02, 0.03, 0.5]),
                               S=np.array([[2.0, 0.3], [0.3, 1.0]]) * 30, df=40.0)
    Phi, Sigma = niw.draw(4000, seed=1)
    np.testing.assert_allclose(Sigma.mean(axis=0), niw.Sigma, rtol=0.03)
    np.testing.assert_allclose(Phi.mean(axis=0), niw.Phi, atol=4 * niw.Phi_std.max() / 60)
    np.testing.assert_allclose(Phi.std(axis=0), niw.Phi_std, rtol=0.05)
    # a degree of freedom that is not whole draws by Bartlett's decomposition
    half = NormalInverseWishart(niw.Phi, niw.XXi, niw.S, 40.5)
    _, S2 = half.draw(4000, seed=2)
    np.testing.assert_allclose(S2.mean(axis=0), half.Sigma, rtol=0.03)


def test_options_and_errors():
    data = _var_data()
    with pytest.raises(ValueError, match="exceed lags"):
        bvar(data, 2, first_obs=2)
    with pytest.raises(ValueError, match="exceed lags"):
        bvar(data, 2, first_obs=10, train=8)
    with pytest.raises(ValueError, match="do not fit"):
        bvar(data, 2, first_obs=3, nobs=1000)
    d = bvar_density(data, 3, first_obs=4)
    assert list(d) == [1, 2, 3] and d[2] == bvar(data, 2, first_obs=4).log_marginal
    # prefilter demeans and drops the constant; an array takes names y1, y2
    Y = np.column_stack([data["x"], data["z"]])
    b = bvar(Y, 1, first_obs=2, prefilter=True)
    assert b.names == ["y1", "y2"] and not b.constant and b.posterior.Phi.shape == (2, 2)
    assert dsg.bvar is bvar


def test_irf_draws_and_statistics():
    data = _var_data()
    b = bvar(data, 2, first_obs=3)
    out = b.irf(12, 300, seed=4)
    Phi, Sigma = b.posterior.draw(300, seed=4)          # the same draws, in the same order
    assert out["draws"] == 300 and out["explosive"] == 0.0 and out["coverage"] == 0.6
    for d in (0, 117, 299):
        C = b._companion(Phi[d])
        L = np.linalg.cholesky(Sigma[d])
        P = np.eye(4)
        for h in range(12):
            resp = P[:2, :2] @ L                          # Psi_h L, from the companion matrix
            for j, e in enumerate(b.names):
                for i, v in enumerate(b.names):
                    assert out["responses"][e][v][d, h] == pytest.approx(resp[i, j], abs=1e-13)
            P = C @ P
    R = out["responses"]["x"]["z"]
    ranked = np.sort(R, axis=0)
    np.testing.assert_allclose(out["mean"]["x"]["z"], R.mean(axis=0), rtol=1e-12)
    np.testing.assert_allclose(out["variance"]["x"]["z"], R.var(axis=0), rtol=1e-12)
    # Dynare's ranked draws: round(0.5 N), round(0.2 N) and round(0.8 N) of N = 300
    np.testing.assert_array_equal(out["median"]["x"]["z"], ranked[149])
    np.testing.assert_array_equal(out["lower_bound"]["x"]["z"], ranked[59])
    np.testing.assert_array_equal(out["upper_bound"]["x"]["z"], ranked[239])
    # the HPD interval holds 90% of the draws and is no wider than the central one
    lo, hi = out["lower"]["x"]["z"], out["upper"]["x"]["z"]
    inside = np.mean((R >= lo - 1e-15) & (R <= hi + 1e-15), axis=0)
    assert np.all(inside >= 0.9)
    q = np.quantile(R, [0.05, 0.95], axis=0)
    assert np.all(hi - lo <= q[1] - q[0] + 1e-12)
    # a Cholesky impulse to the second variable does not move the first on impact
    assert np.all(out["responses"]["z"]["x"][:, 0] == 0.0)


def test_irf_identifications():
    b = bvar(_var_data(), 1, first_obs=2)
    sq = b.irf(5, 50, identification="SquareRoot", seed=1)
    _, Sigma = b.posterior.draw(50, seed=1)
    for d in (0, 49):
        B = np.array([[sq["responses"][e][v][d, 0] for e in b.names] for v in b.names])
        np.testing.assert_allclose(B, B.T, atol=1e-15)
        np.testing.assert_allclose(B @ B, Sigma[d], rtol=1e-12)
    # a callable identification; the upper factor is what Dynare 7.2 takes for 'Cholesky'
    up = b.irf(5, 50, identification=lambda S: np.linalg.cholesky(S).T, seed=1)
    for d in (0, 49):
        U = np.linalg.cholesky(Sigma[d]).T
        assert up["responses"]["x"]["z"][d, 0] == pytest.approx(U[1, 0])
        assert up["responses"]["z"]["x"][d, 0] == pytest.approx(U[0, 1])
    with pytest.raises(ValueError, match="identification"):
        b.irf(5, 10, identification="sign")
    with pytest.raises(ValueError, match="coverage"):
        b.irf(5, 10, coverage=1.2)


def test_plot_posterior_irf_draws_the_bvar_irf():
    pytest.importorskip("matplotlib")
    out = bvar(_var_data(), 1, first_obs=2).irf(6, 40, seed=0)
    fig = dsg.plot_posterior_irf(out, "x")
    axes = [a for a in fig.axes if a.get_visible()]
    assert [a.get_title() for a in axes] == ["x", "z"]
    np.testing.assert_array_equal(axes[1].get_lines()[0].get_ydata(), out["mean"]["x"]["z"])
