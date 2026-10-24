"""Method of moments: numerical invariants and simulation-size corrections."""

import numpy as np
import pytest

import dsgelab as dsg


def _ar1():
    def eqs(nxt, cur, eps, p):
        return [nxt.y - p["rho"] * cur.y - eps.e]
    return dsg.Model(states=["y"], jumps=[], shocks=["e"], equations=eqs, name="ar1")


def _evaluate_at_start(model, data, shocks, x, weighting):
    return dsg.method_of_moments(
        model, data, params={"rho": 0.5},
        free={"rho": x[0], "stderr e": x[1]}, observables=["y"],
        moments=["y*y", "y*y(-1)", "y*y(-2)", "y*y(-3)"], method="SMM",
        cov={"e": 1.0}, guess={"y": 0.0}, simulation_multiple=2, burnin=20,
        shocks=shocks, bartlett_lag=3, weighting=weighting, optimize_=False,
    )


def test_smm_simulation_multiple_corrects_covariance_and_j():
    """With S simulations per observation, Var(theta) has factor 1+1/S and J its inverse."""
    model = _ar1()
    rng = np.random.default_rng(4)
    T, burnin, multiple = 80, 20, 2
    e = rng.standard_normal(T + burnin)
    y = np.zeros(T + burnin)
    for t in range(1, T + burnin):
        y[t] = 0.5 * y[t - 1] + e[t]
    data = {"y": y[burnin:]}
    shocks = rng.standard_normal((burnin + multiple * T, 1))
    x = np.array([0.5, 1.0])

    result = _evaluate_at_start(model, data, shocks, x, ("identity", "optimal"))
    step = 1e-5
    D = np.empty((len(result.moments), len(x)))
    for j in range(len(x)):
        xp, xm = x.copy(), x.copy()
        xp[j] += step
        xm[j] -= step
        mp = _evaluate_at_start(model, data, shocks, xp, ("identity",)).model_moments
        mm = _evaluate_at_start(model, data, shocks, xm, ("identity",)).model_moments
        D[:, j] = (mp - mm) / (2 * step)

    correction = 1 + 1 / multiple
    covariance = np.linalg.inv(D.T @ result.weighting @ D) * correction / T
    np.testing.assert_allclose(result.cov, covariance, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(np.diag(result.cov),
                               [result.stderr[n] ** 2 for n in result.names], rtol=1e-12)
    assert result.J["stat"] == pytest.approx(T * result.objective / correction, rel=1e-12)
    assert result.J["dof"] == 2


def test_gmm_closed_form_moments_duplicates_and_prefilter():
    def eqs(nxt, cur, eps, p):
        return [nxt.y - p["mu"] - p["rho"] * (cur.y - p["mu"]) - eps.e]
    model = dsg.Model(states=["y"], jumps=[], shocks=["e"], equations=eqs)
    data = {"y": np.random.default_rng(5).standard_normal(30)}
    kwargs = dict(params={"mu": 2.0, "rho": 0.5}, free={}, observables=["y"],
                  moments=["y", "y*y", "y*y(-1)", "y^2", "y(-1)*y"], method="GMM",
                  cov={"e": 1.0}, guess={"y": 2.0}, weighting=("identity",), optimize_=False)
    result = dsg.method_of_moments(model, data, **kwargs)
    assert result.moments == ["y", "y*y", "y*y(-1)"]
    variance = 1 / (1 - 0.5 ** 2)
    np.testing.assert_allclose(result.model_moments,
                               [2.0, variance + 4.0, 0.5 * variance + 4.0], rtol=1e-10)
    centred = dsg.method_of_moments(model, data, prefilter=True, **kwargs)
    np.testing.assert_allclose(centred.model_moments, [2.0, variance, 0.5 * variance], rtol=1e-10)


def test_smm_reproduces_manual_simulation_with_fixed_draws():
    model = _ar1()
    T, burnin = 12, 3
    draws = np.random.default_rng(6).standard_normal((T + burnin, 1))
    result = dsg.method_of_moments(
        model, {"y": np.zeros(T)}, params={"rho": 0.5},
        free={"rho": 0.5, "stderr e": 1.0}, observables=["y"],
        moments=["y*y", "y*y(-1)"], method="SMM", cov={"e": 1.0},
        guess={"y": 0.0}, simulation_multiple=1, burnin=burnin, shocks=draws,
        weighting=("identity",), optimize_=False,
    )
    y = np.zeros(T + burnin)
    for t in range(T + burnin):
        y[t] = 0.5 * y[t - 1] + draws[t, 0]
    y = y[burnin:]
    expected = [np.mean(y * y), np.mean(y[1:] * y[:-1])]
    np.testing.assert_allclose(result.model_moments, expected, rtol=1e-12, atol=1e-12)
    repeated = dsg.method_of_moments(
        model, {"y": np.zeros(T)}, params={"rho": 0.5},
        free={"rho": 0.5, "stderr e": 1.0}, observables=["y"],
        moments=["y*y", "y*y(-1)"], method="SMM", cov={"e": 1.0},
        guess={"y": 0.0}, simulation_multiple=1, burnin=burnin, shocks=draws,
        weighting=("identity",), optimize_=False,
    )
    np.testing.assert_array_equal(repeated.model_moments, result.model_moments)


def test_correlated_shocks_enter_gmm_after_standard_deviations():
    def eqs(nxt, cur, eps, p):
        return [nxt.y - p["rho"] * cur.y - eps.e - eps.u]
    model = dsg.Model(states=["y"], jumps=[], shocks=["e", "u"], equations=eqs)
    result = dsg.method_of_moments(
        model, {"y": np.zeros(20)}, params={"rho": 0.5},
        free={"stderr e": 0.3, "stderr u": 0.4, "corr e, u": 0.2},
        observables=["y"], moments=["y*y"], method="GMM", cov={"e": 1.0, "u": 1.0},
        guess={"y": 0.0}, weighting=("identity",), optimize_=False,
    )
    shock_variance = 0.3 ** 2 + 0.4 ** 2 + 2 * 0.2 * 0.3 * 0.4
    assert result.model_moments[0] == pytest.approx(shock_variance / (1 - 0.5 ** 2), rel=1e-10)


def test_mod_file_method_of_moments_uses_estimated_params_and_matched_moments():
    source = """
        var y; varexo e; parameters rho;
        rho = 0.5;
        model; y = rho*y(-1) + e; end;
        steady_state_model; y = 0; end;
        shocks; var e = 0.04; end;
        varobs y;
        estimated_params;
          rho, 0.4, 0, 0.99;
          stderr e, 0.3, 1e-8, 1;
        end;
        matched_moments; y*y; y*y(-1); end;
    """
    mf = dsg.from_mod(source)
    result = mf.method_of_moments({"y": np.zeros(20)}, "GMM",
                                  weighting=("identity",), optimize_=False)
    assert result.names == ["rho", "stderr e"]
    assert result.moments == ["y*y", "y*y(-1)"]
    np.testing.assert_allclose(result.x, [0.4, 0.3])
    variance = 0.3 ** 2 / (1 - 0.4 ** 2)
    np.testing.assert_allclose(result.model_moments, [variance, 0.4 * variance], rtol=1e-10)


def test_moment_estimation_argument_errors():
    model = _ar1()
    kwargs = dict(params={"rho": 0.5}, free={"rho": 0.5}, observables=["y"],
                  cov={"e": 1.0}, guess={"y": 0.0}, optimize_=False)
    with pytest.raises(ValueError, match="up to the second"):
        dsg.method_of_moments(model, {"y": np.zeros(10)}, moments=["y^3"], **kwargs)
    with pytest.raises(ValueError, match="not observables"):
        dsg.method_of_moments(model, {"y": np.zeros(10)}, moments=["x*x"], **kwargs)
    with pytest.raises(ValueError, match="shocks must be"):
        dsg.method_of_moments(model, {"y": np.zeros(10)}, moments=["y*y"], method="SMM",
                              simulation_multiple=1, burnin=2, shocks=np.zeros((3, 1)), **kwargs)
    with pytest.raises(ValueError, match="at least one stage"):
        dsg.method_of_moments(model, {"y": np.zeros(10)}, moments=["y*y"], weighting=(), **kwargs)
