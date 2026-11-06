"""Local identification: closed forms on an AR(1), known failures, and the .mod route."""

import numpy as np
import pytest

import dsgelab as dsg
from dsgelab.examples import nk


def ar1(product: bool = False):
    """``y' = mu + rho (y - mu) + e``, with ``rho = a b`` when ``product``."""
    def eqs(nxt, cur, eps, p):
        rho = p["a"] * p["b"] if product else p["rho"]
        return [nxt.y - p["mu"] - rho * (cur.y - p["mu"]) - eps.e]
    model = dsg.Model(states=["y"], jumps=["w"], shocks=["e"],
                      equations=lambda nxt, cur, eps, p: eqs(nxt, cur, eps, p)
                      + [cur.w - 2 * cur.y], name="ar1")
    params = dict(mu=0.5, rho=0.8, a=1.6, b=0.5, unused=3.0)
    return model, params, {"y": 0.5, "w": 1.0}


def test_moments_and_their_derivatives_match_the_closed_form():
    model, params, guess = ar1()
    mu, rho, s = 0.5, 0.8, 0.3
    r = dsg.identification(model, params, free=["mu", "rho", "stderr e"], observables=["y"],
                           cov={"e": s}, guess=guess, lags=2)
    v = s ** 2 / (1 - rho ** 2)
    assert r.moment_labels == ["E[y]", "cov(y, y)", "cov(y, y(-1))", "cov(y, y(-2))"]
    # a model declared in Python is linearised by finite differences: 1e-10 or so
    np.testing.assert_allclose(r.moments, [mu, v, rho * v, rho ** 2 * v], rtol=1e-9)
    dv_drho = 2 * rho * s ** 2 / (1 - rho ** 2) ** 2
    dv_ds = 2 * s / (1 - rho ** 2)
    exact = np.array([[1, 0, 0],
                      [0, dv_drho, dv_ds],
                      [0, v + rho * dv_drho, rho * dv_ds],
                      [0, 2 * rho * v + rho ** 2 * dv_drho, rho ** 2 * dv_ds]])
    np.testing.assert_allclose(r["moments"].jacobian, exact, rtol=1e-9, atol=1e-8)
    assert r.identified
    assert r.minimal_states == 1
    assert r["minimal"].required == 3 + 1 + 1


def test_spectrum_gram_matches_the_closed_form():
    model, params, guess = ar1()
    rho, s, grid = 0.8, 0.3, 400
    r = dsg.identification(model, params, free=["rho", "stderr e"], observables=["y"],
                           cov={"e": s}, guess=guess, grid=grid, criteria=["spectrum"])
    w = np.linspace(0, np.pi, grid // 2 + 1)
    z = np.exp(-1j * w)
    q = np.abs(1 - rho * z) ** 2                         # f = s^2 / (2 pi q)
    dq = -2 * np.real(z) + 2 * rho                       # dq/drho
    df = np.stack([-s ** 2 * dq / q ** 2, 2 * s / q], axis=1) / (2 * np.pi)
    weights = np.where(w == 0, 1.0, 2.0)
    G = 2 * np.pi * (df.T * weights) @ df / (2 * w.size - 1)
    np.testing.assert_allclose(r["spectrum"].jacobian, G, rtol=1e-8)


def test_a_product_of_two_parameters_is_not_identified():
    model, params, guess = ar1(product=True)
    r = dsg.identification(model, params, free=["a", "b", "stderr e", "unused"],
                           observables=["y"], cov={"e": 0.3}, guess=guess, max_group=2)
    assert not r.identified
    for c in r.checks.values():
        assert c.insensitive == ["unused"]
        assert ("a", "b") in c.dependent
        assert set(c.weak) == {"a", "b"}
        assert c.weak_pairs == [("a", "b")]
    assert r["moments"].rank == 2 and r["moments"].required == 4
    group, cos = r.collinearity[1]["a"]
    assert group == ("b",) and cos == pytest.approx(1.0, abs=1e-9)
    assert r.collinearity[1]["unused"] == ((), 0.0)
    assert "linearly dependent: a, b" in r.summary()


def test_moments_agree_with_the_solution_on_a_model_with_jumps():
    model, p, guess = nk.build()
    r = dsg.identification(model, p, free=["kappa", "rho_a"], observables=["x", "pi"],
                           cov={"eps_a": 0.01, "eps_v": 0.002}, guess=guess,
                           criteria=["moments"], max_group=0)
    sol = dsg.solve(model, p, guess=guess)
    V = sol.moments(cov={"eps_a": 0.01, "eps_v": 0.002})["cov"]
    ix, ip = model.position("x"), model.position("pi")
    np.testing.assert_allclose(r.moments[2:5], [V[ix, ix], V[ip, ix], V[ip, ip]], rtol=1e-10)
    assert r["moments"].identified


def test_measurement_error_enters_the_moments_and_skips_komunjer_ng():
    model, params, guess = ar1()
    r = dsg.identification(model, params, free=["rho", "stderr y"], observables=["y"],
                           cov={"e": 0.3}, measurement={"y": 0.1}, guess=guess, max_group=0)
    v = 0.09 / (1 - 0.64)
    assert r.moments[1] == pytest.approx(v + 0.01, rel=1e-9)
    assert "minimal" in r.skipped and "minimal" not in r.checks
    assert r["moments"].identified and r["spectrum"].identified


def test_bad_arguments():
    model, params, guess = ar1()
    kw = dict(free=["rho"], observables=["y"], guess=guess)
    with pytest.raises(ValueError, match="unknown criteria"):
        dsg.identification(model, params, criteria=["hessian"], **kw)
    with pytest.raises(ValueError, match="even"):
        dsg.identification(model, params, grid=5, **kw)
    with pytest.raises(ValueError, match="stationary"):
        dsg.identification(model, dict(params, rho=1.0), qz_criterium=1 + 1e-6, **kw)
    with pytest.raises(ValueError, match="no value"):
        dsg.identification(model, dict(params, rho=np.nan), **kw)


AR1_MOD = """
var y;
varexo e;
parameters mu rho;
mu = 0.5; rho = 0.8;
model;
y = mu + rho*(y(-1) - mu) + e;
end;
steady_state_model;
y = mu;
end;
shocks;
var e; stderr 0.3;
end;
estimated_params;
rho, beta_pdf, 0.6, 0.1;
stderr e, inv_gamma_pdf, 0.2, inf;
end;
varobs y;
"""


def test_mod_file_route_uses_the_prior_mean_by_default(tmp_path):
    pytest.importorskip("sympy")
    f = tmp_path / "ar1.mod"
    f.write_text(AR1_MOD)
    mf = dsg.from_mod(f)
    r = mf.identification()
    assert r.point == pytest.approx({"rho": 0.6, "stderr e": 0.2})
    assert r.moments[1] == pytest.approx(0.04 / (1 - 0.36), rel=1e-10)
    assert r.identified
    cal = mf.identification(point="calibration")
    assert cal.point == pytest.approx({"rho": 0.8, "stderr e": 0.3})
    some = mf.identification(point={"rho": 0.5})
    assert some.point == pytest.approx({"rho": 0.5, "stderr e": 0.2})


def _ar1_information(y, mu, rho, s):
    """The asymptotic information of an observed AR(1), by hand: the filter is exact."""
    v0 = s ** 2 / (1 - rho ** 2)
    # prediction errors and variances, and their derivatives in (mu, rho, s)
    dv = [np.array([-1.0, 0.0, 0.0])]
    F = [v0]
    dF = [np.array([0.0, 2 * rho * s ** 2 / (1 - rho ** 2) ** 2, 2 * s / (1 - rho ** 2)])]
    for t in range(1, y.size):
        dv.append(np.array([-(1 - rho), -(y[t - 1] - mu), 0.0]))
        F.append(s ** 2)
        dF.append(np.array([0.0, 0.0, 2 * s]))
    return sum(np.outer(a, a) / f + 0.5 * np.outer(b, b) / f ** 2 for a, f, b in zip(dv, F, dF))


def test_identification_strength_matches_the_closed_form():
    model, params, guess = ar1()
    rng = np.random.default_rng(3)
    y = np.empty(200)
    y[0] = 0.5
    for t in range(1, 200):
        y[t] = 0.5 + 0.8 * (y[t - 1] - 0.5) + 0.3 * rng.standard_normal()
    priors = {"mu": ("normal", 0.5, 0.25), "rho": ("beta", 0.8, 0.1),
              "stderr e": ("inv_gamma", 0.3, float("inf"))}
    r = dsg.identification(model, params, free=["mu", "rho", "stderr e"], observables=["y"],
                           cov={"e": 0.3}, guess=guess, strength=True, data={"y": y},
                           priors=priors, max_group=0)
    s = r.strength
    A = _ar1_information(y, 0.5, 0.8, 0.3)
    # a model declared in Python is linearised by finite differences
    np.testing.assert_allclose(s.information, A, rtol=1e-7, atol=1e-8 * np.abs(A).max())
    sd = np.sqrt(np.diag(np.linalg.inv(A)))
    theta = np.array([0.5, 0.8, 0.3])
    np.testing.assert_allclose(s.uncertainty, sd, rtol=1e-6)
    np.testing.assert_allclose(s.strength, theta / sd, rtol=1e-6)
    np.testing.assert_allclose(s.strength_prior[:2], np.array([0.25, 0.1]) / sd[:2], rtol=1e-6)
    assert s.strength_prior[2] == np.inf                       # inverse gamma with sd = inf
    np.testing.assert_allclose(s.sensitivity, np.sqrt(np.diag(A)) * theta, rtol=1e-7)
    assert s.identified == ["mu", "rho", "stderr e"]
    # the moments' standard deviations come from the inverse information
    J = r["moments"].jacobian
    cmm = J @ np.linalg.solve(A, J.T)
    expect = np.sqrt(np.sum((J / np.sqrt(np.diag(cmm))[:, None]) ** 2, axis=0))
    np.testing.assert_allclose(s.moments_sensitivity[:2], expect[:2] * [0.25, 0.1], rtol=1e-6)
    np.testing.assert_allclose(s.singular_values, np.linalg.svd(A, compute_uv=False), rtol=1e-7)
    np.testing.assert_allclose(s.patterns.T @ s.patterns, np.eye(3), atol=1e-12)
    # without data, a sample of `periods` is simulated, the same for the same seed
    a, b = (dsg.identification(model, params, free=["mu", "rho"], observables=["y"],
                               cov={"e": 0.3}, guess=guess, strength=True, periods=50,
                               max_group=0) for _ in range(2))
    assert a.strength.data.shape == (50, 1)
    np.testing.assert_array_equal(a.strength.data, b.strength.data)
    assert np.isnan(a.strength.moments_sensitivity).all()     # no priors, as in Dynare
    plain = dsg.identification(model, params, free=["mu"], observables=["y"], cov={"e": 0.3},
                               guess=guess, max_group=0)
    assert plain.strength is None


def test_identification_strength_leaves_out_dependent_parameters():
    model, params, guess = ar1(product=True)
    r = dsg.identification(model, params, free=["a", "b", "mu"], observables=["y"],
                           cov={"e": 0.3}, guess=guess, strength=True, periods=100,
                           max_group=0)
    s = r.strength
    assert s.identified == ["mu"]
    assert np.isnan(s.strength[:2]).all() and np.isfinite(s.strength[2])
    assert s.singular_values[-1] < 1e-6 * s.singular_values[0]
    # the weakest direction moves a and b against each other, with a b fixed
    v = np.abs(s.patterns[:, -1])
    assert v[2] < 1e-6 and v[0] / v[1] == pytest.approx(1.6 / 0.5, rel=1e-4)
