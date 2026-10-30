"""IRF matching and local projections: closed forms, ordering and covariance."""

import numpy as np
import pytest

import dsgelab as dsg


def _ar1(with_jump=False, two_shocks=False):
    shocks = ["e", "u"] if two_shocks else ["e"]
    jumps = ["x"] if with_jump else []

    def eqs(nxt, cur, eps, p):
        out = [nxt.y - p["rho"] * cur.y - eps.e]
        if two_shocks:
            out[0] -= eps.u
        if with_jump:
            out.append(cur.x - cur.y)
        return out

    return dsg.Model(states=["y"], jumps=jumps, shocks=shocks, equations=eqs, name="ar1")


def _match(model, targets, **kwargs):
    kw = dict(params={"rho": 0.5}, free={"rho": 0.5}, cov={"e": 1.0},
              guess={"y": 0.0, "x": 0.0}, optimize_=False)
    kw.update(kwargs)
    return dsg.irf_matching(model, targets, **kw)


def test_irf_closed_form_and_matlab_target_order():
    model = _ar1(with_jump=True)
    targets = [("y", "e", 2, 0.5), ("x", "e", 2, 0.5),
               ("y", "e", 1, 1.0), ("x", "e", 1, 1.0)]
    result = _match(model, targets, observables=["x", "y"])
    assert [(v, h) for v, _, h, _ in result.targets] == [("x", 1), ("x", 2), ("y", 1), ("y", 2)]
    np.testing.assert_allclose(result.model_irfs, [1.0, 0.5, 1.0, 0.5], atol=1e-12)
    assert result.objective == pytest.approx(0.0, abs=1e-25)


def test_irfs_are_absolute_deviations_from_a_nonzero_steady_state():
    def eqs(nxt, cur, eps, p):
        return [nxt.y - p["mu"] - p["rho"] * (cur.y - p["mu"]) - eps.e,
                cur.x - 2 * cur.y]
    model = dsg.Model(states=["y"], jumps=["x"], shocks=["e"], equations=eqs)
    result = dsg.irf_matching(model, [("x", "e", 1, 2.0)],
                              params={"mu": 2.0, "rho": 0.5}, free={"rho": 0.5},
                              cov={"e": 1.0}, guess={"y": 2.0, "x": 4.0}, optimize_=False)
    np.testing.assert_allclose(result.model_irfs, [2.0], atol=1e-12)


def test_cholesky_impulses_and_relative_irf_with_correlated_shocks():
    model = _ar1(two_shocks=True)
    Sigma = np.array([[0.09, 0.03], [0.03, 0.04]])
    targets = [("y", "e", 1, 0.4), ("y", "u", 1, np.sqrt(0.03))]
    result = _match(model, targets, cov=Sigma)
    np.testing.assert_allclose(result.model_irfs, [0.4, np.sqrt(0.03)], atol=1e-12)
    relative = _match(model, targets, cov=Sigma, relative_irf=True)
    np.testing.assert_allclose(relative.model_irfs, [400 / 3, 100.0], atol=1e-10)


def test_external_covariance_is_inverted_and_hessian_has_no_sample_factor():
    model = _ar1()
    targets = [("y", "e", 1, 1.0), ("y", "e", 2, 0.5), ("y", "e", 3, 0.25)]
    Omega = np.diag([4.0, 9.0, 16.0])
    result = _match(model, targets, target_cov=Omega)
    np.testing.assert_allclose(result.weighting, np.linalg.inv(Omega))
    D = np.array([[0.0], [1.0], [1.0]])  # d rho^(h-1) / d rho at rho=.5
    np.testing.assert_allclose(result.hessian, D.T @ result.weighting @ D, rtol=2e-6)
    np.testing.assert_allclose(result.cov, np.linalg.inv(D.T @ result.weighting @ D), rtol=2e-6)
    np.testing.assert_array_equal(result.target_cov, Omega)


def test_irf_estimation_recovers_ar1_persistence():
    model = _ar1()
    rho = 0.73
    targets = [("y", "e", h, rho ** (h - 1)) for h in range(1, 9)]
    result = _match(model, targets, free={"rho": 0.4}, optimize_=True,
                    bounds={"rho": (0.0, 0.99)})
    assert result.success
    assert result.values["rho"] == pytest.approx(rho, abs=2e-8)
    assert result.objective < 1e-15


def test_mod_irf_blocks_vectors_cross_weights_and_route():
    source = """
        var y x; varexo e; parameters rho; rho=0.5;
        model; y=rho*y(-1)+e; x=y; end;
        steady_state_model; y=0; x=0; end;
        shocks; var e=1; end;
        varobs y x;
        estimated_params; rho, 0.5, 0, 0.99; end;
        xx=[0.5,0.25]; ww=[2,3];
        matched_irfs;
          var y; varexo e; periods 1 2:3; values 1, (xx); weights 2;
          var x; varexo e; periods 1:2; values 1, 0.5; weights 4;
        end;
        matched_irfs_weights;
          var y; periods 1; varexo e; var x; periods 2; varexo e; weights 0.2;
        end;
        method_of_moments(mom_method=irf_matching, mode_compute=0);
    """
    mf = dsg.from_mod(source)
    result = mf.irf_matching(optimize_=False)
    assert len(result.targets) == 5
    np.testing.assert_allclose(result.model_irfs, [1.0, 0.5, 0.25, 1.0, 0.5], atol=1e-12)
    i = result.targets.index(("y", "e", 1, 1.0))
    j = result.targets.index(("x", "e", 2, 0.5))
    assert result.weighting[i, j] == result.weighting[j, i] == 0.2
    assert mf.mom_options["mom_method"] == "irf_matching"


def test_irf_matching_argument_errors():
    model = _ar1()
    with pytest.raises(ValueError, match="declared twice"):
        _match(model, [("y", "e", 1, 1), ("y", "e", 1, 1)])
    with pytest.raises(ValueError, match="starting at 1"):
        _match(model, [("y", "e", 0, 1)])
    with pytest.raises(ValueError, match="positive definite"):
        _match(model, [("y", "e", 1, 1), ("y", "e", 2, 0.5)],
               weighting=[[1, 2], [2, 1]])


def test_local_projection_recovers_ar1_irfs_in_population():
    rng = np.random.default_rng(20261024)
    T, burn, rho = 120_000, 200, 0.72
    e = rng.standard_normal(T + burn)
    y = np.zeros(T + burn)
    for t in range(1, T + burn):
        y[t] = rho * y[t - 1] + e[t]
    lp = dsg.local_projection({"y": y[burn:], "e": e[burn:]}, "e",
                              variables=["y"], horizon=10, lags=1, hac_lags=12)
    np.testing.assert_allclose(lp.irfs[:, 0], rho ** np.arange(11), atol=0.012)
    assert lp.covariance.shape == (11, 11)
    assert np.all(np.diag(lp.covariance) > 0)
    targets, Omega = lp.targets(periods=[1, 2, 3])
    assert [(v, h) for v, _, h, _ in targets] == [("y", 1), ("y", 2), ("y", 3)]
    np.testing.assert_allclose([r[3] for r in targets], lp.irfs[:3, 0])
    np.testing.assert_allclose(Omega, lp.covariance[:3, :3])


def test_local_projection_joint_target_order_and_argument_errors():
    rng = np.random.default_rng(8)
    T = 300
    data = {"y": rng.standard_normal(T), "x": rng.standard_normal(T),
            "e": rng.standard_normal(T)}
    lp = dsg.local_projection(data, "e", variables=["y", "x"], horizon=2, lags=1)
    targets, Omega = lp.targets()
    assert [(v, h) for v, _, h, _ in targets] == [("y", 1), ("y", 2), ("y", 3),
                                                   ("x", 1), ("x", 2), ("x", 3)]
    expected = lp.irfs.flatten(order="F")
    np.testing.assert_allclose([r[3] for r in targets], expected)
    assert Omega.shape == (6, 6)
    with pytest.raises(ValueError, match="horizon"):
        dsg.local_projection(data, "e", variables=["y"], horizon=-1)


def test_local_projection_hac_zero_matches_manual_hc0():
    rng = np.random.default_rng(19)
    T = 400
    e = rng.standard_normal(T)
    y = np.empty(T)
    y[0] = 0.0
    for t in range(1, T):
        y[t] = 0.6 * y[t - 1] + e[t]
    lp = dsg.local_projection({"y": y, "e": e}, "e", variables=["y"],
                              horizon=0, lags=1, hac_lags=0)
    X = np.column_stack([np.ones(T - 1), e[1:], y[:-1]])
    beta = np.linalg.lstsq(X, y[1:], rcond=None)[0]
    u = y[1:] - X @ beta
    bread = np.linalg.inv(X.T @ X)
    expected = bread @ (X.T @ (X * u[:, None] ** 2)) @ bread
    assert lp.irfs[0, 0] == pytest.approx(beta[1])
    assert lp.covariance[0, 0] == pytest.approx(expected[1, 1])
