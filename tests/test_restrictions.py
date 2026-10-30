"""Calibration restrictions on IRFs and moments: closed forms, conventions and sampling."""

import math

import numpy as np
import pytest

import dsgelab as dsg
from dsgelab.examples import nk

RHO, SD = 0.5, 0.3


def _ar1(two_shocks=False):
    """``y`` an AR(1), ``z`` its lag, ``w = 2 + y`` a jump with a nonzero steady state."""
    shocks = ["e", "u"] if two_shocks else ["e"]

    def eqs(nxt, cur, eps, p):
        out = [nxt.y - p["rho"] * cur.y - eps.e, nxt.z - cur.y, cur.w - 2.0 - cur.y]
        if two_shocks:
            out.append(nxt.v - p["rho"] * cur.v - eps.u)
        return out

    states = ["y", "z", "v"] if two_shocks else ["y", "z"]
    model = dsg.Model(states=states, jumps=["w"], shocks=shocks, equations=eqs, name="ar1")
    guess = {"y": 0.0, "z": 0.0, "v": 0.0, "w": 2.0}
    return model, guess


def _solution(two_shocks=False):
    model, guess = _ar1(two_shocks)
    return dsg.solve(model, {"rho": RHO}, guess=guess)


def test_irfs_are_one_standard_deviation_in_absolute_deviations():
    sol = _solution()
    check = dsg.check_restrictions(sol, irf=[("y", "e", "1:3", "+"), ("w", "e", [2], "+")],
                                   cov={"e": SD})
    np.testing.assert_allclose(check.values[0], SD * RHO ** np.arange(3), atol=1e-14)
    # w has a steady state of 2: its value is still the absolute deviation, not 100 * proportional
    np.testing.assert_allclose(check.values[1], [SD * RHO], atol=1e-14)
    relative = dsg.check_restrictions(sol, irf=[("w", "e", 1, "+")], cov={"e": SD},
                                      relative_irf=True)
    np.testing.assert_allclose(relative.values[0], [100.0], atol=1e-11)


def test_impulse_ignores_shock_correlation_as_dynare_does():
    sol = _solution(two_shocks=True)
    Sigma = np.array([[0.09, 0.03], [0.03, 0.04]])
    check = dsg.check_restrictions(sol, irf=[("y", "u", 1, "+"), ("v", "u", 1, "+")], cov=Sigma)
    np.testing.assert_allclose(check.values[0], [0.0], atol=1e-14)    # a Cholesky impulse moves y
    np.testing.assert_allclose(check.values[1], [0.2], atol=1e-14)
    assert check.holds_each.tolist() == [False, True]


def test_moments_variance_correlation_and_the_sign_of_leads():
    sol = _solution()
    var = SD ** 2 / (1 - RHO ** 2)
    check = dsg.check_restrictions(sol, cov={"e": SD}, moments=[
        ("y", "y", 0, (0, 1)), ("w", "w", 0, (0, 1)), ("y", "w", 0, "+"),
        ("y", "y", "1:2", "+"), ("y", "z", "-1:1", "+")])
    np.testing.assert_allclose(check.values[0], [var], rtol=1e-12)
    np.testing.assert_allclose(check.values[1], [var], rtol=1e-9)       # absolute, not proportional
    np.testing.assert_allclose(check.values[2], [1.0], rtol=1e-12)
    np.testing.assert_allclose(check.values[3], [RHO, RHO ** 2], rtol=1e-12)
    # z_t = y_{t-1}: corr(y_t, z_{t+l}) = corr(y_t, y_{t+l-1}) = rho^|l-1|
    np.testing.assert_allclose(check.values[4], [RHO ** 2, RHO, 1.0], rtol=1e-10)
    assert check.holds


def test_ranges_hold_by_or_and_restrictions_by_and():
    sol = _solution()
    one_inside = ("y", "e", "1:3", (0.1, 0.2))           # values 0.3, 0.15, 0.075
    none_inside = ("y", "e", "1:3", (1.0, 2.0))
    check = dsg.check_restrictions(sol, irf=[one_inside, none_inside], cov={"e": SD})
    assert check.inside[0].tolist() == [False, True, False]
    assert check.holds_each.tolist() == [True, False]
    assert not check.holds and check.failed == ["y(1:3), e"]
    assert "FAIL" in check.summary()


def test_bounds_are_strict_and_nan_never_holds():
    r = dsg.Restriction("irf", "y", "e", (1,), 0.0, 1.0)
    assert r.inside([0.0, 0.5, 1.0, math.nan]).tolist() == [False, True, False, False]
    assert r.bound == "[0, 1]"
    assert dsg.Restriction("moment", "y", "y", (0,), 0.0, math.inf).bound == "+"


def test_labels_and_period_forms():
    sol = _solution()
    check = dsg.check_restrictions(sol, cov={"e": SD},
                                   irf=[("y", "e", range(1, 5), "-"), ("y", "e", [3, 1], "-")],
                                   moments=[("y", "z", -2, "-"), ("y", "y", 0, "+")])
    assert [r.label for r in check.restrictions] == ["y(1:4), e", "y([1,3]), e", "y, z(-2)", "y, y"]
    assert check.restrictions[1].periods == (1, 3)


@pytest.mark.parametrize("irf, moments, message", [
    ([("q", "e", 1, "+")], [], "unknown variable"),
    ([("y", "q", 1, "+")], [], "unknown shock"),
    ([("y", "e", 0, "+")], [], "start at 1"),
    ([("y", "e", "1-4", "+")], [], "cannot read the periods"),
    ([("y", "e", 1, "*")], [], "a bound is"),
    ([("y", "e", 1, (1, 0))], [], "lower < upper"),
    ([("y", "e", 1)], [], "an IRF restriction is"),
    ([], [("y", "q", 0, "+")], "unknown variable"),
    ([], [], "no restrictions"),
])
def test_argument_errors(irf, moments, message):
    with pytest.raises(ValueError, match=message):
        dsg.check_restrictions(_solution(), irf=irf, moments=moments, cov={"e": SD})


def test_sampling_skips_indeterminate_draws_and_reproduces_point_checks():
    model, params, guess = nk.build()
    cov = {"eps_a": 0.01, "eps_v": 0.0025}
    irf = [("pi", "eps_v", "1:4", "-"), ("x", "eps_v", 1, (-0.01, -0.002))]
    moments = [("x", "pi", 0, "+")]
    priors = {"phi_pi": ("gamma", 1.5, 0.5), "stderr eps_v": ("inv_gamma", 0.0025, 0.001)}
    sample = dsg.sample_restrictions(model, params, priors, irf=irf, moments=moments, cov=cov,
                                     guess=guess, draws=120, seed=3)
    again = dsg.sample_restrictions(model, params, priors, irf=irf, moments=moments, cov=cov,
                                    guess=guess, draws=120, seed=3)
    np.testing.assert_array_equal(sample.draws, again.draws)
    # phi_pi below about 0.99 is indeterminate (see the example's docstring)
    assert set(sample.status) <= {"ok", "blanchard_kahn"}
    bk = sample.status == "blanchard_kahn"
    assert bk.any() and np.all(sample.draws[bk, 0] < 1.0) and np.all(sample.draws[~bk, 0] > 0.98)
    assert np.all(np.isnan(sample.values[0][bk])) and not sample.holds[bk].any()
    assert sample.share([0]) == pytest.approx(np.mean(sample.holds_each[~bk, 0]))
    assert 0 < sample.share([1]) < 1
    assert sample.share() == pytest.approx(np.mean(sample.holds[~bk]))
    np.testing.assert_allclose(sample.shares, [sample.share([i]) for i in range(3)])
    assert "all together" in sample.summary()

    t = int(np.flatnonzero(~bk)[0])
    phi, sd = sample.draws[t]
    sol = dsg.solve(model, dsg.set_params(params, {"phi_pi": phi}), guess=guess)
    point = dsg.check_restrictions(sol, irf=irf, moments=moments,
                                   cov={"eps_a": 0.01, "eps_v": sd})
    given = dsg.sample_restrictions(model, params, irf=irf, moments=moments, cov=cov,
                                    guess=guess, points=sample.draws[[t]],
                                    names=["phi_pi", "stderr eps_v"])
    for a, b, c in zip(point.values, sample.values, given.values):
        np.testing.assert_allclose(b[t], a, rtol=1e-10)
        np.testing.assert_allclose(c[0], a, rtol=1e-10)


def test_sampling_argument_errors():
    model, params, guess = nk.build()
    irf = [("pi", "eps_v", 1, "-")]
    with pytest.raises(ValueError, match="nothing varies"):
        dsg.sample_restrictions(model, params, irf=irf)
    with pytest.raises(ValueError, match="no prior for"):
        dsg.sample_restrictions(model, params, {"phi_pi": ("gamma", 1.5, 0.5)}, irf=irf,
                                names=["phi_pi", "kappa"])
    with pytest.raises(ValueError, match="columns"):
        dsg.sample_restrictions(model, params, irf=irf, points=[[1.5, 0.1]], names=["phi_pi"])


MOD = """
    var y z; varexo e; parameters rho; rho = 0.5;
    model; y = rho*y(-1) + e; z = y(-1); end;
    steady_state_model; y = 0; z = 0; end;
    shocks; var e = 0.09; end;
    estimated_params; rho, beta_pdf, 0.5, 0.2; stderr e, inv_gamma_pdf, 0.3, 0.1; end;
    irf_calibration;
      y(9), e, [0, 1];
    end;
    irf_calibration(relative_irf, overwrite);
      y(1:3), e, [ 20, 60 ];
      z(2), e, +;
    end;
    moment_calibration;
      y, y, [0, 1];
      y, y(1:2), +;
      y, z(-1), -;
    end;
"""


def test_mod_calibration_blocks_are_read_and_checked():
    mf = dsg.from_mod(MOD)
    assert mf.relative_irf
    assert mf.irf_calibration == [("y", "e", [1, 2, 3], (20.0, 60.0)), ("z", "e", [2], "+")]
    assert mf.moment_calibration == [("y", "y", [0], (0.0, 1.0)), ("y", "y", [1, 2], "+"),
                                     ("y", "z", [-1], "-")]
    check = mf.check_restrictions()
    np.testing.assert_allclose(check.values[0], [100, 50, 25], atol=1e-10)
    np.testing.assert_allclose(check.values[2], [0.09 / 0.75], rtol=1e-12)
    np.testing.assert_allclose(check.values[4], [RHO ** 2], rtol=1e-10)
    assert check.holds_each.tolist() == [True, True, True, True, False]
    at_mean = mf.check_restrictions("prior_mean")
    np.testing.assert_allclose(at_mean.values[2], [0.09 / 0.75], rtol=1e-12)
    moved = mf.check_restrictions({"rho": 0.8, "stderr e": 0.1})
    np.testing.assert_allclose(moved.values[2], [0.01 / 0.36], rtol=1e-12)
    sample = mf.sample_restrictions(draws=50, seed=1)
    assert sample.names == ["rho", "stderr e"] and sample.solved.all()
    np.testing.assert_allclose(sample.values[0][:, 0], 100.0, atol=1e-9)


@pytest.mark.parametrize("block", [
    "irf_calibration; y, e, +; end;",
    "irf_calibration; y(1), e, [0]; end;",
    "moment_calibration; y(1), y, +; end;",
    "moment_calibration; y, q, +; end;",
])
def test_mod_calibration_block_errors(block):
    source = MOD[:MOD.index("irf_calibration;")] + block
    with pytest.raises(dsg.DynareImportError):
        dsg.from_mod(source)


def test_mod_without_blocks_needs_restrictions():
    mf = dsg.from_mod(MOD[:MOD.index("irf_calibration;")])
    with pytest.raises(dsg.DynareImportError, match="no irf_calibration"):
        mf.check_restrictions()
    check = mf.check_restrictions(irf=[("y", "e", 1, "+")])
    np.testing.assert_allclose(check.values[0], [0.3], atol=1e-12)
