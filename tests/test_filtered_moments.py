"""Moments of HP- and band-pass-filtered variables, against integrals of the spectrum."""

import numpy as np
import pytest
from scipy.integrate import quad

import dsgelab as dsg


def ar1(rho):
    """``y' = rho y + e``, with a static ``w = 2 y``; ``rho = 1`` is a random walk."""
    def eqs(nxt, cur, eps, p):
        return [nxt.y - p["rho"] * cur.y - eps.e, cur.w - 2 * cur.y]
    model = dsg.Model(states=["y"], jumps=["w"], shocks=["e"], equations=eqs, name="ar1")
    return dsg.solve(model, dict(rho=rho), guess={"y": 0.0, "w": 0.0}, qz_criterium=1 + 1e-6)


def hp_gain(w, lam):
    c = 4 * lam * (1 - np.cos(w)) ** 2
    return c / (1 + c)


def filtered_variance(spectrum, gain, lag=0):
    """``int_{-pi}^{pi} gain(w)^2 f(w) cos(w lag) dw``, the continuous-frequency answer."""
    def f(w):
        return gain(w) ** 2 * spectrum(w) * np.cos(w * lag)
    return 2 * quad(f, 0, np.pi, limit=400, epsabs=1e-14, epsrel=1e-12)[0]


@pytest.mark.parametrize("lam", [1600, 100])
def test_hp_filtered_ar1_matches_the_integral_of_its_spectrum(lam):
    rho, s = 0.9, 0.01
    sol = ar1(rho)

    def spectrum(w):
        return s ** 2 / (2 * np.pi * (1 - 2 * rho * np.cos(w) + rho ** 2))

    mom = sol.moments({"e": s}, lags=2, hp_filter=lam, grid=2 ** 14)
    y, w = sol.model.position("y"), sol.model.position("w")
    exact = [filtered_variance(spectrum, lambda x: hp_gain(x, lam), k) for k in range(3)]
    assert mom["cov"][y, y] == pytest.approx(exact[0], rel=1e-9)
    assert mom["cov"][w, w] == pytest.approx(4 * exact[0], rel=1e-9)
    for k in (1, 2):
        assert mom["autocov"][k][y, y] == pytest.approx(exact[k], rel=1e-8)
        assert mom["autocorr"][k]["y"] == pytest.approx(exact[k] / exact[0], rel=1e-8)


def test_hp_filter_lets_a_unit_root_through():
    s = 0.01
    sol = ar1(1.0)
    plain = sol.moments({"e": s})
    assert np.isnan(plain["std_by_name"]["y"])            # no unfiltered variance
    mom = sol.moments({"e": s}, hp_filter=1600, grid=2 ** 14)

    def spectrum(w):
        return s ** 2 / (2 * np.pi * 2 * (1 - np.cos(w)))

    exact = filtered_variance(spectrum, lambda x: hp_gain(x, 1600))
    assert mom["std_by_name"]["y"] ** 2 == pytest.approx(exact, rel=1e-8)


def test_band_pass_of_white_noise_keeps_the_share_of_the_band():
    s, grid = 0.02, 512
    sol = ar1(0.0)
    mom = sol.moments({"e": s}, lags=3, bandpass=(6, 32), grid=grid)
    w = np.arange(grid) * 2 * np.pi / grid
    inside = ((w >= 2 * np.pi / 32) & (w <= 2 * np.pi / 6)) | \
             ((w <= 2 * np.pi - 2 * np.pi / 32) & (w >= 2 * np.pi - 2 * np.pi / 6))
    y = sol.model.position("y")
    assert mom["cov"][y, y] == pytest.approx(s ** 2 * inside.mean(), rel=1e-12)
    for k in (1, 2, 3):
        expected = s ** 2 * np.mean(inside * np.cos(w * k))
        assert mom["autocov"][k][y, y] == pytest.approx(expected, rel=1e-10, abs=1e-18)


def test_complementary_bands_add_up_to_the_variance_without_the_zero_frequency():
    s, grid = 0.01, 512
    sol = ar1(0.7)
    y = sol.model.position("y")
    bands = [(2, 6), (6.0001, 32), (32.0001, 1e9)]
    total = sum(sol.moments({"e": s}, bandpass=b, grid=grid)["cov"][y, y] for b in bands)
    w = np.arange(1, grid) * 2 * np.pi / grid
    f = s ** 2 / (1 - 2 * 0.7 * np.cos(w) + 0.49)
    assert total == pytest.approx(f.sum() / grid, rel=1e-10)


def test_decomposition_of_filtered_variance_and_covariance():
    def eqs(nxt, cur, eps, p):
        return [nxt.a - 0.9 * cur.a - eps.ea, nxt.b - 0.5 * cur.b - eps.eb,
                cur.y - cur.a - cur.b]
    model = dsg.Model(states=["a", "b"], jumps=["y"], shocks=["ea", "eb"], equations=eqs)
    sol = dsg.solve(model, {}, guess={"a": 0.0, "b": 0.0, "y": 0.0})
    cov = {"ea": 0.01, "eb": 0.02}
    dec = sol.variance_decomposition(cov, hp_filter=1600)
    V = sol.covariance(cov, hp_filter=1600)
    va = sol.covariance({"ea": 0.01, "eb": 0.0}, hp_filter=1600)
    y = model.position("y")
    assert dec["variance"][y].sum() == pytest.approx(V[y, y], rel=1e-12)
    assert dec["share"][y, 0] == pytest.approx(va[y, y] / V[y, y], rel=1e-12)


def test_unfiltered_moments_are_unchanged_and_expose_the_autocovariances():
    sol = ar1(0.8)
    mom = sol.moments({"e": 0.1}, lags=2)
    v = 0.01 / (1 - 0.64)
    y = sol.model.position("y")
    assert mom["cov"][y, y] == pytest.approx(v, rel=1e-9)
    assert mom["autocov"][2][y, y] == pytest.approx(0.64 * v, rel=1e-9)
    assert mom["autocorr"][1]["y"] == pytest.approx(0.8, rel=1e-9)


def test_bad_filter_arguments():
    sol = ar1(0.8)
    with pytest.raises(ValueError, match="not both"):
        sol.moments(hp_filter=1600, bandpass=(6, 32))
    with pytest.raises(ValueError, match="shortest < longest"):
        sol.moments(bandpass=(32, 6))
    with pytest.raises(ValueError, match="positive"):
        sol.moments(hp_filter=0)
    with pytest.raises(ValueError, match="horizon"):
        sol.variance_decomposition(horizon=4, hp_filter=1600)
