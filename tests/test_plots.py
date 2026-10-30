"""Plots and the densities they draw.

The densities are compared with Dynare's draw_prior_density and posterior_moments in
validation/compare_plots.py; here they are checked against closed forms, and the plots against
the numbers they were given.
"""

import math
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import stats

import dsgelab as dsg
from dsgelab.plots import _repeated_draws
from dsgelab.priors import prior


def test_prior_density_covers_the_bulk_of_the_prior():
    for spec, mass in [(("beta", 0.5, 0.2), 0.998), (("gamma", 1.5, 0.5), 0.998),
                       (("normal", 0.0, 2.0), 0.998), (("inv_gamma", 0.1, 0.05), 0.98)]:
        x, f = dsg.prior_density(spec)
        assert x.size == 200 and np.all(np.diff(x) > 0)
        assert np.trapezoid(f, x) == pytest.approx(mass, abs=0.004)
        pr = prior(*spec)
        assert f[100] == pytest.approx(math.exp(pr.logpdf(x[100])), rel=1e-12)
    x, f = dsg.prior_density(("uniform", 0.5, 0.2))
    assert np.allclose(f, 1 / (x[-1] - x[0]))
    # a density with a pole at the edge of its grid loses the values above 10
    x, f = dsg.prior_density(("beta", 0.05, 0.1))
    assert np.isnan(f).any() and np.nanmax(f) <= 10


def test_posterior_density_is_a_gaussian_kernel_estimate():
    rng = np.random.default_rng(0)
    draws = rng.normal(0.3, 0.05, 4000)
    x, f = dsg.posterior_density(draws)
    assert x.size == 512 and x[0] < draws.min() and x[-1] > draws.max()
    assert np.trapezoid(f, x) == pytest.approx(1.0, abs=1e-3)
    h = 2 * draws.std(ddof=1) * (math.sqrt(math.pi) / (2 * math.sqrt(math.pi)) / (12 * 4000)) ** 0.2
    direct = stats.gaussian_kde(draws, bw_method=h / draws.std(ddof=1))(x[200:300])
    np.testing.assert_allclose(f[200:300], direct, rtol=0.03)   # Dynare bins at (hi - lo)/G
    # a chain that repeats each draw has twice the draws and the same density, over a wider kernel
    assert _repeated_draws(np.sort(np.repeat(draws, 2))) == pytest.approx(2.0)
    assert _repeated_draws(np.sort(draws)) == pytest.approx(1.0)
    _, f2 = dsg.posterior_density(np.repeat(draws, 2))
    assert np.max(np.abs(f2 - f)) < 0.05 * f.max()
    assert np.isnan(dsg.posterior_density(np.ones(10))[1]).all()
    with pytest.raises(ValueError, match="power of two"):
        dsg.posterior_density(draws, grid_points=500)


def _lines(ax):
    return [(line.get_xdata(), line.get_ydata()) for line in ax.get_lines()]


def test_plots_draw_the_numbers_they_are_given():
    pytest.importorskip("matplotlib")
    irf = {"y": 0.9 ** np.arange(10), "c": 0.5 * 0.9 ** np.arange(10)}
    fig = dsg.plot_irf(irf, columns=2)
    axes = [a for a in fig.axes if a.get_visible()]
    assert [a.get_title() for a in axes] == ["y", "c"]
    x, y = _lines(axes[0])[0]
    np.testing.assert_array_equal(x, np.arange(1, 11))
    np.testing.assert_array_equal(y, irf["y"])
    two = dsg.plot_irf({"low": irf, "high": {k: 2 * v for k, v in irf.items()}}, ["c"])
    ax = two.axes[0]
    assert len(ax.get_lines()) == 3 and ax.get_legend() is not None      # two models and zero
    paths = dsg.plot_paths({"y": np.arange(5.0)}, start=1990)
    np.testing.assert_array_equal(_lines(paths.axes[0])[0][0], np.arange(1990, 1995))

    dec = {"shocks": {"a": {"y": np.array([1.0, -1.0, 0.5])}, "b": {"y": np.array([-0.5, 2.0, 0.5])}},
           "initial": {"y": np.array([0.2, 0.0, -0.1])}}
    dec["total"] = {"y": dec["shocks"]["a"]["y"] + dec["shocks"]["b"]["y"] + dec["initial"]["y"]}
    fig = dsg.plot_decomposition(dec, "y")
    ax = fig.axes[0]
    tops = np.zeros(3)
    bottoms = np.zeros(3)
    for bar in ax.patches:
        t = int(round(bar.get_x() + bar.get_width() / 2)) - 1
        if bar.get_height() > 0:
            tops[t] = max(tops[t], bar.get_y() + bar.get_height())
        elif bar.get_height() < 0:
            bottoms[t] = min(bottoms[t], bar.get_y() + bar.get_height())
    np.testing.assert_allclose(tops + bottoms, dec["total"]["y"], atol=1e-12)
    np.testing.assert_array_equal(_lines(ax)[0][1], dec["total"]["y"])

    fc = dsg.Forecast(mean={"y": np.array([1.0, 1.5])}, std={"y": np.ones(2)},
                      lower={"y": np.array([0.0, 0.2])}, upper={"y": np.array([2.0, 2.8])},
                      horizon=2, coverage=0.9)
    fig = dsg.plot_forecast(fc, history={"y": np.array([0.4, 0.8, 0.9])})
    lines = _lines(fig.axes[0])
    np.testing.assert_array_equal(lines[0][0], [1, 2, 3])
    np.testing.assert_array_equal(lines[1][0], [4, 5])
    np.testing.assert_array_equal(lines[1][1], fc.mean["y"])
    assert len(fig.axes[0].collections) == 1                            # the band

    rng = np.random.default_rng(1)
    chain = SimpleNamespace(draws=np.column_stack([rng.normal(0.5, 0.05, 2000),
                                                   rng.gamma(4.0, 0.05, 2000)]),
                            names=["rho", "stderr e"])
    fig = dsg.plot_posterior(chain, priors={"rho": ("beta", 0.5, 0.2)}, mode={"rho": 0.5})
    rho, sd = fig.axes[0], fig.axes[1]
    assert len(rho.get_lines()) == 3 and len(sd.get_lines()) == 1       # prior, posterior, mode
    x, f = dsg.posterior_density(chain.draws[:, 0])
    np.testing.assert_array_equal(_lines(rho)[1][1], f)
    with pytest.raises(ValueError, match="names"):
        dsg.plot_posterior(chain.draws)
