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


# ----------------------------------------------------------------- the statistics
def test_brooks_gelman_follows_its_definition():
    from dsgelab.plots import _round
    rng = np.random.default_rng(2)
    chains = rng.normal(size=(3, 1300, 2)).cumsum(axis=1) * 0.01 + rng.normal(size=(3, 1300, 2))
    lp = rng.normal(size=(3, 1300))
    bg = dsg.brooks_gelman(chains, names=["a", "b"], log_posterior=lp)
    np.testing.assert_array_equal(bg.iterations, np.arange(1000, 1301, 3))
    assert bg.univariate.shape == (101, 6, 2) and bg.multivariate.shape == (101, 6)
    it = 1150
    w = list(bg.iterations).index(it)
    first = math.ceil(0.5 * it)
    W = chains[:, first - 1:it, 1]
    n = it - first + 1
    pooled = np.sort(W.ravel())
    lo, hi = max(1, _round(3 * n * 0.1)), _round(3 * n * 0.9)
    assert bg.univariate[w, 0, 1] == pytest.approx(pooled[hi - 1] - pooled[lo - 1], rel=1e-14)
    assert bg.univariate[w, 2, 1] == pytest.approx(W.var(ddof=1), rel=1e-12)
    assert bg.univariate[w, 3, 1] == pytest.approx(W.var(axis=1, ddof=1).mean(), rel=1e-12)
    m3 = np.mean([np.sum(np.abs(c - c.mean()) ** 3) / (n - 1) for c in W])
    assert bg.univariate[w, 5, 1] == pytest.approx(m3, rel=1e-12)
    assert bg.multivariate[w, 2] == pytest.approx(lp[:, first - 1:it].var(ddof=1), rel=1e-12)
    # an MCMCResult brings its own log posterior; too few draws for the first window is an error
    res = SimpleNamespace(chains=chains, names=["a", "b"], log_posterior=lp)
    np.testing.assert_array_equal(dsg.brooks_gelman(res).multivariate, bg.multivariate)
    with pytest.raises(ValueError, match="origin"):
        dsg.brooks_gelman(chains[:, :900])


def test_mh_autocorrelation_and_moving_average():
    from dsgelab.plots import _moving_average
    x = np.array([1.0, 3.0, 2.0, 5.0, 4.0, 6.0])
    d = x - x.mean()
    acf = dsg.mh_autocorrelation(x, 2)
    cov = [d[2:] @ d[2:], d[2:] @ d[1:5], d[2:] @ d[:4]]
    np.testing.assert_allclose(acf, np.array(cov) / cov[0], rtol=1e-14)
    rng = np.random.default_rng(3)
    ar = np.zeros(20000)
    for t in range(1, ar.size):
        ar[t] = 0.7 * ar[t - 1] + rng.normal()
    np.testing.assert_allclose(dsg.mh_autocorrelation(ar, 3), 0.7 ** np.arange(4), atol=0.02)
    ma = _moving_average(np.arange(10.0), 2)
    assert np.isnan(ma[:2]).all() and np.isnan(ma[-2:]).all()
    np.testing.assert_allclose(ma[2:-2], np.arange(2.0, 8.0))
    with pytest.raises(ValueError, match="lags"):
        dsg.mh_autocorrelation(x, 6)


def test_posterior_moments_are_dynares():
    from dsgelab.plots import _posterior_moments
    rng = np.random.default_rng(4)
    X = rng.gamma(2.0, size=(1001, 3))
    st = _posterior_moments(X, 0.9)
    for j in range(3):
        x = np.sort(X[:, j])
        h = 100                                    # round(0.1 * 1001)
        widths = [x[1001 - h - 1 + i] - x[i] for i in range(h)]
        i = int(np.argmin(widths))
        assert st["lower"][j] == x[i] and st["upper"][j] == pytest.approx(x[i + 1001 - h - 1])
        # the interval holds all but 100 draws, and no other such interval is shorter
        assert np.sum((x >= st["lower"][j]) & (x <= st["upper"][j])) == 1001 - h
        np.testing.assert_array_equal(st["deciles"][j],
                                      x[[99, 199, 299, 399, 500, 600, 700, 800, 900]])
        assert st["median"][j] == np.median(x) and st["mean"][j] == pytest.approx(x.mean())
    assert np.isnan(_posterior_moments(X[:20], 0.9)["lower"]).all()


@pytest.fixture(scope="module")
def ar_problem():
    ar = dsg.from_equations(["a(+1) = rho*a + e", "y = 1 + a"], states=["a"], jumps=["y"],
                            shocks=["e"], parameters=["rho"])
    sol = dsg.solve(ar, dict(rho=0.8), guess=dict(a=0, y=1))
    series = sol.to_levels(sol.simulate(periods=200, cov={"e": 0.1}, seed=1))["y"]
    return dsg.EstimationProblem(
        model=ar, data={"y": series}, params=dict(rho=0.5), observables=["y"],
        priors={"rho": ("beta", 0.5, 0.2), "stderr e": ("inv_gamma", 0.1, float("inf"))},
        guess=dict(a=0, y=1))


def test_mode_check_grid_and_kernels(ar_problem):
    pb = ar_problem
    mode = np.array([0.8, 0.1])
    mc = dsg.mode_check(pb, dict(zip(pb.names, mode)))
    rho, sd = mc["rho"], mc["stderr e"]
    # a beta prior on (0, 1): 0.4 to 1.2 is cut at 1, and the grid made symmetric around the mode
    np.testing.assert_allclose(rho["x"], np.linspace(0.6, 1.0, 21), atol=1e-15)
    assert rho["mode"] == 0.8 and np.isnan(rho["log_posterior"][-1])
    # an infinite upper bound takes Dynare's other rule: from 0 to twice the mode
    np.testing.assert_allclose(sd["x"], np.linspace(0.0, 0.2, 21), atol=1e-15)
    assert np.isnan(sd["log_posterior"][0])
    i = 10
    assert rho["x"][i] == pytest.approx(0.8)
    assert rho["log_posterior"][i] == pytest.approx(pb.log_posterior(mode))
    assert rho["log_likelihood"][i] == pytest.approx(rho["log_posterior"][i])
    theta = np.array([rho["x"][3], 0.1])
    assert rho["log_likelihood"][3] == pytest.approx(pb.log_likelihood(theta) + pb.log_prior(mode))
    asym = dsg.mode_check(pb, mode, symmetric=False, points=10)["rho"]["x"]
    np.testing.assert_allclose(asym, np.r_[np.linspace(0.4, 0.8, 6), np.linspace(0.84, 1.0, 5)])


def test_posterior_irf_statistics(ar_problem):
    pb = ar_problem
    draws = np.column_stack([np.linspace(0.7, 0.9, 30), np.linspace(0.08, 0.12, 30)])
    pi = dsg.posterior_irf(pb, draws, variables=["y", "a"], periods=8, n_draws=None)
    assert pi["draws"] == 30 and set(pi["mean"]) == {"e"}
    R = pi["responses"]["e"]["y"]
    assert R.shape == (30, 8)
    sol, space = pb.solve(draws[4])
    one = sol.irf_impulse([math.sqrt(space.Sigma[0, 0] + 1e-14)], 8)["y"]
    np.testing.assert_allclose(R[4], one, rtol=1e-12)
    np.testing.assert_allclose(pi["mean"]["e"]["y"], R.mean(axis=0), rtol=1e-12)
    assert np.all(pi["lower"]["e"]["y"] <= pi["median"]["e"]["y"])
    assert np.all(pi["median"]["e"]["y"] <= pi["upper"]["e"]["y"])
    assert pi["deciles"]["e"]["y"].shape == (8, 9)
    some = dsg.posterior_irf(pb, draws, periods=8, n_draws=10, seed=1)
    assert some["draws"] == 10 and list(some["mean"]["e"]) == ["y"]   # the observables


# ------------------------------------------------------------------- the figures
def _panels(fig):
    return [a for a in fig.axes if a.get_visible() and a.get_label() != "<colorbar>"]


def test_mcmc_figures_draw_their_statistics():
    pytest.importorskip("matplotlib")
    rng = np.random.default_rng(5)
    chains = rng.normal(size=(2, 1500, 2)).cumsum(axis=1) * 0.01 + rng.normal(size=(2, 1500, 2))
    res = SimpleNamespace(chains=chains, names=["a", "b"], log_posterior=rng.normal(size=(2, 1500)))
    fig = dsg.trace_plot(res, posterior=True, window=50)
    axes = _panels(fig)
    assert [a.get_title() for a in axes] == ["log posterior", "a", "b"]
    (t, x), (_, ma) = _lines(axes[1])
    np.testing.assert_array_equal(x, chains[0, :, 0])
    assert np.isnan(ma[:50]).all() and ma[50] == pytest.approx(chains[0, :101, 0].mean())
    assert len(_panels(dsg.trace_plot(res, ["b"], chains=[0, 1]))[0].get_lines()) == 2

    fig = dsg.plot_mh_autocorrelation(res, ["b"], lags=10)
    heights = [p.get_height() for p in _panels(fig)[0].patches]
    np.testing.assert_allclose(heights, dsg.mh_autocorrelation(chains[0, :, 1], 10))

    bg = dsg.brooks_gelman(res)
    fig = dsg.plot_brooks_gelman(bg)
    axes = _panels(fig)
    assert len(axes) == 6 and axes[4].get_title() == "b (m2)"
    np.testing.assert_array_equal(_lines(axes[4])[0][1], bg.univariate[:, 2, 1])
    np.testing.assert_array_equal(_lines(axes[4])[1][1], bg.univariate[:, 3, 1])
    multi = _panels(dsg.plot_brooks_gelman(res, multivariate=True))
    assert [a.get_title() for a in multi] == ["Interval", "m2", "m3"]
    np.testing.assert_array_equal(_lines(multi[2])[0][1], bg.multivariate[:, 4])
    with pytest.raises(ValueError, match="log posterior"):
        dsg.plot_brooks_gelman(dsg.brooks_gelman(chains), multivariate=True)


def test_estimation_figures(ar_problem):
    pytest.importorskip("matplotlib")
    pb = ar_problem
    axes = _panels(dsg.plot_priors(pb))
    assert [a.get_title() for a in axes] == ["rho", "stderr e"]
    np.testing.assert_array_equal(_lines(axes[0])[0][1], dsg.prior_density(pb.priors["rho"])[1])

    mc = dsg.mode_check(pb, [0.8, 0.1], points=6)
    axes = _panels(dsg.plot_mode_check(mc))
    (x, post), (_, lik), (xm, _), (xbad, _) = _lines(axes[0])
    np.testing.assert_array_equal(post, mc["rho"]["log_posterior"])
    np.testing.assert_array_equal(lik, mc["rho"]["log_likelihood"])
    np.testing.assert_array_equal(xm, [0.8, 0.8])
    np.testing.assert_array_equal(xbad, [1.0])                          # rho = 1 cannot be solved

    sm = pb.smoother(np.array([0.8, 0.1]))
    axes = _panels(dsg.plot_smoothed_shocks(sm, start=0))
    assert [a.get_title() for a in axes] == ["e"]
    np.testing.assert_array_equal(_lines(axes[0])[1][1], sm.shocks[:, 0])
    np.testing.assert_array_equal(_lines(axes[0])[1][0], np.arange(sm.shocks.shape[0]))
    axes = _panels(dsg.plot_smoothed_observables(sm, pb.Y))
    (_, fitted), (_, observed) = _lines(axes[0])
    np.testing.assert_allclose(fitted, pb.Y[:, 0], atol=1e-10)         # no measurement error
    np.testing.assert_array_equal(observed, pb.Y[:, 0])
    with pytest.raises(ValueError, match="measurement"):
        dsg.plot_smoothed_shocks(sm, measurement=True)

    draws = np.column_stack([np.linspace(0.7, 0.9, 40), np.full(40, 0.1)])   # 4 left out
    pi = dsg.posterior_irf(pb, draws, variables=["y", "a"], periods=6, n_draws=None)
    fig = dsg.plot_posterior_irf(pi, "e")
    axes = _panels(fig)
    assert [a.get_title() for a in axes] == ["y", "a"] and "e" in fig._suptitle.get_text()
    np.testing.assert_array_equal(_lines(axes[0])[0][1], pi["mean"]["e"]["y"])
    band = axes[0].collections[0].get_paths()[0].vertices
    assert band[:, 1].min() == pytest.approx(pi["lower"]["e"]["y"].min())
    assert band[:, 1].max() == pytest.approx(pi["upper"]["e"]["y"].max())
    with pytest.raises(ValueError, match="threshold"):
        dsg.plot_posterior_irf(pi, "e", threshold=1e3)


def test_decomposition_options():
    pytest.importorskip("matplotlib")
    dec = {"shocks": {"a": {"y": np.array([1.0, -1.0, 0.5])}, "b": {"y": np.array([-0.5, 2.0, 0.5])},
                      "c": {"y": np.array([0.25, 0.25, 0.0])}},
           "initial": {"y": np.array([0.2, 0.0, -0.1])}}
    dec["total"] = {"y": sum(d["y"] for d in dec["shocks"].values()) + dec["initial"]["y"]}
    fig = dsg.plot_decomposition(dec, "y", groups={"demand": ["a", "b"]}, steady_state=2.0)
    ax = fig.axes[0]
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels == ["total", "demand", "Others + Initial Values"]
    first = [b for b in ax.patches if b.get_x() < 1.5 and b.get_height() != 0]
    np.testing.assert_allclose(sorted(b.get_height() for b in first), [0.45, 0.5])
    assert ax.yaxis.get_major_formatter()(1.0, 0) == "3"
    assert ax.yaxis.get_major_formatter()(-2.0, 0) == "0"
    # without shocks outside the groups, the last contribution is the initial condition
    fig = dsg.plot_decomposition(dec, "y", groups={"all": ["a", "b", "c"]})
    assert [t.get_text() for t in fig.axes[0].get_legend().get_texts()] == ["total", "all", "initial"]
    fig = dsg.plot_decomposition(dec, "y", detail=True)
    axes = _panels(fig)
    assert [a.get_title() for a in axes] == ["a", "b", "c", "initial"]
    assert len({a.get_ylim() for a in axes}) == 1
    with pytest.raises(ValueError, match="detail"):
        dsg.plot_decomposition(dec, "y", detail=True, ax=axes[0])


def test_forecast_occbin_identification_and_filtering_figures():
    pytest.importorskip("matplotlib")
    cf = {"y": np.array([1.0, 1.2, 1.1, 1.05]), "eps": np.zeros((4, 1))}
    fig = dsg.plot_conditional_forecast(cf, unconditional={"y": np.array([1.0, 0.9, 0.85, 0.8])},
                                        conditions={"y": [1.2, 1.1, np.nan]})
    (t, y), (_, u), (tc, c) = _lines(_panels(fig)[0])
    np.testing.assert_array_equal(t, [0, 1, 2, 3])
    np.testing.assert_array_equal(tc, [1, 2, 3])
    np.testing.assert_array_equal(u, [1.0, 0.9, 0.85, 0.8])

    ob = SimpleNamespace(piecewise={"c": np.array([1.0, 0.9, 0.95])},
                         linear={"c": np.array([1.0, 0.8, 0.9])}, steady={"c": 1.0},
                         binding={"zlb": np.array([False, True, False])},
                         regimes=[(1, {"zlb": np.array([False, True, True, False])})])
    axes = _panels(dsg.plot_occbin(ob, deviations=True))
    assert [a.get_title() for a in axes] == ["c", "zlb binds"]
    np.testing.assert_allclose(_lines(axes[0])[1][1], [0.0, -0.1, -0.05])
    np.testing.assert_allclose(_lines(axes[0])[2][1], [0.0, -0.2, -0.1])
    np.testing.assert_array_equal(_lines(axes[1])[0][1], [0, 1, 0])
    axes = _panels(dsg.plot_occbin_regimes(ob))
    marks = [line for line in _lines(axes[0]) if line[0].size == 1]
    np.testing.assert_array_equal([m[1][0] for m in marks], [1, 2, 4])  # slack, binding, slack

    ident = SimpleNamespace(names=["p", "q", "r"],
                            collinearity={1: {"p": (("q",), 0.9), "q": (("p",), 0.9),
                                              "r": (("p",), 0.2)}})
    im = _panels(dsg.plot_identification(ident))[0].images[0].get_array()
    assert im[0, 1] == 0.9 and im[2, 0] == 0.2 and im.mask[0, 0] and im.mask[2, 1]

    rng = np.random.default_rng(6)
    pts = rng.uniform(size=(300, 2))
    fr = dsg.monte_carlo_filtering(pts, pts[:, 0] > 0.4, ["u", "v"])
    axes = _panels(dsg.plot_filtering(fr))
    assert len(axes) == 1 and axes[0].get_title() == "u"               # only u is significant
    _, F = _lines(axes[0])[0]
    assert F[0] == 0 and F[-1] == 1 and np.all(np.diff(F) >= 0)
    axes = _panels(dsg.plot_filtering(fr, ["u", "v"], scatter=False))
    assert axes[1].get_title().startswith("v. p-value")
    axes = _panels(dsg.plot_filtering(fr, ["u", "v"]))
    assert len(axes) == 4
    np.testing.assert_array_equal(_lines(axes[1])[0][0], pts[fr.behaviour, 1])


def test_posterior_limits_and_transition_paths():
    pytest.importorskip("matplotlib")
    rng = np.random.default_rng(7)
    chain = SimpleNamespace(draws=rng.normal(0.5, 0.05, (3000, 1)), names=["rho"])
    ax = dsg.plot_posterior(chain, priors={"rho": ("beta", 0.5, 0.2)}).axes[0]
    xp, fp = dsg.prior_density(("beta", 0.5, 0.2))
    x, f = dsg.posterior_density(chain.draws[:, 0])
    assert ax.get_xlim() == pytest.approx((min(xp[0], x[0]), max(xp[-1], x[-1])))
    assert ax.get_ylim() == pytest.approx((0, 1.1 * max(np.nanmax(fp), f.max())))
    tp = SimpleNamespace(levels=np.arange(6.0).reshape(3, 2), model=None,
                         values=lambda: {"k": np.array([1.0, 2.0, 3.0]), "c": np.zeros(3)})
    fig = dsg.plot_paths(tp, ["k"], start=0)
    np.testing.assert_array_equal(_lines(fig.axes[0])[0][0], [0, 1, 2])


def test_identification_strength_figures():
    pytest.importorskip("matplotlib")
    V = np.linalg.qr(np.random.default_rng(2).standard_normal((3, 3)))[0]
    strength = SimpleNamespace(strength=np.array([2.0, 0.5, np.inf]),
                               strength_prior=np.array([1.0, np.nan, 0.0]),
                               sensitivity=np.array([3.0, 1.0, 2.0]),
                               sensitivity_prior=np.array([np.e, 1.0, 4.0]),
                               moments_sensitivity=np.array([1.0, 4.0, np.inf]),
                               singular_values=np.array([9.0, 4.0, 1.0]), patterns=V)
    ident = SimpleNamespace(names=["p", "q", "r"], strength=strength)
    upper, lower = _panels(dsg.plot_identification_strength(ident))
    assert [t.get_text() for t in upper.get_xticklabels()] == ["q", "p", "r"]   # by strength
    heights = sorted(round(b.get_height(), 12) for b in upper.patches)
    np.testing.assert_allclose(heights, [math.log(0.5), 0.0, math.log(2.0)])
    # the infinite strength of r and the zero strength relative to the prior are marked at zero
    marks = sorted((line.get_xdata()[0], line.get_markerfacecolor()) for line in upper.lines
                   if line.get_markerfacecolor() in ("red", "white"))
    assert marks == [(pytest.approx(2.8), "white"), (pytest.approx(3.2), "red")]
    np.testing.assert_allclose(sorted(b.get_height() for b in lower.patches),
                               sorted(np.log([1.0, 3.0, 2.0, 1.0, np.e, 4.0])))
    (ax,) = _panels(dsg.plot_identification_sensitivity(ident))
    np.testing.assert_allclose(sorted(b.get_height() for b in ax.patches),
                               sorted([math.log(100.0), math.log(25.0)]))
    axes = _panels(dsg.plot_identification_patterns(ident, count=2))
    np.testing.assert_allclose([b.get_height() for b in axes[0].patches], np.abs(V[:, 2]))
    assert axes[1].get_title() == "Singular value 4"
    axes = _panels(dsg.plot_identification_patterns(ident, largest=True, count=1))
    np.testing.assert_allclose([b.get_height() for b in axes[0].patches], np.abs(V[:, 0]))
    with pytest.raises(ValueError, match="strength=True"):
        dsg.plot_identification_strength(SimpleNamespace(names=["p"], strength=None))


def test_realtime_annualised_and_initial_condition_decompositions(ar_problem):
    pb = ar_problem
    theta = pb.start()
    sol, space = pb.solve(theta)
    rt = dsg.realtime_shock_decomposition(space, pb.Y, variables=["a", "y"], forecast=2,
                                          vintages=[50])
    n = pb.Y.shape[0]
    assert rt["total"]["y"].size == n + 2 and rt["forecast"]["total"]["y"].size == n + 1
    for key in ("conditional", "forecast"):
        d = rt[key]
        np.testing.assert_allclose(d["shocks"]["e"]["a"] + d["initial"]["a"], d["total"]["a"],
                                   atol=1e-14)
    fc, vint, cond = (rt[k][50] for k in ("forecast_vintages", "vintages",
                                          "conditional_vintages"))
    assert fc["total"]["a"].size == 3 and vint["total"]["a"].size == 52
    # the forecast starts from the cut's last date and goes on without shocks
    assert fc["total"]["a"][0] == vint["total"]["a"][49]
    np.testing.assert_allclose(fc["total"]["a"][1:], vint["total"]["a"][50:], atol=1e-14)
    np.testing.assert_allclose(fc["total"]["a"][1], 0.5 * fc["total"]["a"][0])   # rho = 0.5
    assert rt["forecast"]["total"]["a"][50] == fc["total"]["a"][1]
    # what the cut did not forecast: the pool's total, less the forecast's contributions
    np.testing.assert_allclose(cond["total"]["a"], rt["total"]["a"][49:52])
    np.testing.assert_allclose(cond["shocks"]["e"]["a"],
                               rt["shocks"]["e"]["a"][49:52] - fc["shocks"]["e"]["a"])
    # the one-step conditional pool is the impact of each date's innovation
    sm = dsg.kalman_smoother(space, pb.Y[:40])
    assert rt["conditional"]["shocks"]["e"]["a"][39] == pytest.approx(sm.shocks[39, 0])
    pytest.importorskip("matplotlib")
    ax = dsg.plot_decomposition(rt, "a", realtime=3, vintage=50).axes[0]
    assert ax.get_title() == "a 2-step ahead forecast (given 50)"
    assert min(b.get_x() + b.get_width() / 2 for b in ax.patches) == pytest.approx(50)
    ax = dsg.plot_decomposition(rt, "a", realtime=2).axes[0]
    assert ax.get_title() == "a 1-step ahead conditional forecast (rolling)"
    with pytest.raises(ValueError, match="vintage 7"):
        dsg.plot_decomposition(rt, "a", realtime=1, vintage=7)

    dec = {"shocks": {"e": {"y": np.arange(8.0)}}, "initial": {"y": np.ones(8)},
           "total": {"y": np.arange(8.0) + 1}}
    yoy = dsg.annualise_decomposition(dec, "yoy")
    np.testing.assert_array_equal(yoy["shocks"]["e"]["y"], [6, 10, 14, 18, 22])
    np.testing.assert_array_equal(yoy["initial"]["y"], [4] * 5)
    aoa = dsg.annualise_decomposition(dec, "aoa", quarter=3)
    np.testing.assert_array_equal(aoa["total"]["y"], [2, 6])      # the fourth quarters
    ax = dsg.plot_decomposition(dec, "y", frequency="yoy", steady_state=1.0).axes[0]
    assert min(b.get_x() + b.get_width() / 2 for b in ax.patches) == pytest.approx(4)
    assert ax.yaxis.get_major_formatter()(0.0, 0) == "4"
    with pytest.raises(ValueError, match="kind"):
        dsg.annualise_decomposition(dec, "monthly")

    sm = pb.smoother(theta)
    ic = dsg.initial_condition_decomposition(sm, variables=["a", "y"])
    total = sum(ic["initial"][c]["a"] for c in ic["initial"]) + ic["shocks"]["a"]
    np.testing.assert_allclose(total, ic["total"]["a"], atol=1e-14)
    a0 = ic["total"]["a"][0]
    np.testing.assert_allclose(ic["initial"]["a"]["a"][:4], a0 * 0.5 ** np.arange(4))
    own = ic["initial"]["y"]["y"]
    assert own[0] == ic["total"]["y"][0] and not own[1:].any()
    assert ic["shocks"]["a"][0] == 0


def test_conditional_forecast_bands():
    two = dsg.from_equations(["a(+1) = 0.8*a + e", "b(+1) = 0.5*b + u", "y = a + b"],
                             states=["a", "b"], jumps=["y"], shocks=["e", "u"], parameters=[])
    sol = dsg.solve(two, {}, guess=dict(a=0, b=0, y=0))
    path = {"y": np.array([0.3, 0.2, np.nan, 0.1])}
    cf = dsg.conditional_forecast_bands(sol, path, controlled=["e"], periods=6,
                                        cov={"e": 0.1, "u": 0.2}, replic=500, seed=4)
    assert cf.replic == 500 and cf.coverage == 0.8
    for t in (1, 2, 4):        # the conditions hold in every draw, so the bands close on them
        assert cf.lower["y"][t] == pytest.approx(path["y"][t - 1])
        assert cf.upper["y"][t] == pytest.approx(path["y"][t - 1])
    assert cf.upper["y"][3] > cf.lower["y"][3] + 0.1        # date 3 is free
    assert cf.shocks["e"]["mean"].size == 4
    # with the drawn shocks at zero, the mean is the mean conditional forecast
    zeros = np.zeros((3, 6, 2))
    flat = dsg.conditional_forecast_bands(sol, {"y": np.array([0.3, 0.2, 0.1])}, controlled=["e"],
                                          periods=6, cov={"e": 0.1, "u": 0.2}, normals=zeros)
    ref = sol.conditional_forecast({"y": np.r_[[0.3, 0.2, 0.1], [np.nan] * 3]}, controlled=["e"])
    for v in ("a", "b", "y"):
        np.testing.assert_allclose(flat.mean[v], ref[v], atol=1e-14)
        np.testing.assert_allclose(flat.unconditional_mean[v], 0.0, atol=1e-14)
    # the bands are the draws ranked round(n 0.1) and round(n 0.9), as Dynare ranks them
    rng = np.random.default_rng(0)
    Z = rng.standard_normal((20, 6, 2))
    cf = dsg.conditional_forecast_bands(sol, {"y": np.array([0.3])}, controlled=["e"], periods=6,
                                        cov={"e": 0.1, "u": 0.2}, normals=(Z, Z))
    b = np.zeros((20, 7))
    for t in range(6):
        b[:, t + 1] = 0.5 * b[:, t] + 0.2 * Z[:, t, 1]
    np.testing.assert_allclose(cf.unconditional_lower["b"], np.sort(b, axis=0)[1], atol=1e-14)
    np.testing.assert_allclose(cf.unconditional_upper["b"], np.sort(b, axis=0)[17], atol=1e-14)
    with pytest.raises(ValueError, match="normals"):
        dsg.conditional_forecast_bands(sol, path, controlled=["e"], normals=zeros)
    pytest.importorskip("matplotlib")
    ax = _panels(dsg.plot_conditional_forecast(cf, ["y"], conditions={"y": [0.3]}))[0]
    lines = _lines(ax)
    np.testing.assert_allclose(lines[0][1], cf.mean["y"])
    np.testing.assert_allclose(lines[2][1], cf.unconditional_lower["y"])
    assert len(ax.collections) == 1                          # the band


def test_posterior_conditional_forecast(ar_problem):
    pb = ar_problem
    theta = pb.start()
    sol, space = pb.solve(theta)
    sm = pb.smoother(theta)
    k0 = sm.states[-1, :space.n_model_states]
    cf = dsg.posterior_conditional_forecast(pb, np.tile(theta, (3, 1)), {"y": [1.1, 1.05]},
                                            periods=4, n_draws=3, replic=2)
    assert cf.replic == 6
    ref = sol.conditional_forecast({"y": np.r_[[1.1, 1.05], [np.nan] * 2]}, initial=k0,
                                   levels=True)
    # one shock, controlled: every path is the mean conditional forecast
    np.testing.assert_allclose(cf.mean["y"], ref["y"], atol=1e-12)
    np.testing.assert_allclose(cf.lower["a"], cf.upper["a"], atol=1e-12)
    # without conditions the controlled shock is zero: the path from the smoothed state
    np.testing.assert_allclose(cf.unconditional_lower["a"], k0[0] * 0.5 ** np.arange(5))
