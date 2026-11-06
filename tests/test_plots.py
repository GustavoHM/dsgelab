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
