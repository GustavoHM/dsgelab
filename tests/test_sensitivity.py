"""Global sensitivity: stability mapping, Monte Carlo filtering and Sobol indices.

Against answers known in closed form. The comparison with Dynare's sensitivity toolbox and with
SALib is validation/compare_gsa.py.
"""

import math

import numpy as np
import pytest
from scipy import stats

import dsgelab as dsg
from dsgelab.priors import prior


def _fisher():
    """``phi pi_t = E pi_{t+1} + a_t``, ``a`` an AR(1): determinate iff ``phi > 1``, stable iff ``rho < 1``."""
    def eqs(nxt, cur, eps, p):
        return [nxt.a - p["rho"] * cur.a - eps.e, p["phi"] * cur.pi - nxt.pi - cur.a]
    model = dsg.Model(states=["a"], jumps=["pi"], shocks=["e"], equations=eqs, name="fisher")
    return model, {"a": 0.0, "pi": 0.0}


def test_distributions_have_their_moments_and_quantiles():
    u = np.array([0.1, 0.5, 0.9])
    t = dsg.Triangular(0.0, 1.0, 4.0)
    ref = stats.triang(c=0.25, loc=0.0, scale=4.0)
    np.testing.assert_allclose(t.ppf(u), ref.ppf(u), rtol=1e-12)
    assert t.mean == pytest.approx(ref.mean()) and t.sd == pytest.approx(ref.std())
    assert not t.symmetric and dsg.Triangular(0, 1, 2).symmetric
    np.testing.assert_allclose(dsg.Normal(1.0, 2.0).ppf(u), stats.norm(1, 2).ppf(u), rtol=1e-12)
    assert dsg.Uniform(2.0, 5.0).sd == pytest.approx(3 / math.sqrt(12))
    with pytest.raises(ValueError):
        dsg.Uniform(1.0, 1.0)
    # a prior's quantile function is the inverse of the distribution it samples from
    rng = np.random.default_rng(0)
    for spec in [("beta", 0.5, 0.2), ("gamma", 1.5, 0.5), ("inv_gamma", 0.4, 0.07),
                 ("normal", 0.0, 2.0), ("uniform", 0.2, 0.1), ("inv_gamma2", 0.5, 0.2)]:
        pr = prior(*spec)
        draws = pr.sample(rng, 40000)
        np.testing.assert_allclose(np.quantile(draws, u), pr.ppf(u),
                                   rtol=0.03, atol=0.01 * pr.sd)


def test_smirnov_test_is_dynares():
    rng = np.random.default_rng(1)
    x, y = rng.normal(size=80), rng.normal(0.4, 1.0, size=50)
    d, p = dsg.smirnov_test(x, y)
    assert d == pytest.approx(stats.ks_2samp(x, y).statistic, abs=1e-15)
    m = 80 * 50 / 130
    lam = (math.sqrt(m) + 0.12 + 0.11 / math.sqrt(m)) * d
    j = np.arange(1, 102)
    assert p == pytest.approx(2 * np.sum((-1.0) ** (j - 1) * np.exp(-2 * lam ** 2 * j ** 2)),
                              rel=1e-12)
    dl, pl = dsg.smirnov_test(x, y, "less")
    dg, _ = dsg.smirnov_test(x, y, "greater")
    assert max(dl, dg) == pytest.approx(d) and pl == pytest.approx(math.exp(-2 * lam ** 2 * (dl / d) ** 2))
    assert dsg.smirnov_test(x, x) == (0.0, 1.0)


def test_filtering_finds_the_parameter_that_drives_the_behaviour():
    rng = np.random.default_rng(2)
    X = rng.uniform(size=(2000, 3))
    f = dsg.monte_carlo_filtering(X, X[:, 0] > 0.6, ["a", "b", "c"])
    assert f.significant == ["a"]
    assert f.n_behaviour + f.n_nonbehaviour == 2000
    assert "a" in f.summary()
    # a behaviour that depends on a sum makes the parameters correlated inside the group
    g = dsg.monte_carlo_filtering(X, X[:, 0] + X[:, 1] > 1.2, ["a", "b", "c"])
    assert set(g.significant) == {"a", "b"}
    pairs = g.correlations("behaviour")
    assert [(a, b) for a, b, *_ in pairs] == [("a", "b")] and pairs[0][2] < 0
    r, pv = stats.pearsonr(X[g.behaviour, 0], X[g.behaviour, 1])
    assert pairs[0][2] == pytest.approx(r, rel=1e-12) and pairs[0][3] == pytest.approx(pv, rel=1e-6)
    # draws in neither group are left out
    h = dsg.monte_carlo_filtering(X, np.arange(100), nonbehaviour=np.arange(100, 300))
    assert h.n_behaviour == 100 and h.n_nonbehaviour == 200 and h.names[0] == "x0"
    with pytest.raises(ValueError, match="both groups"):
        dsg.monte_carlo_filtering(X, np.arange(10), nonbehaviour=np.arange(5, 20))


def test_stability_map_finds_the_determinacy_and_stability_regions():
    model, guess = _fisher()
    dist = {"phi": dsg.Uniform(0.5, 2.0), "rho": dsg.Uniform(0.3, 1.2)}
    st = dsg.stability_map(model, {"phi": 1.5, "rho": 0.5}, dist, draws=256, guess=guess,
                           cov={"e": 0.1})
    phi, rho = st.draws[:, 0], st.draws[:, 1]
    expected = np.where(phi > 1, np.where(rho < 1, "determinate", "explosive"),
                        np.where(rho < 1, "indeterminate", "?"))
    known = expected != "?"
    np.testing.assert_array_equal(st.status[known], expected[known])
    # an unstable root and a missing one at once: as many stable roots as states, Z11 singular
    assert set(st.status[~known]) <= {"indeterminate"}
    assert st.shares["determinate"] == pytest.approx(np.mean(expected == "determinate"))
    assert st.filtering("indeterminate").significant[0] == "phi"
    assert st.filtering("explosive").significant[0] == "rho"
    assert set(st.filtering("determinate").significant) == {"phi", "rho"}
    assert "unique stable solution" in st.summary()
    # given points are classified as they are, and stderr columns are left out of the filtering
    pts = dsg.stability_map(model, {"phi": 1.5, "rho": 0.5},
                            points=[[1.5, 0.5, 0.2], [0.8, 0.5, 0.2]],
                            names=["phi", "rho", "stderr e"], guess=guess)
    assert list(pts.status) == ["determinate", "indeterminate"]
    assert pts.filtering().names == ["phi", "rho"]
    # priors are distributions too, and the designs differ but cover the same region
    pr = dsg.stability_map(model, {"phi": 1.5, "rho": 0.5},
                           {"phi": ("gamma", 1.5, 0.5), "rho": ("beta", 0.5, 0.2)},
                           draws=64, sampling="latin", guess=guess)
    assert pr.draws.shape == (64, 2) and np.all(pr.draws > 0)
    with pytest.raises(ValueError, match="sampling"):
        dsg.stability_map(model, {}, dist, draws=8, sampling="grid", guess=guess)


def test_sobol_indices_of_known_functions():
    pi = math.pi
    ishigami = {f"x{i}": dsg.Uniform(-pi, pi) for i in (1, 2, 3)}

    def f(v):
        return (math.sin(v["x1"]) + 7.0 * math.sin(v["x2"]) ** 2
                + 0.1 * v["x3"] ** 4 * math.sin(v["x1"]))
    res = dsg.sobol(f, ishigami, n=2048, bootstrap=50)
    idx = res.indices()["result"]
    np.testing.assert_allclose(idx["S1"], [0.3139, 0.4424, 0.0], atol=0.02)
    np.testing.assert_allclose(idx["ST"], [0.5576, 0.4424, 0.2437], atol=0.02)
    assert np.all(idx["S1_conf"] > 0) and idx["n"] == 2048
    # additive: first-order and total indices coincide, in proportion to the variances
    lin = dsg.sobol(lambda v: {"y": 2 * v["a"] + v["b"]},
                    {"a": dsg.Normal(0, 1), "b": dsg.Normal(0, 1)}, n=1024, bootstrap=0)
    r = lin.indices()["y"]
    np.testing.assert_allclose(r["S1"], [0.8, 0.2], atol=0.02)
    np.testing.assert_allclose(r["ST"], [0.8, 0.2], atol=0.02)
    assert np.isnan(r["S1_conf"]).all()
    assert lin.A.shape == (1024, 2) and lin.values["y"]["AB"].shape == (1024, 2)
    # a base point with a missing output drops out
    gap = dsg.sobol(lambda v: math.nan if v["a"] > 2.5 else v["a"], {"a": dsg.Normal(0, 1),
                                                                     "b": dsg.Normal(0, 1)},
                    n=256, bootstrap=0)
    assert 240 < gap.indices()["result"]["n"] < 256
    with pytest.raises(ValueError, match="power of two"):
        dsg.sobol(f, ishigami, n=1000)


def test_solution_outputs_are_the_restrictions_values():
    model, guess = _fisher()
    run = dsg.solution_outputs(model, {"phi": 1.5, "rho": 0.5}, guess=guess, cov={"e": 0.1},
                               irf=[("a", "e", "1:2")], moments=[("a", "a", 0), ("a", "a", 1)])
    assert run.outputs == ["a(1), e", "a(2), e", "a, a", "a, a(1)"]
    out = run({"rho": 0.6, "phi": 1.5})
    np.testing.assert_allclose([out["a(1), e"], out["a(2), e"]], [0.1, 0.06], rtol=1e-12)
    assert out["a, a"] == pytest.approx(0.01 / (1 - 0.36), rel=1e-12)
    assert out["a, a(1)"] == pytest.approx(0.6, rel=1e-12)
    assert run({"stderr e": 0.2})["a(1), e"] == pytest.approx(0.2, rel=1e-12)
    assert all(math.isnan(v) for v in run({"phi": 0.8}).values())
    res = dsg.sobol(run, {"rho": dsg.Uniform(0.2, 0.8), "phi": dsg.Uniform(1.2, 2.0)},
                    n=64, bootstrap=0)
    # nothing but rho moves the shock process
    st = res.indices()["a, a(1)"]["ST"]
    assert st[1] == pytest.approx(0.0, abs=1e-12) and st[0] == pytest.approx(1.0, abs=0.05)
    assert "a(2), e" in res.summary()


def test_restriction_sample_filtering():
    model, guess = _fisher()
    sample = dsg.sample_restrictions(model, {"phi": 1.5, "rho": 0.5},
                                     {"phi": ("gamma", 1.5, 0.3), "rho": ("beta", 0.5, 0.2)},
                                     moments=[("a", "a", 1, (0.6, 1.0))], guess=guess,
                                     cov={"e": 0.1}, draws=400, seed=3)
    f = sample.filtering()
    assert f.significant == ["rho"]
    assert f.n_behaviour + f.n_nonbehaviour == int(sample.solved.sum())
    np.testing.assert_array_equal(f.behaviour, sample.solved & sample.holds)
