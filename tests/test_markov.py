"""Markov chains and the discretisation of an AR(1): Tauchen and Rouwenhorst."""

import numpy as np
import pytest
from scipy.stats import binom, norm

import dsgelab as dsg


@pytest.mark.parametrize("rho", [-0.6, 0.0, 0.5, 0.95, 0.999])
@pytest.mark.parametrize("n", [2, 3, 9, 40])
def test_rouwenhorst_has_the_moments_of_the_ar1(rho, n):
    sigma, mean = 0.02, 1.5
    chain = dsg.rouwenhorst(rho, sigma, n, mean=mean)
    var = sigma ** 2 / (1 - rho ** 2)
    assert chain.mean == pytest.approx(mean, rel=1e-13)
    assert chain.variance == pytest.approx(var, rel=1e-11)
    assert chain.autocorrelation() == pytest.approx(rho, abs=1e-11)
    assert chain.autocorrelation(4) == pytest.approx(rho ** 4, abs=1e-11)
    np.testing.assert_allclose(chain.stationary, binom.pmf(np.arange(n), n - 1, 0.5), atol=1e-14)
    s = np.sqrt(var * (n - 1))
    np.testing.assert_allclose(chain.nodes, mean + np.linspace(-s, s, n), rtol=1e-14)


def test_tauchen_probabilities_are_normal_intervals():
    rho, sigma, n, n_std = 0.8, 0.1, 5, 2.5
    chain = dsg.tauchen(rho, sigma, n, mean=-0.4, n_std=n_std)
    s = sigma / np.sqrt(1 - rho ** 2)
    x = np.linspace(-n_std * s, n_std * s, n)
    np.testing.assert_allclose(chain.nodes, x - 0.4, rtol=1e-15)
    h = (x[1] - x[0]) / 2
    for i in range(n):
        m = rho * x[i]
        expected = [norm.cdf(x[0] + h, m, sigma)]
        expected += [norm.cdf(x[j] + h, m, sigma) - norm.cdf(x[j] - h, m, sigma) for j in range(1, n - 1)]
        expected += [norm.sf(x[-1] - h, m, sigma)]
        np.testing.assert_allclose(chain.transition[i], expected, rtol=1e-12, atol=1e-16)


def test_tauchen_approaches_the_ar1_as_nodes_are_added():
    rho, sigma = 0.9, 0.01
    var = sigma ** 2 / (1 - rho ** 2)
    errors = [abs(dsg.tauchen(rho, sigma, n, n_std=4).variance / var - 1) for n in (11, 51, 201)]
    assert errors[0] > errors[1] > errors[2]
    assert errors[2] < 2e-3
    assert dsg.tauchen(rho, sigma, 201, n_std=4).autocorrelation() == pytest.approx(rho, abs=2e-3)


def test_tauchen_keeps_tail_probabilities_that_one_minus_cdf_would_lose():
    chain = dsg.tauchen(0.99, 0.01, 2)
    assert 0.0 < chain.transition[0, 1] < 1e-90
    assert chain.stationary == pytest.approx([0.5, 0.5])


def test_stationary_distribution_of_a_two_state_chain():
    p, q = 0.3, 0.05
    chain = dsg.MarkovChain([0.0, 1.0], [[1 - p, p], [q, 1 - q]])
    np.testing.assert_allclose(chain.stationary, [q / (p + q), p / (p + q)], rtol=1e-15)
    assert chain.mean == pytest.approx(p / (p + q))
    assert chain.autocorrelation() == pytest.approx(1 - p - q)


def test_stationary_distribution_is_invariant():
    chain = dsg.tauchen(0.95, 0.02, 31)
    pi = chain.stationary
    np.testing.assert_allclose(pi @ chain.transition, pi, atol=1e-16)
    assert pi.sum() == pytest.approx(1.0, abs=1e-15)


def test_reducible_chain_is_refused():
    chain = dsg.MarkovChain([0.0, 1.0], np.eye(2))
    with pytest.raises(ValueError, match="reducible"):
        chain.stationary
    with pytest.raises(ValueError, match="reducible"):
        dsg.tauchen(0.999, 0.01, 2).stationary


@pytest.mark.parametrize("args, message", [
    ((1.0, 0.1, 5), "rho"),
    ((0.5, 0.0, 5), "sigma"),
    ((0.5, 0.1, 1), "n"),
    ((0.5, 0.1, 2.5), "n"),
])
def test_bad_processes_are_refused(args, message):
    for f in (dsg.tauchen, dsg.rouwenhorst):
        with pytest.raises(ValueError, match=message):
            f(*args)


def test_bad_chains_are_refused():
    with pytest.raises(ValueError, match="rows"):
        dsg.MarkovChain([0.0, 1.0], [[0.5, 0.4], [0.5, 0.5]])
    with pytest.raises(ValueError, match="2 x 2"):
        dsg.MarkovChain([0.0, 1.0], np.ones((3, 3)) / 3)


def test_simulation_has_the_chain_s_moments():
    chain = dsg.rouwenhorst(0.9, 0.05, 7)
    x = chain.simulate(200_000, seed=3)
    assert x.mean() == pytest.approx(chain.mean, abs=0.01)
    assert x.var() == pytest.approx(chain.variance, rel=0.03)
    assert np.corrcoef(x[1:], x[:-1])[0, 1] == pytest.approx(0.9, abs=0.01)
    idx = chain.simulate(200_000, seed=3, indices=True)
    np.testing.assert_array_equal(chain.nodes[idx], x)
    np.testing.assert_allclose(np.bincount(idx, minlength=7) / idx.size, chain.stationary, atol=0.01)


def test_simulation_starts_where_asked_and_repeats_with_the_seed():
    chain = dsg.tauchen(0.5, 1.0, 5)
    a = chain.simulate(50, start=4, seed=11, indices=True)
    assert a[0] == 4
    np.testing.assert_array_equal(a, chain.simulate(50, start=4, seed=11, indices=True))
    assert chain.simulate(0).size == 0
    with pytest.raises(ValueError, match="start"):
        chain.simulate(5, start=5)
