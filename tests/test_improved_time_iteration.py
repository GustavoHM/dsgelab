"""Improved time iteration and expectations iteration: the fixed point of time iteration."""

import numpy as np
import pytest

import dsgelab as dsg
from dsgelab.examples import rbc


def _closed_form_c(p, a, k):
    return (1.0 - p.alpha * p.beta) * a * k ** p.alpha


def _growth_with_expectation():
    """Stochastic growth with the Euler equation's expectation declared as the jump ``z``."""
    def equations(nxt, cur, eps, p):
        y1 = nxt.a * nxt.k ** p.alpha
        return [
            nxt.k - ((1.0 - p.delta) * cur.k + cur.a * cur.k ** p.alpha - cur.c),
            cur.c ** (-p.gamma) - cur.z,
            cur.z - p.beta * nxt.c ** (-p.gamma) * (p.alpha * y1 / nxt.k + 1.0 - p.delta),
            np.log(nxt.a) - p.rho * np.log(cur.a) - eps.eps_a,
        ]
    return dsg.Model(states=["k", "a"], jumps=["c", "z"], shocks=["eps_a"], equations=equations)


def _consumption_savings(beta=0.96, gamma=4.0, r=1.02):
    def equations(nxt, cur, eps, q):
        return [
            nxt.w - (np.exp(eps.y) + (cur.w - cur.c) * r),
            beta * (nxt.c / cur.c) ** (-gamma) * r - 1.0,
        ]
    return dsg.Model(states=["w"], jumps=["c"], shocks=["y"], equations=equations)


@pytest.mark.parametrize("kind", ["linear", "cubic"])
def test_improved_time_iteration_reaches_time_iterations_fixed_point_in_fewer_steps(kind):
    model, p, guess = rbc.build()
    kss, _ = rbc.steady_levels(p)
    kw = dict(grid={"k": (0.6 * kss, 1.4 * kss, 15), "a": (0.95, 1.05, 5)},
              cov={"eps_a": 0.007}, guess=guess, interpolation=kind, tol=1e-11)
    ti = dsg.time_iteration(model, p, **kw)
    it = dsg.improved_time_iteration(model, p, **kw)
    assert it.converged and isinstance(it, dsg.TimeIterationSolution)
    assert it.iterations <= 6 < ti.iterations
    assert np.max(np.abs(it.values - ti.values)) < 1e-9
    # Newton's convergence: each step's move at most about the square of the previous one
    h = it.history
    assert h[-1] < 1e-11 and h[-2] < 1e-5
    mids = it.grid.midpoints
    c = it.controls(mids)[:, 0]
    assert np.max(np.abs(c / _closed_form_c(p, mids[:, 1], mids[:, 0]) - 1)) < 1e-3


def test_improved_time_iteration_with_a_borrowing_limit_and_a_markov_chain():
    model = _consumption_savings()
    bounds = {"c": (0.0, lambda s, _: s.w)}
    kw = dict(grid={"w": (0.05, 4.0, 40)}, cov={"y": 0.1}, bounds=bounds, tol=1e-11,
              interpolation="cubic")
    ti = dsg.time_iteration(model, {}, initial={"c": lambda s, _: 0.9 * s.w}, **kw)
    it = dsg.improved_time_iteration(model, {}, initial={"c": lambda s, _: np.minimum(s.w, 1.0)},
                                     **kw)
    assert it.converged and it.iterations < 15 < ti.iterations
    assert np.max(np.abs(it.values - ti.values)) < 1e-9
    w = it.grid.points[:, 0]
    binding = np.isclose(it.values[0, :, 0], w, atol=1e-10)
    assert binding[0] and not binding[-1]

    # productivity as a Markov chain, with a SmolyakGrid along capital
    p = rbc.Params()

    def equations(nxt, cur, eps, q):
        return [
            nxt.k - cur.a * cur.k ** q.alpha + cur.c,
            1.0 / cur.c - q.beta * q.alpha * nxt.a * nxt.k ** (q.alpha - 1) / nxt.c,
            np.log(nxt.a) - q.rho * np.log(cur.a),
        ]

    growth = dsg.Model(states=["k", "a"], jumps=["c"], shocks=[], equations=equations)
    ch = dsg.rouwenhorst(p.rho, 0.007, 3)
    ch = dsg.MarkovChain(np.exp(ch.nodes), ch.transition)
    kss, _ = rbc.steady_levels(p)
    grid = dsg.SmolyakGrid({"k": (0.6 * kss, 1.4 * kss)}, 4)
    kw = dict(grid=grid, markov={"a": ch}, tol=1e-11,
              initial={"c": lambda s, q: 0.5 * s.a * s.k ** q.alpha})
    ti = dsg.time_iteration(growth, p, **kw)
    it = dsg.improved_time_iteration(growth, p, **kw)
    assert it.converged and it.iterations < ti.iterations
    assert np.max(np.abs(it.values - ti.values)) < 1e-9


@pytest.mark.parametrize("kind", ["linear", "cubic"])
def test_expectations_iteration_reaches_time_iterations_fixed_point(kind):
    _, p, guess = rbc.build()
    kss, _ = rbc.steady_levels(p)
    model = _growth_with_expectation()
    start = dict(guess) | {"z": 1.0 / guess["c"]}
    kw = dict(grid={"k": (0.6 * kss, 1.4 * kss, 15), "a": (0.95, 1.05, 5)},
              cov={"eps_a": 0.007}, guess=start, interpolation=kind, tol=1e-11)
    ti = dsg.time_iteration(model, p, **kw)
    ei = dsg.expectations_iteration(model, p, expectations=["z"], **kw)
    assert ei.converged
    assert np.max(np.abs(ei.values - ti.values)) < 1e-9
    mids = ei.grid.midpoints
    c, z = ei.controls(mids).T
    assert np.max(np.abs(c / _closed_form_c(p, mids[:, 1], mids[:, 0]) - 1)) < 1e-3
    assert np.allclose(z, 1.0 / c, rtol=1e-3)
    # without damping, the update of c overshoots by about (1 - alpha beta)/(alpha beta)
    with np.errstate(all="ignore"):
        plain = dsg.expectations_iteration(model, p, expectations=["z"], damping=1.0,
                                           max_iter=30, **kw)
    assert not plain.converged


def test_expectations_iteration_with_a_borrowing_limit():
    def equations(nxt, cur, eps, q):
        return [
            nxt.w - (np.exp(eps.y) + (cur.w - cur.c) * 1.02),
            cur.c - cur.z ** -0.25,
            cur.z - 0.96 * 1.02 * nxt.c ** (-4.0),
        ]

    model = dsg.Model(states=["w"], jumps=["c", "z"], shocks=["y"], equations=equations)
    kw = dict(grid={"w": (0.05, 4.0, 40)}, cov={"y": 0.1}, bounds={"c": (0.0, lambda s, _: s.w)},
              initial={"c": lambda s, _: 0.9 * s.w, "z": lambda s, _: (0.9 * s.w) ** -4.0},
              tol=1e-11)
    ei = dsg.expectations_iteration(model, {}, expectations=["z"], **kw)
    ref = dsg.time_iteration(_consumption_savings(), {}, grid=kw["grid"], cov=kw["cov"],
                             bounds=kw["bounds"], initial={"c": lambda s, _: 0.9 * s.w},
                             tol=1e-12)
    assert ei.converged
    assert np.max(np.abs(ei.values[0, :, 0] - ref.values[0, :, 0])) < 1e-8
    w = ei.grid.points[:, 0]
    assert np.isclose(ei.values[0, 0, 0], w[0], atol=1e-10)


def test_misdeclared_expectations_are_refused():
    _, p, guess = rbc.build()
    kss, _ = rbc.steady_levels(p)
    model = _growth_with_expectation()
    start = dict(guess) | {"z": 1.0 / guess["c"]}
    kw = dict(grid={"k": (0.6 * kss, 1.4 * kss, 5), "a": (0.95, 1.05, 3)},
              cov={"eps_a": 0.007}, guess=start)
    with pytest.raises(KeyError, match="not jumps"):
        dsg.expectations_iteration(model, p, expectations=["q"], **kw)
    with pytest.raises(ValueError, match="do not fit"):
        dsg.expectations_iteration(model, p, expectations=["c"], **kw)
    with pytest.raises(ValueError, match="some jump must be a control"):
        dsg.expectations_iteration(model, p, expectations=["c", "z"], **kw)
    with pytest.raises(ValueError, match="not to the expectations"):
        dsg.expectations_iteration(model, p, expectations=["z"], bounds={"z": (0, None)}, **kw)
    with pytest.raises(ValueError, match="damping"):
        dsg.expectations_iteration(model, p, expectations=["z"], damping=0.0, **kw)
    with pytest.raises(ValueError, match="max_iter"):
        dsg.improved_time_iteration(model, p, max_iter=0, **kw)
