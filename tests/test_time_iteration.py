"""Time iteration: the grid's interpolation, closed forms, Markov chains and complementarity."""

import math

import numpy as np
import pytest
from scipy.interpolate import CubicSpline

import dsgelab as dsg
from dsgelab.examples import rbc


# -- the grid -------------------------------------------------------------------------------------

def test_cubic_interpolation_is_scipys_natural_spline_extended_linearly():
    x = np.array([0.0, 0.3, 0.7, 1.5, 2.0, 3.1])
    f = np.sin(x) + x ** 2
    grid = dsg.CartesianGrid({"x": x})
    s = np.linspace(-0.5, 3.6, 57)
    ours = grid.interpolate(f[:, None], s[:, None], kind="cubic")[:, 0]
    spline = CubicSpline(x, f, bc_type="natural")
    inside = (s >= x[0]) & (s <= x[-1])
    assert np.allclose(ours[inside], spline(s[inside]), rtol=0, atol=1e-13)
    low, high = s < x[0], s > x[-1]
    assert np.allclose(ours[low], f[0] + spline(x[0], 1) * (s[low] - x[0]), atol=1e-13)
    assert np.allclose(ours[high], f[-1] + spline(x[-1], 1) * (s[high] - x[-1]), atol=1e-13)


def test_tensor_interpolation_applies_the_spline_along_each_axis():
    x, y = np.linspace(0.0, 1.0, 5), np.array([-1.0, -0.2, 0.5, 2.0])
    grid = dsg.CartesianGrid({"x": x, "y": y})
    F = np.exp(x)[:, None] * np.cos(y)[None, :] + x[:, None] * y[None, :] ** 2
    rng = np.random.default_rng(0)
    pts = np.column_stack([rng.uniform(-0.2, 1.2, 40), rng.uniform(-1.5, 2.5, 40)])
    ours = grid.interpolate(F.reshape(-1, 1), pts, kind="cubic")[:, 0]

    def along(nodes, vals, s):
        sp = CubicSpline(nodes, vals, bc_type="natural")
        return np.where(s < nodes[0], vals[0] + sp(nodes[0], 1) * (s - nodes[0]),
                        np.where(s > nodes[-1], vals[-1] + sp(nodes[-1], 1) * (s - nodes[-1]),
                                 sp(s)))

    ref = [along(y, np.array([along(x, F[:, j], p[0]) for j in range(y.size)]), p[1])
           for p in pts]
    assert np.allclose(ours, ref, rtol=0, atol=1e-12)
    # linear: exact for a bilinear function, inside the grid and beyond it
    G = 1.0 + 2.0 * x[:, None] - y[None, :] + 0.5 * x[:, None] * y[None, :]
    lin = grid.interpolate(G.reshape(-1, 1), pts, kind="linear")[:, 0]
    assert np.allclose(lin, 1.0 + 2.0 * pts[:, 0] - pts[:, 1] + 0.5 * pts[:, 0] * pts[:, 1],
                       atol=1e-13)


def test_grid_from_bounds_and_its_points():
    grid = dsg.CartesianGrid({"k": (1.0, 2.0, 3), "a": [0.9, 1.1]})
    assert grid.shape == (3, 2) and grid.n == 6
    assert np.allclose(grid.points[:2], [[1.0, 0.9], [1.0, 1.1]])
    assert np.allclose(grid.midpoints, [[1.25, 1.0], [1.75, 1.0]])
    with pytest.raises(ValueError):
        dsg.CartesianGrid({"k": [1.0, 1.0, 2.0]})


# -- closed forms ---------------------------------------------------------------------------------

def _closed_form_c(p, a, k):
    return (1.0 - p.alpha * p.beta) * a * k ** p.alpha


def test_stochastic_growth_matches_the_closed_form():
    model, p, guess = rbc.build()
    kss, _ = rbc.steady_levels(p)
    sol = dsg.time_iteration(model, p, grid={"k": (0.6 * kss, 1.4 * kss, 25),
                                             "a": (0.95, 1.05, 7)},
                             cov={"eps_a": 0.007}, guess=guess, interpolation="cubic")
    assert sol.converged and sol.distance < 1e-8
    mids = sol.grid.midpoints
    c = sol.controls(mids)[:, 0]
    assert np.max(np.abs(c / _closed_form_c(p, mids[:, 1], mids[:, 0]) - 1)) < 1e-4
    # one state, by name; and the Euler equation off the grid
    one = sol.controls({"k": kss, "a": 1.0})
    assert one.shape == (1,) and abs(one[0] / _closed_form_c(p, 1.0, kss) - 1) < 1e-5
    assert np.max(np.abs(sol.euler_errors())) < 2e-3
    sim = sol.simulate(50, start={"k": kss, "a": 1.0}, seed=3)
    assert sim["states"].shape == (50, 2) and sim["jumps"].shape == (50, 1)
    k, a, c = sim["states"][:, 0], sim["states"][:, 1], sim["jumps"][:, 0]
    assert np.allclose(k[1:], p.alpha * p.beta * a[:-1] * k[:-1] ** p.alpha, rtol=1e-4)
    assert np.allclose(c, _closed_form_c(p, a, k), rtol=1e-4)


def test_markov_productivity_and_a_product_of_two_chains():
    """The closed form holds for any process of productivity, so for a Markov chain too."""
    p = rbc.Params()

    def equations(nxt, cur, eps, q):
        a, a1 = cur.a1 * cur.a2, nxt.a1 * nxt.a2
        return [
            nxt.k - a * cur.k ** q.alpha + cur.c,
            1.0 / cur.c - q.beta * q.alpha * a1 * nxt.k ** (q.alpha - 1) / nxt.c,
            np.log(nxt.a1) - q.rho * np.log(cur.a1) - eps.e1,
            np.log(nxt.a2) - 0.5 * np.log(cur.a2) - eps.e2,
        ]

    model = dsg.Model(states=["k", "a1", "a2"], jumps=["c"], shocks=["e1", "e2"],
                      equations=equations)
    ch1 = dsg.rouwenhorst(p.rho, 0.007, 3)
    ch2 = dsg.tauchen(0.5, 0.01, 2)
    ch1 = dsg.MarkovChain(np.exp(ch1.nodes), ch1.transition)
    ch2 = dsg.MarkovChain(np.exp(ch2.nodes), ch2.transition)
    kss, css = rbc.steady_levels(p)
    sol = dsg.time_iteration(model, p, grid={"k": (0.6 * kss, 1.4 * kss, 30)},
                             markov={"a1": ch1, "a2": ch2},
                             initial={"c": lambda s, q: 0.5 * s.a1 * s.a2 * s.k ** q.alpha},
                             interpolation="cubic")
    assert sol.converged and sol.values.shape == (6, 30, 1)
    k = np.linspace(0.7 * kss, 1.3 * kss, 9)
    for a1 in ch1.nodes:
        for a2 in ch2.nodes:
            X = np.column_stack([k, np.full_like(k, a1), np.full_like(k, a2)])
            c = sol.controls(X)[:, 0]
            assert np.allclose(c, _closed_form_c(p, a1 * a2, k), rtol=2e-5)
    with pytest.raises(ValueError, match="nodes of their chains"):
        sol.controls([kss, 1.0001, 1.0])
    sim = sol.simulate(40, seed=1)
    assert set(np.unique(sim["nodes"])) <= set(range(6))
    assert np.allclose(sim["jumps"][:, 0],
                       _closed_form_c(p, sim["states"][:, 1] * sim["states"][:, 2],
                                      sim["states"][:, 0]), rtol=1e-4)


# -- bounds ---------------------------------------------------------------------------------------

def _consumption_savings(beta=0.96, gamma=4.0, r=1.02):
    def equations(nxt, cur, eps, q):
        return [
            nxt.w - (np.exp(eps.y) + (cur.w - cur.c) * r),
            beta * (nxt.c / cur.c) ** (-gamma) * r - 1.0,
        ]
    return dsg.Model(states=["w"], jumps=["c"], shocks=["y"], equations=equations)


def test_borrowing_limit_binds_with_the_complementary_sign():
    model = _consumption_savings()
    sol = dsg.time_iteration(model, {}, grid={"w": (0.05, 4.0, 60)}, cov={"y": 0.1},
                             bounds={"c": (0.0, lambda s, _: s.w)},
                             initial={"c": lambda s, _: 0.9 * s.w}, tol=1e-11)
    assert sol.converged
    w = sol.grid.points[:, 0]
    c = sol.values[0, :, 0]
    assert np.all(c <= w + 1e-12) and np.all(c > 0)
    binding = np.isclose(c, w, atol=1e-10)
    assert binding[0] and not binding[-1]
    # constrained: the Euler condition is negative (it would pay to borrow); free: it is zero
    F = sol._problem.equilibrium(sol.values[0], np.zeros(w.size, dtype=int), w[:, None],
                                 sol._problem.coefficients(sol.values))[:, 0]
    assert np.all(F[binding] < 0)
    assert np.max(np.abs(F[~binding])) < 1e-9
    assert np.max(np.abs(sol.euler_errors(w[:, None]))) < 1e-9


def test_zero_lower_bound_is_a_max_with_the_standard_sign():
    def equations(nxt, cur, eps, q):
        return [nxt.z - 0.8 * cur.z - eps.e, cur.i - (0.01 + 1.5 * cur.z)]

    model = dsg.Model(states=["z"], jumps=["i"], shocks=["e"], equations=equations)
    sol = dsg.time_iteration(model, {}, grid={"z": (-0.03, 0.03, 13)}, cov={"e": 0.005},
                             bounds={"i": (0.0, None)}, initial={"i": 0.01})
    z = sol.grid.points[:, 0]
    assert np.allclose(sol.values[0, :, 0], np.maximum(0.0, 0.01 + 1.5 * z), atol=1e-12)
    # between the grid points, except in the cell where the rule kinks
    off = np.linspace(-0.04, 0.04, 33)
    off = off[np.abs(off + 0.01 / 1.5) > 0.005]
    assert np.allclose(sol.controls(off[:, None])[:, 0], np.maximum(0.0, 0.01 + 1.5 * off),
                       atol=1e-12)


# -- the model's rows, and models that do not vectorise -------------------------------------------

def test_a_scalar_only_model_gives_the_same_policy():
    model, p, guess = rbc.build()

    def scalar(nxt, cur, eps, q):
        y = float(cur.a) * float(cur.k) ** q.alpha
        y1 = float(nxt.a) * float(nxt.k) ** q.alpha
        return [nxt.k - (y - cur.c),
                1.0 / cur.c - q.beta / nxt.c * q.alpha * y1 / nxt.k,
                math.log(nxt.a) - q.rho * math.log(cur.a) - float(eps.eps_a)]

    other = dsg.Model(states=["k", "a"], jumps=["c"], shocks=["eps_a"], equations=scalar)
    kss, _ = rbc.steady_levels(p)
    kw = dict(grid={"k": (0.7 * kss, 1.3 * kss, 6), "a": (0.97, 1.03, 3)},
              cov={"eps_a": 0.007}, n_quad=3, guess=guess, tol=1e-10)
    a = dsg.time_iteration(model, p, **kw)
    b = dsg.time_iteration(other, p, **kw)
    assert a._problem._vectorised and not b._problem._vectorised
    assert np.allclose(a.values, b.values, rtol=0, atol=1e-10)


def test_refining_on_a_finer_grid_starts_from_the_coarse_policy():
    model, p, guess = rbc.build()
    kss, _ = rbc.steady_levels(p)
    coarse = dsg.time_iteration(model, p, grid={"k": (0.6 * kss, 1.4 * kss, 8),
                                                "a": (0.95, 1.05, 5)},
                                cov={"eps_a": 0.007}, guess=guess)
    fine = dsg.time_iteration(model, p, grid={"k": (0.6 * kss, 1.4 * kss, 30),
                                              "a": (0.95, 1.05, 5)},
                              cov={"eps_a": 0.007}, initial=coarse)
    assert fine.converged and fine.history[0] < 1e-2


def test_misdeclared_problems_are_refused():
    model, p, guess = rbc.build()
    kss, _ = rbc.steady_levels(p)
    with pytest.raises(ValueError, match="neither a grid axis"):
        dsg.time_iteration(model, p, grid={"k": (0.5, 1.5, 5)}, guess=guess)
    with pytest.raises(ValueError, match="not both"):
        dsg.time_iteration(model, p, grid={"k": (0.5, 1.5, 5), "a": (0.9, 1.1, 3)},
                           markov={"a": dsg.rouwenhorst(0.9, 0.01, 3)}, guess=guess)
    with pytest.raises(KeyError, match="not a jump"):
        dsg.time_iteration(model, p, grid={"k": (0.5, 1.5, 5), "a": (0.9, 1.1, 3)},
                           bounds={"k": (0.0, None)}, guess=guess)
    with pytest.raises(ValueError, match="guess or steady"):
        dsg.time_iteration(model, p, grid={"k": (0.5, 1.5, 5), "a": (0.9, 1.1, 3)})

    def static_state(nxt, cur, eps, q):
        # capital's law written without k at t+1: no row gives the next state
        return [cur.k - 0.2, 1.0 / cur.c - 1.0, nxt.a - cur.a - eps.eps_a]

    bad = dsg.Model(states=["k", "a"], jumps=["c"], shocks=["eps_a"], equations=static_state)
    with pytest.raises(ValueError, match="transitions"):
        dsg.time_iteration(bad, p, grid={"k": (0.1, 0.3, 4), "a": (0.9, 1.1, 3)},
                           initial={"c": 1.0})
