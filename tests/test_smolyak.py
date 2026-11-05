"""Smolyak grids: nodes, the interpolant, and collocation with more than three states."""

from itertools import product

import numpy as np
import pytest

import dsgelab as dsg
from dsgelab.examples import rbc


def _unit_grid(d, level):
    return dsg.SmolyakGrid({f"x{j}": (-1.0, 1.0) for j in range(d)}, level)


# -- the grid -------------------------------------------------------------------------------------

def test_node_counts_are_those_of_jmmv():
    # JMMV (2014): 1 + 2d points at level 1 and 1 + 4d + 4d(d - 1)/2 at level 2; the counts at
    # level 3 are Tasmanian's (validation/compare_smolyak.py)
    for d in (2, 5, 10):
        assert _unit_grid(d, 1).n == 1 + 2 * d
        assert _unit_grid(d, 2).n == 1 + 4 * d + 4 * d * (d - 1) // 2
    assert _unit_grid(2, 3).n == 29 and _unit_grid(5, 3).n == 241
    g = _unit_grid(3, 3)
    assert g.degrees.shape == (g.n, 3) and len({tuple(r) for r in g.degrees}) == g.n
    assert len({tuple(np.round(r, 14)) for r in g.unit_points}) == g.n


def test_the_levels_are_nested_and_the_nodes_are_chebyshev_extrema():
    coarse, fine = _unit_grid(3, 2), _unit_grid(3, 3)
    known = {tuple(np.round(p, 12)) for p in fine.unit_points}
    assert all(tuple(np.round(p, 12)) in known for p in coarse.unit_points)
    extrema = -np.cos(np.pi * np.arange(9) / 8)
    assert np.allclose(_unit_grid(1, 3).axes[0], extrema, atol=1e-15)
    # the points the next level adds are not nodes
    mids = coarse.midpoints
    assert mids.shape[0] == fine.n - coarse.n
    assert not any(tuple(np.round(p, 12)) in {tuple(np.round(q, 12)) for q in coarse.unit_points}
                   for p in mids)


def test_anisotropic_levels_cap_each_axis():
    g = dsg.SmolyakGrid({"x": (0.0, 1.0), "y": (2.0, 5.0), "z": (-1.0, 0.0)},
                        {"x": 3, "y": 1, "z": 0})
    assert g.levels.tolist() == [3, 1, 0] and g.level == 3
    assert np.all(g.points[:, 2] == -0.5)                 # level 0: the centre only
    assert g.axes[1].tolist() == [2.0, 3.5, 5.0]
    assert g.axes[0].size == 9
    assert all(i[0] - 1 <= 3 and i[1] - 1 <= 1 and i[2] == 1 for i in g.indices)
    assert sum(i[0] + i[1] - 2 <= 3 for i in g.indices) == len(g.indices)
    assert dsg.SmolyakGrid({"x": (0, 1), "y": (0, 1)}, [3, 1]).n == g.n
    with pytest.raises(ValueError):
        dsg.SmolyakGrid({"x": (0, 1), "y": (0, 1)}, [2])
    with pytest.raises(ValueError):
        dsg.SmolyakGrid({"x": (0, 1)}, {"y": 2})
    with pytest.raises(ValueError):
        dsg.SmolyakGrid({"x": (1, 0)}, 2)
    with pytest.raises(ValueError):
        dsg.SmolyakGrid({"x": (0, 1)}, 1.5)


# -- the interpolant ------------------------------------------------------------------------------

def _lagrange(nodes, x):
    """The 1-D Lagrange basis on ``nodes`` at the points ``x``, one column per node."""
    L = np.ones((x.size, nodes.size))
    for a in range(nodes.size):
        for b in range(nodes.size):
            if a != b:
                L[:, a] *= (x - nodes[b]) / (nodes[a] - nodes[b])
    return L


def _combination(grid, f, X):
    """Smolyak's combination of tensor interpolants on the same multi-indices, built apart.

    The coefficient of the tensor interpolant of multi-index ``i`` is the sum of ``(-1)^|e|``
    over the ``e`` in ``{0, 1}^d`` with ``i + e`` in the set (Wasilkowski and Woźniakowski, 1995).
    """
    U = 2.0 * (X - grid.bounds[:, 0]) / (grid.bounds[:, 1] - grid.bounds[:, 0]) - 1.0
    index = set(grid.indices)

    def nodes(i):
        m = 1 if i == 1 else 2 ** (i - 1) + 1
        return np.zeros(1) if m == 1 else -np.cos(np.pi * np.arange(m) / (m - 1))

    out = np.zeros(X.shape[0])
    for i in grid.indices:
        c = sum((-1) ** sum(e) for e in product((0, 1), repeat=grid.d)
                if tuple(a + b for a, b in zip(i, e)) in index)
        if c == 0:
            continue
        axes = [nodes(ij) for ij in i]
        Ls = [_lagrange(a, U[:, j]) for j, a in enumerate(axes)]
        for corner in product(*[range(a.size) for a in axes]):
            point = np.array([a[k] for a, k in zip(axes, corner)])
            level = grid.bounds[:, 0] + (point + 1.0) * (grid.bounds[:, 1] - grid.bounds[:, 0]) / 2
            w = np.prod([L[:, k] for L, k in zip(Ls, corner)], axis=0)
            out += c * w * f(level[None, :])[0]
    return out


@pytest.mark.parametrize("levels", [2, 3, [3, 1, 2], [0, 2, 3]])
def test_interpolant_is_smolyaks_combination_of_tensor_interpolants(levels):
    grid = dsg.SmolyakGrid({"x": (0.5, 2.0), "y": (-1.0, 3.0), "z": (0.0, 1.0)}, levels)

    def f(X):
        return np.exp(0.3 * X[:, 0]) * np.cos(X[:, 1]) + X[:, 2] ** 3 / (1.0 + X[:, 0])

    rng = np.random.default_rng(0)
    X = rng.uniform(grid.bounds[:, 0], grid.bounds[:, 1], (25, 3))
    ours = grid.interpolate(f(grid.points)[:, None], X)[:, 0]
    assert np.allclose(ours, _combination(grid, f, X), rtol=0, atol=1e-12)
    W = grid.interpolation_weights(X)
    assert np.allclose(W @ f(grid.points), ours, atol=1e-12)
    assert np.allclose(W.sum(axis=1), 1.0, atol=1e-12)


def test_interpolant_takes_the_values_and_reproduces_its_polynomials():
    grid = dsg.SmolyakGrid({"x": (1.0, 3.0), "y": (0.0, 2.0), "z": (-2.0, 2.0), "w": (0.0, 1.0)},
                           3)
    rng = np.random.default_rng(1)
    V = rng.standard_normal((grid.n, 2))
    assert np.allclose(grid.interpolate(V, grid.points), V, atol=1e-11)
    # any combination of the basis is reproduced away from the nodes
    C = rng.standard_normal((grid.n, 2))
    X = rng.uniform(grid.bounds[:, 0], grid.bounds[:, 1], (30, 4))
    values = grid.basis(grid.points) @ C
    assert np.allclose(grid.interpolate(values, X), grid.basis(X) @ C, atol=1e-10)
    one = grid.interpolate(values, X[0])
    assert one.shape == (2,) and np.allclose(one, grid.basis(X[:1]) @ C)
    # in one dimension it is the Chebyshev interpolant through the extrema
    line = dsg.SmolyakGrid({"x": (-1.0, 1.0)}, 4)
    s = np.linspace(-1, 1, 41)
    fit = np.polynomial.chebyshev.Chebyshev.fit(line.points[:, 0], np.sin(3 * line.points[:, 0]),
                                                line.n - 1, domain=[-1, 1])
    assert np.allclose(line.interpolate(np.sin(3 * line.points), s[:, None])[:, 0], fit(s),
                       atol=1e-12)


# -- collocation ----------------------------------------------------------------------------------

def _growth(n):
    """A planner with ``n`` capital stocks, log utility and full depreciation."""
    shares = np.array([0.15, 0.10, 0.07, 0.05][:n])
    p = {"beta": 0.96, "rho": 0.9, "alpha": 0.36 * shares / shares.sum()}
    ks, inv = [f"k{j}" for j in range(n)], [f"i{j}" for j in range(n)]

    def equations(nxt, cur, eps, p):
        def output(x):
            y = x.a * np.prod([x[k] ** a for k, a in zip(ks, p["alpha"])], axis=0)
            return y, y - sum(x[i] for i in inv)
        y, c = output(cur)
        y1, c1 = output(nxt)
        return ([nxt[k] - cur[i] for k, i in zip(ks, inv)]
                + [1 - p["beta"] * c / c1 * a * y1 / nxt[k] for k, a in zip(ks, p["alpha"])]
                + [np.log(nxt.a) - p["rho"] * np.log(cur.a) - eps.e])

    model = dsg.Model(states=ks + ["a"], jumps=inv, shocks=["e"], equations=equations)
    alpha = p["alpha"]
    kss = p["beta"] * alpha * np.exp(np.sum(alpha * np.log(p["beta"] * alpha)) / (1 - alpha.sum()))
    guess = dict(zip(ks, kss)) | {"a": 1.0} | dict(zip(inv, kss))
    axes = {k: (0.75 * v, 1.25 * v) for k, v in zip(ks, kss)} | {"a": (0.97, 1.03)}

    def exact(X):
        return p["beta"] * alpha[None, :] * (X[:, -1] * np.prod(X[:, :n] ** alpha, axis=1))[:, None]

    rng = np.random.default_rng(2)
    lo, hi = np.array(list(axes.values())).T
    return model, p, guess, axes, exact, rng.uniform(lo, hi, (200, n + 1))


def test_projection_with_four_states_matches_the_closed_form():
    model, p, guess, axes, exact, X = _growth(3)
    errors = []
    for level in (1, 2, 3):
        proj = dsg.solve_projection(model, p, grid=dsg.SmolyakGrid(axes, level), guess=guess,
                                    cov={"e": 0.01}, n_quad=3)
        assert proj.success and proj.residual < 1e-10
        errors.append(np.max(np.abs(proj.jumps(X) / exact(X) - 1)))
    assert errors[-1] < 3e-6
    assert errors[0] / errors[1] > 20 and errors[1] / errors[2] > 20
    assert proj.nodes.shape == (137, 4) and proj.order == 8
    assert proj.jumps(X[0]).shape == (3,)


def test_projection_reorders_a_grid_whose_axes_are_not_in_state_order():
    model, p, guess = rbc.build()
    kss, _ = rbc.steady_levels(p)
    grid = dsg.SmolyakGrid({"a": (0.95, 1.05), "k": (0.7 * kss, 1.4 * kss)}, {"a": 1, "k": 4})
    proj = dsg.solve_projection(model, p, grid=grid, guess=guess, cov={"eps_a": 0.007})
    assert np.allclose(proj.bounds, [[0.7 * kss, 1.4 * kss], [0.95, 1.05]])
    k = np.linspace(0.8, 1.3, 7) * kss
    a = np.linspace(0.96, 1.04, 7)
    c = proj.jumps(np.column_stack([k, a]))[:, 0]
    assert np.max(np.abs(c / ((1 - p.alpha * p.beta) * a * k ** p.alpha) - 1)) < 2e-3
    with pytest.raises(ValueError):
        dsg.solve_projection(model, p, grid=grid, bounds=proj.bounds, guess=guess)
    with pytest.raises(ValueError):
        dsg.solve_projection(model, p, guess=guess)
    with pytest.raises(ValueError):
        dsg.solve_projection(model, p, guess=guess,
                             grid=dsg.SmolyakGrid({"k": (1.0, 2.0), "z": (0.0, 1.0)}, 2))


def test_time_iteration_on_a_smolyak_grid_agrees_with_projection():
    model, p, guess, axes, exact, X = _growth(3)
    grid = dsg.SmolyakGrid(axes, {"k0": 3, "k1": 3, "k2": 3, "a": 1})
    sol = dsg.time_iteration(model, p, grid=grid, cov={"e": 0.01}, n_quad=3, guess=guess,
                             tol=1e-11)
    assert sol.converged and sol.interpolation == "smolyak"
    assert np.max(np.abs(sol.controls(X) / exact(X) - 1)) < 3e-6
    proj = dsg.solve_projection(model, p, grid=grid, guess=guess, cov={"e": 0.01}, n_quad=3)
    assert np.max(np.abs(sol.controls(X) / proj.jumps(X) - 1)) < 1e-9
    errors = sol.euler_errors()
    assert errors.shape == (grid.midpoints.shape[0], 3) and np.max(np.abs(errors)) < 1e-5
    path = sol.simulate(20, seed=0)
    assert path["states"].shape == (20, 4) and np.all(np.isfinite(path["jumps"]))
    with pytest.raises(ValueError):
        dsg.time_iteration(model, p, grid=grid, cov={"e": 0.01}, guess=guess,
                           interpolation="cubic")
