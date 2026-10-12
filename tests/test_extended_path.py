"""Extended path against answers known without it.

The comparison with Dynare (EP, stochastic EP of order 1 and 2 on the sparse and the
full tree, the hybrid correction) is validation/compare_ep.py.
"""

import numpy as np
import pytest

pytest.importorskip("sympy")

import dsgelab as dk  # noqa: E402
from dsgelab.extended_path import extended_path, quadrature_nodes  # noqa: E402

# Burnside (1998): the price-dividend ratio of a Lucas tree with AR(1) dividend growth.
BURNSIDE = dk.from_equations(
    ["x(+1) = (1 - rho)*xbar + rho*x + e",
     "1 = beta*exp(theta*x(+1))*(1 + y(+1))/y"],
    states=["x"], jumps=["y"], shocks=["e"], parameters=["beta", "theta", "rho", "xbar"])
PB = dict(beta=0.95, theta=-1.5, rho=-0.139, xbar=0.0179)
_g = PB["beta"] * np.exp(PB["theta"] * PB["xbar"])
SS_B = dict(x=PB["xbar"], y=_g / (1 - _g))


def test_deterministic_extended_path_is_the_truncated_backward_recursion():
    ss = dk.steady_state(BURNSIDE, SS_B, PB)
    H = 40
    E = 0.0348 * np.random.default_rng(1).standard_normal((8, 1))
    r = extended_path(BURNSIDE, ss, PB, E, horizon=H)
    b, th, rho, xb = PB["beta"], PB["theta"], PB["rho"], PB["xbar"]
    for t in range(1, 9):
        x_t = r["x"][t]
        assert x_t == pytest.approx((1 - rho) * xb + rho * r["x"][t - 1] + E[t - 1, 0], abs=1e-15)
        # agents expect no more shocks: x reverts deterministically, y is pinned at date H
        y = SS_B["y"]
        for j in range(H, 0, -1):
            x_j = xb + rho ** j * (x_t - xb)
            y = b * np.exp(th * x_j) * (1 + y)
        assert r["y"][t] == pytest.approx(y, rel=1e-12)


LINEAR = dk.from_equations(
    ["a(+1) = 0.8*a + e1",
     "b(+1) = 0.5*b + e2",
     "y = 0.6*y(+1) + a - 0.5*pie",
     "pie = 0.9*pie(+1) + 0.2*y + b"],
    states=["a", "b"], jumps=["y", "pie"], shocks=["e1", "e2"], parameters=[])
COV = np.diag([0.01, 0.04])


def test_in_a_linear_model_every_extended_path_is_the_first_order_solution():
    ss = dk.steady_state(LINEAR, dict(a=0, b=0, y=0, pie=0), {})
    sol = dk.solve(LINEAR, {}, steady=ss)
    E = np.random.default_rng(2).standard_normal((12, 2)) * np.sqrt(np.diag(COV))
    k = np.zeros(2)
    lin = []
    for t in range(12):
        k = sol.P @ k + sol.Q @ E[t]
        lin.append(np.concatenate([k, sol.F @ k]))
    lin = np.array(lin)
    for kw in (dict(order=0), dict(order=1), dict(order=2, tree="full")):
        r = extended_path(LINEAR, ss, {}, E, cov=COV, horizon=60, **kw)
        assert np.allclose(r.levels[1:], lin, atol=1e-11), kw


def test_risk_raises_the_price_of_the_tree_and_order_matters():
    # with theta < 0, uncertainty about dividend growth raises the price-dividend ratio
    ss = dk.steady_state(BURNSIDE, SS_B, PB)
    cov = np.array([[0.0348 ** 2]])
    E = np.zeros((3, 1))
    det = extended_path(BURNSIDE, ss, PB, E, horizon=40)
    s1 = extended_path(BURNSIDE, ss, PB, E, cov=cov, horizon=40, order=1)
    s2 = extended_path(BURNSIDE, ss, PB, E, cov=cov, horizon=40, order=2)
    full = extended_path(BURNSIDE, ss, PB, E, cov=cov, horizon=40, order=2, tree="full")
    hyb = extended_path(BURNSIDE, ss, PB, E, cov=cov, horizon=40, order=1, hybrid=True)
    y = [m["y"][1] for m in (det, s1, s2, full)]
    assert y[0] == pytest.approx(SS_B["y"], rel=1e-12)
    assert y[0] < y[1] < y[2] < y[3]
    assert hyb["y"][1] > s1["y"][1]


def test_quadrature_nodes_integrate_the_first_two_moments():
    cov = np.diag([0.04, 0.0, 0.09])
    nodes, w = quadrature_nodes(cov, 5)
    assert nodes.shape == (25, 3) and np.all(nodes[0] == 0)
    assert w.sum() == pytest.approx(1.0, abs=1e-14)
    assert np.allclose(w @ nodes, 0, atol=1e-15)
    assert np.allclose((nodes * w[:, None]).T @ nodes, cov, atol=1e-14)
    assert np.allclose(w @ nodes[:, 0] ** 4, 3 * 0.04 ** 2, atol=1e-14)   # exact to degree 9
    with pytest.raises(ValueError):
        quadrature_nodes(cov, 4)
