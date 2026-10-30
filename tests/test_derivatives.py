"""Derivatives of first-order solution matrices with respect to parameters."""

import numpy as np
import pytest

import dsgelab as dsg
from dsgelab.examples import rbc

pytest.importorskip("sympy")


def test_solution_jacobian_matches_closed_form_rbc():
    model, params, guess = rbc.build_symbolic()
    first = dsg.solve(model, params, guess=guess)
    before = dict(vars(params))
    jac = dsg.solution_jacobian(first, ["alpha", "rho"])
    alpha, beta = params.alpha, params.beta
    k = (alpha * beta) ** (1 / (1 - alpha))
    log_k = np.log(k)
    dk_dalpha = k * (1 / (alpha * (1 - alpha)) + log_k / (1 - alpha))
    dc_dalpha = k ** alpha * (log_k + alpha * dk_dalpha / k) - dk_dalpha

    np.testing.assert_allclose(jac.P[:, :, 0], [[1.0, 0.0], [0.0, 0.0]], rtol=1e-7, atol=1e-10)
    np.testing.assert_allclose(jac.P[:, :, 1], [[0.0, 0.0], [0.0, 1.0]], rtol=1e-7, atol=1e-10)
    np.testing.assert_allclose(jac.F[:, :, 0], [[1.0, 0.0]], rtol=1e-7, atol=1e-10)
    np.testing.assert_allclose(jac.F[:, :, 1], [[0.0, 0.0]], rtol=1e-7, atol=1e-10)
    np.testing.assert_allclose(jac.Q, 0.0, atol=1e-10)
    np.testing.assert_allclose(jac.steady[:, 0], [dk_dalpha, 0.0, dc_dalpha],
                               rtol=2e-6, atol=1e-10)
    np.testing.assert_allclose(jac.steady[:, 1], 0.0, atol=1e-10)
    assert jac.names == ["alpha", "rho"]
    assert jac.P.shape == (2, 2, 2)
    assert jac.F.shape == (1, 2, 2)
    assert jac.Q.shape == (2, 1, 2)
    assert jac.steady.shape == (3, 2)
    assert set(jac.error) == {"P", "F", "Q", "steady"}
    assert vars(params) == before


def test_implicit_qz_jacobian_matches_one_jump_closed_form():
    def eqs(nxt, cur, eps, p):
        return [nxt.x - p["rho"] * cur.x - p["phi"] * eps.e,
                p["a"] * nxt.y - cur.y + p["b"] * cur.x]

    model = dsg.Model(states=["x"], jumps=["y"], shocks=["e"], equations=eqs)
    params = {"rho": 0.7, "phi": 0.2, "a": 0.5, "b": 0.3}
    first = dsg.solve(model, params, guess={"x": 0.0, "y": 0.0})
    jac = dsg.solution_jacobian(first, ["rho", "phi", "a", "b"])
    denominator = 1 - params["a"] * params["rho"]
    np.testing.assert_allclose(jac.P, np.array([1.0, 0.0, 0.0, 0.0]).reshape(1, 1, 4),
                               rtol=1e-7, atol=1e-10)
    np.testing.assert_allclose(jac.Q, np.array([0.0, 1.0, 0.0, 0.0]).reshape(1, 1, 4),
                               rtol=1e-7, atol=1e-10)
    expected_F = [params["b"] * params["a"] / denominator ** 2, 0.0,
                  params["b"] * params["rho"] / denominator ** 2, 1.0 / denominator]
    np.testing.assert_allclose(jac.F.reshape(4), expected_F, rtol=1e-7, atol=1e-10)
    np.testing.assert_allclose(jac.steady, 0.0, atol=1e-12)


def test_implicit_qz_jacobian_handles_complex_stable_pair():
    def eqs(nxt, cur, eps, p):
        return [nxt.x1 - p["c"] * cur.x1 + p["s"] * cur.x2 - p["phi"] * eps.e,
                nxt.x2 - p["s"] * cur.x1 - p["c"] * cur.x2,
                p["a"] * nxt.y - cur.y + p["b"] * cur.x1]

    model = dsg.Model(states=["x1", "x2"], jumps=["y"], shocks=["e"], equations=eqs)
    params = {"c": 0.5, "s": 0.3, "phi": 0.2, "a": 0.5, "b": 0.3}
    first = dsg.solve(model, params, guess={"x1": 0.0, "x2": 0.0, "y": 0.0})
    jac = dsg.solution_jacobian(first, ["c", "s", "phi"])
    P = np.array([[params["c"], -params["s"]], [params["s"], params["c"]]])
    D = np.eye(2) - params["a"] * P
    F = np.array([[params["b"], 0.0]]) @ np.linalg.inv(D)
    dP = [np.eye(2), np.array([[0.0, -1.0], [1.0, 0.0]])]
    expected_F = [params["a"] * F @ value @ np.linalg.inv(D) for value in dP]
    np.testing.assert_allclose(jac.P[..., 0], dP[0], rtol=1e-7, atol=1e-10)
    np.testing.assert_allclose(jac.P[..., 1], dP[1], rtol=1e-7, atol=1e-10)
    np.testing.assert_allclose(jac.F[..., 0], expected_F[0], rtol=1e-7, atol=1e-10)
    np.testing.assert_allclose(jac.F[..., 1], expected_F[1], rtol=1e-7, atol=1e-10)
    np.testing.assert_allclose(jac.Q[..., 2], [[1.0], [0.0]], rtol=1e-7, atol=1e-10)


def test_second_order_jacobian_matches_log_ar1_derivatives():
    model, params, guess = rbc.build_symbolic()
    second = dsg.solve_second_order(dsg.solve(model, params, guess=guess),
                                    cov={"eps_a": 0.007})
    before = dict(vars(params))
    jac = dsg.second_order_jacobian(second, ["rho", "stderr eps_a"])
    ia = model.states.index("a")
    assert isinstance(jac, dsg.SecondOrderSolutionJacobian)
    assert jac.names == ["rho", "stderr eps_a"]
    assert jac.g_kk[ia, ia, ia, 0] == pytest.approx(2 * params.rho - 1, rel=1e-8)
    assert jac.g_ku[ia, ia, 0, 0] == pytest.approx(1.0, rel=1e-8)
    assert jac.g_uu[ia, 0, 0, 0] == 0.0
    assert jac.g_s2[ia, 0] == 0.0
    assert jac.g_k[ia, ia, 0] == pytest.approx(1.0, rel=1e-8)
    assert jac.g_u[ia, 0, 0] == 0.0
    assert jac.g_uu[ia, 0, 0, 1] == 0.0
    assert jac.g_s2[ia, 1] == 0.0
    assert vars(params) == before


def test_second_order_jacobian_argument_errors():
    model, params, guess = rbc.build_symbolic()
    second = dsg.solve_second_order(dsg.solve(model, params, guess=guess))
    with pytest.raises(ValueError, match="step must be positive"):
        dsg.second_order_jacobian(second, ["rho"], step=0.0)
    with pytest.raises(ValueError, match="at least 3"):
        dsg.second_order_jacobian(second, ["rho"], levels=2)


def test_third_order_jacobian_matches_log_ar1_derivatives():
    model, params, guess = rbc.build_symbolic()
    third = dsg.solve_third_order(dsg.solve_second_order(dsg.solve(model, params, guess=guess),
                                                        cov={"eps_a": 0.007}))
    before = dict(vars(params))
    jac = dsg.third_order_jacobian(third, ["rho"])
    ia = model.states.index("a")
    assert isinstance(jac, dsg.ThirdOrderSolutionJacobian)
    assert jac.g_kkk[ia, ia, ia, ia, 0] == pytest.approx(
        3 * params.rho ** 2 - 6 * params.rho + 2, rel=1e-5)
    assert jac.g_kku[ia, ia, ia, 0, 0] == pytest.approx(2 * params.rho - 1, rel=1e-5)
    assert jac.g_kuu[ia, ia, 0, 0, 0] == pytest.approx(1.0, rel=1e-5)
    assert jac.g_uuu[ia, 0, 0, 0, 0] == 0.0
    assert jac.g_s2[ia, 0] == 0.0
    assert vars(params) == before
    sparse = dsg.third_order_jacobian(third, ["rho"], sparse=True, direct_limit=0)
    for name in ("g_kkk", "g_kku", "g_kuu", "g_uuu", "g_s2", "steady"):
        np.testing.assert_allclose(getattr(sparse, name), getattr(jac, name),
                                   rtol=1e-6, atol=1e-8)


def test_third_order_jacobian_argument_errors():
    model, params, guess = rbc.build_symbolic()
    third = dsg.solve_third_order(dsg.solve_second_order(dsg.solve(model, params, guess=guess)))
    with pytest.raises(ValueError, match="step must be positive"):
        dsg.third_order_jacobian(third, ["rho"], step=0.0)
    with pytest.raises(ValueError, match="at least 3"):
        dsg.third_order_jacobian(third, ["rho"], levels=2)


def test_solution_jacobian_argument_errors():
    model, params, guess = rbc.build_symbolic()
    first = dsg.solve(model, params, guess=guess)
    with pytest.raises(ValueError, match="step must be positive"):
        dsg.solution_jacobian(first, ["alpha"], step=0.0)
    with pytest.raises(ValueError, match="at least 3"):
        dsg.solution_jacobian(first, ["alpha"], levels=2)
    with pytest.raises(AttributeError):
        dsg.solution_jacobian(first, ["not_a_parameter"])


def test_backward_solution_jacobian_uses_implicit_derivatives():
    def eqs(nxt, cur, eps, p):
        return [nxt.y - p["rho"] * cur.y - p["phi"] * eps.e]

    model = dsg.Model(states=["y"], jumps=[], shocks=["e"], equations=eqs, name="ar1")
    params = {"rho": 0.5, "phi": 0.2}
    first = dsg.solve(model, params, guess={"y": 0.0})
    jac = dsg.solution_jacobian(first, ["rho", "phi"])
    np.testing.assert_allclose(jac.P[..., 0], [[1.0]], atol=1e-10)
    np.testing.assert_allclose(jac.P[..., 1], [[0.0]], atol=1e-10)
    np.testing.assert_allclose(jac.Q[..., 0], [[0.0]], atol=1e-10)
    np.testing.assert_allclose(jac.Q[..., 1], [[1.0]], atol=1e-10)
    np.testing.assert_allclose(jac.steady, 0.0, atol=1e-10)
