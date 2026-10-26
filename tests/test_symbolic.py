"""The text front end must produce the same model as the hand-written residuals."""

import numpy as np
import pytest

import dsgelab as dsg
from dsgelab.examples import rbc

sympy = pytest.importorskip("sympy")
from dsgelab.symbolic import SymbolicError, from_equations  # noqa: E402


def test_symbolic_and_callable_models_agree():
    m1, p1, g1 = rbc.build()
    m2, p2, g2 = rbc.build_symbolic()
    s1 = dsg.solve(m1, p1, guess=g1)
    s2 = dsg.solve(m2, p2, guess=g2)
    assert np.max(np.abs(s1.P - s2.P)) < 1e-8
    assert np.max(np.abs(s1.F - s2.F)) < 1e-8
    assert dsg.compare_irfs(s1, s2, "eps_a")["worst"] < 1e-6


def test_exact_derivatives_are_used_and_beat_finite_differences():
    """The symbolic model carries analytic derivatives, so it matches the closed form better."""
    m2, p2, g2 = rbc.build_symbolic()
    assert m2.jacobian_fn is not None and m2.hessian_fn is not None
    s2 = dsg.solve(m2, p2, guess=g2)
    cf = rbc.closed_form(p2)
    assert np.max(np.abs(s2.P - cf["P"])) < 1e-12       # exact, not 1e-9
    assert np.max(np.abs(s2.F - cf["F"])) < 1e-12


def test_second_order_with_exact_hessian():
    m2, p2, g2 = rbc.build_symbolic()
    s2 = dsg.solve(m2, p2, guess=g2)
    so = dsg.solve_second_order(s2, cov={"eps_a": 0.007})
    assert so.h_ss[m2.states.index("a")] == pytest.approx(0.007 ** 2, rel=1e-10)
    assert all(2.9 < o < 3.1 for o in so.accuracy()["implied_order"])


def test_undeclared_name_is_rejected():
    with pytest.raises(SymbolicError, match="undeclared"):
        from_equations(["k(+1) = rho*k + gamma*a", "a(+1) = rho*a + eps_a"],
                       states=["k", "a"], jumps=[], shocks=["eps_a"],
                       parameters=["rho"])


def test_lags_and_leads_are_carried_by_listed_auxiliaries():
    import numpy as np
    import dsgelab as dsg
    m = from_equations(["a(+1) = 1.2*a - 0.35*a(-1) + eps_a", "y = a + 0.5*a(-2) + 0.25*y(+2)"],
                       states=["a"], jumps=["y"], shocks=["eps_a"], parameters=[])
    assert m.states == ["a", "LAG1_a", "LAG2_a"] and m.jumps == ["y", "LEAD1_y"]
    assert m.auxiliary == {"LAG1_a": ("lag", "a"), "LAG2_a": ("lag", "a"),
                           "LEAD1_y": ("lead", "y")}
    sol = dsg.solve(m, {}, guess={v: 0.0 for v in m.names})
    assert np.allclose(sol.P, [[1.2, -0.35, 0], [1, 0, 0], [0, 1, 0]], atol=1e-14)
    # y_t = sum_j 0.25^j (a_{t+2j} + 0.5 a_{t+2j-2}), from the AR(2) path of a
    a = sol.irf("eps_a", periods=80, percent=False)["a"]
    b = a + 0.5 * np.r_[0.0, 0.0, a[:-2]]
    y = [sum(0.25 ** j * b[t + 2 * j] for j in range((80 - t) // 2)) for t in range(20)]
    assert np.allclose(sol.irf("eps_a", periods=20, percent=False)["y"], y, atol=1e-13)


def test_a_lagged_innovation_is_refused_with_an_explanation():
    with pytest.raises(SymbolicError, match="innovation"):
        from_equations(["k(+1) = rho*k + eps_a(-1)", "a(+1) = rho*a + eps_a"],
                       states=["k", "a"], jumps=[], shocks=["eps_a"], parameters=["rho"])


def test_equation_count_is_checked():
    with pytest.raises(SymbolicError, match="equations for"):
        from_equations(["a(+1) = rho*a + eps_a"],
                       states=["a"], jumps=["c"], shocks=["eps_a"], parameters=["rho"])


def test_exact_third_derivatives_are_opt_in():
    m = from_equations(["x(+1) = rho*x + 0.1*x**3 + eps"],
                       states=["x"], jumps=[], shocks=["eps"], parameters=["rho"],
                       derivatives="third")
    ss = dsg.steady_state(m, {"x": 2.0}, {"rho": 0.5})
    system = dsg.first_order_system(m, ss, {"rho": 0.5})
    T = dsg.linearize.third_derivatives(m, ss, {"rho": 0.5},
                                        row_scale=system.row_scale)
    expected = -0.6 * abs(ss.x[0]) ** 3 / system.row_scale[0]
    assert ss.x[0] == pytest.approx(np.sqrt(5.0), rel=1e-8)
    assert m.third_fn is not None
    assert T.shape == (1, 3, 3, 3)
    assert T[0, 1, 1, 1] == pytest.approx(expected, abs=1e-14)
    np.testing.assert_allclose(T[0, 0], 0.0, atol=1e-14)
    np.testing.assert_allclose(T[0, 2], 0.0, atol=1e-14)
    sparse = dsg.third_derivatives(m, ss, {"rho": 0.5}, row_scale=system.row_scale,
                                   sparse=True)
    assert isinstance(sparse, dsg.SparseThirdDerivatives)
    assert sparse.nnz == 1
    np.testing.assert_allclose(sparse.to_dense(), T, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(sparse.contract(np.eye(3), np.eye(3), np.eye(3)), T,
                               rtol=0.0, atol=1e-14)
    H = dsg.second_derivatives(m, ss, {"rho": 0.5}, row_scale=system.row_scale)
    H_sparse = dsg.second_derivatives(m, ss, {"rho": 0.5}, row_scale=system.row_scale,
                                      sparse=True)
    assert isinstance(H_sparse, dsg.SparseSecondDerivatives)
    assert H_sparse.nnz == 1
    np.testing.assert_allclose(H_sparse.to_dense(), H, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(H_sparse.contract(np.eye(3), np.eye(3)), H,
                               rtol=0.0, atol=1e-14)

    default = from_equations(["x(+1) = rho*x + eps"], states=["x"], jumps=[],
                             shocks=["eps"], parameters=["rho"])
    assert default.third_fn is None


def test_derivatives_option_rejects_unknown_values():
    with pytest.raises(SymbolicError, match="third.*both.*jacobian.*none"):
        from_equations(["x(+1) = x"], states=["x"], jumps=[], shocks=[],
                       parameters=[], derivatives="cubic")
