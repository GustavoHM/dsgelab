"""The text front end must produce the same model as the hand-written residuals."""

import numpy as np
import pytest

import dsgekit as dk
from dsgekit.examples import rbc

sympy = pytest.importorskip("sympy")
from dsgekit.symbolic import SymbolicError, from_equations  # noqa: E402


def test_symbolic_and_callable_models_agree():
    m1, p1, g1 = rbc.build()
    m2, p2, g2 = rbc.build_symbolic()
    s1 = dk.solve(m1, p1, guess=g1)
    s2 = dk.solve(m2, p2, guess=g2)
    assert np.max(np.abs(s1.P - s2.P)) < 1e-8
    assert np.max(np.abs(s1.F - s2.F)) < 1e-8
    assert dk.compare_irfs(s1, s2, "eps_a")["worst"] < 1e-6


def test_exact_derivatives_are_used_and_beat_finite_differences():
    """The symbolic model carries analytic derivatives, so it matches the closed form better."""
    m2, p2, g2 = rbc.build_symbolic()
    assert m2.jacobian_fn is not None and m2.hessian_fn is not None
    s2 = dk.solve(m2, p2, guess=g2)
    cf = rbc.closed_form(p2)
    assert np.max(np.abs(s2.P - cf["P"])) < 1e-12       # exact, not 1e-9
    assert np.max(np.abs(s2.F - cf["F"])) < 1e-12


def test_second_order_with_exact_hessian():
    m2, p2, g2 = rbc.build_symbolic()
    s2 = dk.solve(m2, p2, guess=g2)
    so = dk.solve_second_order(s2, cov={"eps_a": 0.007})
    assert so.h_ss[m2.states.index("a")] == pytest.approx(0.007 ** 2, rel=1e-10)
    assert all(2.9 < o < 3.1 for o in so.accuracy()["implied_order"])


def test_undeclared_name_is_rejected():
    with pytest.raises(SymbolicError, match="undeclared"):
        from_equations(["k(+1) = rho*k + gamma*a", "a(+1) = rho*a + eps_a"],
                       states=["k", "a"], jumps=[], shocks=["eps_a"],
                       parameters=["rho"])


def test_lag_operator_is_rejected_with_an_explanation():
    with pytest.raises(SymbolicError, match="lag"):
        from_equations(["k(+1) = rho*k(-1)", "a(+1) = rho*a + eps_a"],
                       states=["k", "a"], jumps=[], shocks=["eps_a"],
                       parameters=["rho"])


def test_equation_count_is_checked():
    with pytest.raises(SymbolicError, match="equations for"):
        from_equations(["a(+1) = rho*a + eps_a"],
                       states=["a"], jumps=["c"], shocks=["eps_a"], parameters=["rho"])
