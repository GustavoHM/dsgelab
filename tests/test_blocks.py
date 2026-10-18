"""Derivation of the first-order conditions from the agent's problem."""

import numpy as np
import pytest

import dsgelab as dsg

pytest.importorskip("sympy")
from dsgelab.blocks import FOCError, Problem, optimality_conditions  # noqa: E402
from dsgelab.examples import rbc  # noqa: E402
from dsgelab.symbolic import from_equations  # noqa: E402


def _rbc_problem():
    return Problem(
        "log(c)",
        controls=["c", "i"], states=["k"], exogenous=["a"],
        constraints=["c + i = a*k**alpha", "k(+1) = (1-delta)*k + i"],
        discount="beta", parameters=["alpha", "beta", "delta", "rho"], name="rbc")


def test_derives_the_rbc_euler_equation_and_solves_at_the_same_point():
    """What is tested is the SOLUTION, not the form of the equations.

    The derived block has four jumps (two controls and two multipliers) where the hand-written
    model has one. If the derivation is right, the state transition matrix that the two produce is
    the same, and for this RBC it is known on paper, `[[alpha, 1], [0, rho]]`.
    """
    foc = optimality_conditions(_rbc_problem())
    eqs = foc["equations"] + ["log(a(+1)) = rho*log(a) + eps_a"]
    model = from_equations(eqs, states=["k", "a"], jumps=foc["jumps"], shocks=["eps_a"],
                           parameters=["alpha", "beta", "delta", "rho"], name="rbc-foc")

    p = dict(alpha=0.36, beta=0.99, delta=1.0, rho=0.95)
    k, c = rbc.steady_levels(rbc.Params())
    guess = dict(k=k, a=1.0, c=c, i=k, lm_1=-1 / c, lm_2=-1 / c)
    sol = dsg.solve(model, p, guess=guess)

    ik, ia = model.states.index("k"), model.states.index("a")
    assert np.allclose(sol.P[np.ix_([ik, ia], [ik, ia])],
                       [[p["alpha"], 1.0], [0.0, p["rho"]]], atol=1e-8)


def test_consumption_foc_is_marginal_utility_minus_the_multiplier():
    foc = optimality_conditions(_rbc_problem())
    assert any("1/c" in e and "lm_1" in e for e in foc["equations"])
    assert foc["multipliers"] == ["lm_1", "lm_2"]
    assert foc["jumps"] == ["c", "i", "lm_1", "lm_2"]


def test_each_equation_says_where_it_came_from():
    """The result has to be auditable, with each equation traced to its origin."""
    foc = optimality_conditions(_rbc_problem())
    origens = set(foc["detail"].values())
    assert "FOC for control c" in origens
    assert "Euler equation for state k" in origens
    assert any(o.startswith("constraint 1") for o in origens)


def test_undeclared_name_is_refused():
    with pytest.raises(FOCError, match="undeclared"):
        optimality_conditions(Problem("log(c)", controls=["c"],
                                      constraints=["c = w*n"], parameters=["beta"]))


def test_problem_requiring_t_plus_2_is_refused():
    """An adjustment cost on the CHANGE in the stock puts k(+2) in the Euler equation.

    The function refuses, naming the state that caused it, instead of returning equations that the
    rest of the package cannot represent.
    """
    p = Problem("log(c)", controls=["c"], states=["k"],
                constraints=["c + k(+1) - k = -phi*(k(+1) - k)**2"],
                discount="beta", parameters=["beta", "phi"])
    with pytest.raises(FOCError, match="t\\+2"):
        optimality_conditions(p)
