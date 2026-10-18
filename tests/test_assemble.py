"""assemble: several agents' problems and the closure, as one Model."""

import numpy as np
import pytest

pytest.importorskip("sympy")

import dsgelab as dsg  # noqa: E402
from dsgelab.blocks import FOCError, Problem, assemble  # noqa: E402


def _household(name="household", mult=None):
    return Problem("log(c)", controls=["c"], states=["kh"],
                   constraints=["c + kh(+1) = r*kh + w + (1-delta)*kh"], discount="beta",
                   parameters=["delta"], exogenous=["r", "w"], name=name, multipliers=mult)


def _firm():
    return Problem("a*kf**alpha*nf**(1-alpha) - w*nf - r*kf", controls=["kf", "nf"],
                   discount="beta", parameters=["alpha"], exogenous=["a", "w", "r"], name="firm")


def test_decentralised_economy_is_the_planners():
    # First welfare theorem: a household renting capital to a firm, prices clearing the
    # markets, gives the planner's allocation -- with log utility and full depreciation,
    # k' = alpha beta a k^alpha, i.e. P = [[alpha, 1], [0, rho]] in log deviations.
    m = assemble([_household(), _firm()],
                 equations=["kf = kh", "nf = 1", "log(a(+1)) = rho*log(a) + e"],
                 states=["a"], jumps=["w", "r"], shocks=["e"], parameters=["rho"])
    p = dict(alpha=0.36, beta=0.99, delta=1.0, rho=0.95)
    k = (0.36 * 0.99) ** (1 / (1 - 0.36))
    c = k ** 0.36 * (1 - 0.36 * 0.99)
    guess = dict(kh=k, a=1, c=c, lm_1=-1 / c, kf=k, nf=1, w=(1 - 0.36) * k ** 0.36,
                 r=0.36 * k ** (0.36 - 1))
    sol = dsg.solve(m, p, guess=guess)
    ik, ia = m.states.index("kh"), m.states.index("a")
    assert np.allclose(sol.P[np.ix_([ik, ia], [ik, ia])], [[0.36, 1], [0, 0.95]], atol=1e-12)
    blocks = {o[0] for o in m.assembly["origin"]}
    assert blocks == {"household", "firm", "closure"}


def test_a_variable_chosen_by_two_agents_is_refused():
    rival = Problem("log(kf)", controls=["kf"], discount="beta", name="rival")
    with pytest.raises(FOCError, match="chosen by two agents"):
        assemble([_firm(), rival], equations=[])


def test_the_count_is_checked_with_the_blocks_named():
    with pytest.raises(FOCError, match="equations by block"):
        assemble([_household(), _firm()], equations=["kf = kh"], states=["a"],
                 jumps=["w", "r"], shocks=["e"], parameters=["rho"])


def test_repeated_default_multipliers_get_the_agents_prefix():
    a = _household("patient")
    b = Problem("log(d)", controls=["d"], states=["kb"],
                constraints=["d + kb(+1) = r*kb + w + (1-delta)*kb"], discount="beta",
                parameters=["delta"], exogenous=["r", "w"], name="impatient")
    m = assemble([a, b], equations=["r = 0.03", "w = 1"], jumps=["w", "r"])
    assert "patient_lm_1" in m.jumps and "impatient_lm_1" in m.jumps
    assert m.assembly["renamed_multipliers"]["patient_lm_1"] == ("patient", "lm_1")
