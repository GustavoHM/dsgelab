"""Deterministic transitions, against a path that is known in closed form."""

import numpy as np
import pytest

import dsgelab as dsg
from dsgelab.examples import rbc


def test_transition_matches_the_exact_policy():
    """With log utility and full depreciation the transition is k' = alpha*beta*k^alpha.

    A stacked-time Newton that got the boundary conditions wrong still converges
    to SOMETHING, so comparing against an exact path -- not merely checking that
    the residual fell -- is what makes this a test.
    """
    model, p, guess = rbc.build()
    ss = dsg.steady_state(model, guess, p)
    k_bar = ss["k"]
    k0 = 0.5 * k_bar

    path = dsg.perfect_foresight(model, ss, ss, params=p, T=150,
                                initial_states=[k0, 1.0])
    assert path.converged
    assert path.residual < 1e-10

    exact = np.empty(60)
    exact[0] = k0
    for t in range(1, 60):
        exact[t] = p.alpha * p.beta * exact[t - 1] ** p.alpha
    assert np.max(np.abs(path["k"][:60] - exact)) < 1e-8
    assert path["k"][-1] == pytest.approx(k_bar, rel=1e-8)


def test_announced_parameter_path_differs_from_a_surprise():
    """An announced change moves the economy before it happens; a surprise does not."""
    import dataclasses

    model, p, guess = rbc.build()
    ss0 = dsg.steady_state(model, guess, p)
    p1 = dataclasses.replace(p, beta=0.985)
    ss1 = dsg.steady_state(model, guess, p1)

    T = 120
    surprise = dsg.perfect_foresight(model, ss0, ss1, params=p1, T=T)
    schedule = [p] * 10 + [p1] * (T + 1 - 10)
    announced = dsg.perfect_foresight(model, ss0, ss1, params=schedule, T=T)

    assert surprise.converged and announced.converged
    # Consumption jumps on impact in both, but by different amounts: under the
    # announcement the old parameters still govern the first ten periods.
    assert abs(announced["c"][0] - surprise["c"][0]) > 1e-6
    assert announced["k"][-1] == pytest.approx(ss1["k"], rel=1e-7)


def test_boundary_conditions_are_counted_correctly():
    """States pinned at the start, jumps at the end; the interior is free."""
    model, p, guess = rbc.build()
    ss = dsg.steady_state(model, guess, p)
    k0 = 0.8 * ss["k"]
    path = dsg.perfect_foresight(model, ss, ss, params=p, T=80, initial_states=[k0, 1.0])
    assert path["k"][0] == pytest.approx(k0, rel=1e-14)
    assert path["a"][0] == pytest.approx(1.0, rel=1e-14)
    assert path["c"][-1] == pytest.approx(ss["c"], rel=1e-9)
