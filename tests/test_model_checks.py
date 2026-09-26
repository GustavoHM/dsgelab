"""Independent checks on a solved model."""

import dataclasses

import numpy as np
import pytest

import dsgekit as dk
from dsgekit.examples import nk, rbc


def test_identity_check_separates_a_true_identity_from_a_false_one():
    model, p, guess = rbc.build()
    sol = dk.solve(model, p, guess=guess)
    path = np.array([sol.coords.to_level(np.zeros(model.n_vars)) for _ in range(3)])

    def resource(nxt, cur, params, ctx):
        y = cur["a"] * cur["k"] ** params.alpha
        inv = nxt["k"] - (1 - params.delta) * cur["k"]
        return y - cur["c"] - inv

    def wrong(nxt, cur, params, ctx):
        return resource(nxt, cur, params, ctx) + 0.01

    assert dk.check_identity(model, resource, path, p)["ok"]
    assert not dk.check_identity(model, wrong, path, p)["ok"]


def test_blanchard_kahn_report_gives_the_margin_not_only_the_pass():
    model, p, guess = nk.build()
    sol = dk.solve(model, p, guess=guess)
    rep = dk.blanchard_kahn_report(sol)
    assert rep["satisfied"]
    assert rep["margin"] > 0.0
    tight = dk.solve(model, dataclasses.replace(p, phi_pi=1.01), guess=guess)
    assert dk.blanchard_kahn_report(tight)["margin"] < rep["margin"]


def test_slackness_is_checked_along_the_impulse_responses_not_only_at_the_steady_state():
    model, p, guess = rbc.build()
    sol = dk.solve(model, p, guess=guess)
    small = dk.check_slack(sol, {"c_positive": lambda v: v["c"]}, size=-0.01)
    huge = dk.check_slack(sol, {"c_positive": lambda v: v["c"]}, size=-2.0)
    assert small["ok"]
    assert not huge["ok"]


def test_steady_state_invariance_detects_a_broken_invariance():
    """Capital does not depend on the persistence of technology; it does depend on beta."""
    model, p, guess = rbc.build()

    def by_rho(v):
        return dk.steady_state(model, guess, dataclasses.replace(p, rho=v))

    def by_beta(v):
        return dk.steady_state(model, guess, dataclasses.replace(p, beta=v))

    assert dk.steady_state_invariance(by_rho, [0.5, 0.8, 0.95], ["k", "c"])["ok"]
    assert not dk.steady_state_invariance(by_beta, [0.97, 0.98, 0.99], ["k"])["ok"]
