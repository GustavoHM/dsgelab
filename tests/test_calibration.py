"""Pinning free parameters on targets, and diagnosing when that cannot be done."""

import numpy as np
import pytest

import dsgelab as dk
from dsgelab.examples import rbc


def test_calibrates_beta_to_a_capital_target():
    """With delta = 1 the answer is beta = k^(1-alpha)/alpha, so the fit is checkable."""
    model, p, guess = rbc.build()
    k_target = 0.22
    res = dk.calibrate(model, p, free={"beta": 0.95}, targets={"k": k_target},
                       guess=guess)
    assert res.success
    assert res.steady["k"] == pytest.approx(k_target, rel=1e-9)
    assert res.values["beta"] == pytest.approx(k_target ** (1 - p.alpha) / p.alpha, rel=1e-7)


def test_calibrates_two_parameters_on_two_derived_targets():
    """Targets that are functions of the steady state, not variables of it."""
    model, p, guess = rbc.build()

    def observables(ss, params):
        y = ss["a"] * ss["k"] ** params.alpha
        return {"ky": ss["k"] / y, "cy": ss["c"] / y}

    res = dk.calibrate(model, p, free={"beta": 0.95, "alpha": 0.30},
                       targets={"ky": 0.35, "cy": 0.65}, guess=guess,
                       observables=observables, bounds={"alpha": (0.05, 0.9),
                                                        "beta": (0.5, 0.999)})
    assert res.success
    assert res.achieved["ky"] == pytest.approx(0.35, rel=1e-8)
    assert res.achieved["cy"] == pytest.approx(0.65, rel=1e-8)


def test_target_jacobian_flags_an_unidentified_calibration():
    """Two parameters against one target: some direction moves nothing.

    The point of the diagnostic. Without it the symptom is a solver that stops
    at an arbitrary point on a ridge and reports success.
    """
    model, p, guess = rbc.build()
    ident = dk.target_jacobian(model, p, free=["beta", "alpha"], targets=["k"],
                               guess=guess)
    assert ident["elasticity"].shape == (1, 2)
    assert not ident["identified"]
    assert ident["rank"] == 1 and ident["n_free"] == 2
    assert ident["null_directions"].shape == (1, 2)

    # With a second, independent target the same two parameters ARE identified.
    def observables(ss, params):
        return {"k": ss["k"], "cy": ss["c"] / (ss["a"] * ss["k"] ** params.alpha)}

    ok = dk.target_jacobian(model, p, free=["beta", "alpha"], targets=["k", "cy"],
                            guess=guess, observables=observables)
    assert ok["identified"]


def test_calibration_failure_raises_rather_than_returning_a_wrong_number():
    model, p, guess = rbc.build()
    with pytest.raises(RuntimeError):
        # No beta in (0.5, 0.999) delivers a capital stock of 50.
        dk.calibrate(model, p, free={"beta": 0.95}, targets={"k": 50.0},
                     guess=guess, bounds={"beta": (0.5, 0.999)})
