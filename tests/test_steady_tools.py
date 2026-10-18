"""Calibration equations and homotopy in the steady state, against closed forms."""

import pytest

pytest.importorskip("sympy")

import dsgelab as dsg  # noqa: E402

RBC = ["1/c = beta/c(+1)*(alpha*exp(z(+1))*k(+1)**(alpha-1) + 1 - delta)",
       "c + k(+1) = exp(z)*k**alpha + (1-delta)*k", "z(+1) = rho*z + e", "q = exp(z)*k**alpha"]
KW = dict(states=["k", "z"], jumps=["c", "q"], shocks=["e"],
          parameters=["alpha", "beta", "delta", "rho"])


def test_a_calibration_equation_pins_down_its_parameter():
    # k/q = alpha / (1/beta - 1 + delta), so k/q = 2.5 needs beta = 1/(alpha/2.5 + 1 - delta)
    m = dsg.from_equations(RBC, **KW, calibration_equations={"beta": "k/q = 2.5"})
    p = dict(alpha=0.5, beta=0.95, delta=0.02, rho=0.2)
    ss = dsg.steady_state(m, dict(k=5, z=0, c=2, q=2.2), p)
    assert ss.calibrated["beta"] == pytest.approx(1 / (0.5 / 2.5 + 1 - 0.02), rel=1e-13)
    assert ss.values["k"] / ss.values["q"] == pytest.approx(2.5, rel=1e-13)
    assert ss.params["beta"] == ss.calibrated["beta"] and p["beta"] == 0.95   # no mutation
    # the calibrated parameters solve the model as plain parameters
    plain = dsg.from_equations(RBC, **KW)
    again = dsg.steady_state(plain, ss.values, ss.params)
    assert again.values["k"] == pytest.approx(ss.values["k"], rel=1e-12)


def test_calibrate_argument_and_two_targets():
    m = dsg.from_equations(RBC, **KW)
    p = dict(alpha=0.4, beta=0.95, delta=0.05, rho=0.2)
    ss = dsg.steady_state(m, dict(k=5, z=0, c=2, q=2.2), p,
                         calibrate={"beta": "k/q = 3", "delta": "(q - c)/k = 0.025"})
    # investment share pins delta: delta = i/k; then beta from the capital ratio
    assert ss.calibrated["delta"] == pytest.approx(0.025, rel=1e-12)
    assert ss.calibrated["beta"] == pytest.approx(1 / (0.4 / 3 + 1 - 0.025), rel=1e-12)


def test_homotopy_reaches_a_steady_state_newton_cannot():
    m = dsg.from_equations(["1/c = beta/c(+1)*(alpha*k(+1)**(alpha-1) + 1 - delta)",
                           "c + k(+1) = k**alpha + (1-delta)*k"], states=["k"], jumps=["c"],
                          shocks=[], parameters=["alpha", "beta", "delta"])
    p = dict(alpha=0.33, beta=0.99, delta=0.9)
    guess = dict(k=30.0, c=2.0)
    with pytest.raises(RuntimeError):
        dsg.steady_state(m, guess, p)
    ss = dsg.steady_state(m, guess, p, homotopy={"delta": (0.02, 0.9)})
    k = ((1 / 0.99 - 1 + 0.9) / 0.33) ** (1 / (0.33 - 1))
    assert ss.values["k"] == pytest.approx(k, rel=1e-12)
    assert ss.homotopy_path[0] == 0.0 and ss.homotopy_path[-1] == 1.0
