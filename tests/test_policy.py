"""Discretion, optimal simple rules and welfare, against answers known on paper.

The comparison with Dynare (discretionary_policy, osr, evaluate_planner_objective) is
validation/compare_policy.py; these tests need neither Dynare nor Octave.
"""

import numpy as np
import pytest

pytest.importorskip("sympy")

import dsgelab as dsg  # noqa: E402
from dsgelab.dynare import from_mod  # noqa: E402
from dsgelab.optimal import DiscretionError, discretionary_policy, osr  # noqa: E402
from dsgelab.welfare import consumption_equivalent, welfare  # noqa: E402

NK = ["x - x(+1) + (1/sigma)*(i - pi(+1)) = 0",
      "pi = beta*pi(+1) + kappa*x + u",
      "u(+1) = rho*u + e"]
NK_KW = dict(states=["u"], jumps=["x", "pi", "i"], shocks=["e"],
             parameters=["beta", "sigma", "kappa", "lam", "rho"], instruments=["i"],
             objective="-(pi**2 + lam*x**2)/2", discount="beta")
NK_PARAMS = dict(beta=0.99, sigma=1.0, kappa=0.1, lam=0.25, rho=0.5)


def test_discretion_matches_gali_closed_form():
    # Gali (2015, ch. 5): under discretion the targeting rule is pi_t = -(lam/kappa) x_t, and
    # x_t = -kappa/(kappa^2 + lam(1 - beta rho)) u_t, pi_t = lam/(kappa^2 + lam(1 - beta rho)) u_t.
    ds = discretionary_policy(NK, params=NK_PARAMS, **NK_KW)
    b, k, l, r = (NK_PARAMS[n] for n in ("beta", "kappa", "lam", "rho"))
    den = k ** 2 + l * (1 - b * r)
    irf = ds.irf("e", size=1.0, periods=20)
    u = irf["u"]
    assert u[0] == pytest.approx(1.0)                         # Dynare dating: u_0 = e_0
    assert np.max(np.abs(irf["x"] + k / den * u)) < 1e-10
    assert np.max(np.abs(irf["pi"] - l / den * u)) < 1e-10
    assert np.max(np.abs(irf["pi"] + l / k * irf["x"])) < 1e-10


def test_discretion_is_worse_than_commitment():
    ds = discretionary_policy(NK, params=NK_PARAMS, cov={"e": 0.01}, **NK_KW)
    m = dsg.ramsey_model(NK, **NK_KW)
    ss = dsg.ramsey_steady_state(m, dict(u=0, x=0, pi=0, i=0), NK_PARAMS)
    w_commit = welfare(m, NK_PARAMS, steady=ss, cov={"e": 0.01})
    w_disc = ds.welfare()
    assert w_disc["unconditional"] < w_commit.unconditional < 0


def test_discretion_refuses_what_it_cannot_do():
    with pytest.raises(DiscretionError, match="not linear"):
        discretionary_policy(["x - x(+1) + (i - pi(+1))**3 = 0", "pi = 0.99*pi(+1) + 0.1*x + u",
                              "u(+1) = 0.5*u + e"], params={}, states=["u"],
                             jumps=["x", "pi", "i"], shocks=["e"], parameters=[],
                             instruments=["i"], objective="-(pi**2 + x**2)", discount=0.99)
    with pytest.raises(DiscretionError, match="linear term"):
        discretionary_policy(NK, params=NK_PARAMS, **{**NK_KW, "objective": "-(pi**2) + x"})


RULE = dsg.from_equations(["y = z - r", "r = theta*z", "z(+1) = rho*z + e"],
                         states=["z"], jumps=["y", "r"], shocks=["e"],
                         parameters=["theta", "rho"], derivatives="jacobian")


@pytest.mark.parametrize("w, c", [(0.5, 0.0), (0.5, 0.3), (2.0, -0.4)])
def test_osr_finds_the_closed_form_coefficient(w, c):
    # y = (1-theta) z, r = theta z: the loss Var(y) + w Var(r) + c Cov(y, r) is
    # Var(z) [(1-theta)^2 + w theta^2 + c theta (1-theta)], minimised at
    # theta = (2 - c) / (2 + 2w - 2c). The pair weight counts ONCE, as in Dynare.
    params = dict(theta=0.0, rho=0.9)
    weights = {"y": 1.0, "r": w}
    if c:
        weights[("y", "r")] = c
    res = osr(RULE, params, rule_params=["theta"], weights=weights, cov={"e": 0.1},
              guess=dict(z=0, y=0, r=0))
    assert res.params["theta"] == pytest.approx((2 - c) / (2 + 2 * w - 2 * c), abs=1e-6)
    t = res.params["theta"]
    var_z = 0.01 / (1 - 0.81)
    assert res.loss == pytest.approx(var_z * ((1 - t) ** 2 + w * t ** 2 + c * t * (1 - t)),
                                     rel=1e-10)
    assert res.loss < res.initial_loss


LOGNORMAL = dsg.from_equations(["c = exp(z)", "z(+1) = rho*z + e"], states=["z"], jumps=["c"],
                              shocks=["e"], parameters=["rho", "beta"])


def test_welfare_of_a_lognormal_endowment():
    # U = c = exp(z): to second order E exp(z_t) = 1 + Var(z_t)/2. Unconditional welfare is
    # (1 + s2/2)/(1 - beta) with s2 = sigma^2/(1 - rho^2); conditional on z_0 = 0, Var(z_t) =
    # s2 (1 - rho^(2t)), so W = 1/(1-beta) + s2/2 [1/(1-beta) - 1/(1-beta rho^2)].
    p = dict(rho=0.9, beta=0.96)
    sigma = 0.02
    ss = dsg.steady_state(LOGNORMAL, dict(c=1.0, z=0.0), p)
    w = welfare(LOGNORMAL, p, steady=ss, cov={"e": sigma}, objective="c", discount="beta")
    s2 = sigma ** 2 / (1 - p["rho"] ** 2)
    b, r = p["beta"], p["rho"]
    assert w.unconditional == pytest.approx((1 + s2 / 2) / (1 - b), rel=1e-12)
    assert w.conditional == pytest.approx(1 / (1 - b) + s2 / 2 * (1 / (1 - b) - 1 / (1 - b * r ** 2)),
                                          rel=1e-12)
    assert w.deterministic == pytest.approx(1 / (1 - b))
    assert w.conditional_zero_multipliers is None


def test_consumption_equivalent_in_logs():
    # U = log(c): scaling c by (1 + lambda) adds log(1 + lambda)/(1 - beta) to welfare.
    p = dict(rho=0.9, beta=0.96)
    ss = dsg.steady_state(LOGNORMAL, dict(c=1.0, z=0.0), p)
    ref = welfare(LOGNORMAL, p, steady=ss, cov={"e": 0.02}, objective="log(c)",
                  discount="beta").unconditional
    lam = consumption_equivalent("log(c)", "c", target=ref + 0.5, reference=ref,
                                 model=LOGNORMAL, params=p, steady=ss, cov={"e": 0.02},
                                 discount="beta")
    assert lam == pytest.approx(np.exp(0.5 * (1 - p["beta"])) - 1, rel=1e-10)


def test_from_mod_reads_the_policy_commands():
    text = """
    var y inflation r;
    varexo e;
    parameters g rho;
    g = 1.2; rho = 0.5;
    model(linear);
    y = y(+1) - (r - inflation(+1)) + e;
    inflation = 0.99*inflation(+1) + 0.1*y;
    r = g*inflation;
    end;
    shocks; var e; stderr 0.1; end;
    optim_weights;
    inflation 1;
    y 0.5;
    y, inflation 0.1;
    end;
    osr_params g;
    osr_params_bounds;
    g, 1.01, 5;
    end;
    osr;
    """
    mf = from_mod(text)
    assert mf.osr_params == ["g"]
    assert mf.optim_weights == {"inflation": 1.0, "y": 0.5, ("y", "inflation"): 0.1}
    assert mf.osr_bounds == {"g": (1.01, 5.0)}
    nk = """
    var x pi i u;
    varexo e;
    parameters beta lam;
    beta = 0.99; lam = 0.25;
    model(linear);
    x = x(+1) - (i - pi(+1));
    pi = beta*pi(+1) + 0.1*x + u;
    u = 0.5*u(-1) + e;
    end;
    planner_objective pi^2 + lam*x^2;
    discretionary_policy(instruments=(i), planner_discount=beta);
    """
    mf = from_mod(nk)
    assert mf.model is None and mf.planner["kind"] == "discretion"
    assert mf.planner["instruments"] == ["i"]
    ds = discretionary_policy(mf, params=mf.params)
    # the .mod objective is a loss; read with the sign flipped, x moves against u
    assert ds.irf("e", periods=2)["x"][0] < 0
