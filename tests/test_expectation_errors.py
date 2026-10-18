"""Perfect foresight with expectation errors, and Dynare's deterministic syntax.

The comparison with Dynare's perfect_foresight_solver and
perfect_foresight_with_expectation_errors_solver, on Dynare's own test files, is
validation/compare_foresight.py.
"""

import numpy as np
import pytest

pytest.importorskip("sympy")

import dsgelab as dsg  # noqa: E402
from dsgelab.dynare import from_mod  # noqa: E402
from dsgelab.foresight import News, perfect_foresight_with_expectation_errors  # noqa: E402

RBC = dsg.from_equations(
    ["1/c = beta/c(+1)*(alpha*a(+1)*k(+1)**(alpha-1) + 1 - delta)",
     "c + k(+1) = a*k**alpha + (1-delta)*k",
     "a(+1) = 1 - rho + rho*a + e"],
    states=["k", "a"], jumps=["c"], shocks=["e"],
    parameters=["alpha", "beta", "delta", "rho"])
P = dict(alpha=0.33, beta=0.99, delta=0.025, rho=0.9)


def _ss():
    k = ((1 / P["beta"] - 1 + P["delta"]) / P["alpha"]) ** (1 / (P["alpha"] - 1))
    return dsg.steady_state(RBC, dict(k=k, a=1.0, c=k ** P["alpha"] - P["delta"] * k), P)


def test_news_that_is_no_news_changes_nothing():
    ss = _ss()
    shocks = np.zeros((60, 1)); shocks[4, 0] = 0.01
    plain = dsg.perfect_foresight(RBC, ss, ss, params=P, T=60, shocks=shocks, tol=1e-12)
    same = perfect_foresight_with_expectation_errors(
        RBC, ss, ss, params=P, T=60, shocks=shocks, tol=1e-12,
        news=[News(period=10, shocks=shocks)])
    assert np.max(np.abs(plain.levels - same.levels)) < 1e-10


def test_a_surprise_follows_the_old_plan_until_it_arrives():
    ss = _ss()
    revised = np.zeros((60, 1)); revised[9, 0] = 0.02     # the innovation into date 10
    path = perfect_foresight_with_expectation_errors(
        RBC, ss, ss, params=P, T=60, tol=1e-12, news=[News(period=10, shocks=revised)])
    assert np.max(np.abs(path.levels[:10] - ss.x)) < 1e-12        # nothing was expected
    a10 = 1 - P["rho"] + P["rho"] * 1.0 + 0.02
    assert path["a"][10] == pytest.approx(a10, abs=1e-12)          # the surprise lands at 10
    # from date 10 on, the realised path is a fresh plan from the states reached there
    k0 = np.array([path["k"][10], a10])
    fresh = dsg.perfect_foresight(RBC, ss, ss, params=P, T=50, initial_states=k0, tol=1e-12)
    assert np.max(np.abs(path.levels[10:] - fresh.levels)) < 1e-9
    # and agents who had known it all along would have moved earlier
    known = dsg.perfect_foresight(RBC, ss, ss, params=P, T=60, shocks=revised, tol=1e-12)
    assert abs(known["c"][0] - ss.values["c"]) > 1e-6


def test_constant_simulation_length_extends_the_horizon():
    ss = _ss()
    shocks = np.zeros((70, 1)); shocks[12, 0] = 0.01
    path = perfect_foresight_with_expectation_errors(
        RBC, ss, ss, params=P, T=60, tol=1e-12, constant_simulation_length=True,
        news=[News(period=10, shocks=shocks)])
    assert path.levels.shape[0] == 10 + 60 + 1
    assert path.plan_dates == [0, 10]


PFWEE = """
var c k;
varexo x;
parameters alph gam delt bet aa;
alph=0.5; gam=0.5; delt=0.02; bet=0.05; aa=0.5;
model;
c + k - aa*x*k(-1)^alph - (1-delt)*k(-1);
c^(-gam) - (1+bet)^(-1)*(aa*alph*x(+1)*k^(alph-1) + 1 - delt)*c(+1)^(-gam);
end;
initval;
x = 1;
k = ((delt+bet)/(1.0*aa*alph))^(1/(alph-1));
c = aa*k^alph-delt*k;
end;
shocks;
var x; periods 2:3; values 1.2;
end;
"""


def test_exogenous_levels_and_deterministic_shocks_from_a_mod():
    mf = from_mod(PFWEE)
    assert mf.params["EXO_SS_x"] == 1.0
    assert mf.det_shocks == {"x": {2: 1.2, 3: 1.2}}
    ss = mf.steady_state()
    assert ss.values["SHK_x"] == 1.0
    shocks, k0 = mf.deterministic(20, ss)
    path = dsg.perfect_foresight(mf.model, ss, ss, params=mf.params, T=20, shocks=shocks,
                                initial_states=k0, tol=1e-12)
    # Dynare period t+1 is date t: x is 1 in period 1, 1.2 in periods 2 and 3
    assert np.allclose(path["SHK_x"][:5], [1.0, 1.2, 1.2, 1.0, 1.0])


def test_information_table_becomes_news():
    mf = from_mod(PFWEE)
    table = {1: {"paths": {"x": {1: 1.2}}, "terminal": {"x": 1.0}},
             2: {"paths": {"x": {2: 1.0}}, "terminal": {"x": 1.0}},     # confirms: not news
             3: {"paths": {"x": {3: 1.3}}, "terminal": {"x": 1.1}}}
    news, start = mf.news(table, 10)
    assert [n.period for n in news] == [0, 2]
    path = perfect_foresight_with_expectation_errors(
        mf.model, start, start, params=mf.params, T=10, news=news,
        initial_states=mf.initial_states(start), tol=1e-12)
    assert np.allclose(path["SHK_x"][:4], [1.2, 1.0, 1.3, 1.1])
    k_new = ((0.02 + 0.05) / (0.5 * 0.5 * 1.1)) ** (1 / (0.5 - 1))
    assert path["k"][-1] == pytest.approx(k_new, rel=1e-10)        # the new steady state


def test_osr_on_welfare_finds_the_closed_form():
    # y = (1-theta) z, r = theta z, and U = -(y^2 + w r^2)/2: welfare is maximised where
    # Var(y) + w Var(r) is minimised, at theta = 1/(1+w), whatever the discount factor.
    rule = dsg.from_equations(["y = z - r", "r = theta*z", "z(+1) = rho*z + e"],
                             states=["z"], jumps=["y", "r"], shocks=["e"],
                             parameters=["theta", "rho", "beta"])
    p = dict(theta=0.0, rho=0.9, beta=0.96)
    w = 0.5
    res = dsg.osr(rule, p, rule_params=["theta"], objective=f"-(y**2 + {w}*r**2)/2",
                 discount="beta", cov={"e": 0.1}, guess=dict(z=0, y=0, r=0))
    assert res.params["theta"] == pytest.approx(1 / (1 + w), abs=1e-6)
    var_z = 0.01 / (1 - 0.81)
    t = res.params["theta"]
    assert res.loss == pytest.approx(0.5 * var_z * ((1 - t) ** 2 + w * t ** 2) / (1 - 0.96),
                                     rel=1e-9)
