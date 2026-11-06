"""Nonstationary models: declared trends, the balanced-growth check, the growth path.

The comparison with Dynare (its detrending of fs2000_nonstationary.mod, its bgp.write
problem and the assertions of its tests/bgp) is validation/compare_import.py and
validation/compare_bgp.py.
"""

import numpy as np
import pytest

pytest.importorskip("sympy")

import dsgelab as dsg  # noqa: E402
from dsgelab.dynare import DynareImportError, from_mod  # noqa: E402
from dsgelab.symbolic import SymbolicError  # noqa: E402

# An RBC with labour-augmenting technology growing at the factor g, written in levels.
TRENDED = """
trend_var(growth_factor = g) A;
var(deflator = A) c k y;
parameters alpha beta delta g;
alpha = 0.33; beta = 0.99; delta = 0.025; g = 1.005;
model;
1/c = beta/c(+1)*(alpha*y(+1)/k + 1 - delta);
c + k = y + (1-delta)*k(-1);
y = k(-1)^alpha*A^(1-alpha);
end;
initval; c = 0.8; k = 10; y = 1; end;
"""

# The same model detrended by hand: x = X/A with A(+1) = g A.
BY_HAND = """
var c k y;
parameters alpha beta delta g;
alpha = 0.33; beta = 0.99; delta = 0.025; g = 1.005;
model;
1/c = beta/(g*c(+1))*(alpha*g*y(+1)/k + 1 - delta);
c + k = y + (1-delta)*k(-1)/g;
y = (k(-1)/g)^alpha;
end;
initval; c = 0.8; k = 10; y = 1; end;
"""


def test_declared_trends_give_the_model_detrended_by_hand():
    a, b = from_mod(TRENDED), from_mod(BY_HAND)
    sa, sb = a.steady_state(), b.steady_state()
    for v in ("c", "k", "y"):
        assert sa.values[v] == pytest.approx(sb.values[v], rel=1e-13)
    # closed form: k/y = alpha / (g/beta - 1 + delta) * g
    ky = 0.33 * 1.005 / (1.005 / 0.99 - 1 + 0.025)
    assert sa.values["k"] / sa.values["y"] == pytest.approx(ky, rel=1e-12)
    assert a.trends["deflators"]["c"] == ("mul", "A")


def test_a_log_trend_is_additive():
    log_model = """
log_trend_var(log_growth_factor = lg) LA;
var(log_deflator = LA) lc lk ly;
parameters alpha beta delta lg;
alpha = 0.33; beta = 0.99; delta = 0.025; lg = 0.005;
model;
exp(-lc) = beta*exp(-lc(+1))*(alpha*exp(ly(+1)-lk) + 1 - delta);
exp(lc) + exp(lk) = exp(ly) + (1-delta)*exp(lk(-1));
ly = alpha*lk(-1) + (1-alpha)*LA;
end;
initval; lc = 0.7; lk = 3.1; ly = 1; end;
"""
    s = from_mod(log_model).steady_state()
    h = from_mod(BY_HAND.replace("g = 1.005", f"g = {float(np.exp(0.005))!r}")).steady_state()
    for v in ("c", "k", "y"):
        assert s.values["l" + v] == pytest.approx(np.log(h.values[v]), abs=1e-12)


def test_a_variable_declared_twice_is_refused():
    with pytest.raises(DynareImportError, match="twice"):
        from_mod("var c k y;\n" + TRENDED)


def test_an_unbalanced_trend_is_refused():
    bad = TRENDED.replace("y = k(-1)^alpha*A^(1-alpha);", "y = k(-1)^alpha*A^(1-alpha) + 0.01;")
    with pytest.raises(DynareImportError, match="balanced growth"):
        from_mod(bad)
    bad = TRENDED.replace("c + k = y + (1-delta)*k(-1);", "c + k = y + (1-delta)*k(-1)*A;")
    with pytest.raises(DynareImportError, match="balanced growth"):
        from_mod(bad)


def test_the_growth_path_of_a_solow_model_in_levels():
    m = dsg.from_equations(
        ["E(+1) = gE*E", "N(+1) = gN*N", "y = k**alpha*(E*N)**(1-alpha)",
         "k(+1) = (1-delta)*k + s*y"],
        states=["E", "N", "k"], jumps=["y"], shocks=[], parameters=["alpha", "delta", "s",
                                                                     "gE", "gN"])
    p = dict(alpha=0.33, delta=0.02, s=0.2, gE=1.01, gN=1.005)
    r = dsg.balanced_growth(m, p, seed=3)
    assert r.success and not r.degenerate
    G = p["gE"] * p["gN"]
    assert r.growth["y"] == pytest.approx(G, rel=1e-12)
    assert r.growth["k"] == pytest.approx(G, rel=1e-12)
    # k is the capital of the period: k(+1)/y = s G / (G - 1 + delta), so k/y = s/(G-1+delta)
    assert r.ratio("k", "y") == pytest.approx(p["s"] / (G - 1 + p["delta"]), rel=1e-10)
    assert set(r.growing) == {"E", "N", "k", "y"}
    with pytest.raises(ValueError):
        r.ratio("E", "y")


# --- trends declared in from_equations, against the .mod path

def _irfs(model, params, ss, names, shock):
    sol = dsg.solve(model, params, steady=ss)
    out = sol.irf(shock, 0.01, periods=30, percent=False)
    return {v: np.asarray(out[v]) * sol.coords.scale[model.position(v)] for v in names}


RBC_EQUATIONS = ["1/c = beta/c(+1)*(alpha*y(+1)/k + 1 - delta)",
                 "c + k = y + (1-delta)*k(-1)",
                 "y = k(-1)**alpha*A**(1-alpha)"]


def test_trends_declared_in_from_equations_give_the_mod_path():
    m = dsg.from_equations(RBC_EQUATIONS, states=[], jumps=["c", "k", "y"], shocks=[],
                           parameters=["alpha", "beta", "delta", "g"],
                           trend_vars={"A": "g"}, deflators={"c": "A", "k": "A", "y": "A"})
    assert m.trends == {"deflators": {v: ("mul", "A") for v in ("c", "k", "y")},
                        "trend_vars": {"A": ("mul", "g")}}
    mf = from_mod(TRENDED)
    s = dsg.steady_state(m, dict(c=0.8, k=10, y=1), mf.params)
    sm = mf.steady_state()
    for v in ("c", "k", "y"):
        assert s.values[v] == pytest.approx(sm.values[v], rel=1e-13)


# Cash in advance with two stochastic trends, technology A and money M, a deflator that lags
# (M(-1)/A) and leads of order 2, as in Dynare's tests/trend_var/fs2000_nonstationary.mod.
CIA_MOD = """
var gM gA;
trend_var(growth_factor=gA) A;
trend_var(growth_factor=gM) M;
var(deflator=A) k c;
var(deflator=M(-1)/A) P;
var n;
varexo e_a e_m;
parameters alp bet gam mst rho delta;
alp = 0.33; bet = 0.99; gam = 0.003; mst = 1.011; rho = 0.7; delta = 0.02;
model;
gA = exp(gam + e_a);
log(gM) = (1-rho)*log(mst) + rho*log(gM(-1)) + e_m;
c + k = k(-1)^alp*(A*n)^(1-alp) + (1-delta)*k(-1);
P*c = M;
P/(c(+1)*P(+1)) = bet*P(+1)*(alp*k^(alp-1)*(A(+1)*n(+1))^(1-alp) + 1 - delta)/(c(+2)*P(+2));
n = 1/3;
end;
initval; k = 6; c = 0.45; P = 2.25; n = 0.33; gA = 1.003; gM = 1.011; end;
shocks; var e_a; stderr 0.014; var e_m; stderr 0.005; end;
"""

CIA_EQUATIONS = [
    "gA(+1) = exp(gam + e_a)",
    "log(gM(+1)) = (1-rho)*log(mst) + rho*log(gM) + e_m",
    "c + k = k(-1)**alp*(A*n)**(1-alp) + (1-delta)*k(-1)",
    "P*c = M",
    "P/(c(+1)*P(+1)) = bet*P(+1)*(alp*k**(alp-1)*(A(+1)*n(+1))**(1-alp) + 1 - delta)"
    "/(c(+2)*P(+2))",
    "n = 1/3"]


def test_two_stochastic_trends_and_a_lagged_deflator_give_the_mod_path():
    mf = from_mod(CIA_MOD)
    m = dsg.from_equations(CIA_EQUATIONS, states=["gA", "gM"], jumps=["k", "c", "P", "n"],
                           shocks=["e_a", "e_m"],
                           parameters=["alp", "bet", "gam", "mst", "rho", "delta"],
                           trend_vars={"A": "gA", "M": "gM"},
                           deflators={"k": "A", "c": "A", "P": "M(-1)/A"})
    # gA(+2) appears after detrending: the lead of a state starts from the jump LEAD0_gA
    assert "LEAD0_gA" in m.jumps
    names = ["gA", "gM", "k", "c", "P", "n"]
    sm = mf.steady_state()
    s = dsg.steady_state(m, {v: mf.guess[v] for v in names}, mf.params)
    for v in names:
        assert s.values[v] == pytest.approx(sm.values[v], rel=1e-12)
    for shock in ("e_a", "e_m"):
        a, b = _irfs(m, mf.params, s, names, shock), _irfs(mf.model, mf.params, sm, names, shock)
        for v in names:
            assert np.max(np.abs(a[v] - b[v])) < 1e-12 * max(1.0, np.max(np.abs(b[v])))


def test_log_trends_declared_in_from_equations_give_the_mod_path():
    log_mod = """
log_trend_var(log_growth_factor = lg) LA;
var(log_deflator = LA) lc lk ly;
parameters alpha beta delta lg;
alpha = 0.33; beta = 0.99; delta = 0.025; lg = 0.005;
model;
exp(-lc) = beta*exp(-lc(+1))*(alpha*exp(ly(+1)-lk) + 1 - delta);
exp(lc) + exp(lk) = exp(ly) + (1-delta)*exp(lk(-1));
ly = alpha*lk(-1) + (1-alpha)*LA;
end;
initval; lc = 0.7; lk = 3.1; ly = 1; end;
"""
    m = dsg.from_equations(
        ["exp(-lc) = beta*exp(-lc(+1))*(alpha*exp(ly(+1)-lk) + 1 - delta)",
         "exp(lc) + exp(lk) = exp(ly) + (1-delta)*exp(lk(-1))",
         "ly = alpha*lk(-1) + (1-alpha)*LA"],
        states=[], jumps=["lc", "lk", "ly"], shocks=[],
        parameters=["alpha", "beta", "delta", "lg"],
        log_trend_vars={"LA": "lg"}, log_deflators={v: "LA" for v in ("lc", "lk", "ly")})
    mf = from_mod(log_mod)
    s = dsg.steady_state(m, dict(lc=0.7, lk=3.1, ly=1), mf.params)
    for v in ("lc", "lk", "ly"):
        assert s.values[v] == pytest.approx(mf.steady_state().values[v], abs=1e-12)


def test_declared_trends_are_checked_in_from_equations():
    kw = dict(states=[], jumps=["c", "k", "y"], shocks=[],
              parameters=["alpha", "beta", "delta", "g"])
    every = {"c": "A", "k": "A", "y": "A"}
    bad = RBC_EQUATIONS[:2] + ["y = k(-1)**alpha*A**(1-alpha) + 0.01"]
    with pytest.raises(SymbolicError, match="balanced growth"):
        dsg.from_equations(bad, **kw, trend_vars={"A": "g"}, deflators=every)
    with pytest.raises(SymbolicError, match="not a state or a jump"):
        dsg.from_equations(RBC_EQUATIONS, **kw, trend_vars={"A": "g"}, deflators={"x": "A"})
    with pytest.raises(SymbolicError, match="also declared"):
        dsg.from_equations(RBC_EQUATIONS, **kw, trend_vars={"g": "1.01"}, deflators={"c": "g"})
    with pytest.raises(SymbolicError, match="growth factor of A"):
        dsg.from_equations(RBC_EQUATIONS, **kw, trend_vars={"A": "g*c"}, deflators=every)
    with pytest.raises(SymbolicError, match="deflator of c"):
        dsg.from_equations(RBC_EQUATIONS, **kw, trend_vars={"A": "g"},
                           deflators={**every, "c": "A*k"})
    # without a declaration the model is the one it was
    m = dsg.from_equations(["k(+1) = 0.9*k"], states=["k"], jumps=[], shocks=[], parameters=[])
    assert m.trends == {"deflators": {}, "trend_vars": {}}
