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
