"""OccBin against answers known without it.

The comparison with Dynare's OccBin (one and two constraints, surprise shocks,
curb_retrench) is validation/compare_occbin.py.
"""

import numpy as np
import pytest

pytest.importorskip("sympy")

from dsgelab.dynare import from_mod  # noqa: E402
from dsgelab.occbin import OccbinError, OccbinModel, occbin_simulate  # noqa: E402

# A floor on a backward-looking variable: the piecewise solution is max(x, floor) exactly.
FLOOR = OccbinModel.from_equations(
    ["x(+1) = rho*x + e",
     "[name='floor', relax='zlb'] y = x",
     "[name='floor', bind='zlb'] y = lo"],
    constraints={"zlb": {"bind": "x < lo", "relax": "x > lo"}},
    states=["x"], jumps=["y"], shocks=["e"], parameters=["rho", "lo"])


def test_a_floor_on_a_backward_variable_is_the_max():
    p = dict(rho=0.8, lo=-0.01)
    r = occbin_simulate(FLOOR, p, {"e": {3: -0.05, 12: 0.02}}, periods=30,
                        guess=dict(x=0.0, y=0.0))
    x = r.piecewise["x"]
    assert np.allclose(x, r.linear["x"], atol=1e-15)        # x does not feel the floor
    assert np.allclose(r.piecewise["y"], np.maximum(x, -0.01), atol=1e-14)
    assert np.array_equal(r.binding["zlb"], x < -0.01)


# Borrowing up to a multiple of income (Guerrieri and Iacoviello, 2015, in this
# package's syntax): either the limit holds with a positive multiplier or the
# multiplier is zero and the household borrows less.
BORROW = OccbinModel.from_equations(
    ["c = y + b - R*LB",
     "LB(+1) = b",
     "[name='debt', relax='slack'] b = m*y",
     "[name='debt', bind='slack'] lam = 0",
     "lam = 1/c - beta*R/c(+1)",
     "log(y(+1)) = rho*log(y) + u"],
    constraints={"slack": {"bind": "lam < 0", "relax": "b > m*y"}},
    states=["LB", "y"], jumps=["c", "b", "lam"], shocks=["u"],
    parameters=["R", "beta", "m", "rho"])
P = dict(R=1.04, beta=0.95, m=0.8, rho=0.85)
SS = dict(LB=0.8, y=1.0, b=0.8, c=1.0 + 0.8 - 1.04 * 0.8, lam=(1 - 0.95 * 1.04) / (1 + 0.8 - 1.04 * 0.8))


@pytest.mark.parametrize("curb", [False, True])
def test_complementary_slackness_on_the_piecewise_path(curb):
    r = occbin_simulate(BORROW, P, {"u": {2: 0.06, 20: -0.04}}, periods=60,
                        guess=SS, curb_retrench=curb)
    lam, b, y = r.piecewise["lam"], r.piecewise["b"], r.piecewise["y"]
    slack = r.binding["slack"]
    assert slack.any() and not slack.all()
    assert np.all(np.abs(lam[slack]) < 1e-14)                     # multiplier at zero ...
    assert np.all(b[slack] <= 0.8 * y[slack] + 1e-12)              # ... and borrowing below the limit
    assert np.allclose(b[~slack], 0.8 * y[~slack], atol=1e-14)     # limit holds ...
    assert np.all(lam[~slack] >= -1e-12)                           # ... with a nonnegative multiplier
    # far from the shocks the piecewise path is back on the linear one
    assert np.allclose(r.piecewise["c"][-5:], r.linear["c"][-5:], atol=1e-6)


def test_without_a_binding_constraint_the_paths_are_linear():
    r = occbin_simulate(BORROW, P, {"u": {2: -0.01}}, periods=30, guess=SS)
    assert not r.binding["slack"].any()
    for v in ("c", "b", "lam", "y"):
        assert np.allclose(r.piecewise[v], r.linear[v], atol=1e-15)


def test_too_few_iterations_are_reported():
    with pytest.raises(OccbinError):
        occbin_simulate(BORROW, P, {"u": {2: 0.06}}, periods=30, guess=SS, maxit=1)


def test_from_mod_reads_tags_constraints_and_surprise_shocks():
    mf = from_mod("""
var x y; varexo e; parameters rho lo;
rho = 0.8; lo = -0.01;
model;
x = rho*x(-1) + e;
[name='floor', relax='zlb'] y = x;
[name='floor', bind='zlb'] y = lo;
end;
occbin_constraints;
name 'zlb'; bind x < lo; relax x > lo;
end;
initval; x = 0; y = 0; end;
shocks(surprise); var e; periods 3; values -0.05; end;
occbin_setup(simul_periods=20);
""")
    assert mf.surprise == {"e": {3: -0.05}}
    r = mf.occbin_simulate()
    assert len(r.piecewise["y"]) == 20
    assert np.allclose(r.piecewise["y"], np.maximum(r.piecewise["x"], -0.01), atol=1e-14)


def test_a_tag_naming_an_undefined_constraint_is_refused():
    with pytest.raises(ValueError, match="undefined"):
        OccbinModel.from_equations(
            ["x(+1) = e", "[name='s', bind='nope'] y = 0", "[name='s', relax='nope'] y = x"],
            constraints={}, states=["x"], jumps=["y"], shocks=["e"], parameters=[])
