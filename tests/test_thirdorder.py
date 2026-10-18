"""Third order: the cubic block against the nonlinear system."""

import numpy as np
import pytest

import dsgelab as dsg
from dsgelab.examples import rbc

pytest.importorskip("sympy")


def _solutions(sigma=0.007):
    model, params, guess = rbc.build_symbolic()
    ss = dsg.steady_state(model, guess, params)
    first = dsg.solve(model, params, steady=ss)
    second = dsg.solve_second_order(first, cov={"eps_a": sigma})
    third = dsg.solve_third_order(second, cov={"eps_a": sigma})
    return model, params, first, second, third


def test_each_order_gains_one_degree_of_accuracy():
    """The implied order of accuracy has to be 2, 3 and 4.

    Halving the displacement has to divide the residual by four at first order, by eight at second
    and by sixteen at third. A wrong cubic coefficient does not survive the third step, because the
    residual keeps falling by eight, and the ratio shows it.
    """
    _, _, first, second, third = _solutions()
    assert all(1.9 < o < 2.1 for o in first.accuracy()["implied_order"])
    assert all(2.85 < o < 3.15 for o in second.accuracy()["implied_order"])
    assert all(3.8 < o < 4.2 for o in third.accuracy()["implied_order"])


def test_third_order_sylvester_solves_exactly():
    _, _, _, _, third = _solutions()
    assert third.sylvester_residual < 1e-10


def test_os_termos_cubicos_sao_simetricos_nos_tres_indices():
    """Exact symmetry in theory. Imposing it removes the noise from differentiating the equation."""
    _, _, _, _, third = _solutions()
    for W in (third.g_kkk, third.h_kkk):
        assert np.max(np.abs(W - np.transpose(W, (0, 2, 1, 3)))) < 1e-12
        assert np.max(np.abs(W - np.transpose(W, (0, 1, 3, 2)))) < 1e-12


def test_pruned_third_order_simulation_does_not_explode():
    """An iterated cubic map diverges faster than a quadratic one, and pruning keeps it bounded."""
    _, _, _, _, third = _solutions(sigma=0.02)
    sim = third.simulate(periods=20_000, seed=3)
    assert np.all(np.isfinite(sim["k"]))
    assert np.max(np.abs(sim["k"])) < 1.0


def test_risk_slope_of_the_log_ar1_is_rho_sigma2():
    """E[a'|a] = a^ρ exp(σ²/2). In deviations, the cross term between â and σ² is ρ σ².

    h_ssk[a, a] has to come out with that value and that sign. The capital row,
    whose rule is exact and carries no risk, has to come out at zero. The same holds
    for the consumption jump.
    """
    model, params, _, _, third = _solutions(sigma=0.02)
    ia = model.states.index("a")
    ik = model.states.index("k")
    target = params.rho * 0.02 ** 2
    assert third.h_ssk[ia, ia] == pytest.approx(target, rel=1e-4, abs=1e-8)
    assert np.max(np.abs(third.h_ssk[ik])) < 1e-6
    assert np.max(np.abs(third.h_ssk[:, ik])) < 1e-6
    assert np.max(np.abs(third.g_ssk)) < 1e-6
    assert third.risk_slope_residual < 1e-6


def test_linear_model_has_negligible_third_order():
    """In the NK model, written in log deviations, third order has nothing to add.

    The tolerance is 1e-6 and not zero because this model is declared by a FUNCTION, so the
    Hessian comes from finite differences and carries noise of order 1e-10. The third derivative
    differentiates that object again, and the noise rises to 1e-8. The second-order terms, which
    come from the Hessian directly, come out at exactly zero. Declared through the symbolic front
    end, with an analytical Hessian, the cubic terms would also come out at zero.
    """
    from dsgelab.examples import nk
    model, params, guess = nk.build()
    first = dsg.solve(model, params, guess=guess)
    second = dsg.solve_second_order(first, cov={"eps_a": 0.01, "eps_v": 0.0025})
    third = dsg.solve_third_order(second, cov={"eps_a": 0.01, "eps_v": 0.0025})
    assert np.max(np.abs(second.g_kk)) == 0.0        # exact Hessian: exactly zero
    assert np.max(np.abs(third.g_kkk)) < 1e-6        # noise from differentiating the Hessian
    assert np.max(np.abs(third.h_kkk)) < 1e-6


def test_third_order_at_another_covariance_uses_its_own_risk_terms():
    import dsgelab as dsg
    from dsgelab.examples import rbc
    model, params, guess = rbc.build()
    sol = dsg.solve(model, params, guess=guess)
    low = dsg.solve_second_order(sol, cov={"eps_a": 0.007})
    high = dsg.solve_second_order(sol, cov={"eps_a": 0.014})
    t_mixed = dsg.solve_third_order(low, cov={"eps_a": 0.014})
    t_direct = dsg.solve_third_order(high, cov={"eps_a": 0.014})
    np.testing.assert_allclose(t_mixed.second.h_ss, high.h_ss, rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(t_mixed.h_ssk, t_direct.h_ssk, rtol=1e-8, atol=1e-12)
    a = dsg.solve_third_order(low).simulate(200, {"eps_a": 0.014}, seed=5, burn=20)
    b = t_direct.simulate(200, seed=5, burn=20)
    np.testing.assert_allclose(a["k"], b["k"], rtol=1e-8, atol=1e-12)
