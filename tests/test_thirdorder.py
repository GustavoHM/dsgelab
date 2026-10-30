"""Third order: the cubic block against the nonlinear system."""

from itertools import permutations

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


def test_cubic_innovation_block_matches_exact_rbc():
    model, params, _, _, third = _solutions()
    ia = model.states.index("a")
    ik = model.states.index("k")
    ic = model.n_states + model.jumps.index("c")
    assert third.h_uuu[ia, 0, 0, 0] == pytest.approx(1.0, rel=1e-4)
    assert np.max(np.abs(third.h_uuu[ik])) < 1e-6
    assert third.h_kuu[ia, ia, 0, 0] == pytest.approx(params.rho, rel=1e-4)
    assert np.max(np.abs(third.h_kuu[ik])) < 1e-6
    assert third.h_kku[ia, ia, ia, 0] == pytest.approx(params.rho * (params.rho - 1), rel=1e-4)
    assert np.max(np.abs(third.h_kku[ik])) < 1e-6
    assert third.h_kkk[ia, ia, ia, ia] == pytest.approx(
        params.rho * (params.rho - 1) * (params.rho - 2), rel=1e-4)
    assert third.innovation_residual < 1e-6
    rule = third.decision_rule()
    gamma, beta = params.alpha ** 2, params.rho + params.alpha
    assert rule.g_kkk[ic, ik, ik, ik] == pytest.approx(
        gamma * (gamma - 1) * (gamma - 2), rel=2e-4)
    assert rule.g_kkk[ic, ik, ik, ia] == pytest.approx(
        gamma * (gamma - 1) * beta, rel=2e-4)
    assert rule.g_kkk[ic, ik, ia, ia] == pytest.approx(
        gamma * beta * (beta - 1), rel=2e-4)
    assert rule.g_kkk[ic, ia, ia, ia] == pytest.approx(
        beta * (beta - 1) * (beta - 2), rel=2e-4)
    assert rule.g_uuu[ic, 0, 0, 0] == pytest.approx(1.0, rel=2e-4)
    assert rule.g_kuu[ic, ia, 0, 0] == pytest.approx(params.rho + params.alpha, rel=2e-4)
    assert rule.g_kuu[ic, ik, 0, 0] == pytest.approx(params.alpha ** 2, rel=2e-4)
    assert rule.g_kku[ic, ia, ia, 0] == pytest.approx(
        (params.rho + params.alpha) * (params.rho + params.alpha - 1), rel=2e-4)
    assert rule.g_kku[ic, ik, ik, 0] == pytest.approx(
        params.alpha ** 2 * (params.alpha ** 2 - 1), rel=2e-4)
    assert rule.g_kku[ic, ik, ia, 0] == pytest.approx(
        params.alpha ** 2 * (params.rho + params.alpha), rel=2e-4)
    assert np.max(np.abs(rule.g_uuu - np.transpose(rule.g_uuu, (0, 2, 1, 3)))) < 1e-8
    assert np.max(np.abs(rule.g_uuu - np.transpose(rule.g_uuu, (0, 1, 3, 2)))) < 1e-8
    assert np.max(np.abs(rule.g_kuu - np.transpose(rule.g_kuu, (0, 1, 3, 2)))) < 1e-8
    assert np.max(np.abs(rule.g_kku - np.transpose(rule.g_kku, (0, 2, 1, 3)))) < 1e-8
    assert np.max(np.abs(rule.g_kkk - np.transpose(rule.g_kkk, (0, 2, 1, 3)))) < 1e-8
    assert np.max(np.abs(rule.g_kkk - np.transpose(rule.g_kkk, (0, 1, 3, 2)))) < 1e-8


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
    assert np.max(np.abs(third.h_uuu)) < 1e-6
    assert np.max(np.abs(third.h_kuu)) < 1e-6
    assert np.max(np.abs(third.h_kku)) < 1e-6
    assert np.max(np.abs(third.decision_rule().g_uuu)) < 1e-6
    assert np.max(np.abs(third.decision_rule().g_kuu)) < 1e-6
    assert np.max(np.abs(third.decision_rule().g_kku)) < 1e-6


def test_third_order_irf_matches_first_order_on_linear_model():
    from dsgelab.examples import nk
    model, params, guess = nk.build()
    first = dsg.solve(model, params, guess=guess)
    third = dsg.solve_third_order(dsg.solve_second_order(first, cov={"eps_a": 0.01, "eps_v": 0.0025}))
    linear = first.irf("eps_a", size=0.01, periods=12)
    cubic = third.irf("eps_a", size=0.01, periods=12, percent=True)
    for name in linear:
        np.testing.assert_allclose(cubic[name], linear[name], rtol=1e-8, atol=1e-10)


def test_third_order_irf_uses_pruned_recursion():
    model, params, first, _, third = _solutions()
    linear = first.irf("eps_a", size=0.01, periods=12)
    cubic = third.irf("eps_a", size=0.01, periods=12, percent=True)
    for name in linear:
        assert np.all(np.isfinite(cubic[name]))
    exact = np.zeros(12)
    exact[0] = np.expm1(0.01)
    for t in range(1, 12):
        exact[t] = (1.0 + exact[t - 1]) ** params.rho - 1.0
    np.testing.assert_allclose(cubic["a"], exact * 100.0, rtol=3e-4, atol=1e-9)
    assert cubic["a"][0] == pytest.approx(100 * np.expm1(0.01), rel=3e-4)
    assert not np.allclose(cubic["a"][1:], linear["a"][1:])


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


@pytest.mark.parametrize("direct_limit", [None, 0])
def test_sparse_third_derivatives_match_dense_solution(direct_limit):
    from dsgelab.symbolic import model_from_expressions
    from dsgelab.tensors import SparseThirdDerivatives
    base, params, guess = rbc.build_symbolic()
    model = model_from_expressions(
        base.symbolic["expressions"], states=base.states, jumps=base.jumps,
        shocks=base.shocks, parameters=base.symbolic["parameters"], derivatives="third")
    first = dsg.solve(model, params, guess=guess)
    second = dsg.solve_second_order(first, cov={"eps_a": 0.007})
    options = {} if direct_limit is None else {"direct_limit": direct_limit}
    dense = dsg.solve_third_order(second, **options)
    sparse = dsg.solve_third_order(second, sparse=True, **options)
    for name in ("g_kkk", "h_kkk", "h_uuu", "h_kuu", "h_kku", "g_ssk", "h_ssk",
                 "g_xss", "h_xss", "g_uss", "h_uss"):
        value = getattr(sparse, name)
        if isinstance(value, SparseThirdDerivatives):
            value = value.to_dense()
        np.testing.assert_allclose(value, getattr(dense, name), rtol=1e-7, atol=1e-9)
    assert sparse.sylvester_residual < 1e-8
    assert sparse.innovation_residual < 1e-8
    dense_rule, sparse_rule = dense.decision_rule(), sparse.decision_rule()
    for name in ("g_uuu", "g_kuu", "g_kku", "g_kkk"):
        np.testing.assert_allclose(getattr(sparse_rule, name), getattr(dense_rule, name),
                                   rtol=1e-7, atol=1e-9)
    dense_irf = dense.irf("eps_a", periods=8)
    sparse_irf = sparse.irf("eps_a", periods=8)
    for name in dense_irf:
        np.testing.assert_allclose(sparse_irf[name], dense_irf[name], rtol=1e-7, atol=1e-9)
    dense_sim = dense.simulate(periods=40, seed=7)
    sparse_sim = sparse.simulate(periods=40, seed=7)
    for name in dense_sim:
        np.testing.assert_allclose(sparse_sim[name], dense_sim[name], rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(sparse.stochastic_steady_state().x,
                               dense.stochastic_steady_state().x, rtol=1e-7, atol=1e-9)


def test_canonical_sparse_third_derivatives_expand_all_permutations():
    from dsgelab.tensors import SparseThirdDerivatives
    canonical = [(0, 0, 1), (0, 1, 2), (1, 1, 1)]
    values = np.array([[2.0, 3.0, 0.0], [0.0, 0.0, 5.0]])
    sparse = SparseThirdDerivatives.from_canonical(values, canonical, (2, 3, 3, 3))
    np.testing.assert_allclose(sparse.to_canonical()[:, [1, 4, 6]], values)
    dense = sparse.to_dense()
    assert sparse.nnz == 3
    assert dense[0, 0, 0, 1] == dense[0, 0, 1, 0] == dense[0, 1, 0, 0] == 2.0
    for index in permutations((0, 1, 2)):
        assert dense[(0, *index)] == 3.0
    assert dense[1, 1, 1, 1] == 5.0
    u = np.array([[2.0], [3.0], [4.0]])
    contracted = sparse.contract(u, u, u)[:, 0, 0, 0]
    np.testing.assert_allclose(contracted, np.einsum("eabc,a,b,c->e", dense, u[:, 0], u[:, 0], u[:, 0]))
    weights = np.array([[1.0, 2.0], [-1.0, 0.5]])
    weighted = sparse.contract(u, u, u, weights=weights)[:, 0, 0, 0]
    np.testing.assert_allclose(weighted, weights @ contracted)
    U = np.array([[2.0, 1.0, 0.0], [3.0, 0.0, 1.0], [4.0, 1.0, 2.0]])
    full = sparse.contract(U, U, U)
    independent = sparse.contract_canonical(U, U, U)
    independent_indices = [(i, j, k) for i in range(3) for j in range(i, 3) for k in range(j, 3)]
    for column, index in enumerate(independent_indices):
        np.testing.assert_allclose(independent[:, column], full[(slice(None), *index)])


def test_sparse_sylvester_residual_does_not_expand_rule_tensors(monkeypatch):
    from dsgelab.tensors import SparseThirdDerivatives
    from dsgelab.thirdorder import _sparse_sylvester_residual
    rng = np.random.default_rng(8)
    Abar, Bbar = rng.normal(size=(3, 3)), rng.normal(size=(3, 3))
    P = rng.normal(size=(2, 2))
    dense = rng.normal(size=(3, 2, 2, 2))
    K3 = rng.normal(size=dense.shape)
    sparse = SparseThirdDerivatives.from_dense(dense)
    transformed = np.einsum("fabc,ai,bj,cl->fijl", dense, P, P, P)
    expected = np.max(np.abs(np.einsum("ea,aijl->eijl", Abar, dense)
                             + np.einsum("ef,fijl->eijl", Bbar, transformed) + K3))

    def no_dense(self):
        raise AssertionError("the sparse residual expanded a rule tensor")

    monkeypatch.setattr(SparseThirdDerivatives, "to_dense", no_dense)
    assert _sparse_sylvester_residual(sparse, Abar, Bbar, P, K3) == pytest.approx(expected)


def test_sparse_third_order_solve_does_not_expand_rule_tensors(monkeypatch):
    from dsgelab.tensors import SparseThirdDerivatives
    _, _, _, second, dense = _solutions()

    def no_dense(self):
        raise AssertionError("the sparse solve expanded a rule tensor")

    monkeypatch.setattr(SparseThirdDerivatives, "to_dense", no_dense)
    sparse = dsg.solve_third_order(second, sparse=True)
    assert sparse.sylvester_residual == pytest.approx(dense.sylvester_residual, abs=1e-12)
    assert sparse.risk_state_residual == pytest.approx(dense.risk_state_residual, abs=1e-12)
    assert sparse.risk_shock_residual == pytest.approx(dense.risk_shock_residual, abs=1e-12)


def test_third_order_reuses_precomputed_derivatives(monkeypatch):
    from dsgelab.linearize import second_derivatives, third_derivatives
    model, params, first, second, dense = _solutions()
    H = second_derivatives(model, first.steady, params, ctx=first.ctx,
                           row_scale=first.system.row_scale)
    T = third_derivatives(model, first.steady, params, ctx=first.ctx,
                          row_scale=first.system.row_scale)

    def unexpected(*args, **kwargs):
        raise AssertionError("a precomputed derivative was recomputed")

    monkeypatch.setattr("dsgelab.thirdorder.second_derivatives", unexpected)
    monkeypatch.setattr("dsgelab.thirdorder.third_derivatives", unexpected)
    reused = dsg.solve_third_order(second, hessian=H, third=T)
    np.testing.assert_allclose(reused.g_kkk, dense.g_kkk, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(reused.h_kkk, dense.h_kkk, rtol=1e-10, atol=1e-12)


def test_iterative_cubic_sylvester_uses_symmetric_solution():
    _, _, _, second, dense = _solutions()
    iterative = dsg.solve_third_order(second, direct_limit=0)
    np.testing.assert_allclose(iterative.g_kkk, dense.g_kkk, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(iterative.h_kkk, dense.h_kkk, rtol=1e-7, atol=1e-9)
    assert iterative.sylvester_residual < 1e-8


def test_canonical_iterative_solution_matches_dense_with_three_states():
    def eqs(nxt, cur, eps, p):
        return [cur.c ** -1 - p["beta"] * nxt.c ** -1
                * (p["alpha"] * np.exp(nxt.a) * nxt.k ** (p["alpha"] - 1) + 1 - p["delta"]),
                nxt.k - np.exp(cur.a) * cur.k ** p["alpha"] - cur.z + cur.c,
                nxt.a - p["rho"] * cur.a - eps.ea,
                nxt.z - p["phi"] * cur.z - eps.ez]

    model = dsg.Model(states=["a", "k", "z"], jumps=["c"], shocks=["ea", "ez"], equations=eqs)
    params = {"alpha": 0.36, "beta": 0.99, "rho": 0.95, "phi": 0.8, "delta": 1.0}
    k = (params["alpha"] * params["beta"]) ** (1 / (1 - params["alpha"]))
    guess = {"a": 0.0, "k": k, "z": 0.0, "c": k ** params["alpha"]}
    first = dsg.solve(model, params, guess=guess)
    second = dsg.solve_second_order(first, cov={"ea": 0.007, "ez": 0.005})
    dense = dsg.solve_third_order(second)
    sparse = dsg.solve_third_order(second, sparse=True, direct_limit=0)
    np.testing.assert_allclose(sparse.g_kkk.to_dense(), dense.g_kkk, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(sparse.h_kkk.to_dense(), dense.h_kkk, rtol=1e-6, atol=1e-8)
    assert sparse.sylvester_residual < 1e-7


def test_stochastic_steady_state_is_a_fixed_point_not_the_ergodic_mean():
    model, _, _, _, third = _solutions()
    ss = dsg.stochastic_steady_state(third)
    assert ss.residual < 1e-10
    assert ss["a"] == pytest.approx(1.0, abs=1e-14)
    k = third.second.first.coords.to_hat(ss.x)[: model.n_states]
    np.testing.assert_allclose(third.next_states(k), k, rtol=0.0, atol=1e-12)
    assert ss["a"] != third.second.ergodic_mean()["a"]


def test_stochastic_steady_state_accepts_a_full_guess_and_linear_model():
    from dsgelab.examples import nk
    model, params, guess = nk.build()
    third = dsg.solve_third_order(dsg.solve_second_order(
        dsg.solve(model, params, guess=guess), cov={"eps_a": 0.01, "eps_v": 0.0025}))
    ss = dsg.stochastic_steady_state(third, guess=third.second.first.steady.values)
    assert ss.residual < 1e-10
    np.testing.assert_allclose(ss.x, 0.0, atol=1e-12)
    with pytest.raises(ValueError, match="entries"):
        dsg.stochastic_steady_state(third, guess=np.zeros(1))


def test_third_order_jumps_follow_the_pruned_state_rule():
    # x copies capital, so in Dynare's timing, where k is the end-of-period stock, x[t] = k[t-1]
    p = dict(alpha=0.36, beta=0.99, delta=0.025, rho=0.95)
    model = dsg.from_equations([
        "k(+1) = (1-delta)*k + a*k**alpha - c",
        "1/c = beta/c(+1) * (alpha*a(+1)*k(+1)**(alpha-1) + 1 - delta)",
        "log(a(+1)) = rho*log(a) + e",
        "x = k",
    ], states=["k", "a"], jumps=["c", "x"], shocks=["e"], parameters=list(p))
    k = (p["alpha"] * p["beta"] / (1 - p["beta"] * (1 - p["delta"]))) ** (1 / (1 - p["alpha"]))
    guess = dict(k=k, a=1.0, c=k ** p["alpha"] - p["delta"] * k, x=k)
    first = dsg.solve(model, p, guess=guess)
    third = dsg.solve_third_order(dsg.solve_second_order(first, cov={"e": 0.02 ** 2}))
    simulated = third.simulate(periods=200, seed=1)
    np.testing.assert_allclose(simulated["x"][1:], simulated["k"][:-1], rtol=0, atol=1e-14)
    for draws in (0, 5):
        irf = third.irf("e", size=0.05, periods=20, n_draws=draws)
        np.testing.assert_allclose(irf["x"][1:], irf["k"][:-1], rtol=0, atol=1e-14)
        assert abs(irf["x"][0]) < 1e-14
