"""Value function iteration: closed forms, discrete choices, Howard's steps and the declaration."""

import math

import numpy as np
import pytest

import dsgelab as dsg

ALPHA, BETA = 0.36, 0.95
KSS = (ALPHA * BETA) ** (1 / (1 - ALPHA))
F = ALPHA / (1 - ALPHA * BETA)          # the value's coefficient on log k


def growth():
    """Stochastic growth with log utility and full depreciation, consumption the control."""
    return dsg.DynamicProgram(
        states=["k", "z"], controls=["c"],
        reward=lambda s, x, p: np.log(x.c),
        transition=lambda s, x, e, p: {"k": np.exp(s.z) * s.k ** ALPHA - x.c},
        discount=BETA,
        bounds={"c": (1e-9, lambda s, p: np.exp(s.z) * s.k ** ALPHA - 1e-9)})


def closed_form(chain, extra=0.0):
    """V(k, z_i) = E + F log k + H_i, with H = (I - beta P)^{-1} log(z) / (1 - alpha beta)."""
    H = np.linalg.solve(np.eye(chain.n) - BETA * chain.transition,
                        chain.nodes / (1 - ALPHA * BETA))
    E = (np.log(1 - ALPHA * BETA) + BETA * F * np.log(ALPHA * BETA) + extra) / (1 - BETA)
    return lambda k, i: E + F * np.log(k) + H[i]


# -- closed forms ---------------------------------------------------------------------------------

@pytest.mark.parametrize("kind, limits", [("linear", (3e-3, 6e-4)), ("cubic", (3e-6, 1e-7))])
def test_stochastic_growth_value_and_policy_match_the_closed_form(kind, limits):
    chain = dsg.rouwenhorst(0.9, 0.02, 3)
    V = closed_form(chain)
    errors = []
    for n in (20, 40):
        sol = dsg.value_iteration(growth(), {}, grid={"k": (0.5 * KSS, 1.5 * KSS, n)},
                                  markov={"z": chain}, interpolation=kind, tol=1e-11)
        assert sol.converged
        k = sol.grid.points[:, 0]
        errors.append(max(np.abs(sol.values[i] - V(k, i)).max() for i in range(chain.n)))
        y = np.exp(chain.nodes)[:, None] * k ** ALPHA
        assert np.abs(sol.policy[..., 0] - (1 - ALPHA * BETA) * y).max() < 5e-3
    assert errors[0] < limits[0] and errors[1] < limits[1]
    assert errors[1] < errors[0] / 3


def test_two_controls_by_projected_newton_match_the_closed_form_with_labour():
    # log c + theta log(1 - n), y = z k^a n^(1-a), c = y - k': k' = alpha beta y and
    # n = (1 - alpha) / ((1 - alpha) + theta (1 - alpha beta)) at every state. Next period's
    # capital is bounded to the grid, where the value is interpolated and not extrapolated.
    theta = 1.5
    chain = dsg.rouwenhorst(0.9, 0.02, 3)
    n_star = (1 - ALPHA) / ((1 - ALPHA) + theta * (1 - ALPHA * BETA))
    k_star = (ALPHA * BETA * n_star ** (1 - ALPHA)) ** (1 / (1 - ALPHA))
    lo, hi = 0.5 * k_star, 1.5 * k_star

    def output(s, x):
        return np.exp(s.z) * s.k ** ALPHA * x.n ** (1 - ALPHA)

    prog = dsg.DynamicProgram(
        states=["k", "z"], controls=["kp", "n"],
        reward=lambda s, x, p: np.log(output(s, x) - x.kp) + p["theta"] * np.log(1 - x.n),
        transition=lambda s, x, e, p: {"k": x.kp},
        discount=lambda p: p["beta"], bounds={"kp": (lo, hi), "n": (0.01, 0.99)})
    sol = dsg.value_iteration(prog, {"theta": theta, "beta": BETA}, grid={"k": (lo, hi, 30)},
                              markov={"z": chain}, interpolation="cubic", tol=1e-11)
    assert sol.converged
    k = sol.grid.points[:, 0]
    y = np.exp(chain.nodes)[:, None] * k ** ALPHA * n_star ** (1 - ALPHA)
    assert np.abs(sol.policy[..., 0] - ALPHA * BETA * y).max() < 1e-4
    assert np.abs(sol.policy[..., 1] - n_star).max() < 1e-4
    extra = (theta * np.log(1 - n_star) + (1 + BETA * F) * (1 - ALPHA) * np.log(n_star))
    V = closed_form(chain, extra)
    assert max(np.abs(sol.values[i] - V(k, i)).max() for i in range(chain.n)) < 1e-5


def test_iid_shocks_on_a_two_dimensional_grid_match_the_closed_form():
    # log z follows an AR(1) on the grid; the value is linear in log z, so the extrapolation of
    # the shocked next states beyond the grid is exact in that direction
    rho, sigma = 0.8, 0.03
    prog = dsg.DynamicProgram(
        states=["k", "a"], controls=["c"], shocks=["e"],
        reward=lambda s, x, p: np.log(x.c),
        transition=lambda s, x, e, p: {"k": np.exp(s.a) * s.k ** ALPHA - x.c,
                                       "a": rho * s.a + e.e},
        discount=BETA, bounds={"c": (1e-9, lambda s, p: np.exp(s.a) * s.k ** ALPHA - 1e-9)})
    sd = sigma / math.sqrt(1 - rho ** 2)
    sol = dsg.value_iteration(prog, {}, grid={"k": (0.5 * KSS, 1.5 * KSS, 25),
                                              "a": (-2 * sd, 2 * sd, 5)},
                              cov={"e": sigma}, n_quad=5, interpolation="cubic", tol=1e-11)
    k, a = sol.grid.points.T
    E = (np.log(1 - ALPHA * BETA) + BETA * F * np.log(ALPHA * BETA)) / (1 - BETA)
    V = E + F * np.log(k) + a / ((1 - ALPHA * BETA) * (1 - rho * BETA))
    assert np.abs(sol.values[0] - V).max() < 1e-5
    assert abs(sol.value({"k": KSS, "a": 0.0}) - (E + F * np.log(KSS))) < 1e-5


# -- discrete choices -----------------------------------------------------------------------------

def test_discrete_choices_solve_the_discretised_problem_exactly():
    chain = dsg.rouwenhorst(0.9, 0.05, 2)
    K = np.linspace(0.5 * KSS, 1.5 * KSS, 40)
    prog = dsg.DynamicProgram(
        states=["k", "z"], controls=["kp"],
        reward=lambda s, x, p: np.log(np.exp(s.z) * s.k ** ALPHA - x.kp),
        transition=lambda s, x, e, p: {"k": x.kp}, discount=BETA)
    sol = dsg.value_iteration(prog, {}, grid={"k": K}, markov={"z": chain}, choices={"kp": K},
                              tol=1e-12)
    # the same problem by policy iteration on the states (z, k), choosing the index of k'
    nz, nk = chain.n, K.size
    with np.errstate(invalid="ignore", divide="ignore"):
        R = np.log(np.exp(chain.nodes)[:, None, None] * K[None, :, None] ** ALPHA
                   - K[None, None, :])
    R = np.where(np.isfinite(R), R, -np.inf)
    sigma = np.zeros((nz, nk), dtype=int)
    for _ in range(100):
        r = np.take_along_axis(R, sigma[..., None], axis=2)[..., 0].ravel()
        Pm = np.zeros((nz * nk, nz * nk))
        for iz in range(nz):
            for ik in range(nk):
                Pm[iz * nk + ik, np.arange(nz) * nk + sigma[iz, ik]] = chain.transition[iz]
        v = np.linalg.solve(np.eye(nz * nk) - BETA * Pm, r).reshape(nz, nk)
        q = R + BETA * np.einsum("ij,jk->ik", chain.transition, v)[:, None, :]
        new = np.argmax(q, axis=2)
        if np.array_equal(new, sigma):
            break
        sigma = new
    assert np.abs(sol.values - v).max() < 1e-10
    assert np.array_equal(sol.policy[..., 0], K[sigma])


# -- the iteration --------------------------------------------------------------------------------

def test_howard_steps_reach_the_same_fixed_point_in_fewer_iterations():
    chain = dsg.rouwenhorst(0.9, 0.02, 2)
    kw = dict(grid={"k": (0.5 * KSS, 1.5 * KSS, 15)}, markov={"z": chain}, tol=1e-9)
    plain = dsg.value_iteration(growth(), {}, howard=0, **kw)
    fast = dsg.value_iteration(growth(), {}, howard=50, **kw)
    assert plain.converged and fast.converged
    assert fast.iterations * 5 < plain.iterations
    assert np.abs(plain.values - fast.values).max() < 1e-7
    assert np.abs(plain.policy - fast.policy).max() < 1e-6


def test_refining_from_a_coarse_solution_and_bellman_errors():
    chain = dsg.rouwenhorst(0.9, 0.02, 2)
    coarse = dsg.value_iteration(growth(), {}, grid={"k": (0.5 * KSS, 1.5 * KSS, 10)},
                                 markov={"z": chain}, interpolation="cubic", tol=1e-10)
    fine_grid = {"k": (0.5 * KSS, 1.5 * KSS, 40)}
    cold = dsg.value_iteration(growth(), {}, grid=fine_grid, markov={"z": chain},
                               interpolation="cubic", tol=1e-10)
    warm = dsg.value_iteration(growth(), {}, grid=fine_grid, markov={"z": chain},
                               interpolation="cubic", tol=1e-10, initial=coarse)
    assert warm.iterations < cold.iterations
    assert np.abs(warm.values - cold.values).max() < 1e-8
    at_nodes = cold.bellman_errors(np.column_stack([cold.grid.points[:, 0],
                                                    np.full(40, chain.nodes[0])]))
    assert np.abs(at_nodes).max() <= 1e-9
    assert np.abs(cold.bellman_errors()).max() < np.abs(coarse.bellman_errors()).max()


def test_simulation_follows_the_policy_and_the_transition():
    chain = dsg.rouwenhorst(0.9, 0.02, 3)
    sol = dsg.value_iteration(growth(), {}, grid={"k": (0.5 * KSS, 1.5 * KSS, 20)},
                              markov={"z": chain}, interpolation="cubic", tol=1e-10)
    path = sol.simulate(50, seed=3)
    k, z = path["states"].T
    c = path["controls"][:, 0]
    assert np.allclose(z, chain.nodes[path["nodes"]])
    assert np.allclose(k[1:], np.exp(z[:-1]) * k[:-1] ** ALPHA - c[:-1], atol=1e-14)
    assert np.allclose(c, sol.controls(path["states"])[:, 0])
    assert np.allclose(path["rewards"], np.log(c))
    # the policy is close to its closed form along the path
    assert np.abs(c - (1 - ALPHA * BETA) * np.exp(z) * k ** ALPHA).max() < 1e-4


def test_scalar_only_functions_are_evaluated_point_by_point():
    chain = dsg.rouwenhorst(0.9, 0.02, 2)
    scalar = dsg.DynamicProgram(
        states=["k", "z"], controls=["c"],
        reward=lambda s, x, p: math.log(x.c),
        transition=lambda s, x, e, p: [math.exp(s.z) * s.k ** ALPHA - x.c, 0.0],
        discount="beta",
        bounds={"c": (1e-9, lambda s, p: math.exp(s.z) * s.k ** ALPHA - 1e-9)})
    kw = dict(grid={"k": (0.5 * KSS, 1.5 * KSS, 8)}, markov={"z": chain}, tol=1e-10)
    a = dsg.value_iteration(scalar, {"beta": BETA}, **kw)
    b = dsg.value_iteration(growth(), {}, **kw)
    assert np.allclose(a.values, b.values, atol=1e-10)


# -- the declaration ------------------------------------------------------------------------------

def test_misdeclared_programs_are_refused():
    chain = dsg.rouwenhorst(0.9, 0.02, 2)
    grid = {"k": (0.5 * KSS, 1.5 * KSS, 5)}
    with pytest.raises(ValueError, match="more than once"):
        dsg.DynamicProgram(states=["k"], controls=["k"], reward=None, transition=None,
                           discount=0.9)
    with pytest.raises(KeyError, match="not controls"):
        dsg.DynamicProgram(states=["k"], controls=["c"], reward=None, transition=None,
                           discount=0.9, bounds={"k": (0, 1)})
    unbounded = dsg.DynamicProgram(states=["k", "z"], controls=["c"],
                                   reward=lambda s, x, p: np.log(x.c),
                                   transition=lambda s, x, e, p: {"k": s.k - x.c}, discount=0.9)
    with pytest.raises(ValueError, match="finite bounds"):
        dsg.value_iteration(unbounded, {}, grid=grid, markov={"z": chain})
    patient = dsg.DynamicProgram(states=["k", "z"], controls=["c"], reward=lambda s, x, p: x.c,
                                 transition=lambda s, x, e, p: {"k": s.k}, discount=1.0,
                                 bounds={"c": (0, 1)})
    with pytest.raises(ValueError, match="discount"):
        dsg.value_iteration(patient, {}, grid=grid, markov={"z": chain})
    with pytest.raises(ValueError, match="choices need every control"):
        dsg.value_iteration(growth(), {}, grid=grid, markov={"z": chain}, choices={"x": [1.0]})
    with pytest.raises(ValueError, match="neither a grid axis"):
        dsg.value_iteration(growth(), {}, grid=grid)
    with pytest.raises(KeyError, match="gives no value"):
        dsg.value_iteration(
            dsg.DynamicProgram(states=["k", "z"], controls=["c"], reward=lambda s, x, p: x.c,
                               transition=lambda s, x, e, p: {"z": s.z}, discount=0.9,
                               bounds={"c": (0, 1)}),
            {}, grid=grid, markov={"z": chain})
