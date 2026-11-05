"""The endogenous grid method: closed forms, time iteration, the borrowing limit and declarations."""

import math

import numpy as np
import pytest

import dsgelab as dsg

ALPHA, BETA = 0.36, 0.95
KSS = (ALPHA * BETA) ** (1 / (1 - ALPHA))
SAVING = {"beta": 0.96, "gamma": 4.0, "R": 1.02}
NODES = np.array([-0.12815516, -0.05244005, 0.0, 0.05244005, 0.12815516])[:, None]


def growth():
    """Stochastic growth with log utility and full depreciation; the poststate is next capital."""
    return dsg.EGMProgram(
        states=["k", "z"], controls=["c"], poststate="kp", expectations=["q"],
        expectation=lambda s, x, e, p: {"q": BETA * ALPHA * np.exp(s.z) * s.k ** (ALPHA - 1)
                                        / x.c},
        direct_response=lambda a, z, p: {"c": 1 / z.q},
        reverse_state=lambda a, x, p: ((a.kp + x.c) / np.exp(a.z)) ** (1 / ALPHA),
        half_transition=lambda a, e, p: a.kp,
        post_decision=lambda s, x, p: np.exp(s.z) * s.k ** ALPHA - x.c)


def saving(markov=False):
    """Consumption and saving with cash on hand w and a borrowing limit, c <= w."""
    return dsg.EGMProgram(
        states=["w", "y"] if markov else ["w"], controls=["c"], poststate="a",
        expectations=["mr"], shocks=[] if markov else ["y"],
        expectation=lambda s, x, e, p: p["beta"] * p["R"] * x.c ** -p["gamma"],
        direct_response=lambda a, z, p: z.mr ** (-1 / p["gamma"]),
        reverse_state=lambda a, x, p: a.a + x.c,
        half_transition=lambda a, e, p: p["R"] * a.a + np.exp(e.y),
        post_decision=lambda s, x, p: s.w - x.c,
        bounds={"c": (0.0, lambda s, p: s.w)})


def solve_saving(n=100, **kw):
    return dsg.egm(saving(), SAVING, grid={"a": (0.0, 8.0, n)}, initial={"c": 0.9},
                   quadrature=(NODES, np.full(5, 0.2)), tol=1e-11, **kw)


# -- closed forms ---------------------------------------------------------------------------------

def test_a_linear_policy_is_reproduced_exactly():
    # with no income, w' = R (w - c) and c = kappa w, kappa = 1 - (beta R)^(1/gamma) / R; the
    # policy is linear, so its linear interpolation and extrapolation are exact
    beta, R, gamma = 0.96, 1.03, 2.0
    kappa = 1 - (beta * R) ** (1 / gamma) / R
    prog = dsg.EGMProgram(
        states=["w"], controls=["c"], poststate="a", expectations=["mr"],
        expectation=lambda s, x, e, p: beta * R * x.c ** -gamma,
        direct_response=lambda a, z, p: z.mr ** (-1 / gamma),
        reverse_state=lambda a, x, p: a.a + x.c,
        half_transition=lambda a, e, p: R * a.a,
        post_decision=lambda s, x, p: s.w - x.c)
    sol = dsg.egm(prog, {}, grid={"a": (0.1, 10.0, 20)}, initial={"c": lambda s, p: s.w},
                  tol=1e-13)
    assert sol.converged
    w = np.linspace(0.5, 20.0, 50)
    assert np.abs(sol.controls(w[:, None])[:, 0] / (kappa * w) - 1).max() < 1e-11
    assert np.abs(sol.euler_errors()).max() < 1e-11


@pytest.mark.parametrize("kind, limit", [("linear", 3e-3), ("cubic", 2e-3)])
def test_stochastic_growth_policy_matches_the_closed_form(kind, limit):
    chain = dsg.rouwenhorst(0.9, 0.02, 3)
    errors = []
    for n in (20, 40):
        sol = dsg.egm(growth(), {}, grid={"kp": (0.5 * KSS, 1.5 * KSS, n)}, markov={"z": chain},
                      interpolation=kind, tol=1e-12,
                      initial={"c": lambda s, p: 0.5 * np.exp(s.z) * s.k ** ALPHA})
        assert sol.converged
        worst = 0.0
        for m, z in enumerate(chain.nodes):
            k = 0.5 * (sol.endogenous[m, 1:] + sol.endogenous[m, :-1])
            c = sol.controls(np.column_stack([k, np.full(k.size, z)]))[:, 0]
            exact = (1 - ALPHA * BETA) * np.exp(z) * k ** ALPHA
            worst = max(worst, np.abs(c / exact - 1).max())
        errors.append(worst)
    assert errors[1] < limit
    assert 3.5 < errors[0] / errors[1] < 5.0


# -- against time iteration and the borrowing limit -----------------------------------------------

def test_agrees_with_time_iteration_on_saving_with_a_borrowing_limit():
    sol = solve_saving(200)
    assert sol.converged

    def equations(nxt, cur, eps, p):
        return [nxt.w - (p["R"] * (cur.w - cur.c) + np.exp(eps.y)),
                p["beta"] * p["R"] * (nxt.c / cur.c) ** -p["gamma"] - 1]

    model = dsg.Model(states=["w"], jumps=["c"], shocks=["y"], equations=equations)
    w = np.linspace(0.01, 8.0, 200)
    ti = dsg.time_iteration(model, SAVING, grid={"w": w}, quadrature=(NODES, np.full(5, 0.2)),
                            bounds={"c": (0.0, lambda s, p: s.w)},
                            initial={"c": lambda s, p: 0.9 * s.w}, tol=1e-11)
    gap = np.abs(sol.controls(w[:, None])[:, 0] - ti.values[0, :, 0])[w <= 4]
    assert gap.max() < 5e-4 and gap.mean() < 1e-4


def test_the_borrowing_limit_binds_below_the_lowest_endogenous_point():
    sol = solve_saving()
    w0 = sol.endogenous[0, 0]
    assert sol.values[0, 0, 0] == pytest.approx(w0)        # a = 0 there, so c = w
    w = np.linspace(0.05, w0, 10)
    assert np.allclose(sol.controls(w[:, None])[:, 0], w, rtol=0, atol=1e-14)
    above = np.linspace(w0 * 1.05, 6.0, 20)
    assert np.all(sol.controls(above[:, None])[:, 0] < above)
    # where the limit binds the household would consume more, and the error is zero
    assert np.abs(sol.euler_errors(w[:, None])).max() == 0.0
    assert np.abs(sol.euler_errors()).max() < 5e-3


def test_markov_income_and_refining_from_a_coarse_solution():
    log_y = dsg.rouwenhorst(0.9, 0.1, 3)
    kw = dict(markov={"y": log_y}, tol=1e-11)
    coarse = dsg.egm(saving(markov=True), SAVING, grid={"a": (0.0, 8.0, 20)},
                     initial={"c": lambda s, p: 0.9 * s.w}, **kw)
    cold = dsg.egm(saving(markov=True), SAVING, grid={"a": (0.0, 8.0, 80)},
                   initial={"c": lambda s, p: 0.9 * s.w}, **kw)
    warm = dsg.egm(saving(markov=True), SAVING, grid={"a": (0.0, 8.0, 80)}, initial=coarse,
                   **kw)
    assert cold.converged and warm.converged and warm.iterations < cold.iterations
    assert np.abs(warm.values - cold.values).max() < 1e-9
    # with persistent income, a richer household consumes more at the same cash on hand
    c = cold.controls([[3.0, y] for y in log_y.nodes])[:, 0]
    assert np.all(np.diff(c) > 0)
    assert cold.endogenous.shape == (3, 80) and cold.values.shape == (3, 80, 1)


# -- the solution ---------------------------------------------------------------------------------

def test_simulation_follows_the_policy_and_the_transitions():
    log_y = dsg.rouwenhorst(0.9, 0.1, 3)
    sol = dsg.egm(saving(markov=True), SAVING, grid={"a": (0.0, 8.0, 40)}, markov={"y": log_y},
                  initial={"c": lambda s, p: 0.9 * s.w}, tol=1e-10)
    path = sol.simulate(60, seed=2)
    w, y = path["states"].T
    c, a = path["controls"][:, 0], path["poststates"]
    assert np.allclose(y, log_y.nodes[path["nodes"]])
    assert np.allclose(c, sol.controls(path["states"])[:, 0])
    assert np.allclose(a, w - c)
    assert np.allclose(w[1:], SAVING["R"] * a[:-1] + np.exp(y[1:]))
    iid = solve_saving(40)
    shocks = np.linspace(-0.1, 0.1, 9)
    path = iid.simulate(10, start=[2.0], shocks=shocks)
    assert path["states"][0, 0] == 2.0
    assert np.allclose(path["states"][1:, 0], SAVING["R"] * path["poststates"][:-1]
                       + np.exp(shocks))


def test_scalar_only_functions_are_evaluated_point_by_point():
    scalar = dsg.EGMProgram(
        states=["w"], controls=["c"], poststate="a", expectations=["mr"], shocks=["y"],
        expectation=lambda s, x, e, p: [float(p["beta"] * p["R"] * x.c ** -p["gamma"])],
        direct_response=lambda a, z, p: [float(z.mr ** (-1 / p["gamma"]))],
        reverse_state=lambda a, x, p: {"w": float(a.a + x.c)},
        half_transition=lambda a, e, p: {"w": p["R"] * a.a + math.exp(e.y)},
        post_decision=lambda s, x, p: {"a": float(s.w - x.c)},
        bounds={"c": (0.0, lambda s, p: float(s.w))})
    a = dsg.egm(scalar, SAVING, grid={"a": (0.0, 8.0, 15)}, initial={"c": 0.9},
                quadrature=(NODES, np.full(5, 0.2)), tol=1e-11)
    b = solve_saving(15)
    assert np.allclose(a.values, b.values, atol=1e-12)
    assert np.allclose(a.endogenous, b.endogenous, atol=1e-12)


# -- the declaration ------------------------------------------------------------------------------

def test_misdeclared_programs_are_refused():
    def program(**changes):
        kw = dict(states=["w"], controls=["c"], poststate="a", expectations=["mr"],
                  shocks=["y"], expectation=saving().expectation,
                  direct_response=saving().direct_response,
                  reverse_state=saving().reverse_state, half_transition=saving().half_transition,
                  post_decision=saving().post_decision)
        kw.update(changes)
        return dsg.EGMProgram(**kw)

    grid = {"a": (0.0, 8.0, 10)}
    quad = (NODES, np.full(5, 0.2))
    with pytest.raises(ValueError, match="more than once"):
        program(poststate="w")
    with pytest.raises(KeyError, match="not controls"):
        program(bounds={"w": (0, 1)})
    with pytest.raises(ValueError, match="one axis, the poststate"):
        dsg.egm(program(), SAVING, grid={"w": (0.0, 8.0, 10)}, initial={"c": 0.9},
                quadrature=quad)
    with pytest.raises(ValueError, match="one state without a Markov chain"):
        dsg.egm(program(states=["w", "y2"]), SAVING, grid=grid, initial={"c": 0.9},
                quadrature=quad)
    with pytest.raises(ValueError, match="does not invert"):
        dsg.egm(program(post_decision=lambda s, x, p: s.w - 2 * x.c), SAVING, grid=grid,
                initial={"c": 0.9}, quadrature=quad)
    with pytest.raises(ValueError, match="does not increase"):
        dsg.egm(program(reverse_state=lambda a, x, p: x.c - a.a,
                        post_decision=lambda s, x, p: x.c - s.w), SAVING, grid=grid,
                initial={"c": 0.9}, quadrature=quad)
    with pytest.raises(TypeError, match="initial"):
        dsg.egm(program(), SAVING, grid=grid, initial=0.9, quadrature=quad)
    with pytest.raises(KeyError, match="no value"):
        dsg.egm(program(), SAVING, grid=grid, initial={"x": 0.9}, quadrature=quad)
    with pytest.raises(KeyError, match="gives no value"):
        dsg.egm(program(direct_response=lambda a, z, p: {"x": 1.0}), SAVING, grid=grid,
                initial={"c": 0.9}, quadrature=quad)
