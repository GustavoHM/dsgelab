"""Heterogeneous agents: the household block, the fake-news Jacobians and general equilibrium."""

import numpy as np
import pytest

import dsgelab as dsg
from dsgelab import BlockModel, HetBlock, SimpleBlock, SolvedBlock, simple
from dsgelab.heterogeneous import _Lottery

CALIBRATION = {"eis": 1.0, "delta": 0.025, "alpha": 0.11, "rho_e": 0.9, "sd_e": 0.5, "L": 1.0,
               "nE": 3, "nA": 40, "amin": 0.0, "amax": 50.0}


def household_init(a_grid, e_grid, r, w, eis):
    coh = (1 + r) * a_grid[None, :] + w * e_grid[:, None]
    return (1 + r) * (0.1 * coh) ** (-1 / eis)


def household_step(Va_p, a_grid, e_grid, r, w, beta, eis):
    """The endogenous grid method for a household with CRRA utility and a borrowing limit."""
    c_next = (beta * Va_p) ** (-eis)
    coh = (1 + r) * a_grid[None, :] + w * e_grid[:, None]
    a = np.maximum(dsg.interpolate_rows(c_next + a_grid, coh, a_grid), a_grid[0])
    c = coh - a
    return (1 + r) * c ** (-1 / eis), a, c


def make_grid(rho_e, sd_e, nE, amin, amax, nA):
    chain = dsg.rouwenhorst(rho_e, sd_e * np.sqrt(1 - rho_e ** 2), nE)
    e = np.exp(chain.nodes)
    return e / (chain.stationary @ e), chain.transition, dsg.log_grid(amin, amax, nA)


def household(**options):
    return HetBlock(household_step, ["Va", "a", "c"], backward="Va", policy="a",
                    init=household_init, hetinputs=[(make_grid, ["e_grid", "Pi", "a_grid"])],
                    backward_tol=1e-12, name="household", **options)


@simple("r", "w", "Y")
def firm(K, L, Z, alpha, delta):
    r = alpha * Z * (K(-1) / L) ** (alpha - 1) - delta
    w = (1 - alpha) * Z * (K(-1) / L) ** alpha
    Y = Z * K(-1) ** alpha * L ** (1 - alpha)
    return r, w, Y


@simple("asset_mkt", "goods_mkt")
def mkt_clearing(K, A, Y, C, delta):
    return A - K, Y - C - (K - (1 - delta) * K(-1))


@pytest.fixture(scope="module")
def ks():
    hh = household()
    model = BlockModel([hh, firm, mkt_clearing], name="ks")
    ss = model.steady_state(CALIBRATION, {"beta": 0.98, "Z": 0.85, "K": 3.0},
                            {"r": 0.01, "Y": 1.0, "asset_mkt": 0.0}, solver="hybr")
    return hh, model, ss


# -- grids, interpolation and the lottery ---------------------------------------------------------

def test_log_grid_is_evenly_spaced_in_the_log_of_assets_plus_the_pivot():
    g = dsg.log_grid(0.0, 200.0, 50)
    assert g[0] == 0.0 and g[-1] == pytest.approx(200.0, rel=1e-14)
    assert np.allclose(np.diff(np.log(g + 0.25)), np.log(200.25 / 0.25) / 49, rtol=1e-12)
    assert np.all(np.diff(g) > 0)
    with pytest.raises(ValueError):
        dsg.log_grid(1.0, 0.0, 5)


def test_interpolate_rows_matches_np_interp_inside_and_extrapolates_linearly():
    rng = np.random.default_rng(0)
    x = np.sort(rng.random((3, 20)), axis=1) * 10
    y = rng.random((3, 20))
    q = np.sort(rng.random((3, 15)) * 14 - 2, axis=1)
    out = dsg.interpolate_rows(x, q, y)
    for r in range(3):
        inside = (q[r] >= x[r, 0]) & (q[r] <= x[r, -1])
        assert np.allclose(out[r, inside], np.interp(q[r, inside], x[r], y[r]), atol=1e-14)
        lo = q[r] < x[r, 0]
        slope = (y[r, 1] - y[r, 0]) / (x[r, 1] - x[r, 0])
        assert np.allclose(out[r, lo], y[r, 0] + slope * (q[r, lo] - x[r, 0]))
    # one grid shared by all rows broadcasts
    assert np.allclose(dsg.interpolate_rows(x[0], q, y[0]),
                       np.vstack([dsg.interpolate_rows(x[0], q[r], y[0]) for r in range(3)]))


def test_lottery_keeps_mass_and_the_mean_of_the_policy():
    grid = dsg.log_grid(0.0, 10.0, 30)
    rng = np.random.default_rng(1)
    policy = rng.uniform(0.0, 10.0, (2, 30))
    D = rng.random((2, 30))
    D /= D.sum()
    lot = _Lottery(policy, grid)
    after = lot.forward(D)
    assert after.sum() == pytest.approx(1.0, abs=1e-15)
    assert (after * grid).sum() == pytest.approx((D * policy).sum(), rel=1e-13)
    X = rng.random((2, 30))
    assert np.vdot(after, X) == pytest.approx(np.vdot(D, lot.expectation(X)), rel=1e-13)
    assert np.allclose(lot.matrix() @ D.ravel(), after.ravel(), atol=1e-16)


# -- the household block's steady state -----------------------------------------------------------

def test_steady_state_hits_the_targets_and_the_distribution_is_stationary(ks):
    hh, model, ss = ks
    assert ss["r"] == pytest.approx(0.01, abs=1e-11)
    assert ss["Y"] == pytest.approx(1.0, abs=1e-11)
    assert abs(ss["asset_mkt"]) < 1e-10
    internals = ss.internals["household"]
    D, Dbeg = internals["D"], internals["Dbeg"]
    assert D.sum() == pytest.approx(1.0, abs=1e-14) and np.all(D >= 0) and np.all(Dbeg >= 0)
    assert np.allclose(internals["Pi"].T @ Dbeg, D, atol=1e-16)
    # Dbeg is the fixed point of the policy's lottery after the income shock
    assert np.abs(_Lottery(internals["a"], internals["a_grid"]).forward(D) - Dbeg).max() < 1e-15
    assert ss["A"] == pytest.approx(np.vdot(D, internals["a"]), rel=1e-14)


def test_walras_law_holds_at_the_steady_state(ks):
    # c + a' = (1 + r) a + w e, so C = r A + w and Y - C - delta K = 0 when A = K
    _, _, ss = ks
    assert abs(ss["goods_mkt"]) < 1e-10


def test_a_steady_state_evaluated_again_from_its_internals_is_the_same(ks):
    _, model, ss = ks
    again = model.steady_state(ss)
    for k in ("A", "C", "r", "goods_mkt"):
        assert again[k] == pytest.approx(ss[k], abs=1e-12)


# -- Jacobians ------------------------------------------------------------------------------------

def test_fake_news_jacobians_match_the_direct_ones(ks):
    _, _, ss = ks
    hh = household(h=1e-6)
    T = 25
    J = hh.jacobian(ss, ["r", "w", "beta"], T)
    Jd = hh.jacobian(ss, ["r", "w", "beta"], T, method="direct")
    for O in ("A", "C"):
        for i in ("r", "w", "beta"):
            scale = np.abs(Jd[O][i]).max()
            assert np.abs(J[O][i] - Jd[O][i]).max() < 1e-6 * scale, (O, i)


def test_forward_and_central_differences_agree_to_the_step(ks):
    hh, _, ss = ks
    one = household(twosided=False).jacobian(ss, ["r"], 30)["A"]["r"]
    two = hh.jacobian(ss, ["r"], 30)["A"]["r"]
    assert np.abs(one - two).max() < 1e-3 * np.abs(two).max()


def test_jacobian_columns_sum_to_the_response_to_a_permanent_change(ks):
    # a change in w at every date, from 0 to T - 1, moves C_t by the row sums of J
    hh, _, ss = ks
    T = 40
    J = hh.jacobian(ss, ["w"], T)["C"]["w"]
    h = 1e-6
    up = hh.path(ss, {"w": np.full(T, ss["w"] + h)}, T)["C"]
    dn = hh.path(ss, {"w": np.full(T, ss["w"] - h)}, T)["C"]
    assert np.allclose((up - dn) / (2 * h), J.sum(axis=1), rtol=1e-5, atol=1e-8)


def test_simple_block_jacobian_is_exact(ks):
    _, _, ss = ks
    T = 6
    J = firm.jacobian(ss, ["K", "Z"], T)
    K, Z, a = ss["K"], ss["Z"], ss["alpha"]
    drdK = a * (a - 1) * Z * K ** (a - 2)
    assert np.allclose(J["r"]["K"], drdK * np.eye(T, k=-1), rtol=1e-14, atol=0)
    assert np.allclose(J["Y"]["Z"], K ** a * np.eye(T), rtol=1e-14, atol=0)


def test_simple_block_falls_back_to_central_differences_without_complex_arithmetic():
    def f(x):
        return np.asarray(x(1) ** 2, dtype=float) + x
    block = SimpleBlock(f, "y")
    J = block.jacobian({"x": 2.0}, ["x"], 4)["y"]["x"]
    assert np.allclose(J, np.eye(4) + 4.0 * np.eye(4, k=1), rtol=1e-8)


# -- general equilibrium --------------------------------------------------------------------------

def test_G_solves_the_targets_by_the_chain_rule(ks):
    _, model, ss = ks
    T = 50
    H = model.jacobian(ss, ["K", "Z"], ["asset_mkt"], T)["asset_mkt"]
    G = model.solve_jacobian(ss, ["K"], ["asset_mkt"], ["Z"], T=T)
    assert np.allclose(G["K"]["Z"], -np.linalg.solve(H["K"], H["Z"]), atol=1e-12)
    assert np.abs(G["asset_mkt"]["Z"]).max() < 1e-10
    # Walras's law: with c + a' = (1 + r) a + w e, the goods market clears along the path too
    assert np.abs(G["goods_mkt"]["Z"]).max() < 1e-7
    J = model.jacobian(ss, ["K", "Z"], ["r"], T)["r"]
    assert np.allclose(G["r"]["Z"], J["Z"] + J["K"] @ G["K"]["Z"], atol=1e-12)


def test_linear_response_is_G_times_the_shock(ks):
    _, model, ss = ks
    T = 60
    dZ = 0.01 * 0.8 ** np.arange(T)
    irf = model.impulse_linear(ss, ["K"], ["asset_mkt"], {"Z": dZ})
    G = model.solve_jacobian(ss, ["K"], ["asset_mkt"], ["Z"], T=T)
    for o in ("K", "r", "w", "C"):
        assert np.allclose(irf[o], G[o]["Z"] @ dZ, atol=1e-15)
    assert np.array_equal(irf["Z"], dZ)


def test_nonlinear_transition_approaches_the_linear_one_as_the_square_of_the_shock(ks):
    _, model, ss = ks
    T = 80
    gaps = []
    for size in (1e-2, 1e-3):
        shock = {"Z": size * 0.8 ** np.arange(T)}
        lin = model.impulse_linear(ss, ["K"], ["asset_mkt"], shock)
        nonlin = model.impulse_nonlinear(ss, ["K"], ["asset_mkt"], shock, tol=1e-13)
        assert np.abs(nonlin["asset_mkt"]).max() < 1e-13
        gaps.append(np.abs(nonlin["r"] - lin["r"]).max())
    assert 60 < gaps[0] / gaps[1] < 160


def test_solved_block_gives_the_forward_solution():
    # x = a x(+1) + z, so x_t = sum_k a^k z_{t+k}
    @simple("res")
    def forward(x, z, a):
        return x - a * x(1) - z

    block = SolvedBlock(forward, {"x": 0.0}, ["res"], name="solved")
    assert block.inputs == ["z", "a"] and "x" in block.outputs
    model = BlockModel([block])
    ss = model.steady_state({"z": 1.0, "a": 0.5})
    assert ss["x"] == pytest.approx(2.0, abs=1e-12)
    T = 8
    J = block.jacobian(ss, ["z"], T)["x"]["z"]
    expected = np.triu(0.5 ** (np.arange(T)[None, :] - np.arange(T)[:, None]))
    assert np.allclose(J, expected, atol=1e-12)
    dz = np.zeros(T)
    dz[3] = 0.1
    path = block.path(ss, {"z": ss["z"] + dz}, T)
    assert np.allclose(path["x"] - ss["x"], expected @ dz, atol=1e-12)


def test_steady_state_by_brentq_with_a_bracket(ks):
    @simple("x")
    def square(u):
        return u ** 2

    ss = BlockModel([square]).steady_state({}, {"u": (0.0, 3.0)}, {"x": 2.0})
    assert ss["u"] == pytest.approx(np.sqrt(2.0), abs=1e-12)


# -- the declaration ------------------------------------------------------------------------------

def test_block_inputs_and_outputs():
    hh = household()
    assert hh.outputs == ["A", "C"]
    assert set(hh.inputs) == {"r", "w", "beta", "eis", "rho_e", "sd_e", "nE", "amin", "amax",
                              "nA"}


def test_a_cycle_and_two_makers_of_one_output_are_refused():
    a = SimpleBlock(lambda x: x, "y", name="a")
    b = SimpleBlock(lambda y: y, "x", name="b")
    with pytest.raises(ValueError, match="cycle"):
        BlockModel([a, b])
    c = SimpleBlock(lambda z: z, "y", name="c")
    with pytest.raises(ValueError, match="both"):
        BlockModel([a, c])


def test_an_input_that_moves_the_markov_matrix_is_refused(ks):
    _, _, ss = ks

    def grid_with(rho_e, sd_e, nE, amin, amax, nA, w):
        e, Pi, a = make_grid(rho_e, sd_e, nE, amin, amax, nA)
        return e, (1 - 0.01 * w) * Pi + 0.01 * w * np.eye(int(nE)), a

    hh = HetBlock(household_step, ["Va", "a", "c"], backward="Va", policy="a",
                  init=household_init, hetinputs=[(grid_with, ["e_grid", "Pi", "a_grid"])])
    ss2 = BlockModel([hh]).steady_state(ss)
    with pytest.raises(NotImplementedError, match="Pi"):
        hh.jacobian(ss2, ["w"], 5)
    assert "C" in hh.jacobian(ss2, ["r"], 5)


def test_a_step_that_misses_the_backward_expectation_is_refused():
    with pytest.raises(ValueError, match="Va_p"):
        HetBlock(lambda a_grid, r: (a_grid, a_grid, a_grid), ["Va", "a", "c"], backward="Va",
                 policy="a", init=lambda a_grid: a_grid)
