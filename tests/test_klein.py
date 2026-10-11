"""The QZ solver against a case whose answer is known on paper."""

import numpy as np
import pytest

from dsgelab import BlanchardKahnError, klein


def test_rbc_closed_form():
    """Log utility with full depreciation: P and F are exactly (alpha, rho) and (alpha, 1)."""
    alpha, beta, rho = 0.36, 0.99, 0.95
    A = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    B = np.array([[alpha, 1.0, 0.0], [0.0, rho, 0.0], [alpha, 1.0, -1.0]])
    C = np.array([[0.0], [1.0], [0.0]])
    sol = klein(A, B, n_k=2, C=C)
    assert np.allclose(sol.P, [[alpha, 1.0], [0.0, rho]], atol=1e-10)
    assert np.allclose(sol.F, [[alpha, 1.0]], atol=1e-10)
    assert sol.n_stable == 2
    assert np.allclose(sol.Q, [[0.0], [1.0]], atol=1e-10)


def test_too_few_stable_roots_raises():
    A = np.eye(2)
    B = np.diag([1.5, 2.0])          # both roots outside the unit circle
    with pytest.raises(BlanchardKahnError) as exc:
        klein(A, B, n_k=2)
    assert exc.value.n_stable == 0


def test_too_many_stable_roots_raises():
    A = np.eye(2)
    B = np.diag([0.5, 0.4])
    with pytest.raises(BlanchardKahnError):
        klein(A, B, n_k=1)


def test_singular_A_is_accepted():
    """Static equations make A singular; nothing may invert it."""
    A = np.array([[1.0, 0.0], [0.0, 0.0]])
    B = np.array([[0.5, 0.0], [1.0, -1.0]])
    sol = klein(A, B, n_k=1)
    assert np.allclose(sol.P, [[0.5]])
    assert np.allclose(sol.F, [[1.0]])


def test_shock_loading_uses_only_the_rows_that_hold_ex_post():
    """Rows containing a next-period JUMP hold only in expectation.

    The textbook loading formula, Q = Z11 T11^{-1} (Q^H C)_1, solves the system
    as though every equation held exactly ex post. Under this package's timing
    the innovation is dated t+1, so an equation containing a t+1 jump does not:
    the jump responds to the innovation and E_t of that response is zero.
    Imposing such a row adds a false restriction and the loading comes out
    dense where it should be diagonal.

    Two independent AR(1) processes and one forward-looking equation are enough
    to show it. The loading MUST be the identity.
    """
    rho_a, rho_v, beta = 0.9, 0.5, 0.99
    # x = [a, v, pi]; a and v are AR(1) states, pi a jump with a forward term.
    A = np.array([[1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0],
                  [0.0, 0.0, beta]])
    B = np.array([[rho_a, 0.0, 0.0],
                  [0.0, rho_v, 0.0],
                  [-0.3, -0.2, 1.0]])
    C = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    sol = klein(A, B, n_k=2, C=C)
    assert np.allclose(sol.Q, np.eye(2), atol=1e-12), sol.Q
    assert sol.shock_residual < 1e-12
    assert set(sol.shock_rows.tolist()) == {0, 1}       # the Phillips curve is excluded


def test_a_state_without_a_law_of_motion_is_refused():
    """If no ex-post row mentions a state, its shock loading is not determined."""
    A = np.array([[1.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0],       # this row carries a next-period jump
                  [0.0, 1.0, 0.0]])
    B = np.array([[0.9, 0.0, 0.0],
                  [0.0, 0.5, 0.6],
                  [0.0, 0.0, 1.0]])
    C = np.array([[1.0], [0.0], [0.0]])
    try:
        klein(A, B, n_k=2, C=C)
    except BlanchardKahnError as exc:
        assert "not determined" in str(exc) or "stable" in str(exc)
    except Exception as exc:                     # pragma: no cover - shape guard
        raise AssertionError(f"unexpected error type {type(exc).__name__}: {exc}")
