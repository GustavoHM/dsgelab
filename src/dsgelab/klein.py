"""First-order rational-expectations solver via the generalised Schur (QZ) form.

Canonical form (Klein, 2000)::

    A E_t x_{t+1} = B x_t + C eps_{t+1},      x_t = [k_t ; u_t]

with ``k_t`` (``n_k`` entries) predetermined and ``u_t`` non-predetermined. The
solution returned is::

    u_t     = F k_t
    k_{t+1} = P k_t + Q eps_{t+1}

The Blanchard-Kahn condition -- the number of stable generalised eigenvalues of
the pencil ``(B, A)`` must equal ``n_k`` -- is enforced rather than reported, so
an indeterminate or non-existent equilibrium raises instead of silently
returning a matrix that means nothing.

The decomposition is taken in COMPLEX Schur form. The real form produces 2x2
diagonal blocks whenever generalised eigenvalues come close in modulus, which
mixes the invariant subspaces; that case is not exotic, it is what several
exogenous AR(1) processes with similar persistence look like.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import ordqz

__all__ = ["klein", "KleinSolution", "BlanchardKahnError", "shock_loading"]


class BlanchardKahnError(RuntimeError):
    """Raised when the stable-root count does not match the number of states."""

    def __init__(self, message: str, n_stable: int, n_k: int,
                 eigenvalues: np.ndarray | None = None):
        super().__init__(message)
        self.n_stable = n_stable
        self.n_k = n_k
        self.eigenvalues = eigenvalues


@dataclass
class KleinSolution:
    """Matrices of the first-order solution and the diagnostics behind them."""

    F: np.ndarray                 # jumps on states:  u_t = F k_t
    P: np.ndarray                 # state transition: k_{t+1} = P k_t + Q eps
    Q: np.ndarray | None          # shock loading on the states
    n_stable: int
    n_k: int
    eigenvalues: np.ndarray = field(repr=False)
    #: Rows used to pin the shock loading -- those holding ex post.
    shock_rows: np.ndarray = field(repr=False, default_factory=lambda: np.array([], dtype=int))
    #: How exactly those rows are satisfied by ``Q``; must be at round-off.
    shock_residual: float = 0.0

    @property
    def n_u(self) -> int:
        return self.F.shape[0]

    def companion(self) -> np.ndarray:
        """Transition of the full vector ``x_t = [k_t; u_t]`` implied by (P, F)."""
        n_k, n_u = self.n_k, self.n_u
        M = np.zeros((n_k + n_u, n_k + n_u))
        M[:n_k, :n_k] = self.P
        M[n_k:, :n_k] = self.F @ self.P
        return M


def klein(A, B, n_k: int, C=None, tol: float = 1e-9,
          qz_criterium: float | None = None) -> KleinSolution:
    """Solve ``A E_t x_{t+1} = B x_t + C eps_{t+1}`` for the saddle path.

    Parameters
    ----------
    A, B : (n, n) array_like
        Coefficient matrices. ``A`` is allowed to be singular, which is what
        static equations produce, so nothing here inverts it.
    n_k : int
        Number of predetermined variables, which must occupy the FIRST ``n_k``
        positions of ``x``.
    C : (n, n_eps) array_like, optional
        Loading of the innovations. Supplied, the shock response ``Q`` is
        returned as well.
    tol : float
        Modulus tolerance for counting a generalised eigenvalue as stable.
    qz_criterium : float, optional
        Count a root as stable when its modulus is below this number instead of
        below ``1 - tol``. Dynare's default is ``1 + 1e-6``, which admits unit
        roots: a price LEVEL in a model with an inflation target, a random-walk
        technology. Left at ``None`` the count is strict, and a unit root is a
        Blanchard-Kahn failure rather than a solution with a non-stationary
        variable in it.

    Raises
    ------
    BlanchardKahnError
        If the count of stable roots differs from ``n_k``, or if the stable
        subspace does not span the states (singular ``Z11``).
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if A.shape != B.shape or A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"A and B must be square and conformable; got {A.shape} and {B.shape}")
    n = A.shape[0]
    if not 0 <= n_k <= n:
        raise ValueError(f"n_k={n_k} outside [0, {n}]")

    # Stability is governed by the generalised eigenvalues of the pencil (B, A);
    # order the ones inside the unit circle into the leading block.
    if qz_criterium is None:
        S, T, alpha, beta, Qz, Z = ordqz(B, A, sort="iuc", output="complex")
        cut = 1.0 - tol
    else:
        cut = float(qz_criterium)
        S, T, alpha, beta, Qz, Z = ordqz(
            B, A, sort=lambda a, b: np.abs(a) < cut * np.abs(b), output="complex")
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = np.abs(alpha) / np.where(np.abs(beta) < tol, tol, np.abs(beta))
    n_stable = int(np.sum(lam < cut))

    if n_stable < n_k:
        raise BlanchardKahnError(
            f"no stable equilibrium: {n_stable} stable roots for {n_k} predetermined "
            f"variables (too few stable roots -- the model explodes)",
            n_stable, n_k, np.sort(lam))
    if n_stable > n_k:
        raise BlanchardKahnError(
            f"indeterminate equilibrium: {n_stable} stable roots for {n_k} predetermined "
            f"variables (a continuum of solutions)",
            n_stable, n_k, np.sort(lam))

    Z11 = Z[:n_k, :n_k]
    Z21 = Z[n_k:, :n_k]
    if n_k > 0 and np.linalg.matrix_rank(Z11, tol=1e-10) < n_k:
        raise BlanchardKahnError(
            "Z11 is singular: the stable subspace does not span the state space, so no "
            "state-to-control mapping exists", n_stable, n_k, np.sort(lam))

    if n_k == 0:
        F = np.zeros((n, 0))
        P = np.zeros((0, 0))
        Qm = np.zeros((0, 0 if C is None else np.asarray(C).shape[1]))
        return KleinSolution(F=F, P=P, Q=None if C is None else Qm,
                             n_stable=n_stable, n_k=0, eigenvalues=np.sort(lam))

    Z11i = np.linalg.inv(Z11)
    F = np.real(Z21 @ Z11i)

    # With ordqz(B, A), S is the Schur form of B and T that of A. The recursion is
    # T z_{t+1} = S z_t, hence k_{t+1} = Z11 T11^{-1} S11 Z11^{-1} k_t.
    S11 = S[:n_k, :n_k]
    T11 = T[:n_k, :n_k]
    P = np.real(Z11 @ np.linalg.solve(T11, S11) @ Z11i)

    Qm = None
    shock_rows = np.array([], dtype=int)
    shock_residual = 0.0
    if C is not None:
        C = np.asarray(C, dtype=float)
        if C.ndim == 1:
            C = C[:, None]
        if C.shape[0] != n:
            raise ValueError(f"C must have {n} rows, got {C.shape[0]}")
        Qm, shock_residual, shock_rows = shock_loading(A, C, F, n_k, tol=tol)

    return KleinSolution(F=F, P=P, Q=Qm, n_stable=n_stable, n_k=n_k,
                         eigenvalues=np.sort(lam), shock_rows=shock_rows,
                         shock_residual=shock_residual)


def shock_loading(A, C, F, n_k: int, tol: float = 1e-9):
    """Loading of the innovations on the states: solve for ``Q`` in ``k' = P k + Q eps``.

    This is NOT read off the QZ transformation, and the reason is worth stating,
    because the textbook formula ``Q = Z11 T11^{-1} (Q^H C)_1`` is wrong for a
    large class of models under this package's timing convention.

    That formula solves ``A x_{t+1} = B x_t + C eps`` as though EVERY equation
    held exactly ex post. Here the innovation is dated ``t+1``, so an equation
    containing a next-period JUMP holds only in expectation: the jump responds
    to ``eps_{t+1}``, and ``E_t`` of that response is zero. Imposing such a row
    ex post adds a false restriction on ``Q``, the resulting system is
    inconsistent, and the QZ returns a compromise between the true loading and
    the false restriction. A three-equation New Keynesian model is enough to
    show it: the loading on two independent AR(1) processes must be the
    identity, and the textbook formula returns a dense matrix with off-diagonal
    terms of order 0.12.

    The restriction on ``Q`` comes from the rows that DO hold ex post, which are
    the rows containing no next-period jump: the transition of each exogenous
    process (where the innovation enters) and the accumulation identity of each
    endogenous state (where it does not, forcing that row of ``Q`` to zero).
    Those rows determine ``Q`` exactly, and there are enough of them precisely
    when every predetermined variable has a law of motion -- which is what being
    predetermined means.

    Returns ``(Q, residual, rows)``. A nonzero residual means some equation
    without a next-period jump nevertheless fails to hold ex post; see the error
    message for what that implies.
    """
    A = np.asarray(A, dtype=float)
    C = np.asarray(C, dtype=float)
    n = A.shape[0]
    n_u = n - n_k
    scale = max(float(np.abs(A).max()), 1.0)
    if n_u > 0:
        forward_jump = np.abs(A[:, n_k:]).max(axis=1)
    else:
        forward_jump = np.zeros(n)
    rows = np.where(forward_jump <= tol * scale)[0]

    if rows.size == 0:
        raise BlanchardKahnError(
            "no equation is free of next-period jumps, so no row determines how the "
            "innovations reach the states. Every predetermined variable needs a law of "
            "motion written without a t+1 jump.", 0, n_k)

    M = A[rows][:, :n_k] + A[rows][:, n_k:] @ F      # the second term is zero by construction
    R = C[rows]
    rank = np.linalg.matrix_rank(M, tol=1e-10 * max(scale, 1.0))
    if rank < n_k:
        raise BlanchardKahnError(
            f"the equations without a next-period jump span only {rank} of the {n_k} "
            f"predetermined variables, so the shock loading is not determined. A state "
            f"whose law of motion is missing, or written with a t+1 jump in it, produces "
            f"this.", rank, n_k)

    Q, *_ = np.linalg.lstsq(M, R, rcond=None)
    residual = float(np.max(np.abs(M @ Q - R))) if M.size else 0.0
    limit = 1e-8 * max(float(np.abs(C).max()), 1.0)
    if residual > limit:
        raise BlanchardKahnError(
            f"the ex-post rows are inconsistent (residual {residual:.3e}). This happens "
            f"when an equation contains no next-period jump and yet holds only in "
            f"expectation -- typically one whose only forward term is an EXOGENOUS state, "
            f"such as q_t = beta*f(a_{{t+1}}). Introduce the forward term as a jump with "
            f"its own definition, so that the expectational equations are marked as such.",
            0, n_k)
    return Q, residual, rows
