"""First-order solution of models whose parameters switch with a Markov chain.

The regime ``s_t`` follows a chain with ``transition[i, j]`` the probability of
``s_{t+1} = j`` given ``s_t = i``, and some parameters take a value per regime.
The model is the package's ``f(nxt, cur, eps, params)``, which holds in
expectation over ``s_{t+1}`` and ``eps_{t+1}`` given ``s_t``. A switching
parameter takes the value of the current regime ``s_t``, or, if it is named in
``next_regime``, that of ``s_{t+1}``: the volatility of an innovation dated
``t+1``, or a growth rate that enters an Euler equation at ``t+1``.

Approximation point (Foerster, Rubio-Ramírez, Waggoner and Zha, 2016)
--------------------------------------------------------------------
The steady state is that of the ergodic mean of the switching parameters, and
it is common to all regimes. A switching parameter that moves the steady state
is perturbed, ``theta(s) = theta_bar + chi (theta(s) - theta_bar)``, and enters
the first-order solution as a constant per regime. One that does not move it
keeps its value in each regime, and the coefficients of the solution then differ
across regimes. This is their *partition principle*. Which parameters move the
steady state is found by evaluating the residuals at the steady state with each
parameter at each regime's value, or given in ``perturbed``.

The solution (Farmer, Waggoner and Zha, 2011)
---------------------------------------------
In the rescaled deviations of :mod:`dsgelab.linearize`, the minimum-state-variable
solution is::

    u_t     = F[s_t] k_t + f[s_t]
    k_{t+1} = P[s_t, s_{t+1}] k_t + Q[s_t, s_{t+1}] eps_{t+1} + p[s_t, s_{t+1}]

The rows of the system that contain a state at ``t+1`` and no jump at ``t+1`` are
the laws of motion of the states. They hold for every ``s_{t+1}`` and give
``P``, ``Q`` and ``p`` from ``F`` and ``f``. The other rows hold in expectation,
and they are a system of quadratic matrix equations in ``F[0], ..., F[h-1]``.
Unlike the constant-parameter case, no eigenvalue ordering selects a solution:
the system has in general several, real and complex. It is solved by Newton's
method, as Farmer, Waggoner and Zha (2011) do, from the Klein solution of the
regime-averaged model, from the Klein solution of each regime held forever and,
with ``starts``, from random points. Every distinct solution found is kept.

A solution is accepted when it is mean-square stable (Costa, Fragoso and
Marques, 2005; Farmer, Waggoner and Zha, 2009): the second moments of ``k_t``
stay bounded, which holds when the spectral radius of the operator with blocks
``transition[i, j] * kron(P[i, j], P[i, j])`` is below one. Newton's method from
a finite set of starting points cannot prove that a solution is the only stable
one, which a Gröbner-basis search does (Foerster et al., 2016). The result
therefore reports how many starting points were used, how many converged and
how many stable solutions they found.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike

from .klein import BlanchardKahnError, klein
from .linearize import Coordinates, first_order_system, second_derivatives
from .markov import MarkovChain
from .model import Covariance, Model, Params, Point, get_param, set_params
from .solution import _psd_factor, shock_covariance
from .steady import SteadyState, steady_state

__all__ = ["MarkovSwitchingError", "SwitchingSystem", "MSVSolution", "MarkovSwitchingSolution",
           "switching_system", "msv_solutions", "mean_square_stability",
           "solve_markov_switching", "SecondOrderSwitchingSolution", "SwitchingDecisionRule",
           "solve_second_order_switching"]


class MarkovSwitchingError(RuntimeError):
    """No mean-square stable solution was found, or the system cannot be solved.

    ``solutions`` holds whatever the search found, stable or not.
    """

    def __init__(self, message: str, solutions: Sequence["MSVSolution"] = ()):
        super().__init__(message)
        self.solutions = list(solutions)


def _transition(transition: MarkovChain | ArrayLike) -> np.ndarray:
    T = transition.transition if isinstance(transition, MarkovChain) else transition
    T = np.asarray(T, dtype=float)
    if T.ndim != 2 or T.shape[0] != T.shape[1] or T.shape[0] < 1:
        raise ValueError("transition must be a square matrix, one row and column per regime")
    if np.any(T < 0) or not np.allclose(T.sum(axis=1), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("transition must be nonnegative, with rows that sum to one")
    return T


def _ergodic(T: np.ndarray) -> np.ndarray:
    """The ergodic distribution; with several, the left eigenvector closest to uniform."""
    vals, vecs = np.linalg.eig(T.T)
    k = int(np.argmin(np.abs(vals - 1.0)))
    pi = np.abs(np.real(vecs[:, k]))
    return pi / pi.sum()


# ======================================================================= the system
@dataclass
class SwitchingSystem:
    """``A[i,j] E y' = B[i,j] y + C[i,j] eps' + c[i,j]``, for ``s_t = i`` and ``s_{t+1} = j``.

    The matrices are in the rescaled deviations of :mod:`dsgelab.linearize`,
    with one row scaling common to every pair of regimes, so that rows averaged
    over ``s_{t+1}`` are commensurate. ``motion`` lists the rows that are laws of
    motion of the states, which hold for every ``s_{t+1}``; ``expected`` the
    rest, which hold in expectation. ``row_scale`` is the divisor of each row.
    """

    A: np.ndarray                 # (h, h, n, n)
    B: np.ndarray                 # (h, h, n, n)
    C: np.ndarray                 # (h, h, n, n_eps)
    c: np.ndarray                 # (h, h, n)
    transition: np.ndarray        # (h, h)
    n_k: int
    motion: np.ndarray = field(default=None, repr=False)
    expected: np.ndarray = field(default=None, repr=False)
    row_scale: np.ndarray = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.A = np.asarray(self.A, dtype=float)
        self.B = np.asarray(self.B, dtype=float)
        self.C = np.asarray(self.C, dtype=float)
        self.c = np.asarray(self.c, dtype=float)
        self.transition = _transition(self.transition)
        h, n = self.h, self.n
        self.row_scale = (np.ones(n) if self.row_scale is None
                          else np.asarray(self.row_scale, dtype=float))
        if self.A.shape != (h, h, n, n) or self.B.shape != (h, h, n, n):
            raise ValueError(f"A and B must be ({h}, {h}, n, n), one matrix per pair of regimes")
        if self.C.shape[:3] != (h, h, n) or self.c.shape != (h, h, n):
            raise ValueError("C and c must have one block per pair of regimes and n rows")
        nk = self.n_k
        if not 0 <= nk <= n:
            raise ValueError(f"n_k={nk} outside [0, {n}]")
        scale = max(float(np.abs(self.A).max(initial=0.0)), 1.0)
        jump_next = np.abs(self.A[..., nk:]).max(axis=(0, 1, 3), initial=0.0) > 1e-12 * scale
        state_next = np.abs(self.A[..., :nk]).max(axis=(0, 1, 3), initial=0.0) > 1e-12 * scale
        self.motion = np.flatnonzero(state_next & ~jump_next)
        self.expected = np.flatnonzero(~(state_next & ~jump_next))
        if self.motion.size != nk:
            raise MarkovSwitchingError(
                f"{self.motion.size} equations contain a state at t+1 and no jump at t+1, for "
                f"{nk} states. Each state needs exactly one law of motion, written without a "
                f"t+1 jump, and every other equation that looks forward needs a jump at t+1: "
                f"an equation whose only forward term is an exogenous state, such as "
                f"q_t = beta*f(a_{{t+1}}), should introduce the forward term as a jump.")
        for i in range(h):
            for j in range(h):
                Ak = self.A[i, j][np.ix_(self.motion, np.arange(nk))]
                if nk and np.linalg.matrix_rank(Ak) < nk:
                    raise MarkovSwitchingError(
                        f"the laws of motion do not determine the states at t+1 in regimes "
                        f"({i}, {j}): their matrix has rank {np.linalg.matrix_rank(Ak)} < {nk}")

    @property
    def h(self) -> int:
        return self.transition.shape[0]

    @property
    def n(self) -> int:
        return self.A.shape[2]

    @property
    def n_u(self) -> int:
        return self.n - self.n_k

    def _blocks(self):
        """Per pair of regimes, the pieces of the expected rows and of the laws of motion."""
        nk, L, E = self.n_k, self.motion, self.expected
        out = {}
        for i in range(self.h):
            for j in range(self.h):
                A, B = self.A[i, j], self.B[i, j]
                M = np.linalg.inv(A[np.ix_(L, np.arange(nk))]) if nk else np.zeros((0, 0))
                out[i, j] = dict(
                    AkE=A[E][:, :nk], AuE=A[E][:, nk:], BkE=B[E][:, :nk], BuE=B[E][:, nk:],
                    M=M, BkL=B[L][:, :nk], BuL=B[L][:, nk:], CL=self.C[i, j][L],
                    cL=self.c[i, j][L], cE=self.c[i, j][E])
        return out


def _residual_matrix(model, params, steady, ctx, h_step):
    system = first_order_system(model, steady, params, ctx=ctx, h=h_step, equilibrate=False)
    return system.A, system.B, system.C


def _regime_params(params, switching, names_next, perturbed, means, i, j, chi):
    """Parameters for ``s_t = i`` and ``s_{t+1} = j``, with the perturbed ones scaled by ``chi``."""
    updates = {}
    for name, values in switching.items():
        regime = j if name in names_next else i
        v = float(values[regime])
        if name in perturbed:
            v = means[name] + chi * (v - means[name])
        updates[name] = v
    return set_params(params, updates)


def switching_system(model: Model, params: Params, switching: Mapping[str, Sequence[float]],
                     transition: MarkovChain | ArrayLike, *, steady: SteadyState | None = None,
                     guess: Point | None = None, next_regime: Sequence[str] = (),
                     perturbed: Sequence[str] | None = None, ctx: Any = None,
                     positive: Sequence[str] = (), h: float = 1e-6,
                     tol: float = 1e-11) -> tuple[SwitchingSystem, SteadyState, list[str]]:
    """Linearise ``model`` around the steady state of the ergodic-mean parameters.

    Returns ``(system, steady, perturbed)``. The steady state is computed from
    ``guess`` at the parameters with every switching parameter at its ergodic
    mean, unless ``steady`` is given. ``perturbed`` lists the switching
    parameters scaled by the perturbation parameter; by default, those whose
    regime values leave a nonzero residual at that steady state.
    """
    T = _transition(transition)
    hreg = T.shape[0]
    switching = {k: np.asarray(v, dtype=float).ravel() for k, v in switching.items()}
    for k, v in switching.items():
        if np.size(v) != hreg:
            raise ValueError(f"{k!r} has {np.size(v)} values for {hreg} regimes")
        get_param(params, k)                         # raises if the name is unknown
    names_next = set(next_regime)
    unknown = names_next - set(switching)
    if unknown:
        raise ValueError(f"next_regime names {sorted(unknown)}, which do not switch")
    pi = _ergodic(T)
    means = {k: float(pi @ v) for k, v in switching.items()}
    mean_params = set_params(params, means)
    if steady is None:
        if guess is None:
            raise ValueError("pass either a solved `steady` state or a `guess` for one")
        steady = steady_state(model, guess, mean_params, ctx=ctx, positive=positive, tol=tol)
    ctx = steady.ctx
    x, zero = steady.x, np.zeros(model.n_shocks)

    def resid(p):
        return model.residuals(x, x, zero, p, ctx)

    limit = max(1e-9, 10.0 * float(np.max(np.abs(resid(mean_params)))))
    if perturbed is None:
        perturbed = []
        for k, v in switching.items():
            if any(np.max(np.abs(resid(set_params(mean_params, {k: float(vi)})))) > limit
                   for vi in v):
                perturbed.append(k)
    else:
        perturbed = list(perturbed)
        unknown = set(perturbed) - set(switching)
        if unknown:
            raise ValueError(f"perturbed names {sorted(unknown)}, which do not switch")
    for i in range(hreg):
        for j in range(hreg):
            r = resid(_regime_params(mean_params, switching, names_next, perturbed,
                                     means, i, j, 0.0))
            if np.max(np.abs(r)) > limit:
                raise MarkovSwitchingError(
                    f"the steady state does not solve the model in regimes ({i}, {j}) with the "
                    f"perturbed parameters {perturbed} at their means (residual "
                    f"{np.max(np.abs(r)):.2e}): add to `perturbed` the switching parameters "
                    f"that move the steady state")

    n, ne = model.n_vars, model.n_shocks
    A = np.zeros((hreg, hreg, n, n))
    B = np.zeros_like(A)
    C = np.zeros((hreg, hreg, n, ne))
    c = np.zeros((hreg, hreg, n))
    for i in range(hreg):
        for j in range(hreg):
            p0 = _regime_params(mean_params, switching, names_next, perturbed, means,
                                i, j, 0.0)
            A[i, j], B[i, j], C[i, j] = _residual_matrix(model, p0, steady, ctx, h)
            # The constant is minus the derivative of the residuals with respect to the
            # perturbation parameter, a central difference along the perturbed parameters.
            if perturbed:
                step = 1e-6
                up = _regime_params(mean_params, switching, names_next, perturbed,
                                    means, i, j, step)
                down = _regime_params(mean_params, switching, names_next, perturbed,
                                      means, i, j, -step)
                c[i, j] = -(resid(up) - resid(down)) / (2 * step)
    # One row scaling for every pair of regimes, as in dsgelab.linearize.
    row = np.maximum(np.abs(A), np.abs(B)).max(axis=(0, 1, 3))
    row[row < 1e-12] = 1.0
    A, B = A / row[:, None], B / row[:, None]
    C, c = C / row[:, None], c / row
    return SwitchingSystem(A=A, B=B, C=C, c=c, transition=T, n_k=model.n_states,
                           row_scale=row), steady, perturbed


# ======================================================================= solutions
@dataclass
class MSVSolution:
    """One minimum-state-variable solution, with ``s_t = i`` and ``s_{t+1} = j``.

    ``u_t = F[i] k_t + f[i]`` and ``k_{t+1} = P[i, j] k_t + Q[i, j] eps_{t+1} + p[i, j]``.
    ``radius`` is the spectral radius of the mean-square stability operator, and
    ``residual`` the largest residual of the quadratic system. A complex solution
    (found only with ``complex=True``) has complex arrays.
    """

    F: np.ndarray                 # (h, n_u, n_k)
    f: np.ndarray                 # (h, n_u)
    P: np.ndarray                 # (h, h, n_k, n_k)
    Q: np.ndarray                 # (h, h, n_k, n_eps)
    p: np.ndarray                 # (h, h, n_k)
    radius: float
    residual: float
    iterations: int = 0

    @property
    def stable(self) -> bool:
        """Mean-square stable: real, with a radius below one."""
        return bool(self.radius < 1.0 and np.isrealobj(self.F))

    @property
    def is_real(self) -> bool:
        return bool(np.isrealobj(self.F))


def mean_square_stability(P: ArrayLike, transition: MarkovChain | ArrayLike) -> float:
    """Spectral radius of the mean-square stability operator of ``k' = P[s, s'] k``.

    ``P`` is ``(h, h, n, n)``, one matrix per pair of regimes, or ``(h, n, n)``
    for matrices that depend on ``s'`` only, the form of Costa, Fragoso and
    Marques (2005). The second moments ``M_j = E[k k' 1{s=j}]`` evolve as
    ``M_j' = sum_i transition[i, j] P[i, j] M_i P[i, j]'``, and the solution is
    mean-square stable when the radius of that map is below one.
    """
    T = _transition(transition)
    P = np.asarray(P)
    h = T.shape[0]
    if P.ndim == 3:
        P = np.broadcast_to(P[None], (h,) + P.shape)
    n = P.shape[-1]
    if P.shape != (h, h, n, n):
        raise ValueError(f"P must be ({h}, {h}, n, n) or ({h}, n, n)")
    if n == 0:
        return 0.0
    m = n * n
    if h * m <= 4000:
        Op = np.zeros((h * m, h * m), dtype=P.dtype)
        for i in range(h):
            for j in range(h):
                if T[i, j] != 0.0:
                    Op[j * m:(j + 1) * m, i * m:(i + 1) * m] = T[i, j] * np.kron(P[i, j], P[i, j])
        return float(np.max(np.abs(np.linalg.eigvals(Op))))
    from scipy.sparse.linalg import LinearOperator, eigs

    def apply(v):
        Mi = v.reshape(h, n, n)
        out = np.zeros_like(Mi)
        for i in range(h):
            for j in range(h):
                if T[i, j] != 0.0:
                    out[j] += T[i, j] * P[i, j] @ Mi[i] @ P[i, j].T
        return out.ravel()
    op = LinearOperator((h * m, h * m), matvec=apply, dtype=P.dtype)
    return float(np.max(np.abs(eigs(op, k=1, which="LM", return_eigenvectors=False))))


def _quadratic(system, blocks, F):
    """Residuals ``G[i]`` of the expected rows and the state transitions ``P[i, j]``."""
    h, T = system.h, system.transition
    P = np.empty((h, h, system.n_k, system.n_k), dtype=F.dtype)
    G = np.zeros((h, system.n_u, system.n_k), dtype=F.dtype)
    for i in range(h):
        for j in range(h):
            b = blocks[i, j]
            P[i, j] = b["M"] @ (b["BkL"] + b["BuL"] @ F[i])
            if T[i, j] != 0.0:
                G[i] += T[i, j] * ((b["AkE"] + b["AuE"] @ F[j]) @ P[i, j]
                                   - b["BkE"] - b["BuE"] @ F[i])
    return G, P


def _jacobian(system, blocks, F, P):
    """Derivative of ``vec G`` with respect to ``vec F``, column-major blocks per regime."""
    h, T, nu, nk = system.h, system.transition, system.n_u, system.n_k
    m = nu * nk
    J = np.zeros((h * m, h * m), dtype=F.dtype)
    Ik = np.eye(nk)
    for i in range(h):
        for j in range(h):
            if T[i, j] == 0.0:
                continue
            b = blocks[i, j]
            J[i * m:(i + 1) * m, j * m:(j + 1) * m] += T[i, j] * np.kron(P[i, j].T, b["AuE"])
            inner = (b["AkE"] + b["AuE"] @ F[j]) @ b["M"] @ b["BuL"] - b["BuE"]
            J[i * m:(i + 1) * m, i * m:(i + 1) * m] += T[i, j] * np.kron(Ik, inner)
    return J


def _newton(system, blocks, F0, tol, maxiter):
    F = np.array(F0)
    h, nu, nk = system.h, system.n_u, system.n_k
    for it in range(1, maxiter + 1):
        G, P = _quadratic(system, blocks, F)
        if not np.all(np.isfinite(G)) or np.max(np.abs(F), initial=0.0) > 1e8:
            return None, it
        res = float(np.max(np.abs(G), initial=0.0))
        if res < tol:
            return F, it
        J = _jacobian(system, blocks, F, P)
        try:
            step = np.linalg.solve(J, -np.concatenate([G[i].ravel(order="F") for i in range(h)]))
        except np.linalg.LinAlgError:
            return None, it
        for i in range(h):
            F[i] += step[i * nu * nk:(i + 1) * nu * nk].reshape(nu, nk, order="F")
    G, _ = _quadratic(system, blocks, F)
    return (F, maxiter) if np.max(np.abs(G), initial=0.0) < tol else (None, maxiter)


def _complete(system, blocks, F, iterations) -> MSVSolution:
    """Constants, shock loadings and stability of a solution ``F`` of the quadratic system."""
    h, T, nu, nk = system.h, system.transition, system.n_u, system.n_k
    G, P = _quadratic(system, blocks, F)
    # Constants: the expected rows are linear in f, given F, once p[i, j] is substituted.
    K = np.zeros((h * nu, h * nu), dtype=F.dtype)
    rhs = np.zeros(h * nu, dtype=F.dtype)
    for i in range(h):
        for j in range(h):
            if T[i, j] == 0.0:
                continue
            b = blocks[i, j]
            lead = b["AkE"] + b["AuE"] @ F[j]
            K[i * nu:(i + 1) * nu, i * nu:(i + 1) * nu] += T[i, j] * (
                lead @ b["M"] @ b["BuL"] - b["BuE"])
            K[i * nu:(i + 1) * nu, j * nu:(j + 1) * nu] += T[i, j] * b["AuE"]
            rhs[i * nu:(i + 1) * nu] += T[i, j] * (b["cE"] - lead @ b["M"] @ b["cL"])
    if nu:
        f = np.linalg.solve(K, rhs).reshape(h, nu) if np.any(rhs) else np.zeros((h, nu), F.dtype)
    else:
        f = np.zeros((h, 0), dtype=F.dtype)
    ne = system.C.shape[3]
    Q = np.zeros((h, h, nk, ne))
    p = np.zeros((h, h, nk), dtype=F.dtype)
    for i in range(h):
        for j in range(h):
            b = blocks[i, j]
            Q[i, j] = b["M"] @ b["CL"]
            p[i, j] = b["M"] @ (b["BuL"] @ f[i] + b["cL"])
    radius = mean_square_stability(P, T) if nk else 0.0
    return MSVSolution(F=F, f=f, P=P, Q=Q, p=p, radius=radius,
                       residual=float(np.max(np.abs(G), initial=0.0)), iterations=iterations)


def _klein_start(system, A, B):
    try:
        ks = klein(A, B, system.n_k)
    except (BlanchardKahnError, np.linalg.LinAlgError, ValueError):
        return None
    return ks.F


def msv_solutions(system: SwitchingSystem, *, starts: int = 0, seed: int | None = 0,
                  initial: Sequence[ArrayLike] = (), complex: bool = False,
                  tol: float = 1e-11, maxiter: int = 100) -> tuple[list[MSVSolution], int, int]:
    """Solve the quadratic system of ``system`` by Newton's method from several points.

    The starting points are the ``initial`` arrays given, of shape
    ``(h, n_u, n_k)``, then the Klein solution of the system averaged over the
    regimes (with the ergodic distribution), then the Klein solution of each
    regime held forever, when they exist, then ``starts`` random points around
    the first of these, with standard deviations from one to a hundred times its
    largest entry. With ``complex=True`` the random points are complex, and
    Newton's method in complex arithmetic also finds complex solutions.

    Returns ``(solutions, tried, converged)``: the distinct solutions in the
    order they were found, the number of starting points and how many of them
    converged.
    """
    h, nu, nk = system.h, system.n_u, system.n_k
    T = system.transition
    blocks = system._blocks()
    points = [np.asarray(F0, dtype=np.complex128 if complex else float).reshape(h, nu, nk)
              for F0 in initial]
    pi = _ergodic(T)
    weights = pi[:, None] * T
    mean = _klein_start(system, np.einsum("ij,ijab->ab", weights, system.A),
                        np.einsum("ij,ijab->ab", weights, system.B))
    if mean is not None:
        points.append(np.broadcast_to(mean, (h, nu, nk)).copy())
    for i in range(h):
        Fi = _klein_start(system, system.A[i, i], system.B[i, i])
        if Fi is not None:
            points.append(np.broadcast_to(Fi, (h, nu, nk)).copy())
    if not points:
        points.append(np.zeros((h, nu, nk)))
    base = points[0]
    rng = np.random.default_rng(seed)
    scale = 1.0 + float(np.max(np.abs(base), initial=0.0))
    for _ in range(starts):
        # Spread over two orders of magnitude: solutions far from the Klein ones are not rare.
        spread = scale * 10.0 ** rng.uniform(0.0, 2.0)
        draw = base + spread * rng.standard_normal(base.shape)
        if complex:
            draw = draw + 1j * spread * rng.standard_normal(base.shape)
        points.append(draw)
    found: list[MSVSolution] = []
    converged = 0
    for F0 in points:
        F, it = _newton(system, blocks, F0, tol, maxiter)
        if F is None:
            continue
        converged += 1
        if np.iscomplexobj(F) and np.max(np.abs(F.imag), initial=0.0) < 1e-9 * (
                1.0 + np.max(np.abs(F), initial=0.0)):
            F = F.real.copy()
        size = 1.0 + float(np.max(np.abs(F), initial=0.0))
        if any(s.F.shape == F.shape and np.max(np.abs(s.F - F), initial=0.0) < 1e-7 * size
               for s in found):
            continue
        found.append(_complete(system, blocks, F, it))
    return found, len(points), converged


def _regime_moments(P, p, Q, Sigma, T):
    """Ergodic probabilities and regime-conditional first and second moments of ``k``."""
    h, nk = P.shape[0], P.shape[2]
    pi = _ergodic(T)
    Lq = np.eye(h * nk)
    rq = np.zeros(h * nk)
    for i in range(h):
        for j in range(h):
            Lq[j * nk:(j + 1) * nk, i * nk:(i + 1) * nk] -= T[i, j] * P[i, j]
            rq[j * nk:(j + 1) * nk] += T[i, j] * pi[i] * p[i, j]
    q = np.linalg.solve(Lq, rq).reshape(h, nk) if nk else np.zeros((h, 0))
    rhs = np.zeros((h, nk, nk))
    for i in range(h):
        for j in range(h):
            a = P[i, j] @ np.outer(q[i], p[i, j])
            rhs[j] += T[i, j] * (a + a.T + pi[i] * (np.outer(p[i, j], p[i, j])
                                                    + Q[i, j] @ Sigma @ Q[i, j].T))
    m = nk * nk
    M = np.zeros((h, nk, nk))
    if nk and h * m <= 4000:
        L = np.eye(h * m)
        for i in range(h):
            for j in range(h):
                L[j * m:(j + 1) * m, i * m:(i + 1) * m] -= T[i, j] * np.kron(P[i, j], P[i, j])
        M = np.linalg.solve(L, rhs.reshape(-1)).reshape(h, nk, nk)
    elif nk:
        # Too large for the dense operator: iterate the recursion, which converges
        # geometrically at the rate of the stability radius.
        for _ in range(100000):
            new = rhs + np.einsum("ij,ijab,ibc,ijdc->jad", T, P, M, P)
            if np.max(np.abs(new - M)) < 1e-15 * max(1.0, np.max(np.abs(new))):
                M = new
                break
            M = new
    M = 0.5 * (M + np.transpose(M, (0, 2, 1)))
    w = np.where(pi > 0, pi, 1.0)
    mean = np.where(pi[:, None] > 0, q / w[:, None], 0.0)
    second = np.where(pi[:, None, None] > 0, M / w[:, None, None], 0.0)
    return pi, mean, second


# ================================================================ solution object
@dataclass
class MarkovSwitchingSolution:
    """A mean-square stable first-order solution of a Markov-switching model.

    ``solution`` is the solution used, the first stable one found. ``solutions``
    holds every distinct solution the search found, stable or not, ``tried`` the
    number of starting points and ``converged`` how many of them converged.
    """

    model: Model
    steady: SteadyState
    system: SwitchingSystem
    solution: MSVSolution
    solutions: list
    params: Params
    switching: dict
    next_regime: list
    perturbed: list
    tried: int = 0
    converged: int = 0

    # -- shorthands ------------------------------------------------------
    @property
    def transition(self) -> np.ndarray:
        return self.system.transition

    @property
    def h(self) -> int:
        return self.system.h

    @property
    def F(self) -> np.ndarray:
        return self.solution.F

    @property
    def f(self) -> np.ndarray:
        return self.solution.f

    @property
    def P(self) -> np.ndarray:
        return self.solution.P

    @property
    def Q(self) -> np.ndarray:
        return self.solution.Q

    @property
    def p(self) -> np.ndarray:
        return self.solution.p

    @property
    def coords(self) -> Coordinates:
        return Coordinates.from_steady_state(self.steady)

    @property
    def stable_solutions(self) -> list:
        return [s for s in self.solutions if s.stable]

    @property
    def ergodic(self) -> np.ndarray:
        """Ergodic distribution of the regimes."""
        return _ergodic(self.transition)

    def regime_params(self, regime: int) -> Params:
        """The parameters of ``regime``, with every switching parameter at its value there."""
        return set_params(self.params, {k: float(v[regime]) for k, v in self.switching.items()})

    # -- paths -----------------------------------------------------------
    def variables(self, k: np.ndarray, regimes: ArrayLike,
                  constants: bool = True) -> dict[str, np.ndarray]:
        """Deviation paths of every variable from a state path and a regime path."""
        regimes = np.asarray(regimes, dtype=int)
        u = np.einsum("tab,tb->ta", self.F[regimes], k)
        if constants:
            u = u + self.f[regimes]
        out = {n: k[:, i] for i, n in enumerate(self.model.states)}
        out.update({n: u[:, i] for i, n in enumerate(self.model.jumps)})
        return out

    def to_levels(self, paths: Mapping[str, ArrayLike]) -> dict[str, np.ndarray]:
        """Convert deviation paths to levels using the steady state and the scaling."""
        c = self.coords
        out = {}
        for name, dev in paths.items():
            if name == "regime":
                out[name] = np.asarray(dev)
                continue
            i = self.model.position(name)
            out[name] = c.center[i] + c.scale[i] * np.asarray(dev)
        return out

    def irf(self, name: str, size: float = 1.0, periods: int = 40, *,
            regime: int | Sequence[int] = 0, expected: bool = False, levels: bool = False,
            percent: bool = True) -> dict[str, np.ndarray]:
        """Impulse response of every variable to an innovation of ``size`` in shock ``name``.

        ``regime`` is a regime held for the whole response, or a path of
        ``periods`` regimes; the impulse arrives in the first, with the regime of
        the period before equal to it. With ``expected=True`` and an integer
        ``regime``, the response is averaged over the regimes that follow the
        first, with the chain's probabilities. The response is the difference
        between the paths with and without the innovation, so the regime
        constants cancel. Units as in :meth:`dsgelab.FirstOrderSolution.irf`.
        """
        e = np.zeros(self.model.n_shocks)
        e[self.model.shock_position(name)] = size
        h, nk = self.h, self.model.n_states
        if expected:
            if not np.isscalar(regime):
                raise ValueError("expected=True needs one starting regime")
            r = int(np.asarray(regime))
            v = np.zeros((h, nk))
            v[r] = self.Q[r, r] @ e
            k = np.zeros((periods, nk))
            u = np.zeros((periods, self.model.n_jumps))
            for t in range(periods):
                k[t] = v.sum(axis=0)
                u[t] = np.einsum("iab,ib->a", self.F, v)
                v = np.einsum("ij,ijab,ib->ja", self.transition, self.P, v)
            paths = {n: k[:, i] for i, n in enumerate(self.model.states)}
            paths.update({n: u[:, i] for i, n in enumerate(self.model.jumps)})
        else:
            path = (np.full(periods, int(np.asarray(regime))) if np.isscalar(regime)
                    else np.asarray(regime, dtype=int))
            if path.size != periods:
                raise ValueError(f"the regime path has {path.size} entries for {periods} periods")
            if np.any(path < 0) or np.any(path >= h):
                raise ValueError(f"regimes run from 0 to {h - 1}")
            k = np.zeros((periods, nk))
            if periods:
                k[0] = self.Q[path[0], path[0]] @ e
            for t in range(1, periods):
                k[t] = self.P[path[t - 1], path[t]] @ k[t - 1]
            paths = self.variables(k, path, constants=False)
        if levels:
            c = self.coords
            return {n: c.center[self.model.position(n)] + c.scale[self.model.position(n)] * v
                    for n, v in paths.items()}
        return {n: v * (100.0 if percent else 1.0) for n, v in paths.items()}

    def simulate(self, periods: int = 1000, cov: Covariance = None, *, seed: int | None = None,
                 burn: int = 100, levels: bool = False,
                 regimes: ArrayLike | None = None) -> dict[str, np.ndarray]:
        """Simulate regimes and variables; the regime path is under ``"regime"``.

        The regimes are drawn from the chain, starting from its ergodic
        distribution, unless a path of ``periods + burn`` regimes is given.
        ``burn`` periods are discarded.
        """
        rng = np.random.default_rng(seed)
        total = periods + burn
        if regimes is None:
            chain = MarkovChain(np.arange(self.h, dtype=float), self.transition)
            path = chain.simulate(total, seed=int(rng.integers(2 ** 32)), indices=True)
        else:
            path = np.asarray(regimes, dtype=int)
            if path.size != total:
                raise ValueError(f"the regime path needs {total} entries, periods + burn")
        Sigma = shock_covariance(self.model.shocks, cov)
        eps = rng.standard_normal((total, self.model.n_shocks)) @ _psd_factor(Sigma).T
        k = np.zeros((total, self.model.n_states))
        for t in range(1, total):
            i, j = path[t - 1], path[t]
            k[t] = self.P[i, j] @ k[t - 1] + self.Q[i, j] @ eps[t] + self.p[i, j]
        paths = self.variables(k[burn:], path[burn:])
        paths["regime"] = path[burn:]
        return self.to_levels(paths) if levels else paths

    # -- moments ---------------------------------------------------------
    def regime_moments(self, cov: Covariance = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(pi, mean, second)``: the ergodic probabilities, ``E[k | s=i]`` and ``E[k k' | s=i]``.

        Fixed points of the recursions ``q_j = sum_i T[i,j] (P[i,j] q_i + pi_i p[i,j])`` and
        ``M_j = sum_i T[i,j] (P M_i P' + P q_i p' + p q_i' P' + pi_i (p p' + Q Sigma Q'))``
        for ``q_i = pi_i E[k | s=i]`` and ``M_i = pi_i E[k k' | s=i]``.
        """
        if not self.solution.stable:
            raise MarkovSwitchingError("moments need a mean-square stable solution")
        Sigma = shock_covariance(self.model.shocks, cov)
        return _regime_moments(self.P, self.p, self.Q, Sigma, self.transition)

    def moments(self, cov: Covariance = None) -> dict[str, Any]:
        """Unconditional mean, covariance and standard deviations of every variable.

        Exact under the ergodic distribution of the regimes and of the states,
        from :meth:`regime_moments`. ``conditional_mean[i]`` is the mean of every
        variable given ``s_t = i``. Deviations are in the rescaled units.
        """
        pi, mean_k, second_k = self.regime_moments(cov)
        nk, h, n = self.model.n_states, self.h, self.model.n_vars
        cond = np.zeros((h, n))
        second = np.zeros((n, n))
        for i in range(h):
            # Given s_t = i, x_t = J k_t + a with J = [I; F_i] and a = [0; f_i].
            J = np.vstack([np.eye(nk), self.F[i]])
            a = np.concatenate([np.zeros(nk), self.f[i]])
            cond[i] = J @ mean_k[i] + a
            cross = np.outer(J @ mean_k[i], a)
            second += pi[i] * (J @ second_k[i] @ J.T + cross + cross.T + np.outer(a, a))
        mean = pi @ cond
        cov_x = second - np.outer(mean, mean)
        cov_x = 0.5 * (cov_x + cov_x.T)
        names = self.model.names
        sd = np.sqrt(np.maximum(np.diag(cov_x), 0.0))
        return dict(names=names, mean=mean, cov=cov_x, std=sd, std_by_name=dict(zip(names, sd)),
                    conditional_mean=cond, probabilities=pi)

    # -- diagnostics -----------------------------------------------------
    def summary(self) -> str:
        stable = self.stable_solutions
        lines = [
            f"{self.model.name}: {self.model.n_states} states, {self.model.n_jumps} jumps, "
            f"{self.model.n_shocks} shocks, {self.h} regimes",
            f"  switching parameters  : {sorted(self.switching)}"
            + (f" (dated t+1: {sorted(self.next_regime)})" if self.next_regime else ""),
            f"  perturbed             : {sorted(self.perturbed) or 'none'}",
            f"  steady-state residual : {self.steady.residual:.3e}",
            f"  Newton starts         : {self.tried} tried, {self.converged} converged",
            f"  solutions found       : {len(self.solutions)}, of which {len(stable)} "
            f"mean-square stable",
            f"  stability radius      : {self.solution.radius:.6f}",
        ]
        if len(stable) > 1:
            lines.append("  several stable solutions: the first found is used")
        return "\n".join(lines)


def solve_markov_switching(model: Model, params: Params,
                           switching: Mapping[str, Sequence[float]],
                           transition: MarkovChain | ArrayLike, *,
                           steady: SteadyState | None = None, guess: Point | None = None,
                           next_regime: Sequence[str] = (), perturbed: Sequence[str] | None = None,
                           starts: int = 0, seed: int | None = 0,
                           initial: Sequence[ArrayLike] = (), ctx: Any = None,
                           positive: Sequence[str] = (), h: float = 1e-6, tol: float = 1e-11,
                           maxiter: int = 100, order: int = 1, cov: Covariance = None
                           ) -> MarkovSwitchingSolution | SecondOrderSwitchingSolution:
    """Solve a Markov-switching model to first or second order.

    Parameters
    ----------
    model, params
        The model and its parameters; ``params`` holds the parameters that do
        not switch.
    switching : mapping
        ``{name: (value in regime 0, value in regime 1, ...)}``.
    transition : (h, h) array or MarkovChain
        ``transition[i, j]`` is the probability of moving from regime ``i`` to ``j``.
    steady, guess
        A steady state at the ergodic-mean parameters, or a guess for one.
    next_regime : sequence of str
        Switching parameters that take the value of the regime at ``t+1``.
    perturbed : sequence of str, optional
        Switching parameters that move the steady state, which are perturbed;
        found from the residuals by default.
    starts : int
        Random starting points for Newton's method, besides the Klein
        solutions; with more of them, a second stable solution is more likely to
        be found if one exists.
    initial : sequence of arrays
        Starting points of shape ``(h, n_jumps, n_states)`` tried first.
    order : int
        1 for the first-order solution, a :class:`MarkovSwitchingSolution`; 2
        for :func:`solve_second_order_switching` around it, a
        :class:`SecondOrderSwitchingSolution`.
    cov : mapping, array or None
        Shock covariance of the second-order risk terms; unused at first order.

    Raises
    ------
    MarkovSwitchingError
        When Newton's method converges from no starting point, or when none of
        the solutions it finds is mean-square stable. ``solutions`` on the
        exception holds what was found.
    """
    if order not in (1, 2):
        raise ValueError(f"order must be 1 or 2, not {order!r}")
    system, steady, perturbed = switching_system(
        model, params, switching, transition, steady=steady, guess=guess,
        next_regime=next_regime, perturbed=perturbed, ctx=ctx, positive=positive, h=h)
    found, tried, converged = msv_solutions(system, starts=starts, seed=seed, initial=initial,
                                            tol=tol, maxiter=maxiter)
    if not found:
        raise MarkovSwitchingError(
            f"Newton's method converged from none of {tried} starting points; pass more "
            f"`starts`, or starting points in `initial`")
    stable = [s for s in found if s.stable]
    if not stable:
        radii = ", ".join(f"{s.radius:.4g}" for s in found)
        raise MarkovSwitchingError(
            f"none of the {len(found)} solutions found from {tried} starting points is "
            f"mean-square stable (stability radii {radii})", found)
    first = MarkovSwitchingSolution(
        model=model, steady=steady, system=system, solution=stable[0], solutions=found,
        params=params, switching={k: np.asarray(v, dtype=float).ravel()
                                  for k, v in switching.items()},
        next_regime=list(next_regime), perturbed=perturbed, tried=tried, converged=converged)
    return first if order == 1 else solve_second_order_switching(first, cov)



# ==================================================================== second order
def _pair_hessian(model, steady, params, switching, names_next, perturbed, means, i, j, row, h):
    """Hessian of the residuals in ``[y'; y; eps; chi]`` for ``s_t = i``, ``s_{t+1} = j``.

    In the rescaled deviations and divided by the row scaling, as the first-order
    matrices are. The block in ``[y'; y; eps]`` is :func:`second_derivatives` at the
    parameters of the pair with ``chi = 0``. The derivatives along ``chi`` move the
    perturbed parameters and are taken by central differences with the same step.
    """
    n, ne = model.n_vars, model.n_shocks
    m = 2 * n + ne
    p0 = _regime_params(params, switching, names_next, perturbed, means, i, j, 0.0)
    H = np.zeros((n, m + 1, m + 1))
    H[:, :m, :m] = second_derivatives(model, steady, p0, ctx=steady.ctx, h=h, row_scale=row)
    if not perturbed:
        return H
    coords = Coordinates.from_steady_state(steady)

    def F(w, chi):
        p = _regime_params(params, switching, names_next, perturbed, means, i, j, chi)
        return model.residuals(coords.to_level(w[:n]), coords.to_level(w[n:2 * n]), w[2 * n:],
                               p, steady.ctx) / row

    zero = np.zeros(m)
    H[:, m, m] = (F(zero, h) - 2.0 * F(zero, 0.0) + F(zero, -h)) / (h * h)
    scale = np.concatenate([coords.scale, coords.scale, np.ones(ne)])

    def J(chi):
        p = _regime_params(params, switching, names_next, perturbed, means, i, j, chi)
        exact = model.jacobian(coords.center, coords.center, np.zeros(ne), p, steady.ctx)
        return None if exact is None else exact * scale[None, :] / row[:, None]

    if J(0.0) is not None:
        # With exact first derivatives, one difference along chi, with a first-derivative step.
        step = 1e-6
        val = (J(step) - J(-step)) / (2.0 * step)
        H[:, :m, m] = val
        H[:, m, :m] = val
        return H
    for a in range(m):
        e = np.zeros(m)
        e[a] = h
        val = (F(e, h) - F(e, -h) - F(-e, h) + F(-e, -h)) / (4.0 * h * h)
        H[:, a, m] = val
        H[:, m, a] = val
    return H


@dataclass
class SwitchingDecisionRule:
    """Second-order rule for ``[k_{t+1}; u_{t+1}]`` given ``s_t`` and ``s_{t+1}``.

    The arguments are the states ``k_t``, the innovation ``eps_{t+1}`` and the
    perturbation parameter ``chi``, which is one in the model and is kept as an
    argument, as Foerster, Rubio-Ramírez, Waggoner and Zha (2016) keep it. The
    rule is ``g_k k + g_u eps + g_s + 1/2 g_kk[k, k] + g_ku[k, eps] + 1/2 g_uu[eps, eps]
    + g_ks k + g_us eps + 1/2 g_ss``. Rows follow ``model.states + model.jumps``.
    """

    g_k: np.ndarray             # (n, n_k)
    g_u: np.ndarray             # (n, n_eps)
    g_s: np.ndarray             # (n,)
    g_kk: np.ndarray            # (n, n_k, n_k)
    g_ku: np.ndarray            # (n, n_k, n_eps)
    g_uu: np.ndarray            # (n, n_eps, n_eps)
    g_ks: np.ndarray            # (n, n_k)
    g_us: np.ndarray            # (n, n_eps)
    g_ss: np.ndarray            # (n,)


@dataclass
class SecondOrderSwitchingSolution:
    """Second-order solution of a Markov-switching model, around a first-order one.

    With ``i = s_t``, ``j = s_{t+1}``, in the rescaled deviations and at ``chi = 1``::

        u_t     = F[i] k + f[i] + 1/2 g_kk[i](k, k) + g_ks[i] k + 1/2 g_ss[i]
        k_{t+1} = P[i,j] k + Q[i,j] eps + p[i,j] + 1/2 h_kk[i,j](k, k) + h_ku[i,j](k, eps)
                  + 1/2 h_uu[i,j](eps, eps) + h_ks[i,j] k + h_us[i,j] eps + 1/2 h_ss[i,j]

    with ``k = k_t`` and ``eps = eps_{t+1}``. The index ``s`` stands for the
    perturbation parameter ``chi``, which scales the innovations and the perturbed
    switching parameters. ``g_ss`` and ``h_ss`` are its full second derivatives,
    the convention of Dynare's ``g_s2``, and they depend on ``Sigma``; the other
    blocks do not. ``residual`` is the largest residual of the second-order
    conditions of the expected rows, recomputed from the solution.
    """

    first: MarkovSwitchingSolution
    g_kk: np.ndarray            # (h, n_u, n_k, n_k)
    g_ks: np.ndarray            # (h, n_u, n_k)
    g_ss: np.ndarray            # (h, n_u)
    h_kk: np.ndarray            # (h, h, n_k, n_k, n_k)
    h_ku: np.ndarray            # (h, h, n_k, n_k, n_eps)
    h_uu: np.ndarray            # (h, h, n_k, n_eps, n_eps)
    h_ks: np.ndarray            # (h, h, n_k, n_k)
    h_us: np.ndarray            # (h, h, n_k, n_eps)
    h_ss: np.ndarray            # (h, h, n_k)
    Sigma: np.ndarray = field(repr=False)
    residual: float = 0.0
    step: float = 1e-4

    # -- shorthands ------------------------------------------------------
    @property
    def model(self) -> Model:
        return self.first.model

    @property
    def h(self) -> int:
        return self.first.h

    @property
    def transition(self) -> np.ndarray:
        return self.first.transition

    @property
    def steady(self) -> SteadyState:
        return self.first.steady

    @property
    def F(self) -> np.ndarray:
        return self.first.F

    @property
    def f(self) -> np.ndarray:
        return self.first.f

    @property
    def P(self) -> np.ndarray:
        return self.first.P

    @property
    def Q(self) -> np.ndarray:
        return self.first.Q

    @property
    def p(self) -> np.ndarray:
        return self.first.p

    def to_levels(self, paths: Mapping[str, ArrayLike]) -> dict[str, np.ndarray]:
        """Convert deviation paths to levels using the steady state and the scaling."""
        return self.first.to_levels(paths)

    def at_covariance(self, cov: Covariance) -> "SecondOrderSwitchingSolution":
        """This solution if ``cov`` is the covariance it was computed at, else one solved at ``cov``."""
        if cov is None:
            return self
        Sigma = shock_covariance(self.model.shocks, cov)
        if Sigma.shape == self.Sigma.shape and np.allclose(Sigma, self.Sigma, rtol=1e-12, atol=0.0):
            return self
        return solve_second_order_switching(self.first, Sigma, h=self.step)

    # -- the rules -------------------------------------------------------
    def jumps(self, k: ArrayLike, regime: int) -> np.ndarray:
        """``u_t`` given ``k_t`` and ``s_t = regime``, without pruning."""
        k, i = np.asarray(k, dtype=float), int(regime)
        return (self.F[i] @ k + self.f[i] + 0.5 * np.einsum("rab,a,b->r", self.g_kk[i], k, k)
                + self.g_ks[i] @ k + 0.5 * self.g_ss[i])

    def next_states(self, k: ArrayLike, eps: ArrayLike | None, regime: int,
                    next_regime: int) -> np.ndarray:
        """``k_{t+1}`` given ``k_t``, ``eps_{t+1}``, ``s_t = regime`` and ``s_{t+1} = next_regime``."""
        k = np.asarray(k, dtype=float)
        e = np.zeros(self.model.n_shocks) if eps is None else np.asarray(eps, dtype=float)
        i, j = int(regime), int(next_regime)
        return (self.P[i, j] @ k + self.Q[i, j] @ e + self.p[i, j]
                + 0.5 * np.einsum("rab,a,b->r", self.h_kk[i, j], k, k)
                + np.einsum("rab,a,b->r", self.h_ku[i, j], k, e)
                + 0.5 * np.einsum("rab,a,b->r", self.h_uu[i, j], e, e)
                + self.h_ks[i, j] @ k + self.h_us[i, j] @ e + 0.5 * self.h_ss[i, j])

    def _g_full(self, i):
        """Second derivatives of the jumps in ``(k, chi)``: ``(n_u, n_k + 1, n_k + 1)``."""
        nk = self.model.n_states
        G = np.zeros((self.model.n_jumps, nk + 1, nk + 1))
        G[:, :nk, :nk] = self.g_kk[i]
        G[:, :nk, nk] = self.g_ks[i]
        G[:, nk, :nk] = self.g_ks[i]
        G[:, nk, nk] = self.g_ss[i]
        return G

    def _h_full(self, i, j):
        """Second derivatives of the states in ``(k, eps, chi)``: ``(n_k, n_z, n_z)``."""
        nk, ne = self.model.n_states, self.model.n_shocks
        nz = nk + ne + 1
        kz, ez, c = slice(0, nk), slice(nk, nk + ne), nz - 1
        H = np.zeros((nk, nz, nz))
        H[:, kz, kz] = self.h_kk[i, j]
        H[:, kz, ez] = self.h_ku[i, j]
        H[:, ez, kz] = np.transpose(self.h_ku[i, j], (0, 2, 1))
        H[:, ez, ez] = self.h_uu[i, j]
        H[:, kz, c] = self.h_ks[i, j]
        H[:, c, kz] = self.h_ks[i, j]
        H[:, ez, c] = self.h_us[i, j]
        H[:, c, ez] = self.h_us[i, j]
        H[:, c, c] = self.h_ss[i, j]
        return H

    def decision_rule(self, regime: int, next_regime: int) -> SwitchingDecisionRule:
        """The rule for ``[k_{t+1}; u_{t+1}]`` given ``s_t = regime`` and ``s_{t+1} = next_regime``.

        The states and jumps of ``t+1`` as functions of ``(k_t, eps_{t+1}, chi)``,
        Dynare's decision-rule form with the perturbation parameter kept as an
        argument, which is the form in which Foerster et al. (2016) print their
        solutions.
        """
        i, j = int(regime), int(next_regime)
        nk, ne = self.model.n_states, self.model.n_shocks
        nz = nk + ne + 1
        kz, ez, c = slice(0, nk), slice(nk, nk + ne), nz - 1
        D = np.hstack([self.P[i, j], self.Q[i, j], self.p[i, j][:, None]])    # d k'/dz
        Y = np.vstack([D, np.eye(nz)[c][None]])                              # d (k', chi)/dz
        Hk = self._h_full(i, j)
        Hu = (np.einsum("ap,rab,bq->rpq", Y, self._g_full(j), Y)
              + np.einsum("ra,apq->rpq", self.F[j], Hk))
        Du = self.F[j] @ D
        Du[:, c] += self.f[j]
        Dy, Hy = np.vstack([D, Du]), np.concatenate([Hk, Hu])
        return SwitchingDecisionRule(
            g_k=Dy[:, kz], g_u=Dy[:, ez], g_s=Dy[:, c], g_kk=Hy[:, kz, kz], g_ku=Hy[:, kz, ez],
            g_uu=Hy[:, ez, ez], g_ks=Hy[:, kz, c], g_us=Hy[:, ez, c], g_ss=Hy[:, c, c])

    # -- paths -----------------------------------------------------------
    def _path(self, k0, path, eps, pruning=True):
        """States and jumps along a regime path, from ``k0`` in the period before the first.

        ``path[t]`` is the regime of period ``t``, and the regime of the period
        before the first is taken equal to ``path[0]``. ``eps[t]`` arrives in
        period ``t``. With pruning (Kim, Kim, Schaumburg and Sims, 2008; Andreasen,
        Fernández-Villaverde and Rubio-Ramírez, 2018) the first-order part, with
        its regime constants, feeds the quadratic terms, which do not feed
        themselves.
        """
        periods, nk = len(path), self.model.n_states
        kf = np.zeros((periods, nk))
        ks = np.zeros((periods, nk))
        u = np.zeros((periods, self.model.n_jumps))
        f_prev, s_prev = np.asarray(k0, dtype=float), np.zeros(nk)
        for t in range(periods):
            i, j, e = (path[t - 1] if t else path[0]), path[t], eps[t]
            if pruning:
                kf[t] = self.P[i, j] @ f_prev + self.Q[i, j] @ e + self.p[i, j]
                ks[t] = (self.P[i, j] @ s_prev
                         + 0.5 * np.einsum("rab,a,b->r", self.h_kk[i, j], f_prev, f_prev)
                         + np.einsum("rab,a,b->r", self.h_ku[i, j], f_prev, e)
                         + 0.5 * np.einsum("rab,a,b->r", self.h_uu[i, j], e, e)
                         + self.h_ks[i, j] @ f_prev + self.h_us[i, j] @ e
                         + 0.5 * self.h_ss[i, j])
                u[t] = (self.F[j] @ (kf[t] + ks[t]) + self.f[j]
                        + 0.5 * np.einsum("rab,a,b->r", self.g_kk[j], kf[t], kf[t])
                        + self.g_ks[j] @ kf[t] + 0.5 * self.g_ss[j])
                f_prev, s_prev = kf[t], ks[t]
            else:
                kf[t] = self.next_states(f_prev, e, i, j)
                u[t] = self.jumps(kf[t], j)
                f_prev = kf[t]
        return kf + ks, u

    def _named(self, k, u):
        out = {n: k[:, i] for i, n in enumerate(self.model.states)}
        out.update({n: u[:, i] for i, n in enumerate(self.model.jumps)})
        return out

    def simulate(self, periods: int = 1000, cov: Covariance = None, *, seed: int | None = None,
                 burn: int = 100, levels: bool = False, regimes: ArrayLike | None = None,
                 pruning: bool = True) -> dict[str, np.ndarray]:
        """Simulate regimes and variables with the second-order rules, pruned by default.

        The draws are those of :meth:`MarkovSwitchingSolution.simulate` with the
        same ``seed``: the regimes from the chain from its ergodic distribution,
        unless a path of ``periods + burn`` regimes is given, and the innovations.
        The simulation starts at the steady state and ``burn`` periods are
        discarded. Without pruning the quadratic terms feed on themselves.
        """
        other = self.at_covariance(cov)
        if other is not self:
            return other.simulate(periods, seed=seed, burn=burn, levels=levels, regimes=regimes,
                                  pruning=pruning)
        rng = np.random.default_rng(seed)
        total = periods + burn
        if regimes is None:
            chain = MarkovChain(np.arange(self.h, dtype=float), self.transition)
            path = chain.simulate(total, seed=int(rng.integers(2 ** 32)), indices=True)
        else:
            path = np.asarray(regimes, dtype=int)
            if path.size != total:
                raise ValueError(f"the regime path needs {total} entries, periods + burn")
        eps = rng.standard_normal((total, self.model.n_shocks)) @ _psd_factor(self.Sigma).T
        k, u = self._path(np.zeros(self.model.n_states), path, eps, pruning)
        paths = self._named(k[burn:], u[burn:])
        paths["regime"] = path[burn:]
        return self.to_levels(paths) if levels else paths

    def irf(self, name: str, size: float = 1.0, periods: int = 40, *,
            regime: int | Sequence[int] = 0, state: ArrayLike | None = None,
            pruning: bool = True, levels: bool = False,
            percent: bool = True) -> dict[str, np.ndarray]:
        """Impulse response to an innovation of ``size`` in shock ``name``, along a regime path.

        ``regime`` is a regime held for the whole response, or a path of
        ``periods`` regimes; the impulse arrives in the first, with the regime of
        the period before equal to it, as in :meth:`MarkovSwitchingSolution.irf`.
        The response is the path with the innovation minus the path without it,
        from the same ``state`` (the deviation of the states in the period
        before, the steady state by default) and along the same regimes, with no
        other innovation. At second order the regime constants interact with the
        innovation, so the response depends on the regime path and on the state,
        and a negative innovation is not the mirror of a positive one. With
        ``levels=True`` the path with the innovation is returned, in levels.
        """
        path = (np.full(periods, int(np.asarray(regime))) if np.isscalar(regime)
                else np.asarray(regime, dtype=int))
        if path.size != periods:
            raise ValueError(f"the regime path has {path.size} entries for {periods} periods")
        if np.any(path < 0) or np.any(path >= self.h):
            raise ValueError(f"regimes run from 0 to {self.h - 1}")
        k0 = np.zeros(self.model.n_states) if state is None else np.asarray(state, dtype=float)
        eps = np.zeros((periods, self.model.n_shocks))
        base = self._named(*self._path(k0, path, eps, pruning))
        if periods:
            eps[0, self.model.shock_position(name)] = size
        shocked = self._named(*self._path(k0, path, eps, pruning))
        if levels:
            return self.to_levels(shocked)
        return {n: (shocked[n] - base[n]) * (100.0 if percent else 1.0) for n in shocked}

    # -- moments ---------------------------------------------------------
    def conditional_mean(self, cov: Covariance = None) -> np.ndarray:
        """``(h, n_vars)``: the mean of every variable given each regime, under the pruned rules.

        In closed form, in the rescaled deviations, from the first-order moments
        of :meth:`MarkovSwitchingSolution.regime_moments`. The second-order part of
        the states solves ``q_j = sum_i T[i,j] (P[i,j] q_i + pi_i E[d_ij | s=i])``,
        with ``d_ij`` the quadratic terms of the law of motion, whose innovation
        terms have mean ``1/2 h_uu : Sigma``.
        """
        other = self.at_covariance(cov)
        if other is not self:
            return other.conditional_mean()
        if not self.first.solution.stable:
            raise MarkovSwitchingError("moments need a mean-square stable solution")
        T, h, nk = self.transition, self.h, self.model.n_states
        pi, mk, Mk = _regime_moments(self.P, self.p, self.Q, self.Sigma, T)
        Lq = np.eye(h * nk)
        rq = np.zeros(h * nk)
        for i in range(h):
            for j in range(h):
                if T[i, j] == 0.0:
                    continue
                Lq[j * nk:(j + 1) * nk, i * nk:(i + 1) * nk] -= T[i, j] * self.P[i, j]
                d = 0.5 * (np.einsum("rab,ab->r", self.h_kk[i, j], Mk[i])
                           + np.einsum("rab,ab->r", self.h_uu[i, j], self.Sigma)
                           + self.h_ss[i, j]) + self.h_ks[i, j] @ mk[i]
                rq[j * nk:(j + 1) * nk] += T[i, j] * pi[i] * d
        q = np.linalg.solve(Lq, rq).reshape(h, nk) if nk else np.zeros((h, 0))
        w = np.where(pi > 0, pi, 1.0)
        second = np.where(pi[:, None] > 0, q / w[:, None], 0.0)
        out = np.zeros((h, self.model.n_vars))
        for j in range(h):
            k = mk[j] + second[j]
            u = (self.F[j] @ k + self.f[j] + 0.5 * np.einsum("rab,ab->r", self.g_kk[j], Mk[j])
                 + self.g_ks[j] @ mk[j] + 0.5 * self.g_ss[j])
            out[j] = np.concatenate([k, u])
        return out

    def ergodic_mean(self, cov: Covariance = None, levels: bool = True) -> dict[str, float]:
        """Unconditional mean of every variable under the pruned rules, in levels by default."""
        mean = self.first.ergodic @ self.conditional_mean(cov)
        out = {n: float(mean[i]) for i, n in enumerate(self.model.names)}
        if not levels:
            return out
        return {n: float(v) for n, v in self.to_levels(out).items()}

    def summary(self) -> str:
        return self.first.summary() + f"\n  second-order residual : {self.residual:.3e}"


def _coupled_solve(blocks, rhs, direct_limit, label):
    """Solve ``sum_j blocks[i][j] x_j = rhs_i``, one dense block per pair of regimes."""
    h, size = rhs.shape
    if h * size > direct_limit:
        raise MarkovSwitchingError(
            f"the {label} system has {h * size} unknowns, more than direct_limit={direct_limit}; "
            f"raise it")
    M = np.block([[blocks[i][j] for j in range(h)] for i in range(h)])
    return np.linalg.solve(M, rhs.ravel()).reshape(rhs.shape)


def solve_second_order_switching(first: MarkovSwitchingSolution, cov: Covariance = None, *,
                                 h: float = 1e-4, direct_limit: int = 20000
                                 ) -> SecondOrderSwitchingSolution:
    """Second-order terms of a Markov-switching model around its first-order solution.

    Foerster, Rubio-Ramírez, Waggoner and Zha (2016, Proposition 2): given the
    first-order solution, the second derivatives of the decision rules solve a
    linear system. The laws of motion give the second derivatives of the states
    in terms of those of the jumps, and the expected rows, averaged over the next
    regime and the next innovation, then give three linear systems in the jumps'
    blocks, solved in turn: in ``(k, k)``, in ``(k, chi)`` and in ``(chi, chi)``.
    Each has one block of unknowns per regime, coupled through the transition
    matrix. The derivatives of the residuals are taken at each pair of regimes,
    with the switching parameters that are not perturbed at their regime values,
    so that every coefficient may differ across regimes.

    Parameters
    ----------
    first : MarkovSwitchingSolution
        Output of :func:`solve_markov_switching`.
    cov : mapping, array or None
        Shock covariance, which enters ``g_ss`` and ``h_ss`` only.
    h : float
        Step for the second differences.
    direct_limit : int
        Largest number of unknowns of the ``(k, k)`` system,
        ``h * n_jumps * n_states**2``, which is solved as one dense system.
    """
    if not first.solution.is_real:
        raise MarkovSwitchingError("the second order needs a real first-order solution")
    model, sysm = first.model, first.system
    hr, T = first.h, first.transition
    nk, nu, ne = model.n_states, model.n_jumps, model.n_shocks
    n = nk + nu
    m = 2 * n + ne
    nz = nk + ne + 1
    kz, ez, c = slice(0, nk), slice(nk, nk + ne), nz - 1
    Sigma = shock_covariance(model.shocks, cov)
    blocks = sysm._blocks()
    L, E = sysm.motion, sysm.expected
    F, f, P, Q, p = first.F, first.f, first.P, first.Q, first.p
    pi = _ergodic(T)
    means = {k: float(pi @ v) for k, v in first.switching.items()}
    mean_params = set_params(first.params, means)
    names_next = set(first.next_regime)

    hK = np.zeros((hr, hr, nk, nz, nz))     # the states' Hessian with the jumps' blocks at zero
    R = np.zeros((hr, hr, nu, nz, nz))      # the expected rows with the jumps' blocks at zero
    D = np.zeros((hr, hr, nu, nu))          # multiplies the blocks of the current regime
    MB = np.zeros((hr, hr, nk, nu))         # the states' blocks per unit of the current jumps'
    for i in range(hr):
        for j in range(hr):
            b = blocks[i, j]
            Hij = _pair_hessian(model, first.steady, mean_params, first.switching, names_next,
                                first.perturbed, means, i, j, sysm.row_scale, h)
            # d [y'; y; eps; chi] / d (k, eps, chi) at first order
            V = np.zeros((m + 1, nz))
            V[:nk, kz], V[:nk, ez], V[:nk, c] = P[i, j], Q[i, j], p[i, j]
            V[nk:n, kz], V[nk:n, ez] = F[j] @ P[i, j], F[j] @ Q[i, j]
            V[nk:n, c] = F[j] @ p[i, j] + f[j]
            V[n:n + nk, kz] = np.eye(nk)
            V[n + nk:2 * n, kz], V[n + nk:2 * n, c] = F[i], f[i]
            V[2 * n:m, ez] = np.eye(ne)
            V[m, c] = 1.0
            K = np.einsum("ap,rab,bq->rpq", V, Hij, V)
            lead = b["AkE"] + b["AuE"] @ F[j]
            hK[i, j] = -np.einsum("ka,apq->kpq", b["M"], K[L])
            R[i, j] = K[E] + np.einsum("ra,apq->rpq", lead, hK[i, j])
            MB[i, j] = b["M"] @ b["BuL"]
            D[i, j] = lead @ MB[i, j] - b["BuE"]

    def coupled(kron_of, size):
        """``sum_j T[i,j] (kron(D_ij, I) x_i + kron(AuE_ij, kron_of(i, j)) x_j)``, as blocks."""
        out = []
        for i in range(hr):
            row = [np.zeros((nu * size, nu * size)) for _ in range(hr)]
            for j in range(hr):
                if T[i, j] != 0.0:
                    row[i] += T[i, j] * np.kron(D[i, j], np.eye(size))
                    row[j] += T[i, j] * np.kron(blocks[i, j]["AuE"], kron_of(i, j))
            out.append(row)
        return out

    # (k, k): sum_j T_ij [D_ij G_i + AuE_ij G_j(P_ij, P_ij)] = -sum_j T_ij R_ij
    rhs = np.stack([-np.einsum("j,jrab->rab", T[i], R[i][:, :, kz, kz]).ravel()
                    for i in range(hr)])
    g_kk = _coupled_solve(coupled(lambda i, j: np.kron(P[i, j], P[i, j]).T, nk * nk), rhs,
                          direct_limit, "(k, k)").reshape(hr, nu, nk, nk)
    g_kk = 0.5 * (g_kk + np.transpose(g_kk, (0, 1, 3, 2)))

    # (k, chi): sum_j T_ij [D_ij G_i + AuE_ij G_j P_ij] = -sum_j T_ij [R_ij + AuE_ij g_kk_j(P, p)]
    rhs = np.zeros((hr, nu * nk))
    for i in range(hr):
        for j in range(hr):
            AuE = blocks[i, j]["AuE"]
            rhs[i] -= T[i, j] * (R[i, j][:, kz, c]
                                 + AuE @ np.einsum("abc,bp,c->ap", g_kk[j], P[i, j], p[i, j])
                                 ).ravel()
    g_ks = _coupled_solve(coupled(lambda i, j: P[i, j].T, nk), rhs, direct_limit,
                          "(k, chi)").reshape(hr, nu, nk)

    # (chi, chi), with the variance of the innovations:
    # sum_j T_ij [D_ij G_i + AuE_ij G_j] = -sum_j T_ij [R_ij + R_ij[eps, eps] : Sigma
    #     + AuE_ij (g_kk_j(p, p) + 2 g_ks_j p + g_kk_j(Q, Q) : Sigma)]
    rhs = np.zeros((hr, nu))
    for i in range(hr):
        for j in range(hr):
            AuE = blocks[i, j]["AuE"]
            known = (R[i, j][:, c, c] + np.einsum("rab,ab->r", R[i, j][:, ez, ez], Sigma)
                     + AuE @ (np.einsum("abc,b,c->a", g_kk[j], p[i, j], p[i, j])
                              + 2.0 * g_ks[j] @ p[i, j]
                              + np.einsum("abc,bp,cq,pq->a", g_kk[j], Q[i, j], Q[i, j], Sigma)))
            rhs[i] -= T[i, j] * known
    g_ss = _coupled_solve(coupled(lambda i, j: np.eye(1), 1), rhs, direct_limit,
                          "(chi, chi)").reshape(hr, nu)

    # The states, from the laws of motion.
    h_kk = np.zeros((hr, hr, nk, nk, nk))
    h_ks = np.zeros((hr, hr, nk, nk))
    h_ss = np.zeros((hr, hr, nk))
    for i in range(hr):
        for j in range(hr):
            h_kk[i, j] = hK[i, j][:, kz, kz] + np.einsum("ka,abc->kbc", MB[i, j], g_kk[i])
            h_ks[i, j] = hK[i, j][:, kz, c] + MB[i, j] @ g_ks[i]
            h_ss[i, j] = hK[i, j][:, c, c] + MB[i, j] @ g_ss[i]
    out = SecondOrderSwitchingSolution(
        first=first, g_kk=g_kk, g_ks=g_ks, g_ss=g_ss, h_kk=h_kk,
        h_ku=hK[:, :, :, kz, ez].copy(), h_uu=hK[:, :, :, ez, ez].copy(), h_ks=h_ks,
        h_us=hK[:, :, :, ez, c].copy(), h_ss=h_ss, Sigma=Sigma, step=h)

    # Residual of the expected rows' conditions, recomputed from the assembled solution.
    X = np.zeros((nk + 1, nz))
    X[:nk, kz] = np.eye(nk)
    X[nk, c] = 1.0
    worst = 0.0
    for i in range(hr):
        acc = np.zeros((nu, nz, nz))
        for j in range(hr):
            if T[i, j] == 0.0:
                continue
            b = blocks[i, j]
            Hk = out._h_full(i, j)
            Y = np.vstack([np.hstack([P[i, j], Q[i, j], p[i, j][:, None]]), np.eye(nz)[c][None]])
            u_next = (np.einsum("ap,rab,bq->rpq", Y, out._g_full(j), Y)
                      + np.einsum("ra,apq->rpq", F[j], Hk))
            u_now = np.einsum("ap,rab,bq->rpq", X, out._g_full(i), X)
            lead = b["AkE"] + b["AuE"] @ F[j]
            acc += T[i, j] * (R[i, j] - np.einsum("ra,apq->rpq", lead, hK[i, j])
                              + np.einsum("ra,apq->rpq", b["AkE"], Hk)
                              + np.einsum("ra,apq->rpq", b["AuE"], u_next)
                              - np.einsum("ra,apq->rpq", b["BuE"], u_now))
        for x in (acc[:, kz, kz], acc[:, kz, c],
                  acc[:, c, c] + np.einsum("rab,ab->r", acc[:, ez, ez], Sigma)):
            worst = max(worst, float(np.max(np.abs(x), initial=0.0)))
    out.residual = worst
    return out
