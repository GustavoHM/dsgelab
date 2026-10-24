"""Local identification of the parameters at one point of the parameter space.

The question is whether the distribution of the observables pins down the free
parameters near a given point. Three criteria answer it, each a rank condition
on a Jacobian with respect to the free parameters, as in Dynare's
``identification`` command:

* **moments** (Iskrev 2010): the mean, the covariance matrix and the
  autocovariances up to ``lags`` of the observables;
* **spectrum** (Qu and Tkachenko 2012): the Gram matrix of the derivatives of the
  mean and of the spectral density, summed over a grid of frequencies;
* **minimal system** (Komunjer and Ng 2011): the derivatives of a minimal state
  space representation, next to the similarity transforms of its state and of
  its shocks, which change the representation without changing the observables.

A rank below the number of columns means a direction of the parameters that the
criterion cannot see. Each criterion is then read the way Dynare reads it: which
parameters do not move it at all, which sets of parameters are linearly
dependent, and which parameters are weakly identified because their column is
almost a combination of the others. A brute-force search finds, for each
parameter, the group of other parameters that best reproduces its column.

The derivatives of the state space matrices with respect to the parameters are
central differences with one Richardson step, at the step size, chosen for each
parameter, where a second Richardson estimate from half the steps agrees best.
Everything computed from those matrices, at the point and at the shifted points,
is exact. The disagreement between the two estimates measures the error of the
derivatives, and a singular value below that error is counted as zero.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy import linalg

from .estimation import unpack_point
from .kalman import StateSpace, state_space
from .model import Covariance, Model, Params, Point, get_param

__all__ = ["Identification", "IdentificationCheck", "identification"]

CRITERIA = ("moments", "spectrum", "minimal")


# ===================================================================== results
@dataclass
class IdentificationCheck:
    """The verdict of one criterion.

    Attributes
    ----------
    criterion : ``"moments"``, ``"spectrum"`` or ``"minimal"``.
    jacobian : the matrix whose rank is tested, before its rows are normalised.
        For ``"moments"`` the derivatives of :attr:`Identification.moments`; for
        ``"spectrum"`` the Gram matrix :math:`G`; for ``"minimal"`` the matrix
        :math:`\\bar\\Delta` of Komunjer and Ng, whose columns after the parameters
        belong to the similarity transforms.
    rank, required : the rank found and the rank that identification needs.
    condition : condition number of the normalised matrix, on the columns that move.
    tolerance : the singular value below which a direction counted as zero.
    sensitive : for each parameter, whether its column is nonzero.
    dependent : sets of parameters that are linearly dependent, one per direction
        of the null space.
    multicollinearity : for each parameter, the cosine between its column and its
        projection on the other columns. One means not identified, and values
        near one mean weakly identified.
    weak : parameters whose multicollinearity is within the rank tolerance of one.
    weak_pairs : pairs of weak parameters whose two columns are collinear.
    """

    criterion: str
    jacobian: np.ndarray
    rank: int
    required: int
    condition: float
    tolerance: float
    sensitive: dict[str, bool]
    dependent: list[tuple[str, ...]]
    multicollinearity: dict[str, float]
    weak: list[str]
    weak_pairs: list[tuple[str, str]]

    @property
    def identified(self) -> bool:
        """Full rank: every direction of the parameters moves the criterion."""
        return self.rank == self.required

    @property
    def insensitive(self) -> list[str]:
        """Parameters that do not move the criterion at all."""
        return [k for k, v in self.sensitive.items() if not v]


@dataclass
class Identification:
    """Local identification at one point. See :func:`identification`.

    Attributes
    ----------
    names : the free parameters, in the order of every Jacobian's columns.
    point : their values.
    observables : the observed variables.
    lags : autocovariance lags in :attr:`moments`.
    moments : mean, covariances and autocovariances of the observables, in
        Dynare's order: the means, the lower triangle of the covariance matrix row
        by row, then each autocovariance matrix column by column.
    moment_labels : one label per entry of :attr:`moments`.
    checks : the criteria that were computed, by name.
    skipped : the criteria that were not, with the reason.
    minimal_states : size of the minimal state vector (Komunjer and Ng).
    collinearity : ``collinearity[k][name] = (group, cosine)``, the group of ``k``
        other parameters whose moment derivatives best reproduce those of
        ``name``, from the brute-force search. ``group`` is shorter than ``k`` when
        a larger group adds nothing.
    """

    names: list[str]
    point: dict[str, float]
    observables: list[str]
    lags: int
    moments: np.ndarray
    moment_labels: list[str]
    checks: dict[str, IdentificationCheck]
    skipped: dict[str, str] = field(default_factory=dict)
    minimal_states: int | None = None
    collinearity: dict[int, dict[str, tuple[tuple[str, ...], float]]] = field(
        default_factory=dict)

    @property
    def identified(self) -> bool:
        """Every criterion that was computed has full rank."""
        return all(c.identified for c in self.checks.values())

    def __getitem__(self, criterion: str) -> IdentificationCheck:
        return self.checks[criterion]

    def summary(self) -> str:
        titles = {"moments": "moments (Iskrev 2010)",
                  "spectrum": "spectrum (Qu and Tkachenko 2012)",
                  "minimal": "minimal system (Komunjer and Ng 2011)"}
        n, m = len(self.names), len(self.observables)
        out = [f"Local identification of {n} parameter{'s' * (n != 1)} from "
               f"{m} observable{'s' * (m != 1)} ({', '.join(self.observables)})"]
        for name, c in self.checks.items():
            verdict = "identified" if c.identified else "NOT identified"
            out.append(f"  {titles[name]:40s} rank {c.rank} of {c.required}: {verdict}"
                       f"  (condition {c.condition:.3g})")
            if c.insensitive:
                out.append(f"      do not move it: {', '.join(c.insensitive)}")
            for group in c.dependent:
                if set(group) != set(c.insensitive):
                    out.append(f"      linearly dependent: {', '.join(group)}")
            weak = [w for w in c.weak if w not in c.insensitive]
            if weak:
                out.append(f"      weakly identified: {', '.join(weak)}")
        for name, why in self.skipped.items():
            out.append(f"  {titles[name]:40s} not computed: {why}")
        return "\n".join(out)


# ======================================================================= entry
def identification(model: Model, params: Params, *,
                   free: Sequence[str] | Mapping[str, float],
                   observables: Sequence[str],
                   cov: Covariance = None,
                   measurement: Mapping[str, float] | float | None = None,
                   transforms: str | Mapping[str, str] | None = None,
                   guess: Point | None = None,
                   steady_fn: Callable[[Params], Any] | None = None,
                   qz_criterium: float | None = None,
                   lags: int = 1, grid: int = 5000, max_group: int = 2,
                   criteria: Sequence[str] = CRITERIA,
                   tol_rank: float | str = "auto", tol_deriv: float = 1e-8,
                   tol_sv: float = 1e-3, step: float = 1e-2,
                   levels: int = 8) -> Identification:
    """Local identification of ``free`` at one point, from the observables' distribution.

    Parameters
    ----------
    model, params : the model and its parameters.
    free : the parameters whose identification is checked, named as in
        :class:`~dsgelab.EstimationProblem`: a parameter by its name, ``"stderr e"``
        for the standard deviation of shock ``e`` and ``"stderr y"`` for that of
        the measurement error on observable ``y``. A mapping also sets the point;
        a sequence takes the values in ``params``, ``cov`` and ``measurement``.
    observables : the observed variables.
    cov, measurement, transforms : the shocks' covariance, the measurement errors'
        standard deviations and the transform of each observable, as for
        :class:`~dsgelab.EstimationProblem`.
    guess, steady_fn : how the steady state is found at each point: from ``guess``
        by :func:`dsgelab.steady_state`, or by ``steady_fn(params)``.
    lags : autocovariance lags in the moments (Dynare's ``ar``).
    grid : number of frequencies in :math:`[-\\pi, \\pi]` for the spectrum (even;
        Dynare's ``grid_nbr``).
    max_group : largest group size in the brute-force collinearity search
        (Dynare's ``max_dim_cova_group``); 0 skips the search.
    criteria : which of ``"moments"``, ``"spectrum"`` and ``"minimal"`` to compute.
    tol_rank : tolerance below which a singular value counts as zero. ``"robust"`` is
        Dynare's default, ``max(shape) * eps(norm)`` of each matrix, which assumes
        exact derivatives. ``"auto"`` (the default) is the larger of that and ten
        times the estimated error of the matrix, the 2-norm of the difference
        between the matrices built from the two Richardson estimates of the
        derivatives: by Weyl's inequality a singular value below the error of the
        matrix cannot be told from zero. For the spectrum's G, a Gram matrix whose
        eigenvalues are squared singular values of the stacked derivatives, the
        error is that of the derivatives and the tolerance its square.
    tol_deriv : rows of a Jacobian below this are dropped before normalising.
    tol_sv : entries of a null vector above this mark a parameter as involved.
    step, levels : the derivatives are central differences at ``levels`` steps that
        halve from ``step`` times each parameter's value (``step`` itself for a
        parameter at zero). Every three consecutive steps give two Richardson
        estimates; the steps whose two estimates of the moments' derivatives agree
        best are kept for that parameter, and a step at which the model has no
        stationary solution is skipped.

    Apart from ``tol_rank="auto"``, the tolerances and the reading of each matrix
    are Dynare's, so the verdicts can be compared with its ``identification``
    command at the same point. A model declared in Python is linearised by finite
    differences, and the derivatives of its solution carry an error near
    ``1e-10 / step``; that is what ``"auto"`` is for. A model read from a ``.mod``
    file or declared by equations has exact first derivatives.
    """
    criteria = [c for c in criteria]
    unknown = set(criteria) - set(CRITERIA)
    if unknown:
        raise ValueError(f"unknown criteria {sorted(unknown)}: choose from {CRITERIA}")
    if grid % 2:
        raise ValueError("grid must be even")
    observables = list(observables)
    names = list(free)
    base_meas = _measurement(measurement, observables)
    theta0 = _point(model, params, cov, base_meas, names, free)

    from .solve import solve
    from .steady import steady_state

    def space_at(theta, start):
        p, S, meas = unpack_point(model, params, cov, base_meas, names, theta)
        ss = steady_fn(p) if steady_fn is not None else steady_state(model, start, p)
        sol = solve(model, p, steady=ss, qz_criterium=qz_criterium)
        return state_space(sol, observables, transforms=transforms,
                           measurement=meas or None, cov=S), ss.values

    space0, ss0 = space_at(theta0, guess)
    _check_stationary(space0)

    def moment_vector(s):
        return _moments(s, lags, observables)[0]

    # For each parameter, central differences at steps halving from `step`; the
    # Richardson pair kept is the one whose estimate agrees best with the next.
    pairs = []
    for j in range(len(names)):
        h0 = step * abs(theta0[j]) if theta0[j] != 0 else step
        level: list[tuple[float, StateSpace, StateSpace] | None] = []
        for k in range(levels):
            h = h0 / 2 ** k
            try:
                sides = []
                for sign in (1, -1):
                    th = theta0.copy()
                    th[j] += sign * h
                    sides.append(space_at(th, ss0)[0])
                    _check_stationary(sides[-1])
            except Exception:      # no steady state, no stable solution, a unit root
                level.append(None)
                continue
            level.append((h, sides[0], sides[1]))
        best, best_err = None, np.inf
        for k in range(levels - 2):
            if None in level[k:k + 3]:
                continue
            r1, r2 = _richardson(level[k:k + 3], moment_vector)
            err = np.max(np.abs(r1 - r2)) / max(np.max(np.abs(r1)), 1e-300)
            if err < best_err:
                best, best_err = k, err
        if best is None:
            raise ValueError(f"no step of {names[j]!r} from {h0:.3g} down to "
                             f"{h0 / 2 ** (levels - 1):.3g} keeps the model solvable")
        pairs.append(level[best:best + 3])

    def jac(f: Callable[[StateSpace], np.ndarray]) -> np.ndarray:
        """``[estimate, check]`` for every parameter, stacked as columns."""
        return np.stack([np.stack(_richardson(trio, f)) for trio in pairs], axis=-1)

    moments, labels = _moments(space0, lags, observables)
    dm = jac(moment_vector)
    d_moments = dm[0]
    no = len(observables)
    n_par = len(names)

    checks: dict[str, IdentificationCheck] = {}
    skipped: dict[str, str] = {}
    minimal_states = None
    if "moments" in criteria:
        X, E = _normalise_rows(dm, tol_deriv)
        checks["moments"] = _check("moments", d_moments, X, _norm2(E), 1, names, tol_rank,
                                   tol_sv)
    if "spectrum" in criteria:
        def entries(key: str) -> Callable[[StateSpace], np.ndarray]:
            return lambda s: _matrices(s)[key].ravel()

        d_mats = {k: jac(entries(k)) for k in _MATS}
        G, _, K = _spectrum_gram(space0, [{k: v[i] for k, v in d_mats.items()} for i in (0, 1)],
                                 [dm[0][:no], dm[1][:no]], grid)
        X, err = _gram_tilde(G, K, tol_deriv)
        checks["spectrum"] = _check("spectrum", G, X, err, 3, names, tol_rank, tol_sv)
    if "minimal" in criteria:
        why = _komunjer_ng_unavailable(space0)
        if why:
            skipped["minimal"] = why
        else:
            real = _MinimalRealisation(space0)
            minimal_states = real.m
            if real.m == 0:
                skipped["minimal"] = "the observables do not depend on any state"
            else:
                dl = jac(real.vector)
                Ds = np.stack([_komunjer_ng(real, space0, dl[i], dm[i][:no]) for i in range(2)])
                X, E = _normalise_rows(Ds, tol_deriv)
                checks["minimal"] = _check("minimal", Ds[0], X, _norm2(E), 2, names, tol_rank,
                                           tol_sv)

    collinearity = {}
    rows = np.max(np.abs(d_moments), axis=1) > tol_deriv
    if max_group > 0 and n_par > 1 and rows.any():
        X, E = _normalise_rows(dm, tol_deriv)
        tol = (checks["moments"].tolerance if "moments" in checks
               else _check("moments", d_moments, X, _norm2(E), 1, names, tol_rank,
                           tol_sv).tolerance)
        collinearity = _brute_force(d_moments[rows], names, min(max_group, n_par - 1),
                                    tol_deriv, tol / np.linalg.norm(X, 2))
    return Identification(names=names, point=dict(zip(names, map(float, theta0))),
                          observables=observables, lags=lags, moments=moments,
                          moment_labels=labels, checks=checks, skipped=skipped,
                          minimal_states=minimal_states, collinearity=collinearity)


def _richardson(trio, f):
    """Two Richardson estimates from three halving steps: ``(h, h/2)`` and ``(h/2, h/4)``.

    Both cancel the ``h^2`` error of a central difference, so what separates them
    is rounding noise and the far smaller ``h^4`` term. The first is the estimate
    and the difference measures its error.
    """
    d = [(f(p) - f(m)) / (2 * h) for h, p, m in trio]
    return (4 * d[1] - d[0]) / 3, (4 * d[2] - d[1]) / 3


# ================================================================== the point
def _measurement(measurement, observables) -> dict[str, float]:
    if measurement is None:
        return {}
    if isinstance(measurement, Mapping):
        return {k: float(v) for k, v in measurement.items()}
    return {n: float(measurement) for n in observables}


def _point(model, params, cov, meas, names, free) -> np.ndarray:
    if isinstance(free, Mapping):
        return np.array([float(free[n]) for n in names])
    from .solution import shock_covariance
    S = shock_covariance(model.shocks, cov)
    out = []
    for n in names:
        if n.startswith("stderr "):
            target = n[7:]
            if target in model.shocks:
                out.append(float(np.sqrt(S[model.shocks.index(target), model.shocks.index(target)])))
            elif target in meas:
                out.append(meas[target])
            else:
                raise KeyError(f"{n!r}: no shock or measurement error called {target!r}")
        else:
            out.append(float(get_param(params, n)))
    theta = np.array(out)
    bad = [n for n, v in zip(names, theta) if not np.isfinite(v)]
    if bad:
        raise ValueError(f"no value for {bad}: pass free as a mapping to set the point")
    return theta


def _check_stationary(space: StateSpace) -> None:
    rho = np.max(np.abs(np.linalg.eigvals(space.T))) if space.T.size else 0.0
    if rho >= 1 - 1e-10:
        raise ValueError(f"the state transition has an eigenvalue of modulus {rho:.12g}: "
                         "identification needs stationary observables")


# ===================================================================== moments
_MATS = ("T", "R", "Z", "Sigma", "H")


def _matrices(s: StateSpace) -> dict[str, np.ndarray]:
    return dict(T=s.T, R=s.R, Z=s.Z, Sigma=s.Sigma, H=s.H)


def _moments(s: StateSpace, lags: int, observables) -> tuple[np.ndarray, list[str]]:
    """Mean, covariance (lower triangle by rows) and autocovariances (by columns)."""
    V = linalg.solve_discrete_lyapunov(s.T, s.R @ s.Sigma @ s.R.T)
    G0 = s.Z @ V @ s.Z.T + s.H
    rows, cols = np.tril_indices(len(observables))
    parts = [s.d, G0[rows, cols]]
    labels = [f"E[{y}]" for y in observables]
    labels += [f"cov({observables[i]}, {observables[j]})" for i, j in zip(rows, cols)]
    TjV = V
    for j in range(1, lags + 1):
        TjV = s.T @ TjV
        parts.append((s.Z @ TjV @ s.Z.T).ravel(order="F"))
        labels += [f"cov({a}, {b}(-{j}))" for b in observables for a in observables]
    return np.concatenate(parts), labels


# ==================================================================== spectrum
def _spectrum_gram(s: StateSpace, d_sets: Sequence[dict[str, np.ndarray]],
                   d_means: Sequence[np.ndarray], grid: int,
                   chunk: int = 128) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Qu and Tkachenko's G from two sets of derivatives, and the Gram of their difference.

    The spectral density of the observables is
    ``f(w) = (H(w) Sigma H(w)^* + H_e) / (2 pi)`` with the transfer function
    ``H(w) = Z (I - e^{-iw} T)^{-1} R``. G is the Gram matrix of the stacked
    derivatives of the mean and of ``f`` over Dynare's grid of ``grid + 1``
    frequencies in ``[-pi, pi]``, each positive frequency counted twice. Returns
    ``(G_1, G_2, K)``, with ``K`` the Gram matrix of the difference between the
    two stacked derivatives: its largest eigenvalue is the squared error of the
    first, by the measure of the second.
    """
    nk, ne, no = s.T.shape[0], s.R.shape[1], s.Z.shape[0]
    npar = d_means[0].shape[1]
    sets = []
    for d in d_sets:
        sets.append((d["T"].T.reshape(npar, nk, nk), d["R"].T.reshape(npar, nk, ne),
                     d["Z"].T.reshape(npar, no, nk), d["Sigma"].T.reshape(npar, ne, ne),
                     d["H"].T.reshape(npar, no, no)))
    freqs = np.linspace(0.0, np.pi, grid // 2 + 1)
    weights = np.full(freqs.size, 2.0)
    weights[0] = 1.0
    I = np.eye(nk)
    G1, G2, K = (np.zeros((npar, npar)) for _ in range(3))
    for a in range(0, freqs.size, chunk):
        zbar = np.exp(-1j * freqs[a:a + chunk])
        w = weights[a:a + chunk]
        M = np.linalg.inv(I[None] - zbar[:, None, None] * s.T[None])      # (c, nk, nk)
        ZM = s.Z @ M                                                      # (c, no, nk)
        MR = M @ s.R                                                      # (c, nk, ne)
        Tr = s.Z @ MR                                                     # (c, no, ne)
        TrH = np.conj(np.swapaxes(Tr, 1, 2))                              # (c, ne, no)
        flats = []
        for dT, dR, dZ, dS, dH in sets:
            dTr = (np.einsum("pok,cke->cpoe", dZ, MR)
                   + np.einsum("cok,pke->cpoe", ZM, dR)
                   + zbar[:, None, None, None] * np.einsum(
                       "cpol,cle->cpoe", np.einsum("cok,pkl->cpol", ZM, dT), MR))
            A = np.einsum("cpoe,ef,cfq->cpoq", dTr, s.Sigma, TrH)
            dOmega = (A + np.conj(np.swapaxes(A, 2, 3))
                      + np.einsum("coe,pef,cfq->cpoq", Tr, dS, TrH)
                      + dH[None]) / (2 * np.pi)
            flats.append(dOmega.reshape(dOmega.shape[0], npar, no * no))
        diff = flats[0] - flats[1]
        for out, x in ((G1, flats[0]), (G2, flats[1]), (K, diff)):
            out += np.real(np.einsum("c,cpi,cqi->pq", w, np.conj(x), x))
    scale = 2 * np.pi / (2 * freqs.size - 1)
    dm = d_means[0] - d_means[1]
    return (scale * G1 + d_means[0].T @ d_means[0], scale * G2 + d_means[1].T @ d_means[1],
            scale * K + dm.T @ dm)


def _gram_tilde(G: np.ndarray, K: np.ndarray, tol_deriv: float) -> tuple[np.ndarray, float]:
    """Dynare's correlation-like normalisation of G on its nonzero rows, and its root error.

    Normalising G by its diagonal gives the Gram matrix of the stacked derivatives
    with unit columns; the error returned is the 2-norm of the error of those
    columns, from ``K`` normalised the same way.
    """
    rows = np.flatnonzero(np.max(np.abs(G), axis=1) > tol_deriv)
    out = np.zeros_like(G)
    err = 0.0
    if rows.size:
        delta = np.sqrt(np.diag(G)[rows])
        scale = np.outer(delta, delta)
        out[np.ix_(rows, rows)] = G[np.ix_(rows, rows)] / scale
        Kt = K[np.ix_(rows, rows)] / scale
        err = float(np.sqrt(max(np.max(np.linalg.eigvalsh((Kt + Kt.T) / 2)), 0.0)))
    return out, err


# ================================================================ minimal system
def _komunjer_ng_unavailable(s: StateSpace) -> str:
    if np.any(s.H != 0):
        return "measurement errors are not handled by this criterion"
    if s.Z.shape[0] < s.R.shape[1]:
        return "fewer observables than shocks, as in Dynare"
    return ""


class _MinimalRealisation:
    """A minimal ``(A, B, C, D)`` of the observables, smooth in the parameters.

    Dynare's timing: ``x_t = A x_{t-1} + B u_t``, ``y_t = C x_{t-1} + D u_t``, which
    for this package's state space is ``A = T``, ``B = R``, ``C = Z T``, ``D = Z R``.
    The minimal size ``m`` comes from Rosenbrock's staircase, as in Dynare. The
    realisation itself is Ho and Kalman's from the block Hankel matrix
    ``O(A, C) K(A, B)``, with left and right projections fixed at the point; it is
    therefore a smooth function of the parameters, and its finite differences are
    derivatives of one minimal system and not of several bases glued together.
    """

    def __init__(self, s: StateSpace, tol: float = 1e-7):
        A, B, C, D = _abcd(s)
        self.n = A.shape[0]
        self.m = _minimal_order(A, B, C, tol)
        if self.m == 0:
            return
        H = _observability(A, C) @ _controllability(A, B)
        U, sv, Vt = np.linalg.svd(H)
        self.left = U[:, :self.m].T
        self.right = Vt[:self.m].T
        self.A, self.B, self.C, self.D = self.realise(A, B, C, D)

    def realise(self, A, B, C, D):
        X = self.left @ _observability(A, C)               # m x n
        Y = _controllability(A, B) @ self.right             # n x m
        XY = X @ Y
        return (np.linalg.solve(XY, X @ A @ Y), np.linalg.solve(XY, X @ B), C @ Y, D)

    def vector(self, s: StateSpace) -> np.ndarray:
        A, B, C, D = self.realise(*_abcd(s))
        ne = B.shape[1]
        vech = s.Sigma.T[np.triu_indices(ne)]               # column-major lower triangle
        return np.concatenate([A.ravel(order="F"), B.ravel(order="F"),
                               C.ravel(order="F"), D.ravel(order="F"), vech])


def _abcd(s: StateSpace):
    return s.T, s.R, s.Z @ s.T, s.Z @ s.R


def _controllability(A, B):
    blocks, P = [], B
    for _ in range(A.shape[0]):
        blocks.append(P)
        P = A @ P
    return np.hstack(blocks)


def _observability(A, C):
    blocks, P = [], C
    for _ in range(A.shape[0]):
        blocks.append(P)
        P = P @ A
    return np.vstack(blocks)


def _staircase(a, b, tol):
    """Rosenbrock's controllability staircase: the size of the controllable part."""
    n = a.shape[0]
    aj, bj = a, b
    total = 0
    for _ in range(n):
        u, _, _ = np.linalg.svd(bj)
        u = u[:, ::-1]
        bb = u.T @ bj
        sv = np.linalg.svd(bb, compute_uv=False)
        roj = int(np.sum(sv > tol))
        sigma = bb.shape[0] - roj
        total += roj
        if roj == 0 or sigma == 0:
            break
        abxy = u.T @ aj @ u
        aj, bj = abxy[:sigma, :sigma], abxy[:sigma, sigma:sigma + roj]
    return total


def _minimal_order(A, B, C, tol) -> int:
    """Controllable part, then its observable part, as Dynare's ``minrealold``."""
    n = A.shape[0]
    nc = _staircase(A, B, tol)
    if nc == 0:
        return 0
    if nc < n:
        # the controllable subspace, from an orthonormal basis of K(A, B)
        Q = np.linalg.svd(_controllability(A, B), full_matrices=False)[0][:, :nc]
        A, B, C = Q.T @ A @ Q, Q.T @ B, C @ Q
    return _staircase(A.T, C.T, tol)


def _komunjer_ng(real: _MinimalRealisation, s: StateSpace, d_lambda: np.ndarray,
                 d_mean: np.ndarray) -> np.ndarray:
    """``[dmean 0; dlambda  Delta_T  Delta_U]`` of Komunjer and Ng (2011), as Dynare builds it."""
    A, B, C, D, Sigma = real.A, real.B, real.C, real.D, s.Sigma
    m, ne, no = A.shape[0], B.shape[1], C.shape[0]
    Im, Ie = np.eye(m), np.eye(ne)
    Dp = _duplication(ne)
    Dp_pinv = np.linalg.solve(Dp.T @ Dp, Dp.T)
    nv = ne * (ne + 1) // 2
    DT = np.vstack([np.kron(A.T, Im) - np.kron(Im, A), np.kron(B.T, Im), -np.kron(Im, C),
                    np.zeros((no * ne, m * m)), np.zeros((nv, m * m))])
    DU = np.vstack([np.zeros((m * m, ne * ne)), np.kron(Ie, B), np.zeros((no * m, ne * ne)),
                    np.kron(Ie, D), -2 * Dp_pinv @ np.kron(Sigma, Ie)])
    top = np.hstack([d_mean, np.zeros((d_mean.shape[0], m * m + ne * ne))])
    return np.vstack([top, np.hstack([d_lambda, DT, DU])])


def _duplication(p: int) -> np.ndarray:
    """``vec(S) = D vech(S)`` for symmetric ``S``, with ``vech`` by columns."""
    idx = np.zeros((p, p), dtype=int)
    k = 0
    for j in range(p):
        for i in range(j, p):
            idx[i, j] = idx[j, i] = k
            k += 1
    D = np.zeros((p * p, k))
    for c, v in enumerate(idx.ravel(order="F")):
        D[c, v] = 1.0
    return D


# ===================================================================== reading
def _normalise_rows(Js: np.ndarray, tol_deriv: float) -> tuple[np.ndarray, np.ndarray]:
    """Dynare's normalisation: nonzero rows, each divided by its largest entry.

    ``Js`` stacks the matrix built from the two derivative estimates; the error is
    their difference, scaled by the same rows.
    """
    J = Js[0]
    rows = np.max(np.abs(J), axis=1) > tol_deriv
    scale = np.max(np.abs(J[rows]), axis=1, keepdims=True)
    return J[rows] / scale, (Js[0] - Js[1])[rows] / scale


def _rank(X, tol) -> int:
    if X.size == 0:
        return 0
    return int(np.sum(np.linalg.svd(X, compute_uv=False) > tol))


def _cosn(y, X, rcond: float | None = None) -> float:
    """Cosine between ``y`` and its least-squares projection on the columns of ``X``.

    Directions of ``X`` whose singular value is below ``rcond`` times the largest
    are left out of the projection: they are numerical noise, and a regression on
    them, with a huge coefficient, would make any column look collinear.
    """
    if X.shape[1] == 0:
        return 0.0
    b = np.linalg.lstsq(X, y, rcond=rcond)[0]
    if not np.all(np.isfinite(b)):
        return 0.0
    yhat = X @ b
    ny, nh = float(y @ y), float(yhat @ yhat)
    if nh == 0.0 or not np.isfinite(nh) or _rank(yhat[:, None], _robust(yhat[:, None])) == 0:
        return 0.0
    return float(abs(y @ yhat) / np.sqrt(ny * nh))


def _robust(X) -> float:
    norm = np.linalg.norm(X, 2) if X.size else 0.0
    return max(X.shape) * np.spacing(norm)


def _norm2(E: np.ndarray) -> float:
    return float(np.linalg.norm(E, 2)) if E.size else 0.0


def _check(name, J, X, error, kind, names, tol_rank, tol_sv) -> IdentificationCheck:
    """Dynare's ``identification.checks``: kind 1 a Jacobian, 2 Komunjer and Ng's, 3 a Gram.

    ``X`` is the normalised matrix and ``error`` the estimated 2-norm error of the
    Jacobian behind it: of ``X`` itself, or for a Gram matrix of the derivatives
    whose Gram it is, so that its tolerance is the square.
    """
    p = len(names)
    if tol_rank == "auto":
        floor = 10 * error
        tol = max(_robust(X), floor ** 2 if kind == 3 else floor)
    elif tol_rank == "robust":
        tol = _robust(X)
    else:
        tol = float(tol_rank)
    Xpar, Xrest = (X[:, :p], X[:, p:]) if kind == 2 else (X, X[:, :0])
    top = np.linalg.norm(X, 2) if X.size else 0.0
    rcond = tol / top if top > 0 else None
    norms = np.sqrt(np.sum(Xpar ** 2, axis=0)) if Xpar.shape[0] > 1 else np.abs(Xpar).ravel()
    ind1 = np.flatnonzero(norms >= np.finfo(float).eps)

    def nonzero(cols):
        return Xpar[np.ix_(cols, cols)] if kind == 3 else Xpar[:, cols]

    def with_rest(M):
        return np.hstack([M, Xrest]) if kind == 2 else M

    def right_vectors(M):
        return np.linalg.svd(M, full_matrices=True)[2].T if M.size else np.zeros((M.shape[1],) * 2)

    def cond(M):
        if M.size == 0:
            return np.inf
        sv = np.linalg.svd(M, compute_uv=False)
        return float(sv[0] / sv[-1]) if sv[-1] > 0 else np.inf

    full = with_rest(nonzero(ind1))
    V = right_vectors(full)
    rank = _rank(X, tol)
    required = X.shape[1]
    if p and rank < required:
        keep = np.ones(ind1.size, dtype=bool)
        found = False
        for j in range(rank, full.shape[1]):
            v = np.abs(V[:, j])
            if np.sum(v > tol_sv) == 1:
                found = True
                keep &= (v <= tol_sv)[:ind1.size]
        if found:
            ind1 = ind1[keep]
            full = with_rest(nonzero(ind1))
            V = right_vectors(full)
    condition = cond(full)
    sensitive = np.zeros(p, dtype=bool)
    sensitive[ind1] = True

    mco = np.full(p, np.nan)
    Xnz = nonzero(ind1)
    if kind == 3:
        if ind1.size:
            sub = X[np.ix_(ind1, ind1)]
            delta = np.sqrt(np.diag(sub))
            tilde = sub / np.outer(delta, delta)
            try:
                inv_diag = np.diag(np.linalg.inv(tilde))
            except np.linalg.LinAlgError:
                inv_diag = np.full(ind1.size, np.inf)
            with np.errstate(divide="ignore"):
                mco[ind1] = 1 - 1 / inv_diag
    else:
        for ii in range(ind1.size):
            others = np.delete(np.arange(ind1.size), ii)
            mco[ind1[ii]] = _cosn(Xnz[:, ii], with_rest(Xnz[:, others]), rcond)

    dependent: list[tuple[str, ...]] = []
    finite = mco[np.isfinite(mco)]
    if p and (rank < required or (finite.size and np.min(1 - finite) < tol)):
        if ind1.size < p:
            dependent.append(tuple(names[i] for i in range(p) if not sensitive[i]))
        for j in range(rank, full.shape[1]):
            v = np.abs(V[:ind1.size, j])
            group = tuple(names[ind1[i]] for i in np.flatnonzero(v > tol_sv))
            if group:
                dependent.append(group)

    pco = np.full((p, p), np.nan)
    for ii in range(ind1.size):
        pco[ind1[ii], ind1[ii]] = 1.0
        for jj in range(ii + 1, ind1.size):
            c = _cosn(Xnz[:, ii], with_rest(Xnz[:, [jj]]), rcond)
            pco[ind1[ii], ind1[jj]] = pco[ind1[jj], ind1[ii]] = c
    weak, pairs = [], []
    for j in range(p):
        if mco[j] > 1 - tol:
            weak.append(names[j])
            for k in range(j + 1, p):
                if pco[j, k] > 1 - tol:
                    pairs.append((names[j], names[k]))
    return IdentificationCheck(criterion=name, jacobian=J, rank=rank, required=required,
                               condition=condition, tolerance=tol,
                               sensitive={n: bool(sensitive[i]) for i, n in enumerate(names)},
                               dependent=dependent,
                               multicollinearity={n: float(mco[i]) for i, n in enumerate(names)},
                               weak=weak, weak_pairs=pairs)


def _brute_force(J: np.ndarray, names, max_group: int, tol_deriv: float,
                 rcond: float | None = None) -> dict:
    """For each parameter and group size, the other parameters that best reproduce its column."""
    p = len(names)
    out: dict[int, dict[str, tuple[tuple[str, ...], float]]] = {}
    for size in range(1, max_group + 1):
        out[size] = {}
        for i in range(p):
            others = [k for k in range(p) if k != i]
            best: tuple[int, ...] = ()
            best_c = -1.0
            for combo in itertools.combinations(others, size):
                c = _cosn(J[:, i], J[:, list(combo)], rcond)
                if c > best_c:
                    best, best_c = combo, c
            if best_c <= tol_deriv:
                out[size][names[i]] = ((), best_c)
                continue
            if size > 1 and best_c - out[size - 1][names[i]][1] < tol_deriv:
                out[size][names[i]] = out[size - 1][names[i]]
                continue
            out[size][names[i]] = (tuple(names[k] for k in best), best_c)
    return out
