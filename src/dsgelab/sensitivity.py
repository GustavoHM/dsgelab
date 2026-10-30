"""Global sensitivity: where in the parameter space the model works, and what moves its results.

Three tools, the first two those of Dynare's sensitivity toolbox (``+gsa``):

* :func:`stability_map` draws parameters, solves the model at each draw, and
  sorts the draws into those with a unique stable solution, the indeterminate,
  the explosive and those without a solution (Dynare's ``stab=1``).
* :func:`monte_carlo_filtering` splits a sample of parameters into the draws
  that show a behaviour and those that do not, and asks which parameters drive
  the split: a two-sample Smirnov test per parameter, and the correlations
  between parameters inside each group. :meth:`StabilityMap.filtering` applies it
  to determinacy, indeterminacy and explosiveness, as Dynare does after the
  stability mapping, and :meth:`~dsgelab.RestrictionSample.filtering` to the
  calibration restrictions, as its ``map_calibration`` does.
* :func:`sobol` computes first-order and total Sobol indices of any function of
  the parameters on Saltelli's design, and :func:`solution_outputs` makes that
  function from impulse responses and moments of the first-order solution.

The distributions :class:`Uniform`, :class:`Normal` and :class:`Triangular` have
the interface of ``cgelab.sensitivity``'s (``mean``, ``sd``, ``symmetric``,
``draw`` and ``ppf``), and so does :class:`~dsgelab.Prior`, so a prior can be
given wherever a distribution is. Neither package imports the other.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike

from .model import Covariance, Model, Params, Point

__all__ = ["Uniform", "Normal", "Triangular", "smirnov_test", "FilteringResult",
           "monte_carlo_filtering", "StabilityMap", "stability_map", "SobolResult", "sobol",
           "solution_outputs"]


# ============================================================= distributions
@dataclass(frozen=True)
class Uniform:
    low: float
    high: float
    symmetric = True

    def __post_init__(self):
        if not self.high > self.low:
            raise ValueError(f"Uniform needs high > low, got {self.low}, {self.high}")

    @property
    def mean(self) -> float:
        return 0.5 * (self.low + self.high)

    @property
    def sd(self) -> float:
        return (self.high - self.low) / math.sqrt(12.0)

    def draw(self, rng, n):
        return rng.uniform(self.low, self.high, n)

    def ppf(self, u):
        """The quantile at probability ``u``: how a point of the unit cube becomes a value."""
        return self.low + np.asarray(u, dtype=float) * (self.high - self.low)


@dataclass(frozen=True)
class Normal:
    mean: float
    sd: float
    symmetric = True

    def __post_init__(self):
        if not self.sd > 0:
            raise ValueError(f"Normal needs sd > 0, got {self.sd}")

    def draw(self, rng, n):
        return rng.normal(self.mean, self.sd, n)

    def ppf(self, u):
        from scipy.stats import norm
        return self.mean + self.sd * norm.ppf(np.asarray(u, dtype=float))


@dataclass(frozen=True)
class Triangular:
    low: float
    mode: float
    high: float

    def __post_init__(self):
        if not self.low <= self.mode <= self.high or not self.high > self.low:
            raise ValueError("Triangular needs low <= mode <= high and high > low")

    @property
    def symmetric(self) -> bool:
        return abs((self.mode - self.low) - (self.high - self.mode)) <= 1e-12 * (self.high - self.low)

    @property
    def mean(self) -> float:
        return (self.low + self.mode + self.high) / 3.0

    @property
    def sd(self) -> float:
        a, c, b = self.low, self.mode, self.high
        return math.sqrt((a * a + b * b + c * c - a * b - a * c - b * c) / 18.0)

    def draw(self, rng, n):
        return rng.triangular(self.low, self.mode, self.high, n)

    def ppf(self, u):
        a, c, b = self.low, self.mode, self.high
        u = np.asarray(u, dtype=float)
        cut = (c - a) / (b - a)
        with np.errstate(invalid="ignore"):
            left = a + np.sqrt(u * (b - a) * (c - a))
            right = b - np.sqrt((1.0 - u) * (b - a) * (b - c))
        return np.where(u < cut, left, right)


def _distribution(spec):
    """A distribution with ``ppf``: one of the three above, a Prior, or a prior tuple."""
    from .priors import prior as make_prior
    if hasattr(spec, "ppf"):
        return spec
    if isinstance(spec, tuple):
        return make_prior(*spec)
    raise TypeError(f"a distribution is Uniform, Normal, Triangular, a Prior or a prior "
                    f"tuple, got {spec!r}")


def _sample(dists: Sequence, n: int, seed: int | None, sampling: str) -> np.ndarray:
    from scipy.stats import qmc
    k = len(dists)
    if sampling == "random":
        rng = np.random.default_rng(seed)
        U = rng.random((n, k))
    elif sampling == "sobol":
        engine = qmc.Sobol(d=k, scramble=True, seed=seed)
        U = engine.random_base2(int(math.log2(n))) if n & (n - 1) == 0 else engine.random(n)
    elif sampling == "latin":
        U = qmc.LatinHypercube(d=k, seed=seed).random(n)
    else:
        raise ValueError("sampling must be 'sobol', 'latin' or 'random'")
    return np.column_stack([np.asarray(d.ppf(U[:, j]), dtype=float) for j, d in enumerate(dists)])


# ===================================================== Monte Carlo filtering
def smirnov_test(x1: ArrayLike, x2: ArrayLike, alternative: str = "two-sided") -> tuple[float, float]:
    """Two-sample Smirnov test: ``(d, p)``.

    ``d`` is the largest distance between the two empirical distribution
    functions (``"two-sided"``), or the largest excess of the second over the
    first (``"less"``) or of the first over the second (``"greater"``). The
    p-value is the asymptotic one with Stephens' small-sample factor,
    ``lambda = (sqrt(m) + 0.12 + 0.11/sqrt(m)) d`` for ``m = n1 n2 / (n1 + n2)``,
    summed over 101 terms of the Kolmogorov series: Dynare's ``smirnov_test``.
    """
    a = np.sort(np.asarray(x1, dtype=float).ravel())
    b = np.sort(np.asarray(x2, dtype=float).ravel())
    if a.size == 0 or b.size == 0:
        raise ValueError("both samples need at least one value")
    grid = np.concatenate([a, b])
    f1 = np.searchsorted(a, grid, side="right") / a.size
    f2 = np.searchsorted(b, grid, side="right") / b.size
    if alternative == "two-sided":
        d = float(np.max(np.abs(f1 - f2)))
    elif alternative == "less":
        d = max(float(np.max(f2 - f1)), 0.0)
    elif alternative == "greater":
        d = max(float(np.max(f1 - f2)), 0.0)
    else:
        raise ValueError("alternative must be 'two-sided', 'less' or 'greater'")
    m = a.size * b.size / (a.size + b.size)
    lam = max((math.sqrt(m) + 0.12 + 0.11 / math.sqrt(m)) * d, 0.0)
    if alternative == "two-sided":
        j = np.arange(1, 102)
        p = 2 * float(np.sum((-1.0) ** (j - 1) * np.exp(-2 * lam * lam * j * j)))
        p = min(max(p, 0.0), 1.0)
    else:
        p = math.exp(-2 * lam * lam)
    return d, p


def _correlation(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pearson correlations and their two-sided p-values (Student's t, ``n - 2`` df)."""
    from scipy.stats import t as student
    n = X.shape[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        R = np.corrcoef(X, rowvar=False)
        R = np.atleast_2d(R)
        T = R * np.sqrt((n - 2) / np.maximum(1 - R * R, 0.0))
        P = 2 * student.sf(np.abs(T), n - 2)
    np.fill_diagonal(P, 1.0)
    return R, P


@dataclass
class FilteringResult:
    """Monte Carlo filtering of a sample into two groups. See :func:`monte_carlo_filtering`.

    ``d`` and ``p`` are the Smirnov statistic and p-value of each parameter, in
    the order of ``names``. ``significant`` lists the parameters with ``p`` below
    ``pvalue_ks``, the most significant first: the parameters that drive the
    behaviour.
    """

    names: list
    points: np.ndarray
    behaviour: np.ndarray
    nonbehaviour: np.ndarray
    d: np.ndarray
    p: np.ndarray
    pvalue_ks: float = 0.001
    title: str = "behaviour"

    @property
    def n_behaviour(self) -> int:
        return int(self.behaviour.sum())

    @property
    def n_nonbehaviour(self) -> int:
        return int(self.nonbehaviour.sum())

    @property
    def significant(self) -> list:
        idx = [i for i in np.argsort(self.p, kind="stable") if self.p[i] < self.pvalue_ks]
        return [self.names[i] for i in idx]

    def correlations(self, group: str = "behaviour", *, pvalue: float = 1e-5,
                     alpha2: float = 0.0) -> list[tuple[str, str, float, float]]:
        """Pairs of parameters correlated inside one group: ``(name1, name2, r, p)``.

        ``group`` is ``"behaviour"`` or ``"nonbehaviour"``. A pair is kept when
        ``|r| > alpha2`` and its p-value is below ``pvalue`` (Dynare's
        ``alpha2_stab`` and ``pvalue_corr``). Filtering on an outcome induces
        correlations between parameters that are independent a priori, when the
        outcome depends on a combination of them.
        """
        if group not in ("behaviour", "nonbehaviour"):
            raise ValueError("group must be 'behaviour' or 'nonbehaviour'")
        X = self.points[self.behaviour if group == "behaviour" else self.nonbehaviour]
        if X.shape[0] < 3:
            return []
        R, P = _correlation(X)
        out = []
        k = len(self.names)
        for j in range(k):
            for i in range(j + 1, k):
                if abs(R[i, j]) > alpha2 and P[j, i] < pvalue:
                    out.append((self.names[j], self.names[i], float(R[i, j]), float(P[j, i])))
        return out

    def summary(self) -> str:
        out = [f"Smirnov statistics in driving {self.title}: {self.n_behaviour} draws with it, "
               f"{self.n_nonbehaviour} without"]
        sig = self.significant
        if not sig:
            out.append(f"  no parameter with p-value below {self.pvalue_ks:g}")
        else:
            out.append(f"  {'parameter':16s} {'d-stat':>8s} {'p-value':>10s}")
            for n in sig:
                i = self.names.index(n)
                out.append(f"  {n:16s} {self.d[i]:8.3f} {self.p[i]:10.2e}")
        return "\n".join(out)


def _mask(which, n: int) -> np.ndarray:
    w = np.asarray(which)
    if w.dtype == bool:
        if w.shape != (n,):
            raise ValueError(f"a mask needs {n} entries, got {w.shape}")
        return w.copy()
    m = np.zeros(n, dtype=bool)
    m[w.astype(int)] = True
    return m


def monte_carlo_filtering(points: ArrayLike, behaviour: ArrayLike,
                          names: Sequence[str] | None = None, *,
                          nonbehaviour: ArrayLike | None = None, pvalue_ks: float = 0.001,
                          title: str = "behaviour") -> FilteringResult:
    """Which parameters separate the draws that show a behaviour from those that do not.

    Parameters
    ----------
    points : ``(draws, parameters)``
    behaviour : boolean mask or indices of the draws that show the behaviour.
    names : the parameters, by default ``x0, x1, ...``.
    nonbehaviour : the draws that do not; by default every other draw. Draws in
        neither group (a draw that could not be solved, say) are left out.
    pvalue_ks : the threshold of :attr:`FilteringResult.significant`, Dynare's
        ``pvalue_ks``.

    For each parameter, :func:`smirnov_test` compares its distribution in the
    two groups: a parameter whose distribution differs drives the behaviour.
    This is the Monte Carlo filtering of Hornberger and Spear (1981) and Saltelli
    et al. (2008), as Dynare's ``+gsa`` runs it.
    """
    X = np.atleast_2d(np.asarray(points, dtype=float))
    n, k = X.shape
    names = list(names) if names is not None else [f"x{i}" for i in range(k)]
    if len(names) != k:
        raise ValueError(f"{len(names)} names for {k} columns")
    beh = _mask(behaviour, n)
    nob = ~beh if nonbehaviour is None else _mask(nonbehaviour, n)
    if (beh & nob).any():
        raise ValueError("a draw cannot be in both groups")
    d, p = np.full(k, np.nan), np.full(k, np.nan)
    if beh.any() and nob.any():
        for j in range(k):
            d[j], p[j] = smirnov_test(X[beh, j], X[nob, j])
    return FilteringResult(names=names, points=X, behaviour=beh, nonbehaviour=nob, d=d, p=p,
                           pvalue_ks=pvalue_ks, title=title)


# ========================================================== stability mapping
_STATUSES = ("determinate", "indeterminate", "explosive", "no_steady_state", "no_solution")


@dataclass
class StabilityMap:
    """Draws of the parameters and what the model does at each. See :func:`stability_map`.

    ``status`` is, per draw, ``"determinate"`` (a unique stable solution),
    ``"indeterminate"`` (more stable roots than predetermined variables, or as
    many with a stable subspace that does not span the states, Dynare's rank
    failure), ``"explosive"`` (fewer), ``"no_steady_state"`` or ``"no_solution"``
    (any other failure). These are the verdicts of Dynare's solver (its
    ``info`` codes 0, 4 and 5, and 3).
    """

    names: list
    draws: np.ndarray
    status: np.ndarray
    pvalue_ks: float = 0.001

    @property
    def shares(self) -> dict:
        """Share of the draws in each status."""
        n = len(self.status)
        return {s: float(np.sum(self.status == s)) / n for s in _STATUSES}

    def filtering(self, kind: str = "determinate", *,
                  columns: Sequence[str] | None = None) -> FilteringResult:
        """Monte Carlo filtering of determinacy, indeterminacy or explosiveness.

        The groups are Dynare's. For ``"determinate"``, the draws with a unique
        stable solution against all the others. For ``"indeterminate"`` and
        ``"explosive"``, the draws in that status against the other draws the
        model could be solved at, leaving out those without a steady state or a
        solution. ``columns`` are the parameters tested; by default every name
        but the shock standard deviations (``"stderr ..."``), which cannot move
        the roots, as in Dynare.
        """
        s = self.status
        solved = (s != "no_steady_state") & (s != "no_solution")
        if kind == "determinate":
            beh, nob, title = s == "determinate", s != "determinate", "a unique stable solution"
        elif kind == "indeterminate":
            beh, nob, title = solved & (s != "indeterminate"), s == "indeterminate", "determinacy"
        elif kind == "explosive":
            beh, nob, title = solved & (s != "explosive"), s == "explosive", "stability"
        else:
            raise ValueError("kind must be 'determinate', 'indeterminate' or 'explosive'")
        cols = list(columns) if columns is not None else [
            n for n in self.names if not n.startswith("stderr ")]
        idx = [self.names.index(c) for c in cols]
        return monte_carlo_filtering(self.draws[:, idx], beh, cols, nonbehaviour=nob,
                                     pvalue_ks=self.pvalue_ks, title=title)

    def summary(self) -> str:
        sh = self.shares
        out = [f"Stability mapping over {len(self.status)} draws",
               f"  {100 * sh['determinate']:5.1f}% unique stable solution",
               f"  {100 * sh['explosive']:5.1f}% explosive",
               f"  {100 * sh['indeterminate']:5.1f}% indeterminate"]
        if sh["no_steady_state"]:
            out.append(f"  {100 * sh['no_steady_state']:5.1f}% no steady state")
        if sh["no_solution"]:
            out.append(f"  {100 * sh['no_solution']:5.1f}% no solution")
        for kind in ("determinate", "indeterminate", "explosive"):
            f = self.filtering(kind)
            if f.n_behaviour and f.n_nonbehaviour:
                out.append(f.summary())
        return "\n".join(out)


def stability_map(model: Model, params: Params, distributions: Mapping[str, Any] | None = None, *,
                  draws: int = 2048, seed: int | None = 0, sampling: str = "sobol",
                  points: ArrayLike | None = None, names: Sequence[str] | None = None,
                  cov: Covariance = None, guess: Point | None = None,
                  steady_fn: Callable[[Params], Any] | None = None,
                  qz_criterium: float | None = None, pvalue_ks: float = 0.001) -> StabilityMap:
    """Where in the parameter space the model has a unique stable solution.

    Parameters
    ----------
    distributions : ``{name: distribution}``, with :class:`Uniform`,
        :class:`Normal`, :class:`Triangular`, a :class:`~dsgelab.Prior` or a
        prior tuple ``(shape, mean, sd[, p3, p4])``. A name is a parameter or
        ``"stderr e"``.
    draws, seed, sampling : how many draws, the seed, and the design:
        ``"sobol"`` (a scrambled Sobol sequence, as Dynare's quasi-random
        default), ``"latin"`` (Latin hypercube) or ``"random"``.
    points : classify these rows instead of drawing, one column per name in
        ``names`` (by default the keys of ``distributions``).
    cov : the shock covariance held fixed.
    guess, steady_fn : the steady state at each draw is ``steady_fn(params)``, or
        :func:`steady_state` from ``guess``.
    qz_criterium : as for :func:`solve`.
    pvalue_ks : the threshold of the filtering that follows.

    This is the stability mapping of Dynare's sensitivity toolbox,
    ``dynare_sensitivity(stab=1)``.
    """
    from .estimation import unpack_point
    from .klein import BlanchardKahnError
    from .solve import solve
    from .steady import steady_state

    dists = {k: _distribution(v) for k, v in dict(distributions or {}).items()}
    names = list(names) if names is not None else list(dists)
    if not names:
        raise ValueError("nothing varies: give distributions, or points with names")
    if points is None:
        missing = [n for n in names if n not in dists]
        if missing:
            raise ValueError(f"no distribution for {missing}")
        if draws < 1:
            raise ValueError("draws must be positive")
        X = _sample([dists[n] for n in names], draws, seed, sampling)
    else:
        X = np.atleast_2d(np.asarray(points, dtype=float))
        if X.shape[1] != len(names):
            raise ValueError(f"points have {X.shape[1]} columns for {len(names)} names")
    status = np.full(X.shape[0], "no_solution", dtype=object)
    for t, x in enumerate(X):
        try:
            p, _, _ = unpack_point(model, params, cov, {}, names, x)
            ss = steady_fn(p) if steady_fn is not None else steady_state(model, guess, p)
            solve(model, p, steady=ss, qz_criterium=qz_criterium)
        except BlanchardKahnError as err:
            # the right count of stable roots with a singular Z11 is a rank failure, which
            # Dynare reports as indeterminacy (info 5); a count with no roots is a model error
            if err.n_stable > err.n_k or (err.n_stable == err.n_k and err.eigenvalues is not None):
                status[t] = "indeterminate"
            elif err.n_stable < err.n_k:
                status[t] = "explosive"
            continue
        except RuntimeError:
            status[t] = "no_steady_state"
            continue
        except (ValueError, ArithmeticError, np.linalg.LinAlgError):
            continue
        status[t] = "determinate"
    return StabilityMap(names=names, draws=X, status=status.astype(str), pvalue_ks=pvalue_ks)


# ============================================================ Sobol indices
@dataclass
class SobolResult:
    """Saltelli's design, the outputs at each point, and the indices. See :func:`sobol`.

    ``A`` and ``B`` are ``(n, k)``; ``values[output]`` holds the output on ``A``
    and ``B``, ``(n,)`` each, and on every ``AB_i``, ``(n, k)``.
    """

    names: list
    outputs: list
    n: int
    A: np.ndarray
    B: np.ndarray
    values: dict
    bootstrap: int
    conf_level: float
    seed: int | None

    def indices(self) -> dict:
        """``{output: {"S1", "S1_conf", "ST", "ST_conf", "n"}}``, arrays in the order of ``names``.

        First order by Saltelli et al. (2010), total by Jansen (1999), each over
        the variance of the ``A`` and ``B`` outputs together, on outputs centred
        on the mean of every evaluation: the estimators of SALib, which give the
        same numbers on the same design. The half-widths at ``conf_level`` come
        from a bootstrap over the base points. A base point where the output is
        missing at ``A``, ``B`` or any ``AB_i`` (a draw without a solution) is
        left out, and ``n`` is the number of base points used.
        """
        from scipy.stats import norm

        z = float(norm.ppf(0.5 + self.conf_level / 2.0))
        out = {}
        for name in self.outputs:
            v = self.values[name]
            A, B, AB = v["A"], v["B"], v["AB"]
            ok = np.isfinite(A) & np.isfinite(B) & np.isfinite(AB).all(axis=1)
            A, B, AB = A[ok], B[ok], AB[ok]
            m = int(ok.sum())
            k = len(self.names)
            if m < 2:
                nan = np.full(k, np.nan)
                out[name] = {"S1": nan, "S1_conf": nan, "ST": nan, "ST_conf": nan, "n": m}
                continue
            centre = np.mean(np.concatenate([A, B, AB.ravel()]))
            A, B, AB = A - centre, B - centre, AB - centre
            s1, st = _sobol_indices(A, B, AB)
            if self.bootstrap:
                rng = np.random.default_rng(self.seed)
                draws = rng.integers(0, m, size=(self.bootstrap, m))
                boot = [_sobol_indices(A[r], B[r], AB[r]) for r in draws]
                s1_conf = z * np.std([b[0] for b in boot], axis=0, ddof=1)
                st_conf = z * np.std([b[1] for b in boot], axis=0, ddof=1)
            else:
                s1_conf = st_conf = np.full(k, np.nan)
            out[name] = {"S1": s1, "S1_conf": s1_conf, "ST": st, "ST_conf": st_conf, "n": m}
        return out

    def summary(self) -> str:
        idx = self.indices()
        w = max([len(n) for n in self.names] + [9])
        out = []
        for name in self.outputs:
            r = idx[name]
            out.append(f"{name}  ({r['n']} of {self.n} base points)")
            out.append(f"  {'parameter':{w}s} {'S1':>7s} {'ST':>7s}")
            for i, p in enumerate(self.names):
                out.append(f"  {p:{w}s} {r['S1'][i]:7.3f} {r['ST'][i]:7.3f}")
        return "\n".join(out)


def _sobol_indices(A, B, AB):
    """First-order (Saltelli 2010) and total (Jansen 1999) indices of each column of ``AB``."""
    var = np.var(np.concatenate([A, B]))
    if not var > 0:
        k = AB.shape[1]
        return np.full(k, np.nan), np.full(k, np.nan)
    s1 = np.mean(B[:, None] * (AB - A[:, None]), axis=0) / var
    st = 0.5 * np.mean((A[:, None] - AB) ** 2, axis=0) / var
    return s1, st


def _as_dict(value) -> dict:
    if isinstance(value, Mapping):
        return {str(k): float(v) for k, v in value.items()}
    arr = np.atleast_1d(np.asarray(value, dtype=float))
    if arr.size == 1:
        return {"result": float(arr[0])}
    return {f"result_{i}": float(x) for i, x in enumerate(arr)}


def sobol(run: Callable[[dict], Any], parameters: Mapping[str, Any], *, n: int = 1024,
          seed: int | None = 20260724, bootstrap: int = 200,
          conf_level: float = 0.95) -> SobolResult:
    """First-order and total Sobol indices of ``run`` over independent parameters.

    Saltelli's design: two matrices ``A`` and ``B`` of ``n`` points each, from a
    scrambled Sobol sequence of dimension ``2k`` (``scipy.stats.qmc``), carried
    to each parameter's distribution by its quantile function; and, for each
    parameter ``i``, the matrix ``AB_i``, ``A`` with column ``i`` taken from
    ``B``. That is ``n (k + 2)`` evaluations. ``n`` must be a power of two, which
    is what keeps a Sobol sequence balanced.

    ``run`` maps ``{name: value}`` to a number, a mapping of numbers or an array;
    :func:`solution_outputs` builds one from a model. ``parameters`` maps a name
    to a distribution, as for :func:`stability_map`. The first-order index
    ``S1`` is the share of the output's variance that the parameter explains
    alone; the total index ``ST`` adds every interaction it takes part in.
    The design and the estimators are those of ``cgelab.sobol`` and SALib.
    """
    from scipy.stats import qmc

    if not parameters:
        raise ValueError("no parameters to vary")
    if n < 2 or n & (n - 1):
        raise ValueError(f"n must be a power of two, at least 2, got {n}: a Sobol sequence "
                         f"is balanced only in powers of two")
    if not 0.0 < conf_level < 1.0:
        raise ValueError(f"conf_level must lie strictly between 0 and 1, got {conf_level}")
    if bootstrap < 0 or bootstrap == 1:
        raise ValueError("bootstrap must be zero (no intervals) or at least two resamples")
    names = list(parameters)
    dists = [_distribution(parameters[k]) for k in names]
    k = len(names)
    base = qmc.Sobol(d=2 * k, scramble=True, seed=seed).random(n)
    to_values = lambda u: np.column_stack(  # noqa: E731
        [np.asarray(d.ppf(u[:, j]), dtype=float) for j, d in enumerate(dists)])
    A, B = to_values(base[:, :k]), to_values(base[:, k:])

    outputs: list[str] = []
    values: dict = {}

    def collect(label, i, row, point):
        got = _as_dict(run(dict(zip(names, map(float, point)))))
        for name, y in got.items():
            if name not in values:
                outputs.append(name)
                values[name] = {"A": np.full(n, np.nan), "B": np.full(n, np.nan),
                                "AB": np.full((n, k), np.nan)}
            if label == "AB":
                values[name]["AB"][row, i] = y
            else:
                values[name][label][row] = y

    for j in range(n):
        collect("A", None, j, A[j])
        collect("B", None, j, B[j])
    for i in range(k):
        AB = A.copy()
        AB[:, i] = B[:, i]
        for j in range(n):
            collect("AB", i, j, AB[j])
    return SobolResult(names=names, outputs=outputs, n=n, A=A, B=B, values=values,
                       bootstrap=int(bootstrap), conf_level=float(conf_level), seed=seed)


def solution_outputs(model: Model, params: Params, *, irf: Sequence[Sequence] = (),
                     moments: Sequence[Sequence] = (), cov: Covariance = None,
                     guess: Point | None = None,
                     steady_fn: Callable[[Params], Any] | None = None,
                     qz_criterium: float | None = None,
                     relative_irf: bool = False) -> Callable[[dict], dict]:
    """A ``run`` for :func:`sobol`: impulse responses and moments at a parameter point.

    ``irf`` lists ``(variable, shock, horizons)`` and ``moments`` lists
    ``(variable1, variable2, lags)``, with the units and conventions of
    :func:`check_restrictions`: absolute deviations, a one-standard-deviation
    impulse with horizon 1 the impact period, the variance at lag 0 for a
    variable with itself and correlations otherwise. Each horizon and lag is one
    output, named as the restriction's label (``"y(4), e"``, ``"y, pi(1)"``).
    A point without a determinate solution gives NaN for every output.
    """
    from .estimation import unpack_point
    from .restrictions import Restriction, _restrictions, _values
    from .solve import solve
    from .steady import steady_state

    rows = [(*r, (-math.inf, math.inf)) for r in irf]
    mrows = [(*r, (-math.inf, math.inf)) for r in moments]
    full = _restrictions(model, rows, mrows)
    single = [Restriction(r.kind, r.first, r.second, (p,), r.lower, r.upper)
              for r in full for p in r.periods]
    labels = [r.label for r in single]
    if len(set(labels)) != len(labels):
        raise ValueError("an output is listed twice")

    def run(point: dict) -> dict:
        names = list(point)
        try:
            p, Sigma, _ = unpack_point(model, params, cov, {}, names,
                                       np.array([point[n] for n in names], dtype=float))
            ss = steady_fn(p) if steady_fn is not None else steady_state(model, guess, p)
            sol = solve(model, p, steady=ss, qz_criterium=qz_criterium)
            vals = _values(sol, Sigma, single, relative_irf)
        except (RuntimeError, ValueError, ArithmeticError, np.linalg.LinAlgError):
            return {lab: math.nan for lab in labels}
        return {lab: float(v[0]) for lab, v in zip(labels, vals)}

    run.outputs = labels  # type: ignore[attr-defined]
    return run
