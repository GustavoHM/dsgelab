"""Estimation by the method of moments: GMM and SMM, as Dynare's ``method_of_moments``.

The parameters are chosen to bring the model's moments close to the data's, in
the metric of a weighting matrix ``W``: the objective is ``(m_data - m_model)' W
(m_data - m_model)``. The moments are declared as Dynare's ``matched_moments``
block declares them, as products of observed variables with leads, lags and
powers: ``"c"`` is a mean, ``"c*iv"`` a second moment, ``"c*c(-1)"`` a first
autocovariance, ``"c^2"`` the same as ``"c*c"``.

* **GMM** computes means and second moments in closed form. First order uses the
  steady state and a Lyapunov equation; second order uses the pruned ABCD system.
* **SMM** simulates the model on a fixed set of draws, ``simulation_multiple``
  times as long as the data after a burn-in, and takes the same sample moments
  of the simulation as of the data.

The estimation runs in stages, one per entry of ``weighting``: ``"identity"``,
``"diagonal"`` or ``"optimal"`` (Newey and West's estimate of the inverse of the
long-run covariance of the moment conditions, with a Bartlett kernel), or a
matrix. The first stage evaluates the optimal matrix at the data's moments and
the later ones at the model's moments of the previous stage, as Dynare does.
Standard errors come from the Jacobian of the model's moments, by central
differences, with the sandwich formula unless the last weighting matrix is the
optimal one; Hansen's J test checks the overidentifying restrictions.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike
from scipy import linalg, optimize, stats

from .estimation import unpack_point
from .kalman import StateSpace, state_space
from .model import Covariance, Model, Params, Point, get_param

__all__ = ["MomentsResult", "method_of_moments", "parse_moment"]

_FACTOR = re.compile(r"^\s*([A-Za-z_]\w*)\s*(?:\(\s*([+-]?\d+)\s*\))?\s*(?:\^\s*(\d+))?\s*$")


# ==================================================================== moments
def parse_moment(text: str) -> tuple[list[str], list[int], list[int]]:
    """``"c*iv(-1)"`` -> ``(["c", "iv"], [0, -1], [1, 1])``: variables, leads and powers."""
    names, leads, powers = [], [], []
    for factor in text.split("*"):
        m = _FACTOR.match(factor)
        if not m:
            raise ValueError(f"cannot read the moment {text!r}: write products of "
                             f"variables, as 'c*iv(-1)' or 'c^2'")
        names.append(m.group(1))
        leads.append(int(m.group(2) or 0))
        powers.append(int(m.group(3) or 1))
    return names, leads, powers


def _normalised(spec):
    """Dynare's ``matched_moments_block``: powers spelled out, the latest date at 0."""
    names, leads, powers = spec
    n2 = [n for n, p in zip(names, powers) for _ in range(p)]
    l2 = [l for l, p in zip(leads, powers) for _ in range(p)]
    top = max(l2)
    l2 = [l - top for l in l2]
    order = sorted(range(len(l2)), key=lambda i: -l2[i])          # stable, as MATLAB's sort
    return [n2[i] for i in order], [l2[i] for i in order], [1] * len(order)


def _unique(normal):
    """Indices of the moments without duplicates; the first of each, in the declared order."""
    seen, keep = set(), []
    for i, s in enumerate(normal):
        key = (sum(s[2]), tuple(s[0]), tuple(s[1]), tuple(s[2]))
        if key not in seen:
            seen.add(key)
            keep.append(i)
    return keep


def _sample_moments(Y: np.ndarray, specs, index: Mapping[str, int]) -> tuple[np.ndarray, np.ndarray]:
    """Dynare's ``get_data_moments``: each moment's mean, and its series with gaps filled.

    A lead or a lag shortens the product; missing values are skipped in the mean
    and replaced by it in the series, which feeds the optimal weighting matrix.
    """
    T = Y.shape[0]
    out = np.full(len(specs), np.nan)
    series = np.full((T, len(specs)), np.nan)
    for j, (names, leads, powers) in enumerate(specs):
        prod = np.ones(T)
        for n, l, p in zip(names, leads, powers):
            y = np.full(T, np.nan)
            y[max(-l, 0):T - max(l, 0)] = Y[max(l, 0):T + min(l, 0), index[n]] ** p
            prod = prod * y
        out[j] = np.nanmean(prod)
        prod[np.isnan(prod)] = out[j]
        series[:, j] = prod
    return out, series


def _label(spec) -> str:
    names, leads, powers = spec
    parts = []
    for n, l, p in zip(names, leads, powers):
        parts.append(n + (f"({l:+d})" if l else "") + (f"^{p}" if p != 1 else ""))
    return "*".join(parts)


def optimal_weighting_matrix(series: np.ndarray, moments: np.ndarray, lags: int) -> np.ndarray:
    """Newey and West's inverse long-run covariance of the moment conditions (Bartlett kernel)."""
    h = series - moments[None, :]
    T = h.shape[0]
    S = h.T @ h / T
    for v in range(1, lags + 1):
        G = h[:-v].T @ h[v:] / T
        S = S + (1 - v / (lags + 1)) * (G + G.T)
    W = np.linalg.solve(S, np.eye(S.shape[0]))
    return 0.5 * (W + W.T)


# ===================================================================== result
@dataclass
class MomentsResult:
    """An estimate by the method of moments. See :func:`method_of_moments`.

    Attributes
    ----------
    method : ``"GMM"`` or ``"SMM"``.
    names, x, values : the estimated parameters and their estimates.
    stderr, cov : standard errors and asymptotic covariance of the estimates.
    objective : the moment distance at the estimate, with the last weighting matrix.
    moments, moment_labels : the matched moments, in the order used.
    data_moments, model_moments : their values in the data and in the model at the estimate.
    weighting : the weighting matrix of the last stage.
    stages : one dict per stage, with ``weighting``, ``x`` and ``objective``.
    J : Hansen's test of the overidentifying restrictions, ``{"stat", "dof", "p"}``,
        or ``None`` when there are as many moments as parameters.
    nobs : observations used.
    """

    method: str
    names: list[str]
    x: np.ndarray
    stderr: dict[str, float]
    cov: np.ndarray
    objective: float
    moments: list[str]
    data_moments: np.ndarray
    model_moments: np.ndarray
    weighting: np.ndarray
    stages: list[dict]
    J: dict | None
    nobs: int
    success: bool = True
    message: str = ""

    @property
    def values(self) -> dict[str, float]:
        return dict(zip(self.names, map(float, self.x)))

    def summary(self) -> str:
        out = [f"{self.method}, {len(self.moments)} moments, {len(self.names)} parameters, "
               f"{self.nobs} observations, {len(self.stages)} stage(s)",
               f"  {'parameter':16s} {'estimate':>12s} {'s.e.':>12s}"]
        for n, v in zip(self.names, self.x):
            out.append(f"  {n:16s} {v:12.6g} {self.stderr[n]:12.4g}")
        out.append(f"  objective {self.objective:.6g}")
        if self.J is not None:
            out.append(f"  J test {self.J['stat']:.4g} with {self.J['dof']} degrees of "
                       f"freedom, p-value {self.J['p']:.4g}")
        return "\n".join(out)


# ===================================================================== entry
def method_of_moments(model: Model, data: Mapping[str, ArrayLike] | ArrayLike, *,
                      params: Params, free: Sequence[str] | Mapping[str, float],
                      observables: Sequence[str], moments: Sequence[str],
                      method: str = "GMM", order: int = 1,
                      bounds: Mapping[str, tuple[float, float]] | None = None,
                      cov: Covariance = None,
                      transforms: str | Mapping[str, str] | None = None,
                      guess: Point | None = None,
                      steady_fn: Callable[[Params], Any] | None = None,
                      qz_criterium: float | None = None,
                      weighting: Sequence[str | ArrayLike] = ("diagonal", "optimal"),
                      bartlett_lag: int = 20, prefilter: bool = False,
                      scaling: float = 1.0, first_obs: int = 1, nobs: int | None = None,
                      simulation_multiple: int = 7, burnin: int = 500,
                      seed: int | None = 0, shocks: ArrayLike | None = None,
                      se_step: float = 1e-5, optimize_: bool = True,
                      options: Mapping[str, Any] | None = None) -> MomentsResult:
    """Estimate ``free`` by GMM or SMM on the matched ``moments`` of the ``observables``.

    Parameters
    ----------
    model, params : the model and the parameters that stay fixed.
    data : a mapping from each observable to its series, or a ``(T, n_obs)`` array
        in the order of ``observables``. Missing values are ``nan``.
    free : the estimated names: parameters, ``"stderr e"`` for a shock's standard
        deviation and ``"corr e1, e2"`` for the correlation of two shocks. A
        mapping also gives the starting point.
    observables : the observed variables, in the units of the data.
    moments : the matched moments, as Dynare's ``matched_moments`` block writes
        them (``"c"``, ``"c*iv"``, ``"c*c(-1)"``, ``"c^2"``).
    method : ``"GMM"`` (closed-form moments) or ``"SMM"`` (moments of a simulation).
    order : perturbation order, 1 or 2. Second order uses the pruned decision rule.
    bounds : ``{name: (lower, upper)}`` for the estimated names.
    cov : the shocks' covariance at the start (standard deviations or a matrix).
    transforms : ``"level"`` or ``"log"`` per observable, as in :func:`dsgelab.state_space`.
    guess, steady_fn : how the steady state is found at each trial point.
    weighting : one weighting matrix per stage: ``"identity"``, ``"diagonal"``,
        ``"optimal"`` or an array. Dynare's default, ``("diagonal", "optimal")``.
    bartlett_lag : lags of the Bartlett kernel in the optimal weighting matrix.
    prefilter : demean the data (and the simulation) and match centred second
        moments; the means are still those of the raw series.
    scaling : multiplies the weighting matrix in the objective.
    first_obs, nobs : the sample, with ``first_obs`` counted from 1.
    simulation_multiple, burnin, seed, shocks : SMM's simulation. ``shocks`` gives the
        standard normal draws, ``(burnin + simulation_multiple * nobs, n_shocks)``;
        otherwise they are drawn with ``seed``. The same draws serve every trial.
    se_step : step of the central differences for the standard errors.
    optimize_ : ``False`` evaluates every stage at the starting point, as Dynare's
        ``mode_compute=0``.
    options : passed to :func:`scipy.optimize.least_squares`.
    """
    method = method.upper()
    if method not in ("GMM", "SMM"):
        raise ValueError(f"method must be 'GMM' or 'SMM', not {method!r}")
    if order not in (1, 2):
        raise ValueError(f"order must be 1 or 2, got {order!r}")
    observables = list(observables)
    names = list(free)
    index = {n: i for i, n in enumerate(observables)}
    declared = [parse_moment(m) if isinstance(m, str) else m for m in moments]
    for spec in declared:
        missing = [n for n in spec[0] if n not in index]
        if missing:
            raise ValueError(f"the moment {_label(spec)!r} involves {missing}, "
                             f"which are not observables")
    if method == "GMM" and any(sum(sp[2]) > 2 for sp in declared):
        raise ValueError("GMM matches moments up to the second; use SMM for higher ones")
    normal = [_normalised(sp) for sp in declared]
    # duplicates are found in the normalised form; GMM uses that form, SMM the declared one
    specs = [normal[i] if method == "GMM" else declared[i] for i in _unique(normal)]
    first = np.array([sum(s[2]) == 1 for s in specs])

    Y = _data_matrix(data, observables)[first_obs - 1:]
    if nobs is not None:
        Y = Y[:nobs]
    T = Y.shape[0]
    Yc = Y - np.nanmean(Y, axis=0) if prefilter else Y
    m_first, s_first = _sample_moments(Y, [s for s, f in zip(specs, first) if f], index)
    m_other, s_other = _sample_moments(Yc, [s for s, f in zip(specs, first) if not f], index)
    data_m = np.empty(len(specs))
    series = np.empty((T, len(specs)))
    data_m[first], data_m[~first] = m_first, m_other
    series[:, first], series[:, ~first] = s_first, s_other

    theta0 = _start(model, params, cov, names, free)
    lo = np.array([(bounds or {}).get(n, (-np.inf, np.inf))[0] for n in names], float)
    hi = np.array([(bounds or {}).get(n, (-np.inf, np.inf))[1] for n in names], float)

    from .solve import solve
    from .steady import steady_state
    last_ss = {"values": guess}

    def solution_at(theta):
        p, S, _ = unpack_point(model, params, cov, {}, names, theta)
        ss = steady_fn(p) if steady_fn is not None else steady_state(model, last_ss["values"], p)
        sol = solve(model, p, steady=ss, qz_criterium=qz_criterium)
        last_ss["values"] = ss.values
        return sol, S

    if method == "SMM":
        long = int(round(simulation_multiple * T))
        n_eps = len(model.shocks)
        if shocks is None:
            draws = np.random.default_rng(seed).standard_normal((long + burnin, n_eps))
        else:
            draws = np.asarray(shocks, float)
            if draws.shape != (long + burnin, n_eps):
                raise ValueError(f"shocks must be {(long + burnin, n_eps)}, got {draws.shape}")
        vcf = 1 + 1 / simulation_multiple
    else:
        long, draws, vcf = 0, None, 1.0

    def model_moments(theta) -> np.ndarray:
        sol, S = solution_at(theta)
        if order == 2:
            from .secondorder import solve_second_order
            second = solve_second_order(sol, cov=S)
            if method == "GMM":
                return _gmm_moments_second(second, observables, transforms, specs, index, prefilter)
            return _smm_moments_second(second, observables, transforms, specs, first, index,
                                       draws, long, prefilter)
        s = state_space(sol, observables, transforms=transforms, cov=S)
        if method == "GMM":
            return _gmm_moments(s, specs, index, prefilter)
        return _smm_moments(s, specs, first, index, draws, long, prefilter)

    def residuals(theta, Sw) -> np.ndarray:
        try:
            m = model_moments(theta)
        except Exception:      # no steady state, no determinate solution
            return np.full(len(specs), 1e7)
        r = math.sqrt(scaling) * Sw @ (data_m - m)
        return r if np.all(np.isfinite(r)) else np.full(len(specs), 1e7)

    if not weighting:
        raise ValueError("weighting needs at least one stage")
    x = np.clip(theta0, lo, hi)
    stages, m_model, W, w_opt = [], None, None, False
    message, success = "", True
    for k, wspec in enumerate(weighting):
        w_opt = False
        if isinstance(wspec, str):
            kind = wspec.lower()
            at = data_m if k == 0 else m_model
            assert at is not None
            if kind in ("identity", "identity_matrix"):
                W = np.eye(len(specs))
            elif kind == "diagonal":
                W = np.diag(np.diag(optimal_weighting_matrix(series, at, bartlett_lag)))
            elif kind == "optimal":
                W = optimal_weighting_matrix(series, at, bartlett_lag)
                w_opt = k > 0
            else:
                raise ValueError(f"unknown weighting {wspec!r}")
        else:
            W = np.asarray(wspec, float)
            if W.shape != (len(specs),) * 2:
                raise ValueError(f"a weighting matrix must be {len(specs)} x {len(specs)}")
        Sw = linalg.cholesky(W)                      # upper: W = Sw' Sw, as MATLAB's chol
        if optimize_:
            opts = dict(method="trf", x_scale="jac", ftol=1e-12, xtol=1e-12, gtol=1e-12,
                        max_nfev=5000)
            opts.update(options or {})
            fit = optimize.least_squares(residuals, x, bounds=(lo, hi), args=(Sw,), **opts)
            x, success, message = fit.x, bool(fit.success), str(fit.message)
        r = residuals(x, Sw)
        m_model = model_moments(x)
        stages.append(dict(weighting=W, x=x.copy(), objective=float(r @ r)))
    objective = stages[-1]["objective"]
    assert m_model is not None and W is not None

    D = np.zeros((len(specs), len(names)))
    has_derivatives = True
    for j in range(len(names)):
        xp, xm = x.copy(), x.copy()
        xp[j] += se_step
        xm[j] -= se_step
        try:
            if not (lo[j] <= xm[j] and xp[j] <= hi[j]):
                raise ValueError("a bound")
            D[:, j] = (model_moments(xp) - model_moments(xm)) / (2 * se_step)
        except Exception as err:   # a bound, or a point without a determinate solution
            message = (f"{message}; " if message else "") + \
                f"no standard errors: the difference in {names[j]!r} reaches {err}"
            has_derivatives = False
            break
    Teff = T / vcf                 # SMM's simulation noise inflates the variance by vcf
    if not has_derivatives:        # as Dynare: no column, no covariance
        V = np.full((len(names), len(names)), np.nan)
    elif w_opt:
        V = np.linalg.inv(D.T @ W @ D) / Teff
    else:
        S = np.linalg.inv(optimal_weighting_matrix(series, m_model, bartlett_lag))
        A = np.linalg.inv(D.T @ W @ D)
        V = A @ D.T @ W @ S @ W @ D @ A / Teff
    se = np.sqrt(np.maximum(np.diag(V), 0.0))

    J = None
    if len(specs) > len(names):
        if w_opt:
            fval = objective
        else:
            Swo = linalg.cholesky(optimal_weighting_matrix(series, m_model, bartlett_lag))
            r = residuals(x, Swo)
            fval = float(r @ r)
        stat = Teff * fval / scaling
        dof = len(specs) - len(names)
        J = dict(stat=stat, dof=dof, p=float(stats.chi2.sf(stat, dof)))
    return MomentsResult(method=method, names=names, x=x, stderr=dict(zip(names, map(float, se))),
                         cov=V, objective=objective, moments=[_label(s) for s in specs],
                         data_moments=data_m, model_moments=m_model, weighting=W,
                         stages=stages, J=J, nobs=T, success=success, message=message)


# ====================================================================== pieces
def _data_matrix(data, observables) -> np.ndarray:
    if isinstance(data, Mapping):
        cols: list[np.ndarray] = [np.asarray(data[n], dtype=float).ravel() for n in observables]
        if len({c.size for c in cols}) != 1:
            raise ValueError("every observed series must have the same length")
        return np.column_stack(cols)
    Y = np.asarray(data, dtype=float)
    Y = Y.reshape(-1, 1) if Y.ndim == 1 else Y
    if Y.shape[1] != len(observables):
        raise ValueError(f"data has {Y.shape[1]} columns for {len(observables)} observables")
    return Y


def _start(model, params, cov, names, free) -> np.ndarray:
    if isinstance(free, Mapping):
        return np.array([float(free[n]) for n in names])
    from .solution import shock_covariance
    S = shock_covariance(model.shocks, cov)
    out = []
    for n in names:
        if n.startswith("stderr "):
            i = model.shocks.index(n[7:].strip())
            out.append(math.sqrt(S[i, i]))
        elif n.startswith("corr "):
            a, b = (x.strip() for x in n[5:].split(","))
            i, j = model.shocks.index(a), model.shocks.index(b)
            out.append(S[i, j] / math.sqrt(S[i, i] * S[j, j]))
        else:
            out.append(float(get_param(params, n)))
    theta = np.array(out)
    if not np.all(np.isfinite(theta)):
        bad = [n for n, v in zip(names, theta) if not np.isfinite(v)]
        raise ValueError(f"no starting value for {bad}: pass free as a mapping")
    return theta


def _gmm_moments(s: StateSpace, specs, index, prefilter) -> np.ndarray:
    """Means and second moments of the observables from the first-order solution."""
    V = linalg.solve_discrete_lyapunov(s.T, s.R @ s.Sigma @ s.R.T)
    lags = max([-min(sp[1]) for sp in specs] + [0])
    G = [s.Z @ V @ s.Z.T + s.H]
    TjV = V
    for _ in range(lags):
        TjV = s.T @ TjV
        G.append(s.Z @ TjV @ s.Z.T)
    out = np.empty(len(specs))
    for j, (names, leads, _) in enumerate(specs):
        a = index[names[0]]
        if len(names) == 1:
            out[j] = s.d[a]
            continue
        b, lag = index[names[1]], -leads[1]
        out[j] = G[lag][a, b] + (0.0 if prefilter else s.d[a] * s.d[b])
    return out


def _commutation(n: int) -> np.ndarray:
    permutation = np.arange(n * n).reshape(n, n).T.ravel()
    return np.eye(n * n)[permutation]


def _second_order_system(second, lags=0):
    """Pruned ABCD system of Andreasen, Fernández-Villaverde and Rubio-Ramírez."""
    from .secondorder import _dynare_decision_rule
    first = second.first
    model = first.model
    nk, ne, n = model.n_states, model.n_shocks, model.n_vars
    rule = _dynare_decision_rule(second)
    P, Q = rule.g_k[:nk], rule.g_u[:nk]
    Sigma = second.Sigma
    hkk = rule.g_kk[:nk].reshape(nk, nk * nk, order="F")
    # Mixed innovations use kron(xf, u): the shock index varies fastest.
    hku = rule.g_ku[:nk].transpose(0, 2, 1).reshape(nk, nk * ne, order="F")
    huu = rule.g_uu[:nk].reshape(nk, ne * ne, order="F")
    gkk = rule.g_kk.reshape(n, nk * nk, order="F")
    gku = rule.g_ku.transpose(0, 2, 1).reshape(n, nk * ne, order="F")
    guu = rule.g_uu.reshape(n, ne * ne, order="F")
    vec_sigma = Sigma.reshape(-1, order="F")
    Vf = linalg.solve_discrete_lyapunov(P, Q @ Sigma @ Q.T)

    z1 = slice(0, nk)
    z2 = slice(nk, 2 * nk)
    z3 = slice(2 * nk, 2 * nk + nk * nk)
    nz = 2 * nk + nk * nk
    i1 = slice(0, ne)
    i2 = slice(ne, ne + ne * ne)
    i3 = slice(ne + ne * ne, ne + ne * ne + nk * ne)
    ni = ne + ne * ne + nk * ne
    A = np.zeros((nz, nz))
    B = np.zeros((nz, ni))
    c = np.zeros(nz)
    A[z1, z1] = P
    A[z2, z2] = P
    A[z2, z3] = 0.5 * hkk
    A[z3, z3] = np.kron(P, P)
    B[z1, i1] = Q
    B[z2, i2] = 0.5 * huu
    B[z2, i3] = hku
    B[z3, i2] = np.kron(Q, Q)
    B[z3, i3] = (np.eye(nk * nk) + _commutation(nk)) @ np.kron(P, Q)
    c[z2] = 0.5 * rule.g_s2[:nk] + 0.5 * huu @ vec_sigma
    c[z3] = np.kron(Q, Q) @ vec_sigma

    C = np.zeros((n, nz))
    D = np.zeros((n, ni))
    C[:, z1] = rule.g_k
    C[:, z2] = rule.g_k
    C[:, z3] = 0.5 * gkk
    D[:, i1] = rule.g_u
    D[:, i2] = 0.5 * guu
    D[:, i3] = gku
    d = 0.5 * rule.g_s2 + 0.5 * guu @ vec_sigma

    Vinov = np.zeros((ni, ni))
    Vinov[i1, i1] = Sigma
    fourth = np.empty((ne, ne, ne, ne))
    for a in range(ne):
        for b in range(ne):
            for k in range(ne):
                for ell in range(ne):
                    fourth[a, b, k, ell] = (Sigma[a, b] * Sigma[k, ell]
                                             + Sigma[a, k] * Sigma[b, ell]
                                             + Sigma[a, ell] * Sigma[b, k])
    M4 = fourth.reshape(ne * ne, ne * ne, order="F")
    Vinov[i2, i2] = M4 - np.outer(vec_sigma, vec_sigma)
    Vinov[i3, i3] = np.kron(Vf, Sigma)
    Vz = linalg.solve_discrete_lyapunov(A, B @ Vinov @ B.T)

    Ez = np.zeros(nz)
    Ez[z2] = np.linalg.solve(np.eye(nk) - P, 0.5 * hkk @ Vf.reshape(-1, order="F") + c[z2])
    Ez[z3] = Vf.reshape(-1, order="F")
    Ey = C @ Ez + d
    Vy = C @ Vz @ C.T + D @ Vinov @ D.T
    autocov = []
    tmp = A @ Vz @ C.T + B @ Vinov @ D.T
    Ai = np.eye(nz)
    for _ in range(lags):
        autocov.append(C @ Ai @ tmp)
        Ai = Ai @ A
    return dict(A=A, B=B, c=c, C=C, D=D, d=d, Vinov=Vinov, Ey=Ey, Vy=Vy,
                Ez=Ez, Vz=Vz, autocov=autocov, nk=nk, vec_sigma=vec_sigma)


def _observation_conversion(first, observables, transforms):
    model = first.model
    if isinstance(transforms, str) or transforms is None:
        tr = [transforms or "level"] * len(observables)
    else:
        tr = [transforms.get(n, "level") for n in observables]
    rows = [model.position(n) for n in observables]
    base = np.zeros(len(observables))
    scale = np.zeros(len(observables))
    for i, (name, kind) in enumerate(zip(observables, tr)):
        j = model.position(name)
        center = first.coords.center[j]
        if kind == "level":
            base[i], scale[i] = center, first.coords.scale[j]
        elif kind == "log":
            if center <= 0:
                raise ValueError(f"{name!r} has steady state {center}: a log "
                                 "observation needs a positive one")
            base[i], scale[i] = np.log(center), 1.0
        else:
            raise ValueError(f"transform must be 'level' or 'log', got {kind!r}")
    return rows, base, scale


def _gmm_moments_second(second, observables, transforms, specs, index, prefilter) -> np.ndarray:
    lags = max([-min(sp[1]) for sp in specs] + [0])
    system = _second_order_system(second, lags)
    rows, base, scale = _observation_conversion(second.first, observables, transforms)
    mean = base + scale * system["Ey"][rows]
    Vy = system["Vy"][np.ix_(rows, rows)] * np.outer(scale, scale)
    autocov = [G[np.ix_(rows, rows)] * np.outer(scale, scale) for G in system["autocov"]]
    out = np.empty(len(specs))
    for j, (names, leads, _) in enumerate(specs):
        a = index[names[0]]
        if len(names) == 1:
            out[j] = mean[a]
            continue
        b, lag = index[names[1]], -leads[1]
        G = Vy if lag == 0 else autocov[lag - 1]
        out[j] = G[a, b] + (0.0 if prefilter else mean[a] * mean[b])
    return out


def _smm_moments_second(second, observables, transforms, specs, first, index,
                        draws, long, prefilter) -> np.ndarray:
    system = _second_order_system(second)
    rows, base, scale = _observation_conversion(second.first, observables, transforms)
    Sigma = second.Sigma
    live = np.flatnonzero(np.diag(Sigma) > 0)
    eps = np.zeros_like(draws)
    if live.size:
        U = linalg.cholesky(Sigma[np.ix_(live, live)])
        eps[:, live] = draws[:, live] @ U
    A, B, c = system["A"], system["B"], system["c"]
    C, D, d = system["C"], system["D"], system["d"]
    nk, vec_sigma = system["nk"], system["vec_sigma"]
    z = np.zeros(A.shape[0])
    Ysim = np.empty((draws.shape[0], len(observables)))
    for t in range(draws.shape[0]):
        u = eps[t]
        inov = np.concatenate([u, np.outer(u, u).reshape(-1, order="F") - vec_sigma,
                               np.outer(z[:nk], u).ravel()])
        Ysim[t] = base + scale * (d + C @ z + D @ inov)[rows]
        z = c + A @ z + B @ inov
    Ysim = Ysim[-long:]
    out = np.empty(len(specs))
    out[first] = _sample_moments(Ysim, [sp for sp, f in zip(specs, first) if f], index)[0]
    if prefilter:
        Ysim = Ysim - Ysim.mean(axis=0)
    out[~first] = _sample_moments(Ysim, [sp for sp, f in zip(specs, first) if not f], index)[0]
    return out


def _smm_moments(s: StateSpace, specs, first, index, draws, long, prefilter) -> np.ndarray:
    """The data's sample moments, on a simulation from the steady state with fixed draws."""
    live = np.flatnonzero(np.diag(s.Sigma) > 0)
    eps = np.zeros_like(draws)
    if live.size:
        U = linalg.cholesky(s.Sigma[np.ix_(live, live)])          # upper, as MATLAB's chol
        eps[:, live] = draws[:, live] @ U
    k = np.zeros(s.T.shape[0])
    Ysim = np.empty((draws.shape[0], s.Z.shape[0]))
    for t in range(draws.shape[0]):
        k = s.T @ k + s.R @ eps[t]
        Ysim[t] = s.d + s.Z @ k
    Ysim = Ysim[-long:]
    out = np.empty(len(specs))
    out[first] = _sample_moments(Ysim, [sp for sp, f in zip(specs, first) if f], index)[0]
    if prefilter:
        Ysim = Ysim - Ysim.mean(axis=0)
    out[~first] = _sample_moments(Ysim, [sp for sp, f in zip(specs, first) if not f], index)[0]
    return out
