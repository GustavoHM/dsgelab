"""Local-projection impulse responses with a joint HAC covariance matrix.

The observed shock is dated at t and the response at t+h. Horizon zero therefore
corresponds to period 1 in :func:`dsgelab.irf_matching`. Each horizon is a
separate OLS regression, as in Jordà (2005). Cross-equation covariances use the
overlapping influence functions and a Bartlett kernel, so the result can be used
directly as external targets for IRF matching.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike

__all__ = ["LocalProjectionResult", "local_projection"]


@dataclass
class LocalProjectionResult:
    """Local-projection IRFs and their joint covariance.

    ``irfs[h, i]`` is variable i's response at horizon h. ``covariance`` follows
    ``irfs.flatten(order='F')``: horizon first, then variable. ``stderr`` has the
    same shape as ``irfs``. ``coefficients`` and ``regressor_names`` retain the
    complete regressions for inspection.
    """

    shock: str
    variables: list[str]
    irfs: np.ndarray
    covariance: np.ndarray
    stderr: np.ndarray
    coefficients: list[np.ndarray]
    regressor_names: list[str]
    lags: int
    hac_lags: int

    def targets(self, shock: str | None = None, periods: Sequence[int] | None = None):
        """Return ``(targets, covariance)`` ready for :func:`dsgelab.irf_matching`.

        Periods are counted from 1, so the default includes the impact response.
        """
        if shock is None:
            shock = self.shock
        H, nv = self.irfs.shape
        use = list(range(1, H + 1)) if periods is None else [int(h) for h in periods]
        if any(h < 1 or h > H for h in use):
            raise ValueError(f"periods must be between 1 and {H}")
        indices = [(h - 1) * nv + i for h in use for i in range(nv)]
        # irf_matching sorts by variable before horizon, as MATLAB does; the
        # covariance is reordered here into that same canonical order.
        order = sorted(range(len(indices)),
                       key=lambda k: (indices[k] % nv, indices[k] // nv))
        indices = [indices[k] for k in order]
        targets = [(v, shock, h, float(self.irfs[h - 1, i]))
                   for h in use for i, v in enumerate(self.variables)]
        targets = [targets[k] for k in order]
        return targets, self.covariance[np.ix_(indices, indices)]


def local_projection(data: Mapping[str, ArrayLike] | ArrayLike, shock: str, *,
                     variables: Sequence[str] | None = None,
                     horizon: int, lags: int = 1,
                     controls: Sequence[str] = (),
                     constant: bool = True,
                     hac_lags: int | None = None) -> LocalProjectionResult:
    """Estimate local projections for an observed structural ``shock``.

    ``data`` is a mapping of equally sized series, or an array whose columns are
    named by ``variables`` (then ``shock`` must name one of those columns). Each
    regression includes the current shock, current controls, a constant by
    default, and ``lags`` lags of every response, the shock and every control.
    Set ``lags=0`` for no lagged controls. ``hac_lags`` defaults to ``horizon``;
    it applies jointly within and across all response/horizon equations.

    The function does not identify a structural shock from reduced-form
    residuals: identification is supplied by the observed series passed as
    ``shock``. For recursive identification, estimate that series first.
    """
    if isinstance(horizon, (bool, np.bool_)) or int(horizon) != horizon or horizon < 0:
        raise ValueError("horizon must be a nonnegative integer")
    horizon = int(horizon)
    if isinstance(lags, (bool, np.bool_)) or int(lags) != lags or lags < 0:
        raise ValueError("lags must be a nonnegative integer")
    lags = int(lags)
    if isinstance(data, Mapping):
        columns = {k: np.asarray(v, dtype=float).ravel() for k, v in data.items()}
        if variables is None:
            variables = [v for v in columns if v != shock and v not in controls]
        names = list(variables)
        if shock not in columns:
            raise ValueError(f"no shock series {shock!r}")
        Y = np.column_stack([columns[v] for v in names])
        z = columns[shock]
        C = np.column_stack([columns[c] for c in controls]) if controls else np.empty((z.size, 0))
    else:
        if variables is None:
            raise ValueError("array data needs variables= to name its columns")
        names = list(variables)
        A = np.asarray(data, dtype=float)
        if A.ndim != 2 or A.shape[1] != len(names):
            raise ValueError("array data must have one column per entry of variables")
        if shock not in names:
            raise ValueError("with array data, shock must be one of variables")
        j = names.index(shock)
        z = A[:, j]
        keep = [i for i in range(len(names)) if i != j and names[i] not in controls]
        control_names = [c for c in controls if c in names]
        if len(control_names) != len(controls):
            raise ValueError("every control must be a column named by variables")
        ci = [names.index(c) for c in control_names]
        C = A[:, ci] if ci else np.empty((A.shape[0], 0))
        Y = A[:, keep]
        names = [names[i] for i in keep]
    controls = list(controls)
    if not names:
        raise ValueError("no response variables")
    if len(set(names)) != len(names) or shock in names or set(controls) & set(names):
        raise ValueError("responses, shock and controls must be distinct")
    T = z.size
    if Y.shape[0] != T or C.shape[0] != T:
        raise ValueError("every series must have the same length")
    if T <= horizon + lags + 1:
        raise ValueError("not enough observations for this horizon and lag count")
    if not np.all(np.isfinite(Y)) or not np.all(np.isfinite(z)) or not np.all(np.isfinite(C)):
        raise ValueError("local_projection does not accept missing or infinite observations")

    base = np.column_stack([Y, z, C])
    base_names = names + [shock] + controls
    regressor_names = (["constant"] if constant else []) + [shock] + controls
    shock_col = int(constant)
    for lag in range(1, lags + 1):
        regressor_names += [f"{v}(-{lag})" for v in base_names]
    k = len(regressor_names)
    all_t = np.arange(lags, T)
    pieces = []
    if constant:
        pieces.append(np.ones((len(all_t), 1)))
    pieces += [z[all_t, None], C[all_t]]
    for lag in range(1, lags + 1):
        pieces.append(base[all_t - lag])
    X_all = np.column_stack(pieces)
    H = horizon + 1
    nv = len(names)
    responses = np.empty((H, nv))
    coefficients = []
    influence = np.zeros((T, H * nv))
    for i in range(nv):
        for h in range(H):
            t = np.arange(lags, T - h)
            X = X_all[:len(t)]
            y = Y[t + h, i]
            beta, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
            if rank < k:
                raise ValueError(f"rank-deficient local projection for {names[i]!r} at horizon {h}")
            bread = np.linalg.inv(X.T @ X)
            u = y - X @ beta
            influence[t, h * nv + i] = (X @ bread[:, shock_col]) * u
            responses[h, i] = beta[shock_col]
            coefficients.append(beta)

    L = horizon if hac_lags is None else int(hac_lags)
    if L < 0 or L >= T - lags - horizon:
        raise ValueError("hac_lags must be nonnegative and smaller than the shortest sample")
    V = influence.T @ influence
    for lag in range(1, L + 1):
        G = influence[:-lag].T @ influence[lag:]
        V += (1 - lag / (L + 1)) * (G + G.T)
    V = 0.5 * (V + V.T)
    variance = np.diag(V).reshape(H, nv, order="F")
    stderr = np.sqrt(np.where(variance > 0, variance, np.nan))
    return LocalProjectionResult(shock=shock, variables=names, irfs=responses,
                                 covariance=V, stderr=stderr, coefficients=coefficients,
                                 regressor_names=regressor_names, lags=lags, hac_lags=L)
