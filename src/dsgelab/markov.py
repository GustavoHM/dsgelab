"""Finite Markov chains, and the discretisation of an AR(1) into one.

``tauchen`` (Tauchen, 1986) and ``rouwenhorst`` (Rouwenhorst, 1995; Kopecky and Suen, 2010)
approximate ``y_t = (1 - rho) mean + rho y_{t-1} + sigma e_t``, with ``e_t ~ N(0, 1)``, by a chain
on ``n`` nodes. Tauchen spaces the nodes evenly over ``n_std`` unconditional standard deviations on
each side of the mean and gives each node the probability of the interval around it. Rouwenhorst
spaces them over ``sqrt(n - 1)`` standard deviations and builds the transition matrix by a
recursion in ``n``. Its chain has the mean, the variance and the first autocorrelation of the AR(1)
exactly, for every ``n``, and it stays accurate when ``rho`` is close to one, where Tauchen's does
not.

The stationary distribution is computed by the GTH algorithm (Grassmann, Taksar and Heyman, 1985),
which only adds and divides nonnegative numbers and so loses no accuracy in the small
probabilities of the tails. The grids and the transition matrices are the ones QuantEcon's
``tauchen`` and ``rouwenhorst`` build, with QuantEcon's ``mu`` equal to ``(1 - rho) mean``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import ndtr

__all__ = ["MarkovChain", "tauchen", "rouwenhorst"]


@dataclass
class MarkovChain:
    """A chain on the values ``nodes``, with ``transition[i, j]`` the probability of ``i -> j``."""

    nodes: np.ndarray        # (n,)
    transition: np.ndarray   # (n, n), rows sum to one

    def __post_init__(self) -> None:
        self.nodes = np.asarray(self.nodes, dtype=float)
        self.transition = np.asarray(self.transition, dtype=float)
        n = self.nodes.shape[0]
        if self.nodes.ndim != 1 or n < 1:
            raise ValueError("nodes must be a nonempty vector")
        if self.transition.shape != (n, n):
            raise ValueError(f"transition must be {n} x {n}, one row and one column per node")
        if np.any(self.transition < 0) or not np.allclose(self.transition.sum(axis=1), 1.0,
                                                           rtol=0.0, atol=1e-12):
            raise ValueError("transition must be nonnegative, with rows that sum to one")

    @property
    def n(self) -> int:
        return self.nodes.shape[0]

    @property
    def stationary(self) -> np.ndarray:
        """The stationary distribution, of an irreducible chain."""
        return _gth(self.transition)

    @property
    def mean(self) -> float:
        return float(self.stationary @ self.nodes)

    @property
    def variance(self) -> float:
        pi = self.stationary
        return float(pi @ (self.nodes - pi @ self.nodes) ** 2)

    def autocorrelation(self, lag: int = 1) -> float:
        """Autocorrelation of the node values at ``lag``, under the stationary distribution."""
        pi = self.stationary
        x = self.nodes - pi @ self.nodes
        ahead = np.linalg.matrix_power(self.transition, lag) @ x
        return float((pi @ (x * ahead)) / (pi @ x ** 2))

    def simulate(self, periods: int, *, start: int | None = None, seed: int | None = None,
                 indices: bool = False) -> np.ndarray:
        """A path of ``periods`` values, from node ``start`` or a draw from the stationary distribution.

        ``indices=True`` returns the node indices instead of their values.
        """
        rng = np.random.default_rng(seed)
        cdf = np.cumsum(self.transition, axis=1)
        cdf[:, -1] = 1.0
        path = np.empty(periods, dtype=int)
        if periods == 0:
            return path if indices else self.nodes[path]
        if start is None:
            pi = np.cumsum(self.stationary)
            pi[-1] = 1.0
            start = int(np.searchsorted(pi, rng.random(), side="right"))
        elif not 0 <= start < self.n:
            raise ValueError(f"start must be a node index, from 0 to {self.n - 1}")
        path[0] = start
        u = rng.random(periods - 1)
        for t in range(1, periods):
            path[t] = np.searchsorted(cdf[path[t - 1]], u[t - 1], side="right")
        return path if indices else self.nodes[path]


def tauchen(rho: float, sigma: float, n: int, *, mean: float = 0.0,
            n_std: float = 3.0) -> MarkovChain:
    """Tauchen's (1986) chain for an AR(1) with persistence ``rho`` and innovation s.d. ``sigma``.

    The ``n`` nodes are evenly spaced from ``mean - n_std s`` to ``mean + n_std s``, where
    ``s = sigma / sqrt(1 - rho^2)`` is the unconditional standard deviation, and each transition
    probability is the normal probability of the interval between the midpoints around the node;
    the two end nodes take the tails.
    """
    _check(rho, sigma, n)
    if n_std <= 0:
        raise ValueError("n_std must be positive")
    s = np.sqrt(sigma ** 2 / (1.0 - rho ** 2))
    x = np.linspace(-n_std * s, n_std * s, n)
    half = 0.5 * (x[-1] - x[0]) / (n - 1)
    z = x[None, :] - rho * x[:, None]                 # next node minus the conditional mean
    a, b = (z - half) / sigma, (z + half) / sigma     # the interval around each next node
    # Above the conditional mean the probability is a difference of survival functions, so that
    # an interval in the upper tail keeps the relative accuracy of one in the lower tail.
    P = np.where(a > 0, ndtr(-a) - ndtr(-b), ndtr(b) - ndtr(a))
    P[:, 0] = ndtr(b[:, 0])
    P[:, -1] = ndtr(-a[:, -1])
    return MarkovChain(x + mean, P)


def rouwenhorst(rho: float, sigma: float, n: int, *, mean: float = 0.0) -> MarkovChain:
    """Rouwenhorst's chain for an AR(1) with persistence ``rho`` and innovation s.d. ``sigma``.

    The ``n`` nodes are evenly spaced over ``sqrt(n - 1)`` unconditional standard deviations on
    each side of ``mean``, and the transition matrix is built by the recursion of Kopecky and Suen
    (2010) with ``p = q = (1 + rho) / 2``. The stationary distribution is binomial with ``n - 1``
    trials of probability one half, and the chain's mean, variance and first autocorrelation are
    those of the AR(1).
    """
    _check(rho, sigma, n)
    p = (1.0 + rho) / 2.0
    s = np.sqrt(sigma ** 2 / (1.0 - rho ** 2))
    x = np.linspace(-s * np.sqrt(n - 1), s * np.sqrt(n - 1), n)
    P = np.array([[p, 1.0 - p], [1.0 - p, p]])
    for m in range(3, n + 1):
        Q = np.zeros((m, m))
        Q[:-1, :-1] += p * P
        Q[:-1, 1:] += (1.0 - p) * P
        Q[1:, :-1] += (1.0 - p) * P
        Q[1:, 1:] += p * P
        Q[1:-1, :] /= 2.0
        P = Q
    return MarkovChain(x + mean, P)


def _check(rho: float, sigma: float, n: int) -> None:
    if not -1.0 < rho < 1.0:
        raise ValueError("rho must be between -1 and 1, exclusive: the process must be stationary")
    if sigma <= 0:
        raise ValueError("sigma, the innovation's standard deviation, must be positive")
    if int(n) != n or n < 2:
        raise ValueError("n, the number of nodes, must be an integer of at least 2")


def _gth(P: ArrayLike) -> np.ndarray:
    """Stationary distribution by Grassmann, Taksar and Heyman's (1985) elimination."""
    A = np.array(P, dtype=float)
    n = A.shape[0]
    for k in range(n - 1):
        scale = A[k, k + 1:].sum()
        if scale <= 0.0:
            raise ValueError("the chain is reducible: not every node can reach every other. A "
                             "Tauchen chain with rho close to one and few nodes is, because its "
                             "probabilities of leaving the end nodes underflow to zero; "
                             "rouwenhorst does not have this problem")
        A[k + 1:, k] /= scale
        A[k + 1:, k + 1:] += np.outer(A[k + 1:, k], A[k, k + 1:])
    x = np.empty(n)
    x[-1] = 1.0
    for k in range(n - 2, -1, -1):
        x[k] = x[k + 1:] @ A[k + 1:, k]
    return x / x.sum()
