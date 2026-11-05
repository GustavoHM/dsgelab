"""Smolyak sparse grids with the basis of Judd, Maliar, Maliar and Valero (2014).

Along one axis, level ``i`` has ``m(i)`` nodes, the extrema of the Chebyshev polynomial of degree
``m(i) - 1``, with ``m(1) = 1`` and ``m(i) = 2**(i-1) + 1``. The sets are nested, and ``A_i``, the
nodes level ``i`` adds to level ``i - 1``, are disjoint. The polynomials are split the same way:
``P_1`` is ``T_0``, and ``P_i`` holds ``T_{m(i-1)}`` to ``T_{m(i)-1}``, as many as ``A_i`` has
nodes. A multi-index ``(i_1, ..., i_d)`` contributes the product ``A_{i_1} x ... x A_{i_d}`` to
the grid and ``P_{i_1} x ... x P_{i_d}`` to the basis, so the basis has one function per node.
The isotropic grid of level ``mu`` takes the multi-indices with ``i_1 + ... + i_d <= d + mu``.
The anisotropic grid has a level ``mu_j`` for each axis, takes the multi-indices with
``i_j <= mu_j + 1`` and ``i_1 + ... + i_d <= d + max(mu)``, and is JMMV's (section 4).

The interpolant is the polynomial in that basis that takes the given values at the nodes. Its
coefficients solve a square linear system, factored once per grid. On nested Chebyshev extrema the
space spanned by the basis is that of Smolyak's combination of tensor interpolants, so the
interpolant is the one Tasmanian builds on its Clenshaw–Curtis global grid of type ``level``.
"""

from __future__ import annotations

from functools import cached_property
from itertools import product
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import lu_factor, lu_solve

__all__ = ["SmolyakGrid"]


class SmolyakGrid:
    """A Smolyak grid on a box: one ``(low, high)`` interval for each named axis.

    ``level`` is an integer for an isotropic grid, or one integer per axis, as a sequence in the
    order of ``axes`` or as a mapping from each name, for an anisotropic one. ``points`` lists the
    nodes in levels, one row per node and one column per axis. They are ordered by multi-index,
    in lexicographic order, and within one multi-index with the last axis varying fastest.
    """

    def __init__(self, axes: Mapping[str, tuple[float, float]],
                 level: int | Sequence[int] | Mapping[str, int]):
        self.names = list(axes)
        if not self.names:
            raise ValueError("a grid needs at least one axis")
        bounds = []
        for name, spec in axes.items():
            lo, hi = (float(v) for v in spec)
            if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
                raise ValueError(f"axis {name!r}: the upper end must exceed the lower one")
            bounds.append((lo, hi))
        self.bounds = np.array(bounds)
        d = len(self.names)
        if isinstance(level, Mapping):
            missing = [n for n in self.names if n not in level]
            extra = [n for n in level if n not in self.names]
            if missing or extra:
                raise ValueError(f"level needs one entry per axis: missing {missing}, "
                                 f"unknown {extra}")
            levels = [level[n] for n in self.names]
        elif np.ndim(level) == 0:
            levels = [level] * d
        else:
            levels = list(level)
            if len(levels) != d:
                raise ValueError(f"level has {len(levels)} entries for {d} axes")
        if not all(float(v).is_integer() and v >= 0 for v in levels):
            raise ValueError("each level must be a nonnegative integer")
        self.levels = np.array([int(v) for v in levels], dtype=int)
        self.level = int(self.levels.max())
        self.indices = _multi_indices(self.levels)
        unit, degrees = [], []
        for idx in self.indices:
            unit.extend(product(*[_new_nodes(i) for i in idx]))
            degrees.extend(product(*[_new_degrees(i) for i in idx]))
        self.unit_points = np.array(unit, dtype=float).reshape(-1, d)
        self.degrees = np.array(degrees, dtype=int).reshape(-1, d)
        self.points = self._from_unit(self.unit_points)
        self._lu = lu_factor(self.basis(self.points))

    @property
    def n(self) -> int:
        """Number of grid points, which is also the number of basis functions."""
        return self.points.shape[0]

    @property
    def d(self) -> int:
        """Number of axes."""
        return len(self.names)

    @cached_property
    def axes(self) -> list[np.ndarray]:
        """The coordinates the nodes take along each axis, increasing."""
        return [np.unique(self.points[:, j]) for j in range(self.d)]

    @cached_property
    def midpoints(self) -> np.ndarray:
        """The points the next level adds, where the interpolant is least constrained.

        They are the nodes of the grid with every level raised by one that are not nodes of this
        one, and play the part of a :class:`CartesianGrid`'s cell centres in
        :meth:`TimeIterationSolution.euler_errors`.
        """
        finer = SmolyakGrid(dict(zip(self.names, map(tuple, self.bounds))), self.levels + 1)
        known = {tuple(np.round(p, 12)) for p in self.unit_points}
        keep = [i for i, p in enumerate(finer.unit_points) if tuple(np.round(p, 12)) not in known]
        return finer.points[keep]

    def basis(self, points: ArrayLike) -> np.ndarray:
        """The basis functions at ``points``, one row per point and one column per function."""
        X = self._to_unit(points)
        top = int(self.degrees.max())
        out = np.ones((X.shape[0], self.n))
        for j in range(self.d):
            T = _chebyshev(X[:, j], top)
            out *= T[:, self.degrees[:, j]]
        return out

    def coefficients(self, values: ArrayLike) -> np.ndarray:
        """The interpolant's coefficients, one row per basis function, from one row per node."""
        values = np.asarray(values, dtype=float)
        if values.shape[0] != self.n:
            raise ValueError(f"values need one row per grid point, {self.n}, not {values.shape[0]}")
        return lu_solve(self._lu, values)

    def interpolate(self, values: ArrayLike, points: ArrayLike) -> np.ndarray:
        """Interpolate ``values``, one row per grid point, at ``points``, one row per point."""
        points = np.asarray(points, dtype=float)
        out = self.basis(np.atleast_2d(points)) @ self.coefficients(values)
        return out[0] if points.ndim == 1 else out

    def interpolation_weights(self, points: ArrayLike) -> np.ndarray:
        """The weights that take the values at the nodes to the interpolant at ``points``.

        Row ``p`` holds one weight per node, so that the interpolant at point ``p`` is the weights
        times the values. These are the numbers Tasmanian's ``getInterpolationWeights`` returns.
        """
        points = np.asarray(points, dtype=float)
        W = lu_solve(self._lu, self.basis(np.atleast_2d(points)).T, trans=1).T
        return W[0] if points.ndim == 1 else W

    # time_iteration reaches the interpolant through these two, as it does a CartesianGrid's;
    # the Smolyak interpolant has one kind
    def _coefficients(self, values: np.ndarray, kind: str = "smolyak") -> np.ndarray:
        return self.coefficients(values)

    def _evaluate(self, G: np.ndarray, points: np.ndarray, kind: str = "smolyak") -> np.ndarray:
        single = points.ndim == 1
        if single:
            points = points[None, :]
        if points.shape[1] != self.d:
            raise ValueError(f"points need {self.d} columns, one per axis ({self.names})")
        out = np.tensordot(self.basis(points), G, axes=1)
        return out[0] if single else out

    def _to_unit(self, points: ArrayLike) -> np.ndarray:
        X = np.atleast_2d(np.asarray(points, dtype=float))
        if X.shape[1] != self.d:
            raise ValueError(f"points need {self.d} columns, one per axis ({self.names})")
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        return 2.0 * (X - lo) / (hi - lo) - 1.0

    def _from_unit(self, X: np.ndarray) -> np.ndarray:
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        return lo + (X + 1.0) * (hi - lo) / 2.0

    def __repr__(self) -> str:
        sides = ", ".join(f"{n}: level {mu} from {b[0]:g} to {b[1]:g}"
                          for n, mu, b in zip(self.names, self.levels, self.bounds))
        return f"SmolyakGrid({self.n} points; {sides})"


def _size(i: int) -> int:
    """``m(i)``, the number of nodes at level ``i`` along one axis."""
    return 1 if i == 1 else 2 ** (i - 1) + 1


def _new_nodes(i: int) -> np.ndarray:
    """``A_i``: the Chebyshev extrema of level ``i`` that level ``i - 1`` does not have."""
    if i == 1:
        return np.zeros(1)
    m = _size(i)
    # level 2 adds the ends to the centre, and each later level the odd j between earlier nodes
    j = np.array([0, m - 1]) if i == 2 else np.arange(1, m, 2)
    # -cos(pi j / (m - 1)), with the centre set to exactly zero
    x = -np.cos(np.pi * j / (m - 1))
    x[np.abs(x) < 1e-15] = 0.0
    return x


def _new_degrees(i: int) -> range:
    """``P_i``: the degrees of the Chebyshev polynomials that level ``i`` adds."""
    return range(1) if i == 1 else range(_size(i - 1), _size(i))


def _multi_indices(levels: np.ndarray) -> list[tuple[int, ...]]:
    """The multi-indices with ``i_j <= mu_j + 1`` and ``sum(i - 1) <= max(mu)``, in order."""
    top = int(levels.max())
    out: list[tuple[int, ...]] = []

    def extend(prefix: list[int], used: int) -> None:
        j = len(prefix)
        if j == levels.size:
            out.append(tuple(prefix))
            return
        for i in range(1, min(int(levels[j]), top - used) + 2):
            extend(prefix + [i], used + i - 1)

    extend([], 0)
    return out


def _chebyshev(x: np.ndarray, order: int) -> np.ndarray:
    """``T_0(x)`` to ``T_order(x)``, one column each."""
    T = np.empty((x.size, order + 1))
    T[:, 0] = 1.0
    if order >= 1:
        T[:, 1] = x
    for n in range(2, order + 1):
        T[:, n] = 2.0 * x * T[:, n - 1] - T[:, n - 2]
    return T
