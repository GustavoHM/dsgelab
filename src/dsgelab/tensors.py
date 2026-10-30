"""Sparse tensors used by higher-order perturbation."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations_with_replacement, permutations

import numpy as np
from numpy.typing import ArrayLike

__all__ = ["SparseSecondDerivatives", "SparseThirdDerivatives"]


@dataclass
class SparseThirdDerivatives:
    """Third derivatives in coordinate form, with one row per nonzero entry."""

    equations: np.ndarray
    indices: np.ndarray
    values: np.ndarray
    shape: tuple[int, int, int, int]
    symmetric: bool = False

    def __post_init__(self) -> None:
        self.equations = np.asarray(self.equations, dtype=np.int64).reshape(-1)
        self.indices = np.asarray(self.indices, dtype=np.int64).reshape(-1, 3)
        self.values = np.asarray(self.values, dtype=float).reshape(-1)
        if len(self.shape) != 4 or any(int(d) < 0 for d in self.shape):
            raise ValueError(f"invalid third-derivative shape {self.shape}")
        if not (len(self.equations) == len(self.indices) == len(self.values)):
            raise ValueError("equations, indices and values must have the same length")
        if len(self.values):
            if np.any(self.equations < 0) or np.any(self.equations >= self.shape[0]):
                raise ValueError("equation index outside the tensor shape")
            if np.any(self.indices < 0) or np.any(self.indices >= self.shape[1]):
                raise ValueError("derivative index outside the tensor shape")
        if self.shape[1] != self.shape[2] or self.shape[1] != self.shape[3]:
            raise ValueError("the last three dimensions must have equal length")

    @property
    def nnz(self) -> int:
        return len(self.values)

    @classmethod
    def from_dense(cls, dense: ArrayLike) -> SparseThirdDerivatives:
        array = np.asarray(dense, dtype=float)
        if array.ndim != 4:
            raise ValueError(f"expected a four-dimensional tensor, got {array.shape}")
        equations, i, j, k = np.nonzero(array)
        return cls(equations, np.column_stack([i, j, k]), array[equations, i, j, k],
                   (int(array.shape[0]), int(array.shape[1]),
                    int(array.shape[2]), int(array.shape[3])))

    @classmethod
    def from_canonical(cls, values: ArrayLike, canonical: ArrayLike,
                       shape: tuple[int, int, int, int]) -> SparseThirdDerivatives:
        values = np.asarray(values, dtype=float)
        canonical = np.asarray(canonical, dtype=np.int64).reshape(-1, 3)
        if values.shape != (shape[0], len(canonical)):
            raise ValueError("values and canonical indices have incompatible shapes")
        equations, positions = np.nonzero(values)
        return cls(equations, canonical[positions], values[equations, positions], shape, True)

    def to_canonical(self) -> np.ndarray:
        canonical = list(combinations_with_replacement(range(self.shape[1]), 3))
        positions = {index: column for column, index in enumerate(canonical)}
        out = np.zeros((self.shape[0], len(canonical)))
        for e, index, value in zip(self.equations, self.indices, self.values):
            out[e, positions[tuple(index)]] += value
        return out

    def to_dense(self) -> np.ndarray:
        out = np.zeros(self.shape)
        for e, (a, b, c), value in zip(self.equations, self.indices, self.values):
            permutations_ = set(permutations((a, b, c))) if self.symmetric else [(a, b, c)]
            for i, j, k in permutations_:
                out[e, i, j, k] += value
        return out

    def scaled(self, scale: ArrayLike, row_scale: ArrayLike | None = None) -> SparseThirdDerivatives:
        scale = np.asarray(scale, dtype=float)
        if scale.shape != (self.shape[1],):
            raise ValueError(f"scale has shape {scale.shape}, expected {(self.shape[1],)}")
        factors = np.prod(scale[self.indices], axis=1)
        if row_scale is not None:
            rows = np.asarray(row_scale, dtype=float)
            if rows.shape != (self.shape[0],):
                raise ValueError(f"row_scale has shape {rows.shape}, expected {(self.shape[0],)}")
            factors = factors / rows[self.equations]
        return SparseThirdDerivatives(self.equations.copy(), self.indices.copy(),
                                      self.values * factors, self.shape, self.symmetric)

    def contract(self, u: ArrayLike, v: ArrayLike, w: ArrayLike, rows=None,
                 weights=None) -> np.ndarray:
        """Evaluate ``T[e,a,b,c] u[a,i] v[b,j] w[c,k]`` without densifying ``T``."""
        u, v, w = (np.asarray(x, dtype=float) for x in (u, v, w))
        if any(x.ndim != 2 or x.shape[0] != self.shape[1] for x in (u, v, w)):
            raise ValueError("each contraction operand must have shape (m, n)")
        if rows is None:
            rows = np.arange(self.shape[0])
        rows = np.asarray(rows, dtype=np.int64)
        if weights is not None:
            weights = np.asarray(weights, dtype=float)
            if weights.shape != (len(rows), self.shape[0]):
                raise ValueError("weights must have shape (len(rows), n_equations)")
        out = np.zeros((len(rows), u.shape[1], v.shape[1], w.shape[1]))
        positions = {int(e): i for i, e in enumerate(rows)}
        for e, (a, b, c), value in zip(self.equations, self.indices, self.values):
            if e not in positions:
                continue
            permutations_ = set(permutations((a, b, c))) if self.symmetric else [(a, b, c)]
            for i, j, k in permutations_:
                outer = value * np.einsum("i,j,k->ijk", u[i], v[j], w[k])
                if weights is None:
                    out[positions[e]] += outer
                else:
                    out += weights[:, e, None, None, None] * outer
        return out

    def contract_canonical(self, u: ArrayLike, v: ArrayLike, w: ArrayLike,
                           weights=None) -> np.ndarray:
        """Contract onto independent symmetric triples, with optional row combinations."""
        u, v, w = (np.asarray(x, dtype=float) for x in (u, v, w))
        nk = self.shape[1]
        if any(x.shape != (nk, nk) for x in (u, v, w)):
            raise ValueError("canonical contraction operands must be square (m, m) matrices")
        canonical = list(combinations_with_replacement(range(nk), 3))
        I, J, K = (np.asarray(index, dtype=np.int64) for index in zip(*canonical))
        rows = self.shape[0] if weights is None else len(weights)
        out = np.zeros((rows, len(canonical)))
        if weights is not None:
            weights = np.asarray(weights, dtype=float)
            if weights.shape != (rows, self.shape[0]):
                raise ValueError("weights must have shape (n_rows, n_equations)")
        for e, (a, b, c), value in zip(self.equations, self.indices, self.values):
            permutations_ = set(permutations((a, b, c))) if self.symmetric else [(a, b, c)]
            for p, q, r in permutations_:
                contribution = value * u[p, I] * v[q, J] * w[r, K]
                if weights is None:
                    out[e] += contribution
                else:
                    out += weights[:, e, None] * contribution
        return out


@dataclass
class SparseSecondDerivatives:
    """Hessians in coordinate form, with one row per nonzero entry."""

    equations: np.ndarray
    indices: np.ndarray
    values: np.ndarray
    shape: tuple[int, int, int]

    def __post_init__(self) -> None:
        self.equations = np.asarray(self.equations, dtype=np.int64).reshape(-1)
        self.indices = np.asarray(self.indices, dtype=np.int64).reshape(-1, 2)
        self.values = np.asarray(self.values, dtype=float).reshape(-1)
        if len(self.shape) != 3 or any(int(d) < 0 for d in self.shape):
            raise ValueError(f"invalid second-derivative shape {self.shape}")
        if not (len(self.equations) == len(self.indices) == len(self.values)):
            raise ValueError("equations, indices and values must have the same length")
        if len(self.values):
            if np.any(self.equations < 0) or np.any(self.equations >= self.shape[0]):
                raise ValueError("equation index outside the tensor shape")
            if np.any(self.indices < 0) or np.any(self.indices >= self.shape[1]):
                raise ValueError("derivative index outside the tensor shape")
        if self.shape[1] != self.shape[2]:
            raise ValueError("the last two dimensions must have equal length")

    @property
    def nnz(self) -> int:
        return len(self.values)

    @classmethod
    def from_dense(cls, dense: ArrayLike) -> SparseSecondDerivatives:
        array = np.asarray(dense, dtype=float)
        if array.ndim != 3:
            raise ValueError(f"expected a three-dimensional tensor, got {array.shape}")
        equations, i, j = np.nonzero(array)
        return cls(equations, np.column_stack([i, j]), array[equations, i, j],
                   (int(array.shape[0]), int(array.shape[1]), int(array.shape[2])))

    def to_dense(self) -> np.ndarray:
        out = np.zeros(self.shape)
        np.add.at(out, (self.equations, self.indices[:, 0], self.indices[:, 1]), self.values)
        return out

    def scaled(self, scale: ArrayLike, row_scale: ArrayLike | None = None) -> SparseSecondDerivatives:
        scale = np.asarray(scale, dtype=float)
        if scale.shape != (self.shape[1],):
            raise ValueError(f"scale has shape {scale.shape}, expected {(self.shape[1],)}")
        factors = np.prod(scale[self.indices], axis=1)
        if row_scale is not None:
            rows = np.asarray(row_scale, dtype=float)
            if rows.shape != (self.shape[0],):
                raise ValueError(f"row_scale has shape {rows.shape}, expected {(self.shape[0],)}")
            factors = factors / rows[self.equations]
        return SparseSecondDerivatives(self.equations.copy(), self.indices.copy(),
                                       self.values * factors, self.shape)

    def contract(self, u: ArrayLike, v: ArrayLike) -> np.ndarray:
        """Evaluate ``H[e,a,b] u[a,i] v[b,j]`` without densifying ``H``."""
        u, v = np.asarray(u, dtype=float), np.asarray(v, dtype=float)
        if any(x.ndim < 1 or x.shape[0] != self.shape[1] for x in (u, v)):
            raise ValueError("each contraction operand must start with the derivative axis")
        out = np.zeros((self.shape[0], *u.shape[1:], *v.shape[1:]))
        for e, (a, b), value in zip(self.equations, self.indices, self.values):
            out[e] += value * np.tensordot(u[a], v[b], axes=0)
        return out
