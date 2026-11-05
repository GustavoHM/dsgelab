"""run_tasmanian.py: Tasmanian's Smolyak grids and interpolation weights, for compare_smolyak.py.

Runs in the environment that setup.sh creates (Python 3.12, Tasmanian 8.2) and writes
validation/out_smolyak/tasmanian.json. Each case is a global grid of type "level" with the
Clenshaw-Curtis rule, whose nodes along one axis are the Chebyshev extrema, 1 at level 0 and
2**l + 1 at level l > 0. Tasmanian's level l is JMMV's i - 1, so its depth is JMMV's mu. The
anisotropic cases cap the level along each axis with liLevelLimits, which gives JMMV's
anisotropic grid: the multi-indices with i_j <= mu_j + 1 and i_1 + ... + i_d <= d + max(mu).
For each case the file holds the box, the grid's points and, at 40 points drawn uniformly in the
box, the weights getInterpolationWeights returns, one per grid point in Tasmanian's order.

Usage:  validation/tasmanian/.venv/bin/python validation/tasmanian/run_tasmanian.py
"""
from __future__ import annotations

import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import Tasmanian

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "out_smolyak"

# isotropic grids: (dimension, levels); anisotropic grids: one level per axis
ISOTROPIC = [(1, range(0, 7)), (2, range(0, 6)), (3, range(0, 5)), (4, range(0, 5)),
             (5, range(0, 5)), (6, range(0, 5))]
ANISOTROPIC = [[3, 1], [1, 4], [3, 1, 2], [0, 2, 1, 3], [3, 2, 1, 1, 0], [3, 3, 2, 2, 1, 1],
               [4, 1, 1, 1, 1, 1]]


def box(d: int, rng) -> np.ndarray:
    lo = rng.uniform(-2.0, 1.0, d)
    return np.column_stack([lo, lo + rng.uniform(0.5, 3.0, d)])


def case(levels: list[int], bounds: np.ndarray | None, rng) -> dict:
    d = len(levels)
    depth = max(levels)
    limits = [] if len(set(levels)) == 1 else list(levels)
    grid = Tasmanian.makeGlobalGrid(d, 1, depth, "level", "clenshaw-curtis",
                                    liLevelLimits=limits)
    if bounds is not None:
        grid.setDomainTransform(bounds)
    lo, hi = (bounds[:, 0], bounds[:, 1]) if bounds is not None else (-np.ones(d), np.ones(d))
    at = rng.uniform(lo, hi, (40, d))
    weights = np.array([grid.getInterpolationWeights(x) for x in at])
    return {"levels": list(levels), "bounds": np.column_stack([lo, hi]).tolist(),
            "points": grid.getPoints().tolist(), "at": at.tolist(),
            "weights": weights.tolist()}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(20261103)
    cases = []
    for d, levels in ISOTROPIC:
        for mu in levels:
            # the canonical box for the odd levels, a box of its own for the even ones
            cases.append(case([mu] * d, box(d, rng) if mu % 2 == 0 else None, rng))
    for levels in ANISOTROPIC:
        cases.append(case(levels, box(len(levels), rng), rng))
    meta = {"tasmanian": version("tasmanian"), "rule": "clenshaw-curtis", "type": "level"}
    (OUT / "tasmanian.json").write_text(json.dumps({"meta": meta, "cases": cases}))
    print(f"Tasmanian {meta['tasmanian']}: {len(cases)} grids written to {OUT / 'tasmanian.json'}")


if __name__ == "__main__":
    main()
