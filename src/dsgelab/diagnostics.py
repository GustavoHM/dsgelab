"""Checks that a solution is a solution -- and evidence to show when it is not.

Every routine here answers a question a referee is entitled to ask. None of them
is a formality: the identity check catches income leaks that no equation of the
model imposes, the slackness check catches a smoothed corner that has stopped
being slack, and the comparison catches a port that drifted from its original.
"""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

import numpy as np

from .model import Model
from .solution import FirstOrderSolution

__all__ = ["check_identity", "blanchard_kahn_report", "check_slack", "compare_irfs",
           "steady_state_invariance"]


def check_identity(model: Model, fn: Callable, points, params, ctx=None,
                   tol: float = 1e-8) -> dict:
    """Evaluate an identity the model does NOT impose, as an independent check.

    The resource constraint of a closed model is the canonical case: it follows
    from the budget constraints, the firm's first-order conditions and the
    definitions of profit, so it is not one of the equations. That is exactly
    what makes it evidence -- a Walras residual is a leak somewhere in the
    accounting, and imposing it as an equation would hide the leak instead of
    revealing it.

    ``fn(values_next, values_now, params, ctx) -> float`` returns the gap.
    ``points`` is a sequence of ``(x_next, x_now)`` pairs, or a ``(T, n)`` path,
    in which case consecutive rows are used.
    """
    pts = _as_pairs(points)
    gaps = []
    for x1, x0 in pts:
        gaps.append(float(fn(model.dict(x1), model.dict(x0), params, ctx)))
    gaps = np.array(gaps)
    worst = float(np.max(np.abs(gaps))) if gaps.size else 0.0
    return dict(gaps=gaps, max_abs=worst, ok=bool(worst < tol), tolerance=tol)


def _as_pairs(points):
    arr = np.asarray(points, dtype=float)
    if arr.ndim == 2:
        return [(arr[t + 1], arr[t]) for t in range(arr.shape[0] - 1)]
    return list(points)


def blanchard_kahn_report(solution: FirstOrderSolution) -> dict:
    """Stable-root count, the roots nearest the unit circle and how much room there is.

    A model that satisfies Blanchard-Kahn with a root at 0.9999 satisfies it in
    the same sense that a bridge holds with one bolt: the classification will
    flip under a parameter change small enough to be invisible in a sweep. The
    margin is the number to report, not the pass.
    """
    eig = np.asarray(solution.eigenvalues, dtype=float)
    finite = eig[np.isfinite(eig)]
    stable = finite[finite < 1.0]
    unstable = finite[finite >= 1.0]
    largest_stable = float(stable.max()) if stable.size else float("nan")
    smallest_unstable = float(unstable.min()) if unstable.size else float("inf")
    return dict(n_stable=int(solution.klein.n_stable), n_states=int(solution.klein.n_k),
                satisfied=int(solution.klein.n_stable) == int(solution.klein.n_k),
                largest_stable_root=largest_stable,
                smallest_unstable_root=smallest_unstable,
                margin=float(smallest_unstable - largest_stable),
                eigenvalues=eig)


def check_slack(solution: FirstOrderSolution, constraints: Mapping[str, Callable],
                shocks: Sequence[str] | None = None, size: float = 1.0,
                periods: int = 60, tol: float = 0.0) -> dict:
    """Verify that smoothed inequality constraints stay slack along the impulse responses.

    A ``max(x, 0)`` or a floor is not differentiable and cannot be perturbed, so
    applied models replace it with its smooth interior branch. That substitution
    is legitimate only while the constraint is slack, and slack at the steady
    state is not slack along a large impulse response. Each entry of
    ``constraints`` maps a name to ``fn(values) -> float``, positive when slack.
    """
    names = list(solution.model.shocks) if shocks is None else list(shocks)
    out = {}
    worst_overall = np.inf
    for shock in names:
        paths = solution.irf(shock, size=size, periods=periods, levels=True)
        T = len(next(iter(paths.values())))
        for cname, fn in constraints.items():
            vals = np.array([fn({k: v[t] for k, v in paths.items()}) for t in range(T)])
            key = f"{shock}:{cname}"
            out[key] = dict(min=float(vals.min()), argmin=int(vals.argmin()),
                            ok=bool(vals.min() > tol))
            worst_overall = min(worst_overall, float(vals.min()))
    return dict(detail=out, worst=worst_overall, ok=bool(worst_overall > tol))


def compare_irfs(a: FirstOrderSolution, b: FirstOrderSolution, shock: str,
                 periods: int = 40, size: float = 1.0, variables=None) -> dict:
    """Largest discrepancy between two solutions' impulse responses, variable by variable.

    For checking a port -- a reimplementation in another language, a refactor of
    the equations -- against the original. Comparing steady states is not enough:
    two different models routinely share a steady state and differ entirely in
    their dynamics.
    """
    ia = a.irf(shock, size=size, periods=periods)
    ib = b.irf(shock, size=size, periods=periods)
    keys = list(variables) if variables is not None else [k for k in ia if k in ib]
    diffs = {}
    for k in keys:
        va, vb = np.asarray(ia[k]), np.asarray(ib[k])
        scale = max(float(np.max(np.abs(va))), 1e-12)
        diffs[k] = float(np.max(np.abs(va - vb)) / scale)
    worst = max(diffs.values()) if diffs else 0.0
    return dict(relative=diffs, worst=worst,
                worst_variable=max(diffs, key=diffs.get) if diffs else None)


def steady_state_invariance(build: Callable, values: Sequence, variables: Sequence[str],
                            tol: float = 1e-9) -> dict:
    """Check that named steady-state quantities do not move over a parameter grid.

    Many models carry an invariance that is a THEOREM, not an accident -- output
    independent of a redistributive policy, a great ratio pinned by the Euler
    equation. Asserting it numerically over a grid is the cheapest way to find
    out that a change broke the block that delivered it. ``build(v)`` returns a
    steady state for the grid value ``v``.
    """
    ref = build(values[0])
    rows = []
    worst = 0.0
    for v in values:
        ss = build(v)
        row = {"value": v}
        for name in variables:
            rel = abs(ss[name] / ref[name] - 1.0) if abs(ref[name]) > 1e-12 \
                else abs(ss[name] - ref[name])
            row[name] = rel
            worst = max(worst, rel)
        rows.append(row)
    return dict(rows=rows, worst=worst, ok=bool(worst < tol), tolerance=tol)
