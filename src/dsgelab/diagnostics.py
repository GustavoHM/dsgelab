"""Checks that a solution is a solution -- and evidence to show when it is not.

Every routine here answers a question a referee is entitled to ask. None of them
is a formality: the identity check catches income leaks that no equation of the
model imposes, the slackness check catches a smoothed corner that has stopped
being slack, and the comparison catches a port that drifted from its original.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike

from .model import Model, Params, Point
from .solution import FirstOrderSolution

if TYPE_CHECKING:
    from .steady import SteadyState

__all__ = ["check_identity", "blanchard_kahn_report", "check_slack", "compare_irfs",
           "steady_state_invariance", "diagnose"]


def check_identity(model: Model, fn: Callable,
                   points: ArrayLike | Sequence[tuple[ArrayLike, ArrayLike]],
                   params: Params, ctx: Any = None, tol: float = 1e-8) -> dict:
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
                 periods: int = 40, size: float = 1.0,
                 variables: Sequence[str] | None = None) -> dict:
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


def _jacobian_blocks(model: Model, x, params, ctx=None, h: float = 1e-7):
    """``(J_next, J_now)`` of the residuals at ``(x, x, 0)``, exact when the model has them."""
    eps = np.zeros(model.n_shocks)
    n = model.n_vars
    J = model.jacobian(x, x, eps, params, ctx) if model.jacobian_fn is not None else None
    if J is not None:
        return np.asarray(J[:, :n], float), np.asarray(J[:, n:2 * n], float)
    base = model.residuals(x, x, eps, params, ctx)
    Jn, Jc = np.zeros((len(base), n)), np.zeros((len(base), n))
    for j in range(n):
        step = h * max(1.0, abs(x[j]))
        a = x.copy(); a[j] += step
        Jn[:, j] = (model.residuals(a, x, eps, params, ctx) - base) / step
        Jc[:, j] = (model.residuals(x, a, eps, params, ctx) - base) / step
    return Jn, Jc


def _nullspace(M: np.ndarray, tol: float) -> np.ndarray:
    u, s, vt = np.linalg.svd(M)
    rank = int(np.sum(s > tol * max(1.0, s[0] if s.size else 0.0)))
    return vt[rank:].T


def diagnose(model: Model, params: Params, point: SteadyState | Point, *, ctx: Any = None,
             tol: float = 1e-9) -> dict:
    """What Dynare's ``model_diagnostics`` checks, at ``point`` (a steady state or a guess).

    * variables that do not appear in any equation at the current date (Dynare's
      "aren't present at the current period"), and variables that appear nowhere;
    * a point with ``nan`` or ``inf``, or residuals that are not finite;
    * the STATIC Jacobian -- of ``f(y, y, 0)`` -- and its rank. A singular one has
      collinear relationships, reported both ways: the groups of VARIABLES the
      equations cannot tell apart (the right null space), and the groups of
      EQUATIONS that are linearly dependent (the left null space). One redundant
      equation with another missing, often by Walras' law, is the usual cause;
    * whether a root of the dynamic system lies at one, which makes the static
      Jacobian singular on its own (a price level, a random walk);
    * the DYNAMIC Jacobian -- next and current values together -- and its rank: an
      equation that is a combination of others at every date is redundant;
    * a dynamic Jacobian with ``nan`` or ``inf`` entries.

    ``point`` is a :class:`~dsgelab.SteadyState`, a mapping, or a vector. The
    report has ``ok`` False when anything above was found.
    """
    from .steady import SteadyState

    if isinstance(point, SteadyState):
        x = np.asarray(point.x, float)
    elif isinstance(point, Mapping):
        x = model.vector(point)
    else:
        x = np.asarray(point, float)
    names = model.names
    labels = list(getattr(model, "symbolic", {}).get("equations", [])) if \
        getattr(model, "symbolic", None) else [f"equation {i + 1}" for i in range(model.n_vars)]
    report = {"ok": True, "messages": []}

    def note(msg):
        report["ok"] = False
        report["messages"].append(msg)

    if not np.all(np.isfinite(x)):
        bad = [n for n, v in zip(names, x) if not np.isfinite(v)]
        note(f"the point contains nan or inf in {bad}")
        report["nonfinite_point"] = bad
        return report
    res = model.residuals(x, x, np.zeros(model.n_shocks), params, ctx)
    report["residual"] = float(np.max(np.abs(res))) if np.all(np.isfinite(res)) else np.inf
    if not np.all(np.isfinite(res)):
        note("the residuals at the point are not finite")
    Jn, Jc = _jacobian_blocks(model, x, params, ctx)
    if not (np.all(np.isfinite(Jn)) and np.all(np.isfinite(Jc))):
        rows, cols = np.where(~np.isfinite(Jn) | ~np.isfinite(Jc))
        note("the dynamic Jacobian contains nan or inf")
        report["nonfinite_jacobian"] = sorted({(labels[r] if r < len(labels) else r, names[c])
                                               for r, c in zip(rows, cols)})
        return report
    scale = max(1.0, float(np.max(np.abs(np.hstack([Jn, Jc])))))
    absent_now = [names[j] for j in range(len(names)) if np.max(np.abs(Jc[:, j])) <= tol * scale]
    nowhere = [v for v in absent_now if np.max(np.abs(Jn[:, names.index(v)])) <= tol * scale]
    # a state of this package appears at t through its own transition; what Dynare calls
    # "not present at the current period" is a variable whose CURRENT value enters nothing
    report["absent_current"] = absent_now
    report["absent"] = nowhere
    if nowhere:
        note(f"variables that appear in no equation: {nowhere}")
    if absent_now:
        report["messages"].append(f"variables not present at the current date: {absent_now}")
    # dynamic Jacobian: an equation that is a combination of others at every date
    Jd = np.hstack([Jn, Jc])
    sd = np.linalg.svd(Jd, compute_uv=False)
    drank = int(np.sum(sd > tol * max(1.0, sd[0])))
    report["redundant_equations"] = Jd.shape[0] - drank
    if drank < Jd.shape[0]:
        note(f"the dynamic Jacobian is singular: {Jd.shape[0] - drank} redundant equation(s)")
        Nd = _nullspace(Jd.T, tol)
        report["redundant_groups"] = [[(int(i) + 1, labels[i] if i < len(labels) else "")
                                       for i in np.where(np.abs(v) > 1e-8)[0]] for v in Nd.T]
    Js = Jn + Jc
    s = np.linalg.svd(Js, compute_uv=False)
    rank = int(np.sum(s > tol * max(1.0, s[0])))
    report["static_rank"] = rank
    report["relations"] = Js.shape[0] - rank
    if rank < Js.shape[0]:
        note(f"the static Jacobian is singular: {Js.shape[0] - rank} collinear "
             f"relationship(s) between variables and equations")
        Nv = _nullspace(Js, tol)
        Ne = _nullspace(Js.T, tol)
        report["collinear_variables"] = [[names[i] for i in np.where(np.abs(v) > 1e-8)[0]]
                                         for v in Nv.T]
        report["collinear_equations"] = [[(int(i) + 1, labels[i] if i < len(labels) else "")
                                           for i in np.where(np.abs(v) > 1e-8)[0]]
                                          for v in Ne.T]
        # a unit root makes I - (dynamic transition) singular by itself
        try:
            from scipy.linalg import eigvals
            lam = eigvals(Jc, -Jn)
            near = np.abs(np.abs(lam[np.isfinite(lam)]) - 1) < 1e-6
            report["unit_root"] = bool(np.any(near))
            if report["unit_root"]:
                report["messages"].append(
                    "a root of the dynamic system is at one: the singularity may be (partly) a "
                    "unit root, which is expected if the model has one")
        except Exception:
            report["unit_root"] = None
    return report
