"""Second-order rules and third-order innovation tensors against Dynare 7.2.

Run ``bash run_decision_rule.sh`` first. State-rule rows are checked in ghx,
ghu, ghxx, ghuu, ghxu, ghxuu and ghxxu. The tensors ghu, ghuu and ghuuu are
checked in every row. Composed jump blocks also have closed-form RBC tests.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat

import dsgelab as dsg
import models
from dsgelab.thirdorder import _dynare_cubic_rule, _dynare_risk_rule

HERE = Path(__file__).resolve().parent


def _load(name, order=2):
    suffix = "" if order == 2 else "_o3"
    path = HERE / "out_decision_rule" / f"{name}{suffix}.mat"
    if not path.exists():
        return None
    return loadmat(path, squeeze_me=True, struct_as_record=False)["ex"]


def _common_solution(model, params, ex):
    endo = [str(x).strip() for x in np.atleast_1d(ex.endo_names)]
    values = {v: float(np.atleast_1d(ex.ys)[endo.index(v)]) for v in model.names}
    x = np.array([values[v] for v in model.names])
    residual = float(np.max(np.abs(model.residuals(x, x, np.zeros(model.n_shocks), params))))
    steady = dsg.SteadyState(x=x, values=values, residual=residual)
    return dsg.solve(model, params, steady=steady, tol=2e-6), endo, residual


def _ordering(ex, endo, model):
    ov = np.atleast_1d(ex.order_var).astype(int) - 1
    if hasattr(ex, "state_var"):
        sv = np.atleast_1d(ex.state_var).astype(int) - 1
    else:
        lli = np.atleast_2d(ex.lead_lag_incidence).astype(int)
        sv = ov[lli[0, ov] > 0]
    dr_names = [endo[i] for i in ov]
    state_names = [endo[i] for i in sv]
    rows = [model.position(v) for v in dr_names]
    cols = [model.states.index(v) for v in state_names]
    state_rows = [i for i, v in enumerate(dr_names) if v in model.states]
    return dr_names, state_names, rows, cols, state_rows


def _dynare_state_timing(second):
    """Compose the extra exogenous transition in Dynare's endogenous-state rows."""
    first = second.first
    model = first.model
    P, Q = first.P, first.Q
    nk = model.n_states
    rule = second.decision_rule()
    out = {key: getattr(rule, key).copy() for key in ("g_k", "g_u", "g_kk", "g_uu", "g_ku")}
    direct = np.array([not np.allclose(Q[i], 0.0, atol=1e-12) for i in range(nk)])
    for i in range(nk):
        if direct[i]:
            continue
        R = np.eye(nk)
        S = np.zeros_like(Q)
        D = np.zeros(nk)
        R[direct], S[direct] = P[direct], Q[direct]
        D[direct] = P[i, direct]
        out["g_k"][i] = P[i] @ R
        out["g_u"][i] = P[i] @ S
        out["g_kk"][i] = (np.einsum("a,aij->ij", D, rule.g_kk[:nk])
                          + np.einsum("ab,ai,bj->ij", second.h_kk[i], R, R))
        out["g_uu"][i] = (np.einsum("a,aij->ij", D, rule.g_uu[:nk])
                          + np.einsum("ab,ai,bj->ij", second.h_kk[i], S, S))
        out["g_ku"][i] = (np.einsum("a,aij->ij", D, rule.g_ku[:nk])
                          + np.einsum("ab,ai,bj->ij", second.h_kk[i], R, S))
    return out


def _dynare_cubic_innovation(third):
    second = third.second
    first = second.first
    P, Q = first.P, first.Q
    nk, ne = first.model.n_states, first.model.n_shocks
    rule = third.decision_rule()
    out = rule.g_uuu.copy()
    direct = np.array([not np.allclose(Q[i], 0.0, atol=1e-12) for i in range(nk)])
    S = np.zeros_like(Q)
    S[direct] = Q[direct]
    for i in range(nk):
        if direct[i]:
            continue
        D = np.zeros(nk)
        D[direct] = P[i, direct]
        for a in range(ne):
            for b in range(ne):
                for c in range(ne):
                    out[i, a, b, c] = (
                        np.einsum("d,d->", D, rule.g_uuu[:nk, a, b, c])
                        + np.einsum("ab,a,b->", second.h_kk[i], S[:, a], rule.g_uu[:nk, b, c])
                        + np.einsum("ab,a,b->", second.h_kk[i], S[:, b], rule.g_uu[:nk, a, c])
                        + np.einsum("ab,a,b->", second.h_kk[i], S[:, c], rule.g_uu[:nk, a, b])
                        + np.einsum("abc,a,b,c->", third.h_kkk[i], S[:, a], S[:, b], S[:, c]))
    return out


def _dynare_cubic_state_innovation(third):
    second = third.second
    first = second.first
    P, Q = first.P, first.Q
    nk, ne = first.model.n_states, first.model.n_shocks
    rule = third.decision_rule()
    out = rule.g_kuu.copy()
    direct = np.array([not np.allclose(Q[i], 0.0, atol=1e-12) for i in range(nk)])
    R = np.eye(nk)
    S = np.zeros_like(Q)
    R[direct], S[direct] = P[direct], Q[direct]
    for row in range(nk):
        if direct[row]:
            continue
        D = np.zeros(nk)
        D[direct] = P[row, direct]
        for i in range(nk):
            for a in range(ne):
                for b in range(ne):
                    out[row, i, a, b] = (
                        np.einsum("d,d->", D, rule.g_kuu[:nk, i, a, b])
                        + np.einsum("ab,a,b->", second.h_kk[row], R[:, i], rule.g_uu[:nk, a, b])
                        + np.einsum("ab,a,b->", second.h_kk[row], S[:, a], rule.g_ku[:nk, i, b])
                        + np.einsum("ab,a,b->", second.h_kk[row], S[:, b], rule.g_ku[:nk, i, a])
                        + np.einsum("abc,a,b,c->", third.h_kkk[row], R[:, i], S[:, a], S[:, b]))
    return out


def _dynare_cubic_state_state_innovation(third):
    second = third.second
    first = second.first
    P, Q = first.P, first.Q
    nk, ne = first.model.n_states, first.model.n_shocks
    rule = third.decision_rule()
    out = rule.g_kku.copy()
    direct = np.array([not np.allclose(Q[i], 0.0, atol=1e-12) for i in range(nk)])
    R = np.eye(nk)
    S = np.zeros_like(Q)
    R[direct], S[direct] = P[direct], Q[direct]
    for row in range(nk):
        if direct[row]:
            continue
        D = np.zeros(nk)
        D[direct] = P[row, direct]
        for i in range(nk):
            for j in range(nk):
                for a in range(ne):
                    inner_ku_i = D * rule.g_ku[:nk, i, a]
                    inner_ku_j = D * rule.g_ku[:nk, j, a]
                    inner_kk = D * rule.g_kk[:nk, i, j]
                    out[row, i, j, a] = (
                        np.einsum("d,d->", D, rule.g_kku[:nk, i, j, a])
                        + np.einsum("ab,a,b->", second.h_kk[row], R[:, i], inner_ku_j)
                        + np.einsum("ab,a,b->", second.h_kk[row], R[:, j], inner_ku_i)
                        + np.einsum("ab,a,b->", second.h_kk[row], S[:, a], inner_kk)
                        + np.einsum("abc,a,b,c->", third.h_kkk[row], R[:, i], R[:, j], S[:, a]))
    return out


def _dynare_cubic_states(third):
    second = third.second
    first = second.first
    P, Q = first.P, first.Q
    nk = first.model.n_states
    rule = third.decision_rule()
    out = rule.g_kkk.copy()
    direct = np.array([not np.allclose(Q[i], 0.0, atol=1e-12) for i in range(nk)])
    R = np.eye(nk)
    R[direct] = P[direct]
    for row in range(nk):
        if direct[row]:
            continue
        D = np.zeros(nk)
        D[direct] = P[row, direct]
        for i in range(nk):
            for j in range(nk):
                for k in range(nk):
                    inner_jk = D * rule.g_kk[:nk, j, k]
                    inner_ik = D * rule.g_kk[:nk, i, k]
                    inner_ij = D * rule.g_kk[:nk, i, j]
                    out[row, i, j, k] = (
                        np.einsum("d,d->", D, rule.g_kkk[:nk, i, j, k])
                        + np.einsum("ab,a,b->", second.h_kk[row], R[:, i], inner_jk)
                        + np.einsum("ab,a,b->", second.h_kk[row], R[:, j], inner_ik)
                        + np.einsum("ab,a,b->", second.h_kk[row], R[:, k], inner_ij)
                        + np.einsum("abc,a,b,c->", third.h_kkk[row], R[:, i], R[:, j], R[:, k]))
    return out


def _compare(name, builder):
    ex = _load(name)
    if ex is None:
        print(f"{name}: second-order reference missing; run bash run_decision_rule.sh")
        return False
    model, params, _, cov, _ = builder()
    first, endo, residual = _common_solution(model, params, ex)
    second = dsg.solve_second_order(first, cov=cov)
    ours = _dynare_state_timing(second)
    dr_names, state_names, rows, cols, state_rows = _ordering(ex, endo, model)
    scales = first.coords.scale
    sr = scales[rows]
    sk = scales[[model.position(v) for v in state_names]]
    converted = {
        "ghx": ours["g_k"][rows][:, cols] * sr[:, None] / sk[None, :],
        "ghu": ours["g_u"][rows] * sr[:, None],
        "ghxx": (ours["g_kk"][rows][:, cols] * sr[:, None, None]
                 / (sk[None, :, None] * sk[None, None, :])),
        "ghuu": ours["g_uu"][rows] * sr[:, None, None],
        "ghxu": ours["g_ku"][rows][:, cols] * sr[:, None, None] / sk[None, :, None],
        "ghs2": np.concatenate([second.h_s2, second.g_ss])[rows] * sr,
    }
    ok = True
    print(f"{name}: steady residual {residual:.2e}, innovation residual "
          f"{second.innovation_residual:.2e}")
    for key, ours_key in converted.items():
        theirs = np.asarray(getattr(ex, key), float).reshape(ours_key.shape)
        if key not in ("ghu", "ghuu", "ghs2"):
            ours_key, theirs = ours_key[state_rows], theirs[state_rows]
        error = float(np.max(np.abs(ours_key - theirs)))
        passed = error < 1e-8
        ok = ok and passed
        scope = "all rows" if key in ("ghu", "ghuu", "ghs2") else "state rows"
        print(f"  {key:4s}  {scope:10s} max abs {error:.2e}  {'OK' if passed else 'FAIL'}")
    return ok


def _compare_third(name, builder):
    ex = _load(name, order=3)
    if ex is None:
        print(f"{name}: third-order reference missing; run bash run_decision_rule.sh")
        return False
    model, params, _, cov, _ = builder()
    first, endo, residual = _common_solution(model, params, ex)
    second = dsg.solve_second_order(first, cov=cov)
    third = dsg.solve_third_order(second, cov=cov)
    _, state_names, rows, cols, state_rows = _ordering(ex, endo, model)
    sr = first.coords.scale[rows]
    sk = first.coords.scale[[model.position(v) for v in state_names]]
    cubic = _dynare_cubic_rule(third)
    converted = {
        "ghuuu": _dynare_cubic_innovation(third)[rows] * sr[:, None, None, None],
        "ghxuu": (cubic["g_kuu"][rows][:, cols]
                  * sr[:, None, None, None] / sk[None, :, None, None]),
        "ghxxu": (cubic["g_kku"][rows][:, cols]
                  * sr[:, None, None, None] / (sk[None, :, None, None] * sk[None, None, :, None])),
        "ghxxx": (cubic["g_kkk"][rows][:, cols]
                  * sr[:, None, None, None]
                  / (sk[None, :, None, None] * sk[None, None, :, None] * sk[None, None, None, :])),
        "ghxss": (_dynare_risk_rule(third)["g_xss"][rows][:, cols]
                  * sr[:, None] / sk[None, :]),
        "ghuss": _dynare_risk_rule(third)["g_uss"][rows] * sr[:, None],
    }
    ok = True
    print(f"{name}, order 3: steady residual {residual:.2e}, innovation residual "
          f"{third.innovation_residual:.2e}")
    for key, ours in converted.items():
        theirs = np.asarray(getattr(ex, key), float).reshape(ours.shape)
        all_rows = key in ("ghuuu", "ghxuu", "ghxxu", "ghxxx", "ghxss", "ghuss")
        scope = "all rows" if all_rows else "state rows"
        if not all_rows:
            ours, theirs = ours[state_rows], theirs[state_rows]
        error = float(np.max(np.abs(ours - theirs)))
        relative = error / max(float(np.max(np.abs(theirs))), 1e-14)
        passed = error < 2e-4 and relative < 5e-5
        ok = ok and passed
        print(f"  {key:5s} {scope:10s} max abs {error:.2e}, relative {relative:.2e}  "
              f"{'OK' if passed else 'FAIL'}")
    return ok


def main():
    results = []
    for name, builder in (("example1", models.example1), ("example2", models.example2)):
        results += [_compare(name, builder), _compare_third(name, builder)]
    if all(results):
        print("Second-order tensors and all third-order cubic blocks agree with Dynare.")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
