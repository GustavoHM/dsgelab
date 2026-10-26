"""Parameter Jacobians of first-order decision rules against Dynare 7.2."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat

import dsgelab as dsg
import models
from compare_decision_rule import _ordering

HERE = Path(__file__).resolve().parent


def _load(name):
    path = HERE / "out_decision_rule" / f"{name}_jac.mat"
    if not path.exists():
        return None
    return loadmat(path, squeeze_me=True, struct_as_record=False)["ex"]


def _solution_at_reference(model, params, ex):
    endo = [str(x).strip() for x in np.atleast_1d(ex.endo_names)]
    values = {v: float(np.atleast_1d(ex.ys)[endo.index(v)]) for v in model.names}
    x = np.array([values[v] for v in model.names])
    residual = float(np.max(np.abs(model.residuals(x, x, np.zeros(model.n_shocks), params))))
    steady = dsg.SteadyState(x=x, values=values, residual=residual)
    first = dsg.solve(model, params, steady=steady, tol=2e-6)
    return first, endo, residual


def _composed_jacobian(first, jac):
    nk, ne = first.model.n_states, first.model.n_shocks
    P, F, Q = first.P, first.F, first.Q
    gk = np.vstack([P, F @ P])
    gu = np.vstack([Q, F @ Q])
    dgk = np.vstack([jac.P, np.moveaxis(np.moveaxis(jac.F, -1, 0) @ P, 0, -1)
                     + np.moveaxis(F @ np.moveaxis(jac.P, -1, 0), 0, -1)])
    dgu = np.vstack([jac.Q, np.moveaxis(np.moveaxis(jac.F, -1, 0) @ Q, 0, -1)
                     + np.moveaxis(F @ np.moveaxis(jac.Q, -1, 0), 0, -1)])
    direct = np.array([not np.allclose(Q[i], 0.0, atol=1e-12) for i in range(nk)])
    R = np.eye(nk)
    S = np.zeros_like(Q)
    dR = np.zeros_like(jac.P)
    dS = np.zeros_like(jac.Q)
    R[direct], S[direct] = P[direct], Q[direct]
    dR[direct], dS[direct] = jac.P[direct], jac.Q[direct]
    for i in range(nk):
        if direct[i]:
            continue
        gk[i], gu[i] = P[i] @ R, P[i] @ S
        dgk[i] = np.einsum("ab,ac->cb", jac.P[i], R) + np.einsum("a,abc->bc", P[i], dR)
        dgu[i] = np.einsum("ab,ac->cb", jac.P[i], S) + np.einsum("a,abc->bc", P[i], dS)
    return gk, gu, dgk, dgu


def _scale_derivative(x, dx):
    out = np.zeros_like(dx)
    mask = np.abs(x) > 1e-8
    out[mask] = np.sign(x[mask])[:, None] * dx[mask]
    return out


def _compare(name, builder):
    ex = _load(name)
    if ex is None:
        print(f"{name}: out_decision_rule/{name}_jac.mat missing; run bash run_derivatives.sh")
        return False
    model, params, _, _, _ = builder()
    first, endo, residual = _solution_at_reference(model, params, ex)
    param_names = [str(x).strip() for x in np.atleast_1d(ex.param_names)]
    jac = dsg.solution_jacobian(first, param_names)
    gk, gu, dgk, dgu = _composed_jacobian(first, jac)
    dr_names, state_names, rows, cols, state_rows = _ordering(ex, endo, model)
    scales = first.coords.scale
    sr, sk = scales[rows], scales[[model.position(v) for v in state_names]]
    dsr = _scale_derivative(first.steady.x[rows], jac.steady[rows])
    dsk = _scale_derivative(first.steady.x[[model.position(v) for v in state_names]],
                            jac.steady[[model.position(v) for v in state_names]])
    base_x = gk[rows][:, cols] * sr[:, None] / sk[None, :]
    dx = np.empty_like(np.asarray(ex.dghx, float))
    du = np.empty_like(np.asarray(ex.dghu, float))
    for j in range(len(param_names)):
        A = dsr[:, j, None] * (gk[rows][:, cols] / sk[None, :])
        B = sr[:, None] * dgk[rows][:, cols, j] / sk[None, :]
        C = base_x * (dsk[:, j] / sk)[None, :]
        dx[:, :, j] = A + B - C
        du[:, :, j] = dsr[:, j, None] * gu[rows] + sr[:, None] * dgu[rows][:, :, j]
    ok = True
    print(f"{name}: steady residual {residual:.2e}")
    for key, ours, scope in (("dghx", dx, "state rows"), ("dghu", du, "all rows")):
        theirs = np.asarray(getattr(ex, key), float).reshape(ours.shape)
        if scope == "state rows":
            ours, theirs = ours[state_rows], theirs[state_rows]
        error = float(np.max(np.abs(ours - theirs)))
        relative = error / max(float(np.max(np.abs(theirs))), 1e-14)
        passed = error < 0.1 and relative < 1e-2
        ok = ok and passed
        print(f"  {key:4s} {scope:10s} max abs {error:.2e}, relative {relative:.2e}  "
              f"{'OK' if passed else 'FAIL'}")
    return ok


def main():
    results = [_compare("example1", models.example1), _compare("example2", models.example2)]
    if all(results):
        print("Both parameter-Jacobian cases agree with Dynare.")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
