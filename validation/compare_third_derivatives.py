"""Parameter Jacobians of third-order cubic decision rules against Dynare 7.2."""
from pathlib import Path

import numpy as np
from scipy.io import loadmat

import dsgelab as dsg
import models
from compare_decision_rule import _common_solution, _load, _ordering
from compare_derivatives import _scale_derivative
from compare_second_derivatives import _g_s2_stderr_diagonal
from dsgelab.secondorder import _dynare_decision_rule
from dsgelab.thirdorder import _dynare_cubic_rule

HERE = Path(__file__).resolve().parent


def compare(name, builder):
    path = HERE / "out_decision_rule" / f"{name}_jac3.mat"
    if not path.exists():
        print(f"{name}: reference missing; run bash run_third_derivatives.sh")
        return False
    ex = loadmat(path, squeeze_me=True, struct_as_record=False)["ex"]
    model, params, _, cov, _ = builder()
    reference = _load(name, order=3)
    first, endo, residual = _common_solution(model, params, reference)
    second = dsg.solve_second_order(first, cov=cov)
    third = dsg.solve_third_order(second, sparse=True, direct_limit=0)
    model_names = [str(x).strip() for x in np.atleast_1d(ex.param_names)]
    param_names = [f"stderr {e}" for e in model.shocks]
    param_names += [f"corr {model.shocks[0]}, {model.shocks[1]}"] + model_names
    jac = dsg.third_order_jacobian(third, param_names, sparse=True, direct_limit=0)
    _, state_names, rows, cols, state_rows = _ordering(reference, endo, model)
    scales = first.coords.scale
    sr, sk = scales[rows], scales[[model.position(v) for v in state_names]]
    dsr = _scale_derivative(first.steady.x[rows], jac.steady[rows])
    dsk = _scale_derivative(first.steady.x[[model.position(v) for v in state_names]],
                            jac.steady[[model.position(v) for v in state_names]])
    cubic = _dynare_cubic_rule(third)
    g_s2_dynare = jac.g_s2.copy()
    for j, parameter in enumerate(param_names):
        if parameter.startswith("stderr "):
            g_s2_dynare[:, j] = _g_s2_stderr_diagonal(second, parameter)
    ok = True
    print(f"{name}: steady residual {residual:.2e}")
    for key, attr, order in (("dghxxx", "g_kkk", 3), ("dghxxu", "g_kku", 2),
                             ("dghxuu", "g_kuu", 1), ("dghuuu", "g_uuu", 0),
                             ("dghs2", "g_s2", -1)):
        if attr == "g_s2":
            G = _dynare_decision_rule(second).g_s2[rows]
            dG = g_s2_dynare[rows]
        else:
            G = cubic[attr][rows]
            dG = jac.__getattribute__(attr)[rows]
            if attr != "g_uuu":
                G, dG = G[:, cols], dG[:, cols]
        ours = np.empty_like(dG)
        for j in range(len(param_names)):
            if attr == "g_s2":
                ours[:, j] = dsr[:, j] * G + sr * dG[:, j]
                continue
            if attr == "g_kkk":
                denominator = (sk[None, :, None, None] * sk[None, None, :, None]
                               * sk[None, None, None, :])
                base = G * sr[:, None, None, None] / denominator
                ours[..., j] = (dsr[:, j, None, None, None] * (G / denominator)
                                + sr[:, None, None, None] * dG[..., j] / denominator
                                - base * (dsk[:, j] / sk)[None, :, None, None]
                                - base * (dsk[:, j] / sk)[None, None, :, None]
                                - base * (dsk[:, j] / sk)[None, None, None, :])
            elif attr == "g_kku":
                denominator = sk[None, :, None, None] * sk[None, None, :, None]
                base = G * sr[:, None, None, None] / denominator
                ours[..., j] = (dsr[:, j, None, None, None] * (G / denominator)
                                + sr[:, None, None, None] * dG[..., j] / denominator
                                - base * (dsk[:, j] / sk)[None, :, None, None]
                                - base * (dsk[:, j] / sk)[None, None, :, None])
            elif attr == "g_kuu":
                base = G * sr[:, None, None, None] / sk[None, :, None, None]
                ours[..., j] = (dsr[:, j, None, None, None] * (G / sk[None, :, None, None])
                                + sr[:, None, None, None] * dG[..., j] / sk[None, :, None, None]
                                - base * (dsk[:, j] / sk)[None, :, None, None])
            else:
                base = G * sr[:, None, None, None]
                ours[..., j] = dsr[:, j, None, None, None] * G + sr[:, None, None, None] * dG[..., j]
        theirs = np.asarray(getattr(ex, key), float).reshape(ours.shape)
        if attr != "g_uuu":
            ours, theirs = ours[state_rows], theirs[state_rows]
        error = float(np.max(np.abs(ours - theirs)))
        relative = error / max(float(np.max(np.abs(theirs))), 1e-14)
        passed = error < 0.1 and relative < 5e-2
        ok = ok and passed
        print(f"  {key:6s} max abs {error:.2e}, relative {relative:.2e}  {'OK' if passed else 'FAIL'}")
    return ok


def main():
    results = [compare(name, builder)
               for name, builder in (("example1", models.example1), ("example2", models.example2))]
    if all(results):
        print("Third-order cubic Jacobians agree with Dynare.")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
