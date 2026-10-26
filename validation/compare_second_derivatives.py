"""Parameter Jacobians of second-order decision rules against Dynare 7.2."""
from pathlib import Path

import numpy as np
from scipy.io import loadmat

import dsgelab as dsg
import models
from compare_decision_rule import _common_solution, _load, _ordering
from compare_derivatives import _scale_derivative
from dsgelab.secondorder import _dynare_decision_rule

HERE = Path(__file__).resolve().parent


def _g_s2_stderr_diagonal(second, name):
    """Dynare 7.2 omits cross-covariance changes in this particular derivative."""
    model = second.first.model
    i = model.shocks.index(name[7:].strip())
    sd = float(np.sqrt(second.Sigma[i, i]))
    h = 1e-5 * sd
    values = []
    for sign in (1, -1):
        Sigma = second.Sigma.copy()
        Sigma[i, i] = (sd + sign * h) ** 2
        values.append(_dynare_decision_rule(dsg.solve_second_order(second.first, cov=Sigma)).g_s2)
    return (values[0] - values[1]) / (2 * h)


def compare(name, builder):
    path = HERE / "out_decision_rule" / f"{name}_jac2.mat"
    if not path.exists():
        print(f"{name}: reference missing; run bash run_second_derivatives.sh")
        return False
    ex = loadmat(path, squeeze_me=True, struct_as_record=False)["ex"]
    model, params, _, cov, _ = builder()
    reference = _load(name)
    first, endo, residual = _common_solution(model, params, reference)
    second = dsg.solve_second_order(first, cov=cov)
    model_names = [str(x).strip() for x in np.atleast_1d(ex.param_names)]
    param_names = [f"stderr {e}" for e in model.shocks]
    param_names += [f"corr {model.shocks[0]}, {model.shocks[1]}"] + model_names
    jac = dsg.second_order_jacobian(second, param_names)
    _, state_names, rows, cols, _ = _ordering(reference, endo, model)
    scales = first.coords.scale
    sr = scales[rows]
    sk = scales[[model.position(v) for v in state_names]]
    dsr = _scale_derivative(first.steady.x[rows], jac.steady[rows])
    dsk = _scale_derivative(first.steady.x[[model.position(v) for v in state_names]],
                            jac.steady[[model.position(v) for v in state_names]])
    rule = _dynare_decision_rule(second)
    g_s2_dynare = jac.g_s2.copy()
    for j, parameter in enumerate(param_names):
        if parameter.startswith("stderr "):
            g_s2_dynare[:, j] = _g_s2_stderr_diagonal(second, parameter)
    ok = True
    print(f"{name}: steady residual {residual:.2e}")
    for key, attr in (("dghxx", "g_kk"), ("dghxu", "g_ku"),
                      ("dghuu", "g_uu"), ("dghs2", "g_s2")):
        G = getattr(rule, attr)[rows]
        dG = g_s2_dynare[rows] if attr == "g_s2" else getattr(jac, attr)[rows]
        if attr in ("g_kk", "g_ku"):
            G, dG = G[:, cols], dG[:, cols]
        ours = np.empty_like(dG)
        for j in range(len(param_names)):
            if attr == "g_kk":
                denominator = sk[None, :, None] * sk[None, None, :]
                base = G * sr[:, None, None] / denominator
                ours[:, :, :, j] = (dsr[:, j, None, None] * (G / denominator)
                                    + sr[:, None, None] * dG[:, :, :, j] / denominator
                                    - base * (dsk[:, j] / sk)[None, :, None]
                                    - base * (dsk[:, j] / sk)[None, None, :])
            elif attr == "g_ku":
                base = G * sr[:, None, None] / sk[None, :, None]
                ours[:, :, :, j] = (dsr[:, j, None, None] * (G / sk[None, :, None])
                                    + sr[:, None, None] * dG[:, :, :, j] / sk[None, :, None]
                                    - base * (dsk[:, j] / sk)[None, :, None])
            elif attr == "g_s2":
                ours[:, j] = dsr[:, j] * G + sr * dG[:, j]
            else:
                ours[:, :, :, j] = dsr[:, j, None, None] * G + sr[:, None, None] * dG[:, :, :, j]
        theirs = np.asarray(getattr(ex, key), float).reshape(ours.shape)
        error = float(np.max(np.abs(ours - theirs)))
        relative = error / max(float(np.max(np.abs(theirs))), 1e-14)
        passed = error < 0.1 and relative < 1e-2
        ok = ok and passed
        print(f"  {key:5s} all rows   max abs {error:.2e}, relative {relative:.2e}  "
              f"{'OK' if passed else 'FAIL'}")
    return ok


def main():
    results = [compare(name, builder)
               for name, builder in (("example1", models.example1), ("example2", models.example2))]
    if all(results):
        print("Second-order decision-rule Jacobians agree with Dynare.")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
