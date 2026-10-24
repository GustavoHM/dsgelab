"""First-order IRF matching against Dynare's official frequentist RBC example."""
from pathlib import Path
import warnings

import numpy as np
import scipy.io as sio

import dsgelab as dsg

HERE = Path(__file__).resolve().parent


def _relative(ours, reference):
    ours, reference = np.asarray(ours, float), np.asarray(reference, float)
    return np.max(np.abs(ours - reference) / np.maximum(np.abs(reference), 1e-10), initial=0.0)


def compare(suffix: str):
    name = f"rbc_irf_{suffix}"
    text = (HERE / "work_mom" / f"{name}.mod").read_text()
    mf = dsg.from_mod(text[:text.index("method_of_moments(")])
    ex = sio.loadmat(HERE / "out_mom" / f"{name}.mat", squeeze_me=True,
                     struct_as_record=False)["ex"]
    ref = ex.mom
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        own = mf.irf_matching(optimize_=suffix != "0")
        dyn_point = {str(n): float(v) for n, v in zip(ex.parameter_names, ex.mode)}
        at_dyn = own if suffix == "0" else mf.irf_matching(
            free=dyn_point, optimize_=False)
    dyn_std = ref.irf_matching_std_at_mode.parameters
    reference_se = np.array([float(getattr(dyn_std, n)) for n in own.names]) if suffix != "0" else np.full(len(own.names), np.nan)
    errors = dict(data=_relative(at_dyn.data_irfs, ref.data_moments),
                  model=_relative(at_dyn.model_irfs, ref.model_moments),
                  weighting=_relative(at_dyn.weighting, ref.weighting_info.W.toarray()),
                  objective=_relative(at_dyn.objective, ref.Q),
                  hessian=_relative(at_dyn.hessian, ex.hessian),
                  stderr=_relative([own.stderr[n] for n in own.names], reference_se)
                  if suffix != "0" else 0.0)
    checks = [errors["data"] < 1e-12, errors["model"] < 2e-8,
              errors["weighting"] < 1e-12, errors["objective"] < 2e-8,
              errors["hessian"] < 3e-5, errors["stderr"] < 1e-4]
    estimate_error = np.max(np.abs(own.x - np.asarray(ex.mode, float)))
    print(f"  {name:10s} data {errors['data']:.1e}  model {errors['model']:.1e}  "
          f"W {errors['weighting']:.1e}  Q {errors['objective']:.1e}  "
          f"H {errors['hessian']:.1e}  s.e. {errors['stderr']:.1e}  est. {estimate_error:.1e}  "
          f"{'OK' if all(checks) else 'FAIL'}")
    return [] if all(checks) else [f"{name}: {errors}"]


def main():
    import sys
    cases = sys.argv[1:] or ["0", "5"]
    failures = []
    for suffix in cases:
        try:
            failures.extend(compare(suffix))
        except Exception as err:
            failures.append(f"{suffix}: {err}")
            print(f"  rbc_irf_{suffix} ERROR {err}")
    if failures:
        print("\n".join(failures))
        raise SystemExit(1)
    print(f"All {len(cases)} IRF-matching cases agree with Dynare.")


if __name__ == "__main__":
    main()
