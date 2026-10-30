"""Calibration restrictions against Dynare's irf_calibration and moment_calibration.

The model and both blocks come from Dynare's tests/gsa/ls2003a.mod (Lubik and
Schorfheide, 2003), prepared by restrictions/prepare.py and run by
run_restrictions.sh. Each case is compared twice:

* at the calibration, with Dynare's endogenous_prior_restrictions: every value at
  every horizon or lag, and each restriction's verdict;
* over the 512 prior draws of Dynare's sensitivity toolbox (map_calibration),
  evaluated by dsgelab at the same parameter values: which draws solve, every
  value of every solved draw, each restriction's verdict per draw, and the shares
  Dynare prints.
"""
from pathlib import Path
import sys
import warnings

import numpy as np
import scipy.io as sio

import dsgelab as dsg

HERE = Path(__file__).resolve().parent
TOL_VALUE = 1e-9          # relative to max(1, |value|)


def _cells(x) -> list:
    return [np.atleast_1d(np.asarray(c, dtype=float)) for c in np.atleast_1d(x)]


def _gap(ours, reference) -> float:
    ours, reference = np.asarray(ours, float), np.asarray(reference, float)
    if ours.shape != reference.shape:
        return np.inf
    both = np.isnan(ours) & np.isnan(reference)
    if np.any(np.isnan(ours) != np.isnan(reference)):
        return np.inf
    d = np.abs(ours - reference) / np.maximum(1.0, np.abs(reference))
    return float(np.max(np.where(both, 0.0, d), initial=0.0))


def compare(name: str) -> list[str]:
    mf = dsg.from_mod((HERE / "work_restrictions" / f"{name}.mod").read_text())
    ex = sio.loadmat(HERE / "out_restrictions" / f"{name}.mat", squeeze_me=True,
                     struct_as_record=False)["ex"]
    pt = ex.point
    failures = []
    n_irf, n_mom = len(mf.irf_calibration), len(mf.moment_calibration)
    if mf.relative_irf != bool(ex.relative_irf):
        failures.append(f"{name}: relative_irf read as {mf.relative_irf}")

    # --- at the calibration
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        point = mf.check_restrictions()
    # data_irf{j} and data_moment{j} are [period, value] rows
    ref_point = [c.reshape(-1, 2)[:, 1] for c in _cells(pt.data_irf) + _cells(pt.data_moment)]
    gap_point = max(_gap(v, r) for v, r in zip(point.values, ref_point))
    dyn_verdict = np.concatenate([np.atleast_2d(pt.info_irf)[:, 0],
                                  np.atleast_2d(pt.info_moment)[:, 0]]) == 0
    verdict_point = bool(np.array_equal(point.holds_each, dyn_verdict))

    # --- over Dynare's prior draws
    names = [str(n) for n in np.atleast_1d(ex.names)]
    shocks = set(mf.exogenous)
    names = [f"stderr {n}" if n in shocks else n for n in names]
    X = np.atleast_2d(ex.lpmat)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sample = mf.sample_restrictions(points=X, names=names)
    solved_ref = np.zeros(X.shape[0], bool)
    solved_ref[np.atleast_1d(ex.irestrictions).astype(int) - 1] = True
    same_solved = bool(np.array_equal(sample.solved, solved_ref))
    rows = solved_ref & sample.solved
    mats = _cells_2d(ex.mat_irf, X.shape[0]) + _cells_2d(ex.mat_moment, X.shape[0])
    gap_draws = max(_gap(v[rows], m[rows]) for v, m in zip(sample.values, mats))
    keep = np.atleast_1d(ex.irestrictions).astype(int) - 1
    indx = np.column_stack([np.atleast_2d(ex.indx_irf)[keep], np.atleast_2d(ex.indx_moment)[keep]])
    ours = sample.holds_each[keep]
    verdict_draws = int(np.sum(ours != (indx == 0)))
    share_gap = max(abs(sample.share([i]) - float(np.mean(indx[:, i] == 0)))
                    for i in range(indx.shape[1]))

    ok = (gap_point < TOL_VALUE and verdict_point and same_solved and gap_draws < TOL_VALUE
          and verdict_draws == 0 and share_gap == 0.0)
    print(f"  {name:11s} {n_irf} IRF + {n_mom} moment restrictions   "
          f"calibration: values {gap_point:.1e}, verdicts {'same' if verdict_point else 'DIFFER'}   "
          f"{X.shape[0]} draws ({int(solved_ref.sum())} solved, "
          f"{'same' if same_solved else 'DIFFERENT'} draws): values {gap_draws:.1e}, "
          f"verdicts differ in {verdict_draws}   {'OK' if ok else 'FAIL'}")
    print(f"  {'':11s} shares: {sample.share([0]) * 100:.1f}% {sample.restrictions[0].label}, "
          f"{sample.share(range(1, n_irf)) * 100:.1f}% the other IRF restrictions together, "
          f"{sample.share() * 100:.1f}% all")
    if not ok:
        failures.append(f"{name}: calibration {gap_point:.2e} {verdict_point}, draws "
                        f"{same_solved} {gap_draws:.2e} {verdict_draws} {share_gap:.2e}")
    return failures


def _cells_2d(x, n: int) -> list[np.ndarray]:
    """mat_irf{j} and mat_moment{j}: one row per draw, one column per period."""
    return [np.asarray(c, dtype=float).reshape(n, -1) for c in np.atleast_1d(x)]


def main():
    cases = sys.argv[1:] or ["ls2003_rel", "ls2003_abs"]
    failures = []
    for name in cases:
        try:
            failures.extend(compare(name))
        except Exception as err:
            failures.append(f"{name}: {err}")
            print(f"  {name} ERROR {err}")
    if failures:
        print("\n".join(failures))
        sys.exit(1)


if __name__ == "__main__":
    main()
