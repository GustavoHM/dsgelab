"""compare_identification.py: local identification, against Dynare's identification command.

Three models, each at the prior mean of its estimated parameters, which is where Dynare's
identification looks by default:

- kim2 (tests/identification/kim of Dynare 7.2), Kim (2003): phi and theta only enter
  together, and dumpy does not enter at all, so no criterion has full rank. Dynare's test runs
  it with tol_rank=1e-8.
- as2007 (tests/identification/as2007), An and Schorfheide (2007): 13 parameters, all
  identified.
- fs2000 (examples of Dynare 6.4): 7 parameters and 2 shock standard deviations, which Dynare
  puts first in its columns.

What is compared:

1. The objects. The moments (means, covariances, first autocovariances of the observables), their
   Jacobian with respect to the parameters (Iskrev's J), and Qu and Tkachenko's G. Dynare
   differentiates analytically and dsgelab by differences with a Richardson step.
   Komunjer and Ng's matrix is compared by its size only, which gives the size of the minimal
   state: its entries depend on the basis chosen for the minimal system, and each package
   chooses its own.
2. The verdicts, for each criterion: the rank, which parameters move it (ind0), the sets of
   linearly dependent parameters (indno), the weakly identified parameters (jweak) and pairs
   (jweak_pair). Twice: with Dynare's tolerance (tol_rank="robust", or 1e-8 for kim2) and with
   dsgelab's default, tol_rank="auto", which also counts as zero a singular value below the
   estimated error of the derivatives.
3. The continuous measures, where they are well determined: the multicollinearity of each
   parameter, for the moments and the spectrum, when the criterion has full rank; and the
   brute-force collinearity search (the best group of k other parameters for each one, with
   its cosine), for groups of one parameter always and for larger groups when the moments have
   full rank. When two regressors are collinear to rounding error, the regression on them
   measures that error, and the two packages disagree for that reason only (kim2: alph on phi
   and theta). The multicollinearity of Komunjer and Ng's matrix depends on the basis, as above.

Prerequisites:
    bash fetch.sh
    bash run_identification.sh      # exports out_identification/<name>.mat
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import scipy.io as sio

from dsgelab.dynare import from_mod

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "identification"))
from working_copy import working_copy  # noqa: E402

MODELS = {"kim2": ("identification/kim2.mod", 3, 1e-8),
          "as2007": ("identification/as2007.mod", 3, "robust"),
          "fs2000": ("examples/fs2000.mod", 2, "robust")}
CRITERIA = ("moments", "spectrum", "minimal")
TOL_OBJECTS = 1e-7
TOL_MEASURES = 1e-6


def _dynare_names(ex) -> list:
    exo = [str(x) for x in np.atleast_1d(ex["exo_names"])]
    par = [str(x) for x in np.atleast_1d(ex["param_names"])]
    out = [f"stderr {exo[int(i) - 1]}" for i in np.atleast_1d(ex["stderr_index"])]
    return out + [par[int(i) - 1] for i in np.atleast_1d(ex["param_index"])]


def _pairs_from_vech(v, names) -> set:
    """Dynare's dyn_vech of a symmetric indicator: the upper triangle, column by column."""
    v = np.atleast_1d(v).astype(bool)
    out, k = set(), 0
    for j in range(len(names)):
        for i in range(j + 1):
            if v[k] and i != j:
                out.add(frozenset((names[i], names[j])))
            k += 1
    return out


def compare(name: str) -> list:
    src, max_group, dynare_tol = MODELS[name]
    ex = sio.loadmat(HERE / "out_identification" / f"{name}.mat", squeeze_me=True)
    mf = from_mod(working_copy((HERE / "mods" / src).read_text(), ""))
    dnames = _dynare_names(ex)
    fails = []

    def check(ok, what):
        print(f"    {'ok  ' if ok else 'FAIL'} {what}")
        if not ok:
            fails.append(f"{name}: {what}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        runs = {str(t): mf.identification(max_group=max_group, tol_rank=t)
                for t in (dynare_tol, "auto")}
    r = runs["auto"]
    order = [r.names.index(n) for n in dnames]
    print(f"  {name}: {len(dnames)} parameters, observing {', '.join(r.observables)}")

    m_dy = np.atleast_1d(ex["MOMENTS"]).astype(float)
    d = float(np.max(np.abs(r.moments - m_dy) / np.maximum(np.abs(m_dy), 1e-12)))
    check(d < 1e-10, f"moments: largest relative difference {d:.1e}")
    J = r["moments"].jacobian[:, order]
    J_dy = np.asarray(ex["dMOMENTS"], float).reshape(J.shape)
    d = float(np.max(np.abs(J - J_dy) / np.maximum(np.linalg.norm(J_dy, axis=0), 1e-300)))
    check(d < TOL_OBJECTS, f"Jacobian of the moments: largest difference / column norm {d:.1e}")
    G = r["spectrum"].jacobian[np.ix_(order, order)]
    G_dy = np.atleast_2d(np.asarray(ex["dSPECTRUM"], float))
    d = float(np.max(np.abs(G - G_dy)) / np.max(np.abs(G_dy)))
    check(d < TOL_OBJECTS, f"G of the spectrum: largest difference / largest entry {d:.1e}")
    size = tuple(int(x) for x in np.atleast_1d(ex["dMINIMAL_size"]))
    check(r["minimal"].jacobian.shape == size,
          f"Komunjer and Ng's matrix {r['minimal'].jacobian.shape}, Dynare {size}; "
          f"minimal state of size {r.minimal_states}")

    for tol, run in runs.items():
        for crit in CRITERIA:
            c = run[crit]
            rank_dy = int(ex[f"{crit}_rank"])
            ind0 = np.atleast_1d(ex[f"{crit}_ind0"]).astype(bool)
            weak = np.atleast_1d(ex[f"{crit}_jweak"]).astype(bool)
            indno = np.atleast_2d(ex[f"{crit}_indno"]).astype(bool)
            dep_dy = sorted(sorted(n for n, b in zip(dnames, row) if b) for row in indno if row.any())
            dep = sorted(sorted(g) for g in c.dependent)
            same = (c.rank == rank_dy
                    and [c.sensitive[n] for n in dnames] == list(ind0)
                    and sorted(c.weak) == sorted(n for n, b in zip(dnames, weak) if b)
                    and dep == dep_dy
                    and {frozenset(p) for p in c.weak_pairs}
                    == _pairs_from_vech(ex[f"{crit}_jweak_pair"], dnames))
            check(same, f"{crit:8s} tol_rank={tol}: rank {c.rank} of {c.required} "
                        f"(Dynare {rank_dy}), sensitive, dependent, weak: "
                        f"{'the same' if same else 'DIFFERENT'}"
                        + (f"; dependent {[', '.join(g) for g in dep]}" if dep else ""))

    for crit in ("moments", "spectrum"):
        if not r[crit].identified:
            continue
        mco = np.array([r[crit].multicollinearity[n] for n in dnames])
        mco_dy = np.atleast_1d(ex[f"{crit}_Mco"]).astype(float)
        d = float(np.max(np.abs(mco - mco_dy)))
        check(d < TOL_MEASURES, f"multicollinearity, {crit}: largest difference {d:.1e}")
    if "cosn" in ex:
        cos_dy = np.atleast_2d(ex["cosn"])
        pars = ex["pars"]
        full = r["moments"].identified
        worst, groups = 0.0, []
        for k, table in r.collinearity.items():
            if k > 1 and not full:
                continue
            for i, n in enumerate(dnames):
                grp, c = table[n]
                worst = max(worst, abs(c - cos_dy[i, k - 1]))
                dy = sorted(dnames[int(x) - 1] for x in np.atleast_1d(pars[i, k - 1])[:k]
                            if np.isfinite(x))
                if sorted(grp) != dy:
                    groups.append(f"{n} ({k})")
        which = "all group sizes" if full else "groups of one"
        check(worst < TOL_MEASURES and not groups,
              f"collinearity search, {which}: largest cosine difference {worst:.1e}, "
              f"groups {'the same' if not groups else 'different for ' + ', '.join(groups)}")
    return fails


def main() -> int:
    missing = [n for n in MODELS if not (HERE / "out_identification" / f"{n}.mat").exists()]
    if missing:
        sys.exit(f"missing out_identification/{missing[0]}.mat: run run_identification.sh")
    fails = []
    for name in MODELS:
        fails += compare(name)
    print()
    if fails:
        print(f"{len(fails)} comparison(s) failed:")
        for f in fails:
            print("  " + f)
        return 1
    print("every comparison passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
