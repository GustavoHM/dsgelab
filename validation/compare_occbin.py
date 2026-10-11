"""compare_occbin.py — restrições ocasionalmente ativas (OccBin) contra o Dynare.

Os dois arquivos são de tests/occbin do Dynare 7.1, lidos por `dsgelab.from_mod` com suas
equações marcadas (`bind=`/`relax=`), o bloco `occbin_constraints`, os choques-surpresa e as
opções de `occbin_setup`/`occbin_solver`:

  borrcon.mod   uma restrição de endividamento; choques-surpresa nos períodos 10 e 30;
                simul_periods=80, simul_maxit=11, simul_curb_retrench
  dynrbc.mod    duas restrições (investimento irreversível e não negativo) com quatro
                regimes; 52 choques-surpresa entre os períodos 1 e 169; simul_periods=200,
                simul_curb_retrench, simul_check_ahead_periods=300 (cenário option=1)

Compara-se, em níveis e em todos os períodos, cada variável endógena da trajetória por partes
(oo_.occbin.endo_piecewise) e da linear (oo_.occbin.endo_linear), e, em cada período de choque,
o regime que os agentes esperam (oo_.occbin.simul.regime_history: regimes e datas de início).

Pré-requisitos:
    bash fetch.sh
    bash run_occbin.sh        # exporta out_occbin/<nome>.mat e escreve work_occbin/<nome>.mod
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from dsgelab.dynare import from_mod

HERE = Path(__file__).resolve().parent
TOL = 1e-12


def _load(name):
    from scipy.io import loadmat
    f = HERE / "out_occbin" / f"{name}.mat"
    if not f.exists():
        sys.exit(f"falta {f}: rode bash run_occbin.sh")
    return loadmat(f, simplify_cells=True)["ex"]


def _starts(ind):
    """Dynare's map_regime: the regime of each spell and the (1-based) date it starts."""
    ind = np.asarray(ind, dtype=int)
    regime, start = [int(ind[0])], [1]
    for i in range(1, len(ind)):
        if ind[i] != regime[-1]:
            regime.append(int(ind[i]))
            start.append(i + 1)
    return regime, start


def _as_list(x):
    return [int(v) for v in np.atleast_1d(x)]


def compare(name, constraints_order):
    ex = _load(name)
    names = [str(v).strip() for v in ex["endo_names"]]
    mf = from_mod((HERE / "work_occbin" / f"{name}.mod").read_text())
    r = mf.occbin_simulate()
    T = ex["piecewise"].shape[0]
    err = {}
    for kind in ("piecewise", "linear"):
        ours = getattr(r, kind)
        err[kind] = max(float(np.max(np.abs(ours[v][:T] - ex[kind][:, j])))
                        for j, v in enumerate(names))
    # regimes: Dynare stores one history per period up to the last surprise
    suffix = [""] if len(constraints_order) == 1 else ["1", "2"]
    bad = 0
    n_hist = len(np.atleast_1d(ex["rh_regime" + suffix[0]]))
    for t in range(1, n_hist + 1):
        exp = r.expected_regimes(t)
        for c, sfx in zip(constraints_order, suffix):
            reg_d = _as_list(np.atleast_1d(ex["rh_regime" + sfx])[t - 1])
            st_d = _as_list(np.atleast_1d(ex["rh_regimestart" + sfx])[t - 1])
            reg_o, st_o = _starts(exp[c])
            if (reg_o, st_o) != (reg_d, st_d):
                bad += 1
    ok = err["piecewise"] < TOL and err["linear"] < TOL and bad == 0
    print(f"{name:10s} {T:4d} períodos  por partes {err['piecewise']:.1e}  linear "
          f"{err['linear']:.1e}  regimes {n_hist - bad}/{n_hist}  iterações "
          f"{sum(r.iterations)}  {'OK' if ok else 'DIVERGE'}")
    return ok


def main() -> int:
    ok = compare("borrcon", ["borrcon"])
    ok &= compare("dynrbc", ["IRR", "INEG"])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
