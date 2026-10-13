"""compare_ep.py — trajetória estendida, determinística e estocástica, contra o Dynare.

Os arquivos são de tests/ep do Dynare 7.2, lidos por `dsgelab.from_mod`; os choques são os que o
Dynare usou, exportados por run_ep.sh.

  burnside.mod      Burnside (1998), 100 períodos, as sete variantes do teste que não pedem
                    perturbação de ordem 4: EP; EP estocástica de ordem 1 e 2 (três nós de
                    Gauss-Hermite), na árvore esparsa e na cheia; e as híbridas de ordem 2, que
                    somam a correção de risco de segunda ordem onde a árvore para de se ramificar
  linearmodel0.mod  modelo linear com dois choques, 200 períodos: a EP e a solução de primeira
                    ordem do próprio Dynare (num modelo linear, as duas coincidem)

Compara-se toda variável endógena em todos os períodos. As híbridas usam a hessiana exata
(`from_mod(..., derivatives="both")`); a tolerância de Newton é 1e-13, abaixo da do Dynare
(1e-12, que run_ep.sh impõe), e é essa que limita a concordância.

Pré-requisitos:
    bash fetch.sh
    bash run_ep.sh            # exporta out_ep/<nome>.mat e escreve work_ep/<nome>.mod
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from dsgelab.dynare import from_mod
from dsgelab.extended_path import extended_path

HERE = Path(__file__).resolve().parent
TOL = 1e-10

BURNSIDE = [
    ("ep0", "EP", dict(order=0)),
    ("sep1", "EP estocástica, ordem 1", dict(order=1)),
    ("sep1h", "  idem, híbrida", dict(order=1, hybrid=True)),
    ("sep2", "EP estocástica, ordem 2", dict(order=2)),
    ("sep2full", "  idem, árvore cheia", dict(order=2, tree="full")),
    ("sep2h", "  idem, híbrida", dict(order=2, hybrid=True)),
    ("sep2hfull", "  idem, híbrida, árvore cheia", dict(order=2, tree="full", hybrid=True)),
]


def _load(name):
    from scipy.io import loadmat
    f = HERE / "out_ep" / f"{name}.mat"
    if not f.exists():
        sys.exit(f"falta {f}: rode bash run_ep.sh")
    return loadmat(f, simplify_cells=True)["ex"]


def _report(label, ours, theirs, names, offset=0):
    err = max(float(np.max(np.abs(ours[v][offset:offset + theirs.shape[0]] - theirs[:, j])))
              for j, v in enumerate(names))
    scale = max(float(np.max(np.abs(theirs))), 1.0)
    ok = err / scale < TOL
    print(f"  {label:34s} {theirs.shape[0]:4d} períodos  desvio máx {err:.1e} "
          f"(relativo {err / scale:.1e})  {'OK' if ok else 'DIVERGE'}")
    return ok


def main() -> int:
    ok = True
    ex = _load("burnside")
    names = [str(v).strip() for v in ex["names"]]
    mf = from_mod((HERE / "work_ep" / "burnside.mod").read_text().split("resid;")[0],
                  derivatives="both")
    ss = mf.steady_state()
    E = np.asarray(ex["e"], dtype=float).reshape(-1, 1)[:100]
    print("burnside.mod")
    for key, label, kw in BURNSIDE:
        r = extended_path(mf.model, ss, mf.params, E, cov=mf.cov, tol=1e-13, **kw)
        ok &= _report(label, r, ex[key], names)

    ex = _load("linearmodel0")
    names = [str(v).strip() for v in ex["names"]]
    mf = from_mod((HERE / "work_ep" / "linearmodel0.mod").read_text().split("stoch_simul")[0])
    r = extended_path(mf.model, mf.steady_state(), mf.params, ex["e"][:200], tol=1e-13)
    print("linearmodel0.mod")
    ok &= _report("EP", r, ex["ep0"], names)
    ok &= _report("solução de primeira ordem do Dynare", r, ex["linear"], names, offset=1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
