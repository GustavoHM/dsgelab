"""compare_bgp.py — crescimento balanceado, contra o bgp.write do Dynare e os testes dele.

Os três modelos são de tests/bgp do Dynare 7.1, lidos por `dsgelab.from_mod`: Solow com
progresso técnico e população crescendo (só defasagens), um Ramsey com tendência determinística
(defasagens e avanços) e um novo-keynesiano em níveis (só avanços).

Parte 1, o problema. O Dynare escreve a função bgpfun, que avalia o modelo em dois períodos
consecutivos com x_t = y g^t; run_bgp.sh a avalia em cinco pontos fixos. O dsgelab avalia
`dsgelab.bgp.bgp_residual` nos mesmos pontos, com a datação de cada um: o nível do Dynare é o da
defasagem quando o modelo tem defasagens (e então o nível corrente do dsgelab é y g, e o da
auxiliar LAG1_v é y), e o corrente num modelo só com avanços. Compara-se resíduo a resíduo.

Parte 2, a solução. `ModFile.balanced_growth` a partir de partidas aleatórias, como nos testes
do Dynare, e as asserções deles: no Solow, K/Y = s gE gN / (gE gN - 1 + delta), crescimento do
produto e do capital gE gN, nível do crescimento da eficiência gE (300 partidas); no Ramsey,
todo fator de crescimento é 1.02; no novo-keynesiano, níveis e fatores iguais a 1.

Pré-requisitos:
    bash fetch.sh
    bash run_bgp.sh           # exporta out_bgp/<nome>.mat
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from dsgelab.bgp import bgp_residual
from dsgelab.dynare import from_mod

HERE = Path(__file__).resolve().parent
FILES = {"solow": "solow-1", "ramsey": "ramsey-1", "nk": "nk-1"}


def _modfile(name):
    text = (HERE / "mods" / "bgp" / f"{FILES[name]}.mod").read_text()
    cut = min(i for i in (text.find("if ~isoctave"), text.find("verbatim;")) if i >= 0)
    return from_mod(text[:cut])


def residuals_match(name) -> bool:
    from scipy.io import loadmat
    f = HERE / "out_bgp" / f"{name}.mat"
    if not f.exists():
        sys.exit(f"falta {f}: rode bash run_bgp.sh")
    ex = loadmat(f, simplify_cells=True)["ex"]
    names = [str(v).strip() for v in ex["endo_names"]]
    mf = _modfile(name)
    m = mf.model
    nd = len(names)
    lags = any(kind == "lag" for kind, _ in mf.auxiliary.values())
    err = 0.0
    for j in range(ex["Z"].shape[1]):
        y, g = ex["Z"][:nd, j], ex["Z"][nd:, j]
        X, G = np.zeros(m.n_vars), np.ones(m.n_vars)
        for i, v in enumerate(names):
            X[m.position(v)] = y[i] * g[i] if lags else y[i]
            G[m.position(v)] = g[i]
        for aux, (kind, of) in mf.auxiliary.items():
            if kind == "lag":
                i = names.index(of)
                X[m.position(aux)], G[m.position(aux)] = y[i], g[i]
        r = bgp_residual(m, mf.params, X, G)
        n = m.n_vars
        ours = np.concatenate([r[:nd], r[n:n + nd]])      # Dynare's equations, both dates
        err = max(err, float(np.max(np.abs(ours - ex["F"][:, j]))))
    ok = err < 1e-10
    print(f"  {name:7s} resíduos em 5 pontos contra bgpfun: desvio máx {err:.1e}  "
          f"{'OK' if ok else 'DIVERGE'}")
    return ok


def solutions() -> bool:
    ok = True
    mf = _modfile("solow")
    p = mf.params
    gE, gN = p["EfficiencyGrowth_ss"], p["PopulationGrowth_ss"]
    theory = p["s"] * gE * gN / (gE * gN - 1 + p["delta"])
    dev = np.zeros(4)
    for seed in range(300):
        r = mf.balanced_growth(seed=seed)
        ok &= r.success and not r.degenerate
        dev = np.maximum(dev, np.abs([r.ratio("PhysicalCapitalStock", "Output") - theory,
                                      r.growth["Output"] - gE * gN,
                                      r.growth["PhysicalCapitalStock"] - gE * gN,
                                      r.levels["EfficiencyGrowth"] - gE]))
    good = bool(np.all(dev < 1e-8))
    print(f"  solow   300 partidas: K/Y {dev[0]:.1e}, g_Y {dev[1]:.1e}, g_K {dev[2]:.1e}, "
          f"gE {dev[3]:.1e}  {'OK' if good else 'DIVERGE'}")
    ok &= good
    mf = _modfile("ramsey")
    r = mf.balanced_growth(seed=1)
    d = max(abs(r.growth[v] - 1.02) for v in mf.endogenous)
    print(f"  ramsey  fatores de crescimento contra 1.02: {d:.1e}  {'OK' if d < 1e-6 else 'DIVERGE'}")
    ok &= d < 1e-6
    mf = _modfile("nk")
    r = mf.balanced_growth(seed=2)
    d = max(max(abs(r.levels[v] - 1), abs(r.growth[v] - 1)) for v in mf.endogenous)
    print(f"  nk      níveis e fatores contra 1: {d:.1e}  {'OK' if d < 1e-9 else 'DIVERGE'}")
    return ok and d < 1e-9


def main() -> int:
    print("o problema (bgp.write)")
    ok = all([residuals_match(n) for n in FILES])
    print("a solução (asserções dos testes do Dynare)")
    ok &= solutions()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
