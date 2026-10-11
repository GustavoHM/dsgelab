"""compare_foresight.py — previsão perfeita, com e sem erros de expectativa, contra o Dynare.

Os cinco arquivos são de tests/deterministic_simulations do Dynare 7.1, lidos por
`dsgelab.from_mod`:

  rbc_det1.mod   condição inicial fora do estado estacionário (histval)
  rbc_det2.mod   um choque no primeiro período
  rbc_det3.mod   choques ANTECIPADOS nos períodos 4 a 8 — `perfect_foresight(..., shocks=)`
  pfwee.mod                      erros de expectativa: notícias nos períodos 1, 2, 3, 6 e 7,
                                 choques transitórios e permanentes (pfwee.csv)
  pfwee_constant_sim_length.mod  o mesmo, com cada plano de horizonte constante

Datação: a data t da trajetória do dsgelab é o período t+1 do Dynare, e a data T é o período
terminal T+1. Compara-se toda variável endógena do Dynare em todos os períodos de 1 a T+1.

O rbc_det1 calcula a condição inicial num trecho de MATLAB (`CapitalSS/2`, com CapitalSS lido de
oo_.steady_state); from_mod não executa MATLAB, e a condição é escrita aqui com o mesmo significado.

Pré-requisitos:
    bash fetch.sh
    bash run_foresight.sh     # exporta out_foresight/<nome>.mat
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import dsgelab as dk
from dsgelab.dynare import from_mod, read_expectation_errors_csv
from dsgelab.foresight import perfect_foresight_with_expectation_errors

HERE = Path(__file__).resolve().parent
MODS = HERE / "mods" / "foresight"


def _load(name):
    from scipy.io import loadmat
    f = HERE / "out_foresight" / f"{name}.mat"
    if not f.exists():
        return None
    ex = loadmat(f, squeeze_me=True, struct_as_record=False)["ex"]
    ex.endo_names = [str(x).strip() for x in np.atleast_1d(ex.endo_names)]
    ex.endo_simul = np.atleast_2d(np.asarray(ex.endo_simul, float))
    return ex


def _compare(path, ex, mf, T):
    """Largest relative gap over every Dynare variable and every period 1..T+1."""
    worst, where = 0.0, ""
    for i, v in enumerate(ex.endo_names):
        if v not in mf.endogenous:
            continue
        dy = ex.endo_simul[i, 1:T + 2]
        ours = path[v][:T + 1]
        scale = max(1.0, float(np.max(np.abs(dy))))
        d = float(np.max(np.abs(ours - dy))) / scale
        if d > worst:
            worst, where = d, v
    return worst, where


def deterministic(name: str, T: int, histval=None):
    ex = _load(name)
    if ex is None:
        return None
    mf = from_mod(MODS / f"{name}.mod")
    ss = mf.steady_state()
    shocks, k0 = mf.deterministic(T, ss)
    if histval:
        k0 = mf.initial_states(ss, histval=histval(ss),
                               exo_first={e: k0[mf.model.states.index(f"SHK_{e}")]
                                          for e in mf.exogenous})
    path = dk.perfect_foresight(mf.model, ss, ss, params=mf.params, T=T, shocks=shocks,
                                initial_states=k0, tol=1e-12)
    return _compare(path, ex, mf, T)


def expectation_errors(name: str, T: int, constant: bool):
    ex = _load(name)
    if ex is None:
        return None
    mf = from_mod(MODS / f"{name}.mod")
    table = read_expectation_errors_csv(MODS / "pfwee.csv")
    news, start = mf.news(table, T)
    path = perfect_foresight_with_expectation_errors(
        mf.model, start, start, params=mf.params, T=T, news=news,
        initial_states=mf.initial_states(start), constant_simulation_length=constant, tol=1e-12)
    TT = path.levels.shape[0] - 1
    if ex.endo_simul.shape[1] != TT + 2:
        return float("inf"), f"comprimentos diferentes: Dynare {ex.endo_simul.shape[1] - 2}, dsgelab {TT}"
    return _compare(path, ex, mf, TT)


def main() -> int:
    cases = [
        ("rbc_det1", "condição inicial (histval)",
         lambda: deterministic("rbc_det1", 20, histval=lambda ss: {("Capital", 0): ss.values["Capital"] / 2})),
        ("rbc_det2", "choque no período 1", lambda: deterministic("rbc_det2", 300)),
        ("rbc_det3", "choques antecipados, períodos 4 a 8", lambda: deterministic("rbc_det3", 300)),
        ("pfwee", "erros de expectativa", lambda: expectation_errors("pfwee", 7, False)),
        ("pfwee_constant_sim_length", "idem, horizonte constante",
         lambda: expectation_errors("pfwee_constant_sim_length", 7, True)),
    ]
    bad = 0
    for name, what, run in cases:
        r = run()
        if r is None:
            print(f"{name:<27} sem out_foresight/{name}.mat — rode run_foresight.sh")
            bad += 1
            continue
        d, where = r
        ok = d < 1e-8
        print(f"{name:<27} {what:<38} desvio relativo máx {d:.1e} ({where})  "
              f"{'OK' if ok else 'DIVERGE'}")
        bad += not ok
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
