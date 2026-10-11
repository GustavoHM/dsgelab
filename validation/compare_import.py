"""compare_import.py — os exemplos do próprio Dynare, lidos por dsgelab.from_mod e resolvidos.

compare.py confronta portes feitos À MÃO de três modelos. Aqui nenhum modelo é reescrito: o
`.mod` do Dynare é lido como está por `dsgelab.dynare.from_mod`, e o que se compara é o que sai
dos dois programas a partir do mesmo arquivo. São oito modelos: os três de compare.py e cinco da
pasta examples/ do repositório do Dynare (tag 6.4), que exercitam o que um porte à mão não
exercitaria — c(+2) e choques dentro da função de produção (fs2000), variáveis locais `#` e o
operador steady_state() (Gali_2015), o macroprocessador (agtrend), choques correlacionados
(example1, example3) e um modelo médio de 28 equações com indexação e rigidez de salários
(NK_baseline).

O que vem do Dynare, e por quê:
  * os parâmetros que o .mod NÃO define. O NK_baseline e o RBC_baseline calculam alguns num
    arquivo MATLAB de estado estacionário, que um leitor de .mod não executa. Os que o .mod
    define são lidos por from_mod e CONFERIDOS contra os do Dynare;
  * o ponto de partida do estado estacionário, só para as variáveis cujo valor o .mod não dá em
    forma avaliável (um auxiliar MATLAB, um steady_state_model ausente). O estado estacionário em
    si é resolvido pelo dsgelab e comparado.

O que se compara, como em compare.py: estado estacionário, IRFs ortogonalizadas (Cholesky) de
todas as variáveis a todos os choques com variância, 40 períodos, e desvios-padrão teóricos.
Nenhuma série é deslocada: from_mod mantém a datação do Dynare.

Pré-requisitos:
    bash fetch.sh
    bash run_examples.sh     # exporta out_import/<nome>.mat
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

import dsgelab as dk
from dsgelab.dynare import from_mod

HERE = Path(__file__).resolve().parent
H = 40
TOL = 1e-6


def _load(name):
    from scipy.io import loadmat
    f = HERE / "out_import" / f"{name}.mat"
    if not f.exists():
        return None
    ex = loadmat(f, squeeze_me=True, struct_as_record=False)["ex"]
    for k in ("endo_names", "exo_names", "param_names"):
        setattr(ex, k, [str(x).strip() for x in np.atleast_1d(getattr(ex, k))])
    return ex


def _mods():
    out = {}
    for f in sorted((HERE / "mods").glob("*.mod")) + sorted((HERE / "mods" / "examples").glob("*.mod")):
        out[f.stem] = f
    return out


def compare(name: str, path: Path) -> dict:
    ex = _load(name)
    if ex is None:
        return dict(name=name, missing=True)
    mf = from_mod(path)
    model = mf.model

    # --- parâmetros: os do .mod conferidos; os que faltam, do Dynare
    dy_par = dict(zip(ex.param_names, np.atleast_1d(ex.params).astype(float)))
    params = dict(mf.params)
    d_par, from_dynare = 0.0, []
    for p, v in dy_par.items():
        if p not in params:
            continue
        if math.isnan(params[p]):
            params[p] = v
            from_dynare.append(p)
        else:
            d_par = max(d_par, abs(params[p] - v) / max(abs(v), 1e-12))

    # --- estado estacionário
    ys = dict(zip(ex.endo_names, np.atleast_1d(ex.ys).astype(float)))
    # um modelo linear sem initval parte de zero, como o próprio Dynare
    start = {} if mf.linear else {v: ys[v] for v in mf.missing_guess}
    ss = mf.steady_state(guess=start, params=params)
    d_ss = max(abs(ss.values[v] - ys[v]) / max(abs(ys[v]), 1e-8) for v in mf.endogenous)
    zero = np.zeros(model.n_shocks)
    x_dy = model.vector(mf.full_guess(ys))
    r_dy = float(np.max(np.abs(model.residuals(x_dy, x_dy, zero, params))))
    r_dk = float(np.max(np.abs(model.residuals(ss.x, ss.x, zero, params))))

    # --- IRFs ortogonalizadas e momentos
    sol = dk.solve(model, params, steady=ss, qz_criterium=1 + 1e-6)   # a omissão do Dynare
    cov = np.atleast_2d(np.asarray(ex.sigma_e, float))
    live = [i for i in range(len(ex.exo_names)) if cov[i, i] > 0]
    L = np.zeros_like(cov)
    if live:
        L[np.ix_(live, live)] = np.linalg.cholesky(cov[np.ix_(live, live)])
    order = [model.shocks.index(e) for e in ex.exo_names]
    scale = sol.coords.scale
    curves = {}
    for j in live:
        imp = np.zeros(model.n_shocks)
        imp[order] = L[:, j]
        dev = sol.irf_impulse(imp, periods=H, percent=False)
        for v in mf.endogenous:
            attr = f"{v}_{ex.exo_names[j]}"
            dy = (np.atleast_1d(getattr(ex.irfs, attr)).astype(float)[:H]
                  if hasattr(ex.irfs, attr) else np.zeros(H))
            curves[(v, ex.exo_names[j])] = (np.asarray(dev[v])[:H] * scale[model.position(v)], dy)
    ruler = {}
    for (v, _), (a, b) in curves.items():
        ruler[v] = max(ruler.get(v, 0.0), float(np.max(np.abs(b))), float(np.max(np.abs(a))))
    d_irf, worst = 0.0, ""
    for (v, e), (a, b) in curves.items():
        if ruler[v] < 1e-12:
            continue
        d = float(np.max(np.abs(a - b))) / ruler[v]
        if d > d_irf:
            d_irf, worst = d, f"{v} <- {e}"

    d_mom = None
    if hasattr(ex, "variance"):
        V = np.atleast_2d(np.asarray(ex.variance, float))
        dy_sd = dict(zip(ex.endo_names, np.sqrt(np.maximum(np.diag(V), 0.0))))
        full = np.zeros((model.n_shocks, model.n_shocks))
        full[np.ix_(order, order)] = cov
        mom = sol.moments(cov=full)
        target = [v for v in mf.endogenous if dy_sd.get(v, 0.0) > 1e-9]
        d_mom = max(abs(mom["std_by_name"][v] * scale[model.position(v)] - dy_sd[v]) / dy_sd[v]
                    for v in target) if target else 0.0

    # Quando o e.e. do Dynare deixa resíduo, o que dele decorre herda o erro; a tolerância acompanha.
    floor = max(TOL, 30.0 * r_dy)
    ss_ok = d_ss < 1e-9 or r_dy > 1e3 * max(r_dk, 1e-16)
    ok = ss_ok and d_irf < floor and (d_mom is None or d_mom < floor) and d_par < 1e-12
    return dict(name=name, ok=ok, par=d_par, from_dynare=from_dynare,
                guess_from_dynare=len(start), ss=d_ss, irf=d_irf, worst=worst, mom=d_mom,
                r_dy=r_dy, r_dk=r_dk, floor=floor, n=len(mf.endogenous),
                aux=len(mf.auxiliary), shocks=len(live))


def main() -> int:
    bad = 0
    print(f"{'modelo':<14}{'vars':>5}{'aux':>5}  {'parâm.':>8} {'e.e.':>8} {'IRFs':>8} "
          f"{'momentos':>9}")
    for name, path in _mods().items():
        r = compare(name, path)
        if r.get("missing"):
            print(f"{name:<14} sem out_import/{name}.mat — rode run_examples.sh")
            bad += 1
            continue
        mom = f"{r['mom']:.1e}" if r["mom"] is not None else "n/d"
        note = ""
        if r["r_dy"] > 1e3 * max(r["r_dk"], 1e-16):
            note = f"  (e.e. do Dynare com resíduo {r['r_dy']:.0e})"
        print(f"{name:<14}{r['n']:>5}{r['aux']:>5}  {r['par']:>8.1e} {r['ss']:>8.1e} "
              f"{r['irf']:>8.1e} {mom:>9}  {'OK' if r['ok'] else 'DIVERGE ' + r['worst']}{note}")
        if r["from_dynare"]:
            print(f"{'':<14}parâmetros que o .mod deixa ao MATLAB, tomados do Dynare: "
                  f"{', '.join(r['from_dynare'])}")
        if r["guess_from_dynare"]:
            print(f"{'':<14}ponto de partida do e.e. tomado do Dynare para "
                  f"{r['guess_from_dynare']} variável(is)")
        bad += not r["ok"]
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
