"""roundtrip.py — exportar do dsgekit para o Dynare, rodar lá, e comparar de volta.

O teste de ida e volta é mais forte do que o `compare.py`. Ali eu escrevi o porte de um `.mod` para
o dsgekit à mão, e um erro meu de datação apareceu — o que o teste pegou. Aqui é o EXPORTADOR que
escreve o `.mod`, e o que se verifica é se a tradução automática preserva o modelo: mesmo estado
estacionário, mesmas respostas a impulso, mesmos momentos.

Usa os mesmos três modelos de `models.py`, que já vieram do Dynare. O caminho completo é portanto
Dynare -> dsgekit (à mão, verificado por `compare.py`) -> Dynare (automático, verificado aqui).

    bash fetch.sh && bash run_dynare.sh      # para o compare.py
    python roundtrip.py                      # este
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dsgekit as dk
import models
from dsgekit.export import to_dynare

HERE = Path(__file__).resolve().parent
OUT = HERE / "out_roundtrip"
H = 40


def _dynare_dir() -> str:
    d = os.environ.get("DYNARE_MATLAB")
    if d:
        return d
    for c in ("/opt/homebrew/opt/dynare/lib/dynare/matlab",
              "/opt/homebrew/lib/dynare/matlab",
              "/usr/local/opt/dynare/lib/dynare/matlab",
              "/usr/lib/dynare/matlab"):
        if Path(c).is_dir():
            return c
    raise RuntimeError("diretório matlab/ do Dynare não encontrado; defina DYNARE_MATLAB")


def roda_dynare(mod: Path) -> Path:
    """Roda um .mod no Octave e devolve o .mat com oo_ exportado."""
    octave = os.environ.get("OCTAVE", "/opt/homebrew/bin/octave")
    base = mod.stem
    mat = OUT / f"{base}.mat"
    script = f"""
      pkg load statistics datatypes;
      addpath '{_dynare_dir()}';
      dynare {mod.name} noclearall nograph;
      options_.order = 1; options_.irf = {H}; options_.periods = 0;
      options_.hp_filter = 0; options_.nograph = 1; options_.noprint = 1;
      [info, oo_, options_, M_] = stoch_simul(M_, options_, oo_, M_.endo_names);
      ex = struct();
      ex.steady_state = oo_.steady_state; ex.endo_names = M_.endo_names;
      ex.exo_names = M_.exo_names; ex.irfs = oo_.irfs; ex.info = info;
      if isfield(oo_, 'var'), ex.variance = oo_.var; end
      save('-mat7-binary', '{mat.name}', 'ex');
    """
    r = subprocess.run([octave, "--no-gui", "--eval", script], cwd=OUT,
                       capture_output=True, text=True)
    if not mat.exists():
        raise RuntimeError(f"Dynare falhou em {mod.name}:\n{r.stdout[-1800:]}\n{r.stderr[-800:]}")
    return mat


def confronta(nome: str) -> dict:
    model, params, guess, cov, _ = models.ALL[nome]()
    ss = dk.steady_state(model, guess, params)
    sol = dk.solve(model, params, steady=ss)

    mod = OUT / f"{nome}_rt.mod"
    to_dynare(model, params, steady=ss, cov=cov, path=mod, irf=H)
    mat = roda_dynare(mod)

    from scipy.io import loadmat
    ex = loadmat(mat, squeeze_me=True, struct_as_record=False)["ex"]
    nomes = [str(x).strip() for x in np.atleast_1d(ex.endo_names)]
    exo = [str(x).strip() for x in np.atleast_1d(ex.exo_names)]
    dy_ss = dict(zip(nomes, np.atleast_1d(ex.steady_state).astype(float)))

    print(f"\n{'=' * 84}\n{nome} — exportado pelo dsgekit e resolvido pelo Dynare\n{'=' * 84}")

    comuns = [v for v in model.names if v in dy_ss]
    d_ss = max(abs(ss[v] - dy_ss[v]) / max(abs(dy_ss[v]), 1e-8) for v in comuns)
    zero = np.zeros(model.n_shocks)
    x_dy = np.array([dy_ss[v] for v in model.names])
    r_dy = float(np.max(np.abs(model.residuals(x_dy, x_dy, zero, params))))
    print(f"  estado estacionário ({len(comuns)} variáveis)     desvio relativo máx {d_ss:.3e}")
    print(f"     resíduo do sistema no e.e. que o Dynare devolveu: {r_dy:.2e}")

    # `predetermined_variables` redata as EQUAÇÕES, não a série reportada: o Dynare continua
    # publicando o `k` DECIDIDO em t, que é o nosso k de t+1. O deslocamento é o mesmo do
    # compare.py, e vale exatamente para os estados cuja linha de Q é nula — os que não recebem
    # inovação diretamente. Medido: com deslocamento 1 o desvio cai de 1,4e-02 para 3,4e-15.
    shift = {model.states[i] for i in range(model.n_states)
             if np.allclose(sol.Q[i, :], 0.0, atol=1e-12)}
    L = np.linalg.cholesky(np.asarray(cov, float))
    curvas, escala = {}, {}
    for m, choque in enumerate(model.shocks):
        dev = sol.irf_impulse(L[:, m], periods=H + 1, percent=False)
        for v in model.names:
            attr = f"{v}_{exo[m]}"
            if not hasattr(ex.irfs, attr):
                continue
            dy = np.atleast_1d(getattr(ex.irfs, attr)).astype(float)[:H]
            bruto = np.asarray(dev[v]) * sol.coords.scale[model.position(v)]
            lvl = bruto[1:H + 1] if v in shift else bruto[:H]
            curvas[(v, choque)] = (lvl, dy)
            escala[v] = max(escala.get(v, 0.0), float(np.max(np.abs(dy))))
    pior, onde = 0.0, ""
    for (v, choque), (a, b) in curvas.items():
        if escala[v] < 1e-12:
            continue
        d = float(np.max(np.abs(a - b))) / escala[v]
        if d > pior:
            pior, onde = d, f"{v} <- {choque}"
    print(f"  IRFs ({len(curvas)} séries x {H} períodos)          desvio relativo máx "
          f"{pior:.3e}  ({onde})")

    d_mom = None
    if hasattr(ex, "variance"):
        V = np.atleast_2d(np.asarray(ex.variance, float))
        dy_sd = dict(zip(nomes, np.sqrt(np.maximum(np.diag(V), 0.0))))
        mom = sol.moments(cov=np.asarray(cov, float))
        alvo = [v for v in model.names if v in dy_sd and dy_sd[v] > 1e-9]
        d_mom = max(abs(mom["std_by_name"][v] * sol.coords.scale[model.position(v)] - dy_sd[v])
                    / dy_sd[v] for v in alvo)
        print(f"  desvios-padrão teóricos ({len(alvo)} variáveis)  desvio relativo máx {d_mom:.3e}")

    piso = max(1e-6, 30.0 * r_dy)
    ok = pior < piso and (d_mom is None or d_mom < piso) and (d_ss < 1e-9 or r_dy > 1e-9)
    if piso > 1e-6:
        print(f"     (tolerância elevada a {piso:.1e}: é o que a precisão do e.e. do Dynare permite)")
    return {"ok": ok, "ss": d_ss, "irf": pior, "mom": d_mom, "r_dy": r_dy}


def main() -> int:
    OUT.mkdir(exist_ok=True)
    print("=" * 84)
    print("IDA E VOLTA — o dsgekit escreve o .mod, o Dynare resolve, comparamos de volta")
    print("=" * 84)
    res = {n: confronta(n) for n in models.ALL}
    print(f"\n{'=' * 84}\nRESUMO")
    for n, r in res.items():
        print(f"  {n:<14} e.e. {r['ss']:.1e}   IRFs {r['irf']:.1e}   "
              f"momentos {('%.1e' % r['mom']) if r['mom'] is not None else '   n/d'}"
              f"   {'OK' if r['ok'] else 'DIVERGE'}")
    print("=" * 84)
    return 0 if all(r["ok"] for r in res.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
