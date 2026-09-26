"""compare.py — o dsgekit contra o Dynare, nos modelos de referência do próprio Dynare.

Por que estes modelos. O `dsgekit` foi verificado contra respostas conhecidas no papel (RBC de
forma fechada, princípio de Taylor, correção de risco lognormal) e contra o modelo do artigo que o
motivou. Nenhum dos dois é independente do autor. Aqui os modelos são de terceiros — os exemplos do
tutorial do Dynare e a coleção de Johannes Pfeifer, ambos citados em dynare.org/resources —, o
solucionador de referência é o Dynare, e nada nesta pasta foi escrito com a resposta em mãos.

O que se compara, em ordem de força:
  1. estado estacionário, variável a variável;
  2. respostas a impulso, todas as variáveis e todos os choques, 40 períodos;
  3. desvios-padrão teóricos, que dependem de P, F E do carregamento de choque Q.

O terceiro é o que o segundo não cobre. Uma IRF por impulso no ESTADO não passa por Q; os momentos
passam, e é neles que um carregamento errado aparece.

Uma tradução de datação é necessária e está documentada em models.py: o `k` do Dynare é o estoque
DECIDIDO em t, o do dsgekit é o DISPONÍVEL em t. As séries diferem por um período — para as
variáveis predeterminadas que NÃO recebem choque diretamente, isto é, cuja linha de Q é nula.

Pré-requisitos:
    bash fetch.sh          # baixa os .mod
    bash run_dynare.sh     # roda cada um e exporta out/<nome>.mat
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dsgekit as dk
import models

HERE = Path(__file__).resolve().parent
H = 40


def _load_mat(name):
    from scipy.io import loadmat
    f = HERE / "out" / f"{name}.mat"
    if not f.exists():
        return None
    ex = loadmat(f, squeeze_me=True, struct_as_record=False)["ex"]
    ex.endo_names = [str(x).strip() for x in np.atleast_1d(ex.endo_names)]
    ex.exo_names = [str(x).strip() for x in np.atleast_1d(ex.exo_names)]
    return ex


def _rel(a, b, floor=1e-8):
    a = np.asarray(a, float); b = np.asarray(b, float)
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), floor)))


def compare(name: str) -> dict:
    build = models.ALL[name]
    model, params, guess, cov, mod_file = build()
    ex = _load_mat(name)
    if ex is None:
        print(f"\n{name}: out/{name}.mat ausente — rode `bash run_dynare.sh`")
        return {"ok": False}

    ss = dk.steady_state(model, guess, params)
    sol = dk.solve(model, params, steady=ss)

    print(f"\n{'=' * 88}\n{name}  ({mod_file}) — {model.n_states} estados, "
          f"{model.n_jumps} saltos, {model.n_shocks} choques\n{'=' * 88}")

    # ---------------------------------------------------------------- 1
    dy_ss = dict(zip(ex.endo_names, np.atleast_1d(ex.steady_state).astype(float)))
    comuns = [v for v in model.names if v in dy_ss]
    d_ss = max(abs(ss[v] - dy_ss[v]) / max(abs(dy_ss[v]), 1e-8) for v in comuns)
    # Quando os dois discordam, a pergunta não é qual número é "o certo" mas qual dos dois
    # PONTOS anula as equações. Mede-se o resíduo do sistema em ambos.
    zero = np.zeros(model.n_shocks)
    x_dy = np.array([dy_ss[v] for v in model.names])
    r_dk = float(np.max(np.abs(model.residuals(ss.x, ss.x, zero, params))))
    r_dy = float(np.max(np.abs(model.residuals(x_dy, x_dy, zero, params))))
    veredito = "OK" if d_ss < 1e-9 else (
        "OK (o Dynare parou antes; ver resíduos)" if r_dy > 1e3 * max(r_dk, 1e-16)
        else "<-- DIVERGE")
    print(f"  1. estado estacionário ({len(comuns)} variáveis)        "
          f"desvio relativo máx {d_ss:.3e}  {veredito}")
    print(f"       resíduo do sistema:  no e.e. do dsgekit {r_dk:.2e}   "
          f"no e.e. do Dynare {r_dy:.2e}")
    if d_ss >= 1e-9:
        pior = max(comuns, key=lambda v: abs(ss[v] - dy_ss[v]) / max(abs(dy_ss[v]), 1e-8))
        print(f"       maior desvio: {pior}  dsgekit {ss[pior]:.10f}  Dynare {dy_ss[pior]:.10f}")
    ss_ok = d_ss < 1e-9 or r_dy > 1e3 * max(r_dk, 1e-16)

    # ---------------------------------------------------------------- 2
    # variáveis predeterminadas SEM choque direto: a série do Dynare adianta-se um período
    shift = {model.states[i] for i in range(model.n_states)
             if np.allclose(sol.Q[i, :], 0.0, atol=1e-12)}
    # O Dynare ORTOGONALIZA os choques para as IRFs: a resposta ao choque i é a resposta a
    # `chol(Sigma)'(:,i)`, e não a um impulso isolado em i. Com covariância diagonal as duas
    # coincidem; com choques correlacionados — o `var e, u = phi*0.009^2` do example1 — não, e a
    # diferença é da ordem de phi. Replica-se aqui a mesma convenção.
    L = np.linalg.cholesky(np.asarray(cov, float))
    curvas = {}
    for m, shock in enumerate(model.shocks):
        dev = sol.irf_impulse(L[:, m], periods=H + 1, percent=False)
        for v in model.names:
            attr = f"{v}_{ex.exo_names[m]}"
            if not hasattr(ex.irfs, attr):
                continue
            dy = np.atleast_1d(getattr(ex.irfs, attr)).astype(float)[:H]
            lvl = np.asarray(dev[v]) * sol.coords.scale[model.position(v)]
            curvas[(v, shock)] = (lvl[1:H + 1] if v in shift else lvl[:H], dy)

    # A régua de cada variável é a sua MAIOR resposta entre todos os choques, e não a resposta ao
    # choque em questão. Uma variável que não responde a um choque — `ghat` a `eps_z`, por
    # construção — tem resposta exatamente nula no dsgekit e ruído de ordem 1e-19 no Dynare;
    # dividir por esse ruído produz um erro relativo enorme sobre uma quantidade que é zero.
    escala = {}
    for (v, _), (_, dy) in curvas.items():
        escala[v] = max(escala.get(v, 0.0), float(np.max(np.abs(dy))))
    pior_irf, pior_nome = 0.0, ""
    for (v, shock), (dk_, dy) in curvas.items():
        if escala[v] < 1e-12:
            continue
        d = float(np.max(np.abs(dk_ - dy))) / escala[v]
        if d > pior_irf:
            pior_irf, pior_nome = d, f"{v} <- {shock}"
    print(f"  2. IRFs ({len(model.names)} variáveis x {model.n_shocks} choques x {H} períodos)"
          f"   desvio relativo máx {pior_irf:.3e}  ({pior_nome})"
          f"  {'OK' if pior_irf < 1e-6 else '<-- DIVERGE'}")

    # ---------------------------------------------------------------- 3
    d_mom = None
    if hasattr(ex, "variance"):
        V = np.atleast_2d(np.asarray(ex.variance, float))
        dy_sd = dict(zip(ex.endo_names, np.sqrt(np.maximum(np.diag(V), 0.0))))
        mom = sol.moments(cov=np.asarray(cov, float))
        dk_sd = {v: mom["std_by_name"][v] * sol.coords.scale[model.position(v)]
                 for v in model.names}
        alvo = [v for v in model.names if v in dy_sd and dy_sd[v] > 1e-9]
        d_mom = max(abs(dk_sd[v] - dy_sd[v]) / dy_sd[v] for v in alvo)
        print(f"  3. desvios-padrão teóricos ({len(alvo)} variáveis)   "
              f"desvio relativo máx {d_mom:.3e}  {'OK' if d_mom < 1e-6 else '<-- DIVERGE'}")
        print(f"       (passam por Q, que as IRFs por impulso no estado não exercitam)")
        pior = max(alvo, key=lambda v: abs(dk_sd[v] - dy_sd[v]) / dy_sd[v])
        print(f"       maior: {pior:<12} dsgekit {dk_sd[pior]:.10f}   Dynare {dy_sd[pior]:.10f}")

    acc = sol.accuracy()
    ordem = ("exata (modelo linear)" if acc.get("exact")
             else ", ".join(f"{o:.3f}" for o in acc["implied_order"]))
    print(f"  4. Blanchard-Kahn {sol.klein.n_stable}/{sol.klein.n_k};  "
          f"ordem de acurácia {ordem};  resíduo do carregamento {sol.klein.shock_residual:.1e}")

    # Quando o e.e. do Dynare deixa um resíduo de 1e-6, tudo o que dele decorre — a linearização,
    # as IRFs, os momentos — herda esse erro. Comparar contra 1e-6 nesse caso seria exigir da
    # comparação uma precisão que a referência não tem.
    piso = max(1e-6, 30.0 * r_dy)
    ok = ss_ok and pior_irf < piso and (d_mom is None or d_mom < piso)
    if piso > 1e-6:
        print(f"       (tolerância elevada a {piso:.1e}: é o que a precisão do e.e. do Dynare"
              f" permite)")
    return {"ok": ok, "ss": d_ss, "irf": pior_irf, "mom": d_mom, "piso": piso,
            "r_dy": r_dy, "r_dk": r_dk}


def main() -> int:
    print("=" * 88)
    print("VALIDAÇÃO DO dsgekit CONTRA O DYNARE — modelos de dynare.org/resources")
    print("=" * 88)
    out = {n: compare(n) for n in models.ALL}
    print(f"\n{'=' * 88}\nRESUMO")
    for n, r in out.items():
        if not r.get("ok") and r.get("ss") is None:
            print(f"  {n:<14} (não executado)")
            continue
        nota = ""
        if r.get("r_dy", 0) > 1e3 * max(r.get("r_dk", 1e-16), 1e-16):
            nota = f"  (limitado pelo e.e. do Dynare, resíduo {r['r_dy']:.0e})"
        print(f"  {n:<14} e.e. {r['ss']:.1e}   IRFs {r['irf']:.1e}   "
              f"momentos {('%.1e' % r['mom']) if r['mom'] is not None else '   n/d'}"
              f"   {'OK' if r['ok'] else 'DIVERGE'}{nota}")
    print("=" * 88)
    return 0 if all(r.get("ok") for r in out.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
