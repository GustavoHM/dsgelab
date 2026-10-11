"""compare_policy.py — regras simples ótimas, discrição e bem-estar, contra o Dynare.

Todos os modelos são lidos por `dsgelab.from_mod` a partir dos arquivos do próprio Dynare, menos o
RBC com imposto, escrito nesta pasta. O que se compara:

  nk_osr_planner.mod   (escrito aqui)  osr com planner_objective: a perda é menos o bem-estar
                       incondicional de 2ª ordem. Mesma conferência: a perda no ótimo do Dynare.
  osr_example.mod      (tests/optimal_policy/OSR)  a função de perda do dsgelab no ótimo do Dynare,
                       contra o valor que o Dynare reporta; e o ótimo do dsgelab, que tem de ter
                       perda menor ou igual. Com Octave e Dynare à mão, a perda do PRÓPRIO Dynare
                       no ponto do dsgelab também é calculada.
  dennis_1.mod         (tests/discretionary_policy)  a regra de decisão sob discrição, como IRFs a
                       cada choque, e o bem-estar incondicional e condicional.
  nk_ramsey_model.mod  (tests/optimal_policy)  Ramsey lido do .mod, sem tradução à mão: estado
                       estacionário com os multiplicadores, IRFs de 1ª ordem e bem-estar de 2ª.
  rbc_labortax.mod     (escrito aqui)  bem-estar de 2ª ordem com um estoque de capital.

Pré-requisitos:
    bash fetch.sh
    bash run_policy.sh      # exporta out_policy/<nome>.mat
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

import dsgelab as dk
from dsgelab.dynare import from_mod
from dsgelab.model import set_params
from dsgelab.optimal import discretionary_policy, osr, ramsey_model, ramsey_steady_state
from dsgelab.welfare import welfare

HERE = Path(__file__).resolve().parent
H = 40


def _load(name):
    from scipy.io import loadmat
    f = HERE / "out_policy" / f"{name}.mat"
    if not f.exists():
        return None
    ex = loadmat(f, squeeze_me=True, struct_as_record=False)["ex"]
    for k in ("endo_names", "exo_names", "param_names"):
        setattr(ex, k, [str(x).strip() for x in np.atleast_1d(getattr(ex, k))])
    return ex


def _dynare_irfs(ex, impulse):
    """First-order IRFs from Dynare's decision rule ``ghx``, ``ghu`` (DR order)."""
    n = len(ex.endo_names)
    ov = np.atleast_1d(ex.order_var).astype(int) - 1
    if hasattr(ex, "state_var"):
        cols = np.atleast_1d(ex.state_var).astype(int) - 1
    else:
        lli = np.atleast_2d(ex.lead_lag_incidence).astype(int)
        cols = ov[lli[0, ov] > 0]
    ghx = np.asarray(ex.ghx, float).reshape(n, -1)
    ghu = np.asarray(ex.ghu, float).reshape(n, -1)
    out, prev = np.zeros((H, n)), np.zeros(n)
    for t in range(H):
        u = impulse if t == 0 else np.zeros_like(impulse)
        y = np.zeros(n)
        y[ov] = ghx @ prev[cols] + ghu @ u
        out[t], prev = y, y
    return {v: out[:, i] for i, v in enumerate(ex.endo_names)}


def _welfare_of(ex):
    w = ex.welfare
    c = w.conditional
    if hasattr(c, "_fieldnames"):
        return float(w.unconditional), float(c.steady_initial_multiplier), \
            float(c.zero_initial_multiplier)
    return float(w.unconditional), float(c), None


def _rel(a, b):
    return abs(a - b) / max(abs(b), 1e-12)


# ---------------------------------------------------------------------------- OSR
def _dynare_loss_at(params: dict) -> float | None:
    """Dynare's own loss at ``params``: osr_example.mod with those starting values,
    evaluated by +osr/objective. None when Octave or Dynare are not at hand."""
    octave = shutil.which("octave") or "/opt/homebrew/bin/octave"
    matlab = Path("/opt/homebrew/opt/dynare/lib/dynare/matlab")
    if not Path(octave).exists() or not matlab.exists():
        return None
    src = (HERE / "mods" / "ramsey" / "osr_example.mod").read_text()
    head = src.split("osr(")[0]
    for k, v in params.items():
        head = head.replace(f"\n{k} = ", f"\n{k} = {v!r}; % ", 1) if f"\n{k} = " in head \
            else head.replace(f"\n{k}=", f"\n{k} = {v!r}; % ", 1)
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "osr_at.mod").write_text(head + "\nstoch_simul(order=1, irf=0, noprint);\n")
        code = (f"pkg load statistics datatypes; addpath '{matlab}'; "
                "dynare osr_at.mod noclearall nograph; "
                "i_var = [find(strcmp(M_.endo_names,'y')); find(strcmp(M_.endo_names,'inflation'))]; "
                "ip = [find(strcmp(M_.param_names,'gammax0')) find(strcmp(M_.param_names,'gammac0')) "
                "find(strcmp(M_.param_names,'gamma_y_')) find(strcmp(M_.param_names,'gamma_inf_'))]; "
                "W = M_.osr.variable_weights; options_.qz_criterium = 1+1e-6; "
                "[iv, ~, ~] = unique(i_var); "
                "l = osr.objective(M_.params(ip), M_, oo_, options_, ip, oo_.dr.inv_order_var(iv), W(iv,iv)); "
                "printf('LOSS %.17g\\n', l);")
        out = subprocess.run([octave, "--no-gui", "--eval", code], cwd=d, capture_output=True,
                             text=True, timeout=600)
        for line in out.stdout.splitlines():
            if line.startswith("LOSS"):
                return float(line.split()[1])
    return None


def check_osr(with_dynare: bool) -> bool:
    ex = _load("osr_example")
    if ex is None:
        print("osr_example: sem out_policy/osr_example.mat — rode run_policy.sh")
        return False
    mf = from_mod(HERE / "mods" / "ramsey" / "osr_example.mod")
    dy_par = {k: float(getattr(ex.osr.optim_params, k)) for k in ex.osr.optim_params._fieldnames}
    dy_loss = float(ex.osr.objective_function)
    sf = lambda p: mf.steady_state(params=p)       # noqa: E731

    # 1. the loss function itself, at Dynare's point
    p = set_params(mf.params, dy_par)
    sol = dk.solve(mf.model, p, steady=sf(p))
    V = sol.covariance(mf.cov) * np.outer(sol.coords.scale, sol.coords.scale)
    from dsgelab.optimal import _weight_matrix
    ours_at_dy = float(np.sum(_weight_matrix(mf.model.names, mf.optim_weights) * V))
    d_fun = _rel(ours_at_dy, dy_loss)

    # 2. the optimum
    r = osr(mf.model, mf.params, rule_params=mf.osr_params, weights=mf.optim_weights,
            cov=mf.cov, steady_fn=sf)
    better = r.loss <= dy_loss * (1 + 1e-12)
    print("OSR — osr_example.mod")
    print(f"  perda do dsgelab no ótimo do Dynare   {ours_at_dy:.16f}")
    print(f"  perda reportada pelo Dynare           {dy_loss:.16f}   desvio relativo {d_fun:.1e}")
    print(f"  ótimo do dsgelab                      {r.loss:.16f}   "
          f"{'menor ou igual' if better else 'MAIOR'} que o do Dynare")
    for k in mf.osr_params:
        print(f"     {k:<11} dsgelab {r.params[k]:>12.6f}   Dynare {dy_par[k]:>12.6f}")
    ok = d_fun < 1e-10 and better
    if with_dynare:
        dl = _dynare_loss_at(r.params)
        if dl is None:
            print("  (Octave/Dynare indisponíveis: a perda do Dynare no ponto do dsgelab não foi calculada)")
        else:
            d2 = _rel(r.loss, dl)
            print(f"  perda do PRÓPRIO Dynare no ótimo do dsgelab {dl:.16f}   desvio {d2:.1e}")
            ok = ok and d2 < 1e-10
    return ok


def check_osr_welfare() -> bool:
    ex = _load("nk_osr_planner")
    if ex is None:
        print("nk_osr_planner: sem out_policy/nk_osr_planner.mat — rode run_policy.sh")
        return False
    mf = from_mod(HERE / "ramsey" / "nk_osr_planner.mod")
    dy_par = {k: float(getattr(ex.osr.optim_params, k)) for k in ex.osr.optim_params._fieldnames}
    dy_loss = float(ex.osr.objective_function)
    sf = lambda p: mf.steady_state(params=p)       # noqa: E731
    U, beta = mf.planner["objective"], mf.planner["discount"]
    p = set_params(mf.params, dy_par)
    ours_at_dy = -welfare(mf.model, p, steady=sf(p), cov=mf.cov, objective=U,
                          discount=beta).unconditional
    r = osr(mf.model, mf.params, rule_params=mf.osr_params, objective=U, discount=beta,
            cov=mf.cov, steady_fn=sf)
    d_fun = _rel(ours_at_dy, dy_loss)
    better = r.loss <= dy_loss * (1 + 1e-12)
    print("\nOSR sobre o bem-estar — nk_osr_planner.mod (planner_objective, ordem 2)")
    print(f"  perda do dsgelab no ótimo do Dynare   {ours_at_dy:.16f}")
    print(f"  perda reportada pelo Dynare           {dy_loss:.16f}   desvio relativo {d_fun:.1e}")
    print(f"  ótimo do dsgelab                      {r.loss:.16f}   "
          f"{'menor ou igual' if better else 'MAIOR'} que o do Dynare")
    for k in mf.osr_params:
        print(f"     {k:<11} dsgelab {r.params[k]:>12.6f}   Dynare {dy_par[k]:>12.6f}")
    return d_fun < 1e-10 and better


# ------------------------------------------------------------------- discretion
def check_discretion() -> bool:
    ex = _load("dennis_1")
    if ex is None:
        print("dennis_1: sem out_policy/dennis_1.mat — rode run_policy.sh")
        return False
    mf = from_mod(HERE / "mods" / "ramsey" / "dennis_1.mod")
    ds = discretionary_policy(mf, params=mf.params, cov=mf.cov)
    L = np.linalg.cholesky(np.atleast_2d(ex.sigma_e))
    order = [ds.shocks.index(e) for e in ex.exo_names]
    ruler, worst = {}, {}
    for j, e in enumerate(ex.exo_names):
        dy = _dynare_irfs(ex, L[:, j])
        imp = np.zeros(len(ds.shocks))
        imp[order] = L[:, j]
        mine = ds.irf_impulse(imp, periods=H)
        for v in ex.endo_names:
            ruler[v] = max(ruler.get(v, 0.0), float(np.max(np.abs(dy[v]))))
            worst[v] = max(worst.get(v, 0.0), float(np.max(np.abs(mine[v] - dy[v]))))
    d_irf = max(worst[v] / ruler[v] for v in ruler if ruler[v] > 1e-12)
    # Dynare's planner_objective is a LOSS here; ours is its negative.
    un_dy, cond_dy, _ = _welfare_of(ex)
    u0 = np.zeros(len(ds.shocks))
    u0[order] = np.atleast_1d(ex.exo_simul_first)
    w = ds.welfare()
    w_draw = ds.welfare(shock=u0)
    d_un = _rel(-w["unconditional"], un_dy)
    d_cond = _rel(-w_draw["conditional"], cond_dy)
    print("\nDiscrição — dennis_1.mod")
    print(f"  IRFs ({len(ex.endo_names)} variáveis x {len(ex.exo_names)} choques x {H})   "
          f"desvio relativo máx {d_irf:.1e}")
    print(f"  bem-estar incondicional  dsgelab {-w['unconditional']:.10f}   Dynare {un_dy:.10f}   "
          f"{d_un:.1e}")
    print(f"  bem-estar condicional    dsgelab {-w_draw['conditional']:.10f}   Dynare {cond_dy:.10f}   "
          f"{d_cond:.1e}")
    print(f"     (com a primeira inovação da simulação do Dynare, que é o que ele usa; "
          f"sem choque: {-w['conditional']:.10f})")
    print(f"  iterações de Dennis: {ds.iterations}")
    # Dynare stops the Dennis iteration at sqrt(eps); that is the floor here.
    return d_irf < 1e-6 and d_un < 1e-8 and d_cond < 1e-8


# --------------------------------------------------------------- Ramsey + welfare
def check_ramsey_mod() -> bool:
    ex = _load("nk_ramsey_model")
    if ex is None:
        print("nk_ramsey_model: sem out_policy — rode run_policy.sh")
        return False
    mf = from_mod(HERE / "mods" / "ramsey" / "nk_ramsey_model.mod")
    m = ramsey_model(mf)
    ss = ramsey_steady_state(m, mf.full_guess(), mf.params)
    ys = dict(zip(ex.endo_names, np.atleast_1d(ex.ys).astype(float)))
    names = [v for v in ex.endo_names if v in m.names]
    d_ss = max(abs(ss.values[v] - ys[v]) for v in names)
    sol = dk.solve(m, mf.params, steady=ss)
    sd = np.sqrt(np.diag(np.atleast_2d(ex.sigma_e)))
    d_irf = 0.0
    for j, e in enumerate(ex.exo_names):
        imp = np.zeros(len(ex.exo_names))
        imp[j] = sd[j]
        dy = _dynare_irfs(ex, imp)
        mine = sol.irf(e, size=sd[j], periods=H, levels=True)
        for v in names:
            d_irf = max(d_irf, float(np.max(np.abs(mine[v] - ss.values[v] - dy[v]))))
    w = welfare(m, mf.params, steady=ss, cov=mf.cov)
    un, c_ss, c_0 = _welfare_of(ex)
    d_w = max(_rel(w.unconditional, un), _rel(w.conditional, c_ss),
              _rel(w.conditional_zero_multipliers, c_0))
    print("\nRamsey lido do .mod — nk_ramsey_model.mod")
    print(f"  {len(names)} séries comparadas, multiplicadores incluídos (mesma numeração do Dynare)")
    print(f"  estado estacionário  {d_ss:.1e}     IRFs de 1ª ordem  {d_irf:.1e}")
    print(f"  bem-estar  incondicional {w.unconditional:.12f}  Dynare {un:.12f}")
    print(f"             condicional   {w.conditional:.12f}  Dynare {c_ss:.12f}")
    print(f"             multiplicadores zerados {w.conditional_zero_multipliers:.12f}  "
          f"Dynare {c_0:.12f}   desvio relativo máx {d_w:.1e}")
    return d_ss < 1e-9 and d_irf < 1e-10 and d_w < 1e-10


def check_welfare_capital() -> bool:
    ex = _load("rbc_labortax_o2")
    if ex is None:
        print("rbc_labortax_o2: sem out_policy — rode run_policy.sh")
        return False
    sys.path.insert(0, str(HERE))
    import compare_ramsey
    m, p, g, _ = compare_ramsey.rbc_labortax()
    ss = ramsey_steady_state(m, g, p)
    w = welfare(m, p, steady=ss, cov={"e": float(np.sqrt(np.atleast_2d(ex.sigma_e)[0, 0]))})
    un, c_ss, c_0 = _welfare_of(ex)
    d_w = max(_rel(w.unconditional, un), _rel(w.conditional, c_ss),
              _rel(w.conditional_zero_multipliers, c_0))
    print("\nBem-estar com capital — rbc_labortax.mod, ordem 2")
    print(f"  incondicional {w.unconditional:.12f}  Dynare {un:.12f}")
    print(f"  condicional   {w.conditional:.12f}  Dynare {c_ss:.12f}   desvio relativo máx {d_w:.1e}")
    return d_w < 1e-10


def main() -> int:
    with_dynare = "--sem-dynare" not in sys.argv
    res = {"OSR": check_osr(with_dynare), "OSR sobre o bem-estar": check_osr_welfare(),
           "discrição": check_discretion(),
           "Ramsey do .mod": check_ramsey_mod(), "bem-estar com capital": check_welfare_capital()}
    print("\n" + "  ".join(f"{k}: {'OK' if v else 'FALHOU'}" for k, v in res.items()))
    return 0 if all(res.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
