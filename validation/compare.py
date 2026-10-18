"""compare.py: dsgelab against Dynare, on Dynare's own reference models.

Why these models. `dsgelab` was checked against answers known on paper (closed-form RBC, the
Taylor principle, the lognormal risk correction) and against the model of the paper that motivated
it. Neither check is independent of the author. Here the models come from third parties (the
examples of the Dynare tutorial and Johannes Pfeifer's collection, both cited at
dynare.org/resources), the reference solver is Dynare, and nothing in this folder was written with
the answer at hand.

What is compared, in order of strength:
  1. steady state, variable by variable;
  2. impulse responses, every variable and every shock, 40 periods;
  3. theoretical standard deviations, which depend on P, F AND the shock loading Q.

The third covers what the second does not. An IRF to an impulse in the STATE does not pass through
Q. The moments do, and a wrong loading shows up in them.

One translation of timing is needed, and it is documented in models.py. Dynare's `k` is the stock
DECIDED in t, dsgelab's is the stock AVAILABLE in t. The series differ by one period for the
predetermined variables that receive NO shock directly, that is, whose row of Q is zero.

Prerequisites:
    bash fetch.sh          # downloads the .mod files
    bash run_dynare.sh     # runs each one and exports out/<name>.mat
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dsgelab as dsg
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
        print(f"\n{name}: out/{name}.mat missing, run `bash run_dynare.sh`")
        return {"ok": False}

    ss = dsg.steady_state(model, guess, params)
    sol = dsg.solve(model, params, steady=ss)

    print(f"\n{'=' * 88}\n{name}  ({mod_file}) — {model.n_states} states, "
          f"{model.n_jumps} jumps, {model.n_shocks} shocks\n{'=' * 88}")

    # ---------------------------------------------------------------- 1
    dy_ss = dict(zip(ex.endo_names, np.atleast_1d(ex.steady_state).astype(float)))
    common = [v for v in model.names if v in dy_ss]
    d_ss = max(abs(ss[v] - dy_ss[v]) / max(abs(dy_ss[v]), 1e-8) for v in common)
    # When the two disagree, the question is not which number is "the right one" but which of the
    # two POINTS sets the equations to zero. The residual of the system is measured at both.
    zero = np.zeros(model.n_shocks)
    x_dy = np.array([dy_ss[v] for v in model.names])
    r_dk = float(np.max(np.abs(model.residuals(ss.x, ss.x, zero, params))))
    r_dy = float(np.max(np.abs(model.residuals(x_dy, x_dy, zero, params))))
    verdict = "OK" if d_ss < 1e-9 else (
        "OK (Dynare stopped early; see residuals)" if r_dy > 1e3 * max(r_dk, 1e-16)
        else "<-- DIVERGE")
    print(f"  1. steady state ({len(common)} variables)        "
          f"max relative deviation {d_ss:.3e}  {verdict}")
    print(f"       system residual:  at the dsgelab s.s. {r_dk:.2e}   "
          f"at the Dynare s.s. {r_dy:.2e}")
    if d_ss >= 1e-9:
        worst = max(common, key=lambda v: abs(ss[v] - dy_ss[v]) / max(abs(dy_ss[v]), 1e-8))
        print(f"       largest deviation: {worst}  dsgelab {ss[worst]:.10f}  Dynare {dy_ss[worst]:.10f}")
    ss_ok = d_ss < 1e-9 or r_dy > 1e3 * max(r_dk, 1e-16)

    # ---------------------------------------------------------------- 2
    # predetermined variables WITHOUT a direct shock: Dynare's series leads by one period
    shift = {model.states[i] for i in range(model.n_states)
             if np.allclose(sol.Q[i, :], 0.0, atol=1e-12)}
    # Dynare ORTHOGONALISES the shocks for the IRFs. The response to shock i is the response to
    # `chol(Sigma)'(:,i)`, not to an isolated impulse in i. With a diagonal covariance the two
    # coincide. With correlated shocks (example1's `var e, u = phi*0.009^2`) they do not, and the
    # difference is of the order of phi. The same convention is replicated here.
    L = np.linalg.cholesky(np.asarray(cov, float))
    curves = {}
    for m, shock in enumerate(model.shocks):
        dev = sol.irf_impulse(L[:, m], periods=H + 1, percent=False)
        for v in model.names:
            attr = f"{v}_{ex.exo_names[m]}"
            if not hasattr(ex.irfs, attr):
                continue
            dy = np.atleast_1d(getattr(ex.irfs, attr)).astype(float)[:H]
            lvl = np.asarray(dev[v]) * sol.coords.scale[model.position(v)]
            curves[(v, shock)] = (lvl[1:H + 1] if v in shift else lvl[:H], dy)

    # The yardstick of each variable is its LARGEST response across all shocks, not the response
    # to the shock in question. A variable that does not respond to a shock (`ghat` to `eps_z`, by
    # construction) has an exactly zero response in dsgelab and noise of order 1e-19 in Dynare.
    # Dividing by that noise produces a large relative error on a quantity that is zero.
    escala = {}
    for (v, _), (_, dy) in curves.items():
        escala[v] = max(escala.get(v, 0.0), float(np.max(np.abs(dy))))
    pior_irf, pior_nome = 0.0, ""
    for (v, shock), (dk_, dy) in curves.items():
        if escala[v] < 1e-12:
            continue
        d = float(np.max(np.abs(dk_ - dy))) / escala[v]
        if d > pior_irf:
            pior_irf, pior_nome = d, f"{v} <- {shock}"
    print(f"  2. IRFs ({len(model.names)} variables x {model.n_shocks} shocks x {H} periods)"
          f"   max relative deviation {pior_irf:.3e}  ({pior_nome})"
          f"  {'OK' if pior_irf < 1e-6 else '<-- DIVERGE'}")

    # ---------------------------------------------------------------- 3
    d_mom = None
    if hasattr(ex, "variance"):
        V = np.atleast_2d(np.asarray(ex.variance, float))
        dy_sd = dict(zip(ex.endo_names, np.sqrt(np.maximum(np.diag(V), 0.0))))
        mom = sol.moments(cov=np.asarray(cov, float))
        dk_sd = {v: mom["std_by_name"][v] * sol.coords.scale[model.position(v)]
                 for v in model.names}
        cols = [v for v in model.names if v in dy_sd and dy_sd[v] > 1e-9]
        d_mom = max(abs(dk_sd[v] - dy_sd[v]) / dy_sd[v] for v in cols)
        print(f"  3. theoretical standard deviations ({len(cols)} variables)   "
              f"max relative deviation {d_mom:.3e}  {'OK' if d_mom < 1e-6 else '<-- DIVERGE'}")
        print(f"       (they pass through Q, which the IRFs to a state impulse do not exercise)")
        worst = max(cols, key=lambda v: abs(dk_sd[v] - dy_sd[v]) / dy_sd[v])
        print(f"       largest: {worst:<12} dsgelab {dk_sd[worst]:.10f}   Dynare {dy_sd[worst]:.10f}")

    acc = sol.accuracy()
    ordem = ("exact (linear model)" if acc.get("exact")
             else ", ".join(f"{o:.3f}" for o in acc["implied_order"]))
    print(f"  4. Blanchard-Kahn {sol.klein.n_stable}/{sol.klein.n_k};  "
          f"order of accuracy {ordem};  loading residual {sol.klein.shock_residual:.1e}")

    # When Dynare's s.s. leaves a residual of 1e-6, everything derived from it (the linearisation,
    # the IRFs, the moments) inherits that error. Comparing against 1e-6 in that case would demand
    # of the comparison a precision that the reference does not have.
    piso = max(1e-6, 30.0 * r_dy)
    ok = ss_ok and pior_irf < piso and (d_mom is None or d_mom < piso)
    if piso > 1e-6:
        print(f"       (tolerance raised to {piso:.1e}, which is what the precision of Dynare's s.s."
              f" allows)")
    return {"ok": ok, "ss": d_ss, "irf": pior_irf, "mom": d_mom, "piso": piso,
            "r_dy": r_dy, "r_dk": r_dk}


def main() -> int:
    print("=" * 88)
    print("VALIDATION OF dsgelab AGAINST DYNARE — models from dynare.org/resources")
    print("=" * 88)
    out = {n: compare(n) for n in models.ALL}
    print(f"\n{'=' * 88}\nSUMMARY")
    for n, r in out.items():
        if not r.get("ok") and r.get("ss") is None:
            print(f"  {n:<14} (not run)")
            continue
        nota = ""
        if r.get("r_dy", 0) > 1e3 * max(r.get("r_dk", 1e-16), 1e-16):
            nota = f"  (limited by Dynare's s.s., residual {r['r_dy']:.0e})"
        print(f"  {n:<14} s.s. {r['ss']:.1e}   IRFs {r['irf']:.1e}   "
              f"moments {('%.1e' % r['mom']) if r['mom'] is not None else '   n/a'}"
              f"   {'OK' if r['ok'] else 'DIVERGE'}{nota}")
    print("=" * 88)
    return 0 if all(r.get("ok") for r in out.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
