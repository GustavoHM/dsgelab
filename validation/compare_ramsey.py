"""compare_ramsey.py: dsgelab's Ramsey planner against Dynare's ramsey_model.

Two models:

  nk_linear         the textbook New Keynesian model (Galí, ch. 5), written in this folder. It has
                    an analytical answer, which tests/test_optimal.py already checks. Here Dynare
                    is the third witness.
  nk_ramsey_model   from Dynare's own test suite (tests/optimal_policy). It is nonlinear, with a
                    Rotemberg Phillips curve, an Euler equation in c(+1) and pai(+1), and
                    productivity in exp(a). The derivatives of the constraints with respect to the
                    t+1 variables depend on those of t, so the FOC needs lagged copies. This is the
                    case that the linear model does not exercise.

What is compared: the steady state and the impulse responses, 40 periods, in levels, of every
private variable AND of the multipliers. Dynare numbers one multiplier per equation, including
those of the exogenous processes, and dsgelab only those of the constraints. dsgelab's MULT_j
belongs to the j-th constraint, and it is compared with the multiplier that Dynare gives to the
SAME equation. Both write the Lagrangian with the same sign, and the values agree without a sign
flip.

The reference steady state is oo_.dr.ys, the point at which Dynare linearised.
oo_.steady_state cannot be used, because in nk_ramsey_model it holds the initval
(c = n = 0.9671684882), which misses the exact root c = 0.875^(1/4) in the seventh decimal place.

Dynare's .mod does not set the order, and Dynare's default is 2, so its IRFs would be generalised
second-order ones. run_ramsey.sh sets order=1 before rerunning stoch_simul.

  rbc_labortax      an RBC with a labour tax as the instrument, written in this folder. The
                    competitive equilibrium with tau = 0 is efficient, and the planner keeps tau
                    at zero. The case exercises the planner's condition for a STOCK, k(+1), chosen
                    in t and holding in expectation.

Timing. Dynare's k in predetermined_variables enters the equations as dsgelab's does, but the
series Dynare reports is that of the redated variable, k(+1). As in compare.py, the predetermined
variables whose row of Q is zero are compared with a lag of one period.

Prerequisites:
    bash fetch.sh          # downloads mods/ramsey/nk_ramsey_model.mod
    bash run_ramsey.sh     # runs Dynare and exports out/<name>.mat
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import dsgelab as dsg
from dsgelab.optimal import ramsey_model, ramsey_steady_state

HERE = Path(__file__).resolve().parent
H = 40
TOL = 1e-8


def nk_linear():
    model = ramsey_model(
        ["x - x(+1) + (1/sigma)*(i - pi(+1)) = 0",
         "pi = beta*pi(+1) + kappa*x + u",
         "u(+1) = rho*u + e"],
        states=["u"], jumps=["x", "pi", "i"], shocks=["e"],
        parameters=["beta", "sigma", "kappa", "lam", "rho"], instruments=["i"],
        objective="-(pi**2 + lam*x**2)/2", discount=0.99, name="nk_linear")
    params = dict(beta=0.99, sigma=1.0, kappa=0.1, lam=0.25, rho=0.5)
    guess = dict(u=0.0, x=0.0, pi=0.0, i=0.0)
    return model, params, guess, {"e": "e"}


def nk_ramsey_model():
    # Literal translation of tests/optimal_policy/nk_ramsey_model.mod. A(-1) becomes the state
    # `a` with a(+1) = rho*a + u. The rest is written as in the .mod, with ^ replaced by **.
    model = ramsey_model(
        ["a(+1) = rho*a + u",
         "1/c = beta*(1/c(+1))*(r/pai(+1))",
         "omega*pai*(pai-1) = beta*omega*(c/c(+1))*pai(+1)*(pai(+1)-1)"
         " + epsilon*exp(a)*n*(c/exp(a)*phi*n**gamma - (epsilon-1)/epsilon)",
         "exp(a)*n = c + (omega/2)*(pai-1)**2"],
        states=["a"], jumps=["pai", "c", "n", "r"], shocks=["u"],
        parameters=["beta", "rho", "epsilon", "omega", "phi", "gamma"], instruments=["r"],
        objective="log(c) - phi*n**(1+gamma)/(1+gamma)", discount=0.99,
        name="nk_ramsey_model")
    params = dict(beta=0.99, gamma=3.0, omega=17.0, epsilon=8.0, phi=1.0, rho=0.95)
    guess = dict(a=0.0, pai=1.0, r=1 / 0.99, c=0.9671684882, n=0.9671684882)
    return model, params, guess, {"u": "u"}


def rbc_labortax():
    # dsgelab's k is the stock available in t. The .mod declares predetermined_variables k,
    # which is the same timing in the equations. a(-1) becomes the state `a`.
    model = ramsey_model(
        ["1/c = beta*(1/c(+1))*(alpha*exp(a(+1))*k(+1)**(alpha-1)*n(+1)**(1-alpha) + 1 - delta)",
         "chi*n**phi*c = (1-tau)*(1-alpha)*exp(a)*k**alpha*n**(-alpha)",
         "c + k(+1) = exp(a)*k**alpha*n**(1-alpha) + (1-delta)*k",
         "a(+1) = rho*a + e"],
        states=["k", "a"], jumps=["c", "n", "tau"], shocks=["e"],
        parameters=["beta", "alpha", "delta", "rho", "chi", "phi"], instruments=["tau"],
        objective="log(c) - chi*n**(1+phi)/(1+phi)", discount=0.99, name="rbc_labortax")
    params = dict(beta=0.99, alpha=0.33, delta=0.025, rho=0.95, chi=8.0, phi=1.0)
    guess = dict(n=0.330881054416, k=9.379954789951, c=0.763215941854, tau=0.0, a=0.0)
    return model, params, guess, {"e": "e"}


ALL = {"nk_linear": nk_linear, "rbc_labortax": rbc_labortax,
       "nk_ramsey_model": nk_ramsey_model}


def _load_mat(name):
    from scipy.io import loadmat
    f = HERE / "out" / f"{name}.mat"
    if not f.exists():
        return None
    ex = loadmat(f, squeeze_me=True, struct_as_record=False)["ex"]
    ex.endo_names = [str(x).strip() for x in np.atleast_1d(ex.endo_names)]
    ex.exo_names = [str(x).strip() for x in np.atleast_1d(ex.exo_names)]
    return ex


def compare(name: str) -> dict:
    model, params, guess, shock_map = ALL[name]()
    ex = _load_mat(name)
    if ex is None:
        return dict(name=name, missing=True)
    private = model.ramsey["private"]

    ss = ramsey_steady_state(model, guess, params)
    # name in dsgelab -> name in Dynare: the private variables and, for MULT_j, the same equation's
    names = {v: v for v in private}
    for j, eq in enumerate(model.ramsey["constraint_index"]):
        names[model.ramsey["multipliers"][j]] = f"MULT_{eq + 1}"
    dy_ss = dict(zip(ex.endo_names, np.atleast_1d(ex.ys)))
    err_ss = max(abs(ss.values[v] - dy_ss[d]) for v, d in names.items())

    sol = dsg.solve(model, params, steady=ss)
    sd = np.sqrt(np.diag(np.atleast_2d(ex.sigma_e)))
    shifted = {s for i, s in enumerate(model.states)
               if s in private and not np.any(sol.Q[i])}
    err_irf = 0.0
    for dy_shock, our_shock in shock_map.items():
        size = sd[ex.exo_names.index(dy_shock)]
        ours = sol.irf(our_shock, size=size, periods=H, levels=True)
        for v, d in names.items():
            field = f"{d}_{dy_shock}"
            dy = np.asarray(getattr(ex.irfs, field), float) if hasattr(ex.irfs, field) \
                else np.zeros(H)
            mine = np.asarray(ours[v], float) - ss.values[v]
            if v in shifted:
                mine, dy = mine[1:H], dy[:H - 1]
            else:
                mine = mine[:H]
            err_irf = max(err_irf, float(np.max(np.abs(mine - dy))))
    return dict(name=name, ss=err_ss, irf=err_irf,
                ok=bool(err_ss < TOL and err_irf < TOL), n_vars=model.n_vars, n_cmp=len(names),
                added=model.n_vars - len(private))


def main() -> int:
    bad = 0
    for name in ALL:
        r = compare(name)
        if r.get("missing"):
            print(f"{name:<17} no out/{name}.mat, run fetch.sh and run_ramsey.sh")
            bad += 1
            continue
        flag = "OK" if r["ok"] else "FAILED"
        print(f"{name:<17} s.s. {r['ss']:.1e}   IRFs {r['irf']:.1e}   "
              f"({r['n_cmp']} series; model with {r['n_vars']} variables, {r['added']} from the "
              f"planner)   {flag}")
        bad += not r["ok"]
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
