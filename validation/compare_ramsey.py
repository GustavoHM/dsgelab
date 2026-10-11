"""compare_ramsey.py — o planejador de Ramsey do dsgelab contra o ramsey_model do Dynare.

Dois modelos:

  nk_linear         o novo-keynesiano de livro-texto (Galí, cap. 5), escrito nesta pasta. Tem
                    resposta analítica, e tests/test_optimal.py já a confere; aqui o Dynare é a
                    terceira testemunha.
  nk_ramsey_model   do conjunto de testes do próprio Dynare (tests/optimal_policy). Não linear:
                    Phillips de Rotemberg, Euler com c(+1) e pai(+1), produtividade em exp(a). As
                    derivadas das restrições em relação às variáveis de t+1 dependem das de t, de
                    modo que a CPO precisa de cópias defasadas — é o caso que o linear não exercita.

O que se compara: o estado estacionário e as respostas a impulso, 40 períodos, em níveis, de todas
as variáveis privadas E dos multiplicadores. O Dynare numera um multiplicador por equação, inclusive
as dos processos exógenos; o dsgelab só os das restrições. O MULT_j do dsgelab é o da j-ésima
restrição, e é comparado ao multiplicador que o Dynare dá à MESMA equação. Os dois escrevem o
lagrangiano com o mesmo sinal, e os valores batem sem troca de sinal.

O estacionário de referência é oo_.dr.ys, o ponto em que o Dynare linearizou. oo_.steady_state não
serve: no nk_ramsey_model ele guarda o initval (c = n = 0.9671684882), que erra a raiz exata
c = 0.875^(1/4) na sétima casa.

O .mod do Dynare não fixa a ordem, e a omissão do Dynare é 2: as IRFs dele seriam generalizadas, de
segunda ordem. run_ramsey.sh fixa order=1 antes de reexecutar o stoch_simul.

  rbc_labortax      um RBC com imposto sobre o trabalho como instrumento, escrito nesta pasta. O
                    equilíbrio competitivo com tau = 0 é eficiente, e o planejador o mantém em zero;
                    o que o caso exercita é a condição do planejador para um ESTOQUE, k(+1),
                    escolhido em t e válida em esperança.

Datação: o k do Dynare em predetermined_variables entra nas equações como o do dsgelab, mas a
série que o Dynare reporta é a da variável redatada, k(+1). Como em compare.py, as variáveis
predeterminadas cuja linha de Q é nula são comparadas com um período de defasagem.

Pré-requisitos:
    bash fetch.sh          # baixa mods/ramsey/nk_ramsey_model.mod
    bash run_ramsey.sh     # roda o Dynare e exporta out/<nome>.mat
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import dsgelab as dk
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
    # Tradução literal de tests/optimal_policy/nk_ramsey_model.mod. A(-1) vira o estado `a`
    # com a(+1) = rho*a + u; o resto é escrito como no .mod, com ^ trocado por **.
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
    # O k do dsgelab é o estoque disponível em t; o .mod declara predetermined_variables k,
    # que é a mesma datação nas equações. a(-1) vira o estado `a`.
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
    # nome no dsgelab -> nome no Dynare: as privadas e, para MULT_j, o da mesma equação
    names = {v: v for v in private}
    for j, eq in enumerate(model.ramsey["constraint_index"]):
        names[model.ramsey["multipliers"][j]] = f"MULT_{eq + 1}"
    dy_ss = dict(zip(ex.endo_names, np.atleast_1d(ex.ys)))
    err_ss = max(abs(ss.values[v] - dy_ss[d]) for v, d in names.items())

    sol = dk.solve(model, params, steady=ss)
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
            print(f"{name:<17} sem out/{name}.mat — rode fetch.sh e run_ramsey.sh")
            bad += 1
            continue
        flag = "OK" if r["ok"] else "FALHOU"
        print(f"{name:<17} e.e. {r['ss']:.1e}   IRFs {r['irf']:.1e}   "
              f"({r['n_cmp']} séries; modelo com {r['n_vars']} variáveis, {r['added']} do "
              f"planejador)   {flag}")
        bad += not r["ok"]
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
