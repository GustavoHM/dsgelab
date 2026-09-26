"""Os modelos de referência do Dynare, declarados em dsgekit.

Cada porte é uma transcrição das equações do `.mod`, com uma única tradução mecânica: a DATAÇÃO.

    Dynare                          dsgekit
    ------------------------------  --------------------------------------------
    x        (variável de salto)    cur["x"]
    x(+1)                           nxt["x"]
    k        (estoque decidido em t)  nxt["k"]      <- k é PREDETERMINADO
    k(-1)    (estoque usado em t)      cur["k"]
    a        (processo com a(-1))      cur["a"], e a equação anda um período à frente
    e        (inovação)                dsgekit datada em t+1

O ponto delicado é o estoque. No Dynare `k` é o estoque de FIM de período e `k(-1)` o que produziu
`y` no período; no dsgekit o estado em t é o que está disponível em t, isto é, o `k(-1)` do Dynare.
Trocar os dois dá um modelo que resolve, satisfaz Blanchard-Kahn e responde a choques — e não é o
mesmo modelo. É por isso que a validação compara IRFs e momentos, e não apenas o estado
estacionário.

Os processos exógenos não sofrem deslocamento: `a = rho*a(-1) + e` no Dynare determina `a_t` a
partir de `t-1` mais a inovação de `t`, que é exatamente a convenção do dsgekit
(`nxt.a = rho*cur.a + eps`).

Cada função devolve `(model, params, guess, cov, dynare_mod)`.
"""
from __future__ import annotations

import numpy as np

from dsgekit.symbolic import from_equations

__all__ = ["example1", "example2", "rbc_baseline", "ALL"]


# ---------------------------------------------------------------------------
# example1.mod — Collard (2001), "Stochastic simulations with DYNARE"
# ---------------------------------------------------------------------------
def example1():
    """Crescimento estocástico com lazer e dois processos exógenos CRUZADOS.

    Exercita: choques CORRELACIONADOS (`var e, u = phi*0.009*0.009`) e processos exógenos que se
    alimentam um do outro (`a` depende de `b(-1)` e vice-versa), de modo que a matriz de transição
    dos estados não é diagonal e o carregamento de choque não é trivial.
    """
    p = dict(alpha=0.36, rho=0.95, tau=0.025, beta=0.99, delta=0.025, psi=0.0, theta=2.95)
    model = from_equations(
        [
            # FOC intratemporal:  c*theta*h^(1+psi) = (1-alpha)*y
            "c*theta*h**(1+psi) - (1-alpha)*y",
            # Euler:  k = beta*((exp(b)*c)/(exp(b(+1))*c(+1)))*(exp(b(+1))*alpha*y(+1)+(1-delta)*k)
            "k(+1) - beta*((exp(b)*c)/(exp(b(+1))*c(+1)))"
            "*(exp(b(+1))*alpha*y(+1) + (1-delta)*k(+1))",
            # produção:  y = exp(a)*k(-1)^alpha*h^(1-alpha)
            "y - exp(a)*k**alpha*h**(1-alpha)",
            # acumulação:  k = exp(b)*(y-c) + (1-delta)*k(-1)
            "k(+1) - exp(b)*(y-c) - (1-delta)*k",
            # processos cruzados
            "a(+1) - rho*a - tau*b - eps_e",
            "b(+1) - tau*a - rho*b - eps_u",
        ],
        states=["k", "a", "b"], jumps=["y", "c", "h"],
        shocks=["eps_e", "eps_u"], parameters=list(p), name="example1",
    )
    guess = {"y": 1.08068253095672, "c": 0.80359242014163, "h": 0.29175631001732,
             "k": 11.08360443260358, "a": 0.0, "b": 0.0}
    phi, sd = 0.1, 0.009
    cov = np.array([[sd ** 2, phi * sd * sd], [phi * sd * sd, sd ** 2]])
    return model, p, guess, cov, "example1.mod"


# ---------------------------------------------------------------------------
# example2.mod — o MESMO modelo, escrito em logaritmos
# ---------------------------------------------------------------------------
def example2():
    """O modelo do example1 com todas as variáveis em log, e choques não correlacionados.

    Por que vale a pena portar os dois: a alocação REAL tem de ser a mesma, e as duas leituras
    diferem apenas por uma transformação das variáveis. É um teste da própria escolha de
    coordenadas do dsgekit, que carrega desvios PROPORCIONAIS para variáveis com e.e. não nulo —
    aqui as variáveis já são logs, e os desvios são absolutos.
    """
    p = dict(alpha=0.36, rho=0.95, tau=0.025, beta=0.99, delta=0.025, psi=0.0, theta=2.95)
    model = from_equations(
        [
            "exp(c)*theta*exp(h)**(1+psi) - (1-alpha)*exp(y)",
            "exp(k(+1)) - beta*((exp(b)*exp(c))/(exp(b(+1))*exp(c(+1))))"
            "*(exp(b(+1))*alpha*exp(y(+1)) + (1-delta)*exp(k(+1)))",
            "exp(y) - exp(a)*exp(k)**alpha*exp(h)**(1-alpha)",
            "exp(k(+1)) - exp(b)*(exp(y)-exp(c)) - (1-delta)*exp(k)",
            "a(+1) - rho*a - tau*b - eps_e",
            "b(+1) - tau*a - rho*b - eps_u",
        ],
        states=["k", "a", "b"], jumps=["y", "c", "h"],
        shocks=["eps_e", "eps_u"], parameters=list(p), name="example2",
    )
    guess = {"y": 0.1, "c": -0.2, "h": -1.2, "k": 2.4, "a": 0.0, "b": 0.0}
    cov = np.diag([0.009 ** 2, 0.009 ** 2])
    return model, p, guess, cov, "example2.mod"


# ---------------------------------------------------------------------------
# RBC_baseline.mod — Johannes Pfeifer, calibrado a dados dos EUA 1947Q4-2016Q1
# ---------------------------------------------------------------------------
def rbc_baseline():
    """RBC com lazer, crescimento e gasto do governo.

    Exercita um modelo de 15 variáveis com seis identidades logarítmicas puras — equações
    ESTÁTICAS, que põem linhas nulas em A e tornam o feixe QZ singular. É a situação em que uma
    implementação que inverta A falha, e uma que passe pela transformação QZ não.

    Os parâmetros derivados (gammax, delta, beta, psi, g_ss) saem dos alvos de calibração pela
    mesma aritmética do bloco `steady_state_model` do `.mod`.
    """
    alpha, sigma, i_y, k_y = 0.33, 1.0, 0.25, 10.4
    x, n, rhoz, rhog, gshare = 0.0055, 0.0027, 0.97, 0.989, 0.2038
    gammax = (1 + n) * (1 + x)
    delta = i_y / k_y - x - n - n * x
    beta = (1 + x) * (1 + n) / (alpha / k_y + (1 - delta))
    l_ss = 0.33
    k_ss = ((1 / beta * (1 + n) * (1 + x) - (1 - delta)) / alpha) ** (1 / (alpha - 1)) * l_ss
    invest_ss = (x + n + delta + n * x) * k_ss
    y_ss = k_ss ** alpha * l_ss ** (1 - alpha)
    g_ss = gshare * y_ss
    c_ss = (1 - gshare) * y_ss - invest_ss
    psi = (1 - alpha) * (k_ss / l_ss) ** alpha * (1 - l_ss) / c_ss ** sigma

    p = dict(alpha=alpha, sigma=sigma, gammax=gammax, delta=delta, beta=beta,
             psi=psi, g_ss=g_ss, rhoz=rhoz, rhog=rhog)
    model = from_equations(
        [
            # Euler: c^-sigma = beta/gammax*c(+1)^-sigma*(alpha*exp(z(+1))*(k/l(+1))^(alpha-1)+1-delta)
            "c**(-sigma) - beta/gammax*c(+1)**(-sigma)"
            "*(alpha*exp(z(+1))*(k(+1)/l(+1))**(alpha-1) + (1-delta))",
            "psi*c**sigma*1/(1-l) - w",                       # FOC do trabalho
            "gammax*k(+1) - (1-delta)*k - invest",            # acumulação
            "y - invest - c - g_ss*exp(ghat)",                # restrição de recursos
            "y - exp(z)*k**alpha*l**(1-alpha)",               # produção
            "w - (1-alpha)*y/l",                              # CPO do trabalho, firma
            "r - 4*alpha*y/k",                                # juro anualizado
            "z(+1) - rhoz*z - eps_z",
            "ghat(+1) - rhog*ghat - eps_g",
            # ATENÇÃO à datação: no .mod `log_k = log(k)` refere-se ao `k` CORRENTE do Dynare,
            # que é o estoque de FIM de período — o `k(+1)` daqui. Escrever `log(k)` daria o log
            # do estoque disponível, uma série atrasada em um período: o e.e. coincidiria e a IRF
            # não, que é exatamente o tipo de erro que só a comparação dinâmica revela.
            "log_y - log(y)", "log_k - log(k(+1))", "log_c - log(c)",
            "log_l - log(l)", "log_w - log(w)", "log_invest - log(invest)",
        ],
        states=["k", "z", "ghat"],
        jumps=["y", "c", "l", "r", "w", "invest",
               "log_y", "log_k", "log_c", "log_l", "log_w", "log_invest"],
        shocks=["eps_z", "eps_g"], parameters=list(p), name="RBC_baseline",
    )
    guess = {"k": k_ss, "z": 0.0, "ghat": 0.0, "y": y_ss, "c": c_ss, "l": l_ss,
             "r": 4 * alpha * y_ss / k_ss, "w": (1 - alpha) * y_ss / l_ss,
             "invest": invest_ss, "log_y": np.log(y_ss), "log_k": np.log(k_ss),
             "log_c": np.log(c_ss), "log_l": np.log(l_ss),
             "log_w": np.log((1 - alpha) * y_ss / l_ss), "log_invest": np.log(invest_ss)}
    cov = np.diag([0.66 ** 2, 1.04 ** 2])
    return model, p, guess, cov, "RBC_baseline.mod"


#: nome -> construtor, na ordem em que a validação os percorre
ALL = {"example1": example1, "example2": example2, "rbc_baseline": rbc_baseline}
