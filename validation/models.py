"""Dynare's reference models, declared in dsgelab.

Each port is a transcription of the `.mod` equations, with a single mechanical translation, the
TIMING.

    Dynare                          dsgelab
    ------------------------------  --------------------------------------------
    x        (jump variable)        cur["x"]
    x(+1)                           nxt["x"]
    k        (stock decided in t)     nxt["k"]      <- k is PREDETERMINED
    k(-1)    (stock used in t)         cur["k"]
    a        (process with a(-1))      cur["a"], and the equation moves one period ahead
    e        (innovation)              dsgelab dated t+1

The stock is where care is needed. In Dynare `k` is the END-of-period stock and `k(-1)` the one
that produced `y` in the period. In dsgelab the state in t is what is available in t, that is,
Dynare's `k(-1)`. Swapping the two gives a model that solves, satisfies Blanchard-Kahn and responds
to shocks, and is not the same model. This is why the validation compares IRFs and moments, and not
only the steady state.

The exogenous processes are not shifted. In Dynare `a = rho*a(-1) + e` determines `a_t` from
`t-1` plus the innovation of `t`, which is exactly dsgelab's convention
(`nxt.a = rho*cur.a + eps`).

Each function returns `(model, params, guess, cov, dynare_mod)`.
"""
from __future__ import annotations

import numpy as np

from dsgelab.symbolic import from_equations

__all__ = ["example1", "example2", "rbc_baseline", "ALL"]


# ---------------------------------------------------------------------------
# example1.mod — Collard (2001), "Stochastic simulations with DYNARE"
# ---------------------------------------------------------------------------
def example1():
    """Stochastic growth with leisure and two CROSS-LINKED exogenous processes.

    Exercises CORRELATED shocks (`var e, u = phi*0.009*0.009`) and exogenous processes that feed
    each other (`a` depends on `b(-1)` and vice versa), so that the state transition matrix is not
    diagonal and the shock loading is not the identity.
    """
    p = dict(alpha=0.36, rho=0.95, tau=0.025, beta=0.99, delta=0.025, psi=0.0, theta=2.95)
    model = from_equations(
        [
            # intratemporal FOC:  c*theta*h^(1+psi) = (1-alpha)*y
            "c*theta*h**(1+psi) - (1-alpha)*y",
            # Euler:  k = beta*((exp(b)*c)/(exp(b(+1))*c(+1)))*(exp(b(+1))*alpha*y(+1)+(1-delta)*k)
            "k(+1) - beta*((exp(b)*c)/(exp(b(+1))*c(+1)))"
            "*(exp(b(+1))*alpha*y(+1) + (1-delta)*k(+1))",
            # production:  y = exp(a)*k(-1)^alpha*h^(1-alpha)
            "y - exp(a)*k**alpha*h**(1-alpha)",
            # accumulation:  k = exp(b)*(y-c) + (1-delta)*k(-1)
            "k(+1) - exp(b)*(y-c) - (1-delta)*k",
            # cross-linked processes
            "a(+1) - rho*a - tau*b - eps_e",
            "b(+1) - tau*a - rho*b - eps_u",
        ],
        states=["k", "a", "b"], jumps=["y", "c", "h"],
        shocks=["eps_e", "eps_u"], parameters=list(p), derivatives="third", name="example1",
    )
    guess = {"y": 1.08068253095672, "c": 0.80359242014163, "h": 0.29175631001732,
             "k": 11.08360443260358, "a": 0.0, "b": 0.0}
    phi, sd = 0.1, 0.009
    cov = np.array([[sd ** 2, phi * sd * sd], [phi * sd * sd, sd ** 2]])
    return model, p, guess, cov, "example1.mod"


# ---------------------------------------------------------------------------
# example2.mod — the SAME model, written in logarithms
# ---------------------------------------------------------------------------
def example2():
    """The example1 model with every variable in logs, and uncorrelated shocks.

    Porting both tests something. The REAL allocation has to be the same, and the two versions
    differ only by a transformation of the variables. This tests dsgelab's own choice of
    coordinates, which carries PROPORTIONAL deviations for variables with a nonzero s.s. Here the
    variables are already logs, and the deviations are absolute.
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
        shocks=["eps_e", "eps_u"], parameters=list(p), derivatives="third", name="example2",
    )
    guess = {"y": 0.1, "c": -0.2, "h": -1.2, "k": 2.4, "a": 0.0, "b": 0.0}
    cov = np.diag([0.009 ** 2, 0.009 ** 2])
    return model, p, guess, cov, "example2.mod"


# ---------------------------------------------------------------------------
# RBC_baseline.mod — Johannes Pfeifer, calibrated to US data 1947Q4-2016Q1
# ---------------------------------------------------------------------------
def rbc_baseline():
    """RBC with leisure, growth and government spending.

    Exercises a 15-variable model with six pure logarithmic identities. These are STATIC
    equations, which put zero rows in A and make the QZ pencil singular. In that situation an
    implementation that inverts A fails, and one that goes through the QZ decomposition does not.

    The derived parameters (gammax, delta, beta, psi, g_ss) follow from the calibration targets by
    the same arithmetic as the `steady_state_model` block of the `.mod`.
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
            "psi*c**sigma*1/(1-l) - w",                       # labour FOC
            "gammax*k(+1) - (1-delta)*k - invest",            # accumulation
            "y - invest - c - g_ss*exp(ghat)",                # resource constraint
            "y - exp(z)*k**alpha*l**(1-alpha)",               # production
            "w - (1-alpha)*y/l",                              # labour FOC, firm
            "r - 4*alpha*y/k",                                # annualised interest rate
            "z(+1) - rhoz*z - eps_z",
            "ghat(+1) - rhog*ghat - eps_g",
            # NOTE the timing. In the .mod, `log_k = log(k)` refers to Dynare's CURRENT `k`,
            # which is the END-of-period stock, the `k(+1)` here. Writing `log(k)` would give the
            # log of the available stock, a series lagged by one period. The s.s. would coincide
            # and the IRF would not, which is exactly the kind of error that only the dynamic
            # comparison reveals.
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


#: name -> constructor, in the order in which the validation goes through them
ALL = {"example1": example1, "example2": example2, "rbc_baseline": rbc_baseline}
