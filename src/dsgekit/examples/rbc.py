"""Stochastic growth model, in the special case that has a closed-form solution.

With log utility and full depreciation the policy functions are exact,
``k_{t+1} = alpha*beta*a_t*k_t^alpha`` and ``c_t = (1-alpha*beta)*a_t*k_t^alpha``,
so the log-linear solution is known on paper: ``P = [[alpha, 1], [0, rho]]`` and
``F = [alpha, 1]``. That makes it the right test case -- a solver can be checked
against an answer rather than against its own previous output.

The nonlinear equations are the ones written below; nothing here is linearised
by hand, so what the test verifies is the whole chain, steady state included.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..model import Model

__all__ = ["Params", "build", "build_symbolic", "closed_form"]


@dataclass
class Params:
    alpha: float = 0.36
    beta: float = 0.99
    rho: float = 0.95
    delta: float = 1.0          # the closed form needs full depreciation
    gamma: float = 1.0          # log utility


def _equations(nxt, cur, eps, p):
    y = cur.a * cur.k ** p.alpha
    y1 = nxt.a * nxt.k ** p.alpha
    return [
        # resource constraint
        nxt.k - ((1.0 - p.delta) * cur.k + y - cur.c),
        # consumption Euler
        cur.c ** (-p.gamma) - p.beta * nxt.c ** (-p.gamma) * (
            p.alpha * y1 / nxt.k + 1.0 - p.delta),
        # technology
        np.log(nxt.a) - p.rho * np.log(cur.a) - eps.eps_a,
    ]


def build(params: Params | None = None):
    """Return ``(model, params, guess)`` ready for :func:`dsgekit.solve`."""
    p = params or Params()
    model = Model(states=["k", "a"], jumps=["c"], shocks=["eps_a"],
                  equations=_equations, name="rbc")
    k, c = steady_levels(p)
    guess = {"k": k, "a": 1.0, "c": c}
    return model, p, guess


def steady_levels(p: Params) -> tuple[float, float]:
    """Steady-state capital and consumption."""
    kk = (p.alpha * p.beta / (1.0 - p.beta * (1.0 - p.delta))) ** (1.0 / (1.0 - p.alpha))
    cc = kk ** p.alpha - p.delta * kk
    return kk, cc


def closed_form(p: Params) -> dict:
    """The exact log-linear solution, valid only at ``delta == 1``."""
    if abs(p.delta - 1.0) > 1e-12 or abs(p.gamma - 1.0) > 1e-12:
        raise ValueError("the closed form holds only for delta = 1 and log utility")
    P = np.array([[p.alpha, 1.0], [0.0, p.rho]])
    F = np.array([[p.alpha, 1.0]])
    return dict(P=P, F=F)


def build_symbolic(params: Params | None = None):
    """The same model declared as text equations, with exact derivatives.

    Requires sympy. Solving this and :func:`build` must give the same matrices;
    the test that checks it is what keeps the two front ends honest about being
    the same model.
    """
    from ..symbolic import from_equations
    p = params or Params()
    model = from_equations(
        [
            "k(+1) = (1-delta)*k + a*k**alpha - c",
            "c**(-gamma) = beta * c(+1)**(-gamma) * (alpha*a(+1)*k(+1)**(alpha-1) + 1 - delta)",
            "log(a(+1)) = rho*log(a) + eps_a",
        ],
        states=["k", "a"], jumps=["c"], shocks=["eps_a"],
        parameters=["alpha", "beta", "rho", "delta", "gamma"], name="rbc-symbolic",
    )
    k, c = steady_levels(p)
    return model, p, {"k": k, "a": 1.0, "c": c}
