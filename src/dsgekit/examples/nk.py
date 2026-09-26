"""Three-equation New Keynesian model, written in log deviations.

Second test case for a reason: every variable has a ZERO steady state, so it
exercises the branch that carries variables in absolute rather than proportional
deviations. That branch is where a scaling bug hides -- a model that scales by
name rather than by level multiplies a mean-zero variable by ``(1 + yhat)``
around zero and annihilates the shock without any error.

It also has a determinacy region with an economic meaning: the Taylor principle.
``phi_pi > 1`` gives a unique equilibrium and ``phi_pi < 1`` gives a continuum,
which makes it a real test of the Blanchard-Kahn machinery rather than a
formality.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..model import Model

__all__ = ["Params", "build"]


@dataclass
class Params:
    beta: float = 0.99
    sigma: float = 1.0          # inverse elasticity of intertemporal substitution
    kappa: float = 0.1          # slope of the Phillips curve
    phi_pi: float = 1.5         # Taylor coefficient on inflation
    phi_x: float = 0.125        # Taylor coefficient on the output gap
    rho_a: float = 0.9
    rho_v: float = 0.5


def _equations(nxt, cur, eps, p):
    return [
        # technology and monetary shocks
        nxt.a - p.rho_a * cur.a - eps.eps_a,
        nxt.v - p.rho_v * cur.v - eps.eps_v,
        # dynamic IS curve, with the natural rate driven by technology
        cur.x - (nxt.x - (1.0 / p.sigma) * (cur.i - nxt.pi - (p.rho_a - 1.0) * cur.a)),
        # New Keynesian Phillips curve
        cur.pi - (p.beta * nxt.pi + p.kappa * cur.x),
        # Taylor rule
        cur.i - (p.phi_pi * cur.pi + p.phi_x * cur.x + cur.v),
    ]


def build(params: Params | None = None):
    """Return ``(model, params, guess)``; the steady state is the origin."""
    p = params or Params()
    model = Model(states=["a", "v"], jumps=["x", "pi", "i"], shocks=["eps_a", "eps_v"],
                  equations=_equations, name="nk3")
    guess = {"a": 0.0, "v": 0.0, "x": 0.0, "pi": 0.0, "i": 0.0}
    return model, p, guess
