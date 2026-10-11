"""Top-level entry point: from a declared model to a usable solution."""

from __future__ import annotations

import numpy as np

from .klein import klein
from .linearize import first_order_system
from .model import Model
from .solution import FirstOrderSolution
from .steady import SteadyState, steady_state

__all__ = ["solve"]


def solve(model: Model, params, *, steady=None, guess=None, ctx=None,
          ctx_update=None, positive=(), h: float = 1e-6,
          equilibrate: bool = True, tol: float = 1e-11) -> FirstOrderSolution:
    """Solve ``model`` at ``params`` to first order.

    Either pass an already-solved ``steady`` state or a ``guess`` from which one
    is computed. Everything else is forwarded to :func:`dsgelab.steady_state`.

    Returns a :class:`~dsgelab.solution.FirstOrderSolution` carrying the Klein
    matrices, the steady state, the coordinate map and the parameters, so
    impulse responses and moments need no further arguments.
    """
    if steady is None:
        if guess is None:
            raise ValueError("pass either a solved `steady` state or a `guess` for one")
        steady = steady_state(model, guess, params, ctx=ctx, ctx_update=ctx_update,
                              positive=positive, tol=tol)
    elif not isinstance(steady, SteadyState):
        raise TypeError("`steady` must be a SteadyState; use dsgelab.steady_state to build one")

    system = first_order_system(model, steady, params, ctx=steady.ctx, h=h,
                                equilibrate=equilibrate)
    if system.steady_residual > max(tol, 1e-8):
        raise RuntimeError(
            f"the point handed to the linearisation is not a steady state "
            f"(residual {system.steady_residual:.3e}); solve it first")
    ks = klein(system.A, system.B, model.n_states, C=system.C)
    return FirstOrderSolution(model=model, steady=steady, system=system, klein=ks,
                              params=params, ctx=steady.ctx)
