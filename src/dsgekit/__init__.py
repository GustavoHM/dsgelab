"""dsgekit -- a general toolkit for solving DSGE models.

A model is declared once, as a list of state names, a list of jump names, a list
of shock names and one function returning the equilibrium residuals. Everything
else -- steady state, first-order solution, impulse responses, moments,
second-order risk corrections, nonlinear perfect-foresight transitions,
calibration on targets -- is computed from that single declaration, so the
stochastic block and the deterministic transition cannot drift apart into two
models that merely look alike.

Quick start::

    import dsgekit as dk
    from dsgekit.examples import rbc

    model, params, guess = rbc.build()
    sol = dk.solve(model, params, guess=guess)
    print(sol.summary())
    irf = sol.irf("eps_a", size=0.01, periods=40)

The canonical form behind it is Klein's::

    A E_t x_{t+1} = B x_t + C eps_{t+1},   u_t = F k_t,   k_{t+1} = P k_t + Q eps_{t+1}

with the Blanchard-Kahn condition enforced rather than reported.
"""

from .calibration import CalibrationResult, calibrate, target_jacobian
from .diagnostics import (blanchard_kahn_report, check_identity, check_slack,
                          compare_irfs, steady_state_invariance)
from .foresight import TransitionPath, perfect_foresight
from .klein import BlanchardKahnError, KleinSolution, klein
from .linearize import (Coordinates, LinearSystem, first_order_system, second_derivatives,
                        third_derivatives)
from .model import Model, VarView, get_param, set_params
from .secondorder import SecondOrderSolution, solve_second_order
from .thirdorder import ThirdOrderSolution, solve_third_order
from .estimate import EstimationResult, estimate, log_likelihood, log_posterior
from .projection import Projection, solve_projection
from .solution import FirstOrderSolution, shock_covariance
from .solve import solve
from .steady import SteadyState, steady_state

__version__ = "0.2.0"

__all__ = [
    "Model", "VarView", "get_param", "set_params",
    "steady_state", "SteadyState",
    "solve", "FirstOrderSolution", "shock_covariance",
    "klein", "KleinSolution", "BlanchardKahnError",
    "first_order_system", "second_derivatives", "LinearSystem", "Coordinates",
    "solve_second_order", "SecondOrderSolution",
    "solve_third_order", "ThirdOrderSolution", "third_derivatives",
    "estimate", "EstimationResult", "log_likelihood", "log_posterior",
    "solve_projection", "Projection",
    "perfect_foresight", "TransitionPath",
    "calibrate", "CalibrationResult", "target_jacobian",
    "check_identity", "check_slack", "blanchard_kahn_report", "compare_irfs",
    "steady_state_invariance",
    "__version__",
    "from_equations", "optimality_conditions", "to_dynare", "to_macromodelling",
]


def from_equations(*args, **kwargs):
    """Declare a model as text equations. Requires sympy; see :mod:`dsgekit.symbolic`."""
    from .symbolic import from_equations as _fe
    return _fe(*args, **kwargs)


def optimality_conditions(*args, **kwargs):
    """Derive first-order conditions from a stated optimisation problem.

    Requires sympy; see :mod:`dsgekit.blocks`.
    """
    from .blocks import optimality_conditions as _oc
    return _oc(*args, **kwargs)


def to_dynare(*args, **kwargs):
    """Write a symbolically-declared model as a Dynare ``.mod``. See :mod:`dsgekit.export`."""
    from .export import to_dynare as _td
    return _td(*args, **kwargs)


def to_macromodelling(*args, **kwargs):
    """Write a symbolically-declared model for MacroModelling.jl. See :mod:`dsgekit.export`."""
    from .export import to_macromodelling as _tm
    return _tm(*args, **kwargs)
