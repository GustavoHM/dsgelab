"""dsgelab -- a general toolkit for solving DSGE models.

A model is declared once, as a list of state names, a list of jump names, a list
of shock names and one function returning the equilibrium residuals. Everything
else -- steady state, first-order solution, impulse responses, moments,
second-order risk corrections, nonlinear perfect-foresight transitions,
calibration on targets, optimal policy and welfare -- is computed from that
single declaration, so the stochastic block and the deterministic transition
cannot drift apart into two models that merely look alike.

Quick start::

    import dsgelab as dsg
    from dsgelab.examples import rbc

    model, params, guess = rbc.build()
    sol = dsg.solve(model, params, guess=guess)
    print(sol.summary())
    irf = sol.irf("eps_a", size=0.01, periods=40)

The canonical form behind it is Klein's::

    A E_t x_{t+1} = B x_t + C eps_{t+1},   u_t = F k_t,   k_{t+1} = P k_t + Q eps_{t+1}

with the Blanchard-Kahn condition enforced rather than reported.
"""

from typing import Any

from .calibration import CalibrationResult, calibrate, target_jacobian
from .diagnostics import (blanchard_kahn_report, check_identity, check_slack,
                          compare_irfs, diagnose, steady_state_invariance)
from .bgp import BalancedGrowth, balanced_growth
from .extended_path import ExtendedPath, extended_path, quadrature_nodes
from .identification import Identification, IdentificationCheck, identification
from .irf_matching import IRFMatchingResult, irf_matching
from .local_projection import LocalProjectionResult, local_projection
from .moment_estimation import MomentsResult, method_of_moments
from .foresight import News, TransitionPath, perfect_foresight, perfect_foresight_with_expectation_errors
from .klein import BlanchardKahnError, KleinSolution, klein
from .linearize import (Coordinates, LinearSystem, first_order_system, second_derivatives,
                        third_derivatives)
from .model import Model, VarView, get_param, set_params
from .dynare import DynareImportError, ModFile, from_mod, read_expectation_errors_csv
from .occbin import OccbinError, OccbinModel, OccbinResult, occbin_simulate
from .optimal import (DiscretionError, DiscretionSolution, OSRError, OSRResult, PrivateBlock,
                      RamseyError, discretionary_policy, osr, ramsey_model, ramsey_multipliers,
                      ramsey_steady_state)
from .welfare import Welfare, WelfareError, consumption_equivalent, welfare, welfare_model
from .secondorder import DecisionRule, SecondOrderSolution, solve_second_order
from .thirdorder import ThirdOrderSolution, solve_third_order, stochastic_steady_state
from .tensors import SparseSecondDerivatives, SparseThirdDerivatives
from .derivatives import (SecondOrderSolutionJacobian, SolutionJacobian, PosteriorGradient,
                          ThirdOrderSolutionJacobian, second_order_jacobian,
                          solution_jacobian, third_order_jacobian, log_likelihood_gradient,
                          log_posterior_gradient)
from .gradient import LikelihoodGradient, kalman_likelihood_gradient
from .hmc import hmc
from .nuts import NUTSResult, nuts
from .estimate import EstimationResult, estimate, log_likelihood, log_posterior
from .estimation import (EstimationProblem, MCMCResult, ModeResult, SMCResult,
                         effective_sample_size, maximum_likelihood, metropolis_hastings,
                         model_probabilities, posterior_forecast, posterior_mode,
                         prior_predictive, rhat, smc)
from .forecast_evaluation import (DieboldMariano, ForecastEvaluation, diebold_mariano,
                                  forecast_evaluation)
from .kalman import (FilterResult, Forecast, SmootherResult, StateSpace, forecast,
                     kalman_filter, kalman_smoother, realtime_shock_decomposition,
                     shock_decomposition, state_space)
from .inversion import InversionFilterResult, inversion_filter, inversion_smoother
from .particle import ParticleFilterResult, ParticleSmootherResult, particle_filter, particle_smoother
from .priors import Prior, prior
from .projection import Projection, solve_projection
from .restrictions import (Restriction, RestrictionCheck, RestrictionSample, check_restrictions,
                           sample_restrictions)
from .solution import FirstOrderSolution, shock_covariance
from .solve import solve
from .steady import SteadyState, SteadyStateError, analytic_steady_state, steady_state

__version__ = "0.5.0"

__all__ = [
    "Model", "VarView", "get_param", "set_params",
    "steady_state", "SteadyState", "analytic_steady_state", "SteadyStateError",
    "solve", "FirstOrderSolution", "shock_covariance",
    "klein", "KleinSolution", "BlanchardKahnError",
    "first_order_system", "second_derivatives", "LinearSystem", "Coordinates",
    "solve_second_order", "SecondOrderSolution", "DecisionRule",
    "solve_third_order", "ThirdOrderSolution", "third_derivatives", "SparseThirdDerivatives",
    "stochastic_steady_state",
    "SparseSecondDerivatives",
    "solution_jacobian", "SolutionJacobian", "second_order_jacobian", "SecondOrderSolutionJacobian",
    "third_order_jacobian", "ThirdOrderSolutionJacobian",
    "log_likelihood_gradient", "LikelihoodGradient", "kalman_likelihood_gradient",
    "log_posterior_gradient", "PosteriorGradient", "hmc", "nuts", "NUTSResult",
    "estimate", "EstimationResult", "log_likelihood", "log_posterior",
    "EstimationProblem", "posterior_mode", "maximum_likelihood", "metropolis_hastings", "smc",
    "prior_predictive", "model_probabilities", "posterior_forecast", "rhat",
    "effective_sample_size", "ModeResult", "MCMCResult", "SMCResult",
    "state_space", "StateSpace", "kalman_filter", "kalman_smoother", "FilterResult",
    "particle_filter", "ParticleFilterResult", "particle_smoother", "ParticleSmootherResult",
    "inversion_filter", "InversionFilterResult", "inversion_smoother",
    "SmootherResult", "shock_decomposition", "realtime_shock_decomposition", "forecast",
    "Forecast", "forecast_evaluation", "ForecastEvaluation", "diebold_mariano", "DieboldMariano",
    "prior", "Prior",
    "solve_projection", "Projection",
    "perfect_foresight", "TransitionPath", "perfect_foresight_with_expectation_errors", "News",
    "calibrate", "CalibrationResult", "target_jacobian",
    "identification", "Identification", "IdentificationCheck",
    "method_of_moments", "MomentsResult", "irf_matching", "IRFMatchingResult",
    "local_projection", "LocalProjectionResult",
    "check_restrictions", "sample_restrictions", "Restriction", "RestrictionCheck",
    "RestrictionSample",
    "check_identity", "check_slack", "blanchard_kahn_report", "compare_irfs",
    "steady_state_invariance", "diagnose",
    "ramsey_model", "ramsey_steady_state", "ramsey_multipliers", "RamseyError",
    "discretionary_policy", "DiscretionSolution", "DiscretionError", "PrivateBlock",
    "osr", "OSRResult", "OSRError",
    "welfare", "welfare_model", "Welfare", "WelfareError", "consumption_equivalent",
    "from_mod", "ModFile", "DynareImportError", "read_expectation_errors_csv",
    "OccbinModel", "OccbinResult", "OccbinError", "occbin_simulate",
    "extended_path", "ExtendedPath", "quadrature_nodes", "balanced_growth", "BalancedGrowth",
    "__version__",
    "from_equations", "optimality_conditions", "assemble", "to_dynare", "to_macromodelling",
]


def from_equations(*args: Any, **kwargs: Any) -> Model:
    """Declare a model as text equations. Requires sympy; see :mod:`dsgelab.symbolic`."""
    from .symbolic import from_equations as _fe
    return _fe(*args, **kwargs)


def optimality_conditions(*args: Any, **kwargs: Any) -> dict:
    """Derive first-order conditions from a stated optimisation problem.

    Requires sympy; see :mod:`dsgelab.blocks`.
    """
    from .blocks import optimality_conditions as _oc
    return _oc(*args, **kwargs)


def assemble(*args: Any, **kwargs: Any) -> Model:
    """Join several optimisation problems and closure equations into one Model.

    Requires sympy; see :func:`dsgelab.blocks.assemble`.
    """
    from .blocks import assemble as _as
    return _as(*args, **kwargs)


def to_dynare(*args: Any, **kwargs: Any) -> str:
    """Write a symbolically-declared model as a Dynare ``.mod``. See :mod:`dsgelab.export`."""
    from .export import to_dynare as _td
    return _td(*args, **kwargs)


def to_macromodelling(*args: Any, **kwargs: Any) -> str:
    """Write a symbolically-declared model for MacroModelling.jl. See :mod:`dsgelab.export`."""
    from .export import to_macromodelling as _tm
    return _tm(*args, **kwargs)
