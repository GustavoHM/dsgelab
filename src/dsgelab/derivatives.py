"""Derivatives of first-order solutions with respect to model parameters."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .estimation import unpack_point
from .gradient import LikelihoodGradient, kalman_likelihood_gradient
from .kalman import _aggregate
from .identification import _richardson
from .linearize import first_order_system
from .klein import _implicit_derivative
from .model import get_param, set_params
from .secondorder import SecondOrderSolution, _dynare_decision_rule, solve_second_order
from .solution import FirstOrderSolution
from .solve import solve
from .steady import steady_state
from .thirdorder import ThirdOrderSolution, _dynare_cubic_rule, solve_third_order

__all__ = ["SecondOrderSolutionJacobian", "SolutionJacobian", "ThirdOrderSolutionJacobian",
           "PosteriorGradient", "second_order_jacobian", "solution_jacobian",
           "third_order_jacobian", "log_likelihood_gradient", "log_posterior_gradient"]


@dataclass
class SolutionJacobian:
    """Derivatives of the solution matrices, with a trailing parameter axis."""

    names: list[str]
    P: np.ndarray
    F: np.ndarray
    Q: np.ndarray
    steady: np.ndarray
    error: dict[str, np.ndarray]


@dataclass
class PosteriorGradient:
    """Log posterior and its gradient in the order of ``problem.names``."""

    logposterior: float
    gradient: np.ndarray
    loglik: float
    logprior: float


@dataclass
class SecondOrderSolutionJacobian:
    """Derivatives of the full second-order decision rule."""

    names: list[str]
    g_k: np.ndarray
    g_u: np.ndarray
    g_kk: np.ndarray
    g_uu: np.ndarray
    g_ku: np.ndarray
    g_s2: np.ndarray
    steady: np.ndarray
    error: dict[str, np.ndarray]


@dataclass
class ThirdOrderSolutionJacobian:
    """Derivatives of the cubic decision-rule blocks."""

    names: list[str]
    g_kkk: np.ndarray
    g_kku: np.ndarray
    g_kuu: np.ndarray
    g_uuu: np.ndarray
    g_s2: np.ndarray
    steady: np.ndarray
    error: dict[str, np.ndarray]


def solution_jacobian(first: FirstOrderSolution, names, *, step: float = 1e-3,
                      levels: int = 4, qz_criterium: float | None = None) -> SolutionJacobian:
    """Differentiate ``P``, ``F``, ``Q`` and the steady state.

    General QZ saddle-path solutions are differentiated through their implicit
    conditions at the reference point. Central parameter differences of the
    steady state and linear system are extrapolated with Richardson's formula;
    perturbed points do not require another QZ decomposition. ``levels`` is the
    number of candidate step sizes, and ``error`` contains the second estimate.
    """
    if step <= 0:
        raise ValueError("step must be positive")
    if levels < 3:
        raise ValueError("levels must be at least 3")
    names = list(names)
    if first.model.n_jumps == 0:
        return _backward_jacobian(first, names, step=step)
    return _implicit_qz_jacobian(first, names, step=step, levels=levels)


def _implicit_qz_jacobian(first, names, *, step, levels):
    model, base, start = first.model, first.params, first.steady.values
    nk = model.n_states
    A, B, C = first.system.A, first.system.B, first.system.C
    Q0 = first.Q if first.Q is not None else np.zeros((nk, model.n_shocks))
    values: dict[str, list[np.ndarray]] = {key: [] for key in ("P", "F", "Q", "steady")}
    errors: dict[str, list[np.ndarray]] = {key: [] for key in values}

    for name in names:
        theta0 = float(get_param(base, name))
        h0 = step * abs(theta0) if theta0 != 0 else step

        def evaluate(h):
            plus_params = set_params(base, {name: theta0 + h})
            minus_params = set_params(base, {name: theta0 - h})
            plus_ss = steady_state(model, start, plus_params)
            minus_ss = steady_state(model, start, minus_params)
            plus = first_order_system(model, plus_ss, plus_params)
            minus = first_order_system(model, minus_ss, minus_params)
            dA = (plus.A - minus.A) / (2 * h)
            dB = (plus.B - minus.B) / (2 * h)
            dC = (plus.C - minus.C) / (2 * h)
            dss = (plus_ss.x - minus_ss.x) / (2 * h)
            dP, dF, dQ = _implicit_derivative(
                A, B, C, first.P, first.F, Q0, first.klein.shock_rows, dA, dB, dC)
            result = (dP, dF, dQ, dss)
            if any(not np.all(np.isfinite(value)) for value in result):
                raise np.linalg.LinAlgError(f"nonfinite derivative of {name!r}")
            return result

        level = []
        for k in range(levels):
            h = h0 / 2 ** k
            try:
                level.append((h, evaluate(h)))
            except Exception:
                continue

        def estimates(trio):
            r1 = tuple((4 * fine - coarse) / 3
                       for coarse, fine in zip(trio[0][1], trio[1][1]))
            r2 = tuple((4 * fine - coarse) / 3
                       for coarse, fine in zip(trio[1][1], trio[2][1]))
            return r1, r2

        best = None
        best_err = np.inf
        for k in range(len(level) - 2):
            estimate, error = estimates(level[k:k + 3])
            err = max(float(np.max(np.abs(a - b))) for a, b in zip(estimate, error))
            scale = max(float(np.max(np.abs(a))) for a in estimate)
            err /= max(scale, 1e-300)
            if err < best_err:
                best, best_err = (estimate, error), err
        if best is None:
            raise ValueError(f"no step of {name!r} from {h0:.3g} down to "
                             f"{h0 / 2 ** (levels - 1):.3g} gives a finite implicit derivative")
        estimate, error = best
        for key, value, err in zip(values, estimate, error):
            values[key].append(value)
            errors[key].append(err)
    return SolutionJacobian(
        names=list(names), **{key: np.stack(value, axis=-1) for key, value in values.items()},
        error={key: np.stack(value, axis=-1) for key, value in errors.items()})


def _backward_jacobian(first: FirstOrderSolution, names, *, step: float) -> SolutionJacobian:
    """Implicit derivatives for models with P = A^-1 B and Q = A^-1 C."""
    model, base, start = first.model, first.params, first.steady.values
    A, P = first.system.A, first.P
    Q = first.Q if first.Q is not None else np.zeros((model.n_states, model.n_shocks))
    C = first.system.C if first.system.C is not None else np.zeros_like(Q)
    Ainv = np.linalg.inv(A)
    P_out, Q_out, steady_out = [], [], []
    errors: dict[str, list[np.ndarray]] = {key: [] for key in ("P", "F", "Q", "steady")}
    for name in names:
        theta0 = float(get_param(base, name))
        h0 = step * abs(theta0) if theta0 != 0 else step

        def evaluate(h):
            plus = steady_state(model, start, set_params(base, {name: theta0 + h}))
            minus = steady_state(model, start, set_params(base, {name: theta0 - h}))
            sp = first_order_system(model, plus, set_params(base, {name: theta0 + h}))
            sm = first_order_system(model, minus, set_params(base, {name: theta0 - h}))
            dA = (sp.A - sm.A) / (2 * h)
            dB = (sp.B - sm.B) / (2 * h)
            dC = (sp.C - sm.C) / (2 * h) if C is not None else np.zeros_like(Q)
            dss = (plus.x - minus.x) / (2 * h)
            return Ainv @ (dB - dA @ P), Ainv @ (dC - dA @ Q), dss

        coarse = evaluate(h0)
        fine = evaluate(h0 / 2)
        estimate = tuple((4 * f - c) / 3 for c, f in zip(coarse, fine))
        error = tuple(f - c for c, f in zip(coarse, fine))
        P_out.append(estimate[0])
        Q_out.append(estimate[1])
        steady_out.append(estimate[2])
        for key, value in zip(("P", "Q", "steady"), error):
            errors[key].append(value)
        errors["F"].append(np.zeros((*first.F.shape, 1)))
    return SolutionJacobian(
        names=list(names), P=np.stack(P_out, axis=-1),
        F=np.zeros((*first.F.shape, len(names))), Q=np.stack(Q_out, axis=-1),
        steady=np.stack(steady_out, axis=-1),
        error={key: (np.stack(value, axis=-1) if key != "F"
                     else np.concatenate(value, axis=-1)) for key, value in errors.items()})


def second_order_jacobian(second: SecondOrderSolution, names, *, step: float = 1e-3,
                          levels: int = 4) -> SecondOrderSolutionJacobian:
    """Differentiate the full second-order decision rule with Richardson extrapolation.

    ``names`` contains model parameters, ``"stderr e"`` or ``"corr e1, e2"``.
    Arrays have a trailing parameter axis.
    """
    if step <= 0:
        raise ValueError("step must be positive")
    if levels < 3:
        raise ValueError("levels must be at least 3")
    names = list(names)
    first = second.first
    model, base, start = first.model, first.params, first.steady.values
    keys = ("g_k", "g_u", "g_kk", "g_uu", "g_ku", "g_s2")

    def solve_at(name, theta):
        params, Sigma, _ = unpack_point(model, base, second.Sigma, {}, [name], [theta])
        ss = steady_state(model, start, params)
        return solve_second_order(solve(model, params, steady=ss), cov=Sigma)

    def starting_value(name):
        if name.startswith("stderr "):
            i = model.shocks.index(name[7:].strip())
            return float(np.sqrt(second.Sigma[i, i]))
        if name.startswith("corr "):
            a, b = (x.strip() for x in name[5:].split(","))
            i, j = model.shocks.index(a), model.shocks.index(b)
            return float(second.Sigma[i, j] / np.sqrt(second.Sigma[i, i] * second.Sigma[j, j]))
        return float(get_param(base, name))

    def metrics(sol):
        rule = _dynare_decision_rule(sol)
        return np.concatenate([getattr(rule, key).ravel() for key in keys]
                              + [sol.first.steady.x.ravel()])

    values: dict[str, list[np.ndarray]] = {key: [] for key in keys + ("steady",)}
    errors: dict[str, list[np.ndarray]] = {key: [] for key in keys + ("steady",)}
    for name in names:
        theta0 = starting_value(name)
        h0 = step * abs(theta0) if theta0 != 0 else step
        level = []
        for k in range(levels):
            h = h0 / 2 ** k
            try:
                sides = [solve_at(name, theta0 + sign * h) for sign in (1, -1)]
            except Exception:
                continue
            level.append((h, *sides))
        best, best_err = None, np.inf
        for k in range(len(level) - 2):
            r1, r2 = _richardson(level[k:k + 3], metrics)
            err = np.max(np.abs(r1 - r2)) / max(np.max(np.abs(r1)), 1e-300)
            if err < best_err:
                best, best_err = k, err
        if best is None:
            raise ValueError(f"no step of {name!r} from {h0:.3g} down to "
                             f"{h0 / 2 ** (levels - 1):.3g} keeps the model solvable")
        trio = level[best:best + 3]
        for key in keys + ("steady",):
            def value(sol, key=key):
                if key == "steady":
                    return sol.first.steady.x
                return getattr(_dynare_decision_rule(sol), key)
            estimate, error = _richardson(trio, value)
            values[key].append(estimate)
            errors[key].append(error)
    return SecondOrderSolutionJacobian(
        names=names, steady=np.stack(values.pop("steady"), axis=-1),
        **{key: np.stack(value, axis=-1) for key, value in values.items()},
        error={key: np.stack(value, axis=-1) for key, value in errors.items()})


def third_order_jacobian(third: ThirdOrderSolution, names, *, step: float = 1e-3,
                         levels: int = 4, sparse: bool = False,
                         direct_limit: int = 6000) -> ThirdOrderSolutionJacobian:
    """Differentiate the cubic decision-rule blocks with Richardson extrapolation.

    ``names`` contains model parameters, ``"stderr e"`` or ``"corr e1, e2"``.
    ``sparse`` and ``direct_limit`` select the storage and Sylvester route of
    :func:`dsgelab.solve_third_order` at every perturbed point.
    """
    if step <= 0:
        raise ValueError("step must be positive")
    if levels < 3:
        raise ValueError("levels must be at least 3")
    names = list(names)
    second = third.second
    first = second.first
    model, base, start = first.model, first.params, first.steady.values
    keys = ("g_kkk", "g_kku", "g_kuu", "g_uuu", "g_s2")

    def solve_at(name, theta):
        params, Sigma, _ = unpack_point(model, base, second.Sigma, {}, [name], [theta])
        ss = steady_state(model, start, params)
        second_at = solve_second_order(solve(model, params, steady=ss), cov=Sigma)
        return solve_third_order(second_at, sparse=sparse, direct_limit=direct_limit)

    def metrics(sol):
        rule = _dynare_cubic_rule(sol)
        rule["g_s2"] = _dynare_decision_rule(sol.second).g_s2
        return np.concatenate([rule[key].ravel() for key in keys]
                              + [sol.second.first.steady.x.ravel()])

    values: dict[str, list[np.ndarray]] = {key: [] for key in keys + ("steady",)}
    errors: dict[str, list[np.ndarray]] = {key: [] for key in keys + ("steady",)}
    for name in names:
        if name.startswith("stderr "):
            i = model.shocks.index(name[7:].strip())
            theta0 = float(np.sqrt(second.Sigma[i, i]))
        elif name.startswith("corr "):
            a, b = (x.strip() for x in name[5:].split(","))
            i, j = model.shocks.index(a), model.shocks.index(b)
            theta0 = float(second.Sigma[i, j] / np.sqrt(second.Sigma[i, i] * second.Sigma[j, j]))
        else:
            theta0 = float(get_param(base, name))
        h0 = step * abs(theta0) if theta0 != 0 else step
        level = []
        for k in range(levels):
            h = h0 / 2 ** k
            try:
                sides = [solve_at(name, theta0 + sign * h) for sign in (1, -1)]
            except Exception:
                continue
            level.append((h, *sides))
        best, best_err = None, np.inf
        for k in range(len(level) - 2):
            r1, r2 = _richardson(level[k:k + 3], metrics)
            err = np.max(np.abs(r1 - r2)) / max(np.max(np.abs(r1)), 1e-300)
            if err < best_err:
                best, best_err = k, err
        if best is None:
            raise ValueError(f"no step of {name!r} from {h0:.3g} down to "
                             f"{h0 / 2 ** (levels - 1):.3g} keeps the model solvable")
        trio = level[best:best + 3]
        for key in keys + ("steady",):
            def value(sol, key=key):
                if key == "steady":
                    return sol.second.first.steady.x
                if key == "g_s2":
                    return _dynare_decision_rule(sol.second).g_s2
                return _dynare_cubic_rule(sol)[key]
            estimate, error = _richardson(trio, value)
            values[key].append(estimate)
            errors[key].append(error)
    return ThirdOrderSolutionJacobian(
        names=names, steady=np.stack(values.pop("steady"), axis=-1),
        **{key: np.stack(value, axis=-1) for key, value in values.items()},
        error={key: np.stack(value, axis=-1) for key, value in errors.items()})


def log_likelihood_gradient(problem, theta) -> LikelihoodGradient:
    """Differentiate a first-order ``EstimationProblem`` likelihood.

    The solution Jacobian supplies model-parameter derivatives. Shock and
    measurement-error standard deviations and correlations use their direct
    formulas. Missing observations and diffuse starts use the univariate
    recursion. Nonlinear filters are outside this routine's scope.
    """
    if problem.order != 1:
        raise NotImplementedError("likelihood gradients currently require order=1")
    theta = np.asarray(theta, dtype=float).reshape(-1)
    names = list(problem.names)
    if theta.size != len(names):
        raise ValueError(f"theta has {theta.size} entries, expected {len(names)}")
    sol, space = problem.solve(theta)
    model_names = [name for name in names
                   if not name.startswith("stderr ") and not name.startswith("corr ")]
    jac = (solution_jacobian(sol, model_names, qz_criterium=problem.qz_criterium)
           if model_names else None)
    _, Sigma, meas = problem.unpack(theta)
    derivatives = {}
    for name in names:
        if jac is not None and name in jac.names:
            j = jac.names.index(name)
            dT, dR, dF, ds = jac.P[..., j], jac.Q[..., j], jac.F[..., j], jac.steady[..., j]
        else:
            dT = np.zeros((sol.model.n_states, sol.model.n_states))
            dR = np.zeros((sol.model.n_states, space.R.shape[1]))
            dF = np.zeros_like(sol.F)
            ds = np.zeros_like(sol.steady.x)
        dSigma = np.zeros_like(Sigma)
        dH = np.zeros_like(space.H)
        if name.startswith("stderr ") and name[7:].strip() in sol.model.shocks:
            i = sol.model.shocks.index(name[7:].strip())
            sd = float(np.sqrt(Sigma[i, i]))
            if sd > 0:
                dSigma[i, :] = Sigma[i, :] / sd
                dSigma[:, i] = Sigma[:, i] / sd
                dSigma[i, i] = 2 * sd
        elif name.startswith("corr "):
            a, b = (x.strip() for x in name[5:].split(","))
            i, j = sol.model.shocks.index(a), sol.model.shocks.index(b)
            sd = float(np.sqrt(Sigma[i, i] * Sigma[j, j]))
            dSigma[i, j] = dSigma[j, i] = sd
        elif name.startswith("stderr ") and name[7:].strip() in problem.observables:
            i = problem.observables.index(name[7:].strip())
            sd = float(meas.get(name[7:].strip(), 0.0))
            dH[i, i] = 2 * sd
        nk = sol.model.n_states
        dZ = np.zeros((len(problem.observables), nk))
        dd = np.zeros(len(problem.observables))
        for i, observable in enumerate(problem.observables):
            position = sol.model.position(observable)
            if observable in sol.model.jumps:
                row = np.asarray(sol.F[sol.model.jumps.index(observable)], dtype=float)
                drow = dF[sol.model.jumps.index(observable)]
            else:
                row = np.zeros(nk)
                row[sol.model.states.index(observable)] = 1.0
                drow = np.zeros(nk)
            if space.transforms[i] == "log":
                dZ[i] = drow
                dd[i] = ds[position] / sol.coords.center[position]
            else:
                scale = sol.coords.scale[position]
                dscale = (np.sign(sol.coords.center[position]) * ds[position]
                          if abs(sol.coords.center[position]) > 1e-8 else 0.0)
                dZ[i] = dscale * row + scale * drow
                dd[i] = ds[position]
        if space.aggregation is not None:
            dT, dR, dZ, dd = _aggregate(dT, dR, dZ, dd, space.aggregation, derivative=True)
        derivatives[name] = {"T": dT, "R": dR, "Sigma": dSigma,
                             "Z": dZ, "d": dd, "H": dH}
    result = kalman_likelihood_gradient(space, problem.Y, derivatives)
    start = problem.presample
    return LikelihoodGradient(
        loglik=float(result.loglik_t[start:].sum()), loglik_t=result.loglik_t,
        gradient=result.gradient_t[start:].sum(axis=0), gradient_t=result.gradient_t)


def log_posterior_gradient(problem, theta) -> PosteriorGradient:
    """Evaluate the log posterior and its gradient in ``problem.names`` order."""
    theta = np.asarray(theta, dtype=float).reshape(-1)
    lp = problem.log_prior(theta)
    if not np.isfinite(lp):
        return PosteriorGradient(-np.inf, np.full(len(problem.names), np.nan),
                                 -np.inf, lp)
    prior_gradient = np.array([
        problem.priors[name].score(value) if name in problem.priors else 0.0
        for name, value in zip(problem.names, theta)])
    try:
        result = log_likelihood_gradient(problem, theta)
    except NotImplementedError:
        raise
    except (ValueError, RuntimeError, np.linalg.LinAlgError):
        return PosteriorGradient(-np.inf, np.full(len(problem.names), np.nan),
                                 -np.inf, lp)
    return PosteriorGradient(lp + result.loglik, prior_gradient + result.gradient,
                             result.loglik, lp)

