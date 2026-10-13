"""Occasionally binding constraints: the piecewise-linear solution of OccBin.

Guerrieri and Iacoviello (2015), as Dynare implements it. A model has a REFERENCE
regime, in which its steady state lies, and one alternative equation for every
combination of constraints that bind. Each regime is linearised at the reference
steady state, so an alternative regime carries a constant: its own residual there.
Given a guess of the dates at which each constraint binds, the path is linear with
time-varying coefficients, found backwards from the first date after which the
reference regime holds for good:

    [A_k + A_u F_{t+1},  -B_u] [k_{t+1}; u_t] = B_k k_t + c_t - A_u g_{t+1},

which gives ``u_t = F_t k_t + g_t`` and ``k_{t+1} = P_t k_t + h_t``, with
``(F, g) = (F_ref, 0)`` from that date on. The constraints are then checked on the
path, in levels, and the guess is revised -- a date joins the binding set where the
binding condition holds and leaves it where the relax condition holds -- until the
guess reproduces itself. Agents expect no further shocks; at a date with a
surprise they re-plan from where they are, starting from the regimes they had
expected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from .linearize import first_order_system
from .model import Model, Point
from .solve import solve
from .steady import SteadyState, steady_state

__all__ = ["OccbinModel", "OccbinResult", "OccbinError", "occbin_simulate"]

_OPS = {"<": np.less, "<=": np.less_equal, ">": np.greater, ">=": np.greater_equal}


class OccbinError(RuntimeError):
    """The regime iteration did not settle."""


@dataclass
class OccbinModel:
    """A model with occasionally binding constraints.

    ``build(binding)`` returns the :class:`~dsgelab.model.Model` of the regime in
    which the constraints named in the frozenset ``binding`` bind (the empty set is
    the reference regime); every regime has the same variables in the same order.
    ``constraints`` maps each constraint to ``{"bind": (expr, op), "relax": (expr,
    op) or None}``: the constraint binds where ``expr op 0`` holds, ``expr`` a SymPy
    expression in current-period variable LEVELS and parameters. Without a relax
    condition the constraint is relaxed wherever it does not bind.
    """

    build: Callable[[frozenset], Model]
    constraints: dict
    _models: dict = field(default_factory=dict, repr=False)
    _tests: dict = field(default_factory=dict, repr=False)

    @property
    def names(self) -> list[str]:
        return list(self.constraints)

    def regime(self, binding: Iterable[str] = ()) -> Model:
        key = frozenset(binding)
        if key not in self._models:
            self._models[key] = self.build(key)
        return self._models[key]

    @property
    def reference(self) -> Model:
        return self.regime(())

    @classmethod
    def from_equations(cls, equations: Sequence[str], *, constraints: Mapping[str, Mapping],
                       states: Sequence[str], jumps: Sequence[str], shocks: Sequence[str],
                       parameters: Sequence[str], name: str = "model",
                       **kwargs: Any) -> "OccbinModel":
        """Equations as in :func:`dsgelab.from_equations`, with Dynare's tags.

        An equation that changes with the regime is written once per variant, each
        prefixed by ``[name='slot', bind='c1,c2', relax='c3']``: the variant applies
        when every constraint in ``bind`` binds and none in ``relax`` does. Each
        constraint of ``constraints`` is ``{"bind": "lb < 0", "relax": "b > M*y"}``.
        """
        import re

        import sympy

        from .dynare import DynareImportError, _csv, _pick_variant, _tags
        from .symbolic import from_equations

        slots, seen = [], {}
        for eq in equations:
            m = re.match(r"^\s*\[([^\]]*)\]\s*", eq)
            if not m:
                slots.append(eq)
                continue
            tags, body = _tags(m.group(1)), eq[m.end():]
            if "name" not in tags:
                raise ValueError(f"a tagged equation needs a name: {eq!r}")
            if tags["name"] not in seen:
                seen[tags["name"]] = len(slots)
                slots.append([])
            slots[seen[tags["name"]]].append((frozenset(_csv(tags.get("bind", ""))),
                                               frozenset(_csv(tags.get("relax", ""))),
                                               body, body))

        def build(binding):
            try:
                eqs = [s if isinstance(s, str) else _pick_variant(s, binding)[1] for s in slots]
            except DynareImportError as exc:
                raise ValueError(str(exc)) from None
            tag = ",".join(sorted(binding))
            return from_equations(eqs, states=states, jumps=jumps, shocks=shocks,
                                  parameters=parameters, name=f"{name}[{tag}]" if tag else name,
                                  **kwargs)

        syms = {v: sympy.Symbol(v) for v in [*states, *jumps, *parameters]}
        parsed = {}
        for cname, spec in constraints.items():
            parsed[cname] = {}
            for kind in ("bind", "relax"):
                text = spec.get(kind)
                if text is None:
                    parsed[cname][kind] = None
                    continue
                mm = re.match(r"(.*?)(<=|>=|<|>)(.*)$", text, flags=re.S)
                if not mm:
                    raise ValueError(f"constraint {cname}: {kind} {text!r} is not a comparison")
                e = sympy.sympify(f"({mm.group(1)}) - ({mm.group(3)})", locals=syms)
                parsed[cname][kind] = (e, mm.group(2))
            if parsed[cname]["bind"] is None:
                raise ValueError(f"constraint {cname} has no bind condition")
        used = {n for s in slots if not isinstance(s, str) for b, r, _, _ in s for n in b | r}
        if used - set(parsed):
            raise ValueError(f"tags name undefined constraints: {sorted(used - set(parsed))}")
        return cls(build=build, constraints=parsed)

    # -- constraint tests on a path in levels ------------------------------------
    def test(self, name: str, kind: str, levels: Mapping[str, np.ndarray],
             params: Mapping[str, float]) -> np.ndarray:
        """Where ``kind`` ("bind" or "relax") of constraint ``name`` holds on ``levels``."""
        spec = self.constraints[name][kind]
        T = len(next(iter(levels.values())))
        if spec is None:          # no relax condition: relaxed wherever it does not bind
            return ~self.test(name, "bind", levels, params)
        key = (name, kind)
        if key not in self._tests:
            import sympy
            e, op = spec
            syms = sorted(e.free_symbols, key=str)
            self._tests[key] = ([str(s) for s in syms], sympy.lambdify(syms, e, "numpy"), op)
        names, fn, op = self._tests[key]
        args = []
        for n in names:
            if n in levels:
                args.append(np.asarray(levels[n], dtype=float))
            elif n in params:
                args.append(float(params[n]))
            else:
                raise KeyError(f"constraint {name}: {n!r} is neither a variable nor a parameter")
        val = np.broadcast_to(np.asarray(fn(*args), dtype=float), (T,))
        return _OPS[op](val, 0.0)


@dataclass
class OccbinResult:
    """Piecewise-linear and linear paths, in levels, from date 1 on."""

    piecewise: dict
    linear: dict
    steady: SteadyState
    regimes: list          # per surprise date: (date, {constraint: bool array of the plan})
    binding: dict          # constraint -> bool array over the simulated dates
    iterations: list

    def expected_regimes(self, date: int) -> dict:
        """The binding dates agents expect at ``date``, from that date on (``[0]`` is ``date``).

        The plan made at the last surprise up to ``date``, which no news has revised.
        """
        s, reg = max(((s, r) for s, r in self.regimes if s <= date), key=lambda p: p[0])
        return {c: arr[date - s:] for c, arr in reg.items()}


def _regime_system(model: Model, ss: SteadyState, params):
    """``A y' = B y + c`` of a regime at the reference steady state (rescaled coordinates)."""
    sys_ = first_order_system(model, ss, params, ctx=ss.ctx, equilibrate=False)
    c = -model.residuals(ss.x, ss.x, np.zeros(model.n_shocks), params, ss.ctx)
    return sys_.A, sys_.B, np.asarray(c, dtype=float)


def occbin_simulate(om: OccbinModel, params: Mapping[str, float],
                    shocks: Mapping[str, Mapping[int, float]],
                    *, periods: int = 100, maxit: int = 30, curb_retrench: bool = False,
                    check_ahead: int = 200, steady: SteadyState | None = None,
                    guess: Point | None = None, initial: Mapping[str, float] | None = None,
                    qz_criterium: float | None = None) -> OccbinResult:
    """Simulate surprise shocks under occasionally binding constraints.

    ``shocks`` is ``{shock: {date: value}}``, dates from 1, each unforeseen until it
    arrives (Dynare's ``shocks(surprise)``). ``periods`` is the length of the
    simulation (Dynare's ``simul_periods``), extended to the last shock date. Each
    plan looks ``max(check_ahead, periods - last shock date)`` periods ahead, and
    further while a constraint still binds at its end. ``maxit`` and
    ``curb_retrench`` are Dynare's ``simul_maxit`` and ``simul_curb_retrench``: with
    the latter, each iteration releases only the last date that should be relaxed.
    ``initial`` gives the states at date 0, in levels, if not the steady state.
    """
    ref = om.reference
    if steady is None:
        if guess is None:
            raise ValueError("pass a solved `steady` state or a `guess`")
        steady = steady_state(ref, guess, params)
    sol = solve(ref, params, steady=steady, qz_criterium=qz_criterium)
    nk, n = ref.n_states, ref.n_vars
    P, F, Q = sol.P, sol.F, sol.Q
    coords = sol.coords
    names = list(ref.states) + list(ref.jumps)
    cn = om.names
    nc = len(cn)

    systems: dict = {}

    def system(binding):
        if binding not in systems:
            systems[binding] = _regime_system(om.regime(binding), steady, params)
        return systems[binding]

    T_shocks = max([t for path in shocks.values() for t in path] + [1])
    n_periods = max(periods, T_shocks)
    E = np.zeros((T_shocks + 1, ref.n_shocks))
    for e, path in shocks.items():
        if e not in ref.shocks:
            raise KeyError(f"{e!r} is not a shock of the model")
        for t, v in path.items():
            if t < 1:
                raise ValueError("surprise dates start at 1")
            E[t, list(ref.shocks).index(e)] = v

    k = np.zeros(nk)
    if initial:
        lv = coords.to_level(np.zeros(n))
        for v, x in initial.items():
            lv[names.index(v)] = x
        k = coords.to_hat(lv)[:nk]
    k0 = k.copy()

    def plan(k0, ind):
        """Path of all variables (hat) at plan dates 1..N under the regimes ``ind`` (N x nc)."""
        N = ind.shape[0]
        active = np.where(ind.any(axis=1))[0]
        T_max = active[-1] + 1 if active.size else 0
        Fs, gs, Ps, hs = [None] * N, [None] * N, [None] * N, [None] * N
        F_next, g_next = F, np.zeros(n - nk)
        for i in range(T_max - 1, -1, -1):
            A, B, c = system(frozenset(cn[j] for j in range(nc) if ind[i, j]))
            M = np.hstack([A[:, :nk] + A[:, nk:] @ F_next, -B[:, nk:]])
            rhs = np.hstack([B[:, :nk], (c - A[:, nk:] @ g_next)[:, None]])
            X = np.linalg.solve(M, rhs)
            Ps[i], hs[i] = X[:nk, :nk], X[:nk, nk]
            Fs[i], gs[i] = X[nk:, :nk], X[nk:, nk]
            F_next, g_next = Fs[i], gs[i]
        y = np.zeros((N, n))
        kk = k0.copy()
        for i in range(N):
            if i < T_max:
                u = Fs[i] @ kk + gs[i]
                y[i] = np.concatenate([kk, u])
                kk = Ps[i] @ kk + hs[i]
            else:
                y[i] = np.concatenate([kk, F @ kk])
                kk = P @ kk
        return y, kk

    def tests(y):
        lv = coords.to_level(y)
        levels = {v: lv[:, j] for j, v in enumerate(names)}
        bind = np.column_stack([om.test(c_, "bind", levels, params) for c_ in cn])
        relax = np.column_stack([om.test(c_, "relax", levels, params) for c_ in cn])
        return bind, relax

    N0 = max(check_ahead, n_periods - T_shocks) + 1
    ind = np.zeros((N0, nc), dtype=bool)
    out = np.zeros((n_periods, n))
    regimes, iterations = [], []
    y = None
    for s in range(1, T_shocks + 1):
        if s > 1 and not E[s].any():
            # no news: the plan made at the last surprise goes on
            k_end = P @ y[-1, :nk]
            y = np.vstack([y[1:], np.concatenate([k_end, F @ k_end])])
            out[s - 1] = y[0]
            k = y[1, :nk]
            ind = np.vstack([ind[1:], np.zeros((1, nc), dtype=bool)])
            continue
        k = k + Q @ E[s]
        history = []
        for it in range(1, maxit + 1):
            if ind[-1].any():          # still binding at the end: look one period further
                ind = np.vstack([ind, np.zeros((1, nc), dtype=bool)])
            if len(ind) < N0:
                ind = np.vstack([ind, np.zeros((N0 - len(ind), nc), dtype=bool)])
            history.append(ind.copy())
            y, _ = plan(k, ind)
            bind, relax = tests(y)
            change = bool(np.any(bind & ~ind) or np.any(relax & ind))
            if not change:
                break
            if curb_retrench:
                retrench = np.zeros_like(ind)
                for j in range(nc):
                    hits = np.where(relax[:, j] & ind[:, j])[0]
                    if hits.size and np.where(relax[:, j])[0][-1] >= np.where(ind[:, j])[0][-1]:
                        retrench[hits[-1], j] = True
                new = (ind | bind) & ~retrench
            else:
                new = (ind | bind) & ~(ind & relax)
            if any(h.shape == new.shape and np.array_equal(h, new) for h in history):
                raise OccbinError(f"date {s}: the regime guess cycles; try curb_retrench=True")
            ind = new
        else:
            raise OccbinError(f"date {s}: no regime fixed point in {maxit} iterations")
        iterations.append(it)
        regimes.append((s, {c_: ind[:, j].copy() for j, c_ in enumerate(cn)}))
        out[s - 1] = y[0]
        k = y[1, :nk]          # the next date starts from the state this plan reaches
        ind = np.vstack([ind[1:], np.zeros((1, nc), dtype=bool)])
    rest = n_periods - T_shocks
    if rest > 0:
        tail = y[1:rest + 1]
        if len(tail) < rest:          # beyond the plan: the reference rule
            kk = y[-1, :nk]
            extra = []
            for _ in range(rest - len(tail)):
                kk = P @ kk
                extra.append(np.concatenate([kk, F @ kk]))
            tail = np.vstack([tail, extra]) if extra else tail
        out[T_shocks:] = tail

    # the linear (reference) solution under the same shocks
    lin = np.zeros((n_periods, n))
    k_lin = k0
    for t in range(1, n_periods + 1):
        if t <= T_shocks:
            k_lin = k_lin + Q @ E[t]
        lin[t - 1] = np.concatenate([k_lin, F @ k_lin])
        k_lin = P @ k_lin

    pw_lv, lin_lv = coords.to_level(out), coords.to_level(lin)
    piecewise = {v: pw_lv[:, j] for j, v in enumerate(names)}
    linear = {v: lin_lv[:, j] for j, v in enumerate(names)}
    bind_out = {c_: np.zeros(n_periods, dtype=bool) for c_ in cn}
    for s, reg in regimes:
        for c_, arr in reg.items():
            m_ = min(len(arr), n_periods - s + 1)
            bind_out[c_][s - 1:s - 1 + m_] = arr[:m_]
    return OccbinResult(piecewise=piecewise, linear=linear, steady=steady, regimes=regimes,
                        binding=bind_out, iterations=iterations)

