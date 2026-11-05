"""compare_kim.py: Kim's filter against statsmodels' Hamilton filter and against the exact likelihood.

RISE, the reference the roadmap names, runs only in MATLAB (see compare_switching.py). The
references here are:

  1. statsmodels 0.14.4 (statsmodels/run_statsmodels.py, in the environment of
     statsmodels/setup.sh), on two series simulated by dsgelab from Markov-switching models in
     which Kim's filter is exact:
       hamilton  a Fisherian model with an i.i.d. real rate and a Taylor rule whose inflation
                 target and response to inflation switch; observed inflation is
                 mu[s_t] + sigma[s_t] e_t, statsmodels' MarkovRegression with a switching
                 constant and variance. The states carry nothing from one date to the next, so
                 the filter is Hamilton's (1989).
       ar        an AR(1) observed without error around a switching mean, Hamilton's (1989)
                 model, statsmodels' MarkovAutoregression of order 1. Given the regime, the
                 observation reveals the state, so each regime's filtered state has no variance
                 and the collapse loses nothing. statsmodels conditions on the first
                 observation, and so does the comparison, through init= and a missing first
                 value.
     At the parameters of the simulating model: the log likelihood, and the filtered and smoothed
     probabilities of regime 0. For the first series, also the maximum-likelihood estimate:
     maximum_likelihood on a MarkovSwitchingProblem in the Fisherian model's own parameters
     (inflation targets, responses to inflation and staying probabilities), against
     MarkovRegression's fit in its reduced form, compared through the reduced form the estimate
     implies.
  2. The exact likelihood, the mixture of the Kalman filters of all h^T regime paths
     (enumerated_likelihood), on short samples where Kim's collapse is an approximation: the AR
     model with a switching mean and persistence observed with measurement error, and the New
     Keynesian model of Foerster et al. (2013) with inflation observed with error. The
     difference measures Kim's approximation.

Prerequisites:
    python validation/compare_kim.py --prepare
    bash validation/statsmodels/setup.sh
    validation/statsmodels/.venv/bin/python validation/statsmodels/run_statsmodels.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import dsgelab as dsg

HERE = Path(__file__).resolve().parent
OUT = HERE / "out_kim"
failures: list[str] = []
T_SIM = 300


def check(label: str, value: float, limit: float) -> None:
    status = "ok" if value <= limit else "FAIL"
    if status == "FAIL":
        failures.append(label)
    print(f"  {status:4s} {label:62s} {value:9.2e}  (limit {limit:.0e})")


# ------------------------------------------------------------------------------------- models
def fisher():
    """i_t = r_t + E_t pi_{t+1}, i_t = pistar + alpha (pi_t - pistar), r_t i.i.d. with s.d. sig."""
    model = dsg.Model(states=["r"], jumps=["pi"], shocks=["e"],
                      equations=lambda nxt, cur, eps, p: [
                          nxt.r - p["sig"] * eps.e,
                          p["pistar"] + p["alpha"] * (cur.pi - p["pistar"]) - cur.r - nxt.pi])
    params = dict(pistar=2.0, alpha=1.5, sig=1.0)
    switching = {"pistar": (2.0, 8.0), "alpha": (2.5, 1.2)}
    T = np.array([[0.95, 0.05], [0.10, 0.90]])
    return model, params, switching, T


def hamilton_ar():
    """y_t = mu[s_t] + z_t, z_t = phi[s_{t-1}] z_{t-1} + sig e_t."""
    model = dsg.Model(states=["z"], jumps=["y"], shocks=["e"],
                      equations=lambda nxt, cur, eps, p: [
                          nxt.z - p["phi"] * cur.z - p["sig"] * eps.e,
                          cur.y - p["mu"] - cur.z])
    return model, dict(mu=1.0, phi=0.6, sig=0.8)


def reduced_form(sol, space):
    """statsmodels' MarkovRegression parameters implied by the Fisherian model's solution."""
    _, _, cov = space.ergodic_start()
    T = space.transition
    sigma2 = [float(space.Z[s, 0] @ cov[s] @ space.Z[s, 0]) for s in range(2)]
    return [T[0, 0], T[1, 0], space.d[0, 0], space.d[1, 0], sigma2[0], sigma2[1]]


# ------------------------------------------------------------------------------------ prepare
def prepare() -> None:
    OUT.mkdir(exist_ok=True)
    model, params, switching, T = fisher()
    ms = dsg.solve_markov_switching(model, params, switching, T, guess=dict(r=0.0, pi=4.0))
    sim = ms.simulate(T_SIM, seed=20261103, levels=True)
    space = dsg.switching_state_space(ms, ["pi"])
    ham = dict(y=sim["pi"].tolist(), params=reduced_form(ms, space),
               regime=sim["regime"].tolist())

    model, params = hamilton_ar()
    Tm = np.array([[0.9, 0.1], [0.2, 0.8]])
    ms = dsg.solve_markov_switching(model, params, {"mu": (2.0, -1.0)}, Tm,
                                    guess=dict(z=0.0, y=1.0))
    sim = ms.simulate(T_SIM, seed=20261104, levels=True)
    ar = dict(y=sim["y"].tolist(), params=[Tm[0, 0], Tm[1, 0], 2.0, -1.0, 0.8 ** 2, 0.6],
              regime=sim["regime"].tolist())
    (OUT / "input.json").write_text(json.dumps(dict(hamilton=ham, ar=ar), indent=1))
    print(f"wrote {OUT / 'input.json'}")


# --------------------------------------------------------------------------------- statsmodels
def statsmodels_part() -> None:
    spec = json.loads((OUT / "input.json").read_text())
    ref = json.loads((OUT / "statsmodels.json").read_text())
    print(f"\n1. Hamilton's filter: statsmodels {ref['statsmodels']}")

    model, params, switching, T = fisher()
    ms = dsg.solve_markov_switching(model, params, switching, T, guess=dict(r=0.0, pi=4.0))
    space = dsg.switching_state_space(ms, ["pi"])
    y = np.asarray(spec["hamilton"]["y"])
    kim = dsg.kim_filter(space, y)
    r = ref["hamilton"]
    print(f"  hamilton, {y.size} observations: log likelihood {kim.loglik:.10f}, "
          f"statsmodels {r['loglik']:.10f}")
    check("hamilton: log likelihood minus statsmodels'", abs(kim.loglik - r["loglik"]), 1e-9)
    check("hamilton: filtered Pr(s=0) minus statsmodels'",
          float(np.max(np.abs(kim.filtered[:, 0] - r["filtered"]))), 1e-11)
    check("hamilton: smoothed Pr(s=0) minus statsmodels'",
          float(np.max(np.abs(kim.smoothed[:, 0] - r["smoothed"]))), 1e-11)
    regime = np.asarray(spec["hamilton"]["regime"])
    print(f"       the smoothed probability puts the simulated regime above one half at "
          f"{np.mean((kim.smoothed[:, 0] > 0.5) == (regime == 0)):.1%} of the dates")
    short = y[:14]
    check("hamilton: Kim minus the exact likelihood, 14 observations",
          abs(dsg.kim_filter(space, short).loglik - dsg.enumerated_likelihood(space, short)),
          1e-11)

    problem = dsg.MarkovSwitchingProblem(
        model, {"pi": y}, params, ["pi"], switching, T, guess=dict(r=0.0, pi=4.0),
        free=["pistar[0]", "pistar[1]", "alpha[0]", "alpha[1]", "p[0,0]", "p[1,1]"],
        bounds={"pistar[0]": (-20, 20), "pistar[1]": (-20, 20), "alpha[0]": (1.0001, 20),
                "alpha[1]": (1.0001, 20)})
    mode = dsg.maximum_likelihood(problem)
    sol, sp = problem.solve(mode.x)
    mine = np.array(reduced_form(sol, sp))
    theirs = np.array(r["mle"])
    print(f"  maximum likelihood: dsgelab {mode.log_likelihood:.8f}, statsmodels "
          f"{r['mle_loglik']:.8f}")
    for name, a, b in zip(r["param_names"], mine, theirs):
        print(f"       {name:10s} {a:12.6f} {b:12.6f}")
    check("hamilton: maximized log likelihood minus statsmodels'",
          abs(mode.log_likelihood - r["mle_loglik"]), 1e-6)
    check("hamilton: implied reduced form minus statsmodels' estimate",
          float(np.max(np.abs(mine - theirs) / np.maximum(1.0, np.abs(theirs)))), 1e-4)

    model, params = hamilton_ar()
    Tm = np.array([[0.9, 0.1], [0.2, 0.8]])
    ms = dsg.solve_markov_switching(model, params, {"mu": (2.0, -1.0)}, Tm,
                                    guess=dict(z=0.0, y=1.0))
    space = dsg.switching_state_space(ms, ["y"])
    y = np.asarray(spec["ar"]["y"])
    # Conditional on y_1, as statsmodels is: z_1 = y_1 - mu[s] given s_1 = s, with no variance,
    # and the regime at its ergodic distribution.
    init = (ms.ergodic, np.array([[y[0] - 2.0], [y[0] + 1.0]]), np.zeros((2, 1, 1)))
    data = y.copy()
    data[0] = np.nan
    kim = dsg.kim_filter(space, data, init=init)
    r = ref["ar"]
    print(f"  ar, {y.size - 1} observations after the first: log likelihood {kim.loglik:.10f}, "
          f"statsmodels {r['loglik']:.10f}")
    # The coefficients of the solution come from central differences, with relative errors near
    # 1e-10, and here the observation equation divides by the steady state of y.
    check("ar: log likelihood minus statsmodels'", abs(kim.loglik - r["loglik"]), 1e-8)
    check("ar: filtered Pr(s=0) minus statsmodels'",
          float(np.max(np.abs(kim.filtered[1:, 0] - r["filtered"]))), 1e-9)
    check("ar: smoothed Pr(s=0) minus statsmodels'",
          float(np.max(np.abs(kim.smoothed[1:, 0] - r["smoothed"]))), 1e-9)


# ------------------------------------------------------------------------------------- exact
def frwz_nk():
    sys.path.insert(0, str(HERE))
    from compare_switching import frwz_nk as build
    return build()


def exact_part() -> None:
    print("\n2. Kim's filter against the exact likelihood, all regime paths enumerated")
    model, params = hamilton_ar()
    Tm = np.array([[0.9, 0.1], [0.2, 0.8]])
    ms = dsg.solve_markov_switching(model, params, {"mu": (2.0, -1.0), "phi": (0.9, 0.3)}, Tm,
                                    guess=dict(z=0.0, y=1.0))
    errors = {}
    for meas in (0.1, 0.5, 2.0):
        space = dsg.switching_state_space(ms, ["y"], measurement={"y": meas})
        worst, size = 0.0, 0.0
        for seed in range(10):
            y = ms.simulate(14, seed=seed, levels=True)["y"]
            exact = dsg.enumerated_likelihood(space, y)
            worst = max(worst, abs(dsg.kim_filter(space, y).loglik - exact))
            size = max(size, abs(exact))
        errors[meas] = worst
        print(f"  ar, mu and phi switching, measurement s.d. {meas}: largest |Kim - exact| over "
              f"10 samples of 14 dates {worst:.2e} (|log likelihood| up to {size:.1f})")
    # Kim's approximation has no error bound; these limits guard the numbers recorded here.
    check("ar: largest |Kim - exact|", max(errors.values()), 5e-2)

    model, params = frwz_nk()
    Rss = params["Rss"]
    ms = dsg.solve_markov_switching(model, params, {"mu": (0.0075, 0.0025), "psi": (3.1, 0.9)},
                                    [[0.9, 0.1], [0.1, 0.9]],
                                    guess=dict(rl=Rss, e=0.0, PI=1.0, Y=0.9, R=Rss),
                                    next_regime=("mu",))
    # Inflation alone: with output and the interest rate both observed, the regime is identified
    # at almost every date, the filtered probabilities are near 0 or 1 and the collapse is exact
    # to 1e-13.
    space = dsg.switching_state_space(ms, ["PI"], measurement={"PI": 0.001}, cov={"eps_r": 1.0})
    worst, size = 0.0, 0.0
    for seed in range(5):
        sim = ms.simulate(12, seed=seed, levels=True)
        data = {"PI": sim["PI"]}
        exact = dsg.enumerated_likelihood(space, data)
        worst = max(worst, abs(dsg.kim_filter(space, data).loglik - exact))
        size = max(size, abs(exact))
    print(f"  Foerster et al. (2013) NK, inflation observed: largest |Kim - exact| over 5 "
          f"samples of 12 dates {worst:.2e} (|log likelihood| up to {size:.1f})")
    check("NK: largest |Kim - exact|", worst, 5e-3)


def main() -> int:
    if "--prepare" in sys.argv:
        prepare()
        return 0
    if not (OUT / "statsmodels.json").exists():
        print(f"{OUT / 'statsmodels.json'} is missing: see the prerequisites in the docstring")
        failures.append("statsmodels reference")
    else:
        statsmodels_part()
    exact_part()
    print()
    if failures:
        print(f"FAILED: {failures}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
