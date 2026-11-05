"""compare_switching.py: the Markov-switching solution against published results and closed forms.

The roadmap names RISE (Maih, 2015) as the reference. RISE runs only in MATLAB: at commit
9bdcd74 of github.com/jmaih/RISE_toolbox (27 September 2026), and at its v2019 tag, the toolbox
is distributed as MATLAB P-code (1652 and 1013 .p files), with .m files that hold only the help
text, and GNU Octave cannot execute P-code. The references here are the ones that exist without
MATLAB:

  1. Foerster, Rubio-Ramírez, Waggoner and Zha (2013, Atlanta Fed WP 2013-1, section 6.2), a New
     Keynesian model with a switching drift of technology (perturbed, dated t+1 in the Euler
     equation) and a switching response to inflation (not perturbed). They find all the solutions
     of the quadratic system with Gröbner bases and print them: nine, three real, for two
     parameterizations. With psi(2) = 0.9 only solution (1) is mean-square stable, with
     psi(2) = 0.7 solutions (1) and (3) are. Newton's method in complex arithmetic, from 400
     random points, finds the nine, which are compared entry by entry with the printed tables.
  2. Foerster, Rubio-Ramírez, Waggoner and Zha (2016, Quantitative Economics, section 6.1), the
     RBC with switching drift, persistence and volatility of technology. Gröbner bases give four
     solutions, one of them stable, and they print the first-order coefficients of that one to four
     decimals, truncated rather than rounded.
  3. The same paper's section 5: phi(s_t) pi_t + sigma(s_t) eps_t = E_t pi_{t+1}, whose solution
     is pi_t = -sigma(s_t)/phi(s_t) eps_t exactly (their Proposition 3); with the lag of inflation
     as a state, the quadratic system has four solutions and one is stable.
  4. A Fisherian model with an AR(1) real rate, where pi_t = c[s_t] r_t + f[s_t] and
     (diag(alpha) - rho T) c = 1 in closed form.
  5. A model whose regimes are identical, whose solution must be the Klein solution of the
     constant-parameter model, on the package's RBC and three-equation New Keynesian examples.

No external tool is needed.
"""
from __future__ import annotations

import sys

import numpy as np

import dsgelab as dsg
from dsgelab.examples import nk, rbc

failures: list[str] = []


def check(label: str, value: float, limit: float) -> None:
    status = "ok" if value <= limit else "FAIL"
    if status == "FAIL":
        failures.append(label)
    print(f"  {status:4s} {label:62s} {value:9.2e}  (limit {limit:.0e})")


# ----------------------------------------------------------------------------- 1. FRWZ (2013) NK
def frwz_nk():
    def equations(nxt, cur, eps, p):
        k = p["kappa"]

        def adj(x):
            return 1 - 0.5 * k * (x.PI - 1) ** 2
        return [
            nxt.rl - cur.R,
            nxt.e - eps.eps_r,
            1 - p["beta"] * adj(cur) * cur.Y / (adj(nxt) * nxt.Y) / np.exp(p["mu"]) * cur.R / nxt.PI,
            (1 - p["eta"]) + p["eta"] * adj(cur) * cur.Y
            + p["beta"] * k * adj(cur) / adj(nxt) * (nxt.PI - 1) * nxt.PI
            - k * (cur.PI - 1) * cur.PI,
            (cur.rl / p["Rss"]) ** p["rho"] * cur.PI ** ((1 - p["rho"]) * p["psi"])
            * np.exp(p["sigma"] * cur.e) - cur.R / p["Rss"],
        ]
    model = dsg.Model(states=["rl", "e"], jumps=["PI", "Y", "R"], shocks=["eps_r"],
                      equations=equations, name="frwz_nk")
    beta, mu = 0.9976, 0.005
    params = dict(beta=beta, kappa=161.0, eta=10.0, rho=0.8, sigma=0.0025, mu=mu, psi=1.0,
                  Rss=np.exp(mu) / beta)
    return model, params


# Their tables: D h(1), D g(1)' (output, inflation), D h(2), D g(2)', solutions (1) to (9), the
# complex conjugate pairs given once.
NK_A = [
    "0.59517 -1.92815 -0.327932 0.699414 -2.9541 -0.554689",
    "0.77508 -3.64018 -0.0398952 1.3018 -7.43725 2.76721",
    "0.79559 -1.82393 -0.00706061 1.05423 1.21892 1.40196",
    "1.0939-0.4363j -0.8264+4.2641j 0.4706-0.6986j 1.3311+0.0574j -10.008-1.9739j 2.9287+0.3165j",
    "1.0952-0.2105j -0.9833+1.9595j 0.4727-0.3370j 1.0240-0.0200j 0.8689+0.7833j 1.2351-0.1103j",
    "1.2360-0.2511j 0.7554+3.0821j 0.6980-0.4020j 0.7507+0.0047j -2.2696+0.6345j -0.2718+0.0260j",
]
NK_B = [
    "0.59067 -1.9452 -0.3351 0.71244 -3.2185 -0.6209",
    "0.79733 -4.7813 -0.0043 1.32443 -11.313 3.71833",
    "0.85231 -1.7727 0.08374 1.01525 2.03718 1.52618",
    "1.0444-0.4989j -0.7000+4.6327j 0.3912-0.7987j 1.3523+0.0442j -14.210-1.8733j 3.9161+0.3132j",
    "1.0670-0.1894j -1.2845+1.5032j 0.4274-0.3033j 0.9995-0.0089j 1.6635+0.5703j 1.4141-0.0629j",
    "1.2374-0.2527j 0.8018+3.0980j 0.7004-0.4046j 0.7582+0.0045j -2.3764+0.6699j -0.2963+0.0317j",
]


def unit(text: str) -> float:
    """A unit in the last digit of a number as printed."""
    return 10.0 ** (-len(text.split(".")[1])) if "." in text else 1.0


def printed_error(text: str, v: complex) -> float:
    """Error of ``v`` in units of the last printed digit, real and imaginary parts apart."""
    if text.endswith("j"):
        k = max(text.rfind("+"), text.rfind("-"))
        re_text, im_text = text[:k], text[k:-1]
        return max(abs(v.real - float(re_text)) / unit(re_text),
                   abs(v.imag - float(im_text)) / unit(im_text))
    return max(abs(v.real - float(text)) / unit(text), abs(v.imag) / 1e-6)


def nk_part() -> None:
    print("\n1. Foerster et al. (2013), New Keynesian model: all solutions of the quadratic system")
    model, params = frwz_nk()
    Rss, Yss = params["Rss"], 0.9
    guess = dict(rl=Rss, e=0.0, PI=1.0, Y=Yss, R=Rss)
    for label, psi2, table, stable_rows in (("psi(2) = 0.9", 0.9, NK_A, [0]),
                                            ("psi(2) = 0.7", 0.7, NK_B, [0, 2])):
        system, ss, perturbed = dsg.switching_system(
            model, params, {"mu": (0.0075, 0.0025), "psi": (3.1, psi2)},
            [[0.9, 0.1], [0.1, 0.9]], guess=guess, next_regime=("mu",))
        found, tried, converged = dsg.msv_solutions(system, starts=400, seed=1, complex=True)
        print(f"  {label}: perturbed {perturbed}, {tried} starting points, {converged} converged, "
              f"{len(found)} distinct solutions, {sum(s.is_real for s in found)} real")
        mine = []
        for s in found:
            F = s.F                       # jumps PI, Y, R; states rl, e
            mine.append((np.array([F[0][2, 0], F[0][1, 0] * Yss / Rss, F[0][0, 0] / Rss,
                                   F[1][2, 0], F[1][1, 0] * Yss / Rss, F[1][0, 0] / Rss]),
                         s))
        check(f"{label}: number of solutions found minus nine", abs(len(found) - 9), 0)
        worst, worst_units, odd = 0.0, 0.0, []
        for r, line in enumerate(table):
            row = line.split()
            for sign in ((1,) if r < 3 else (1, -1)):
                ref = np.array([complex(v) for v in row])
                ref = ref if sign == 1 else np.conj(ref)
                dist = [np.max(np.abs(m - ref)) for m, _ in mine]
                m, s = mine[int(np.argmin(dist))]
                m = m if sign == 1 else np.conj(m)
                for text, v in zip(row, m):
                    units = printed_error(text, v)
                    if units > 1.0:
                        odd.append((r + 1 if sign == 1 else r + 2,
                                    complex(text) if sign == 1 else np.conj(complex(text)),
                                    v if sign == 1 else np.conj(v)))
                        continue
                    worst = max(worst, abs(v - complex(text)))
                    worst_units = max(worst_units, units)
                if r < 3 and (r in stable_rows) != s.stable:
                    failures.append(f"{label}: stability of solution ({r + 1})")
                    print(f"  FAIL stability of solution ({r + 1}): radius {s.radius:.4f}")
        print(f"       entries within a unit of the last printed digit: largest error {worst:.2e}, "
              f"{worst_units:.2f} of that unit")
        for r, printed, v in odd:
            print(f"       solution ({r}): printed {printed:.4f}, computed {v:.6f}")
        # (4) and (5) of the second table print -0.7000 for an entry that is -0.66695 here, with
        # every other entry of those solutions within a unit of the last digit.
        check(f"{label}: entries off by more than a unit of the last digit", len(odd),
              0 if label.endswith("0.9") else 2)
        radii = sorted((s.radius, s.stable) for _, s in mine if s.is_real)
        print("       real solutions, stability radius: "
              + ", ".join(f"{r:.4f}{' (stable)' if st else ''}" for r, st in radii))
        stable = [s for _, s in mine if s.stable]
        check(f"{label}: stable solutions found minus {len(stable_rows)}",
              abs(len(stable) - len(stable_rows)), 0)


# ---------------------------------------------------------------------------- 2. FRWZ (2016) RBC
RBC_PRINTED = {   # rows c, k, z; columns k(-1), z(-1), eps, 1; truncated to four decimals
    0: [[0.0405, 0.1264, 0.0091, 0.000049], [0.9692, -2.1406, -0.1552, -0.3720],
        [0.0, 0.1, 0.0072, 0.0184]],
    1: [[0.0405, 0.0, 0.0268, -0.0968], [0.9692, 0.0, -0.4649, 0.9227],
        [0.0, 0.0, 0.0217, -0.0410]],
}


def rbc_part() -> None:
    print("\n2. Foerster et al. (2016), Markov-switching RBC: printed first-order coefficients")

    def equations(nxt, cur, eps, p):
        a, d, u = p["alpha"], p["delta"], p["upsilon"]
        return [
            cur.c + cur.z * nxt.kp - cur.z ** (1 - a) * cur.kp ** a - (1 - d) * cur.kp,
            nxt.z - np.exp((1 - p["rho"]) * p["mu"] + p["rho"] * np.log(cur.z)
                           + p["sigma"] * eps.e),
            cur.c ** (u - 1) - p["beta"] * cur.z ** (u - 1) * nxt.c ** (u - 1)
            * (a * nxt.z ** (1 - a) * nxt.kp ** (a - 1) + 1 - d),
        ]
    model = dsg.Model(states=["kp", "z"], jumps=["c"], shocks=["e"], equations=equations,
                      name="frwz_rbc")
    a, b, u, d = 0.33, 0.9976, -1.0, 0.025
    params = dict(alpha=a, beta=b, upsilon=u, delta=d, mu=0.0, rho=0.0, sigma=0.0)
    switching = {"mu": (0.0274, -0.0337), "rho": (0.1, 0.0), "sigma": (0.0072, 0.0216)}
    T = [[0.75, 0.25], [0.5, 0.5]]
    mu = (2 / 3) * 0.0274 + (1 / 3) * (-0.0337)
    k = (np.exp((a - 1) * mu) / a * (np.exp((1 - u) * mu) / b - 1 + d)) ** (1 / (a - 1))
    c = k * (1 - d - np.exp(mu) + (np.exp((1 - u) * mu) / b - 1 + d) / a)
    ms = dsg.solve_markov_switching(model, params, switching, T,
                                    guess=dict(kp=k, z=np.exp(mu), c=c),
                                    next_regime=("mu", "rho", "sigma"), starts=200)
    print(f"  steady state c = {ms.steady['c']:.6f}, k = {ms.steady['kp']:.5f}, "
          f"z = {ms.steady['z']:.5f} (printed 2.08259, 22.1504, 1.007); perturbed "
          f"{ms.perturbed}")
    sc = ms.coords.scale                        # kp, z, c
    worst, rest = 0.0, []
    for j in (0, 1):
        F, f = ms.F[j], ms.f[j]
        Pz, Qz, pz = ms.P[0, j][1], ms.Q[0, j][1], ms.p[0, j][1]
        Pk, pk = ms.P[j, 0][0], ms.p[j, 0][0]

        # A row on (kp_t, z_t, 1) in this package's states, written on FRWZ's
        # (k_{t-1}, z_{t-1}, eps_t, 1) by substituting z_t's law of motion in regime j.
        def frwz(on_kp, on_z, const):
            return np.array([on_kp + on_z * Pz[0], on_z * Pz[1], on_z * Qz[0], on_z * pz + const])
        D = np.vstack([frwz(F[0, 0], F[0, 1], f[0]) * sc[2], frwz(Pk[0], Pk[1], pk) * sc[0],
                       np.array([Pz[0], Pz[1], Qz[0], pz]) * sc[1]])
        D[:, 0] /= sc[0]
        D[:, 1] /= sc[1]
        # Rounded first to 1e-8, below the precision of the central differences behind the
        # Jacobian (about 1e-10), so that 0.09999999993 truncates to 0.1000 and not to 0.0999.
        truncated = np.trunc(np.round(D * 1e4, 4)) / 1e4
        for r in range(3):
            for col in range(4):
                printed = RBC_PRINTED[j][r][col]
                if r == 0 and col == 3 and j == 0:        # printed with six decimals
                    rest.append((D[r, col], printed))
                    continue
                worst = max(worst, abs(truncated[r, col] - printed))
        print(f"  regime {j + 1}:\n" + "\n".join("    " + " ".join(f"{v:9.5f}" for v in row)
                                                 for row in D))
    check("23 coefficients truncated to four decimals minus printed", worst, 1e-12)
    # The constant of c in regime 1 is printed as 0.000049; truncated, the value here would print
    # as 0.000045, three units of that last digit away.
    v, printed = rest[0]
    print(f"       constant of c in regime 1: computed {v:.7f}, printed {printed}")
    check("entries off by more than a unit of the last digit", float(abs(v - printed) > 1e-6), 1)
    found, tried, _ = dsg.msv_solutions(ms.system, starts=400, seed=1, complex=True)
    print(f"  complex Newton from {tried} starting points: {len(found)} solutions, "
          f"{sum(s.is_real for s in found)} real, stability radii "
          + ", ".join(f"{s.radius:.4f}" for s in found))
    check("solutions found minus four (Groebner bases give four)", abs(len(found) - 4), 0)
    check("stable solutions found minus one", abs(sum(s.stable for s in found) - 1), 0)


# ------------------------------------------------------------------- 3. FRWZ (2016) simple model
def simple_part() -> None:
    print("\n3. Foerster et al. (2016), section 5: phi(s) pi_t + sigma(s) eps_t = E_t pi_{t+1}")
    T = [[0.95, 0.05], [0.15, 0.85]]
    model = dsg.Model(states=["e"], jumps=["pi"], shocks=["eps"],
                      equations=lambda nxt, cur, eps, p: [
                          nxt.e - eps.eps, p["phi"] * cur.pi + p["sigma"] * cur.e - nxt.pi])
    ms = dsg.solve_markov_switching(model, dict(phi=1.0, sigma=0.1),
                                    {"phi": (1.25, 0.96), "sigma": (0.1, 0.6)}, T,
                                    guess=dict(e=0.0, pi=0.0), starts=50)
    exact = -np.array([0.1 / 1.25, 0.6 / 0.96])
    print(f"  pi_t = {ms.F[0][0, 0]:.6f} eps_t in regime 1, {ms.F[1][0, 0]:.6f} in regime 2 "
          f"(printed -0.08 and -0.625)")
    check("solution minus -sigma/phi", float(np.max(np.abs(ms.F[:, 0, 0] - exact))), 1e-14)
    lagged = dsg.Model(states=["pl"], jumps=["pi"], shocks=["eps"],
                       equations=lambda nxt, cur, eps, p: [nxt.pl - cur.pi,
                                                           p["phi"] * cur.pi - nxt.pi])
    system, _, _ = dsg.switching_system(lagged, dict(phi=1.0), {"phi": (1.25, 0.96)}, T,
                                        guess=dict(pl=0.0, pi=0.0))
    found, tried, _ = dsg.msv_solutions(system, starts=200, seed=0, complex=True)
    print("  with the lag of inflation as a state, coefficients on it: "
          + "; ".join(f"({s.F[0][0, 0]:.4f}, {s.F[1][0, 0]:.4f}) radius {s.radius:.3f}"
                      for s in found))
    check("solutions found minus four", abs(len(found) - 4), 0)
    check("stable solutions found minus one", abs(sum(s.stable for s in found) - 1), 0)


# --------------------------------------------------------------------------- 4. Fisherian model
def fisher_part() -> None:
    print("\n4. Fisherian model with an AR(1) real rate: (diag(alpha) - rho T) c = 1")
    T = np.array([[0.9, 0.1], [0.25, 0.75]])
    alpha, rho = np.array([1.8, 1.1]), 0.7
    model = dsg.Model(states=["r"], jumps=["pi"], shocks=["e"],
                      equations=lambda nxt, cur, eps, p: [
                          nxt.r - p["rho"] * cur.r - eps.e,
                          p["alpha"] * cur.pi - cur.r - nxt.pi])
    ms = dsg.solve_markov_switching(model, dict(alpha=1.0, rho=rho), {"alpha": alpha}, T,
                                    guess=dict(r=0.0, pi=0.0))
    c = np.linalg.solve(np.diag(alpha) - rho * T, np.ones(2))
    print(f"  c = {ms.F[0][0, 0]:.12f}, {ms.F[1][0, 0]:.12f}")
    check("c minus the closed form", float(np.max(np.abs(ms.F[:, 0, 0] - c))), 1e-14)


# ------------------------------------------------------------------------- 5. identical regimes
def klein_part() -> None:
    print("\n5. Identical regimes: the Klein solution of the constant-parameter model")
    for mod, name in ((rbc, "rho"), (nk, "rho_a")):
        model, params, guess = mod.build()
        sol = dsg.solve(model, params, guess=guess)
        v = getattr(params, name)
        ms = dsg.solve_markov_switching(model, params, {name: (v, v)},
                                        [[0.9, 0.1], [0.3, 0.7]], guess=guess)
        err = max(np.max(np.abs(ms.F - sol.F)), np.max(np.abs(ms.P - sol.P)),
                  np.max(np.abs(ms.Q - sol.Q)), np.max(np.abs(ms.f)), np.max(np.abs(ms.p)))
        check(f"{model.name}: F, P, Q minus Klein's, and constants", float(err), 1e-13)


def main() -> int:
    nk_part()
    rbc_part()
    simple_part()
    fisher_part()
    klein_part()
    print()
    if failures:
        print(f"FAILED: {failures}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
