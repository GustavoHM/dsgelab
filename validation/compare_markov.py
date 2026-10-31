"""compare_markov.py: the discretisation of an AR(1), against QuantEcon and the closed form.

  1. tauchen and rouwenhorst against QuantEcon's functions of the same names, on 120 processes
     (persistence from -0.5 to 0.999, two innovation standard deviations, 2 to 101 nodes, two
     means): the nodes and the transition matrices, and Rouwenhorst's stationary distribution.
     QuantEcon's mu is (1 - rho) times the mean.
  2. tauchen against the same probabilities computed with 40 significant digits (mpmath): each
     transition probability in relative terms, and the stationary distribution. QuantEcon's
     errors are printed beside dsgelab's: it computes the upper tail as 1 - cdf, which loses the
     relative accuracy of small probabilities, rounds some of them to zero, and moves the
     stationary distribution of the coarsest chains.
  3. rouwenhorst against the closed form: the binomial stationary distribution, and the mean, the
     variance and the autocorrelations at lags 1 and 4 of the AR(1), for every case.
  4. Every stationary distribution is checked to be invariant, pi P = pi.

A Tauchen chain with rho = 0.999 and few nodes has probabilities that underflow to zero in double
precision, and both packages find it reducible; it is left out of the comparison in 2. The largest
relative errors of dsgelab's probabilities are in the far tail, below 1e-200, where an interval
ends some 35 standard deviations from the conditional mean: there a rounding of the end point in
double precision moves the probability by a relative a^2 eps, about 1e-12, whatever the method.

Prerequisites:
    pip install quantecon        # 0.13.1 for the numbers in README.md
"""
from __future__ import annotations

import sys
import warnings

import mpmath as mp
import numpy as np
from scipy.stats import binom

import dsgelab as dsg

CASES = [(rho, sigma, n, mean)
         for rho in (-0.5, 0.0, 0.5, 0.9, 0.99, 0.999)
         for sigma in (0.007, 1.0)
         for n in (2, 3, 7, 25, 101)
         for mean in (0.0, -1.3)]
N_STD = 3


def exact_tauchen(rho: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Tauchen's transition matrix and stationary distribution, to 40 digits.

    Neither depends on sigma or on the mean, which only scale and shift the nodes.
    """
    with mp.workdps(40):
        r = mp.mpf(rho)
        s = 1 / mp.sqrt(1 - r ** 2)
        x = [-N_STD * s + 2 * N_STD * s * i / (n - 1) for i in range(n)]
        h = (x[1] - x[0]) / 2
        P = mp.matrix(n, n)
        for i in range(n):
            m = r * x[i]
            for j in range(n):
                lo = x[j] - h - m if j > 0 else -mp.inf
                hi = x[j] + h - m if j < n - 1 else mp.inf
                # a tail probability below 1e-40 would vanish in 1 - cdf even at 40 digits
                P[i, j] = mp.ncdf(-lo) - mp.ncdf(-hi) if lo > 0 else mp.ncdf(hi) - mp.ncdf(lo)
        # The stationary distribution by GTH elimination, which stays accurate where LU finds the
        # most persistent chains singular, checked by its residual in the same arithmetic.
        A = P.copy()
        for k in range(n - 1):
            scale = mp.fsum(A[k, j] for j in range(k + 1, n))
            for i in range(k + 1, n):
                A[i, k] /= scale
            for i in range(k + 1, n):
                for j in range(k + 1, n):
                    A[i, j] += A[i, k] * A[k, j]
        pi = [mp.mpf(0)] * n
        pi[-1] = mp.mpf(1)
        for k in range(n - 2, -1, -1):
            pi[k] = mp.fsum(pi[i] * A[i, k] for i in range(k + 1, n))
        total = mp.fsum(pi)
        pi = [v / total for v in pi]
        residual = max(abs(mp.fsum(pi[i] * P[i, j] for i in range(n)) - pi[j]) for j in range(n))
        assert residual < mp.mpf(10) ** -30, f"40-digit stationary distribution off by {residual}"
        return (np.array([[float(P[i, j]) for j in range(n)] for i in range(n)]),
                np.array([float(v) for v in pi]))


def rel(a: np.ndarray, b: np.ndarray) -> float:
    """Largest relative difference over the entries of b that double precision can represent."""
    ok = b > 1e-300
    return float(np.max(np.abs(a[ok] / b[ok] - 1)))


def main() -> bool:
    try:
        import quantecon as qe
    except ImportError:
        print("QuantEcon is not installed: pip install quantecon")
        return False
    print(f"QuantEcon {qe.__version__}, {len(CASES)} processes")
    d = dict.fromkeys(["t_nodes", "t_P", "r_nodes", "r_P", "r_pi", "invariant", "binomial",
                       "moments", "P_ours", "P_qe", "pi_ours", "pi_qe"], 0.0)
    exact = {}
    reducible, zeroed, qe_reducible = set(), set(), set()
    ok = True
    for rho, sigma, n, mean in CASES:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # QuantEcon warns about its changed signature
            qt = qe.markov.tauchen(n, rho, sigma, mu=(1 - rho) * mean, n_std=N_STD)
            qr = qe.markov.rouwenhorst(n, rho, sigma, mu=(1 - rho) * mean)
        t = dsg.tauchen(rho, sigma, n, mean=mean, n_std=N_STD)
        r = dsg.rouwenhorst(rho, sigma, n, mean=mean)
        scale = max(1.0, abs(mean))
        d["t_nodes"] = max(d["t_nodes"], np.abs(t.nodes - qt.state_values).max() / scale)
        d["t_P"] = max(d["t_P"], np.abs(t.transition - qt.P).max())
        d["r_nodes"] = max(d["r_nodes"], np.abs(r.nodes - qr.state_values).max() / scale)
        d["r_P"] = max(d["r_P"], np.abs(r.transition - qr.P).max())

        try:
            pi = t.stationary
        except ValueError:
            reducible.add((rho, n))
            ok &= not qt.is_irreducible           # both must find it reducible
            pi = None
        if pi is not None:
            if (rho, n) not in exact:
                exact[rho, n] = exact_tauchen(rho, n)
            P_ex, pi_ex = exact[rho, n]
            d["P_ours"] = max(d["P_ours"], rel(t.transition, P_ex))
            d["pi_ours"] = max(d["pi_ours"], float(np.abs(pi - pi_ex).max()))
            d["invariant"] = max(d["invariant"], np.abs(pi @ t.transition - pi).max())
            if np.any((qt.P == 0) & (P_ex > 1e-300)):
                zeroed.add((rho, n))
            d["P_qe"] = max(d["P_qe"], rel(qt.P, P_ex))
            if not qt.is_irreducible:
                qe_reducible.add((rho, n))
            else:
                d["pi_qe"] = max(d["pi_qe"], float(np.abs(qt.stationary_distributions[0] - pi_ex).max()))

        p = r.stationary
        d["invariant"] = max(d["invariant"], np.abs(p @ r.transition - p).max())
        d["r_pi"] = max(d["r_pi"], np.abs(p - qr.stationary_distributions[0]).max())
        d["binomial"] = max(d["binomial"], np.abs(p - binom.pmf(np.arange(n), n - 1, 0.5)).max())
        var = sigma ** 2 / (1 - rho ** 2)
        d["moments"] = max(d["moments"], abs(r.mean - mean) / scale, abs(r.variance / var - 1),
                           abs(r.autocorrelation() - rho), abs(r.autocorrelation(4) - rho ** 4))

    print("\ntauchen against QuantEcon")
    print(f"  nodes {d['t_nodes']:.1e}   transition {d['t_P']:.1e} (absolute)")
    print(f"tauchen against 40 digits: {len(exact)} distinct chains (the probabilities depend on "
          f"rho and n only); {len(reducible)} reducible in double precision for both packages, "
          f"left out")
    print(f"  transition probabilities, relative:  dsgelab {d['P_ours']:.1e}   "
          f"QuantEcon {d['P_qe']:.1e}, with probabilities rounded to zero in {len(zeroed)} chains")
    print(f"  stationary distribution:             dsgelab {d['pi_ours']:.1e}   "
          f"QuantEcon {d['pi_qe']:.1e}, and {len(qe_reducible)} chains it finds reducible")
    print("rouwenhorst against QuantEcon")
    print(f"  nodes {d['r_nodes']:.1e}   transition {d['r_P']:.1e}   "
          f"stationary {d['r_pi']:.1e}")
    print("rouwenhorst against the closed form")
    print(f"  binomial stationary distribution {d['binomial']:.1e}   mean, variance and "
          f"autocorrelations of the AR(1) {d['moments']:.1e}")
    print(f"invariance pi P = pi, every chain {d['invariant']:.1e}")
    return bool(ok and reducible
                and d["t_nodes"] < 1e-14 and d["t_P"] < 1e-14
                and d["P_ours"] < 1e-10 and d["pi_ours"] < 1e-13
                and d["r_nodes"] < 1e-14 and d["r_P"] < 1e-14 and d["r_pi"] < 1e-13
                and d["binomial"] < 1e-13 and d["moments"] < 1e-10 and d["invariant"] < 1e-15)


if __name__ == "__main__":
    ok = main()
    print("\n" + ("OK" if ok else "DIVERGE"))
    sys.exit(0 if ok else 1)
