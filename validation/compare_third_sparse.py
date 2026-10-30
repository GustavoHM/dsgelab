"""Closed-form and full-pipeline scalability checks of sparse third order."""
import argparse
import resource
import sys
import time

import numpy as np

import dsgelab as dsg
from dsgelab.secondorder import _solve_sylvester
from dsgelab.tensors import SparseThirdDerivatives
from dsgelab.thirdorder import _sparse_sylvester_residual


def _peak_mb():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e6 if sys.platform == "darwin" else peak / 1024


def _core(n, nk):
    nc = nk * (nk + 1) * (nk + 2) // 6
    rng = np.random.default_rng(20261025)
    Abar = -np.eye(n)
    Bbar = 0.01 * np.eye(n)
    P = 0.5 * np.eye(nk)
    R = rng.normal(size=(n, nc))
    started = time.perf_counter()
    W = _solve_sylvester(Abar, Bbar, P, R, direct_limit=0, order=3,
                         sparse=True, canonical_rhs=True)
    residual = _sparse_sylvester_residual(W, Abar, Bbar, P, -R, canonical=True)
    elapsed = time.perf_counter() - started
    expected = R / (-1.0 + 0.01 * 0.5 ** 3)
    error = float(np.max(np.abs(W.to_canonical() - expected)))
    passed = residual < 1e-10 and error < 1e-10
    print(f"core: equations={n}, states={nk}, canonical columns={nc}, nonzeros={W.nnz}")
    print(f"residual={residual:.3e}, closed-form error={error:.3e}")
    print(f"elapsed={elapsed:.2f}s, peak RSS={_peak_mb():.1f}MB: {'OK' if passed else 'FAIL'}")
    return passed


def _full(n, nk):
    nu, ne = n - nk, min(nk, 5)
    states = [f"x{i}" for i in range(nk)]
    jumps = [f"y{i}" for i in range(nu)]
    shocks = [f"e{i}" for i in range(ne)]

    def equations(nxt, cur, eps, params):
        out = []
        for i in range(nk):
            innovation = getattr(eps, shocks[i]) if i < ne else 0.0
            cubic = (getattr(cur, states[i]) * getattr(cur, states[(i + 1) % nk])
                     * getattr(cur, states[(i + 2) % nk]))
            out.append(getattr(nxt, states[i]) - 0.5 * getattr(cur, states[i])
                       - 0.01 * cubic - innovation)
        for j in range(nu):
            out.append(getattr(nxt, jumps[j]) - 2.0 * getattr(cur, jumps[j])
                       + 0.1 * getattr(cur, states[j % nk]))
        return out

    model = dsg.Model(states=states, jumps=jumps, shocks=shocks, equations=equations,
                      name="sparse-third-benchmark")
    guess = {name: 0.0 for name in model.names}
    started = time.perf_counter()
    first = dsg.solve(model, {}, guess=guess)
    m = 2 * n + ne
    hessian = np.zeros((n, m, m))
    second = dsg.solve_second_order(first, cov={name: 0.01 for name in shocks}, hessian=hessian)
    equations_index, indices, values = [], [], []
    for i in range(nk):
        equations_index.append(i)
        indices.append(sorted(n + (i + offset) % nk for offset in range(3)))
        values.append(-0.01)
    third_derivatives = SparseThirdDerivatives(
        equations_index, indices, values, (n, m, m, m), symmetric=True)
    third = dsg.solve_third_order(second, third=third_derivatives, hessian=hessian,
                                  sparse=True, direct_limit=0)
    elapsed = time.perf_counter() - started
    magnitude = max(float(np.max(np.abs(third.g_kkk.values))),
                    float(np.max(np.abs(third.h_kkk.values))))
    residual = max(third.sylvester_residual, third.risk_slope_residual,
                   third.risk_state_residual, third.risk_shock_residual,
                   third.innovation_residual)
    passed = residual < 1e-8 and magnitude > 1e-8
    print(f"full pipeline: equations={n}, states={nk}, jumps={nu}, shocks={ne}")
    print(f"model third-derivative nonzeros={third_derivatives.nnz}, "
          f"solution nonzeros={third.g_kkk.nnz + third.h_kkk.nnz}")
    print(f"maximum residual={residual:.3e}, maximum |cubic|={magnitude:.3e}")
    print(f"elapsed={elapsed:.2f}s, peak RSS={_peak_mb():.1f}MB: {'OK' if passed else 'FAIL'}")
    return passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--equations", type=int, default=71)
    parser.add_argument("--states", type=int, default=35)
    parser.add_argument("--full", action="store_true", help="run the full nonlinear pipeline")
    args = parser.parse_args()
    n, nk = args.equations, args.states
    if n < 2 or nk < 1 or nk >= n:
        parser.error("require equations >= 2 and 0 < states < equations")
    passed = _full(n, nk) if args.full else _core(n, nk)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
