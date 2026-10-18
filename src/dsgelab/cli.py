"""Command line entry point: a smoke test of the installed package."""

from __future__ import annotations

import argparse
import sys


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="dsgelab", description=__doc__)
    parser.add_argument("--example", default="rbc", choices=("rbc", "nk"),
                        help="which bundled example to solve")
    parser.add_argument("--periods", type=int, default=12, help="impulse-response length")
    args = parser.parse_args(argv)

    import dsgelab as dsg
    from dsgelab.examples import nk, rbc

    build = {"rbc": rbc.build, "nk": nk.build}[args.example]
    model, params, guess = build()
    sol = dsg.solve(model, params, guess=guess)
    print(sol.summary())

    acc = sol.accuracy()
    if acc.get("exact"):
        print("  first-order accuracy  : exact -- the model is linear, so there is "
              "nothing for the approximation to miss")
    else:
        print(f"  first-order accuracy  : implied order "
              f"{', '.join(f'{o:.2f}' for o in acc['implied_order'])} (2 is correct)")

    shock = model.shocks[0]
    irf = sol.irf(shock, size=0.01, periods=args.periods)
    print(f"\nimpulse response to {shock} (per cent deviations):")
    header = "  t   " + "".join(f"{n:>12}" for n in model.names)
    print(header)
    for t in range(min(args.periods, 8)):
        row = "".join(f"{irf[n][t]:>12.5f}" for n in model.names)
        print(f"  {t:<4}{row}")
    return 0


if __name__ == "__main__":       # pragma: no cover
    sys.exit(main())
