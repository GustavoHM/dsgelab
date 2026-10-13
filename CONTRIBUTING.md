# Contributing to dsgelab

Thank you for considering it. Bug reports, failing models, and comparisons with another tool are
as welcome as code.

## What the package promises, and what that asks of a contribution

Every number `dsgelab` produces has been checked against an independent reference: another
implementation (Dynare, MacroModelling.jl, gEconpy, dolo, sequence-jacobian), a closed-form
solution, or a published result. A feature without that check would weaken the promise for the
whole package, so a change that adds or alters a result is merged only when it meets the same
six conditions the [roadmap](ROADMAP.md#o-que-conta-como-pronto) sets for its own items:

1. It is implemented with a public API listed in `__all__`.
2. It is **validated against an independent reference**, in a script anyone can run. Scripts that
   need Dynare, Octave or Julia live in [`validation/`](validation/README.md); the closed forms
   and anything that runs on numpy and scipy alone go in the test suite.
3. It has tests in `tests/` that need no external tool.
4. It is documented in [`docs/api.md`](docs/api.md), with a runnable example where it helps;
   examples on the site are executed by `tools/check_site_examples.py`.
5. The matching row of [`docs/comparison.md`](docs/comparison.md) is updated.
6. It has an entry in [`CHANGELOG.md`](CHANGELOG.md).

A bug fix needs a test that fails before the fix and passes after it. If you are unsure how to
validate something, open an issue first; finding the right reference is often most of the work.

## Reporting a problem

The most useful report is a model that misbehaves: the smallest model that shows it, what
`dsgelab` returns, and what you expected, ideally with the number another tool gives for the same
model. A `.mod` file is fine; `dsgelab.from_mod` reads it. Please include the output of

```
python -c "import dsgelab, numpy, scipy, sys; print(dsgelab.__version__, numpy.__version__, scipy.__version__, sys.version)"
```

## Development set-up

```
git clone https://github.com/GustavoHM/dsgelab && cd dsgelab
python -m venv .venv && . .venv/bin/activate
pip install -e ".[symbolic,test]"
pytest
python tools/check_site_examples.py
```

The suite runs against `src/` by default. `DSGELAB_TEST_INSTALLED=1 pytest` runs it against the
installed package instead, which is how CI tests the wheel on Linux, macOS and Windows, Python
3.11 to 3.14.

The external validation runs with `bash validation/run_all.sh`; it needs GNU Octave with the
packages that `validation/setup_octave_packages.sh` installs, Dynare, and (for one comparison)
Julia. It also runs weekly in CI.

## Style

- The core depends on numpy and scipy only. Anything heavier goes behind an optional extra.
- Variables are addressed by name, never by position in a vector.
- Errors are raised, not reported: a solver that cannot meet its conditions (Blanchard–Kahn, a
  steady state that does not zero the residuals) raises rather than returning something that
  means nothing.
- Code, docstrings and the API reference are in English. Match the surrounding code: its
  comment density, naming, and idiom.

## Versions

Semantic versioning, with the public API being what `__all__` lists. Before 1.0 a minor version
may still change the API, but every removal or incompatible change is preceded by a deprecation
warning that lasts at least one minor version.

## Conduct

Participation is governed by the [code of conduct](CODE_OF_CONDUCT.md).
