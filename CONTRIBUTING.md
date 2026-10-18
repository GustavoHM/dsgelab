# Contributing to dsgelab

Thank you for considering a contribution. Bug reports, failing models and comparisons with
another tool are as welcome as code.

## What the package promises, and what that asks of a contribution

Each feature of `dsgelab` is validated against an independent reference where one exists,
which may be another implementation (Dynare, MacroModelling.jl), a closed-form solution or a
published result, and a few rest on internal tests only. Because a new feature without such a
check would weaken that standard for the whole package, a change that adds or alters a result is
merged only when it meets the six conditions the [roadmap](ROADMAP.md#what-counts-as-done) sets
for its own items.

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
validate something, open an issue first, where the choice of reference can be discussed.

## Reporting a problem

Reports go to the [issue tracker](https://github.com/GustavoHM/dsgelab/issues). A report is
most useful when it contains the smallest model that shows the problem, what `dsgelab` returns
and what you expected, ideally with the number another tool gives for the same model. A `.mod`
file is fine, since `dsgelab.from_mod` reads it. Please include the output of

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

The external validation runs with `bash validation/run_all.sh`. It needs GNU Octave with the
packages that `validation/setup_octave_packages.sh` installs, Dynare and, for one comparison,
Julia. It also runs weekly in CI.

## Style

- The core depends on numpy and scipy only. Any other dependency goes behind an optional extra.
- Variables are addressed by name, since reading them by position in a vector lets a swapped
  index pass without an error.
- A solver that cannot meet its conditions (Blanchard–Kahn, a steady state that does not zero
  the residuals) raises an exception instead of returning a result.
- Code, docstrings and the API reference are in English. New code follows the comment density,
  naming and idiom of the code around it.

## Versions

Versions follow semantic versioning, and the public API is what `__all__` lists. Before 1.0 a
minor version may still change the API, but every removal or incompatible change is preceded by a
deprecation warning that lasts at least one minor version.

## Conduct

Participation is governed by the [code of conduct](CODE_OF_CONDUCT.md).
