import os
import pathlib
import sys

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"

# By default the suite runs against the checkout. With DSGELAB_TEST_INSTALLED=1
# it runs against the installed package instead, which is how CI tests the
# wheel: a file left out of the wheel then fails here rather than on a user's
# machine.
if os.environ.get("DSGELAB_TEST_INSTALLED"):
    import dsgelab

    if SRC in pathlib.Path(dsgelab.__file__).resolve().parents:
        raise RuntimeError(
            f"DSGELAB_TEST_INSTALLED is set but dsgelab was imported from {dsgelab.__file__}"
        )
else:
    sys.path.insert(0, str(SRC))
