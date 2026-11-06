#!/usr/bin/env bash
# setup.sh: creates validation/tasmanian/.venv, a Python 3.12 environment with the versions in
# requirements.txt, and builds Tasmanian 8.2 there from its source on PyPI, the reference of
# compare_smolyak.py. The build needs a C++ compiler (on macOS, the Command Line Tools); CMake comes
# from requirements.txt. On 3 November 2026 it took a minute and a half on macOS 27 (arm64) with
# Apple clang 21.
#
# Usage:  bash validation/tasmanian/setup.sh
#
# It needs uv (https://docs.astral.sh/uv/), which also fetches Python 3.12 if it is missing.
set -euo pipefail
cd "$(dirname "$0")"

uv venv -q --allow-existing --python 3.12 .venv
uv pip install -q --python .venv/bin/python -r requirements.txt
# scikit-build calls cmake by name, so the environment's comes first on the PATH. Build isolation
# would fetch the build tools again, unpinned and without cmake.
PATH="$PWD/.venv/bin:$PATH" uv pip install -q --python .venv/bin/python --no-build-isolation \
  "tasmanian==8.2"
.venv/bin/python -c "from importlib.metadata import version; import Tasmanian; \
print('Tasmanian', version('tasmanian'))"
