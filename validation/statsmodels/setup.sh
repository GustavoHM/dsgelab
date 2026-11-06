#!/usr/bin/env bash
# setup.sh: creates validation/statsmodels/.venv, a Python 3.12 environment with statsmodels 0.14.4
# and the versions in requirements.txt, in which run_statsmodels.py evaluates the reference
# likelihoods and regime probabilities of compare_kim.py.
#
# Usage:  bash validation/statsmodels/setup.sh
#
# It needs uv (https://docs.astral.sh/uv/), which also fetches Python 3.12 if it is missing.
set -euo pipefail
cd "$(dirname "$0")"

uv venv -q --allow-existing --python 3.12 .venv
uv pip install -q --python .venv/bin/python -r requirements.txt
.venv/bin/python -c "from importlib.metadata import version; print('statsmodels', version('statsmodels'))"
