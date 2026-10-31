#!/usr/bin/env bash
# setup.sh: creates validation/dolo/.venv, a Python 3.11 environment with dolo 0.4.9.20 and the
# versions in requirements.txt, and downloads the three dolo models that compare_time_iteration.py
# solves (compare_value_iteration.py solves one of them, rbc_mc), from dolo's v0.4.9.20 tag,
# checking their sha256 sums.
#
# Usage:  bash validation/dolo/setup.sh
#
# It needs uv (https://docs.astral.sh/uv/), which also fetches Python 3.11 if it is missing.
set -euo pipefail
cd "$(dirname "$0")"

uv venv -q --allow-existing --python 3.11 .venv
uv pip install -q --python .venv/bin/python -r requirements.txt
.venv/bin/python -c "from importlib.metadata import version; print('dolo', version('dolo'))"

mkdir -p models
get() {   # get <file> <sha256>
  curl -fsSL --retry 5 --retry-all-errors --retry-delay 3 -o "models/$1" \
    "https://raw.githubusercontent.com/EconForge/dolo/v0.4.9.20/examples/models/$1"
  echo "$2  models/$1" | shasum -a 256 -c -
}
get rbc.yaml 690e3bf96873b8d0238eb1c20b3a7786f096c7cc8d7d54fddc4847f935fc3f03
get rbc_mc.yaml d09c07e564b5613b8b47028062951f327d0186f73db80c6e12f37881f071a614
get consumption_savings_iid.yaml a85a2257285335b2b0f2a9a5c8967d0e85a3fdce0473948a6e02b246948d25ed
