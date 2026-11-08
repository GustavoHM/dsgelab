#!/usr/bin/env bash
# setup.sh: creates validation/sequence_jacobian/.venv, a Python 3.11 environment with
# sequence-jacobian 1.0.0 and the versions in requirements.txt, and downloads the four notebooks
# whose models compare_heterogeneous.py solves, krusell_smith.ipynb, hank.ipynb,
# labor_search.ipynb and two_asset.ipynb, from sequence-jacobian's v1.0.0 tag, checking their
# sha256 sums.
#
# Usage:  bash validation/sequence_jacobian/setup.sh
#
# It needs uv (https://docs.astral.sh/uv/), which also fetches Python 3.11 if it is missing.
set -euo pipefail
cd "$(dirname "$0")"

uv venv -q --allow-existing --python 3.11 .venv
uv pip install -q --python .venv/bin/python -r requirements.txt
.venv/bin/python -c "from importlib.metadata import version; print('sequence-jacobian', version('sequence-jacobian'))"

mkdir -p notebooks
get() {   # get <file> <sha256>
  curl -fsSL --retry 5 --retry-all-errors --retry-delay 3 -o "notebooks/$1" \
    "https://raw.githubusercontent.com/shade-econ/sequence-jacobian/v1.0.0/notebooks/$1"
  echo "$2  notebooks/$1" | shasum -a 256 -c -
}
get krusell_smith.ipynb ddcde2d99edfc20d4183ea750dd39782ee507cea869f0040665ebaff8e359cf3
get hank.ipynb cd517a24fca328eab8969477a04da27530bb6298f29cdde7cf2aa75d8217e2ff
get labor_search.ipynb 7e6c980181b91b17f108a0be41400a5e0f57cda8b1776a71b8cabc595dd5457c
get two_asset.ipynb 493b3ac6ee73fb8dcc867a8846e82e9cc8a2c482e0f9a650f2834a68dde3756d
