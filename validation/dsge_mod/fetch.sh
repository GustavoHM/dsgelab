#!/usr/bin/env bash
# fetch.sh: downloads selected models from Johannes Pfeifer's DSGE_mod collection.
#
# The source is pinned by commit and every downloaded file is checked by SHA-256. The .mod files
# are GPL-3.0 and are not versioned in this MIT package.
set -euo pipefail
cd "$(dirname "$0")"

COMMIT=26c21f79e4c6a6b19b18e6330edc5b23489907a4
BASE="https://raw.githubusercontent.com/JohannesPfeifer/DSGE_mod/$COMMIT"

# destination|source path|sha256
FILES="
Hansen_1985.mod|Hansen_1985/Hansen_1985.mod|e2ce4e5faf0ce7392299c5636fe6222576fb8ca8c75b2799a8937011c642dcc3
Aguiar_Gopinath_2007.mod|Aguiar_Gopinath_2007/Aguiar_Gopinath_2007.mod|4e16b74227230481b6fe92715d8555c919962cf97bdb9ede4aa55dea4aec7d07
Ghironi_Melitz_2005.mod|Ghironi_Melitz_2005/Ghironi_Melitz_2005.mod|8f32fb6a1cf6c9374213c45c58abebb09d5c5130feed8d22c8000deac4113d2b
Ghironi_Melitz_2005_steadystate.m|Ghironi_Melitz_2005/Ghironi_Melitz_2005_steadystate.m|d38dd5e52da0cdd498fff80dd34123212c194b0d8ada2b6dd4eec1a5dfe9171a
Smets_Wouters_2007_45.mod|Smets_Wouters_2007/Smets_Wouters_2007_45.mod|dd1c0498262cd546404e9bf7a2d533ba5d8044a792626269aa4bad935258efb4
"

mkdir -p mods
failed=0
while IFS='|' read -r dst src sha; do
  [ -n "$dst" ] || continue
  if curl -fsSL --retry 5 --retry-all-errors --retry-delay 3 -o "mods/$dst" "$BASE/$src" \
     && echo "$sha  mods/$dst" | shasum -a 256 -c --status -; then
    echo "ok  mods/$dst"
  else
    echo "FAILED  mods/$dst"
    rm -f "mods/$dst"
    failed=1
  fi
done <<< "$FILES"
exit $failed
