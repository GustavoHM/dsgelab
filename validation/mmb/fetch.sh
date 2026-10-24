#!/usr/bin/env bash
# fetch.sh: downloads the central-bank models of the Macroeconomic Model Data Base to mods/.
#
# The files come from Rep-MMB, the replication toolbox of the MMB (macromodelbase.com), at a fixed
# commit of github.com/AlexMeyer-Gohde/Rep-MMB, and each one is checked against its SHA-256. They
# are GPL-3.0 and are not under version control here, for the same reason as ../fetch.sh.
set -euo pipefail
cd "$(dirname "$0")"

COMMIT=5db355464d23a56968e4c4d5164ab7290e9e26db   # 30 January 2025
BASE="https://raw.githubusercontent.com/AlexMeyer-Gohde/Rep-MMB/$COMMIT/Rep-MMB"

# model  sha256 of <model>_rep.mod
MODELS="
BRA_SAMBA08  74a5c2dae055e46c766091818906b98a186d215e880a8cbe45facaa662d2c1ff
CA_LS07      4b6760e8427dd04e73b61ccdebadab6e9dffe8355cf9fba72a4c956e8c1072ca
CL_MS07      00aceed3ba4ea74f314f8ad21e19120047274069d3305b77a2d6a662c9601017
EACZ_GEM03   e623ae1a2a4ee9ebd7a878655bfb06d4c9a396ff19cd54ebb354d67fffef4833
EAUS_NAWM08  271e0bbc1fd7edf0cd81cee44d87904d0d21e46df9a14afd83f754dd7f6d88cb
EA_ALSV06    731ab6783f45455e2bd48ee992b39561a4cc87022b86a82f6b0812a7ddf0a92b
EA_SWW14     128f72c62d5629ae7b2022280737adfe583660a5de6bce6b9c8e6b9570c42a86
HK_FP13      98915655d1bdf338ffa0091632217fb8e498946d3b4a824804ec57f70ae3d710
UK_SM11      348ca8056fa18a58aea649de827cbf9f1c3d55adb0b5e0ab126a8ad54c110ea0
US_DNGS15    8ada225b83f0c8227cc56bd334c573b9e5bd9d168862df7d1a56e210cb0e42cb
US_FRB03     dd4db1a940ff219682c288d5da0082194adc054dc0faa6f1ba689b2688d3c4fd
US_FRB08     24332e8fe65e495604d3f66f8ba34de619346fbe3375a08d41cc0613babc4c76
US_SW07      84e8c3c0933a1075a049137b7958f5206f6802a982384edce5e329a53c62cf7e
"

mkdir -p mods
failed=0
while read -r m sha; do
  [ -n "$m" ] || continue
  if curl -fsSL --retry 5 --retry-all-errors --retry-delay 3 -o "mods/$m.mod" "$BASE/$m/${m}_rep.mod" \
     && echo "$sha  mods/$m.mod" | shasum -a 256 -c --status -; then
    echo "ok  mods/$m.mod"
  else
    echo "FAILED  mods/$m.mod"
    rm -f "mods/$m.mod"
    failed=1
  fi
done <<< "$MODELS"
exit $failed
