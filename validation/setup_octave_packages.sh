#!/usr/bin/env bash
# setup_octave_packages.sh: installs in Octave the packages that the run_*.sh scripts load
# (`pkg load statistics datatypes`), in the versions with which the published validation was made.
#
# Usage:  bash setup_octave_packages.sh
#
# `pkg install -forge` would bring the newest version. Here the versions, the URLs and the sha256
# sums are those of the official index (github.com/gnu-octave/packages), and a file that does not
# match is refused.
set -euo pipefail

OCTAVE="${OCTAVE:-$(command -v octave || echo /opt/homebrew/bin/octave)}"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

install() {   # install <name> <version> <url> <sha256>
  local f="$tmp/$1-$2.tar.gz"
  curl -fsSL --retry 5 --retry-delay 3 -o "$f" "$3"
  echo "$4  $f" | shasum -a 256 -c -
  "$OCTAVE" --no-gui --eval "pkg install '$f'"
}

install datatypes 1.5.0 \
  https://github.com/pr0m1th3as/datatypes/releases/download/release-1.5.0/datatypes-1.5.0.tar.gz \
  5d4eb5efe22a1388d7f0a4337539fd4d8a79f33d40fb37a23c4da0bd7d3de32b
install statistics 2.0.0 \
  https://github.com/gnu-octave/statistics/releases/download/release-2.0.0/statistics-2.0.0.tar.gz \
  e82c1d6957885ee444adc91e35defd00665ec3c3e1eeae0bd39f89b7b28a6490

"$OCTAVE" --no-gui --eval "pkg list"
