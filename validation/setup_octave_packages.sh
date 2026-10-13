#!/usr/bin/env bash
# setup_octave_packages.sh — instala no Octave os pacotes que os run_*.sh carregam
# (`pkg load statistics datatypes`), nas versões com que a validação publicada foi feita.
#
# Uso:  bash setup_octave_packages.sh
#
# `pkg install -forge` traria a versão mais nova; aqui as versões, as URLs e os sha256 são os do
# índice oficial (github.com/gnu-octave/packages), e um arquivo que não bate é recusado.
set -euo pipefail

OCTAVE="${OCTAVE:-$(command -v octave || echo /opt/homebrew/bin/octave)}"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

install() {   # install <nome> <versão> <url> <sha256>
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
