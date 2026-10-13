#!/usr/bin/env bash
# fetch.sh — baixa os arquivos MATLAB de Heutel (2012) para original/.
#
# A cópia pública está no repositório institucional da Georgia State University
# (scholarworks.gsu.edu, item 9a01aa68-89c0-4e66-b7bd-9f09606d259c). Os arquivos continuam
# atribuídos a Garth Heutel e não são versionados aqui. Os seis .m são idênticos, byte a byte, aos
# que a validação usou (source-sha256.json); o ReadMe.txt público é uma versão anterior, de março
# de 2011, que descreve só os arquivos MATLAB.
set -euo pipefail
cd "$(dirname "$0")"

URL=https://scholarworks.gsu.edu/server/api/core/bitstreams/9be302dc-8a22-440f-b0e8-ad22a5022b85/content
SHA256=7ea9b108c7d8b393c8214d4a958c9b7de5b3151180cd3e85ab65d28cd0a5c123

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
curl -fsSL --retry 5 --retry-delay 3 -o "$tmp/WebFiles.zip" "$URL"
echo "$SHA256  $tmp/WebFiles.zip" | shasum -a 256 -c -
mkdir -p original
unzip -oq "$tmp/WebFiles.zip" -d original
ls original
