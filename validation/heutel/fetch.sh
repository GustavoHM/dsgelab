#!/usr/bin/env bash
# fetch.sh: downloads Heutel's (2012) MATLAB files to original/.
#
# The public copy is in Georgia State University's institutional repository
# (scholarworks.gsu.edu, item 9a01aa68-89c0-4e66-b7bd-9f09606d259c). The files remain attributed
# to Garth Heutel and are not under version control here. The six .m files are identical, byte for
# byte, to the ones the validation used (source-sha256.json). The public ReadMe.txt is an earlier
# version, from March 2011, that describes only the MATLAB files.
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
