#!/bin/bash
# Downloads the code the tests and benchmarks cut their prompts from (see tools/corpus.py):
# ripgrep 14.1.1, checked against the commit the recorded numbers were measured with.
set -eu
cd "$(dirname "$0")/.."
TAG=14.1.1
COMMIT=4649aa9700619f94cf9c66876e9549d83420e16c
DIR=data/corpus/ripgrep
if [ ! -d "$DIR/.git" ]; then
    mkdir -p data/corpus
    git -c advice.detachedHead=false clone -q --depth 1 --branch "$TAG" https://github.com/BurntSushi/ripgrep "$DIR"
fi
HEAD=$(git -C "$DIR" rev-parse HEAD)
[ "$HEAD" = "$COMMIT" ] || { echo "$DIR is at $HEAD, expected $COMMIT" >&2; exit 1; }
echo "corpus ready: $DIR ($TAG)"
