#!/bin/bash
# One step from nothing to weights the server can load: downloads Qwen2.5-Coder-1.5B from Hugging
# Face (a pinned revision, checked by SHA-256) and converts it to witness's format.
#
#   tools/get_model.sh [PYTHON]        needs curl, sha256sum and a Python with numpy (default: python3)
#
# Leaves   data/Qwen2.5-Coder-1.5B/   the files of the Hugging Face repository          (3.1 GB)
#          data/qwen2.5-coder-1.5b.wit                                               (3.1 GB)
# and prints the command that starts the server. Safe to run again: finished files are kept and an
# interrupted download is resumed. The model is Alibaba's, under Apache-2.0; nothing of it is in
# this repository.
set -eu
cd "$(dirname "$0")/.."
PY=${1:-python3}
REPO=Qwen/Qwen2.5-Coder-1.5B
REV=df3ce67c0e24480f20468b6ef2894622d69eb73b
DIR=data/Qwen2.5-Coder-1.5B
OUT=data/qwen2.5-coder-1.5b.wit

"$PY" -c "import numpy" 2>/dev/null || { echo "$PY has no numpy: pip install numpy, or pass another Python as the first argument" >&2; exit 1; }
mkdir -p "$DIR"
fetch() {   # FILE SHA256
    if [ -f "$DIR/$1" ] && echo "$2  $DIR/$1" | sha256sum -c --status; then echo "have $1"; return; fi
    echo "downloading $1"
    curl -fL --retry 3 -C - --progress-bar -o "$DIR/$1" "https://huggingface.co/$REPO/resolve/$REV/$1"
    echo "$2  $DIR/$1" | sha256sum -c --status || { echo "$DIR/$1: checksum mismatch; delete it and run again" >&2; exit 1; }
}
fetch config.json       984abd4b3f21edb957012c7dc4352c462b08e15ffe8384b6d55990885e9dcc9a
fetch tokenizer.json    c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539
# (the next four are only read by the Python tools, which load the tokenizer and the model with transformers)
fetch tokenizer_config.json 4a917e4772d9b246ef600e1c5d7d81e3a81ae9f3f2d10d78ccb02c86e5fb184a
fetch vocab.json ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910
fetch merges.txt 599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3
fetch generation_config.json 444ea57ebf361c659d1f5c23508319b691e181d122588ea3cdd5f68d3e618b1f
fetch model.safetensors 535485b26038ac24f4e8a86e1a73ed417d8992e3192f6305b7fcc4fb92efa210

WIT_SHA=e60ddd7e4963a7482edf1a8216c17ea1f44673bb731069a983589ab4dbb8d97d
if [ -f "$OUT" ] && echo "$WIT_SHA  $OUT" | sha256sum -c --status; then
    echo "have $OUT"
else
    "$PY" tools/convert.py "$DIR" "$OUT"
    echo "$WIT_SHA  $OUT" | sha256sum -c --status || { echo "$OUT: not the expected file" >&2; exit 1; }
fi
echo
echo "ready. Start the server with:"
echo "  ./target/release/server $OUT $DIR/tokenizer.json --port 8012"
