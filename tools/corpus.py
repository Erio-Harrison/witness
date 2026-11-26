"""The code every test and benchmark cuts its prompts from: a pinned release of ripgrep.

tools/fetch_corpus.sh downloads it to data/corpus/ripgrep. The corpus is the Rust sources of two of
its crates (searcher and ignore, about 450 KB), in sorted path order, so that everybody who fetches
the same release gets the same prompts. ripgrep is dual-licensed MIT / Unlicense; it is downloaded,
not copied into this repository.

WITNESS_CORPUS=DIR uses every *.rs file under DIR instead (the numbers are then not comparable with
the recorded ones).
"""
import glob
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CRATES = ["searcher", "ignore"]


def files():
    custom = os.environ.get("WITNESS_CORPUS")
    if custom:
        found = sorted(glob.glob(os.path.join(custom, "**", "*.rs"), recursive=True))
    else:
        base = ROOT / "data" / "corpus" / "ripgrep" / "crates"
        found = [f for c in CRATES for f in sorted(glob.glob(str(base / c / "src" / "**" / "*.rs"), recursive=True))]
    if not found:
        raise SystemExit("no corpus: run tools/fetch_corpus.sh (or set WITNESS_CORPUS to a directory of .rs files)")
    return found


def text():
    return "\n".join(open(f, encoding="utf-8").read() for f in files())


def model_dir():
    """The HF snapshot directory of the model: WITNESS_MODEL, or the line in spikes/baseline/model_path.txt."""
    path = os.environ.get("WITNESS_MODEL")
    if path:
        return path
    note = ROOT / "spikes" / "baseline" / "model_path.txt"
    if not note.exists():
        raise SystemExit(f"set WITNESS_MODEL, or write the model directory into {note}")
    return note.read_text().strip()
