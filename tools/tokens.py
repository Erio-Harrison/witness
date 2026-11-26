"""Token ids of real code for the scheduler checks and timings, which run without a tokenizer.

usage: python tools/tokens.py MODEL_DIR OUT_DIR

Writes OUT_DIR/code.i32 (the corpus of tools/corpus.py tokenized, little-endian i32) and
OUT_DIR/special.i32: the ids of <|fim_prefix|>, <|fim_suffix|>, <|fim_middle|>, <|endoftext|>,
<|im_end|>, <|fim_pad|>, <|repo_name|>, <|file_sep|>, in that order.
"""
import corpus
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(sys.argv[1])
out = Path(sys.argv[2])
src = corpus.text()
ids = tok(src).input_ids
np.array(ids, "<i4").tofile(out / "code.i32")
names = ["<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>", "<|endoftext|>", "<|im_end|>", "<|fim_pad|>", "<|repo_name|>", "<|file_sep|>"]
special = [tok.convert_tokens_to_ids(n) for n in names]
assert all(isinstance(i, int) and i >= 0 for i in special), special
np.array(special, "<i4").tofile(out / "special.i32")
print(f"{len(ids)} code tokens; special ids {dict(zip(names, special))}")
