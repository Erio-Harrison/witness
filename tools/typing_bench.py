"""First-token latency while typing: the cache-hit case of code completion, over HTTP.

usage: python tools/typing_bench.py URL witness|openai LABEL [OUT.jsonl]

A fill-in-the-middle prompt, <|fim_prefix|> prefix <|fim_suffix|> suffix <|fim_middle|>, is sent
11 times: once cold, then 10 times with the prefix 3 tokens longer each time and the suffix unchanged,
as an editor does while the user types. Everything up to the old end of the prefix can come from a
prefix cache; the new tokens and the whole suffix cannot. Reports the cold request and the median
first-token time of the 10 typed ones, for several prefix and suffix lengths.

`witness` sends prompt + suffix to /v1/completions (the server builds the FIM prompt); `openai` sends
the same prompt as one string with the FIM tokens written out, which HF-tokenizer servers parse into
the same token ids. The prompt token counts the servers report are printed so that this can be seen.
"""
import corpus
import json
import statistics
import sys
import time
import urllib.request

from transformers import AutoTokenizer

MODEL = corpus.model_dir()
SCENARIOS = [(1500, 0), (1500, 64), (1500, 300), (3000, 300), (1500, 800)]  # prefix, suffix tokens
STEPS, GROW, GEN = 10, 3, 8

url, style, label = sys.argv[1], sys.argv[2], sys.argv[3]
out_path = sys.argv[4] if len(sys.argv) > 4 else None
tok = AutoTokenizer.from_pretrained(MODEL)
src = corpus.text()
ids = tok(src).input_ids
name = json.loads(urllib.request.urlopen(url + "/v1/models", timeout=30).read())["data"][0]["id"]


def body(prefix, suffix, **extra):
    b = {"model": name, "max_tokens": GEN, "temperature": 0, **extra}
    if style == "witness":
        b.update(prompt=prefix, suffix=suffix)
    else:
        b.update(prompt="<|fim_prefix|>" + prefix + "<|fim_suffix|>" + suffix + "<|fim_middle|>")
    return json.dumps(b).encode()


def post(data):
    return urllib.request.urlopen(urllib.request.Request(url + "/v1/completions", data, {"Content-Type": "application/json"}), timeout=300)


def cold(prefix, suffix):
    """Whole response time of a 1-token request, and the prompt tokens the server counted."""
    t = time.perf_counter()
    r = json.loads(post(body(prefix, suffix, max_tokens=1)).read())
    return (time.perf_counter() - t) * 1e3, r["usage"]["prompt_tokens"]


def first_token_ms(prefix, suffix):
    t = time.perf_counter()
    with post(body(prefix, suffix, stream=True)) as r:
        for line in r:
            if line.startswith(b"data: {") and json.loads(line[6:])["choices"][0].get("text"):
                ms = (time.perf_counter() - t) * 1e3
                r.read()
                return ms
    return (time.perf_counter() - t) * 1e3  # no text at all (the model stopped at once)


print(f"{label}: prefix grows by {GROW} tokens per request, {STEPS} requests, {GEN} tokens generated")
for n, (p, s) in enumerate(SCENARIOS):
    a = 5000 + 9000 * n                       # a different part of the code for every scenario
    suffix = tok.decode(ids[a + p + 200:a + p + 200 + s])
    cold_ms, prompt_tokens = cold(tok.decode(ids[a:a + p]), suffix)
    typed = [first_token_ms(tok.decode(ids[a:a + p + GROW * k]), suffix) for k in range(1, STEPS + 1)]
    rec = {"label": label, "prefix": p, "suffix": s, "prompt_tokens": prompt_tokens, "cold_ms": cold_ms,
           "typed_ms_median": statistics.median(typed), "typed_ms_max": max(typed), "typed_ms_min": min(typed)}
    print(f"  prefix {p:4} suffix {s:3} ({prompt_tokens} prompt tokens): cold {cold_ms:7.1f} ms, "
          f"typing median {rec['typed_ms_median']:6.1f} ms (min {rec['typed_ms_min']:.1f}, max {rec['typed_ms_max']:.1f})", flush=True)
    if out_path:
        with open(out_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
