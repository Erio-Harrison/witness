"""Baseline TTFT / decode speed of an OpenAI-compatible server (llama-server, vLLM) for code prompts.

usage: python bench.py PORT LABEL [--lens 128,1024,2048,4096] [--conc 1,2,3,4] [--gen 128]
Prompts are cut from the corpus (tools/corpus.py) to exact token counts with the model's tokenizer; every
request starts at a different offset so no two share a prefix (prefix caching is measured apart).
Decode speed is generated tokens (from the server's usage report, else --gen) per second between the
first and last streamed event. Appends one JSON line per (len, conc) to results.jsonl next to this file.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))
import corpus  # noqa: E402
import http.client
import json
import statistics
import threading
import time
from pathlib import Path

from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
MODEL = "Qwen/Qwen2.5-Coder-1.5B"

ap = argparse.ArgumentParser()
ap.add_argument("port", type=int)
ap.add_argument("label")
ap.add_argument("--lens", default="128,1024,2048,4096")
ap.add_argument("--conc", default="1,2,3,4")
ap.add_argument("--gen", type=int, default=128)
ap.add_argument("--reps", type=int, default=3)
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(MODEL)
src = corpus.text()
ids = tok(src).input_ids
offset = [0]


def prompt(n):
    o = offset[0]
    offset[0] = (o + 7919) % (len(ids) - n - 1)
    return tok.decode(ids[o:o + n])


def model_name():
    c = http.client.HTTPConnection("127.0.0.1", a.port, timeout=30)
    c.request("GET", "/v1/models")
    return json.loads(c.getresponse().read())["data"][0]["id"]


NAME = model_name()


def one(p, out, i):
    body = json.dumps({"model": NAME, "prompt": p, "max_tokens": a.gen, "min_tokens": a.gen, "ignore_eos": True,
                       "temperature": 0, "stream": True, "stream_options": {"include_usage": True}})
    c = http.client.HTTPConnection("127.0.0.1", a.port, timeout=600)
    t0 = time.perf_counter()
    c.request("POST", "/v1/completions", body, {"Content-Type": "application/json"})
    r = c.getresponse()
    stamps, buf, tokens = [], b"", a.gen
    while True:
        d = r.read1(65536)
        if not d:
            break
        now = time.perf_counter()
        buf += d
        while b"\n\n" in buf:
            ev, buf = buf.split(b"\n\n", 1)
            ev = ev.strip()
            if ev.startswith(b"data:") and ev != b"data: [DONE]":
                j = json.loads(ev[5:])
                if any(ch.get("text") for ch in j.get("choices", [])):
                    stamps.append(now)
                if (j.get("usage") or {}).get("completion_tokens"):
                    tokens = j["usage"]["completion_tokens"]
    # Tokens per second from the first event with text to the last. Tokens, not events: a server
    # that decodes speculatively sends several tokens in one event.
    out[i] = (stamps[0] - t0, (tokens - 1) / (stamps[-1] - stamps[0]) if len(stamps) > 1 else 0, len(stamps))


# warm-up (graphs, allocator, first-use kernels)
one(prompt(256), [None], 0)
for n in [int(x) for x in a.lens.split(",")]:
    for c in [int(x) for x in a.conc.split(",")]:
        ttft, tps = [], []
        for _ in range(a.reps):
            out = [None] * c
            ths = [threading.Thread(target=one, args=(prompt(n), out, i)) for i in range(c)]
            for t in ths:
                t.start()
            for t in ths:
                t.join()
            ttft += [o[0] * 1e3 for o in out]
            tps += [o[1] for o in out]
        rec = {"label": a.label, "prompt_tokens": n, "conc": c, "gen": a.gen,
               "ttft_ms_median": statistics.median(ttft), "ttft_ms_max": max(ttft),
               "decode_tok_s_per_user": statistics.median(tps), "decode_tok_s_total": statistics.median(tps) * c}
        print(f"{a.label:<10} prompt={n:<5} conc={c}  TTFT median {rec['ttft_ms_median']:7.1f} ms (max {rec['ttft_ms_max']:7.1f})"
              f"  decode {rec['decode_tok_s_per_user']:6.1f} tok/s/user  {rec['decode_tok_s_total']:6.1f} total", flush=True)
        with open(HERE / "results.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
