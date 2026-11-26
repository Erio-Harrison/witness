"""Checks of the witness server from outside, over HTTP.

usage: python tools/server_test.py tokenize URL MODEL_DIR     tokenizer agrees with HF's
       python tools/server_test.py infill URL OUT.json        run the /infill cases, save the answers
       python tools/server_test.py compare A.json B.json      compare two saved runs (e.g. llama-server's)
       python tools/server_test.py trace URL                  ground-truth trace and injected waits

The /infill cases are cut from the corpus (tools/corpus.py): a cursor in the middle of a file, with and
without extra context, greedy sampling.
"""
import corpus
import json
import sys
import time
import urllib.request

FILES = corpus.files()


def post(url, path, body, raw=False):
    req = urllib.request.Request(url + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read()
    return data if raw else json.loads(data)


def get(url, path):
    with urllib.request.urlopen(url + path, timeout=120) as r:
        return r.read().decode()


def cases():
    out = []
    for i, (f, line, extra) in enumerate([(FILES[0], 40, 0), (FILES[1], 80, 2), (FILES[2], 25, 0), (FILES[3], 120, 3)]):
        lines = open(f, encoding="utf-8").read().split("\n")
        line = min(line, len(lines) - 5)
        cur = lines[line]
        cut = len(cur) // 2
        body = {
            "input_prefix": "\n".join(lines[:line]) + "\n",
            "prompt": cur[:cut],
            "input_suffix": cur[cut:] + "\n" + "\n".join(lines[line + 1:line + 40]) + "\n",
            "input_extra": [{"filename": FILES[4 + k], "text": open(FILES[4 + k], encoding="utf-8").read()[:1500]} for k in range(extra)],
            "n_predict": 32, "n_indent": 0, "t_max_predict_ms": 0, "cache_prompt": True, "stream": False,
            "temperature": 0, "top_k": 1, "samplers": ["top_k"],
        }
        out.append((f"case {i} ({f.split('/')[-1]}:{line}, {extra} extra files)", body))
    return out


def cmd_tokenize(url, model_dir):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    texts = [open(f, encoding="utf-8").read() for f in FILES[:12]]
    texts += [open(corpus.ROOT / "tools" / "zh_sample.md", encoding="utf-8").read(), "", " ", "\n\n\t  x", "émoji 🙂 ok", "a" * 300]
    bad = 0
    for t in texts:
        ours = post(url, "/tokenize", {"content": t})["tokens"]
        ref = tok(t, add_special_tokens=False).input_ids
        if ours != ref:
            bad += 1
            k = next((i for i, (a, b) in enumerate(zip(ours, ref)) if a != b), min(len(ours), len(ref)))
            print(f"MISMATCH at token {k}: ours {ours[k:k+5]} ref {ref[k:k+5]} in {t[:40]!r}")
        elif post(url, "/detokenize", {"tokens": ours})["content"] != t:
            bad += 1
            print(f"ROUND TRIP differs for {t[:40]!r}")
    special = "x <|im_end|> y"
    ours = post(url, "/tokenize", {"content": special})["tokens"]
    as_text = tok.convert_tokens_to_ids("<|im_end|>") not in ours
    print(f"{len(texts)} texts, {sum(len(tok(t).input_ids) for t in texts)} tokens: {bad} mismatches; "
          f"'<|im_end|>' in client text is {'plain text' if as_text else 'A CONTROL TOKEN (bad)'}")
    return bad == 0 and as_text


def cmd_infill(url, out_path):
    results = []
    for name, body in cases():
        t = time.perf_counter()
        r = post(url, "/infill", body)
        ms = (time.perf_counter() - t) * 1e3
        results.append({"name": name, "content": r["content"], "tokens_evaluated": r.get("tokens_evaluated"),
                        "tokens_predicted": r.get("tokens_predicted"), "stop_type": r.get("stop_type"),
                        "cache_n": r.get("timings", {}).get("cache_n"), "ms": ms})
        print(f"{name}: {r.get('tokens_evaluated')} prompt tokens ({r.get('timings', {}).get('cache_n')} cached), "
              f"{r.get('tokens_predicted')} generated in {ms:.0f} ms: {r['content'][:70]!r}")
    # the same cursor again, a few characters later: the prefix should come from the cache
    name, body = cases()[1]
    body = dict(body, prompt=body["prompt"] + body["input_suffix"][:3], input_suffix=body["input_suffix"][3:])
    t = time.perf_counter()
    r = post(url, "/infill", body)
    print(f"typed 3 more characters at case 1: {r['timings']['cache_n']} of {r['tokens_evaluated']} prompt tokens cached, "
          f"{(time.perf_counter() - t) * 1e3:.0f} ms")
    r0 = post(url, "/infill", dict(cases()[2][1], n_predict=0))
    print(f"n_predict 0 (cache warm-up): content {r0['content']!r}, {r0['tokens_predicted']} generated")
    json.dump(results, open(out_path, "w"), indent=1, ensure_ascii=False)
    return True


def cmd_compare(a_path, b_path):
    a, b = json.load(open(a_path)), json.load(open(b_path))
    ok = True
    for x, y in zip(a, b):
        same_prompt = x["tokens_evaluated"] == y["tokens_evaluated"]
        n = next((i for i, (p, q) in enumerate(zip(x["content"], y["content"])) if p != q), min(len(x["content"]), len(y["content"])))
        same_text = x["content"] == y["content"]
        ok &= same_prompt
        print(f"{x['name']}: prompt tokens {x['tokens_evaluated']} vs {y['tokens_evaluated']} {'ok' if same_prompt else 'DIFFER'}; "
              f"content {'identical' if same_text else f'same for {n} characters, then {x['content'][n:n+25]!r} vs {y['content'][n:n+25]!r}'}")
    return ok


def stream(url, body):
    """One streamed /v1/completions; returns the arrival time of every event with text, and the total."""
    req = urllib.request.Request(url + "/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0, stamps = time.perf_counter(), []
    with urllib.request.urlopen(req, timeout=120) as r:
        for line in r:
            if line.startswith(b"data: {") and json.loads(line[6:])["choices"][0]["text"]:
                stamps.append(time.perf_counter() - t0)
    return stamps, time.perf_counter() - t0


def cmd_trace(url):
    prompt = open(FILES[0], encoding="utf-8").read()[:3000]
    body = {"prompt": prompt, "max_tokens": 24, "stream": True, "ignore_eos": True}
    ok = True
    off = {"gpu_thread_spin_us": 0, "gpu_thread_sleep_us": 0, "socket_delay_us": 0, "coalesce_tokens": 0, "cuda_delay_us": 0}
    post(url, "/inject", off)
    post(url, "/trace", {"enabled": False})
    get(url, "/trace")
    for _ in range(8):   # the first requests after start-up are slower: the GPU's clocks are still coming up
        stream(url, body)
    # One request's time moves by 20 ms or so from run to run (GPU clocks; with speculative decoding
    # the 24 tokens take about 10 steps), so the injected waits add up to several times that and both
    # sides of the comparison are medians of several requests.
    runs = sorted((stream(url, body) for _ in range(5)), key=lambda r: r[1])
    base_stamps, base_total = runs[2]
    ok &= get(url, "/trace").strip() == ""
    print(f"trace off: nothing recorded; 24 tokens in {base_total * 1e3:.0f} ms, {len(base_stamps)} events with text")

    post(url, "/trace", {"enabled": True})
    stamps, total = stream(url, body)
    events = [json.loads(l) for l in get(url, "/trace").splitlines()]
    host = [e for e in events if e["src"] == "host"]
    cuda = [e for e in events if e["src"] == "cuda"]
    kinds = {}
    for e in host:
        kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
    sites = {}
    for e in cuda:
        s = sites.setdefault(e["site"], [0, 0, 0])
        s[0] += 1
        s[1] += e["t1"] - e["t0"]
        s[2] += e["cpu_ns"]
    print(f"trace on: {len(host)} host events {kinds}")
    for name, (n, wall, cpu) in sorted(sites.items(), key=lambda kv: -kv[1][1])[:6]:
        print(f"    {name}: {n} calls, {wall / 1e6:.1f} ms wall, {cpu / 1e6:.1f} ms CPU")
    tokens = [e for e in host if e["kind"] == "token"]
    writes = [e for e in host if e["kind"] == "socket_write" and e["b"] > 0]
    ok &= len(tokens) == 24 and sum(e["b"] for e in writes) == 24
    print(f"    {len(tokens)} token events, {sum(e['b'] for e in writes)} tokens in {len(writes)} socket writes; "
          f"tokens are written {max(w['t0'] - t['t0'] for t, w in zip(tokens, writes)) / 1e3:.0f} us after they exist at most")
    order = all(a["t0"] <= b["t0"] for a, b in zip(events, events[1:]))
    ok &= order
    print(f"    events in time order: {order}; decode units: {kinds.get('decode', 0)}, prefill units: {kinds.get('prefill', 0)}")

    def with_injection(label, setting, expect_ms, count):
        nonlocal ok
        post(url, "/inject", dict(off, **setting))
        totals = [stream(url, body)[1] for _ in range(2)]
        get(url, "/trace")
        s, total = stream(url, body)
        ev = [json.loads(l) for l in get(url, "/trace").splitlines()]
        total = sorted(totals + [total])[1]
        post(url, "/inject", off)
        extra = (total - base_total) * 1e3
        seen = count(ev)
        # Not simply additive: with pauses between its units the power-capped GPU runs each unit faster
        # (a decode step takes 17 instead of 21 ms), so the total may grow by much less than was injected.
        good = 0.3 * expect_ms - 30 < extra < 1.35 * expect_ms + 30
        ok &= good
        print(f"injected {label}: total +{extra:.0f} ms (expected about +{expect_ms:.0f}), {seen}; {len(s)} events with text {'ok' if good else 'UNEXPECTED'}")

    units = kinds.get("decode", 0) + kinds.get("prefill", 0)
    with_injection("10 ms spin before each GPU unit", {"gpu_thread_spin_us": 10000}, 10 * units,
                   lambda ev: f"{sum(e.get('kind') == 'injected_spin' for e in ev)} spins recorded")
    with_injection("10 ms sleep before each GPU unit", {"gpu_thread_sleep_us": 10000}, 10 * units,
                   lambda ev: f"{sum(e.get('kind') == 'injected_sleep' for e in ev)} sleeps recorded")
    with_injection("3 ms before each socket write", {"socket_delay_us": 3000}, 0,
                   lambda ev: f"{sum(e.get('kind') == 'injected_socket_delay' for e in ev)} delays recorded (they fall on the HTTP thread, so the total barely moves)")
    with_injection("5 ms inside every cudaStreamSynchronize", {"cuda_delay_us": 5000, "cuda_site": "cudaStreamSynchronize"},
                   5 * sum(1 for e in cuda if e["site"] == "cudaStreamSynchronize"),
                   lambda ev: f"{sum(e.get('injected_us', 0) > 0 for e in ev)} CUDA calls carry the injected wait")
    post(url, "/inject", dict(off, coalesce_tokens=8))
    s, _ = stream(url, body)
    post(url, "/inject", off)
    post(url, "/trace", {"enabled": False})
    good = len(s) == 3
    ok &= good
    print(f"coalescing 8 tokens per write: {len(s)} events with text for 24 tokens {'ok' if good else 'UNEXPECTED'}")
    return ok


if __name__ == "__main__":
    cmd = sys.argv[1]
    ok = {"tokenize": cmd_tokenize, "infill": cmd_infill, "compare": cmd_compare, "trace": cmd_trace}[cmd](*sys.argv[2:])
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
