"""Do two servers give the same greedy completion for the same prompt?

usage: python tools/agree.py run URL OUT.json [--n 24] [--gen 48]     ask one server, save its answers
       python tools/agree.py compare A.json B.json [C.json ...]        compare saved answers pairwise

Prompts are cut from the corpus (tools/corpus.py) at 128, 512 and 1024 tokens, a third each. Requests
go one at a time to /v1/completions with temperature 0. Engines that run the same weights round at
different points, so where the model's two best candidates are nearly level they may pick differently;
this measures how often that happens between any two of them.
"""
import json
import sys
import urllib.request

import corpus


def run(url, out, n, gen):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(corpus.model_dir())
    ids = tok(corpus.text()).input_ids
    model = json.load(urllib.request.urlopen(url + "/v1/models", timeout=30))["data"][0]["id"]
    answers = []
    for i in range(n):
        length = [128, 512, 1024][i % 3]
        start = (i * 4001) % (len(ids) - length - 1)
        prompt = tok.decode(ids[start:start + length])
        body = {"model": model, "prompt": prompt, "max_tokens": gen, "temperature": 0}
        req = urllib.request.Request(url + "/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
        text = json.load(urllib.request.urlopen(req, timeout=120))["choices"][0]["text"]
        answers.append({"length": length, "start": start, "tokens": tok(text, add_special_tokens=False).input_ids})
    json.dump(answers, open(out, "w"))
    print(f"{url}: {n} completions saved to {out}")


def compare(paths):
    runs = {p: json.load(open(p)) for p in paths}
    for i, a in enumerate(paths):
        for b in paths[i + 1:]:
            same, firsts = 0, []
            for x, y in zip(runs[a], runs[b]):
                assert (x["length"], x["start"]) == (y["length"], y["start"])
                k = next((j for j, (s, t) in enumerate(zip(x["tokens"], y["tokens"])) if s != t), None)
                if k is None and len(x["tokens"]) == len(y["tokens"]):
                    same += 1
                else:
                    firsts.append(k if k is not None else min(len(x["tokens"]), len(y["tokens"])))
            where = f"; the others first differ at token {sorted(firsts)}" if firsts else ""
            print(f"{a} vs {b}: {same} of {len(runs[a])} completions identical{where}")


if __name__ == "__main__":
    if sys.argv[1] == "run":
        opt = lambda name, default: int(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default
        run(sys.argv[2], sys.argv[3], opt("--n", 24), opt("--gen", 48))
    else:
        compare(sys.argv[2:])
