"""Compare one witness run with the SGLang and vLLM runs: python compare.py WITNESS SGLANG VLLM (labels)."""
import json, sys
W, S, V = sys.argv[1:4]
rows = [json.loads(l) for l in open('results.jsonl')]
d = {(r['label'], r['prompt_tokens'], r['conc']): r for r in rows}
wt = wd = 0
for n in (128, 1024, 2048, 4096):
    for c in (1, 2, 3, 4):
        x = [d[(l, n, c)] for l in (W, S, V)]
        bt = min(x[1]['ttft_ms_median'], x[2]['ttft_ms_median']); bd = max(x[1]['decode_tok_s_per_user'], x[2]['decode_tok_s_per_user'])
        t, k = x[0]['ttft_ms_median'], x[0]['decode_tok_s_per_user']
        wt += t < bt; wd += k > bd
        print(f"{n}x{c} | " + " | ".join(f"{r['ttft_ms_median']:.0f} / {r['decode_tok_s_per_user']:.1f}" for r in x) + f" | ttft {100*(t/bt-1):+.0f}% dec {100*(k/bd-1):+.0f}%")
print(f"witness ahead of the better framework: TTFT {wt}/16, decode {wd}/16")
ty = {}
for l in open('typing.jsonl'):
    r = json.loads(l); ty[(r['label'], r['prefix'], r['suffix'])] = r
wy = 0
for p, s in [(1500, 0), (1500, 64), (1500, 300), (3000, 300), (1500, 800)]:
    x = [ty.get((l, p, s)) for l in (W, S, V)]
    if not all(x): continue
    b = min(x[1]['typed_ms_median'], x[2]['typed_ms_median']); wy += x[0]['typed_ms_median'] < b
    print(f"typing {p}/{s} | " + " | ".join(f"{r['typed_ms_median']:.1f}" for r in x) + f" | {100*(x[0]['typed_ms_median']/b-1):+.0f}%")
print(f"typing: witness ahead in {wy}/5; total {wt + wd + wy}/37")
