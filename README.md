# witness

An inference engine written for exactly one model, one GPU and one job: **Qwen2.5-Coder-1.5B** (bf16, not quantized) on an **RTX 4060 Laptop**, serving **local code completion**.

General frameworks have to work for every model on every device. witness goes the other way: it narrows the scope as far as it will go, to see how much faster than a general framework one can get on a single combination while staying as close to the original model.

The host side is Rust; the GPU side is hand-written CUDA kernels plus cuBLAS. About 4,700 lines in total.

> **Status: experimental.** Every number here comes from one machine. The code builds and runs on other GPUs (sm_89 or newer), but the tuning was done for this one card and nothing says the lead holds elsewhere.

## Results

Compared with vLLM and SGLang on the same machine, in the same session, with the same weights, the same precision and the same benchmark script. All three run with n-gram (prompt lookup) speculative decoding enabled. The GPU is power-capped at 30 W, which is this laptop's everyday state.

Prompts are cut from a public corpus (the Rust sources of ripgrep 14.1.1, fetched by `tools/fetch_corpus.sh`), so the runs can be repeated.

There are 37 items. In each one witness is compared with **the better of the two frameworks**. witness was run twice:

| Metric | Items where witness leads |
|---|---|
| Time to first token (4 prompt lengths × 1–4 concurrent requests) | 15 / 16 |
| Generation speed per user (same grid) | 6 / 16 |
| Time to first token while typing (prefix cache hit) | 3–4 / 5 |
| Total | **24–25 / 37** |

The short version: **time to first token leads almost everywhere; generation speed does not.**

Single user (median time to first token / generation speed; first witness run):

| Prompt | witness | SGLang | vLLM |
|---|---|---|---|
| 128 tokens | 37 ms / 215 tok/s | 54 ms / 81 tok/s | 50 ms / 163 tok/s |
| 1024 tokens | 195 ms / 107 tok/s | 221 ms / 79 tok/s | 216 ms / 94 tok/s |
| 2048 tokens | 379 ms / 89 tok/s | 423 ms / 86 tok/s | 425 ms / 91 tok/s |
| 4096 tokens | 812 ms / 89 tok/s | 864 ms / 131 tok/s | 876 ms / 95 tok/s |

With 2–4 concurrent requests, time to first token is 1%–38% ahead of the better framework, except at 4096 × 4, which is a tie (2122–2131 ms against 2114 ms).

Typing: an editor resends the request every few keystrokes, and most of the prefix is already cached. Here the prefix grows by 3 tokens per request:

| Prefix / suffix (tokens) | witness | SGLang | vLLM |
|---|---|---|---|
| 1500 / 0 | 25.6 ms | 43.7 ms | 48.8 ms |
| 1500 / 64 | 38.3 ms | 54.5 ms | 47.3 ms |
| 1500 / 300 | 88.7 ms | 99.8 ms | 102.9 ms |
| 3000 / 300 | 95.8 ms | 106.0 ms | 93.4 ms |
| 1500 / 800 | 195.5 ms | 195.8 ms | 206.5 ms |

The full 16-cell table and every earlier comparison are written up in [spikes/baseline/NOTES.md](spikes/baseline/NOTES.md) (section 13 is the current result). Per-run logs are not in the repository; the scripts in `spikes/baseline/` reproduce them.

### What to keep in mind when reading these numbers

- **Generation speed is not a lead.** witness is ahead in 6 of 16 cells and is often 5%–25% behind vLLM under concurrency. Part of that is a scheduling choice: witness serves new prompts first, which is why its time to first token is better in those same cells.
- **Everything above about 55 tok/s comes from speculative decoding.** A single decode step is bound by memory bandwidth; the floor on this machine is 18.3 ms per step, and all three engines sit close to it. How many guesses are accepted depends on how repetitive the text is, so per-cell generation speed varies a lot, and with only 5–20 requests per cell a single cell is not stable: the same cell differs by up to 30% between the two witness runs.
- **One machine, 30 W cap.** The memory clock of this GPU moves between two levels with temperature, so numbers from different sessions cannot be compared. Every comparison here ran all three engines back to back in one session.
- The frameworks' speculative decoding was not tuned in depth: vLLM used 7 draft tokens with lookup lengths 1–8, SGLang its defaults.
- llama.cpp and TensorRT-LLM were only measured early on, without speculative decoding (NOTES section 1b). They are not part of the comparison above.

## Is the output the same as the original model's?

Not bit for bit. What is measured, after every change, on three reference cases (Rust code, Chinese prose, a fill-in-the-middle prompt; 128–233 tokens each):

**Exact configuration** (fp32 KV cache, scalar kernels) against Hugging Face transformers running the same weights in fp32:

| Check | Result |
|---|---|
| Every intermediate tensor of every layer | relative error about 1e-5 |
| Argmax at every position | no difference |

This is what shows the implementation computes the right function.

**Normal configuration** (bf16 KV cache, tensor-core kernels, the one that is benchmarked), measured against the fp32 model, with Hugging Face's own bf16 run as the yardstick:

| Distance from the fp32 model | witness | HF transformers in bf16 |
|---|---|---|
| Mean KL divergence of the next-token distribution | 1.2e-4 – 2.1e-4 | 4.6e-4 – 1.9e-3 |
| Positions where the top token differs | 0 – 3 | 3 – 7 |

So witness is closer to the fp32 model than the reference bf16 run is. The check fails if that stops being true for any case.

**Against HF bf16 token by token:** greedy generation departs from HF's bf16 output in all three cases (at generated token 0, 5 and 9). At each of those positions HF bf16's own top two candidates are 0.125 apart, the smallest step bf16 can represent at that magnitude. Two bf16 implementations that round at different points pick differently there, and from then on the texts differ.

**Against the other engines:** the same 24 prompts (128–1024 tokens), 48 tokens each, greedy, sent twice to each server (`tools/agree.py`):

| Pair | Identical completions out of 24 |
|---|---|
| witness, run 1 vs run 2 | 24 |
| witness, speculative decoding on vs off | 24 |
| vLLM, run 1 vs run 2 | 20 |
| SGLang, run 1 vs run 2 | 19 |
| witness vs vLLM | 16–17 |
| witness vs SGLang | 18–22 |
| vLLM vs SGLang | 17–18 |

No two engines agree on every completion, and vLLM and SGLang do not reproduce their own output from one run to the next. witness disagrees with each of them about as often as they disagree with each other, and is the only one of the three that is deterministic.

Other checks:

| Check | Result |
|---|---|
| Speculative decoding on vs off | identical tokens |
| `/infill` against llama.cpp's `llama-server` | identical text on 4 cases (measured on an earlier corpus; not repeated on the current one) |
| Tokenizer against HF | identical on 18 texts (67,520 tokens) |

Limits of this verification: the comparison with HF covers sequences up to 233 tokens. Longer contexts are covered only by consistency checks between witness's own code paths (chunked prefill against single-step decode), not by a direct comparison with HF.

## Why it is faster

No single reason; a chain of choices specific to this model and this card.

**Prompt processing (prefill)**
- Attention scores, softmax and the weighted sum run in one kernel that drives the tensor cores through inline PTX. Scores never leave the registers.
- cuBLAS picks its matrix-multiply kernel by a heuristic, and on this card the pick is not the fastest one. After timing every configuration cuBLAS offers, the engine asks for a faster one with a bit-identical result: 5%–7% overall.
- The largest product (the MLP) is computed in 8 slices so that the activation that follows reads data still in cache.
- The attention kernel lays its staged data out to avoid shared-memory bank conflicts, which made attention 24% faster.

**Decode steps**
- A whole step is recorded as one CUDA graph, removing a hundred kernel launches per token.
- Matrix-vector products use the tensor cores, so a step with 24 rows costs only 8% more than a step with 1. That is what makes verifying many speculative guesses nearly free.

**Caching and scheduling**
- Prefix cache: a new request is matched against cached KV by longest common prefix, and only the rest is computed.
- Speculative decoding: guesses are looked up in the text so far and verified in the same step as the regular token; wrong ones are discarded.

Measurements for each step, and the attempts that did not work, are in [spikes/baseline/NOTES.md](spikes/baseline/NOTES.md).

## Getting started

### Requirements

- An NVIDIA GPU with compute capability 8.9 or newer (RTX 40 series) and 8 GB of memory. About 4–5 GB is used at run time.
- CUDA Toolkit (developed with 13.0), with `nvcc` under `$CUDA_HOME/bin` (default `/usr/local/cuda`).
- Rust (developed with 1.100).
- Python with `numpy`, to convert the weights. The correctness checks and benchmarks additionally need `torch` and `transformers`; running the server needs neither.
- Developed on WSL2 under Windows 11. Native Linux should work and has not been tested.

### Build

```bash
git clone https://github.com/Erio-Harrison/witness
cd witness
cargo build --release
```

`build.rs` calls `nvcc` to compile `core/` into a static library.

### Get the model

```bash
tools/get_model.sh
```

This downloads Qwen2.5-Coder-1.5B from Hugging Face (a pinned revision, verified by SHA-256) into `data/` and converts it to witness's format. It needs `curl` and about 6.2 GB of disk, and takes a few minutes; it can be interrupted and run again. The weights are not part of this repository.

To convert a checkpoint you already have: `python tools/convert.py /path/to/Qwen2.5-Coder-1.5B data/qwen2.5-coder-1.5b.wit`.

### Start the server

```bash
./target/release/server data/qwen2.5-coder-1.5b.wit data/Qwen2.5-Coder-1.5B/tokenizer.json --port 8012
```

| Flag | Default | Meaning |
|---|---|---|
| `--port` | 8012 | Port to listen on |
| `--host` | 127.0.0.1 | Address to listen on |
| `--ctx` | 8192 | Largest context of a request, in tokens |
| `--slots` | 8 | KV cache slots: more slots remember more prefixes and use more GPU memory |
| `--chunk` | 1024 | Tokens per prefill chunk; the tuning assumes 1024 |
| `--no-speculate` | | Turn speculative decoding off |
| `--trace` | | Start with the ground-truth trace enabled |

### Try it

Fill in the middle (code before and after the cursor; the model writes what goes between):

```bash
curl -s http://127.0.0.1:8012/infill -H 'Content-Type: application/json' -d '{
  "input_prefix": "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n    ",
  "input_suffix": "\n\nprint(fibonacci(10))\n",
  "n_predict": 48
}'
```

OpenAI-compatible completion:

```bash
curl -s http://127.0.0.1:8012/v1/completions -H 'Content-Type: application/json' -d '{
  "prompt": "// Rust: reverse a string\nfn reverse(s: &str) -> String {\n",
  "max_tokens": 40
}'
```

### In an editor

`/infill` is compatible with llama.cpp's `llama-server`, so the [llama.vscode](https://github.com/ggml-org/llama.vscode) extension can be pointed at `http://127.0.0.1:8012`. The endpoint's output has been compared with `llama-server`'s, but **the extension itself has not been run against witness in VS Code yet**.

## API

| Path | Description |
|---|---|
| `POST /infill` | llama-server's fill-in-the-middle endpoint |
| `POST /v1/completions` | OpenAI-compatible; with `suffix` it is treated as fill-in-the-middle; supports `stream` |
| `POST /tokenize`, `POST /detokenize` | Tokenizer |
| `GET /health`, `GET /props`, `GET /v1/models` | Status and model information |
| `GET /trace`, `POST /trace` | Read the ground-truth trace; turn it on or off |
| `POST /inject` | Inject waits (see below) |

## Known limitations

- **Greedy decoding only.** No temperature, top-p or other sampling; the same input always gives the same output.
- **Qwen2.5 architecture only.** Model dimensions are read from the weight file, so the 0.5B model of the same architecture runs (it was used for debugging). Other models do not.
- **Chunks that are not 1024 rows are slower.** The remainder of a prompt after full chunks, and long suffixes, cost 5%–20% more per token, because kernel selection was tuned only for 768 and 1024 rows.
- **No multi-GPU, no quantization, no offline batch inference.**
- The tuning table (`core/lt_table.inc`) is tied to this GPU and this cuBLAS version. Where an entry is not available the engine falls back to cuBLAS's default choice: results stay correct, speed may not. `spikes/gemm/lt.cu` redoes the search.

## Ground-truth trace and injection

witness has a second purpose: to be a test subject for performance attribution tools. For that it can record exactly what it did.

- With the trace on, it records how long every CUDA call took and how much CPU it used, and when each token was produced and written to the socket. With the trace off there is no overhead.
- Waits can be injected on demand (spinning or sleeping on the GPU thread, socket delays, coalescing tokens, delays inside CUDA calls) to see whether an attribution tool finds them.

See `tools/server_test.py trace` and `src/trace.rs`.

## Layout

| Path | Contents |
|---|---|
| `core/engine.cu` | The engine: weight loading, reference kernels, the wide (prefill) path, the step path, CUDA graphs, tracing |
| `core/flash.cuh` | Fused attention on the tensor cores (prefill and decode) |
| `core/tc.cuh`, `core/fast.cuh` | Tensor-core and scalar kernels of a decode step |
| `core/lt_table.inc` | cuBLAS kernel selection table |
| `core/witness_core.h` | The C interface |
| `src/sched.rs` | Scheduler: prefix cache, speculative decoding, queuing of prefill and decode |
| `src/http.rs`, `src/bin/server.rs` | HTTP server |
| `src/bin/check.rs`, `src/bin/sched.rs` | Correctness checks |
| `src/bin/bench.rs` | In-process timing |
| `tools/` | Model download and conversion, HF reference data, HTTP tests, the typing benchmark |
| `spikes/` | Early feasibility experiments, the scripts that run the other frameworks, and the notes |

## Running the checks and benchmarks

```bash
tools/fetch_corpus.sh                          # ripgrep 14.1.1, into data/corpus/
export WITNESS_MODEL=data/Qwen2.5-Coder-1.5B   # what tools/get_model.sh downloaded

# HF reference data, then the engine checks (needs torch)
python tools/ref_dump.py $WITNESS_MODEL data/ref-1.5b
./target/release/check data/qwen2.5-coder-1.5b.wit data/ref-1.5b

# scheduler, prefix cache and speculative decoding
python tools/tokens.py $WITNESS_MODEL data
./target/release/sched check data/qwen2.5-coder-1.5b.wit data

# in-process timing
./target/release/bench data/qwen2.5-coder-1.5b.wit data/ref-1.5b/fim/tokens.i32
```

The comparison with other frameworks uses the scripts in `spikes/baseline/` (`run_witness.sh`, `run_baseline.sh`, `compare.py`). They expect each framework in its own virtual environment; read them before running.

## License

MIT, see [LICENSE](LICENSE). The model is not part of this repository: Qwen2.5-Coder-1.5B is Alibaba Cloud's, under Apache-2.0, and `tools/get_model.sh` downloads it from Hugging Face. The test corpus (ripgrep, MIT / Unlicense) is downloaded as well.

## Acknowledgements

- [Qwen2.5-Coder](https://github.com/QwenLM/Qwen2.5-Coder): the model.
- [llama.cpp](https://github.com/ggml-org/llama.cpp): the `/infill` prompt format is a port of its implementation.
- [ripgrep](https://github.com/BurntSushi/ripgrep): the test corpus.
- [vLLM](https://github.com/vllm-project/vllm), [SGLang](https://github.com/sgl-project/sglang), [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM): the frameworks compared against. They solve a far harder, general problem; the comparison here only shows what specializing for one narrow combination buys.
