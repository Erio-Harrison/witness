# Spike 1: baseline for Qwen2.5(-Coder)-1.5B on the RTX 4060 Laptop

Machine: RTX 4060 Laptop (sm_89, 24 SMs, 8 GB, ~6.8 GiB usable), measured read bandwidth ~250 GB/s
(`~/gpu-spike/bench/bw.cu`), dense GEMM bf16 ~24 TFLOPS, fp8 ~45 TFLOPS (`~/gpu-spike/bench/tflops.py`).
Model: Qwen/Qwen2.5-Coder-1.5B (base, for FIM), same architecture as Qwen2.5-1.5B:
hidden 1536, 28 layers, 12 q heads / 2 kv heads x 128, MLP 8960, vocab 151936, tied embeddings.
bf16 weights 3.09 GB -> ideal decode step 12.35 ms at 250 GB/s (~81 tok/s).

## 3. GEMV bandwidth at 1.5B shapes (`gemv.cu`, 2026-11-26) — zero weights; see section 4

Simple kernel: one warp per output row, 16-byte streaming loads, fp32 accumulate. Weights cycled over
>= 512 MB so the 32 MB L2 cannot serve them.

| shape (N x K) | B=1 ours / cuBLAS GB/s | B=4 ours / cuBLAS |
|---|---|---|
| qkv 2048 x 1536 (6.3 MB) | 150 / 148 | 150 / 145 |
| o 1536 x 1536 (4.7 MB) | 148 / 143 | 165 / 133 |
| gate_up 17920 x 1536 (55 MB) | 211 / 211 | 209 / 210 |
| down 1536 x 8960 (27.5 MB) | 205 / 201 | 203 / 206 |
| lm_head 151936 x 1536 (467 MB) | 217 / 217 | 215 / 215 |

Whole step, all 28 layers + lm_head with distinct weights, one CUDA graph:

| batch | ours | cuBLAS |
|---|---|---|
| 1 | 14.38 ms, 215 GB/s, 69.6 tok/s | 15.63 ms, 198 GB/s, 64.0 tok/s |
| 2 | 14.44 ms -> 138.5 tok/s total | 14.67 ms |
| 3 | 14.53 ms -> 206.4 tok/s total | 14.76 ms |
| 4 | 14.75 ms -> 271.3 tok/s total | 14.73 ms |

Findings:
- A plain GEMV already reaches 86% of measured bandwidth over a whole step (215 of 250 GB/s);
  the matmul-only floor for bf16 B=1 is ~14.4 ms (~70 tok/s). Ideal is 12.35 ms.
- The loss is in the small per-layer matrices (qkv, o: ~150 GB/s, 4-6 MB each, too small to fill
  24 SMs / hide the tail). Fusing qkv+o work or overlapping consecutive matmuls is where the last
  ~2 ms are.
- Batch 2-4 is nearly free: step time 14.38 -> 14.75 ms at B=4, so 4 concurrent users get ~68 tok/s
  each (271 tok/s total). Concurrency 2-4 costs almost nothing in decode.
- Not yet included: attention over the KV cache, norms, RoPE, SiLU, sampling, launch gaps between
  non-matmul kernels. Those decide how close a real engine gets to this floor.

## 4. The decode floor depends on the weight VALUES (2026-11-26) — corrects section 3

Section 3's whole-step numbers (14.4 ms, 215 GB/s, "batch 2-4 is nearly free") were taken with
all-zero weights (`cudaMemset(..., 0, ...)`) over 20 steps. `spikes/decode/gemv_rand.cu` is the same
chain with a step count and a fill argument (`gemv_rand STEPS FILL`, FILL < 0 = random bf16 values
of weight-like size). Same kernel, same shapes, same session, memory clock 5501 MHz, 30 W cap:

| weights | steps | 1 row | 4 rows |
|---|---|---|---|
| zero | 20 | 14.4-14.9 ms | 14.6 ms |
| zero | 200-300 | 14.4-14.6 ms | 17.9-18.2 ms |
| constant 0x55 | 200-400 | 14.5-15.1 ms | 18.4-18.5 ms |
| random | 200-300 | 18.30 ms (169 GB/s, 54.6 tok/s) | 19.1-19.7 ms (~51-52 tok/s per row) |

So with realistic data the matmul-only floor of a bf16 decode step on this machine is 18.3 ms, not
14.4 ms, and 4 rows cost about 1 ms more once the run is long enough. The likely mechanism is the
30 W power cap (constant data toggles fewer bits, so the same kernel draws less and the governor
lets it run faster); that is an inference, not something measured directly. What is measured: the
step time changes by 27% with the data and nothing else. Switching the activations to fp32, or the
number of weight rows a warp takes (1, 2, 4, 8), changes nothing.

Consequences:
- The read-bandwidth figure (`bw.cu`, 219-250 GB/s) was also measured on constant data; for real
  weights use ~169 GB/s.
- The best frameworks (46-48 tok/s, section 1b) were already at 85-88% of this floor. In bf16 at
  30 W nobody can be more than ~15% faster than them on single-user decode.
- The memory clock moves between 5501 and 7001 MHz with temperature over a session (the same witness
  build gave 18.1 ms/step at 7001 MHz early on and 20.2 ms at 5501 MHz later). Only numbers taken
  back to back are comparable; `tools/perf.sh` alternates the prototype and witness for that.
- Per-kernel costs measured by leaving a kernel out (WK_SKIP) are not additive in this regime.

witness, stage 3, same session as the last table row (in-process, no HTTP or tokenizer; decode by
the step graph, prefill in 1024-token chunks):

| context | prefill | decode, 1 row | decode, 4 rows (per row) |
|---|---|---|---|
| 128 | 32 ms | 20.1-20.2 ms/step, 49.5-49.8 tok/s | 22.4-22.5 ms/step, 44.4-44.6 tok/s |
| 1024 | 231-232 ms | 20.45 ms, 48.9 tok/s | 23.7-24.0 ms, 41.7-42.2 tok/s |
| 2048 | 479-486 ms | 21.0-21.2 ms, 47.1-47.7 tok/s | 26.2-26.3 ms, 38.0-38.2 tok/s |
| 4096 | 1048-1050 ms | 21.5-21.7 ms, 46.2-46.5 tok/s | 29.5-29.7 ms, 33.7-33.9 tok/s |

Not yet compared with the frameworks in one session.

## 5. witness over HTTP, same bench.py (2026-11-26 evening) — first reading, not same-session

`run_witness.sh witness-m5`: the witness server (stage 5: tokenizer, HTTP, scheduler; greedy), --ctx 5120,
bench.py --reps 5 exactly as for the frameworks. Rows `witness-m5` in results.jsonl. Median TTFT /
decode tok/s per user, next to the best framework of section 1b per cell:

| prompt x conc | witness | best framework TTFT | best framework decode |
|---|---|---|---|
| 128 x 1 | 35 ms / 60.8 | 40 ms (trtllm) | 48.0 (sglang) |
| 128 x 2 | 58 ms / 49.5 | 58 ms (sglang) | 52.3 (llama.cpp) |
| 128 x 3 | 84 ms / 46.8 | 96 ms (llama.cpp) | 49.8 (llama.cpp) |
| 128 x 4 | 111 ms / 44.4 | 124 ms (llama.cpp) | 50.4 (llama.cpp) |
| 1024 x 1 | 232 ms / 49.0 | 211 ms (trtllm) | 47.3 (sglang) |
| 1024 x 2 | 417 ms / 45.5 | 316 ms (trtllm) | 45.7 (sglang) |
| 1024 x 3 | 619 ms / 42.1 | 468 ms (llama.cpp) | 46.4 (sglang) |
| 1024 x 4 | 689 ms / 38.1 | 631 ms (sglang) | 44.9 (trtllm) |
| 2048 x 1 | 495 ms / 47.5 | 426 ms (sglang) | 47.1 (sglang) |
| 2048 x 2 | 816 ms / 42.8 | 645 ms (sglang) | 43.3 (sglang) |
| 2048 x 3 | 1172 ms / 37.8 | 866 ms (sglang) | 44.9 (vllm) |
| 2048 x 4 | 1471 ms / 32.9 | 1070 ms (sglang) | 43.5 (trtllm) |
| 4096 x 1 | 1073 ms / 47.0 | 891 ms (sglang) | 45.8 (sglang) |
| 4096 x 2 | 1734 ms / 37.5 | 1335 ms (sglang) | 39.6 (sglang) |
| 4096 x 3 | 2427 ms / 30.0 | 1787 ms (sglang) | 36.7 (vllm) |
| 4096 x 4 | 2993 ms / 24.8 | 2228 ms (sglang) | 32.7 (vllm) |

GPU while busy: 29.9 W, SM 1356 MHz, memory clock 5501 MHz 96% / 7001 MHz 4%, max 60 C, peak VRAM
4421 MiB. guarded.sh: clean; measured-phase memory stall 11 ms.

- Ahead: TTFT at 128 tokens (all concurrencies) and single-user decode at every length.
- Behind: TTFT from 1024 tokens up (9-20% single user, more under concurrency) and decode under
  concurrency (up to 24% at 4096 x 4).
- The frameworks were measured about 7 hours earlier; the machine's state drifts (section 4), so the
  margins of a few percent are not established. The concurrency and long-prompt gaps are larger than
  the drift.
- 128 x 1 decode (60.8) is higher than any other witness reading at that state; treat it as an
  outlier until repeated.

## 6. Hand-written tensor-core kernel via inline PTX (`spikes/tc/mma.cu`, 2026-11-26)

Question: can our own kernels use the tensor cores (so attention need not go through cuBLAS calls and
materialized score matrices), and how fast is a simple one? Shape of attention scores: S = A B^T,
A [768][128], B [4096][128] bf16, S fp32 (12.6 MB), 30 W state.

- `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` works on sm_89 from nvcc inline asm.
- Results are bit-identical to cuBLAS GemmEx (max |diff| 0 over all 3.1M scores) and within 4e-6
  of an exact CPU sum. fp32 accumulation, so scores keep full precision.
- Throughput, same session:

| kernel | TFLOPS |
|---|---|
| ours, scores written out | 6.4-7.3 |
| ours, scores kept in registers (as a fused softmax would use them) | 7.9-8.2 |
| cuBLAS GemmEx, standalone at this shape | 11.0 |
| cuBLAS GemmEx as called inside the engine (from the prefill profile: 128-position tiles) | ~6.6 |

The kernel is the simplest design: each warp keeps its 16 rows of A in registers and loads B from
global memory with one 4-byte load per lane per operand word. Overlapping 2-8 column tiles per step,
or changing the chunk and block sizes, moves it by less than 15%: it is limited by those loads, not
by the mma chain. cuBLAS-class throughput needs B staged in shared memory (ldmatrix).

## 7. witness vs SGLang vs vLLM, one session (2026-11-26 16:11-16:40) — superseded by section 8

witness after the stage-3 speed work (speculative decoding by prompt lookup, separate norm kernel,
fused tensor-core attention), then SGLang, then vLLM, then witness again; same bench.py, --reps 5,
through HTTP. Rows `witness-opt`, `witness-opt2`, `sglang-eve`, `vllm-eve` in results.jsonl. Median
TTFT / decode tok/s per user:

| prompt x conc | witness, run 1 | witness, run 2 | SGLang | vLLM |
|---|---|---|---|---|
| 128 x 1 | 35 ms / 91.3 | 33 ms / 92.5 | 39 ms / 50.4 | 37 ms / 50.0 |
| 128 x 2 | 59 ms / 65.2 | 58 ms / 65.6 | 56 ms / 50.1 | 60 ms / 50.6 |
| 128 x 3 | 86 ms / 47.4 | 85 ms / 48.3 | 89 ms / 49.7 | 107 ms / 50.4 |
| 128 x 4 | 113 ms / 45.5 | 111 ms / 46.5 | 117 ms / 49.7 | 131 ms / 50.1 |
| 1024 x 1 | 227 ms / 102.3 | 227 ms / 103.9 | 201 ms / 49.8 | 203 ms / 49.6 |
| 1024 x 2 | 406 ms / 64.3 | 407 ms / 65.7 | 302 ms / 47.8 | 307 ms / 47.7 |
| 1024 x 3 | 606 ms / 43.5 | 604 ms / 44.3 | 585 ms / 48.6 | 590 ms / 48.4 |
| 1024 x 4 | 676 ms / 40.4 | 676 ms / 40.7 | 602 ms / 45.1 | 695 ms / 46.6 |
| 2048 x 1 | 476 ms / 132.1 | 478 ms / 131.8 | 399 ms / 49.3 | 402 ms / 48.7 |
| 2048 x 2 | 803 ms / 60.2 | 803 ms / 59.5 | 607 ms / 45.5 | 617 ms / 45.4 |
| 2048 x 3 | 1148 ms / 39.3 | 1151 ms / 38.8 | 808 ms / 41.4 | 1210 ms / 47.2 |
| 2048 x 4 | 1375 ms / 34.3 | 1376 ms / 33.9 | 1005 ms / 38.5 | 1412 ms / 43.9 |
| 4096 x 1 | 1041 ms / 76.5 | 1043 ms / 75.5 | 838 ms / 48.4 | 846 ms / 47.8 |
| 4096 x 2 | 1701 ms / 47.2 | 1698 ms / 47.1 | 1259 ms / 41.6 | 1286 ms / 41.2 |
| 4096 x 3 | 2376 ms / 31.9 | 2366 ms / 31.4 | 1675 ms / 34.9 | 2100 ms / 38.7 |
| 4096 x 4 | 2941 ms / 26.2 | 3005 ms / 26.6 | 2095 ms / 31.1 | 2528 ms / 34.5 |

GPU while busy, all four runs: 29.8-29.9 W, SM 1355-1394 MHz, memory clock 5501 MHz (99-100%), max 58 C.

- Decode, 1 and 2 concurrent: witness ahead in all 8 cells (+56% to +168% single user, +13% to +34%
  with two). That is speculative decoding: the prompts are code continuations, where the next tokens
  often repeat something in the context. It gives nothing when all 4 rows are taken by requests.
- Decode, 3 and 4 concurrent: witness behind in all 8 cells, by 6% to 24%.
- TTFT: witness ahead at 128 tokens for 1, 3 and 4 concurrent (3-6%); behind everywhere else, by
  13-24% single user and up to 43% under concurrency. The server prefills in 512-token chunks
  (1024 measured faster in-process), and concurrent prompts are prefilled one after another.
- The two witness runs agree within 3% in every cell.

Validity: guarded.sh's measured-phase memory stall was 12 and 11 ms for witness, 72 ms for SGLang and
81 ms for vLLM, over the 50 ms threshold. Their numbers are nevertheless better than their noon runs
of section 1b (SGLang 4096 x 1: 838 ms vs 891 ms; decode 48-50 vs 46-48 tok/s), so the stall did not
flatter witness; they are kept, with this caveat.

## 8. witness vs SGLang vs vLLM, one session (2026-11-26 evening, third round) — superseded by section 9

After: up to 8 rows per decode step with the spare ones carrying guesses for every request, 1024-token
prefill chunks, the MLP's first product in cache-sized slices, 16-byte staging in the fused attention
kernel, tensor-core decode attention. Order: witness, SGLang, vLLM, witness. Rows `witness-r3`,
`witness-r3b`, `sglang-r3`, `vllm-r3` in results.jsonl. Median TTFT / decode tok/s per user:

| prompt x conc | witness, run 1 | witness, run 2 | SGLang | vLLM |
|---|---|---|---|---|
| 128 x 1 | 34 ms / 114.0 | 35 ms / 116.1 | 38 ms / 50.4 | 39 ms / 50.0 |
| 128 x 2 | 60 ms / 77.8 | 59 ms / 78.9 | 55 ms / 50.1 | 60 ms / 50.3 |
| 128 x 3 | 84 ms / 90.9 | 87 ms / 91.6 | 88 ms / 49.6 | 107 ms / 50.4 |
| 128 x 4 | 113 ms / 70.6 | 115 ms / 75.1 | 116 ms / 49.7 | 130 ms / 50.0 |
| 1024 x 1 | 205 ms / 128.6 | 201 ms / 130.6 | 201 ms / 49.8 | 201 ms / 49.6 |
| 1024 x 2 | 316 ms / 70.2 | 312 ms / 75.4 | 302 ms / 47.9 | 308 ms / 47.7 |
| 1024 x 3 | 425 ms / 64.8 | 426 ms / 64.8 | 584 ms / 48.6 | 592 ms / 48.4 |
| 1024 x 4 | 540 ms / 60.1 | 538 ms / 61.0 | 598 ms / 45.1 | 694 ms / 46.5 |
| 2048 x 1 | 421 ms / 215.7 | 421 ms / 211.3 | 399 ms / 49.2 | 404 ms / 48.7 |
| 2048 x 2 | 744 ms / 78.5 | 745 ms / 73.3 | 607 ms / 45.5 | 618 ms / 45.4 |
| 2048 x 3 | 1093 ms / 66.3 | 1091 ms / 64.7 | 811 ms / 41.4 | 1208 ms / 47.2 |
| 2048 x 4 | 1219 ms / 52.3 | 1325 ms / 51.2 | 1008 ms / 38.5 | 1410 ms / 43.9 |
| 4096 x 1 | 911 ms / 94.1 | 911 ms / 94.3 | 836 ms / 48.5 | 848 ms / 47.7 |
| 4096 x 2 | 1499 ms / 66.6 | 1496 ms / 67.2 | 1261 ms / 41.6 | 1284 ms / 41.2 |
| 4096 x 3 | 2117 ms / 48.3 | 2124 ms / 49.5 | 1682 ms / 34.9 | 2099 ms / 38.7 |
| 4096 x 4 | 2617 ms / 42.3 | 2620 ms / 40.7 | 2095 ms / 31.0 | 2524 ms / 34.5 |

GPU while busy, all four runs: 29.8-29.9 W, SM 1349-1374 MHz, memory clock 5501 MHz, max 58 C.

- Decode: witness ahead in all 16 cells, by 17% (2048 x 4) to 330% (2048 x 1). All of the margin is
  speculative decoding on code continuations; with no guesses right it would be within a few percent.
- TTFT, single user: ahead at 128 (-10%), behind by 2% at 1024, 5% at 2048, 9% at 4096.
- TTFT, concurrent: ahead at 128 x 3-4 and 1024 x 3-4 (up to 27%), behind at 2048 and 4096 by
  19-35%. The scheduler prefills the newest request first, so of two prompts that arrive together the
  first is interrupted and both finish late (2048 x 2: 630 and 840 ms instead of 420 and 840).
- The two witness runs agree within 7% in every cell; SGLang and vLLM are within 1% of section 7.

Validity: this time every run had a measured-phase memory stall over the 50 ms threshold (witness 82
and 115 ms, SGLang 152 ms, vLLM 190 ms), so by the rule that a run disturbed by memory pressure is void, none is clean. The numbers
reproduce (see above), and the stalls are larger for the frameworks than for witness; treat margins
of a few percent as unestablished.

## 9. witness vs SGLang vs vLLM, one session (2026-11-26 17:40-18:10, fourth round) — decode columns superseded by section 10

After: prefill order changed (a request that continues one still being served first, otherwise the
least prompt left first). Order: witness, SGLang, vLLM, witness. Rows `witness-r4`, `witness-r4b`,
`sglang-r4`, `vllm-r4` in results.jsonl. Median TTFT / decode tok/s per user:

| prompt x conc | witness, run 1 | witness, run 2 | SGLang | vLLM |
|---|---|---|---|---|
| 128 x 1 | 36 ms / 116.5 | 35 ms / 116.6 | 42 ms / 50.3 | 38 ms / 50.0 |
| 128 x 2 | 60 ms / 78.0 | 58 ms / 77.9 | 58 ms / 50.1 | 62 ms / 50.3 |
| 128 x 3 | 86 ms / 92.4 | 89 ms / 94.2 | 89 ms / 49.7 | 107 ms / 50.4 |
| 128 x 4 | 115 ms / 70.6 | 114 ms / 72.1 | 117 ms / 49.7 | 131 ms / 50.0 |
| 1024 x 1 | 203 ms / 125.4 | 203 ms / 128.9 | 203 ms / 49.9 | 202 ms / 49.6 |
| 1024 x 2 | 314 ms / 75.0 | 313 ms / 74.8 | 302 ms / 47.9 | 306 ms / 47.7 |
| 1024 x 3 | 424 ms / 65.7 | 425 ms / 63.3 | 586 ms / 48.6 | 589 ms / 48.4 |
| 1024 x 4 | 540 ms / 57.7 | 540 ms / 59.2 | 596 ms / 45.0 | 695 ms / 46.5 |
| 2048 x 1 | 424 ms / 213.4 | 424 ms / 215.9 | 400 ms / 49.2 | 407 ms / 48.7 |
| 2048 x 2 | 653 ms / 76.9 | 654 ms / 75.6 | 606 ms / 45.5 | 619 ms / 45.3 |
| 2048 x 3 | 890 ms / 62.8 | 893 ms / 69.5 | 811 ms / 41.4 | 1215 ms / 47.2 |
| 2048 x 4 | 1121 ms / 52.3 | 1128 ms / 50.4 | 1008 ms / 38.5 | 1417 ms / 43.8 |
| 4096 x 1 | 909 ms / 95.0 | 916 ms / 95.6 | 836 ms / 48.5 | 852 ms / 47.7 |
| 4096 x 2 | 1411 ms / 63.5 | 1418 ms / 59.8 | 1260 ms / 41.6 | 1290 ms / 41.1 |
| 4096 x 3 | 1916 ms / 48.3 | 1916 ms / 50.7 | 1686 ms / 34.9 | 2105 ms / 38.7 |
| 4096 x 4 | 2414 ms / 38.4 | 2417 ms / 38.9 | 2095 ms / 31.0 | 2535 ms / 34.5 |

witness (the worse of its two runs) is ahead of the better framework in 5 of 16 cells on TTFT and
16 of 16 on decode.

NOT like for like on decode: witness runs with speculative decoding (prompt lookup) on, the
frameworks in their default configuration, which has theirs off. Both have one in this version
(vLLM: v1/worker/gpu/spec_decode/ngram; SGLang: srt/speculative/ngram_worker.py, algorithm NGRAM);
neither has been run with it here. The TTFT columns are like for like (cold prompts, no cache hits).

Validity: measured-phase memory stall was again over the 50 ms threshold in every run (see the
BENCH-PHASE lines of bench_*-r4*.log; `own_processes` is the stall of the run's own cgroup). Dropping
this project's files from the page cache first (evict.sh) did not reduce it; the cause is not found.
SGLang's numbers are within 1% of section 8 cell by cell regardless.

## 10. Like for like: all three with prompt-lookup speculative decoding, and the typing case (2026-11-26 18:07-18:35) — superseded by section 11

One session: witness, then SGLang with `--speculative-algorithm NGRAM` (its defaults: 12 draft
tokens, a tree, a cache shared across requests), then vLLM with `--speculative-config
{"method":"ngram","num_speculative_tokens":7,"prompt_lookup_min":2,"prompt_lookup_max":8}` (chosen
to mirror witness; its own defaults are min = max = 5; not tuned). bench.py now counts generated
tokens from the server's usage report instead of streamed events: both frameworks send the several
tokens of one speculative step as one event, which the old count undercounted. Rows `witness-r5`,
`sglang-spec`, `vllm-spec` in results.jsonl. Median TTFT / decode tok/s per user:

| prompt x conc | witness | SGLang, NGRAM | vLLM, ngram |
|---|---|---|---|
| 128 x 1 | 34 ms / 114.7 | 56 ms / 140.8 | 49 ms / 120.6 |
| 128 x 2 | 59 ms / 78.7 | 84 ms / 84.0 | 74 ms / 88.8 |
| 128 x 3 | 88 ms / 93.8 | 113 ms / 98.2 | 97 ms / 135.6 |
| 128 x 4 | 116 ms / 72.7 | 136 ms / 86.8 | 119 ms / 113.0 |
| 1024 x 1 | 202 ms / 128.5 | 227 ms / 137.5 | 217 ms / 132.4 |
| 1024 x 2 | 312 ms / 70.6 | 313 ms / 58.5 | 412 ms / 90.7 |
| 1024 x 3 | 425 ms / 67.7 | 591 ms / 60.9 | 586 ms / 88.5 |
| 1024 x 4 | 537 ms / 57.1 | 601 ms / 59.0 | 694 ms / 78.1 |
| 2048 x 1 | 423 ms / 219.0 | 418 ms / 123.3 | 423 ms / 223.3 |
| 2048 x 2 | 651 ms / 78.6 | 619 ms / 62.6 | 627 ms / 91.2 |
| 2048 x 3 | 891 ms / 66.8 | 834 ms / 54.7 | 1211 ms / 88.8 |
| 2048 x 4 | 1121 ms / 50.7 | 1019 ms / 49.5 | 1421 ms / 77.2 |
| 4096 x 1 | 911 ms / 94.5 | 862 ms / 87.8 | 859 ms / 87.6 |
| 4096 x 2 | 1410 ms / 63.7 | 1280 ms / 61.1 | 1298 ms / 57.7 |
| 4096 x 3 | 1912 ms / 49.1 | 1701 ms / 52.5 | 2113 ms / 62.7 |
| 4096 x 4 | 2409 ms / 39.7 | 2113 ms / 41.5 | 2543 ms / 49.9 |

witness is ahead of the better framework in 8 of 16 cells on TTFT and 2 of 16 on decode.

- Decode: vLLM is ahead of witness in 14 cells, by up to 36% (128 x 4: 113.0 vs 72.7). It gives
  every request 7 draft tokens whatever the concurrency; witness has 8 rows per step in total, so 4
  requests get one guess each. SGLang is ahead at 128 tokens, behind or level from 1024 x 2 up.
  Section 9's decode margins were the frameworks running without speculative decoding.
- TTFT: with speculative decoding on, both frameworks are slower on short prompts (128 x 1: 56 and
  49 ms, from 42 and 38 ms without), which is where witness leads; from 2048 tokens up witness is
  still 1-14% behind SGLang.

Typing case (`tools/typing_bench.py`, rows in typing.jsonl): a FIM prompt sent cold, then 10 times with
the prefix 3 tokens longer and the suffix unchanged; first-token time. All three servers report the
same prompt token counts.

| prefix / suffix | prompt tokens | witness cold / typing | SGLang cold / typing | vLLM cold / typing |
|---|---|---|---|---|
| 1500 / 0 | 1503/1503/1503 | 561 / 42.7 ms | 522 / 39.6 ms | 511 / 33.9 ms |
| 1500 / 64 | 1567/1567/1567 | 354 / 49.7 ms | 352 / 56.5 ms | 327 / 46.1 ms |
| 1500 / 300 | 1803/1803/1803 | 409 / 92.8 ms | 401 / 98.6 ms | 366 / 96.0 ms |
| 3000 / 300 | 3303/3303/3303 | 743 / 116.1 ms | 692 / 103.6 ms | 675 / 106.9 ms |
| 1500 / 800 | 2303/2303/2303 | 505 / 195.9 ms | 471 / 203.5 ms | 486 / 186.5 ms |

The three are within 20% of each other in every case; vLLM is fastest in three of five. witness has
no advantage here. Its server re-tokenizes the whole prompt on every request (about 15 ms of the
40-50 ms in the short cases).

Validity: measured-phase memory stall was 6 ms for witness, 0 ms for vLLM, and 984 ms for SGLang
(1128 ms in its own processes), so SGLang's numbers in this section may be understated. Its memory
clock was also at 7001 MHz for 27% of the run against 1-6% for the others, which works the other way.

## 11. After three rounds of speed work, one session (2026-11-26 night) — current comparison

witness: decode products on the tensor cores with up to 32 rows per step (24 used), guesses from
1-8 token matches in the request's own text and in other slots, fused attention split over key
segments for short continuations, prompts before decode steps. SGLang: `--speculative-algorithm
NGRAM`. vLLM: `--speculative-config {"method":"ngram","num_speculative_tokens":7,
"prompt_lookup_min":1,"prompt_lookup_max":8}` (with lookup_min 2, section 10, it was 20-30% slower
at decode). Rows `witness-r10`, `witness-r10b`, `sglang-spec3`, `vllm-spec3`. Median TTFT / decode
tok/s per user:

| prompt x conc | witness, run 1 | witness, run 2 | SGLang, NGRAM | vLLM, ngram (lookup 1-8) |
|---|---|---|---|---|
| 128 x 1 | 32 ms / 157.1 | 32 ms / 157.6 | 62 ms / 144.4 | 52 ms / 160.4 |
| 128 x 2 | 48 ms / 89.8 | 49 ms / 89.3 | 75 ms / 88.8 | 73 ms / 104.0 |
| 128 x 3 | 66 ms / 163.2 | 65 ms / 154.0 | 100 ms / 100.0 | 98 ms / 154.0 |
| 128 x 4 | 94 ms / 123.7 | 92 ms / 126.1 | 130 ms / 87.8 | 122 ms / 121.8 |
| 1024 x 1 | 206 ms / 185.5 | 204 ms / 173.6 | 225 ms / 147.4 | 215 ms / 156.8 |
| 1024 x 2 | 308 ms / 99.3 | 306 ms / 101.0 | 405 ms / 59.3 | 408 ms / 111.0 |
| 1024 x 3 | 411 ms / 87.9 | 406 ms / 86.9 | 417 ms / 64.2 | 586 ms / 87.4 |
| 1024 x 4 | 524 ms / 90.5 | 522 ms / 86.5 | 597 ms / 62.8 | 691 ms / 96.9 |
| 2048 x 1 | 429 ms / 307.1 | 428 ms / 319.7 | 420 ms / 122.8 | 424 ms / 224.2 |
| 2048 x 2 | 655 ms / 94.0 | 651 ms / 92.2 | 623 ms / 61.5 | 626 ms / 98.1 |
| 2048 x 3 | 879 ms / 85.5 | 878 ms / 94.6 | 815 ms / 60.2 | 1221 ms / 104.4 |
| 2048 x 4 | 1105 ms / 74.1 | 1102 ms / 72.5 | 1008 ms / 53.8 | 1416 ms / 94.7 |
| 4096 x 1 | 920 ms / 95.0 | 920 ms / 98.6 | 856 ms / 86.5 | 872 ms / 92.5 |
| 4096 x 2 | 1406 ms / 78.5 | 1405 ms / 73.4 | 1275 ms / 63.0 | 1295 ms / 82.9 |
| 4096 x 3 | 1894 ms / 57.1 | 1889 ms / 61.4 | 1694 ms / 51.7 | 2109 ms / 71.8 |
| 4096 x 4 | 2377 ms / 43.7 | 2375 ms / 44.9 | 2100 ms / 47.2 | 2529 ms / 51.4 |

Typing case, median first-token time:

| prefix / suffix | witness, run 1 | witness, run 2 | SGLang | vLLM |
|---|---|---|---|---|
| 1500 / 0 | 29.1 ms | 29.1 ms | 42.7 ms | 34.5 ms |
| 1500 / 64 | 35.4 ms | 35.7 ms | 53.2 ms | 46.3 ms |
| 1500 / 300 | 90.9 ms | 90.7 ms | 96.8 ms | 96.9 ms |
| 3000 / 300 | 99.2 ms | 99.1 ms | 100.3 ms | 109.4 ms |
| 1500 / 800 | 198.0 ms | 197.1 ms | 203.5 ms | 187.4 ms |

Items in which witness is ahead (16 TTFT cells, 16 decode cells, 5 typing cases), both witness runs:
- against the better of the two frameworks per item: 17-18 of 37 (TTFT 8, decode 5-6, typing 4);
- against SGLang alone: 26-27 of 37; against vLLM alone: 21-23 of 37 (see compare.py and the script
  in the session for the split).

Where it stands: TTFT ahead at 128 and 1024 tokens in every cell (1-38%), behind SGLang from 2048
up (2-13%: cold prefill is 5-8% slower); decode ahead of SGLang nearly everywhere, level with vLLM
single-user and behind it by 4-23% at 2-4 concurrent from 1024 tokens (vLLM prefills all prompts
together and decodes uninterrupted, at the price of TTFT 30-40% worse than witness there; witness
serves prompts first and requests that are already generating wait); typing ahead except with an
800-token suffix.

Validity: measured-phase memory stall 212 / 41 ms (witness), 480 ms (SGLang), 52 ms (vLLM).

## 1a. llama.cpp baseline, first pass (2026-11-26) — NOT KEPT AS BASELINE, see caveats

llama-server ab16df3, Qwen2.5-Coder-1.5B f16 GGUF, -np 4 -c 20480 -ngl 99 --cache-ram 1024, bench.py
(--reps 5, 128 generated tokens, distinct prompts so no prefix reuse). results.jsonl.

| prompt | c=1 TTFT / tok/s | c=2 | c=3 | c=4 |
|---|---|---|---|---|
| 128 | 51 ms / 47.8 | 68 ms / 56.3 each | 98 ms / 51.0 | 124 ms / 52.2 |
| 1024 | 218 ms / 45.9 | 436 ms / 44.9 | 443 ms / 21.5 | 667 ms / 41.3 |
| 2048 | 440 ms / 45.6 | 684 ms / 38.9 | 1350 ms / 21.5 | 1591 ms / 40.3 |
| 4096 | 917 ms / 44.7 | 1417 ms / 33.3 | 2348 ms / 21.1 | 2871 ms / 33.1 |

Caveats found while running it:
- The GPU was power-capped: Windows "Balanced" plan, GPU power limit 30 W (default 35, max 65),
  memory clock averaging 5717 of 8001 MHz, SM clock dipping to 210 MHz; driver counters show long
  SW power-cap and SW thermal slowdowns. Machine: HP OMEN Transcend 14-fb0xxx (Core Ultra 7 155H).
  All numbers so far (bandwidth, GEMV, this table) were taken in that state. Re-run everything
  after switching Windows power mode to Best performance and OMEN Gaming Hub to Performance.
- c=3 decode halving (~21 tok/s per user) is not explained yet; could be the clock swings or
  llama.cpp interleaving other slots' prefill chunks with decode. To check with Nsight Systems.
- Prefill is ~4.5k tok/s (4096 tokens -> ~0.9 s TTFT at c=1): far below the bf16 GEMM roofline
  (1.5B ~3 GFLOP/token -> ~8k tok/s at 24 TFLOPS). TTFT is the weak spot for code completion.

## 1b. llama.cpp vs vLLM vs SGLang vs TensorRT-LLM, same session (2026-11-26 11:38-12:35) — current baseline

Same bf16 weights, same bench.py (--reps 5, 128 generated tokens, distinct prompts), one framework after
another via `run_llamacpp.sh` / `run_baseline.sh`. llama-server ab16df3 (f16 GGUF, -np 4 -c 20480
--cache-ram 1024), vLLM 0.34.0 (--dtype bfloat16 --max-model-len 5120 --gpu-memory-utilization 0.85),
SGLang 0.5.23 (--dtype bfloat16 --context-length 5120 --mem-fraction-static 0.8), TensorRT-LLM 1.3.0
(trtllm-serve, PyTorch backend — the only one this version offers — --max_seq_len 5120
--max_batch_size 4 --free_gpu_memory_fraction 0.5 --no-telemetry). Rows labelled `*-manual` in
results.jsonl. Median TTFT / decode tok/s per user:

| prompt x conc | llama.cpp | vLLM | SGLang | TensorRT-LLM |
|---|---|---|---|---|
| 128 x 1 | 50 ms / 46.1 | 40 ms / 47.8 | 43 ms / 48.0 | 40 ms / 46.7 |
| 128 x 2 | 74 ms / 52.3 | 63 ms / 47.8 | 58 ms / 47.7 | 75 ms / 47.2 |
| 128 x 3 | 96 ms / 49.8 | 115 ms / 47.9 | 96 ms / 47.4 | 115 ms / 45.8 |
| 128 x 4 | 124 ms / 50.4 | 135 ms / 47.8 | 124 ms / 47.5 | 125 ms / 46.0 |
| 1024 x 1 | 227 ms / 45.0 | 211 ms / 46.9 | 216 ms / 47.3 | 211 ms / 46.7 |
| 1024 x 2 | 461 ms / 42.8 | 323 ms / 45.6 | 324 ms / 45.7 | 316 ms / 45.1 |
| 1024 x 3 | 468 ms / 20.7 | 625 ms / 46.2 | 619 ms / 46.4 | 621 ms / 45.6 |
| 1024 x 4 | 737 ms / 43.0 | 738 ms / 44.4 | 631 ms / 43.0 | 824 ms / 44.9 |
| 2048 x 1 | 447 ms / 44.3 | 432 ms / 46.4 | 426 ms / 47.1 | 432 ms / 46.2 |
| 2048 x 2 | 717 ms / 38.3 | 659 ms / 43.2 | 645 ms / 43.3 | 646 ms / 42.5 |
| 2048 x 3 | 1404 ms / 20.6 | 1290 ms / 44.9 | 866 ms / 39.4 | 1276 ms / 44.4 |
| 2048 x 4 | 1660 ms / 41.3 | 1502 ms / 41.8 | 1070 ms / 36.7 | 1717 ms / 43.5 |
| 4096 x 1 | 945 ms / 42.8 | 903 ms / 45.3 | 891 ms / 45.8 | 907 ms / 45.3 |
| 4096 x 2 | 1488 ms / 31.7 | 1367 ms / 39.1 | 1335 ms / 39.6 | 1365 ms / 38.7 |
| 4096 x 3 | 2474 ms / 20.2 | 2240 ms / 36.7 | 1787 ms / 33.2 | 1817 ms / 32.5 |
| 4096 x 4 | 3007 ms / 31.3 | 2698 ms / 32.7 | 2228 ms / 29.5 | 2286 ms / 28.9 |

GPU state while busy (gpu_*-manual.csv): OMEN "manual" mode, driver power limit still 30 W.

| | avg W | avg SM MHz | memory clock | peak VRAM |
|---|---|---|---|---|
| llama.cpp | 29.9 | 1297 | 5501 MHz 81%, 7001 MHz 19% | not logged |
| vLLM | 29.9 | 1321 | 5501 MHz 99% | 5823 MiB |
| SGLang | 29.9 | 1297 | 5501 MHz 100% | 6977 MiB |
| TensorRT-LLM | 29.9 | 1355 | 5501 MHz 99.5% | 5712 MiB |

Findings:
- Single user, the four are within ~7%: decode 43-48 tok/s, prefill ~4.3-4.8k tok/s
  (4096 tokens: 945 / 903 / 891 / 907 ms). The matmul-only prototype in the same state is 69.6 tok/s.
- Best per metric: single-user decode SGLang 45.8-48.0; single-user TTFT vLLM/TensorRT-LLM at 128
  (40 ms), SGLang at 2048-4096; concurrent TTFT SGLang (4096 x 4: 2228 ms; TensorRT-LLM 2286,
  vLLM 2698, llama.cpp 3007); concurrent decode at 2048 x 3-4 vLLM / TensorRT-LLM (42-45 tok/s).
- llama.cpp's c=3 decode halving (~20 tok/s) reproduces and is its own: the other three do not show it.
- VRAM is a preallocated KV pool in vLLM, SGLang and TensorRT-LLM, so the peak is a setting, not a need.

Validity: guarded.sh reported DIRTY for vLLM, SGLang and TensorRT-LLM (whole-run memory stall
470 / 195 / 124 ms), all of it model load: the stall over the measured phase alone (BENCH-PHASE line
in run_baseline.sh) was 0 / 35 / 7 ms, under the guard's 50 ms threshold. Earlier flagged runs are kept
apart in results_{vllm,sglang,trtllm}_dirty.jsonl (first SGLang run: 129 ms stall in the measured
phase and a 6 s TTFT outlier from flashinfer compiling a kernel at first use; first TensorRT-LLM run:
585 ms); their medians agree with the kept runs within ~2%. The llama.cpp memory clock was higher 19%
of the time, slightly in its favour.

Setup problems hit on WSL: SGLang needs its venv's `ninja` on PATH (flashinfer JIT); TensorRT-LLM's
Open MPI worker never connects unless MPI is kept on loopback (OMPI_MCA_btl_tcp_if_include=lo,
OMPI_MCA_oob_tcp_if_include=lo). Both are in run_baseline.sh.

Not done: ExLlamaV2 (not installed), prefix-cache-hit TTFT, real /infill workload.

## WSL crash on 2026-11-26 10:50 (cause found)

llama-server's host-RAM prompt cache (`--cache-ram`, default 8192 MiB) grew with every distinct
4096-token prompt (~115 MB of KV each) until the 7.6 GB WSL VM ran out; the OOM killer killed
llama-server, and systemd's default OOMPolicy=stop then stopped all of init.scope (VS Code server,
Claude Code), taking the session down. Fixes: WSL raised to 12 GB + 8 GB swap (.wslconfig);
experiments run through `spikes/guarded.sh` (own scope, MemoryMax, OOMPolicy=continue, OOM/PSI
check -> GUARD-VERDICT); llama-server runs with `--cache-ram 1024`.

## 12. Round 4: prefill (cuBLAS kernel choice, attention layout)

Goal of the round: lead in most items against the better of the two frameworks per item, not only
against each alone. What changed, all in the wide (prefill) path, measured in one session at 30 W:

| change | effect at 4096 tokens |
|---|---|
| products through cublasLt with a chosen configuration (algorithm 21, tile 24, 9 stages at 1024 rows; `core/lt_table.inc`, found with `spikes/gemm/lt.cu`), result bit-identical to cublasGemmEx | −5% to −7% |
| MLP slices of 1120 units (an eighth) instead of 996 | about −2% |
| fused attention: staged K and V laid out so that a warp's 32 lanes hit 32 shared-memory banks (K reads were 8-way conflicts, the V staging stores about 32-way); staging in a non-inlined function to stay at 128 registers | attention 44.8 → 34.0 us/token (−24%), −5% overall |

In-process prefill, old build and new alternated: 1024 tokens 208 → 186 ms, 2048: 430 → 378,
4096: 917 → 805 (−12%). Checks pass (generation identical to HF on the reference cases).

Tried without gain: double-buffered staging with prefetch plus split score accumulators (register
count rose to 155–160; capped at 128 it was 46.8 against 44.3 us/token), 4 or 16 warps per block,
`-use_fast_math` (1%), bf16 gate/up output and other slice sizes, chunks other than 1024.
A trap worth remembering: with fewer than 8 warps the old staging silently staged half the keys and
looked 19% faster; and the kernel is twice as slow as soon as it needs more than 128 registers
(one block per SM instead of two).

Known weak spot: chunks that are not 1024 rows (a prompt's remainder, a long suffix) cost 5% to
20% more per token; only 768 and 1024 rows have tuned entries.

Same-session comparison, speculative decoding on everywhere (rows `witness-r11`, `witness-r11b`,
`sglang-spec4`, `vllm-spec4`; `compare.py witness-r11 sglang-spec4 vllm-spec4`):

- against the better framework per item: TTFT 15/16, decode 7/16, typing 4/5: **26/37** (both witness runs)
- TTFT: ahead in every cell except 4096 x 4 (2098–2101 against SGLang's 2097 ms, a tie)
- decode: ahead single-user from 1024 tokens (+6% to +42%); behind vLLM at 2–4 concurrent in most
  cells (−4% to −15%), the price of serving prompts first
- typing: ahead except prefix 1500 / suffix 800 (193 against vLLM's 187 ms)

Memory stalls in the measured phase: SGLang 12 ms, vLLM 27 ms, witness 0 ms and 217 ms (the two
witness runs agree cell by cell).

## 13. Public corpus (2026-11-27) — current comparison

Prompts used to be cut from a private repository's Rust sources. They now come from ripgrep 14.1.1
(crates searcher and ignore, 105164 tokens; `tools/corpus.py`, `tools/fetch_corpus.sh`), so every
number from here on can be reproduced. Reference dumps, token files, checks and benchmarks were
redone on it; the engine did not change. Numbers of earlier sections are on the old corpus.

Checks on the new corpus:
- Greedy generation is identical to HF bf16 in 1 of 3 reference cases; in the other two it departs
  at generated token 5 and 9, where HF bf16's own top-2 margin is 0.125 (one bf16 step at that
  magnitude). The exact configuration has no argmax difference from HF fp32 at any position. On the
  old corpus all three happened to be identical.
- Normal configuration, logits against HF fp32: 6.6e-3 to 6.1e-2 (decode path), 5.7e-3 to 3.6e-2
  (prefill). The limit had been 5e-2, set from the old cases (worst 1.8e-2). HF bf16 against HF fp32
  on the same cases is 2.8e-2 to 7.6e-2, so the limit is now 1e-1 and check prints that yardstick.
  The one case over the old limit (fim, 6.1e-2) is also the one where ours is further from fp32
  than HF bf16 is (2.8e-2); not investigated further.
- `tools/server_test.py trace`: with the new prompt 24 tokens take about 10 steps, and the wall-clock
  effect of the injected waits was inside the run-to-run noise; injections are now 10 / 10 / 5 ms.
  Also seen: the total grows by less than what is injected (64-94 ms for 110), because the
  power-capped GPU runs each step faster when there are pauses between steps.

Same session, speculative decoding on everywhere; rows `witness-pub`, `sglang-pub`, `vllm-pub`,
`witness-pub-b` (run in that order). Median TTFT / decode tok/s per user:

| prompt x conc | witness | witness (2nd run) | SGLang | vLLM |
|---|---|---|---|---|
| 128 x 1 | 37 ms / 214.5 | 33 / 195.5 | 54 ms / 80.7 | 50 ms / 163.3 |
| 128 x 2 | 51 / 90.6 | 48 / 82.1 | 76 / 89.4 | 78 / 97.3 |
| 128 x 3 | 71 / 120.1 | 66 / 102.7 | 108 / 84.8 | 97 / 126.0 |
| 128 x 4 | 99 / 99.7 | 94 / 88.5 | 131 / 79.8 | 131 / 83.1 |
| 1024 x 1 | 195 / 107.3 | 184 / 93.1 | 221 / 78.7 | 216 / 94.3 |
| 1024 x 2 | 283 / 154.7 | 276 / 144.2 | 412 / 103.8 | 411 / 192.8 |
| 1024 x 3 | 362 / 131.5 | 366 / 121.0 | 587 / 76.8 | 593 / 141.2 |
| 1024 x 4 | 464 / 96.8 | 469 / 94.9 | 606 / 68.6 | 700 / 106.4 |
| 2048 x 1 | 379 / 88.8 | 382 / 88.2 | 423 / 85.7 | 425 / 91.2 |
| 2048 x 2 | 578 / 92.4 | 584 / 90.3 | 623 / 73.3 | 632 / 95.5 |
| 2048 x 3 | 780 / 75.3 | 786 / 90.5 | 811 / 57.2 | 1237 / 93.8 |
| 2048 x 4 | 984 / 73.9 | 991 / 73.3 | 1022 / 66.3 | 1431 / 73.1 |
| 4096 x 1 | 812 / 88.5 | 816 / 88.4 | 864 / 130.9 | 876 / 94.9 |
| 4096 x 2 | 1244 / 100.8 | 1250 / 130.8 | 1287 / 84.2 | 1323 / 99.1 |
| 4096 x 3 | 1678 / 52.4 | 1686 / 61.6 | 1698 / 54.7 | 2142 / 56.6 |
| 4096 x 4 | 2122 / 53.4 | 2131 / 56.1 | 2114 / 51.0 | 2569 / 46.3 |

Typing (median first-token ms): 1500/0: 25.6, 28.0 | 43.7 | 48.8;  1500/64: 38.3, 34.6 | 54.5 | 47.3;
1500/300: 88.7, 88.9 | 99.8 | 102.9;  3000/300: 95.8, 98.1 | 106.0 | 93.4;  1500/800: 195.5, 197.0 | 195.8 | 206.5.

Against the better framework per item: TTFT 15/16 both runs, decode 6/16 both runs (not the same six),
typing 4/5 and 3/5: **25/37 and 24/37**. TTFT is the solid result; decode is not a lead, and with
5-20 requests per cell its cells move by up to 30% between the two witness runs.
Memory stall in the measured phase: witness 0 and 25 ms, SGLang 7 ms, vLLM 19 ms (its whole run,
including model load, was flagged at 256 ms).

### 13a. The acceptance check, revised (2026-11-27)

GOALS.md was removed; the Chinese reference case, which was its text, is now `tools/zh_sample.md`
(references regenerated). On it greedy generation departs from HF bf16 at generated token 0
(margin 0.125), so all three cases now differ from HF bf16 somewhere, each at a 0.125 margin.

The fixed logit tolerance said little: HF bf16 itself is 2.8e-2 to 2.2e-1 from HF fp32 on these
cases. `check` now measures the normal configuration against the fp32 model with HF's bf16 run as
the yardstick and fails if ours is further away, by mean KL divergence or by argmax flips:

| case | HF bf16: mean KL, flips | witness decode path | witness prefill path |
|---|---|---|---|
| chinese | 1.90e-3, 7 | 2.14e-4, 3 | 2.13e-4, 2 |
| fim | 4.63e-4, 4 | 1.22e-4, 1 | 1.15e-4, 1 |
| rust | 9.03e-4, 3 | 1.19e-4, 0 | 1.21e-4, 0 |

(0.5B model: HF bf16 1.2e-3 to 1.5e-3, ours 2.5e-4 to 3.2e-4.) Still open: no comparison with HF
beyond 233 tokens; the generation comparison stops at the first difference.

### 13b. Do the engines agree with each other? (2026-11-27)

`tools/agree.py`: 24 prompts from the corpus (128, 512, 1024 tokens), 48 tokens each, temperature 0,
one request at a time, each server asked twice; no speculative decoding in the frameworks.
Identical completions out of 24: witness vs itself 24, witness with vs without speculative decoding
24, vLLM vs itself 20, SGLang vs itself 19, witness vs vLLM 16-17, witness vs SGLang 18-22,
vLLM vs SGLang 17-18. First differences are spread from token 1 to 38. So "the same text as vLLM or
SGLang" cannot be an acceptance rule: neither meets it against itself. (Why they are not repeatable
was not looked into.)
