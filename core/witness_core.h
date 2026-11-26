// witness GPU core: weights, KV cache and kernels for one Qwen2.5 model on one GPU.
// C ABI for the Rust host. Every function returning int returns 0 on success; on failure
// wk_last_error() describes it. Not thread safe: one thread owns an engine.
#ifndef WITNESS_CORE_H
#define WITNESS_CORE_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define WK_MAX_ROWS 32

typedef struct wk_engine wk_engine;

typedef struct {
    uint32_t vocab, hidden, layers, q_heads, kv_heads, head_dim, mlp;
    float rms_eps, rope_theta;
} wk_config;

typedef struct {
    uint32_t slots;    // independent sequences the KV cache holds
    uint32_t max_ctx;  // positions per slot
    uint32_t max_chunk;  // most tokens one wk_prefill call takes (0: wk_prefill unavailable)
    uint32_t kv_bf16;    // KV cache element type: 1 = bf16 (normal), 0 = fp32 (exact, for checking)
} wk_options;

// One row of a step: feed `token` at position `pos` of sequence `slot`. Its K/V are written at
// (slot, pos) and it attends to positions 0..pos of that slot, which must already be filled, or be
// filled by earlier rows of the same step: the rows of one slot must be adjacent and at consecutive
// positions. With an fp32 KV cache (the exact configuration) a step takes at most 8 rows.
typedef struct {
    int32_t token, slot, pos;
} wk_row;

// Optional copies of intermediate values for the correctness checks. Each pointer is a host buffer
// of [layers][n_rows][dim] floats (final_norm: [n_rows][hidden]); see tools/ref_dump.py for what
// each one is. qkv is before RoPE.
typedef struct {
    float *x_in, *norm1, *qkv, *attn, *o, *norm2, *act, *mlp, *final_norm;
} wk_taps;

int wk_create(const char* weights_path, const wk_options* opt, wk_engine** out);
void wk_destroy(wk_engine* e);
void wk_get_config(const wk_engine* e, wk_config* out);

// Run n_rows (1..WK_MAX_ROWS) rows with the reference kernels. `logits`
// ([n_rows][vocab]) and `taps` may be NULL. `argmax` ([n_rows]) is always written.
int wk_step(wk_engine* e, const wk_row* rows, int n_rows, float* logits, int32_t* argmax, const wk_taps* taps);

// The same step with the fused kernels.
int wk_step_fast(wk_engine* e, const wk_row* rows, int n_rows, float* logits, int32_t* argmax);

// Greedy generation with no host work between steps: rows[r].token is fed at rows[r].pos, and each
// step's argmax is fed at the next position, n_tokens times. `out` is [n_rows][n_tokens].
int wk_generate(wk_engine* e, const wk_row* rows, int n_rows, int n_tokens, int32_t* out);

#define WK_ALL_LOGITS 1u  // logits for every position ([n][vocab]) instead of the last one
#define WK_EXACT 2u       // fp32 products instead of bf16-input tensor-core ones: slow, for checking

// Feed n consecutive tokens at positions pos0..pos0+n-1 of `slot` in one pass; positions 0..pos0 of
// the slot must already be filled. `logits` receives the last position's logits ([vocab], may be
// NULL), or all positions' with WK_ALL_LOGITS. `argmax` is the last position's. Taps are
// [layers][n][dim].
int wk_prefill(wk_engine* e, int32_t slot, int32_t pos0, const int32_t* tokens, int n, uint32_t flags,
               float* logits, int32_t* argmax, const wk_taps* taps);

// Copy the K/V of positions 0..n of `src_slot` to `dst_slot`, so that a sequence sharing that prefix
// can continue in `dst_slot` while `src_slot` stays in use.
int wk_copy_kv(wk_engine* e, int32_t src_slot, int32_t dst_slot, int32_t n);

// --- Ground-truth trace and injected waits (process-wide) ---

// One CUDA runtime call, cuBLAS call or kernel launch made by the core.
typedef struct {
    uint64_t t0_ns, t1_ns;  // CLOCK_MONOTONIC, as wk_now_ns()
    uint64_t cpu_ns;        // CPU time the calling thread used inside the call
    uint32_t site;          // see wk_trace_site
    uint32_t injected_us;   // wait injected inside this call by wk_inject_cuda_delay
} wk_trace_rec;

uint64_t wk_now_ns(void);
void wk_trace_enable(int on);
// Moves up to `cap` records out of the trace; `dropped` (may be NULL) receives the number lost to a
// full buffer since the last call.
int wk_trace_drain(wk_trace_rec* out, int cap, uint64_t* dropped);
// Name of a call site, e.g. "cudaStreamSynchronize" or "launch k_f_qkv". Returns its length or -1.
int wk_trace_site(uint32_t id, char* out, int cap);
// Sleep `microseconds` inside every call whose text contains `site_substring` (0 turns it off).
void wk_inject_cuda_delay(const char* site_substring, uint32_t microseconds);

const char* wk_last_error(void);

#ifdef __cplusplus
}
#endif
#endif
