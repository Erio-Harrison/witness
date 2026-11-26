// Weights are bf16, activations fp32, the KV cache bf16 or (for checking) fp32.
//
// Reference implementation: one kernel per operation, nothing fused, every intermediate value can be
// tapped. Two paths share its layer loop and every kernel except the matrix products:
//   narrow (wk_step)     a few rows, one token each; products are GEMVs that read each weight once.
//                        With an fp32 KV cache: the same weights in an fp32 forward pass.
//   wide   (wk_prefill)  many consecutive tokens of one sequence; products are cuBLAS GEMMs on the
//                        tensor cores, which take bf16 inputs: the activations entering a product are
//                        rounded to bf16, the product is accumulated and stored in fp32. WK_EXACT
//                        replaces them with fp32 SGEMMs on widened weights (slow, for checking).
//
// Fast narrow path (wk_step_fast, wk_generate): the fused kernels of fast.cuh; wk_generate runs whole
// steps as one CUDA graph each, argmax and feedback included, with no host work between steps.
#include "witness_core.h"

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <map>
#include <mutex>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <vector>

static thread_local std::string g_err;

static int fail(const char* fmt, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    g_err = buf;
    return -1;
}

// ---------------------------------------------------------------------------------------------
// Ground-truth trace and injected waits. Every CUDA runtime call, cuBLAS call and kernel launch of
// the host code goes through WK_CUDA / WK_BLAS / WK_LAUNCH. When the hook is off (the normal case)
// each costs one test of g_hook. When it is on, the call's start, end and thread CPU time go to a
// ring that wk_trace_drain empties, and a wait can be injected inside calls whose text matches.

static bool g_hook;                  // tracing or injection is active
static bool g_trace;
static std::mutex g_trace_mu;
static std::vector<wk_trace_rec> g_ring;
static uint64_t g_ring_dropped;
static std::vector<const char*> g_sites;
static std::string g_delay_match;
static uint32_t g_delay_us;
static const size_t RING_CAP = 1 << 18;

static uint64_t clock_ns(clockid_t id) {
    timespec t;
    clock_gettime(id, &t);
    return (uint64_t)t.tv_sec * 1000000000ull + t.tv_nsec;
}

struct TraceScope {
    const char* site;
    uint64_t t0 = 0, c0 = 0;
    uint32_t injected = 0;
    explicit TraceScope(const char* s) : site(s) {
        if (!g_hook) return;
        t0 = clock_ns(CLOCK_MONOTONIC);
        c0 = clock_ns(CLOCK_THREAD_CPUTIME_ID);
        if (g_delay_us && strstr(site, g_delay_match.c_str())) {
            injected = g_delay_us;
            usleep(injected);
        }
    }
    ~TraceScope() {
        if (!t0 || !g_trace) return;
        uint64_t t1 = clock_ns(CLOCK_MONOTONIC), c1 = clock_ns(CLOCK_THREAD_CPUTIME_ID);
        std::lock_guard<std::mutex> lock(g_trace_mu);
        uint32_t id = 0;
        while (id < g_sites.size() && g_sites[id] != site) id++;
        if (id == g_sites.size()) g_sites.push_back(site);
        if (g_ring.size() >= RING_CAP) {
            g_ring_dropped++;
            return;
        }
        g_ring.push_back({t0, t1, c1 - c0, id, injected});
    }
};

#define WK_CUDA(call)                                                                              \
    do {                                                                                           \
        cudaError_t e_;                                                                            \
        {                                                                                          \
            TraceScope ts_(#call);                                                                 \
            e_ = (call);                                                                           \
        }                                                                                          \
        if (e_ != cudaSuccess) return fail("%s: %s", #call, cudaGetErrorString(e_));               \
    } while (0)
#define WK_BLAS(call)                                                                              \
    do {                                                                                           \
        cublasStatus_t st_;                                                                        \
        {                                                                                          \
            TraceScope ts_(#call);                                                                 \
            st_ = (call);                                                                          \
        }                                                                                          \
        if (st_ != CUBLAS_STATUS_SUCCESS) return fail("%s: cuBLAS status %d", #call, (int)st_);    \
    } while (0)
// A kernel launch. Launches return at once unless the driver's queue is full.
#define WK_LAUNCH(...)                                                                             \
    do {                                                                                           \
        TraceScope ts_("launch " #__VA_ARGS__);                                                    \
        __VA_ARGS__;                                                                               \
    } while (0)

extern "C" uint64_t wk_now_ns(void) { return clock_ns(CLOCK_MONOTONIC); }

extern "C" void wk_trace_enable(int on) {
    std::lock_guard<std::mutex> lock(g_trace_mu);
    g_trace = on != 0;
    g_hook = g_trace || g_delay_us;
}

extern "C" void wk_inject_cuda_delay(const char* site_substring, uint32_t microseconds) {
    std::lock_guard<std::mutex> lock(g_trace_mu);
    g_delay_match = site_substring ? site_substring : "";
    g_delay_us = microseconds;
    g_hook = g_trace || g_delay_us;
}

extern "C" int wk_trace_drain(wk_trace_rec* out, int cap, uint64_t* dropped) {
    std::lock_guard<std::mutex> lock(g_trace_mu);
    int n = (int)g_ring.size() < cap ? (int)g_ring.size() : cap;
    memcpy(out, g_ring.data(), n * sizeof(wk_trace_rec));
    g_ring.erase(g_ring.begin(), g_ring.begin() + n);
    if (dropped) *dropped = g_ring_dropped, g_ring_dropped = 0;
    return n;
}

// The call's name: for "launch k_f_qkv<R, KV><<<...", "launch k_f_qkv"; for "cudaMemcpyAsync(...)",
// "cudaMemcpyAsync".
extern "C" int wk_trace_site(uint32_t id, char* out, int cap) {
    std::lock_guard<std::mutex> lock(g_trace_mu);
    if (id >= g_sites.size() || cap < 1) return -1;
    const char* s = g_sites[id];
    int n = 0;
    for (const char* p = s; *p && n < cap - 1; p++) {
        if (*p == '(' || *p == '<') break;
        out[n++] = *p;
    }
    out[n] = 0;
    return n;
}

// ---------------------------------------------------------------------------------------------
// Weight file (tools/convert.py)

struct FileHeader {
    char magic[8];
    uint32_t version;
    wk_config cfg;
    uint32_t n_tensors, pad;
    // uint64_t offsets[n_tensors]; uint64_t file_size;
};
static_assert(sizeof(FileHeader) == 56, "header layout");

enum { T_EMBED = 0, T_FINAL_NORM = 1, T_LAYER0 = 2, T_PER_LAYER = 7 };
enum { L_NORM1, L_QKV_W, L_QKV_B, L_O_W, L_NORM2, L_GATE_UP_W, L_DOWN_W };

struct wk_engine {
    wk_config c{};
    wk_options o{};
    int qkv_dim = 0, kv_dim = 0;
    size_t cap = 0;                                 // rows the activation buffers hold
    std::vector<uint64_t> off;
    char* blob = nullptr;                           // the whole weight file, on the device
    float *x = nullptr, *h = nullptr, *qkv = nullptr, *attn = nullptr, *y = nullptr;  // [cap][dim]
    float *gu = nullptr, *act = nullptr;
    float* logits = nullptr;                        // [WK_MAX_ROWS][vocab]
    float* scores = nullptr;                        // [cap][q_heads][max_ctx]
    void *K = nullptr, *V = nullptr;                // [slots][layers][max_ctx][kv_dim], bf16 or fp32
    float* q = nullptr;                             // [WK_MAX_ROWS][hidden], fast path: q after RoPE
    float *att_part = nullptr, *att_stat = nullptr; // fast path attention: per-tile outputs and their scales
    int32_t *count = nullptr, *gen_out = nullptr;   // wk_generate: tokens produced per row, [WK_MAX_ROWS][GEN_CAP]
    cudaGraphExec_t graph[WK_MAX_ROWS + 1] = {};    // one decode step for R rows (scalar kernels)
    std::map<std::string, cudaGraphExec_t> tc_graph; // tensor-core decode step, by its rows per slot
    int32_t* pairs = nullptr;                       // [WK_MAX_ROWS][2]: rows that share a warp of the decode attention
    int32_t h_slot[WK_MAX_ROWS] = {}, h_pos[WK_MAX_ROWS] = {};  // the rows last uploaded by set_rows
    float* inv_freq = nullptr;                      // [head_dim / 2]
    float2* rope = nullptr;                         // [max_ctx][head_dim / 2]: (cos, sin) of position * inv_freq
    int32_t *tok = nullptr, *slot = nullptr, *pos = nullptr;  // [cap]
    __nv_bfloat16* xb = nullptr;                    // [cap][max(hidden, mlp)], a product's input in bf16
    float* wide_w = nullptr;                        // WK_EXACT: one weight matrix (or a slab of one) in fp32
    size_t wide_w_count = 0;
    float* all_logits = nullptr;                    // WK_ALL_LOGITS: [all_logits_rows][vocab]
    size_t all_logits_rows = 0;
    __nv_bfloat16 *hb = nullptr, *ab = nullptr;     // fused wide path: [cap][hidden] normalized x, attention output
    __nv_bfloat16 *wa_q = nullptr, *wa_p = nullptr; // wide attention: packed q, a tile's weights
    float *wa_s = nullptr, *wa_o = nullptr;         // a tile's scores and outputs
    float* wa_inv = nullptr;                        // 1 / sum of each score row's weights
    float *fl_part = nullptr, *fl_stat = nullptr;   // fused wide attention split over key segments: partial results
    cudaStream_t s = nullptr;
    cublasHandle_t blas = nullptr;
    cublasLtHandle_t lt = nullptr;
    cublasLtMatmulDesc_t lt_desc = nullptr;           // op(A) = A^T, fp32 accumulation
    struct LtPlan { bool use; cublasLtMatmulAlgo_t algo; };
    std::map<uint64_t, LtPlan> lt_plans;              // by (N, K, R, result type), see gemm_wide
    std::vector<float> host_logits;

    template <class T> const T* w(int tensor) const { return reinterpret_cast<const T*>(blob + off[tensor]); }
    template <class T> const T* w(int layer, int which) const { return w<T>(T_LAYER0 + layer * T_PER_LAYER + which); }
};

// ---------------------------------------------------------------------------------------------
// Kernels. Blocks are 256 threads; "row" is a row of the step (a sequence), not a matrix row.

static const int BLOCK = 256;
static const int GEN_CAP = 4096;  // tokens one wk_generate call can produce per row

__device__ __forceinline__ float kv_get(const float* p) { return *p; }
__device__ __forceinline__ float kv_get(const __nv_bfloat16* p) { return __bfloat162float(*p); }
__device__ __forceinline__ void kv_put(float* p, float v) { *p = v; }
__device__ __forceinline__ void kv_put(__nv_bfloat16* p, float v) { *p = __float2bfloat16(v); }

__device__ float block_sum(float v, float* sh) {
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = v;
    __syncthreads();
    float t = 0;
    for (int i = 0; i < BLOCK / 32; i++) t += sh[i];
    __syncthreads();
    return t;
}

__device__ float block_max(float v, float* sh) {
    for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = v;
    __syncthreads();
    float t = sh[0];
    for (int i = 1; i < BLOCK / 32; i++) t = fmaxf(t, sh[i]);
    __syncthreads();
    return t;
}

// x[row] = embed[token[row]]. Grid: rows.
__global__ void k_embed(const __nv_bfloat16* __restrict__ E, const int32_t* __restrict__ tok, float* __restrict__ x, int H) {
    const __nv_bfloat16* e = E + (size_t)tok[blockIdx.x] * H;
    float* out = x + (size_t)blockIdx.x * H;
    for (int i = threadIdx.x; i < H; i += BLOCK) out[i] = __bfloat162float(e[i]);
}

// y = x * rsqrt(mean(x^2) + eps) * w. Grid: rows.
__global__ void k_rmsnorm(const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ y, int H, float eps) {
    __shared__ float sh[BLOCK / 32];
    const float* in = x + (size_t)blockIdx.x * H;
    float* out = y + (size_t)blockIdx.x * H;
    float ss = 0;
    for (int i = threadIdx.x; i < H; i += BLOCK) ss += in[i] * in[i];
    float r = rsqrtf(block_sum(ss, sh) / H + eps);
    for (int i = threadIdx.x; i < H; i += BLOCK) out[i] = in[i] * r * w[i];
}

// y[r][n] = sum_k W[n][k] * x[r][k] (+ bias[n]). One warp per output n; each lane reads 16 bytes
// (8 bf16) of the weight row per step. K must be a multiple of 8.
__global__ void k_gemv(const __nv_bfloat16* __restrict__ W, const float* __restrict__ x, const float* __restrict__ bias,
                       float* __restrict__ y, int N, int K, int R) {
    int n = (blockIdx.x * BLOCK + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (n >= N) return;
    const uint4* w = reinterpret_cast<const uint4*>(W + (size_t)n * K);
    float acc[WK_MAX_ROWS] = {};
    for (int i = lane; i < K / 8; i += 32) {
        uint4 wv = w[i];
        const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(&wv);
        float wf[8];
        for (int j = 0; j < 4; j++) {
            float2 f = __bfloat1622float2(w2[j]);
            wf[2 * j] = f.x;
            wf[2 * j + 1] = f.y;
        }
        for (int r = 0; r < R; r++) {
            const float* xr = x + (size_t)r * K + 8 * i;
            for (int j = 0; j < 8; j++) acc[r] += wf[j] * xr[j];
        }
    }
    for (int r = 0; r < R; r++) {
        float v = acc[r];
        for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
        if (lane == 0) y[(size_t)r * N + n] = v + (bias ? bias[n] : 0.0f);
    }
}

struct Dims {
    int layers, q_heads, kv_heads, head_dim, max_ctx;
    __host__ __device__ int hidden() const { return q_heads * head_dim; }
    __host__ __device__ int kv_dim() const { return kv_heads * head_dim; }
    __host__ __device__ int qkv_dim() const { return hidden() + 2 * kv_dim(); }
    __host__ __device__ size_t kv_at(int slot, int layer, int pos) const { return (((size_t)slot * layers + layer) * max_ctx + pos) * kv_dim(); }
};

// RoPE on q (in place) and k, then store k and v at (slot, layer, pos). Pairs are (j, j + head_dim/2)
// as in HF's rotate_half. Grid: rows.
template <class KV>
__global__ void k_rope_kv(float* __restrict__ qkv, const int32_t* __restrict__ slot, const int32_t* __restrict__ pos,
                          const float* __restrict__ inv_freq, KV* __restrict__ K, KV* __restrict__ V, int layer, Dims d) {
    int r = blockIdx.x, hd = d.head_dim, half = hd / 2;
    float* q = qkv + (size_t)r * d.qkv_dim();  // q heads, then k heads, then v heads
    const float* v = q + d.hidden() + d.kv_dim();
    size_t at = d.kv_at(slot[r], layer, pos[r]);
    float p = (float)pos[r];
    for (int i = threadIdx.x; i < (d.q_heads + d.kv_heads) * half; i += BLOCK) {
        int head = i / half, j = i % half;
        float s, c;
        sincosf(p * inv_freq[j], &s, &c);
        float* hp = q + head * hd;
        float a = hp[j], b = hp[j + half];
        float ra = a * c - b * s, rb = b * c + a * s;
        if (head < d.q_heads) {
            hp[j] = ra;
            hp[j + half] = rb;
        } else {
            KV* k = K + at + (head - d.q_heads) * hd;
            kv_put(k + j, ra);
            kv_put(k + j + half, rb);
        }
    }
    for (int i = threadIdx.x; i < d.kv_dim(); i += BLOCK) kv_put(V + at + i, v[i]);
}

// out[r][h] = softmax(q_h . K[0..pos] / sqrt(head_dim)) @ V[0..pos]. Grid: (q_heads, rows).
template <class KV>
__global__ void k_attn(const float* __restrict__ qkv, const int32_t* __restrict__ slot, const int32_t* __restrict__ pos,
                       const KV* __restrict__ K, const KV* __restrict__ V, float* __restrict__ scores,
                       float* __restrict__ out, int layer, Dims d) {
    __shared__ float sh[BLOCK / 32];
    int h = blockIdx.x, r = blockIdx.y, hd = d.head_dim, kvd = d.kv_dim();
    int g = h / (d.q_heads / d.kv_heads), T = pos[r] + 1;
    const float* q = qkv + (size_t)r * d.qkv_dim() + h * hd;
    size_t base = d.kv_at(slot[r], layer, 0) + g * hd;
    float* sc = scores + ((size_t)r * d.q_heads + h) * d.max_ctx;
    float scale = rsqrtf((float)hd);

    float m = -INFINITY;
    for (int t = threadIdx.x; t < T; t += BLOCK) {
        const KV* k = K + base + (size_t)t * kvd;
        float dot = 0;
        for (int j = 0; j < hd; j++) dot += q[j] * kv_get(k + j);
        sc[t] = dot * scale;
        m = fmaxf(m, sc[t]);
    }
    m = block_max(m, sh);
    float sum = 0;
    for (int t = threadIdx.x; t < T; t += BLOCK) {
        sc[t] = expf(sc[t] - m);
        sum += sc[t];
    }
    sum = block_sum(sum, sh);
    for (int j = threadIdx.x; j < hd; j += BLOCK) {
        float a = 0;
        for (int t = 0; t < T; t++) a += sc[t] * kv_get(V + base + (size_t)t * kvd + j);
        out[(size_t)r * d.hidden() + h * hd + j] = a / sum;
    }
}

__global__ void k_add(float* __restrict__ x, const float* __restrict__ y, int n) {
    int i = blockIdx.x * BLOCK + threadIdx.x;
    if (i < n) x[i] += y[i];
}

// act[i] = silu(gate_i) * up_i, with gate and up interleaved in gu. n = rows * mlp.
__global__ void k_silu_mul(const float* __restrict__ gu, float* __restrict__ act, int n) {
    int i = blockIdx.x * BLOCK + threadIdx.x;
    if (i >= n) return;
    float g = gu[2 * i];
    act[i] = g / (1.0f + expf(-g)) * gu[2 * i + 1];
}

__global__ void k_to_bf16(const float* __restrict__ x, __nv_bfloat16* __restrict__ out, size_t n) {
    size_t i = (size_t)blockIdx.x * BLOCK + threadIdx.x;
    if (i < n) out[i] = __float2bfloat16(x[i]);
}

__global__ void k_widen(const __nv_bfloat16* __restrict__ w, float* __restrict__ out, size_t n) {
    size_t i = (size_t)blockIdx.x * BLOCK + threadIdx.x;
    if (i < n) out[i] = __bfloat162float(w[i]);
}

// y[r][n] += bias[n]. Grid: rows.
__global__ void k_add_bias(float* __restrict__ y, const float* __restrict__ bias, int N) {
    float* out = y + (size_t)blockIdx.x * N;
    for (int i = threadIdx.x; i < N; i += BLOCK) out[i] += bias[i];
}

// --- Wide attention: tensor-core GEMMs over tiles of positions (bf16 KV cache only). ---
// For one kv head and a tile of consecutive positions, the q heads that share the kv head are the
// rows of one matrix: scores = Q K^T, weights = masked softmax, output = weights V.

#ifndef WK_WA_POS
#define WK_WA_POS 128
#endif
static const int WA_POS = WK_WA_POS;  // positions per tile
static const int FL_BLOCKS = 96;                 // blocks the fused attention aims to run at once
static const size_t FL_PART = (size_t)4 << 20;   // floats of partial results it may use for that

// qb[kv head][row][head in group][head_dim] = bf16(q after RoPE). Grid: rows.
__global__ void k_wa_pack_q(const float* __restrict__ qkv, __nv_bfloat16* __restrict__ qb, int rows, Dims d) {
    const int r = blockIdx.x, hd = d.head_dim, G = d.q_heads / d.kv_heads;
    const float* q = qkv + (size_t)r * d.qkv_dim();
    for (int i = threadIdx.x; i < d.hidden(); i += BLOCK) {
        int h = i / hd, j = i % hd;
        qb[(((size_t)(h / G) * rows + r) * G + h % G) * hd + j] = __float2bfloat16(q[i]);
    }
}

// One block per score row: P = exp((S - max) * scale) over the first `valid` positions, zero after,
// and inv[row] = 1 / sum(P); the division is applied to the row's output (k_wa_scatter). Row `row`
// belongs to tile position row / G, which may attend to first_valid + row / G positions.
__global__ void k_wa_softmax(const float* __restrict__ S, __nv_bfloat16* __restrict__ P, float* __restrict__ inv, int T, int G,
                             int first_valid, float scale) {
    __shared__ float sh[BLOCK / 32];
    const int valid = first_valid + blockIdx.x / G;
    const float* s = S + (size_t)blockIdx.x * T;
    __nv_bfloat16* p = P + (size_t)blockIdx.x * T;
    float m = -INFINITY;
    for (int t = threadIdx.x; t < valid; t += BLOCK) m = fmaxf(m, s[t]);
    m = block_max(m, sh);
    float sum = 0;
    for (int t = threadIdx.x; t < T; t += BLOCK) {
        __nv_bfloat16 e = __float2bfloat16(t < valid ? expf((s[t] - m) * scale) : 0.0f);
        p[t] = e;
        sum += __bfloat162float(e);
    }
    sum = block_sum(sum, sh);
    if (threadIdx.x == 0) inv[blockIdx.x] = 1.0f / sum;
}

// attn[row][heads of the kv head] = og[position in tile][group * head_dim] * inv. Grid: tile positions.
__global__ void k_wa_scatter(const float* __restrict__ og, const float* __restrict__ inv, float* __restrict__ attn, int kvh, Dims d) {
    const int hd = d.head_dim, G = d.q_heads / d.kv_heads, n = G * hd;
    const float* in = og + (size_t)blockIdx.x * n;
    float* out = attn + (size_t)blockIdx.x * d.hidden() + kvh * n;
    for (int i = threadIdx.x; i < n; i += BLOCK) out[i] = in[i] * inv[blockIdx.x * G + i / hd];
}

// --- Fused kernels of the wide path (bf16 KV cache, no taps). The residual stream x stays fp32; what
// enters a product is produced directly in bf16, so no fp32 copy of it is written and read back.

// x[r] = embed[token[r]];  hb[r] = bf16(rmsnorm(x[r]) * g). Grid: rows.
__global__ void k_w_embed_norm(const __nv_bfloat16* __restrict__ E, const int32_t* __restrict__ tok, float* __restrict__ x,
                               const float* __restrict__ g, __nv_bfloat16* __restrict__ hb, int H, float eps) {
    __shared__ float sh[BLOCK / 32];
    const __nv_bfloat16* e = E + (size_t)tok[blockIdx.x] * H;
    float* xr = x + (size_t)blockIdx.x * H;
    __nv_bfloat16* out = hb + (size_t)blockIdx.x * H;
    float ss = 0;
    for (int i = threadIdx.x; i < H; i += BLOCK) {
        float v = __bfloat162float(e[i]);
        xr[i] = v;
        ss += v * v;
    }
    float r = rsqrtf(block_sum(ss, sh) / H + eps);
    for (int i = threadIdx.x; i < H; i += BLOCK) out[i] = __float2bfloat16(__bfloat162float(e[i]) * r * g[i]);
}

// x[r] += y[r];  hb[r] = bf16(rmsnorm(x[r]) * g). Grid: rows.
__global__ void k_w_add_norm(float* __restrict__ x, const float* __restrict__ y, const float* __restrict__ g,
                             __nv_bfloat16* __restrict__ hb, int H, float eps) {
    __shared__ float sh[BLOCK / 32];
    float* xr = x + (size_t)blockIdx.x * H;
    const float* yr = y + (size_t)blockIdx.x * H;
    __nv_bfloat16* out = hb + (size_t)blockIdx.x * H;
    float ss = 0;
    for (int i = threadIdx.x; i < H; i += BLOCK) {
        float v = xr[i] + yr[i];
        xr[i] = v;
        ss += v * v;
    }
    float r = rsqrtf(block_sum(ss, sh) / H + eps);
    for (int i = threadIdx.x; i < H; i += BLOCK) out[i] = __float2bfloat16(xr[i] * r * g[i]);
}

// rope[pos][j] = (cos, sin)(pos * inv_freq[j]), computed on the device with the same sincosf the
// kernels that do not use the table call. Grid: positions.
__global__ void k_rope_table(const float* __restrict__ inv_freq, float2* __restrict__ rope, int half) {
    for (int j = threadIdx.x; j < half; j += BLOCK) {
        float s, c;
        sincosf((float)blockIdx.x * inv_freq[j], &s, &c);
        rope[(size_t)blockIdx.x * half + j] = make_float2(c, s);
    }
}

// After the q/k/v product: bias, RoPE, K and V into the cache, q packed in bf16 for the attention
// GEMMs as qb[kv head][row][head in group][head_dim]. Grid: rows.
__global__ void k_w_qkv_post(const float* __restrict__ qkv, const float* __restrict__ bias, const int32_t* __restrict__ pos,
                             const float2* __restrict__ rope, __nv_bfloat16* __restrict__ K, __nv_bfloat16* __restrict__ V,
                             __nv_bfloat16* __restrict__ qb, const int32_t* __restrict__ slot, int layer, int rows, Dims d) {
    const int r = blockIdx.x, hd = d.head_dim, half = hd / 2, H = d.hidden(), G = d.q_heads / d.kv_heads;
    const float* in = qkv + (size_t)r * d.qkv_dim();
    const size_t at = d.kv_at(slot[r], layer, pos[r]);
    for (int i = threadIdx.x; i < (d.q_heads + d.kv_heads) * half; i += BLOCK) {
        int head = i / half, j = i % half, n1 = head * hd + j, n2 = n1 + half;
        const float2 cs = rope[(size_t)pos[r] * half + j];
        const float c = cs.x, s = cs.y;
        float a = in[n1] + bias[n1], b = in[n2] + bias[n2];
        __nv_bfloat16 ra = __float2bfloat16(a * c - b * s), rb = __float2bfloat16(b * c + a * s);
        if (head < d.q_heads) {
            __nv_bfloat16* q = qb + (((size_t)(head / G) * rows + r) * G + head % G) * hd;
            q[j] = ra;
            q[j + half] = rb;
        } else {
            K[at + n1 - H] = ra;
            K[at + n2 - H] = rb;
        }
    }
    const int v0 = H + d.kv_dim();
    for (int i = threadIdx.x; i < d.kv_dim(); i += BLOCK) V[at + i] = __float2bfloat16(in[v0 + i] + bias[v0 + i]);
}

// The same for a decode step: q stays fp32, in q[rows][hidden] (the attention kernel packs it).
__global__ void k_t_qkv_post(const float* __restrict__ qkv, const float* __restrict__ bias, const int32_t* __restrict__ pos,
                             const float* __restrict__ inv_freq, __nv_bfloat16* __restrict__ K, __nv_bfloat16* __restrict__ V,
                             float* __restrict__ q, const int32_t* __restrict__ slot, int layer, Dims d) {
    const int r = blockIdx.x, hd = d.head_dim, half = hd / 2, H = d.hidden();
    const float* in = qkv + (size_t)r * d.qkv_dim();
    const size_t at = d.kv_at(slot[r], layer, pos[r]);
    const float p = (float)pos[r];
    for (int i = threadIdx.x; i < (d.q_heads + d.kv_heads) * half; i += BLOCK) {
        int head = i / half, j = i % half, n1 = head * hd + j, n2 = n1 + half;
        float s, c;
        sincosf(p * inv_freq[j], &s, &c);
        float a = in[n1] + bias[n1], b = in[n2] + bias[n2];
        float ra = a * c - b * s, rb = b * c + a * s;
        if (head < d.q_heads) {
            q[(size_t)r * H + n1] = ra;
            q[(size_t)r * H + n2] = rb;
        } else {
            K[at + n1 - H] = __float2bfloat16(ra);
            K[at + n2 - H] = __float2bfloat16(rb);
        }
    }
    const int v0 = H + d.kv_dim();
    for (int i = threadIdx.x; i < d.kv_dim(); i += BLOCK) V[at + i] = __float2bfloat16(in[v0 + i] + bias[v0 + i]);
}

// ab[row][heads of the kv head] = bf16(og[position in tile] * inv). Grid: tile positions.
__global__ void k_w_scatter(const float* __restrict__ og, const float* __restrict__ inv, __nv_bfloat16* __restrict__ ab, int kvh, Dims d) {
    const int hd = d.head_dim, G = d.q_heads / d.kv_heads, n = G * hd;
    const float* in = og + (size_t)blockIdx.x * n;
    __nv_bfloat16* out = ab + (size_t)blockIdx.x * d.hidden() + kvh * n;
    for (int i = threadIdx.x; i < n; i += BLOCK) out[i] = __float2bfloat16(in[i] * inv[blockIdx.x * G + i / hd]);
}

// actb[r][c0 + i] = bf16(silu(gate) * up) for `units` MLP units starting at c0: gu is [rows][2 * units]
// with gate and up interleaved, actb is [rows][mlp]. Grid: (ceil(units / BLOCK), rows): the row and
// the unit come from the block and thread indices (a division per element was most of this kernel).
// The same with gate and up in bf16, as the wide path's GEMM leaves them: the fp32 pair is 8 bytes
// per unit to write and read back, per row and layer, and that traffic was 8% of a long prefill.
__global__ void k_w_silu_b(const __nv_bfloat16* __restrict__ gu, __nv_bfloat16* __restrict__ actb, int units, int mlp, int c0) {
    const int i = blockIdx.x * BLOCK + threadIdx.x, r = blockIdx.y;
    if (i >= units) return;
    const float2 v = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(gu + (size_t)r * 2 * units)[i]);
    actb[(size_t)r * mlp + c0 + i] = __float2bfloat16(v.x / (1.0f + expf(-v.x)) * v.y);
}

__global__ void k_w_silu(const float* __restrict__ gu, __nv_bfloat16* __restrict__ actb, int units, int mlp, int c0) {
    const int i = blockIdx.x * BLOCK + threadIdx.x, r = blockIdx.y;
    if (i >= units) return;
    const float2 v = reinterpret_cast<const float2*>(gu + (size_t)r * 2 * units)[i];
    actb[(size_t)r * mlp + c0 + i] = __float2bfloat16(v.x / (1.0f + expf(-v.x)) * v.y);
}

#include "fast.cuh"
#include "flash.cuh"
#include "tc.cuh"

// ---------------------------------------------------------------------------------------------

// WK_PROFILE=1: wk_step_fast waits after every kernel and wk_destroy prints the time per kernel kind.
// wk_prefill does the same per operation of the wide path.
enum { P_EMBED, P_QKV, P_ATTN, P_O, P_GATE_UP, P_DOWN, P_LM_HEAD, P_COUNT,
       W_NORM = 0, W_QKV, W_ROPE, W_ATT_PACK, W_ATT_SCORE, W_ATT_SOFTMAX, W_ATT_OUT, W_O, W_ADD, W_GATE_UP, W_SILU, W_DOWN, W_COUNT };
static const char* const P_NAMES[P_COUNT] = {"embed", "qkv", "attn", "o", "gate_up", "down", "lm_head"};
static const char* const W_NAMES[W_COUNT] = {"norm(+add)", "qkv", "rope+kv", "att_pack", "att_score", "att_softmax", "att_out", "o", "add", "gate_up", "silu", "down"};
static double g_wprof[W_COUNT];
static long g_wprof_tokens;
static bool g_wprof_on;

struct wk_engine;
static void wprof(wk_engine* e, int kind);
static double g_prof[P_COUNT];
static long g_prof_steps;
static bool g_prof_on;

static void prof(wk_engine* e, int kind) {
    static std::chrono::steady_clock::time_point last;
    if (!g_prof_on) return;
    cudaStreamSynchronize(e->s);
    auto now = std::chrono::steady_clock::now();
    if (kind >= 0) g_prof[kind] += std::chrono::duration<double, std::micro>(now - last).count();
    last = now;
}

static int blocks_for(size_t n) { return (int)((n + BLOCK - 1) / BLOCK); }

enum Path { NARROW, WIDE, WIDE_EXACT };

static const size_t WIDE_W_BYTES = 256u << 20;  // fp32 slab of widened weights for WK_EXACT

// y[r] = W x[r] (+ bias) for R rows. W is bf16 [N][K] row-major; x [R][K] and y [R][N] are fp32.
static int linear(wk_engine* e, Path path, const __nv_bfloat16* W, const float* bias, const float* x, float* y, int N, int K, int R) {
    cudaStream_t s = e->s;
    if (path == NARROW) {
        WK_LAUNCH(k_gemv<<<blocks_for((size_t)N * 32), BLOCK, 0, s>>>(W, x, bias, y, N, K, R));
        return 0;
    }
    // Column-major view for cuBLAS: W is K x N, x is K x R, y is N x R, and y = W^T x.
    const float one = 1, zero = 0;
    if (path == WIDE) {
        WK_LAUNCH(k_to_bf16<<<blocks_for((size_t)R * K), BLOCK, 0, s>>>(x, e->xb, (size_t)R * K));
        WK_BLAS(cublasGemmEx(e->blas, CUBLAS_OP_T, CUBLAS_OP_N, N, R, K, &one, W, CUDA_R_16BF, K, e->xb, CUDA_R_16BF, K,
                             &zero, y, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    } else {
        if (!e->wide_w) {
            e->wide_w_count = WIDE_W_BYTES / sizeof(float);
            WK_CUDA(cudaMalloc(&e->wide_w, WIDE_W_BYTES));
        }
        int slab = (int)(e->wide_w_count / K);  // weight rows widened per SGEMM
        for (int n0 = 0; n0 < N; n0 += slab) {
            int m = N - n0 < slab ? N - n0 : slab;
            WK_LAUNCH(k_widen<<<blocks_for((size_t)m * K), BLOCK, 0, s>>>(W + (size_t)n0 * K, e->wide_w, (size_t)m * K));
            WK_BLAS(cublasSgemm(e->blas, CUBLAS_OP_T, CUBLAS_OP_N, m, R, K, &one, e->wide_w, K, x, K, &zero, y + n0, N));
        }
    }
    if (bias) WK_LAUNCH(k_add_bias<<<R, BLOCK, 0, s>>>(y, bias, N));
    return 0;
}

static int tap(const wk_engine* e, float* host, size_t index, const float* dev, size_t count) {
    if (!host) return 0;
    WK_CUDA(cudaMemcpyAsync(host + index * count, dev, count * sizeof(float), cudaMemcpyDeviceToHost, e->s));
    WK_CUDA(cudaStreamSynchronize(e->s));
    return 0;
}

// Attention for R consecutive positions of one slot starting at pos0, reading e->qkv (after RoPE)
// and the bf16 KV cache, writing e->attn.
// With `fused`, e->wa_q is already packed and the output goes to e->ab in bf16.
static int wide_attention(wk_engine* e, int R, int slot, int pos0, int layer, const Dims& d, bool fused = false) {
    typedef __nv_bfloat16 bf;
    cudaStream_t s = e->s;
    const int hd = d.head_dim, kvd = d.kv_dim(), G = d.q_heads / d.kv_heads;
    const float one = 1, zero = 0;
    if (fused && (hd == 64 || hd == 128) && !getenv("WK_NO_FLASH")) {  // one launch per kv head, scores kept in registers
        const int rows = R * G, blocks = (rows + 16 * FA_WARPS - 1) / (16 * FA_WARPS), T = pos0 + R;
        // Few rows over a long context: cut the keys into segments so that about FL_BLOCKS blocks
        // run at once (see k_w_flash_attn). Segments are multiples of 16 keys and at least 256.
        int segs = 1, seg_len = (T + 15) / 16 * 16;
        if (blocks < FL_BLOCKS && (size_t)rows * hd <= FL_PART / (FL_BLOCKS / blocks + 1)) {
            segs = std::min((T + 255) / 256, FL_BLOCKS / blocks);
            if (segs > 1) seg_len = ((T + segs - 1) / segs + 15) / 16 * 16;
            segs = (T + seg_len - 1) / seg_len;
        }
        for (int kvh = 0; kvh < d.kv_heads; kvh++) {
            const bf* Qg = e->wa_q + (size_t)kvh * rows * hd;
            const bf* Kg = (const bf*)e->K + d.kv_at(slot, layer, 0) + kvh * hd;
            const bf* Vg = (const bf*)e->V + d.kv_at(slot, layer, 0) + kvh * hd;
            const dim3 grid(blocks, segs);
#define FLASH(HD, SPLIT) WK_LAUNCH(k_w_flash_attn<HD, SPLIT><<<grid, FA_WARPS * 32, 0, s>>>(Qg, Kg, Vg, e->ab, rows, G, kvd, d.hidden(), kvh * G * hd, pos0, segs, seg_len, e->fl_part, e->fl_stat, nullptr))
            if (hd == 128 && segs > 1) FLASH(128, true);
            else if (hd == 128) FLASH(128, false);
            else if (segs > 1) FLASH(64, true);
            else FLASH(64, false);
#undef FLASH
            if (segs > 1)
                WK_LAUNCH(k_w_flash_merge<<<rows, hd, 0, s>>>(e->fl_part, e->fl_stat, e->ab, rows, segs, hd, G, d.hidden(), kvh * G * hd));
        }
        wprof(e, W_ATT_OUT);
        return 0;
    }
    if (!fused) WK_LAUNCH(k_wa_pack_q<<<R, BLOCK, 0, s>>>(e->qkv, e->wa_q, R, d));
    wprof(e, W_ATT_PACK);
    for (int kvh = 0; kvh < d.kv_heads; kvh++) {
        const bf* Kg = (const bf*)e->K + d.kv_at(slot, layer, 0) + kvh * hd;
        const bf* Vg = (const bf*)e->V + d.kv_at(slot, layer, 0) + kvh * hd;
        for (int p0 = 0; p0 < R; p0 += WA_POS) {
            const int np = R - p0 < WA_POS ? R - p0 : WA_POS, rows = np * G, T = pos0 + p0 + np;
            const bf* Q = e->wa_q + ((size_t)kvh * R + p0) * G * hd;
            // Column-major views: K, V are hd x T (lda = kv_dim), Q is hd x rows, S and P are T x rows.
            WK_BLAS(cublasGemmEx(e->blas, CUBLAS_OP_T, CUBLAS_OP_N, T, rows, hd, &one, Kg, CUDA_R_16BF, kvd, Q, CUDA_R_16BF, hd,
                                 &zero, e->wa_s, CUDA_R_32F, T, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            wprof(e, W_ATT_SCORE);
            WK_LAUNCH(k_wa_softmax<<<rows, BLOCK, 0, s>>>(e->wa_s, e->wa_p, e->wa_inv, T, G, pos0 + p0 + 1, rsqrtf((float)hd)));
            wprof(e, W_ATT_SOFTMAX);
            WK_BLAS(cublasGemmEx(e->blas, CUBLAS_OP_N, CUBLAS_OP_N, hd, rows, T, &one, Vg, CUDA_R_16BF, kvd, e->wa_p, CUDA_R_16BF, T,
                                 &zero, e->wa_o, CUDA_R_32F, hd, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            if (fused)
                WK_LAUNCH(k_w_scatter<<<np, BLOCK, 0, s>>>(e->wa_o, e->wa_inv, e->ab + (size_t)p0 * d.hidden(), kvh, d));
            else
                WK_LAUNCH(k_wa_scatter<<<np, BLOCK, 0, s>>>(e->wa_o, e->wa_inv, e->attn + (size_t)p0 * d.hidden(), kvh, d));
            wprof(e, W_ATT_OUT);
        }
    }
    return 0;
}

// The wide path's products, y[R][N] = xb[R][K] . W[N][K]^T with bf16 operands and fp32 accumulation.
//
// cublasGemmEx picks its kernel by a heuristic, and on this GPU that is not the fastest kernel cuBLAS
// has for our shapes. spikes/gemm/lt.cu times every cublasLt configuration (algorithm, tile, stages)
// for one shape and prints the fastest whose result has the same bits as cublasGemmEx's; LT_TABLE is
// built from its output and from timings inside the engine. Which configuration wins depends on the
// row count: an entry serves its row count and up to 127 rows more, and shapes that have none
// (fewer than 768 rows, other models) use the default.
//   WK_LT=0 turns the table off;  WK_LT=ID,TILE,STAGES uses one configuration for everything.
struct LtEntry { int N, K, R, id, tile, stages; };
static const LtEntry LT_TABLE[] = {
#include "lt_table.inc"
};
static const int LT_MIN_ROWS = 512, LT_ROW_STEP = 128;

static const wk_engine::LtPlan& lt_plan(wk_engine* e, cudaDataType out_t, int N, int K, int R,
                                        const cublasLtMatrixLayoutOpaque_t* la, const cublasLtMatrixLayoutOpaque_t* lb,
                                        const cublasLtMatrixLayoutOpaque_t* ld) {
    const uint64_t key = ((uint64_t)N << 40) | ((uint64_t)K << 20) | ((uint64_t)R << 1) | (out_t == CUDA_R_16BF);
    auto it = e->lt_plans.find(key);
    if (it != e->lt_plans.end()) return it->second;
    wk_engine::LtPlan plan{};
    int id = -1, tile = 0, stages = 0;
    const char* env = getenv("WK_LT");
    if (env && strcmp(env, "0")) sscanf(env, "%d,%d,%d", &id, &tile, &stages);
    else if (!env)
        for (const LtEntry& t : LT_TABLE)
            if (t.N == N && t.K == K && t.R == R / LT_ROW_STEP * LT_ROW_STEP) id = t.id, tile = t.tile, stages = t.stages;
    cublasLtMatmulHeuristicResult_t res{};
    if (id >= 0 && cublasLtMatmulAlgoInit(e->lt, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16BF, CUDA_R_16BF, out_t, out_t, id, &plan.algo) == CUBLAS_STATUS_SUCCESS &&
        cublasLtMatmulAlgoConfigSetAttribute(&plan.algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof tile) == CUBLAS_STATUS_SUCCESS &&
        cublasLtMatmulAlgoConfigSetAttribute(&plan.algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stages, sizeof stages) == CUBLAS_STATUS_SUCCESS &&
        cublasLtMatmulAlgoCheck(e->lt, e->lt_desc, (cublasLtMatrixLayout_t)la, (cublasLtMatrixLayout_t)lb, (cublasLtMatrixLayout_t)ld,
                                (cublasLtMatrixLayout_t)ld, &plan.algo, &res) == CUBLAS_STATUS_SUCCESS &&
        res.state == CUBLAS_STATUS_SUCCESS && res.workspaceSize == 0)
        plan.use = true;
    return e->lt_plans.emplace(key, plan).first->second;
}

static int gemm_wide(wk_engine* e, const __nv_bfloat16* W, const __nv_bfloat16* xb, void* y, cudaDataType out_t, int N, int K, int R) {
    const float one = 1, zero = 0;
    if (R >= LT_MIN_ROWS && e->lt) {
        cublasLtMatrixLayoutOpaque_t la, lb, ld;
        WK_BLAS(cublasLtMatrixLayoutInit(&la, CUDA_R_16BF, K, N, K));
        WK_BLAS(cublasLtMatrixLayoutInit(&lb, CUDA_R_16BF, K, R, K));
        WK_BLAS(cublasLtMatrixLayoutInit(&ld, out_t, N, R, N));
        const wk_engine::LtPlan& plan = lt_plan(e, out_t, N, K, R, &la, &lb, &ld);
        if (plan.use) {
            WK_BLAS(cublasLtMatmul(e->lt, e->lt_desc, &one, W, &la, xb, &lb, &zero, y, &ld, y, &ld, &plan.algo, nullptr, 0, e->s));
            return 0;
        }
    }
    WK_BLAS(cublasGemmEx(e->blas, CUBLAS_OP_T, CUBLAS_OP_N, N, R, K, &one, W, CUDA_R_16BF, K, xb, CUDA_R_16BF, K, &zero, y,
                         out_t, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    return 0;
}

// yb = W xb for R rows, everything bf16 (fp32 accumulation, the result rounded).
static int gemm_bf16_out(wk_engine* e, const __nv_bfloat16* W, const __nv_bfloat16* xb, __nv_bfloat16* yb, int N, int K, int R) {
    return gemm_wide(e, W, xb, yb, CUDA_R_16BF, N, K, R);
}

// y = W xb for R rows, xb already bf16; y fp32.
static int gemm_bf16(wk_engine* e, const __nv_bfloat16* W, const __nv_bfloat16* xb, float* y, int N, int K, int R) {
    return gemm_wide(e, W, xb, y, CUDA_R_32F, N, K, R);
}

// The MLP's first product is done for a slice of its units at a time, over all rows: the fp32 gate
// and up values of a slice (rows x 2 x units x 4 bytes) are then read back for SiLU while still in
// the cache instead of from memory. Measured at 1024 rows: 8 slices (9 MB each) 5% faster overall
// than one; with few rows one slice is already small, and more only add calls.
#ifndef WK_GATE_BF16
#define WK_GATE_BF16 0
#endif
#ifndef WK_MLP_SLICE_BYTES
#define WK_MLP_SLICE_BYTES (9u << 20)
#endif
#ifndef WK_MLP_SLICE_UNITS
#define WK_MLP_SLICE_UNITS 1120
#endif

// The wide path with fused element-wise kernels: same arithmetic as forward(WIDE) with a bf16 KV
// cache, no taps. Leaves the residual stream in e->x and its final norm, in bf16, in e->hb.
static int wide_forward(wk_engine* e, int R, int slot, int pos0) {
    const wk_config& c = e->c;
    cudaStream_t s = e->s;
    const int H = c.hidden, I = c.mlp;
    const Dims d{(int)c.layers, (int)c.q_heads, (int)c.kv_heads, (int)c.head_dim, (int)e->o.max_ctx};
    typedef __nv_bfloat16 bf;

    wprof(e, -1);
    WK_LAUNCH(k_w_embed_norm<<<R, BLOCK, 0, s>>>(e->w<bf>(T_EMBED), e->tok, e->x, e->w<float>(0, L_NORM1), e->hb, H, c.rms_eps));
    wprof(e, W_NORM);
    for (int l = 0; l < (int)c.layers; l++) {
        if (gemm_bf16(e, e->w<bf>(l, L_QKV_W), e->hb, e->qkv, e->qkv_dim, H, R)) return -1;
        wprof(e, W_QKV);
        WK_LAUNCH(k_w_qkv_post<<<R, BLOCK, 0, s>>>(e->qkv, e->w<float>(l, L_QKV_B), e->pos, e->rope, (bf*)e->K, (bf*)e->V, e->wa_q, e->slot, l, R, d));
        wprof(e, W_ROPE);
        if (wide_attention(e, R, slot, pos0, l, d, true)) return -1;
        if (gemm_bf16(e, e->w<bf>(l, L_O_W), e->ab, e->y, H, H, R)) return -1;
        wprof(e, W_O);
        WK_LAUNCH(k_w_add_norm<<<R, BLOCK, 0, s>>>(e->x, e->y, e->w<float>(l, L_NORM2), e->hb, H, c.rms_eps));
        wprof(e, W_NORM);
        // From LT_MIN_ROWS rows on a slice is WK_MLP_SLICE_UNITS units whatever the row count, so
        // that the product has a shape of LT_TABLE. 1120 is an eighth of the MLP: measured at 4096
        // tokens, 800 ms against 815 to 870 for 896, 992, 1024, 1280, 1792 and 2240 units.
        const int slices = (int)(((size_t)R * 2 * I * sizeof(float) + WK_MLP_SLICE_BYTES - 1) / WK_MLP_SLICE_BYTES);
        const int units = R >= LT_MIN_ROWS ? WK_MLP_SLICE_UNITS : (I + slices - 1) / slices;
        for (int c0 = 0; c0 < I; c0 += units) {
            const int n = I - c0 < units ? I - c0 : units;
            if (WK_GATE_BF16) {
                if (gemm_bf16_out(e, e->w<bf>(l, L_GATE_UP_W) + (size_t)2 * c0 * H, e->hb, (bf*)e->gu, 2 * n, H, R)) return -1;
                wprof(e, W_GATE_UP);
                WK_LAUNCH(k_w_silu_b<<<dim3(blocks_for(n), R), BLOCK, 0, s>>>((bf*)e->gu, e->xb, n, I, c0));
            } else {
                if (gemm_bf16(e, e->w<bf>(l, L_GATE_UP_W) + (size_t)2 * c0 * H, e->hb, e->gu, 2 * n, H, R)) return -1;
                wprof(e, W_GATE_UP);
                WK_LAUNCH(k_w_silu<<<dim3(blocks_for(n), R), BLOCK, 0, s>>>(e->gu, e->xb, n, I, c0));
            }
            wprof(e, W_SILU);
        }
        if (gemm_bf16(e, e->w<bf>(l, L_DOWN_W), e->xb, e->y, H, I, R)) return -1;
        wprof(e, W_DOWN);
        const float* next = l + 1 < (int)c.layers ? e->w<float>(l + 1, L_NORM1) : e->w<float>(T_FINAL_NORM);
        WK_LAUNCH(k_w_add_norm<<<R, BLOCK, 0, s>>>(e->x, e->y, next, e->hb, H, c.rms_eps));
        wprof(e, W_NORM);
    }
    return 0;
}

// The decoder stack for R rows whose token / slot / pos are already in the device arrays. Leaves the
// final-norm output in e->h. Each row's K/V are stored before attention runs, so rows that are
// consecutive positions of one slot see each other causally.
static int forward(wk_engine* e, int R, Path path, const wk_taps* taps, int wide_slot = 0, int wide_pos0 = 0) {
    const wk_config& c = e->c;
    cudaStream_t s = e->s;
    const int H = c.hidden, I = c.mlp;
    const Dims d{(int)c.layers, (int)c.q_heads, (int)c.kv_heads, (int)c.head_dim, (int)e->o.max_ctx};
    const wk_taps none{};
    const wk_taps& t = taps ? *taps : none;
    typedef __nv_bfloat16 bf;

    wprof(e, -1);
    WK_LAUNCH(k_embed<<<R, BLOCK, 0, s>>>(e->w<bf>(T_EMBED), e->tok, e->x, H));
    for (int l = 0; l < (int)c.layers; l++) {
        if (tap(e, t.x_in, l, e->x, (size_t)R * H)) return -1;
        WK_LAUNCH(k_rmsnorm<<<R, BLOCK, 0, s>>>(e->x, e->w<float>(l, L_NORM1), e->h, H, c.rms_eps));
        if (tap(e, t.norm1, l, e->h, (size_t)R * H)) return -1;
        wprof(e, W_NORM);
        if (linear(e, path, e->w<bf>(l, L_QKV_W), e->w<float>(l, L_QKV_B), e->h, e->qkv, e->qkv_dim, H, R)) return -1;
        if (tap(e, t.qkv, l, e->qkv, (size_t)R * e->qkv_dim)) return -1;
        wprof(e, W_QKV);
        if (e->o.kv_bf16) {
            WK_LAUNCH(k_rope_kv<bf><<<R, BLOCK, 0, s>>>(e->qkv, e->slot, e->pos, e->inv_freq, (bf*)e->K, (bf*)e->V, l, d));
            wprof(e, W_ROPE);
            if (path == WIDE) {
                if (wide_attention(e, R, wide_slot, wide_pos0, l, d)) return -1;
            } else {
                WK_LAUNCH(k_attn<bf><<<dim3(c.q_heads, R), BLOCK, 0, s>>>(e->qkv, e->slot, e->pos, (bf*)e->K, (bf*)e->V, e->scores, e->attn, l, d));
            }
        } else {
            WK_LAUNCH(k_rope_kv<float><<<R, BLOCK, 0, s>>>(e->qkv, e->slot, e->pos, e->inv_freq, (float*)e->K, (float*)e->V, l, d));
            WK_LAUNCH(k_attn<float><<<dim3(c.q_heads, R), BLOCK, 0, s>>>(e->qkv, e->slot, e->pos, (float*)e->K, (float*)e->V, e->scores, e->attn, l, d));
        }
        if (tap(e, t.attn, l, e->attn, (size_t)R * H)) return -1;
        if (linear(e, path, e->w<bf>(l, L_O_W), nullptr, e->attn, e->y, H, H, R)) return -1;
        if (tap(e, t.o, l, e->y, (size_t)R * H)) return -1;
        wprof(e, W_O);
        WK_LAUNCH(k_add<<<blocks_for((size_t)R * H), BLOCK, 0, s>>>(e->x, e->y, R * H));
        wprof(e, W_ADD);

        WK_LAUNCH(k_rmsnorm<<<R, BLOCK, 0, s>>>(e->x, e->w<float>(l, L_NORM2), e->h, H, c.rms_eps));
        if (tap(e, t.norm2, l, e->h, (size_t)R * H)) return -1;
        wprof(e, W_NORM);
        if (linear(e, path, e->w<bf>(l, L_GATE_UP_W), nullptr, e->h, e->gu, 2 * I, H, R)) return -1;
        wprof(e, W_GATE_UP);
        WK_LAUNCH(k_silu_mul<<<blocks_for((size_t)R * I), BLOCK, 0, s>>>(e->gu, e->act, R * I));
        wprof(e, W_SILU);
        if (tap(e, t.act, l, e->act, (size_t)R * I)) return -1;
        if (linear(e, path, e->w<bf>(l, L_DOWN_W), nullptr, e->act, e->y, H, I, R)) return -1;
        if (tap(e, t.mlp, l, e->y, (size_t)R * H)) return -1;
        wprof(e, W_DOWN);
        WK_LAUNCH(k_add<<<blocks_for((size_t)R * H), BLOCK, 0, s>>>(e->x, e->y, R * H));
        wprof(e, W_ADD);
    }
    WK_LAUNCH(k_rmsnorm<<<R, BLOCK, 0, s>>>(e->x, e->w<float>(T_FINAL_NORM), e->h, H, c.rms_eps));
    return tap(e, t.final_norm, 0, e->h, (size_t)R * H);
}

static int upload_rows(wk_engine* e, const int32_t* tok, const int32_t* slot, const int32_t* pos, int R) {
    WK_CUDA(cudaMemcpyAsync(e->tok, tok, R * sizeof(int32_t), cudaMemcpyHostToDevice, e->s));
    WK_CUDA(cudaMemcpyAsync(e->slot, slot, R * sizeof(int32_t), cudaMemcpyHostToDevice, e->s));
    WK_CUDA(cudaMemcpyAsync(e->pos, pos, R * sizeof(int32_t), cudaMemcpyHostToDevice, e->s));
    return 0;
}

// Copies R rows of logits to the host, waits for the stream and reports any kernel fault.
static int fetch_logits(wk_engine* e, const float* dev, float* host, int R) {
    WK_CUDA(cudaMemcpyAsync(host, dev, (size_t)R * e->c.vocab * sizeof(float), cudaMemcpyDeviceToHost, e->s));
    WK_CUDA(cudaStreamSynchronize(e->s));
    WK_CUDA(cudaGetLastError());
    return 0;
}

static int32_t argmax_of(const float* lg, uint32_t vocab) {
    int best = 0;
    for (uint32_t v = 1; v < vocab; v++)
        if (lg[v] > lg[best]) best = (int)v;
    return best;
}

template <class T> static int dev_alloc(T** p, size_t count) {
    WK_CUDA(cudaMalloc(p, count * sizeof(T)));
    return 0;
}

static int create(const char* path, const wk_options* opt, wk_engine* e) {
    if (opt->slots == 0 || opt->max_ctx == 0) return fail("slots and max_ctx must be positive");
    if ((opt->max_ctx + ATT_TILE - 1) / ATT_TILE * ATT_MAX_GROUP > ATT_BLOCK)
        return fail("max_ctx %u is over the limit of %d", opt->max_ctx, ATT_BLOCK / ATT_MAX_GROUP * ATT_TILE);
    int fd = open(path, O_RDONLY);
    if (fd < 0) return fail("open %s: %s", path, strerror(errno));
    struct stat st;
    fstat(fd, &st);
    size_t size = (size_t)st.st_size;
    void* map = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (map == MAP_FAILED) return fail("mmap %s: %s", path, strerror(errno));
    struct Unmap {
        void* p;
        size_t n;
        ~Unmap() { munmap(p, n); }
    } unmap{map, size};

    const FileHeader* fh = static_cast<const FileHeader*>(map);
    if (size < sizeof(FileHeader) || memcmp(fh->magic, "WITNESS1", 8) != 0 || fh->version != 1)
        return fail("%s is not a witness weight file (version 1)", path);
    e->c = fh->cfg;
    e->o = *opt;
    const wk_config& c = e->c;
    if (fh->n_tensors != T_LAYER0 + T_PER_LAYER * c.layers) return fail("%s: unexpected tensor count", path);
    const uint64_t* offs = reinterpret_cast<const uint64_t*>(fh + 1);
    if (offs[fh->n_tensors] != size) return fail("%s: truncated (%zu of %llu bytes)", path, size, (unsigned long long)offs[fh->n_tensors]);
    if (c.q_heads * c.head_dim != c.hidden || c.q_heads % c.kv_heads || c.head_dim % 2 || c.hidden % 8 || c.mlp % 8 ||
        c.q_heads / c.kv_heads > ATT_MAX_GROUP || c.head_dim % 32 || c.head_dim > 128)
        return fail("%s: unsupported model shape", path);
    e->off.assign(offs, offs + fh->n_tensors);
    e->kv_dim = c.kv_heads * c.head_dim;
    e->qkv_dim = c.hidden + 2 * e->kv_dim;

    WK_CUDA(cudaStreamCreate(&e->s));
    WK_BLAS(cublasCreate(&e->blas));
    WK_BLAS(cublasSetStream(e->blas, e->s));
    WK_BLAS(cublasLtCreate(&e->lt));
    WK_BLAS(cublasLtMatmulDescCreate(&e->lt_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    {
        const cublasOperation_t transpose = CUBLAS_OP_T;
        WK_BLAS(cublasLtMatmulDescSetAttribute(e->lt_desc, CUBLASLT_MATMUL_DESC_TRANSA, &transpose, sizeof transpose));
    }
    WK_CUDA(cudaMalloc(&e->blob, size));
    WK_CUDA(cudaMemcpy(e->blob, map, size, cudaMemcpyHostToDevice));

    const size_t R = e->cap = opt->max_chunk > WK_MAX_ROWS ? opt->max_chunk : WK_MAX_ROWS;
    size_t kv = (size_t)opt->slots * c.layers * opt->max_ctx * e->kv_dim * (opt->kv_bf16 ? sizeof(__nv_bfloat16) : sizeof(float));
    WK_CUDA(cudaMalloc(&e->K, kv));
    WK_CUDA(cudaMalloc(&e->V, kv));
    if (dev_alloc(&e->x, R * c.hidden) || dev_alloc(&e->h, R * c.hidden) || dev_alloc(&e->qkv, R * e->qkv_dim) ||
        dev_alloc(&e->attn, R * c.hidden) || dev_alloc(&e->y, R * c.hidden) || dev_alloc(&e->gu, R * 2 * c.mlp) ||
        dev_alloc(&e->act, R * c.mlp) || dev_alloc(&e->logits, (size_t)WK_MAX_ROWS * c.vocab) ||
        dev_alloc(&e->scores, R * c.q_heads * opt->max_ctx) || dev_alloc(&e->q, (size_t)WK_MAX_ROWS * c.hidden) ||
        dev_alloc(&e->att_part, (size_t)WK_MAX_ROWS * ((opt->max_ctx + ATT_TILE - 1) / ATT_TILE) * c.hidden) ||
        dev_alloc(&e->att_stat, (size_t)WK_MAX_ROWS * c.kv_heads * ((opt->max_ctx + ATT_TILE - 1) / ATT_TILE) * 2 * ATT_MAX_GROUP) ||
        dev_alloc(&e->count, WK_MAX_ROWS) || dev_alloc(&e->gen_out, (size_t)WK_MAX_ROWS * GEN_CAP) ||
        dev_alloc(&e->inv_freq, c.head_dim / 2) || dev_alloc(&e->tok, R) || dev_alloc(&e->slot, R) || dev_alloc(&e->pos, R) ||
        dev_alloc(&e->xb, R * (c.mlp > c.hidden ? c.mlp : c.hidden)))
        return -1;
    if (opt->kv_bf16) {
        const size_t tile_rows = (size_t)WA_POS * (c.q_heads / c.kv_heads);
        if (dev_alloc(&e->hb, R * c.hidden) || dev_alloc(&e->ab, R * c.hidden) ||
            dev_alloc(&e->wa_q, R * c.hidden) || dev_alloc(&e->wa_s, tile_rows * opt->max_ctx) ||
            dev_alloc(&e->wa_p, tile_rows * opt->max_ctx) || dev_alloc(&e->wa_o, tile_rows * c.head_dim) ||
            dev_alloc(&e->wa_inv, tile_rows) || dev_alloc(&e->fl_part, FL_PART) || dev_alloc(&e->fl_stat, FL_PART / 32) ||
            dev_alloc(&e->pairs, (size_t)WK_MAX_ROWS * 2))
            return -1;
    }

    std::vector<float> inv(c.head_dim / 2);
    for (uint32_t j = 0; j < c.head_dim / 2; j++) inv[j] = 1.0f / powf(c.rope_theta, (float)(2 * j) / (float)c.head_dim);
    WK_CUDA(cudaMemcpy(e->inv_freq, inv.data(), inv.size() * sizeof(float), cudaMemcpyHostToDevice));
    WK_CUDA(cudaMalloc(&e->rope, (size_t)opt->max_ctx * (c.head_dim / 2) * sizeof(float2)));
    WK_LAUNCH(k_rope_table<<<opt->max_ctx, BLOCK, 0, e->s>>>(e->inv_freq, e->rope, c.head_dim / 2));
    WK_CUDA(cudaStreamSynchronize(e->s));
    e->host_logits.resize((size_t)WK_MAX_ROWS * c.vocab);
    return 0;
}

extern "C" int wk_create(const char* weights_path, const wk_options* opt, wk_engine** out) {
    wk_engine* e = new wk_engine;
    if (create(weights_path, opt, e) != 0) {
        wk_destroy(e);
        return -1;
    }
    *out = e;
    return 0;
}

extern "C" void wk_destroy(wk_engine* e) {
    if (!e) return;
    if (getenv("WK_PROFILE") && g_prof_steps) {
        double total = 0;
        for (double t : g_prof) total += t;
        fprintf(stderr, "profile over %ld steps, ms per step:", g_prof_steps);
        for (int k = 0; k < P_COUNT; k++) fprintf(stderr, " %s %.3f", P_NAMES[k], g_prof[k] / g_prof_steps / 1e3);
        fprintf(stderr, " total %.3f\n", total / g_prof_steps / 1e3);
    }
    if (getenv("WK_PROFILE") && g_wprof_tokens) {
        double total = 0;
        for (double t : g_wprof) total += t;
        fprintf(stderr, "prefill profile over %ld tokens, %% of %.1f us per token:", g_wprof_tokens, total / g_wprof_tokens);
        for (int k = 0; k < W_COUNT; k++) fprintf(stderr, " %s %.1f", W_NAMES[k], 100 * g_wprof[k] / total);
        fprintf(stderr, "\n");
    }
    void* bufs[] = {e->blob, e->x, e->h, e->qkv, e->attn, e->y, e->gu, e->act, e->logits, e->scores, e->K,
                    e->V, e->inv_freq, e->tok, e->slot, e->pos, e->xb, e->wide_w, e->all_logits, e->q, e->count, e->gen_out, e->att_part, e->att_stat, e->wa_q, e->wa_s, e->wa_p, e->wa_o, e->hb, e->ab, e->wa_inv, e->fl_part, e->fl_stat, e->pairs, e->rope};
    for (void* p : bufs) cudaFree(p);
    for (cudaGraphExec_t g : e->graph)
        if (g) cudaGraphExecDestroy(g);
    for (auto& g : e->tc_graph) cudaGraphExecDestroy(g.second);
    if (e->lt_desc) cublasLtMatmulDescDestroy(e->lt_desc);
    if (e->lt) cublasLtDestroy(e->lt);
    if (e->blas) cublasDestroy(e->blas);
    if (e->s) cudaStreamDestroy(e->s);
    delete e;
}

extern "C" void wk_get_config(const wk_engine* e, wk_config* out) { *out = e->c; }

extern "C" const char* wk_last_error(void) { return g_err.c_str(); }

// Checks the rows of a narrow step that will run `steps` consecutive positions and uploads them.
static int set_rows(wk_engine* e, const wk_row* rows, int R, int steps) {
    const wk_config& c = e->c;
    if (R < 1 || R > WK_MAX_ROWS) return fail("n_rows %d out of range 1..%d", R, WK_MAX_ROWS);
    int32_t tok[WK_MAX_ROWS], slot[WK_MAX_ROWS], pos[WK_MAX_ROWS];
    for (int r = 0; r < R; r++) {
        const wk_row& w = rows[r];
        if (w.token < 0 || (uint32_t)w.token >= c.vocab) return fail("row %d: token %d out of range", r, w.token);
        if (w.slot < 0 || (uint32_t)w.slot >= e->o.slots) return fail("row %d: slot %d out of range", r, w.slot);
        if (w.pos < 0 || (uint64_t)w.pos + steps > e->o.max_ctx) return fail("row %d: position %d (+%d steps) exceeds max_ctx %u", r, w.pos, steps, e->o.max_ctx);
        // Rows of one slot must be consecutive positions in row order: each row's K/V is stored before
        // attention runs, so a later row attends to the earlier ones (several tokens of one sequence
        // in one step, as speculative decoding needs).
        for (int q = r - 1; q >= 0; q--)
            if (rows[q].slot == w.slot) {
                if (q != r - 1 || rows[q].pos + 1 != w.pos) return fail("rows %d and %d share slot %d but are not adjacent, consecutive positions", q, r, w.slot);
                if (steps != 1) return fail("rows %d and %d share slot %d in a multi-step run", q, r, w.slot);
                break;
            }
        tok[r] = w.token, slot[r] = w.slot, pos[r] = w.pos;
        e->h_slot[r] = w.slot, e->h_pos[r] = w.pos;
    }
    if (!e->o.kv_bf16 && R > 8) return fail("%d rows: the exact configuration (fp32 KV cache) takes at most 8", R);
    return upload_rows(e, tok, slot, pos, R);
}

extern "C" int wk_step(wk_engine* e, const wk_row* rows, int R, float* logits, int32_t* argmax, const wk_taps* taps) {
    const wk_config& c = e->c;
    if (set_rows(e, rows, R, 1) || forward(e, R, NARROW, taps)) return -1;
    if (linear(e, NARROW, e->w<__nv_bfloat16>(T_EMBED), nullptr, e->h, e->logits, c.vocab, c.hidden, R)) return -1;
    float* out = logits ? logits : e->host_logits.data();
    if (fetch_logits(e, e->logits, out, R)) return -1;
    for (int r = 0; r < R; r++) argmax[r] = argmax_of(out + (size_t)r * c.vocab, c.vocab);
    return 0;
}

extern "C" int wk_prefill(wk_engine* e, int32_t slot, int32_t pos0, const int32_t* tokens, int n, uint32_t flags,
                          float* logits, int32_t* argmax, const wk_taps* taps) {
    const wk_config& c = e->c;
    if (n < 1 || (size_t)n > e->cap || e->o.max_chunk == 0) return fail("%d tokens, expected 1..%u (max_chunk)", n, e->o.max_chunk);
    if (slot < 0 || (uint32_t)slot >= e->o.slots) return fail("slot %d out of range", slot);
    if (pos0 < 0 || (uint64_t)pos0 + n > e->o.max_ctx) return fail("positions %d..%lld exceed max_ctx %u", pos0, (long long)pos0 + n, e->o.max_ctx);
    std::vector<int32_t> slots(n, slot), pos(n);
    for (int i = 0; i < n; i++) {
        if (tokens[i] < 0 || (uint32_t)tokens[i] >= c.vocab) return fail("token %d out of range", tokens[i]);
        pos[i] = pos0 + i;
    }
    Path path = (flags & WK_EXACT) ? WIDE_EXACT : WIDE;
    g_wprof_on = getenv("WK_PROFILE") != nullptr && path == WIDE && !taps;
    const bool fused = path == WIDE && e->o.kv_bf16 && !taps;
    int rc = upload_rows(e, tokens, slots.data(), pos.data(), n) ||
             (fused ? wide_forward(e, n, slot, pos0) : forward(e, n, path, taps, slot, pos0));
    if (g_wprof_on) g_wprof_tokens += n;
    g_wprof_on = false;
    if (rc) return -1;

    const __nv_bfloat16* E = e->w<__nv_bfloat16>(T_EMBED);
    if (flags & WK_ALL_LOGITS) {
        if (!logits) return fail("WK_ALL_LOGITS needs a logits buffer");
        if (e->all_logits_rows < (size_t)n) {
            WK_CUDA(cudaFree(e->all_logits));
            e->all_logits = nullptr, e->all_logits_rows = 0;
            WK_CUDA(cudaMalloc(&e->all_logits, (size_t)n * c.vocab * sizeof(float)));
            e->all_logits_rows = n;
        }
        if (fused ? gemm_bf16(e, E, e->hb, e->all_logits, c.vocab, c.hidden, n)
                  : linear(e, path, E, nullptr, e->h, e->all_logits, c.vocab, c.hidden, n))
            return -1;
        if (fetch_logits(e, e->all_logits, logits, n)) return -1;
        *argmax = argmax_of(logits + (size_t)(n - 1) * c.vocab, c.vocab);
        return 0;
    }
    // Only the last position predicts a token anyone reads: one GEMV, as in the narrow path.
    if (fused) {  // the fp32 residual stream of the last row, normalized, then one GEMV
        WK_LAUNCH(k_rmsnorm<<<1, BLOCK, 0, e->s>>>(e->x + (size_t)(n - 1) * c.hidden, e->w<float>(T_FINAL_NORM), e->h, c.hidden, c.rms_eps));
        WK_LAUNCH(k_f_lm_head<1><<<blocks_for((size_t)(c.vocab + WK_GEMV_ROWS - 1) / WK_GEMV_ROWS * 32), BLOCK, 0, e->s>>>(E, e->h, e->logits, c.vocab, c.hidden));
    } else if (linear(e, NARROW, E, nullptr, e->h + (size_t)(n - 1) * c.hidden, e->logits, c.vocab, c.hidden, 1))
        return -1;
    float* out = logits ? logits : e->host_logits.data();
    if (fetch_logits(e, e->logits, out, 1)) return -1;
    *argmax = argmax_of(out, c.vocab);
    return 0;
}

// ---------------------------------------------------------------------------------------------
// Fast narrow path.

// One decode step with the fused kernels, for R rows whose token / slot / pos are in the device
// arrays. Leaves the logits in e->logits.
static void wprof(wk_engine* e, int kind) {
    static std::chrono::steady_clock::time_point last;
    if (!g_wprof_on) return;
    cudaStreamSynchronize(e->s);
    auto now = std::chrono::steady_clock::now();
    if (kind >= 0) g_wprof[kind] += std::chrono::duration<double, std::micro>(now - last).count();
    last = now;
}

// The per-tile part of decode attention of the scalar path.
template <class KV> static void attn_tiles(wk_engine* e, int R, KV* K, KV* V, int l, int tiles, const Dims& d) {
    WK_LAUNCH(k_f_attn_tile<KV><<<dim3(tiles, d.kv_heads, R), ATT_TILE, 0, e->s>>>(e->q, e->slot, e->pos, K, V, e->att_part, e->att_stat, l, tiles, d));
}

template <int R, class KV> static void fast_step(wk_engine* e) {
    const wk_config& c = e->c;
    cudaStream_t s = e->s;
    const int H = c.hidden, I = c.mlp;
    const Dims d{(int)c.layers, (int)c.q_heads, (int)c.kv_heads, (int)c.head_dim, (int)e->o.max_ctx};
    typedef __nv_bfloat16 bf;
    KV *K = (KV*)e->K, *V = (KV*)e->V;

    // WK_SKIP=attn,gate_up,...: leave kernel kinds out, to time the rest. The output is garbage.
    const char* skip_env = getenv("WK_SKIP");
    const std::string skip = skip_env ? skip_env : "";
    auto on = [&](const char* kind) { return skip.find(kind) == std::string::npos; };
    prof(e, -1);
    WK_LAUNCH(k_embed<<<R, BLOCK, 0, s>>>(e->w<bf>(T_EMBED), e->tok, e->x, H));
    prof(e, P_EMBED);
    const int warps_o = (H + WK_GEMV_ROWS - 1) / WK_GEMV_ROWS, tiles = (e->o.max_ctx + ATT_TILE - 1) / ATT_TILE;
    for (int l = 0; l < (int)c.layers; l++) {
        WK_LAUNCH(k_rmsnorm<<<R, BLOCK, 0, s>>>(e->x, e->w<float>(l, L_NORM1), e->h, H, c.rms_eps));
        if (on("qkv")) WK_LAUNCH(k_f_qkv<R, KV><<<blocks_for((size_t)e->qkv_dim / 2 * 32), BLOCK, 0, s>>>(
            e->w<bf>(l, L_QKV_W), e->w<float>(l, L_QKV_B), e->h, e->q, e->slot, e->pos, e->inv_freq, K, V, l, d));
        prof(e, P_QKV);
        if (on("tile")) attn_tiles(e, R, K, V, l, tiles, d);
        if (on("merge")) WK_LAUNCH(k_f_attn_merge<<<dim3(c.kv_heads, R), ATT_BLOCK, 0, s>>>(e->pos, e->att_part, e->att_stat, e->attn, tiles, d));
        prof(e, P_ATTN);
        if (on("o_proj")) WK_LAUNCH(k_f_gemv_add<R><<<blocks_for((size_t)warps_o * 32), BLOCK, 0, s>>>(e->w<bf>(l, L_O_W), e->attn, e->x, H, H));
        prof(e, P_O);
        WK_LAUNCH(k_rmsnorm<<<R, BLOCK, 0, s>>>(e->x, e->w<float>(l, L_NORM2), e->h, H, c.rms_eps));
        if (on("gate_up")) WK_LAUNCH(k_f_gate_up<R><<<blocks_for((size_t)(I + WK_GATE_PAIRS - 1) / WK_GATE_PAIRS * 32), BLOCK, 0, s>>>(e->w<bf>(l, L_GATE_UP_W), e->h, e->act, I, H));
        prof(e, P_GATE_UP);
        if (on("down")) WK_LAUNCH(k_f_gemv_add<R><<<blocks_for((size_t)warps_o * 32), BLOCK, 0, s>>>(e->w<bf>(l, L_DOWN_W), e->act, e->x, H, I));
        prof(e, P_DOWN);
    }
    WK_LAUNCH(k_rmsnorm<<<R, BLOCK, 0, s>>>(e->x, e->w<float>(T_FINAL_NORM), e->h, H, c.rms_eps));
    if (on("lm_head")) WK_LAUNCH(k_f_lm_head<R><<<blocks_for((size_t)(c.vocab + WK_GEMV_ROWS - 1) / WK_GEMV_ROWS * 32), BLOCK, 0, s>>>(e->w<bf>(T_EMBED), e->h, e->logits, c.vocab, H));
    prof(e, P_LM_HEAD);
}

// ---------------------------------------------------------------------------------------------
// Decode step on the tensor cores (bf16 KV cache): the wide path's kernels with k_t_gemv for the
// products, rows grouped by slot, attention cut into fixed segments of keys.

static void tc_gemv(wk_engine* e, int CT, const __nv_bfloat16* W, const __nv_bfloat16* x, float* y, int N, int K) {
    const int blocks = (N + 127) / 128;
    switch (CT) {
    case 1: WK_LAUNCH(k_t_gemv<1><<<blocks, BLOCK, 0, e->s>>>(W, x, y, N, K)); break;
    case 2: WK_LAUNCH(k_t_gemv<2><<<blocks, BLOCK, 0, e->s>>>(W, x, y, N, K)); break;
    case 3: WK_LAUNCH(k_t_gemv<3><<<blocks, BLOCK, 0, e->s>>>(W, x, y, N, K)); break;
    default: WK_LAUNCH(k_t_gemv<4><<<blocks, BLOCK, 0, e->s>>>(W, x, y, N, K)); break;
    }
}

// The groups of a step: maximal runs of adjacent rows in one slot (set_rows guarantees that the rows
// of a slot are adjacent). Returns their sizes.
static std::vector<int> tc_groups(const wk_engine* e, int R) {
    std::vector<int> sizes;
    for (int r = 0; r < R; r++) {
        if (r > 0 && e->h_slot[r] == e->h_slot[r - 1]) sizes.back()++;
        else sizes.push_back(1);
    }
    return sizes;
}

// Launches one step for R rows in the given groups; everything that changes from step to step
// (tokens, positions, slots) is read from device memory, so the launches can be captured as a graph.
// Leaves the logits of all rows in e->logits.
static void tc_step(wk_engine* e, int R, const std::vector<int>& groups) {
    const wk_config& c = e->c;
    cudaStream_t s = e->s;
    const int H = c.hidden, I = c.mlp, hd = c.head_dim, CT = (R + 7) / 8;
    const Dims d{(int)c.layers, (int)c.q_heads, (int)c.kv_heads, (int)c.head_dim, (int)e->o.max_ctx};
    typedef __nv_bfloat16 bf;
    const int tiles = (e->o.max_ctx + ATT_TILE - 1) / ATT_TILE;
    int n_pairs = 0;  // two adjacent rows of a group share a warp of the attention
    for (int n : groups) n_pairs += (n + 1) / 2;

    WK_LAUNCH(k_w_embed_norm<<<R, BLOCK, 0, s>>>(e->w<bf>(T_EMBED), e->tok, e->x, e->w<float>(0, L_NORM1), e->hb, H, c.rms_eps));
    for (int l = 0; l < (int)c.layers; l++) {
        tc_gemv(e, CT, e->w<bf>(l, L_QKV_W), e->hb, e->qkv, e->qkv_dim, H);
        WK_LAUNCH(k_t_qkv_post<<<R, BLOCK, 0, s>>>(e->qkv, e->w<float>(l, L_QKV_B), e->pos, e->inv_freq, (bf*)e->K, (bf*)e->V, e->q, e->slot, l, d));
        const dim3 grid(tiles, d.kv_heads, n_pairs);
        if (hd == 128)
            WK_LAUNCH(k_t_attn_tile<128><<<grid, 32, 0, s>>>(e->q, e->slot, e->pos, e->pairs, (bf*)e->K, (bf*)e->V, e->att_part, e->att_stat, l, tiles, d));
        else
            WK_LAUNCH(k_t_attn_tile<64><<<grid, 32, 0, s>>>(e->q, e->slot, e->pos, e->pairs, (bf*)e->K, (bf*)e->V, e->att_part, e->att_stat, l, tiles, d));
        WK_LAUNCH(k_t_attn_merge<<<dim3(d.kv_heads, R), ATT_BLOCK, 0, s>>>(e->pos, e->att_part, e->att_stat, e->ab, tiles, d));
        tc_gemv(e, CT, e->w<bf>(l, L_O_W), e->ab, e->y, H, H);
        WK_LAUNCH(k_w_add_norm<<<R, BLOCK, 0, s>>>(e->x, e->y, e->w<float>(l, L_NORM2), e->hb, H, c.rms_eps));
        tc_gemv(e, CT, e->w<bf>(l, L_GATE_UP_W), e->hb, e->gu, 2 * I, H);
        WK_LAUNCH(k_w_silu<<<dim3(blocks_for(I), R), BLOCK, 0, s>>>(e->gu, e->xb, I, I, 0));
        tc_gemv(e, CT, e->w<bf>(l, L_DOWN_W), e->xb, e->y, H, I);
        const float* next = l + 1 < (int)c.layers ? e->w<float>(l + 1, L_NORM1) : e->w<float>(T_FINAL_NORM);
        WK_LAUNCH(k_w_add_norm<<<R, BLOCK, 0, s>>>(e->x, e->y, next, e->hb, H, c.rms_eps));
    }
    tc_gemv(e, CT, e->w<bf>(T_EMBED), e->hb, e->logits, c.vocab, H);
}

static bool tc_ok(const wk_engine* e) {
    const wk_config& c = e->c;
    return e->o.kv_bf16 && (c.head_dim == 64 || c.head_dim == 128) && c.hidden % 64 == 0 && c.mlp % 64 == 0 &&
           !getenv("WK_NO_TC_STEP");
}

// Uploads the pairing of rows for the attention: two adjacent rows of a group per warp.
static int tc_upload_dyn(wk_engine* e, int R, const std::vector<int>& groups) {
    int32_t pairs[2 * WK_MAX_ROWS];
    int n = 0, r0 = 0;
    for (int size : groups) {
        for (int i = 0; i < size; i += 2) pairs[2 * n] = r0 + i, pairs[2 * n + 1] = i + 1 < size ? r0 + i + 1 : -1, n++;
        r0 += size;
    }
    WK_CUDA(cudaMemcpyAsync(e->pairs, pairs, 2 * n * sizeof(int32_t), cudaMemcpyHostToDevice, e->s));
    return 0;
}

static void fast_step(wk_engine* e, int R) {
    if (e->o.kv_bf16) {
        switch (R) {
        case 1: fast_step<1, __nv_bfloat16>(e); break;
        case 2: fast_step<2, __nv_bfloat16>(e); break;
        case 3: fast_step<3, __nv_bfloat16>(e); break;
        case 4: fast_step<4, __nv_bfloat16>(e); break;
        case 5: fast_step<5, __nv_bfloat16>(e); break;
        case 6: fast_step<6, __nv_bfloat16>(e); break;
        case 7: fast_step<7, __nv_bfloat16>(e); break;
        case 8: fast_step<8, __nv_bfloat16>(e); break;
        }
    } else {
        switch (R) {
        case 1: fast_step<1, float>(e); break;
        case 2: fast_step<2, float>(e); break;
        case 3: fast_step<3, float>(e); break;
        case 4: fast_step<4, float>(e); break;
        case 5: fast_step<5, float>(e); break;
        case 6: fast_step<6, float>(e); break;
        case 7: fast_step<7, float>(e); break;
        case 8: fast_step<8, float>(e); break;
        }
    }
}

extern "C" int wk_step_fast(wk_engine* e, const wk_row* rows, int R, float* logits, int32_t* argmax) {
    if (set_rows(e, rows, R, 1)) return -1;
    g_prof_on = getenv("WK_PROFILE") != nullptr;
    if (tc_ok(e)) {
        const std::vector<int> groups = tc_groups(e, R);
        if (tc_upload_dyn(e, R, groups)) return -1;
        tc_step(e, R, groups);
    } else {
        fast_step(e, R);
    }
    g_prof_on = false;
    g_prof_steps++;
    float* out = logits ? logits : e->host_logits.data();
    if (fetch_logits(e, e->logits, out, R)) return -1;
    for (int r = 0; r < R; r++) argmax[r] = argmax_of(out + (size_t)r * e->c.vocab, e->c.vocab);
    return 0;
}

// The graph of one step for the rows last given to set_rows: kernels, argmax, feedback of the token
// and position.
static int step_graph(wk_engine* e, int R, cudaGraphExec_t* out) {
    const bool tc = tc_ok(e);
    std::vector<int> groups;
    cudaGraphExec_t* slot_for;
    if (tc) {
        groups = tc_groups(e, R);
        std::string key;
        for (int n : groups) key += std::to_string(n) + ",";
        slot_for = &e->tc_graph[key];
    } else {
        slot_for = &e->graph[R];
    }
    if (!*slot_for) {
        cudaGraph_t g;
        WK_CUDA(cudaStreamBeginCapture(e->s, cudaStreamCaptureModeGlobal));
        if (tc) tc_step(e, R, groups);
        else fast_step(e, R);
        WK_LAUNCH(k_argmax_next<<<R, ATT_BLOCK, 0, e->s>>>(e->logits, e->c.vocab, e->tok, e->pos, e->count, e->gen_out, GEN_CAP));
        WK_CUDA(cudaStreamEndCapture(e->s, &g));
        WK_CUDA(cudaGraphInstantiate(slot_for, g, 0));
        WK_CUDA(cudaGraphDestroy(g));
    }
    *out = *slot_for;
    return 0;
}

extern "C" int wk_generate(wk_engine* e, const wk_row* rows, int R, int n_tokens, int32_t* out) {
    if (n_tokens < 1 || n_tokens > GEN_CAP) return fail("n_tokens %d out of range 1..%d", n_tokens, GEN_CAP);
    cudaGraphExec_t graph;
    if (set_rows(e, rows, R, n_tokens) || step_graph(e, R, &graph)) return -1;
    const bool tc = tc_ok(e);
    const std::vector<int> groups = tc ? tc_groups(e, R) : std::vector<int>();
    WK_CUDA(cudaMemsetAsync(e->count, 0, WK_MAX_ROWS * sizeof(int32_t), e->s));
    if (tc && tc_upload_dyn(e, R, groups)) return -1;
    for (int i = 0; i < n_tokens; i++) WK_CUDA(cudaGraphLaunch(graph, e->s));
    for (int r = 0; r < R; r++)
        WK_CUDA(cudaMemcpyAsync(out + (size_t)r * n_tokens, e->gen_out + (size_t)r * GEN_CAP, n_tokens * sizeof(int32_t), cudaMemcpyDeviceToHost, e->s));
    WK_CUDA(cudaStreamSynchronize(e->s));
    WK_CUDA(cudaGetLastError());
    return 0;
}

extern "C" int wk_copy_kv(wk_engine* e, int32_t src_slot, int32_t dst_slot, int32_t n) {
    const wk_config& c = e->c;
    if (src_slot < 0 || (uint32_t)src_slot >= e->o.slots || dst_slot < 0 || (uint32_t)dst_slot >= e->o.slots || src_slot == dst_slot)
        return fail("copy between slots %d and %d", src_slot, dst_slot);
    if (n < 0 || (uint32_t)n > e->o.max_ctx) return fail("%d positions exceed max_ctx %u", n, e->o.max_ctx);
    const Dims d{(int)c.layers, (int)c.q_heads, (int)c.kv_heads, (int)c.head_dim, (int)e->o.max_ctx};
    const size_t elem = e->o.kv_bf16 ? sizeof(__nv_bfloat16) : sizeof(float), bytes = (size_t)n * e->kv_dim * elem;
    for (int l = 0; l < (int)c.layers && n; l++)
        for (void* base : {e->K, e->V}) {
            char* b = static_cast<char*>(base);
            WK_CUDA(cudaMemcpyAsync(b + d.kv_at(dst_slot, l, 0) * elem, b + d.kv_at(src_slot, l, 0) * elem, bytes, cudaMemcpyDeviceToDevice, e->s));
        }
    return 0;
}
