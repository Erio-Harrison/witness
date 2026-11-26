// Fused kernels of the narrow path (decode). Included by engine.cu after its basic kernels.
//
// Per layer, plus two k_rmsnorm:
//   k_f_qkv       q/k/v projection + bias + RoPE + KV store
//   k_f_attn_tile, k_f_attn_merge   attention over the KV cache
//   k_f_gemv_add  o projection + residual add          (also: down projection + residual add)
//   k_f_gate_up   gate/up projection + SiLU * up
// The products that follow an RMSNorm read the normalized rows h[R][hidden], which k_rmsnorm writes
// just before. (Normalizing inside every block, into shared memory, was no faster for one row and
// 1.3 ms per step slower for four: the shared memory lowers how many blocks an SM runs at once.)
// R = rows of the step (1..WK_MAX_ROWS), KV = element type of the KV cache.

// One warp's dot products of M bf16 weight rows (K elements each) with R fp32 rows x[R][K]. Per step
// a lane reads 16 bytes of each weight row (streaming) and 32 bytes of each x row, so x is read once
// for all M weight rows. On return acc[m][r] holds the whole sum in every lane. x must be 16-byte
// aligned and K a multiple of 8.
template <int R, int M>
__device__ __forceinline__ void warp_dot(const __nv_bfloat16* const* rows, const float* __restrict__ x, int K, float (*acc)[R]) {
#pragma unroll
    for (int m = 0; m < M; m++)
#pragma unroll
        for (int r = 0; r < R; r++) acc[m][r] = 0;
    for (int i = threadIdx.x & 31; i < K / 8; i += 32) {
        float4 xa[R], xb[R];
#pragma unroll
        for (int r = 0; r < R; r++) {
            const float4* xr = reinterpret_cast<const float4*>(x + (size_t)r * K) + 2 * i;
            xa[r] = xr[0];
            xb[r] = xr[1];
        }
#pragma unroll
        for (int m = 0; m < M; m++) {
            uint4 wv = __ldcs(reinterpret_cast<const uint4*>(rows[m]) + i);
            const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(&wv);
            float2 w0 = __bfloat1622float2(w2[0]), w1 = __bfloat1622float2(w2[1]);
            float2 w2f = __bfloat1622float2(w2[2]), w3 = __bfloat1622float2(w2[3]);
#pragma unroll
            for (int r = 0; r < R; r++)
                acc[m][r] += w0.x * xa[r].x + w0.y * xa[r].y + w1.x * xa[r].z + w1.y * xa[r].w +
                             w2f.x * xb[r].x + w2f.y * xb[r].y + w3.x * xb[r].z + w3.y * xb[r].w;
        }
    }
#pragma unroll
    for (int m = 0; m < M; m++)
#pragma unroll
        for (int r = 0; r < R; r++) {
            float v = acc[m][r];
            for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
            acc[m][r] = v;
        }
}

// One warp per pair of outputs: the two halves of a RoPE pair of a q or k head, or two v elements.
// h is the normalized input; q goes to q[R][hidden]; k and v go to the cache at (slot, layer, pos).
template <int R, class KV>
__global__ void k_f_qkv(const __nv_bfloat16* __restrict__ W, const float* __restrict__ bias, const float* __restrict__ h,
                        float* __restrict__ q, const int32_t* __restrict__ slot, const int32_t* __restrict__ pos,
                        const float* __restrict__ inv_freq, KV* __restrict__ Kc, KV* __restrict__ Vc, int layer, Dims d) {
    const int H = d.hidden(), hd = d.head_dim, half = hd / 2, rot = (d.q_heads + d.kv_heads) * half;
    int warp = (blockIdx.x * BLOCK + threadIdx.x) >> 5;
    if (warp >= d.qkv_dim() / 2) return;
    int n1, n2, j = 0;
    if (warp < rot) {
        j = warp % half;
        n1 = (warp / half) * hd + j;
        n2 = n1 + half;
    } else {
        n1 = H + d.kv_dim() + 2 * (warp - rot);
        n2 = n1 + 1;
    }
    const __nv_bfloat16* rows[2] = {W + (size_t)n1 * H, W + (size_t)n2 * H};
    float acc[2][R];
    warp_dot<R, 2>(rows, h, H, acc);
    if (threadIdx.x & 31) return;
#pragma unroll
    for (int r = 0; r < R; r++) {
        float va = acc[0][r] + bias[n1], vb = acc[1][r] + bias[n2];
        size_t at = d.kv_at(slot[r], layer, pos[r]);
        if (warp >= rot) {
            kv_put(Vc + at + (n1 - H - d.kv_dim()), va);
            kv_put(Vc + at + (n2 - H - d.kv_dim()), vb);
            continue;
        }
        float s, c;
        sincosf((float)pos[r] * inv_freq[j], &s, &c);
        float ra = va * c - vb * s, rb = vb * c + va * s;
        if (n1 < H) {
            q[(size_t)r * H + n1] = ra;
            q[(size_t)r * H + n2] = rb;
        } else {
            kv_put(Kc + at + (n1 - H), ra);
            kv_put(Kc + at + (n2 - H), rb);
        }
    }
}

// Attention limits: q heads per kv head <= ATT_MAX_GROUP, head_dim a multiple of 32 and <= 128.
static const int ATT_BLOCK = 1024, ATT_WARPS = ATT_BLOCK / 32, ATT_MAX_GROUP = 8;
static const int ATT_TILE = 256;  // positions per block of k_f_attn_tile; also its block size

// Eight consecutive KV elements as floats; p must be 16-byte aligned.
__device__ __forceinline__ void kv_get8(const __nv_bfloat16* p, float* out) {
    uint4 v = *reinterpret_cast<const uint4*>(p);
    const __nv_bfloat162* v2 = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
    for (int j = 0; j < 4; j++) {
        float2 f = __bfloat1622float2(v2[j]);
        out[2 * j] = f.x;
        out[2 * j + 1] = f.y;
    }
}
__device__ __forceinline__ void kv_get8(const float* p, float* out) {
    float4 a = reinterpret_cast<const float4*>(p)[0], b = reinterpret_cast<const float4*>(p)[1];
    out[0] = a.x, out[1] = a.y, out[2] = a.z, out[3] = a.w, out[4] = b.x, out[5] = b.y, out[6] = b.z, out[7] = b.w;
}

// Four consecutive KV elements as floats; p must be 8-byte (bf16) or 16-byte (fp32) aligned.
__device__ __forceinline__ float4 kv_get4(const __nv_bfloat16* p) {
    uint2 v = *reinterpret_cast<const uint2*>(p);
    const __nv_bfloat162* v2 = reinterpret_cast<const __nv_bfloat162*>(&v);
    float2 a = __bfloat1622float2(v2[0]), b = __bfloat1622float2(v2[1]);
    return make_float4(a.x, a.y, b.x, b.y);
}
__device__ __forceinline__ float4 kv_get4(const float* p) { return *reinterpret_cast<const float4*>(p); }

// Attention is split over the positions so that a long context spreads over the SMs: each block of
// k_f_attn_tile takes ATT_TILE positions of one (kv head, row) and produces, for the q heads that
// share that kv head, a softmax-weighted V sum normalized to the tile's own largest score, plus
// that largest score and the tile's sum of weights. k_f_attn_merge rescales and adds the tiles in
// tile order, so the result does not depend on scheduling.
//
// A thread waits for each load from the KV cache before it issues the next instruction, so the
// kernels are arranged to have many loads in flight: loads that a thread needs soon are issued
// together, and positions are spread over threads or warps rather than walked by one.
//
// Grid: (tiles, kv_heads, rows), ATT_TILE threads. Blocks past the row's position do nothing.
//   part  [rows][kv_heads][tiles][group * head_dim]     stat  [rows][kv_heads][tiles][2][ATT_MAX_GROUP]
template <class KV>
__global__ void k_f_attn_tile(const float* __restrict__ q, const int32_t* __restrict__ slot, const int32_t* __restrict__ pos,
                              const KV* __restrict__ Kc, const KV* __restrict__ Vc, float* __restrict__ part,
                              float* __restrict__ stat, int layer, int tiles, Dims d) {
    __shared__ float qs[ATT_BLOCK];                          // the group's q heads
    __shared__ float sc[ATT_MAX_GROUP][ATT_TILE];            // scores, then weights
    __shared__ float pw[ATT_TILE / 32][ATT_BLOCK];           // each warp's partial output
    __shared__ float red[ATT_TILE / 32][ATT_MAX_GROUP];
    const int tile = blockIdx.x, kvh = blockIdx.y, r = blockIdx.z, hd = d.head_dim, kvd = d.kv_dim(), H = d.hidden();
    const int G = d.q_heads / d.kv_heads, h0 = kvh * G, T = pos[r] + 1, t0 = tile * ATT_TILE;
    if (t0 >= T) return;
    const int n = T - t0 < ATT_TILE ? T - t0 : ATT_TILE;     // positions in this tile
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, t = threadIdx.x;
    const size_t base = d.kv_at(slot[r], layer, t0) + kvh * hd;
    const float scale = rsqrtf((float)hd);

    for (int i = threadIdx.x; i < G * hd; i += ATT_TILE) qs[i] = q[(size_t)r * H + h0 * hd + i];
    __syncthreads();

    // Scores: one thread per position. The first 8 elements of each 32-element stretch of K[t] are
    // loaded up front, which brings all of K[t] in at once.
    float m[ATT_MAX_GROUP], sum[ATT_MAX_GROUP];
    for (int i = 0; i < G; i++) m[i] = -INFINITY;
    if (t < n) {
        const KV* k = Kc + base + (size_t)t * kvd;
        float lead[4][8];
#pragma unroll
        for (int seg = 0; seg < 4; seg++)
            if (seg * 32 < hd) kv_get8(k + seg * 32, lead[seg]);
        float dot[ATT_MAX_GROUP] = {};
#pragma unroll
        for (int seg = 0; seg < 4; seg++) {
            if (seg * 32 >= hd) break;
#pragma unroll
            for (int part8 = 0; part8 < 4; part8++) {
                const int j = seg * 32 + part8 * 8;
                float kf[8];
                if (part8 == 0) {
#pragma unroll
                    for (int u = 0; u < 8; u++) kf[u] = lead[seg][u];
                } else {
                    kv_get8(k + j, kf);
                }
                for (int i = 0; i < G; i++) {
                    const float* qi = qs + i * hd + j;
#pragma unroll
                    for (int u = 0; u < 8; u++) dot[i] += qi[u] * kf[u];
                }
            }
        }
        for (int i = 0; i < G; i++) m[i] = sc[i][t] = dot[i] * scale;
    }
    for (int i = 0; i < G; i++) {
        float x = m[i];
        for (int o = 16; o; o >>= 1) x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, o));
        if (lane == 0) red[warp][i] = x;
    }
    __syncthreads();
    for (int i = 0; i < G; i++) {
        float x = red[0][i];
        for (int w = 1; w < ATT_TILE / 32; w++) x = fmaxf(x, red[w][i]);
        m[i] = x;
    }
    __syncthreads();
    for (int i = 0; i < G; i++) {
        float e = t < n ? expf(sc[i][t] - m[i]) : 0.0f;
        if (t < n) sc[i][t] = e;
        for (int o = 16; o; o >>= 1) e += __shfl_xor_sync(0xffffffff, e, o);
        if (lane == 0) red[warp][i] = e;
    }
    __syncthreads();
    for (int i = 0; i < G; i++) {
        float x = 0;
        for (int w = 0; w < ATT_TILE / 32; w++) x += red[w][i];
        sum[i] = x;
    }

    // Output: each warp walks every 8th position of the tile, two positions per step. A lane takes
    // four consecutive elements of V[p] in one load, so lanes 0 .. head_dim/4 cover the head.
    float4 acc[ATT_MAX_GROUP];
    for (int i = 0; i < G; i++) acc[i] = make_float4(0, 0, 0, 0);
    if (lane * 4 < hd) {
        const int step = ATT_TILE / 32;
        for (int p = warp; p < n; p += 2 * step) {
            const bool two = p + step < n;
            float4 va = kv_get4(Vc + base + (size_t)p * kvd + lane * 4);
            float4 vb = two ? kv_get4(Vc + base + (size_t)(p + step) * kvd + lane * 4) : make_float4(0, 0, 0, 0);
            for (int i = 0; i < G; i++) {
                float wa = sc[i][p], wb = two ? sc[i][p + step] : 0.0f;
                acc[i].x += wa * va.x, acc[i].y += wa * va.y, acc[i].z += wa * va.z, acc[i].w += wa * va.w;
                acc[i].x += wb * vb.x, acc[i].y += wb * vb.y, acc[i].z += wb * vb.z, acc[i].w += wb * vb.w;
            }
        }
        for (int i = 0; i < G; i++) {
            float* o = &pw[warp][i * hd + lane * 4];
            o[0] = acc[i].x, o[1] = acc[i].y, o[2] = acc[i].z, o[3] = acc[i].w;
        }
    }
    __syncthreads();
    float* pt = part + (((size_t)r * d.kv_heads + kvh) * tiles + tile) * G * hd;
    for (int o = threadIdx.x; o < G * hd; o += ATT_TILE) {
        float total = 0;
        for (int w = 0; w < ATT_TILE / 32; w++) total += pw[w][o];
        pt[o] = total;
    }
    float* st = stat + (((size_t)r * d.kv_heads + kvh) * tiles + tile) * 2 * ATT_MAX_GROUP;
    if (threadIdx.x < G) st[threadIdx.x] = m[threadIdx.x], st[ATT_MAX_GROUP + threadIdx.x] = sum[threadIdx.x];
}

// out[r][head] = sum over tiles of exp(m_tile - m) * part_tile, divided by the same sum of the
// tiles' weight sums. Grid: (kv_heads, rows), ATT_BLOCK threads: first one thread per (tile, head)
// fetches the tile's statistics, then one thread per output element adds the tiles in order.
__global__ void k_f_attn_merge(const int32_t* __restrict__ pos, const float* __restrict__ part, const float* __restrict__ stat,
                               float* __restrict__ out, int tiles, Dims d) {
    __shared__ float sm[ATT_BLOCK], ss[ATT_BLOCK];   // [tile][head in group]: largest score, sum of weights
    const int kvh = blockIdx.x, r = blockIdx.y, hd = d.head_dim, G = d.q_heads / d.kv_heads;
    const int n_tiles = (pos[r] + ATT_TILE) / ATT_TILE;
    const float* pt = part + ((size_t)r * d.kv_heads + kvh) * tiles * G * hd;
    const float* st = stat + ((size_t)r * d.kv_heads + kvh) * tiles * 2 * ATT_MAX_GROUP;
    for (int k = threadIdx.x; k < n_tiles * ATT_MAX_GROUP; k += ATT_BLOCK) {
        const float* s = st + (size_t)(k / ATT_MAX_GROUP) * 2 * ATT_MAX_GROUP + k % ATT_MAX_GROUP;
        sm[k] = s[0];
        ss[k] = s[ATT_MAX_GROUP];
    }
    __syncthreads();
    if (threadIdx.x >= G * hd) return;
    const int o = threadIdx.x, i = o / hd;
    float m = -INFINITY;
    for (int w = 0; w < n_tiles; w++) m = fmaxf(m, sm[w * ATT_MAX_GROUP + i]);
    float total = 0, denom = 0;
    int w = 0;
    for (; w + 4 <= n_tiles; w += 4) {  // four tiles' loads in flight at a time
        float p0 = pt[(size_t)w * G * hd + o], p1 = pt[(size_t)(w + 1) * G * hd + o];
        float p2 = pt[(size_t)(w + 2) * G * hd + o], p3 = pt[(size_t)(w + 3) * G * hd + o];
        float e0 = expf(sm[w * ATT_MAX_GROUP + i] - m), e1 = expf(sm[(w + 1) * ATT_MAX_GROUP + i] - m);
        float e2 = expf(sm[(w + 2) * ATT_MAX_GROUP + i] - m), e3 = expf(sm[(w + 3) * ATT_MAX_GROUP + i] - m);
        total += e0 * p0, denom += e0 * ss[w * ATT_MAX_GROUP + i];
        total += e1 * p1, denom += e1 * ss[(w + 1) * ATT_MAX_GROUP + i];
        total += e2 * p2, denom += e2 * ss[(w + 2) * ATT_MAX_GROUP + i];
        total += e3 * p3, denom += e3 * ss[(w + 3) * ATT_MAX_GROUP + i];
    }
    for (; w < n_tiles; w++) {
        float e = expf(sm[w * ATT_MAX_GROUP + i] - m);
        total += e * pt[(size_t)w * G * hd + o];
        denom += e * ss[w * ATT_MAX_GROUP + i];
    }
    out[(size_t)r * d.hidden() + kvh * G * hd + o] = total / denom;
}

// Weight rows a warp of the plain products takes at once (x is read once for all of them), and
// (gate, up) pairs a warp of k_f_gate_up takes.
#ifndef WK_GEMV_ROWS
#define WK_GEMV_ROWS 4
#endif
#ifndef WK_GATE_PAIRS
#define WK_GATE_PAIRS 2
#endif

// M consecutive weight rows starting at n0, clamped to the last row.
template <int M>
__device__ __forceinline__ void rows_from(const __nv_bfloat16* W, int n0, int N, int K, const __nv_bfloat16** rows) {
#pragma unroll
    for (int m = 0; m < M; m++) rows[m] = W + (size_t)(n0 + m < N ? n0 + m : N - 1) * K;
}

// y[r][n] += sum_k W[n][k] x[r][k]. One warp per WK_GEMV_ROWS outputs.
template <int R>
__global__ void k_f_gemv_add(const __nv_bfloat16* __restrict__ W, const float* __restrict__ x, float* __restrict__ y, int N, int K) {
    const int M = WK_GEMV_ROWS;
    int n0 = ((blockIdx.x * BLOCK + threadIdx.x) >> 5) * M;
    if (n0 >= N) return;
    const __nv_bfloat16* rows[M];
    rows_from<M>(W, n0, N, K, rows);
    float acc[M][R];
    warp_dot<R, M>(rows, x, K, acc);
    if (threadIdx.x & 31) return;
#pragma unroll
    for (int m = 0; m < M; m++)
#pragma unroll
        for (int r = 0; r < R; r++)
            if (n0 + m < N) y[(size_t)r * N + n0 + m] += acc[m][r];
}

// act[r][i] = silu(gate_i) * up_i of the normalized input h; gate and up rows are interleaved in W.
// One warp per WK_GATE_PAIRS outputs.
template <int R>
__global__ void k_f_gate_up(const __nv_bfloat16* __restrict__ W, const float* __restrict__ h, float* __restrict__ act, int I, int H) {
    const int P = WK_GATE_PAIRS;
    int i0 = ((blockIdx.x * BLOCK + threadIdx.x) >> 5) * P;
    if (i0 >= I) return;
    const __nv_bfloat16* rows[2 * P];
    rows_from<2 * P>(W, 2 * i0, 2 * I, H, rows);
    float acc[2 * P][R];
    warp_dot<R, 2 * P>(rows, h, H, acc);
    if (threadIdx.x & 31) return;
#pragma unroll
    for (int m = 0; m < P; m++)
#pragma unroll
        for (int r = 0; r < R; r++) {
            float gv = acc[2 * m][r];
            if (i0 + m < I) act[(size_t)r * I + i0 + m] = gv / (1.0f + expf(-gv)) * acc[2 * m + 1][r];
        }
}

// logits[r][n] = embed[n] . h[r], h being the final norm's output. One warp per WK_GEMV_ROWS
// vocabulary entries.
template <int R>
__global__ void k_f_lm_head(const __nv_bfloat16* __restrict__ E, const float* __restrict__ h, float* __restrict__ logits, int V, int H) {
    const int M = WK_GEMV_ROWS;
    int n0 = ((blockIdx.x * BLOCK + threadIdx.x) >> 5) * M;
    if (n0 >= V) return;
    const __nv_bfloat16* rows[M];
    rows_from<M>(E, n0, V, H, rows);
    float acc[M][R];
    warp_dot<R, M>(rows, h, H, acc);
    if (threadIdx.x & 31) return;
#pragma unroll
    for (int m = 0; m < M; m++)
#pragma unroll
        for (int r = 0; r < R; r++)
            if (n0 + m < V) logits[(size_t)r * V + n0 + m] = acc[m][r];
}

// Greedy feedback on the device: the argmax of each row (lowest index on ties) becomes the row's
// next token, its position advances, and the token is appended to out[row][count[row]].
// Grid: rows, ATT_BLOCK threads.
__global__ void k_argmax_next(const float* __restrict__ logits, int V, int32_t* __restrict__ tok, int32_t* __restrict__ pos,
                              int32_t* __restrict__ count, int32_t* __restrict__ out, int out_cap) {
    __shared__ float bv[ATT_WARPS];
    __shared__ int bi[ATT_WARPS];
    const int r = blockIdx.x;
    const float* lg = logits + (size_t)r * V;
    float best = -INFINITY;
    int idx = 0x7fffffff;
    for (int i = threadIdx.x; i < V; i += ATT_BLOCK) {
        if (lg[i] > best) best = lg[i], idx = i;
    }
    for (int o = 16; o; o >>= 1) {
        float ov = __shfl_xor_sync(0xffffffff, best, o);
        int oi = __shfl_xor_sync(0xffffffff, idx, o);
        if (ov > best || (ov == best && oi < idx)) best = ov, idx = oi;
    }
    if ((threadIdx.x & 31) == 0) bv[threadIdx.x >> 5] = best, bi[threadIdx.x >> 5] = idx;
    __syncthreads();
    if (threadIdx.x) return;
    for (int w = 1; w < ATT_WARPS; w++)
        if (bv[w] > best || (bv[w] == best && bi[w] < idx)) best = bv[w], idx = bi[w];
    if (idx == 0x7fffffff) idx = 0;
    tok[r] = idx;
    pos[r] += 1;
    int c = count[r];
    if (c < out_cap) out[(size_t)r * out_cap + c] = idx;
    count[r] = c + 1;
}
