// Fused attention of the wide path on the tensor cores (bf16 KV cache). Included by engine.cu.
//
// One launch per kv head does what took a score GEMM, a softmax kernel, an output GEMM and a scatter
// per tile of positions: scores never leave the registers.
//
// Rows are (position, q head of the kv head's group) pairs, 16 per warp (the M of mma.m16n8k16). A
// block of FA_WARPS warps walks the keys 16 at a time. For each step it stages that piece of K, and
// of V transposed, in shared memory for all its warps; then every warp
//   scores   two 16 x 8 products  Q (16 rows x head_dim) . K^T (head_dim x 8 keys), fp32
//   weights  mask, running max, exp: the online softmax of FlashAttention
//   output   head_dim / 8 products  P (16 rows x 16 keys) . V (16 keys x 8 elements), fp32
// The two score tiles come out of the mma in exactly the register layout the output product wants
// for its first operand, so the weights go from one to the other without being moved.
//
// Operands are bf16 (q, K, V as stored; the weights rounded), accumulation is fp32, as in the cuBLAS
// path this replaces.

#ifndef WK_FA_WARPS
#define WK_FA_WARPS 8
#endif
static const int FA_WARPS = WK_FA_WARPS;  // warps (16 rows each) that share one staged piece of K and V

// d (16 x 8, fp32) += a (16 x 16, bf16) * b (16 x 8, bf16), fragments in the mma register layout:
// lane = 4 * g + t;  a[0] = rows g, a[1] = rows g + 8 at k = 2t, 2t+1;  a[2], a[3] the same at k + 8;
// b[0] = column g at k = 2t, 2t+1;  b[1] at k + 8;  d[0], d[1] = row g, columns 2t, 2t+1;  d[2], d[3] = row g + 8.
__device__ __forceinline__ void mma_bf16(float* d, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t pack_bf16(float x, float y) {
    __nv_bfloat162 v = __floats2bfloat162_rn(x, y);
    return *reinterpret_cast<uint32_t*>(&v);
}

// Largest of v over the 4 lanes that share a row (lanes 4g .. 4g+3).
__device__ __forceinline__ float group_max(float v) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, 1));
    return fmaxf(v, __shfl_xor_sync(0xffffffff, v, 2));
}

#define VT_AT(e) ((e) * 8 + ((e) >> 2) * 4)
// Stages keys k0 .. k0+15 (clamped to the last key in the cache; the caller masks the excess) for a
// block of k_w_flash_attn, in the layout described there. K: a thread copies 8 elements of one key.
// V: a thread takes 8 elements of two adjacent keys and stores, per element, the word that holds both.
// Not inlined: inlined, its temporaries cost the kernel 40 registers, and a block more per SM is lost.
template <int HD>
__device__ __noinline__ void fa_stage(__nv_bfloat16* ks, uint32_t* vt, const __nv_bfloat16* K, const __nv_bfloat16* V, int k0, int last_key, int kvd) {
    if (threadIdx.x < 2 * HD) {
        const int key = threadIdx.x / (HD / 8), j = (threadIdx.x % (HD / 8)) * 8;
        *reinterpret_cast<uint4*>(&ks[key * (HD + 8) + j]) = *reinterpret_cast<const uint4*>(K + (size_t)min(k0 + key, last_key) * kvd + j);
    }
    if (threadIdx.x < HD) {
        const int pair = threadIdx.x % 8, j = (threadIdx.x / 8) * 8;
        const uint4 v0 = *reinterpret_cast<const uint4*>(V + (size_t)min(k0 + 2 * pair, last_key) * kvd + j);
        const uint4 v1 = *reinterpret_cast<const uint4*>(V + (size_t)min(k0 + 2 * pair + 1, last_key) * kvd + j);
        const uint16_t* e0 = reinterpret_cast<const uint16_t*>(&v0);
        const uint16_t* e1 = reinterpret_cast<const uint16_t*>(&v1);
#pragma unroll
        for (int u = 0; u < 8; u++) vt[VT_AT(j + u) + pair] = (uint32_t)e0[u] | (uint32_t)e1[u] << 16;
    }
}

// qb  [rows][HD]           q after RoPE of one kv head's group, row = position * G + head in group
// K, V                     that kv head's cache: element (key, j) at K[key * kvd + j]
// ab  [positions][hidden]  output; row `row` goes to ab[(row / G) * hidden + out_off + (row % G) * HD]
// Position p of the chunk attends to keys 0 ..= pos0 + p; all of them are already in the cache.
//
// A block walks its keys one piece after another, so with few rows and a long context (a short
// prompt continuation, the typing case) one block would be all there is. The keys are therefore cut
// into `segs` segments of `seg_len`, one block per (rows, segment): with segs > 1 a block leaves its
// rows' partial result for its segment (weighted V sum relative to the segment's largest score, that
// score, the sum of weights) in part / stat for k_w_flash_merge; with segs == 1 it writes ab itself.
//   part [segs][rows][HD] fp32     stat [segs][rows][2]: largest score, sum of weights
// Grid: (ceil(rows / (16 * FA_WARPS)), segs), FA_WARPS * 32 threads. SPLIT = (segs > 1): the
// unsplit kernel does not pay for the segment tests in its inner loop.
//
// For a decode step, which runs as a CUDA graph and so cannot have its position or slot in the
// arguments: with `dyn` (two ints on the device: the first row's position and the slot's offset in
// the cache, in elements) those replace pos0 and are added to K and V.
template <int HD, bool SPLIT>
__global__ void k_w_flash_attn(const __nv_bfloat16* __restrict__ qb, const __nv_bfloat16* __restrict__ K_arg,
                               const __nv_bfloat16* __restrict__ V_arg, __nv_bfloat16* __restrict__ ab, int rows, int G, int kvd,
                               int hidden, int out_off, int pos0_arg, int segs, int seg_len, float* __restrict__ part,
                               float* __restrict__ stat, const long long* __restrict__ dyn) {
    const int KS = HD / 16, DT = HD / 8;
    const int pos0 = dyn ? (int)dyn[0] : pos0_arg;
    const __nv_bfloat16* K = K_arg + (dyn ? dyn[1] : 0);
    const __nv_bfloat16* V = V_arg + (dyn ? dyn[1] : 0);
    // The staged pieces are laid out for the banks of shared memory (32, one per 4-byte word; lanes of
    // a warp that touch different words of one bank in one instruction take turns). A warp's lanes read
    // the same words of 8 different keys of K, or of 8 different elements of V: with rows of 64 or of 8
    // words those would be 4 or 16 banks for 32 lanes. So a row of K is followed by 4 unused words, and
    // every 4 rows of V by 4 unused words: then the 32 lanes are in 32 banks, when staging as well.
    __shared__ __align__(16) __nv_bfloat16 ks[16 * (HD + 8)];   // K piece: key k at k * (HD + 8), its elements in order
    __shared__ uint32_t vt[HD * 8 + HD];                        // V piece transposed: element e at VT_AT(e), 8 words of 2 keys
    static_assert(FA_WARPS * 32 >= 2 * HD, "staging needs a thread per 8 elements of 16 keys");
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
    const int m0 = (blockIdx.x * FA_WARPS + warp) * 16;
    const int last_row = rows - 1;
    const int row_lo = m0 + g, row_hi = m0 + g + 8;
    const bool has_lo = row_lo <= last_row, has_hi = row_hi <= last_row;
    const int load_lo = has_lo ? row_lo : last_row, load_hi = has_hi ? row_hi : last_row;
    // Keys each row may attend to (inclusive), and the last key any row of this block needs.
    const int valid_lo = pos0 + load_lo / G, valid_hi = pos0 + load_hi / G;
    const int block_last_row = min(last_row, (blockIdx.x + 1) * FA_WARPS * 16 - 1);
    const int block_valid = pos0 + block_last_row / G, last_key = pos0 + last_row / G;
    // The last key any row of this warp needs. The same in all lanes: mma needs the whole warp.
    const int warp_valid = pos0 + min(last_row, m0 + 15) / G;
    const float scale = rsqrtf((float)HD);
    // Where this thread's two words of each output piece go in ab, in 4-byte words (-1: no such row).
    // Worked out here so that nothing but these has to be kept through the loop for it.
    const int out_word_lo = has_lo ? ((load_lo / G) * hidden + out_off + (load_lo % G) * HD) / 2 + t : -1;
    const int out_word_hi = has_hi ? ((load_hi / G) * hidden + out_off + (load_hi % G) * HD) / 2 + t : -1;

    // This warp's 16 rows of q, as first operands of the score products.
    uint32_t aq[KS][4];
    {
        const uint32_t* q_lo = reinterpret_cast<const uint32_t*>(qb + (size_t)load_lo * HD);
        const uint32_t* q_hi = reinterpret_cast<const uint32_t*>(qb + (size_t)load_hi * HD);
#pragma unroll
        for (int s = 0; s < KS; s++) {
            aq[s][0] = q_lo[s * 8 + t];
            aq[s][1] = q_hi[s * 8 + t];
            aq[s][2] = q_lo[s * 8 + 4 + t];
            aq[s][3] = q_hi[s * 8 + 4 + t];
        }
    }
    float o[DT][4];
#pragma unroll
    for (int dt = 0; dt < DT; dt++) o[dt][0] = o[dt][1] = o[dt][2] = o[dt][3] = 0;
    float m_lo = -INFINITY, m_hi = -INFINITY, l_lo = 0, l_hi = 0;

    const int seg = blockIdx.y, k_begin = SPLIT ? seg * seg_len : 0, k_last = SPLIT ? min(block_valid, k_begin + seg_len - 1) : block_valid;
    for (int k0 = k_begin; k0 <= k_last; k0 += 16) {
        fa_stage<HD>(ks, vt, K, V, k0, last_key, kvd);
        __syncthreads();
        if (k0 <= warp_valid) {
            // Scores of this warp's rows against keys k0 + 8u + {2t, 2t+1}, u = 0, 1.
            float s[2][4];
#pragma unroll
            for (int u = 0; u < 2; u++) {
                s[u][0] = s[u][1] = s[u][2] = s[u][3] = 0;
                const uint32_t* kb = reinterpret_cast<const uint32_t*>(&ks[(u * 8 + g) * (HD + 8)]);
#pragma unroll
                for (int q = 0; q < KS; q++) mma_bf16(s[u], aq[q], kb[q * 8 + t], kb[q * 8 + 4 + t]);
            }
            float mx_lo = -INFINITY, mx_hi = -INFINITY;
#pragma unroll
            for (int u = 0; u < 2; u++) {
#pragma unroll
                for (int c = 0; c < 2; c++) {
                    const int key = k0 + u * 8 + 2 * t + c;
                    const bool in_seg = !SPLIT || key < k_begin + seg_len;
                    s[u][c] = key <= valid_lo && in_seg ? s[u][c] * scale : -INFINITY;
                    s[u][2 + c] = key <= valid_hi && in_seg ? s[u][2 + c] * scale : -INFINITY;
                    mx_lo = fmaxf(mx_lo, s[u][c]);
                    mx_hi = fmaxf(mx_hi, s[u][2 + c]);
                }
            }
            // Online softmax: raise each row's running max, rescale what was accumulated under the
            // old one, then add this step's weights.
            const float n_lo = fmaxf(m_lo, group_max(mx_lo)), n_hi = fmaxf(m_hi, group_max(mx_hi));
            // (m = -inf before the row's first key: nothing accumulated yet, the factor is irrelevant but must not be NaN)
            const float f_lo = n_lo == m_lo ? 1.0f : (SPLIT && m_lo == -INFINITY ? 0.0f : expf(m_lo - n_lo));
            const float f_hi = n_hi == m_hi ? 1.0f : (SPLIT && m_hi == -INFINITY ? 0.0f : expf(m_hi - n_hi));
            m_lo = n_lo, m_hi = n_hi;
            if (f_lo != 1.0f || f_hi != 1.0f) {
                l_lo *= f_lo, l_hi *= f_hi;
#pragma unroll
                for (int dt = 0; dt < DT; dt++) o[dt][0] *= f_lo, o[dt][1] *= f_lo, o[dt][2] *= f_hi, o[dt][3] *= f_hi;
            }
            uint32_t ap[4];
#pragma unroll
            for (int u = 0; u < 2; u++) {
                // exp(-inf) = 0 for masked keys. A row with no key yet in this segment has m = -inf and
                // only masked scores: its weights are 0, not exp(-inf + inf).
                // (Unsplit, every row has key 0, so m is finite and exp(-inf - m) = 0 does it.)
                const float p0 = SPLIT && s[u][0] == -INFINITY ? 0.0f : expf(s[u][0] - m_lo), p1 = SPLIT && s[u][1] == -INFINITY ? 0.0f : expf(s[u][1] - m_lo);
                const float p2 = SPLIT && s[u][2] == -INFINITY ? 0.0f : expf(s[u][2] - m_hi), p3 = SPLIT && s[u][3] == -INFINITY ? 0.0f : expf(s[u][3] - m_hi);
                ap[2 * u] = pack_bf16(p0, p1);
                ap[2 * u + 1] = pack_bf16(p2, p3);
                // Sum the weights as they are used (rounded), so that the output is an exact average.
                const __nv_bfloat162 r_lo = *reinterpret_cast<const __nv_bfloat162*>(&ap[2 * u]);
                const __nv_bfloat162 r_hi = *reinterpret_cast<const __nv_bfloat162*>(&ap[2 * u + 1]);
                const float2 w_lo = __bfloat1622float2(r_lo), w_hi = __bfloat1622float2(r_hi);
                l_lo += w_lo.x + w_lo.y;
                l_hi += w_hi.x + w_hi.y;
            }
#pragma unroll
            for (int dt = 0; dt < DT; dt++) {
                const uint32_t* vb = &vt[VT_AT(dt * 8 + g)];
                mma_bf16(o[dt], ap, vb[t], vb[4 + t]);
            }
        }
        __syncthreads();
    }

#undef VT_AT
    // A row's weight sum is spread over its 4 lanes.
    l_lo += __shfl_xor_sync(0xffffffff, l_lo, 1);
    l_lo += __shfl_xor_sync(0xffffffff, l_lo, 2);
    l_hi += __shfl_xor_sync(0xffffffff, l_hi, 1);
    l_hi += __shfl_xor_sync(0xffffffff, l_hi, 2);
    if (SPLIT) {
        float* p_lo = part + ((size_t)seg * rows + load_lo) * HD;
        float* p_hi = part + ((size_t)seg * rows + load_hi) * HD;
#pragma unroll
        for (int dt = 0; dt < DT; dt++) {
            if (has_lo) p_lo[dt * 8 + 2 * t] = o[dt][0], p_lo[dt * 8 + 2 * t + 1] = o[dt][1];
            if (has_hi) p_hi[dt * 8 + 2 * t] = o[dt][2], p_hi[dt * 8 + 2 * t + 1] = o[dt][3];
        }
        if (t == 0) {
            float* st = stat + ((size_t)seg * rows) * 2;
            if (has_lo) st[2 * load_lo] = m_lo, st[2 * load_lo + 1] = l_lo;
            if (has_hi) st[2 * load_hi] = m_hi, st[2 * load_hi + 1] = l_hi;
        }
        return;
    }
    const float i_lo = 1.0f / l_lo, i_hi = 1.0f / l_hi;
    uint32_t* out_lo = reinterpret_cast<uint32_t*>(ab) + out_word_lo;
    uint32_t* out_hi = reinterpret_cast<uint32_t*>(ab) + out_word_hi;
#pragma unroll
    for (int dt = 0; dt < DT; dt++) {
        // o[dt][0], o[dt][1]: row lo, elements 8 dt + 2t, +1;  o[dt][2], o[dt][3]: row hi.
        if (out_word_lo >= 0) out_lo[dt * 4] = pack_bf16(o[dt][0] * i_lo, o[dt][1] * i_lo);
        if (out_word_hi >= 0) out_hi[dt * 4] = pack_bf16(o[dt][2] * i_hi, o[dt][3] * i_hi);
    }
}

// Adds a row's segments in segment order: ab = sum_s exp(m_s - m) part_s / sum_s exp(m_s - m) l_s.
// Grid: rows, HD threads.
__global__ void k_w_flash_merge(const float* __restrict__ part, const float* __restrict__ stat, __nv_bfloat16* __restrict__ ab,
                                int rows, int segs, int HD, int G, int hidden, int out_off) {
    const int row = blockIdx.x, j = threadIdx.x;
    float m = -INFINITY;
    for (int s = 0; s < segs; s++) m = fmaxf(m, stat[((size_t)s * rows + row) * 2]);
    float total = 0, denom = 0;
    for (int s = 0; s < segs; s++) {
        const float* st = stat + ((size_t)s * rows + row) * 2;
        if (st[1] == 0.0f) continue;  // no key of this row in the segment
        const float f = expf(st[0] - m);
        total += f * part[((size_t)s * rows + row) * HD + j];
        denom += f * st[1];
    }
    ab[(size_t)(row / G) * hidden + out_off + (row % G) * HD + j] = __float2bfloat16(total / denom);
}

// --- Decode attention on the tensor cores (bf16 KV cache) ---
//
// One warp takes ATT_TILE keys of one kv head for one or two rows of the step and leaves, per row and
// q head of the kv head's group, the weighted V sum relative to the tile's largest score, that score
// and the tile's sum of weights, for k_t_attn_merge. A row's q heads are 8 of the 16 rows of the
// products (the first 8 for the pair's first row, the last 8 for its second); a pair is two adjacent
// positions of one slot, so both read the same K and V. K and V are read straight from the cache:
// K's operand is pairs of consecutive elements of one key, V's is pairs of consecutive keys of one
// element, gathered with two loads. A row's result does not depend on whether it has a partner.
//
// pairs [2 * n_pairs]: the rows of each pair; the second is -1 if the pair has one row.
// Grid: (tiles, kv_heads, n_pairs), 32 threads.
//   part  [rows][kv_heads][tiles][group * head_dim]     stat  [rows][kv_heads][tiles][2][ATT_MAX_GROUP]
template <int HD>
__global__ void k_t_attn_tile(const float* __restrict__ q, const int32_t* __restrict__ slot, const int32_t* __restrict__ pos,
                              const int32_t* __restrict__ pairs, const __nv_bfloat16* __restrict__ Kc,
                              const __nv_bfloat16* __restrict__ Vc, float* __restrict__ part, float* __restrict__ stat,
                              int layer, int tiles, Dims d) {
    const int KS = HD / 16, DT = HD / 8;
    const int tile = blockIdx.x, kvh = blockIdx.y, kvd = d.kv_dim(), H = d.hidden();
    const int r_lo = pairs[2 * blockIdx.z], r_hi = pairs[2 * blockIdx.z + 1];
    const int G = d.q_heads / d.kv_heads, h0 = kvh * G, t0 = tile * ATT_TILE;
    const int T_lo = pos[r_lo] + 1, T_hi = r_hi >= 0 ? pos[r_hi] + 1 : 0, T = max(T_lo, T_hi);
    if (t0 >= T) return;
    const int e_lo = min(T_lo, t0 + ATT_TILE), e_hi = min(T_hi, t0 + ATT_TILE), t_end = max(e_lo, e_hi);
    const int g = threadIdx.x >> 2, t = threadIdx.x & 3;
    const bool real = g < G, has_hi = real && r_hi >= 0;   // rows g >= G of each half are padding
    const size_t base = d.kv_at(slot[r_lo], layer, 0) + kvh * HD;
    const float scale = rsqrtf((float)HD);

    uint32_t aq[KS][4];
    {
        const float* q_lo = q + (size_t)r_lo * H + (h0 + (real ? g : 0)) * HD;
        const float* q_hi = q + (size_t)(has_hi ? r_hi : r_lo) * H + (h0 + (real ? g : 0)) * HD;
#pragma unroll
        for (int s = 0; s < KS; s++) {
            aq[s][0] = real ? pack_bf16(q_lo[s * 16 + 2 * t], q_lo[s * 16 + 2 * t + 1]) : 0;
            aq[s][2] = real ? pack_bf16(q_lo[s * 16 + 8 + 2 * t], q_lo[s * 16 + 8 + 2 * t + 1]) : 0;
            aq[s][1] = has_hi ? pack_bf16(q_hi[s * 16 + 2 * t], q_hi[s * 16 + 2 * t + 1]) : 0;
            aq[s][3] = has_hi ? pack_bf16(q_hi[s * 16 + 8 + 2 * t], q_hi[s * 16 + 8 + 2 * t + 1]) : 0;
        }
    }
    float o[DT][4];
#pragma unroll
    for (int dt = 0; dt < DT; dt++) o[dt][0] = o[dt][1] = o[dt][2] = o[dt][3] = 0;
    float m_lo = -INFINITY, m_hi = -INFINITY, l_lo = 0, l_hi = 0;

    for (int k0 = t0; k0 < t_end; k0 += 16) {
        float s[2][4];
#pragma unroll
        for (int u = 0; u < 2; u++) {
            s[u][0] = s[u][1] = s[u][2] = s[u][3] = 0;
            const uint32_t* kb = reinterpret_cast<const uint32_t*>(Kc + base + (size_t)min(k0 + u * 8 + g, T - 1) * kvd);
#pragma unroll
            for (int x = 0; x < KS; x++) mma_bf16(s[u], aq[x], kb[x * 8 + t], kb[x * 8 + 4 + t]);
        }
        float mx_lo = -INFINITY, mx_hi = -INFINITY;
#pragma unroll
        for (int u = 0; u < 2; u++)
#pragma unroll
            for (int c = 0; c < 2; c++) {
                const int key = k0 + u * 8 + 2 * t + c;
                s[u][c] = key < e_lo ? s[u][c] * scale : -INFINITY;
                s[u][2 + c] = key < e_hi ? s[u][2 + c] * scale : -INFINITY;
                mx_lo = fmaxf(mx_lo, s[u][c]);
                mx_hi = fmaxf(mx_hi, s[u][2 + c]);
            }
        // Online softmax per row, as in k_w_flash_attn. A row with no key in this step (the pair's
        // other row has one more) keeps its state: its scores are all -inf and its weights 0.
        const float n_lo = fmaxf(m_lo, group_max(mx_lo)), n_hi = fmaxf(m_hi, group_max(mx_hi));
        if (n_lo != m_lo) {
            const float f = m_lo == -INFINITY ? 0.0f : expf(m_lo - n_lo);
            l_lo *= f;
#pragma unroll
            for (int dt = 0; dt < DT; dt++) o[dt][0] *= f, o[dt][1] *= f;
            m_lo = n_lo;
        }
        if (n_hi != m_hi) {
            const float f = m_hi == -INFINITY ? 0.0f : expf(m_hi - n_hi);
            l_hi *= f;
#pragma unroll
            for (int dt = 0; dt < DT; dt++) o[dt][2] *= f, o[dt][3] *= f;
            m_hi = n_hi;
        }
        uint32_t ap[4];
#pragma unroll
        for (int u = 0; u < 2; u++) {
            ap[2 * u] = pack_bf16(s[u][0] == -INFINITY ? 0.0f : expf(s[u][0] - m_lo), s[u][1] == -INFINITY ? 0.0f : expf(s[u][1] - m_lo));
            ap[2 * u + 1] = pack_bf16(s[u][2] == -INFINITY ? 0.0f : expf(s[u][2] - m_hi), s[u][3] == -INFINITY ? 0.0f : expf(s[u][3] - m_hi));
            const float2 w_lo = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&ap[2 * u]));
            const float2 w_hi = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&ap[2 * u + 1]));
            l_lo += w_lo.x + w_lo.y;
            l_hi += w_hi.x + w_hi.y;
        }
        // V operand: keys k0 + 2t, +1 and k0 + 8 + 2t, +1 of element 8 dt + g (clamped keys carry weight 0).
        const __nv_bfloat16* v0 = Vc + base + (size_t)min(k0 + 2 * t, T - 1) * kvd + g;
        const __nv_bfloat16* v1 = Vc + base + (size_t)min(k0 + 2 * t + 1, T - 1) * kvd + g;
        const __nv_bfloat16* v2 = Vc + base + (size_t)min(k0 + 8 + 2 * t, T - 1) * kvd + g;
        const __nv_bfloat16* v3 = Vc + base + (size_t)min(k0 + 8 + 2 * t + 1, T - 1) * kvd + g;
#pragma unroll
        for (int dt = 0; dt < DT; dt++) {
            const uint32_t b0 = (uint32_t)*reinterpret_cast<const uint16_t*>(v0 + dt * 8) | (uint32_t)*reinterpret_cast<const uint16_t*>(v1 + dt * 8) << 16;
            const uint32_t b1 = (uint32_t)*reinterpret_cast<const uint16_t*>(v2 + dt * 8) | (uint32_t)*reinterpret_cast<const uint16_t*>(v3 + dt * 8) << 16;
            mma_bf16(o[dt], ap, b0, b1);
        }
    }
    l_lo += __shfl_xor_sync(0xffffffff, l_lo, 1);
    l_lo += __shfl_xor_sync(0xffffffff, l_lo, 2);
    l_hi += __shfl_xor_sync(0xffffffff, l_hi, 1);
    l_hi += __shfl_xor_sync(0xffffffff, l_hi, 2);
    if (!real) return;
    {
        float* pt = part + (((size_t)r_lo * d.kv_heads + kvh) * tiles + tile) * G * HD + g * HD;
#pragma unroll
        for (int dt = 0; dt < DT; dt++) pt[dt * 8 + 2 * t] = o[dt][0], pt[dt * 8 + 2 * t + 1] = o[dt][1];
        float* st = stat + (((size_t)r_lo * d.kv_heads + kvh) * tiles + tile) * 2 * ATT_MAX_GROUP;
        if (t == 0) st[g] = m_lo, st[ATT_MAX_GROUP + g] = l_lo;
    }
    if (has_hi) {
        float* pt = part + (((size_t)r_hi * d.kv_heads + kvh) * tiles + tile) * G * HD + g * HD;
#pragma unroll
        for (int dt = 0; dt < DT; dt++) pt[dt * 8 + 2 * t] = o[dt][2], pt[dt * 8 + 2 * t + 1] = o[dt][3];
        float* st = stat + (((size_t)r_hi * d.kv_heads + kvh) * tiles + tile) * 2 * ATT_MAX_GROUP;
        if (t == 0) st[g] = m_hi, st[ATT_MAX_GROUP + g] = l_hi;   // l = 0: no key of this row in the tile
    }
}

// k_f_attn_merge for a row whose tiles may hold no key of it (stat sum 0), with the output in bf16.
// Grid: (kv_heads, rows), ATT_BLOCK threads.
__global__ void k_t_attn_merge(const int32_t* __restrict__ pos, const float* __restrict__ part, const float* __restrict__ stat,
                               __nv_bfloat16* __restrict__ out, int tiles, Dims d) {
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
    for (int w = 0; w < n_tiles; w++) {
        const float e = expf(sm[w * ATT_MAX_GROUP + i] - m);
        total += e * pt[(size_t)w * G * hd + o];
        denom += e * ss[w * ATT_MAX_GROUP + i];
    }
    out[(size_t)r * d.hidden() + kvh * G * hd + o] = __float2bfloat16(total / denom);
}
