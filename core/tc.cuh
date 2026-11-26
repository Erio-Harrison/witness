// Matrix products of a decode step on the tensor cores, for 1 to 32 rows. Included by engine.cu.
//
// y[r][n] = sum_k W[n][k] x[r][k]: W [N][K] bf16 row-major, read once per step; x [rows][K] bf16;
// y [rows][N] fp32. One warp takes 16 weight rows. Per 64 elements along k a lane makes four 16-byte
// streaming loads (two from each of its two weight rows) and, per 8 x-rows, two loads of x and four
// mma.m16n8k16. The k index inside an mma is only a label, so it is assigned such that what a lane
// loaded is exactly its operand: mma j of a block pairs elements 64b + 16t + 2j, +1 and + 8 of the
// lane's rows. No shuffling, and the weights are read 16 bytes per lane in order, as a plain GEMV
// reads them.
//
// Measured on the matmul-only chain with random weights (spikes/tc/tcgemv.cu): 8 rows 18.5 ms per
// step, 16 rows 18.8, 24 rows 19.0, 32 rows 21.1, against 18.3 for one row of the scalar GEMV. A
// row's result does not depend on which row it is or on how many rows there are (bit for bit).
//
// CT = column tiles of 8 x-rows: x and y must hold 8 * CT rows; rows past the caller's count are
// computed from whatever x holds and written, and concern nobody.
// Grid: ceil(N / 128), 256 threads. K a multiple of 64.
template <int CT>
__global__ void k_t_gemv(const __nv_bfloat16* __restrict__ W, const __nv_bfloat16* __restrict__ x, float* __restrict__ y, int N, int K) {
    const int warp = (blockIdx.x * BLOCK + threadIdx.x) >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int n0 = warp * 16;
    if (n0 >= N) return;
    const uint4* w_lo = reinterpret_cast<const uint4*>(W + (size_t)min(n0 + g, N - 1) * K) + 2 * t;
    const uint4* w_hi = reinterpret_cast<const uint4*>(W + (size_t)min(n0 + g + 8, N - 1) * K) + 2 * t;
    float d[CT][4];
#pragma unroll
    for (int c = 0; c < CT; c++) d[c][0] = d[c][1] = d[c][2] = d[c][3] = 0;
    for (int b = 0; b < K / 64; b++) {
        const uint4 l0 = __ldcs(w_lo + b * 8), l1 = __ldcs(w_lo + b * 8 + 1), h0 = __ldcs(w_hi + b * 8), h1 = __ldcs(w_hi + b * 8 + 1);
        const uint32_t* L0 = reinterpret_cast<const uint32_t*>(&l0);
        const uint32_t* L1 = reinterpret_cast<const uint32_t*>(&l1);
        const uint32_t* H0 = reinterpret_cast<const uint32_t*>(&h0);
        const uint32_t* H1 = reinterpret_cast<const uint32_t*>(&h1);
#pragma unroll
        for (int c = 0; c < CT; c++) {
            const uint4* xr = reinterpret_cast<const uint4*>(x + (size_t)(c * 8 + g) * K) + 2 * t + b * 8;
            const uint4 x0 = xr[0], x1 = xr[1];
            const uint32_t* X0 = reinterpret_cast<const uint32_t*>(&x0);
            const uint32_t* X1 = reinterpret_cast<const uint32_t*>(&x1);
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const uint32_t a[4] = {L0[j], H0[j], L1[j], H1[j]};
                mma_bf16(d[c], a, X0[j], X1[j]);
            }
        }
    }
#pragma unroll
    for (int c = 0; c < CT; c++) {
        // d[c][0], d[c][1]: weight row n0 + g, x-rows 8c + 2t, +1;  d[c][2], d[c][3]: weight row n0 + g + 8.
        float* y0 = y + (size_t)(c * 8 + 2 * t) * N + n0 + g;
        if (n0 + g < N) y0[0] = d[c][0], y0[N] = d[c][1];
        if (n0 + g + 8 < N) y0[8] = d[c][2], y0[N + 8] = d[c][3];
    }
}

// hb[r] = bf16(rmsnorm(x[r]) * g). Grid: rows.
__global__ void k_t_norm(const float* __restrict__ x, const float* __restrict__ g, __nv_bfloat16* __restrict__ hb, int H, float eps) {
    __shared__ float sh[BLOCK / 32];
    const float* in = x + (size_t)blockIdx.x * H;
    __nv_bfloat16* out = hb + (size_t)blockIdx.x * H;
    float ss = 0;
    for (int i = threadIdx.x; i < H; i += BLOCK) ss += in[i] * in[i];
    float r = rsqrtf(block_sum(ss, sh) / H + eps);
    for (int i = threadIdx.x; i < H; i += BLOCK) out[i] = __float2bfloat16(in[i] * r * g[i]);
}
