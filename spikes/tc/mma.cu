// Spike: can a hand-written kernel drive the tensor cores on this GPU (sm_89) through inline PTX,
// and how much of cuBLAS's throughput does a simple one reach at the shape of attention scores?
//
//   nvcc -O3 -arch=sm_89 -lcublas -o mma mma.cu && ./mma
//
// S[m][n] = sum_k A[m][k] * B[n][k]   with A [M][K], B [N][K] bf16 row-major, K = 128, S fp32:
// the scores of M query rows against N cached positions. One mma.m16n8k16 multiplies a 16 x 16 piece
// of A by a 16 x 8 piece of B^T into a 16 x 8 piece of S with fp32 accumulation.
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e) { printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)
#define CB(x) do { cublasStatus_t st_ = (x); if (st_) { printf("%s:%d cublas %d\n", __FILE__, __LINE__, (int)st_); exit(1); } } while (0)

static const int K = 128, KS = K / 16;   // k-steps per output tile
#ifndef TILES
#define TILES 4
#endif
#ifndef NCHUNK
#define NCHUNK 256                       // columns of S one warp produces (a multiple of 8 * TILES)
#endif
#ifndef WARPS
#define WARPS 8
#endif

// d (16 x 8, fp32) += a (16 x 16, bf16) * b (16 x 8, bf16). Per lane: a = 4 words of 2 bf16,
// b = 2 words, c/d = 4 floats, in the fragment layout of the PTX mma instruction.
__device__ __forceinline__ void mma_bf16(float* d, const uint32_t* a, const uint32_t* b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// Grid: (N / (NCHUNK * WARPS), M / 16), WARPS * 32 threads. Each warp keeps its 16 rows of A in
// registers and walks NCHUNK columns, 8 at a time. With `store` the scores are written; without, only
// each row's running maximum is kept (as a fused softmax would consume them) and written at the end.
__global__ void scores(const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B, float* __restrict__ S,
                       float* __restrict__ rowmax, int N, bool store) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
    const int m0 = blockIdx.y * 16, n_begin = (blockIdx.x * WARPS + warp) * NCHUNK;
    const uint32_t* a_lo = reinterpret_cast<const uint32_t*>(A + (size_t)(m0 + g) * K);      // row g
    const uint32_t* a_hi = reinterpret_cast<const uint32_t*>(A + (size_t)(m0 + g + 8) * K);  // row g + 8
    uint32_t a[KS][4];
#pragma unroll
    for (int ks = 0; ks < KS; ks++) {
        a[ks][0] = a_lo[ks * 8 + t];        // k = 16 ks + 2t, +1
        a[ks][1] = a_hi[ks * 8 + t];
        a[ks][2] = a_lo[ks * 8 + 4 + t];    // k = 16 ks + 8 + 2t, +1
        a[ks][3] = a_hi[ks * 8 + 4 + t];
    }
    float mx_lo = -INFINITY, mx_hi = -INFINITY;
    // TILES column tiles (8 columns each) per step: their mma chains are independent, so they overlap.
    for (int n0 = n_begin; n0 < n_begin + NCHUNK && n0 < N; n0 += 8 * TILES) {
        const uint32_t* b_row[TILES];
        float d[TILES][4];
#pragma unroll
        for (int u = 0; u < TILES; u++) {
            b_row[u] = reinterpret_cast<const uint32_t*>(B + (size_t)(n0 + 8 * u + g) * K);  // column g of tile u
            d[u][0] = d[u][1] = d[u][2] = d[u][3] = 0;
        }
#pragma unroll
        for (int ks = 0; ks < KS; ks++) {
#pragma unroll
            for (int u = 0; u < TILES; u++) {
                uint32_t b[2] = {b_row[u][ks * 8 + t], b_row[u][ks * 8 + 4 + t]};
                mma_bf16(d[u], a[ks], b);
            }
        }
        // d[u][0], d[u][1]: row g, columns 2t and 2t+1 of tile u;  d[u][2], d[u][3]: row g + 8.
#pragma unroll
        for (int u = 0; u < TILES; u++) {
            if (store) {
                float2* lo = reinterpret_cast<float2*>(S + (size_t)(m0 + g) * N + n0 + 8 * u + 2 * t);
                float2* hi = reinterpret_cast<float2*>(S + (size_t)(m0 + g + 8) * N + n0 + 8 * u + 2 * t);
                *lo = make_float2(d[u][0], d[u][1]);
                *hi = make_float2(d[u][2], d[u][3]);
            }
            mx_lo = fmaxf(mx_lo, fmaxf(d[u][0], d[u][1]));
            mx_hi = fmaxf(mx_hi, fmaxf(d[u][2], d[u][3]));
        }
    }
    if (!store) {  // one value per (row, lane-in-group, warp): enough to keep the work from being optimized away
        rowmax[((size_t)(m0 + g) * WARPS + warp) * 4 + t] = mx_lo;
        rowmax[((size_t)(m0 + g + 8) * WARPS + warp) * 4 + t] = mx_hi;
    }
}

static float ms_of(cudaEvent_t a, cudaEvent_t b) {
    float ms;
    CK(cudaEventSynchronize(b));
    CK(cudaEventElapsedTime(&ms, a, b));
    return ms;
}

int main(int argc, char** argv) {
    const int M = argc > 1 ? atoi(argv[1]) : 768, N = argc > 2 ? atoi(argv[2]) : 4096, REPS = 200;
    printf("S = A B^T: M = %d rows, N = %d positions, K = %d (%.1f MB of scores per product)\n", M, N, K, (double)M * N * 4 / 1e6);
    std::vector<__nv_bfloat16> ha((size_t)M * K), hb((size_t)N * K);
    unsigned long long st = 88172645463325252ull;
    auto rnd = [&]() { st ^= st << 13; st ^= st >> 7; st ^= st << 17; return ((int)(st >> 40) % 2001 - 1000) * 1e-3f; };
    for (auto& v : ha) v = __float2bfloat16(rnd());
    for (auto& v : hb) v = __float2bfloat16(rnd());
    __nv_bfloat16 *A, *B; float *S, *S2, *rowmax;
    CK(cudaMalloc(&A, ha.size() * 2)); CK(cudaMalloc(&B, hb.size() * 2));
    CK(cudaMalloc(&S, (size_t)M * N * 4)); CK(cudaMalloc(&S2, (size_t)M * N * 4)); CK(cudaMalloc(&rowmax, (size_t)M * WARPS * 4 * 4));
    CK(cudaMemcpy(A, ha.data(), ha.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(B, hb.data(), hb.size() * 2, cudaMemcpyHostToDevice));
    cudaStream_t s; CK(cudaStreamCreate(&s));
    cublasHandle_t h; CB(cublasCreate(&h)); CB(cublasSetStream(h, s));
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    const dim3 grid((N + NCHUNK * WARPS - 1) / (NCHUNK * WARPS), M / 16);
    const float one = 1, zero = 0;
    // cuBLAS, as the engine calls it: column-major S^T (N x M) = B (K x N)^T * A (K x M).
    auto blas = [&]() { CB(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &one, B, CUDA_R_16BF, K, A, CUDA_R_16BF, K, &zero, S2, CUDA_R_32F, N,
                                         CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP)); };

    // 1. correctness against the CPU and against cuBLAS
    scores<<<grid, WARPS * 32, 0, s>>>(A, B, S, rowmax, N, true);
    blas();
    std::vector<float> hs((size_t)M * N), hs2((size_t)M * N);
    CK(cudaMemcpyAsync(hs.data(), S, hs.size() * 4, cudaMemcpyDeviceToHost, s));
    CK(cudaMemcpyAsync(hs2.data(), S2, hs2.size() * 4, cudaMemcpyDeviceToHost, s));
    CK(cudaStreamSynchronize(s)); CK(cudaGetLastError());
    double worst_cpu = 0, worst_blas = 0, scale = 0;
    for (int i = 0; i < 4000; i++) {
        st = st * 6364136223846793005ull + 1442695040888963407ull;
        int m = (int)((st >> 33) % M), n = (int)((st >> 13) % N);
        double ref = 0;
        for (int k = 0; k < K; k++) ref += (double)__bfloat162float(ha[(size_t)m * K + k]) * __bfloat162float(hb[(size_t)n * K + k]);
        worst_cpu = fmax(worst_cpu, fabs(hs[(size_t)m * N + n] - ref));
        scale = fmax(scale, fabs(ref));
    }
    for (size_t i = 0; i < hs.size(); i++) worst_blas = fmax(worst_blas, fabs((double)hs[i] - hs2[i]));
    printf("correctness: max |ours - exact| over 4000 sampled scores %.2e (largest score %.1f); max |ours - cuBLAS| over all %.2e\n",
           worst_cpu, scale, worst_blas);

    // 2. throughput
    const double flop = 2.0 * M * N * K;
    for (int variant = 0; variant < 3; variant++) {
        const char* name = variant == 0 ? "ours, scores stored" : variant == 1 ? "ours, scores consumed in registers" : "cuBLAS GemmEx (stores scores)";
        auto run = [&]() { if (variant == 2) blas(); else scores<<<grid, WARPS * 32, 0, s>>>(A, B, S, rowmax, N, variant == 0); };
        for (int i = 0; i < 20; i++) run();
        CK(cudaEventRecord(e0, s));
        for (int i = 0; i < REPS; i++) run();
        CK(cudaEventRecord(e1, s));
        float ms = ms_of(e0, e1) / REPS;
        printf("  %-36s %7.3f ms  %5.1f TFLOPS\n", name, ms, flop / (ms * 1e-3) / 1e12);
    }
    CK(cudaGetLastError());
    return 0;
}
