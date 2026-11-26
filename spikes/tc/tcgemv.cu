// Spike: a decode step's matrix products on the tensor cores, for 1 to 32 rows.
//
//   nvcc -O3 -arch=sm_89 -o tcgemv tcgemv.cu && ./tcgemv [STEPS]
//
// y[r][n] = sum_k W[n][k] x[r][k], W [N][K] bf16 row-major (read once per step, as in decode),
// x [R][K] bf16, y fp32. One warp takes 16 weight rows; per 64 elements along k a lane makes four
// 16-byte loads (two from each of its two weight rows) and four mma.m16n8k16 per 8 x-rows. The
// k index inside an mma is only a label, so it is chosen such that a lane's own loads are its
// operands: no shuffling, and the weights are read as in the plain GEMV (16 bytes per lane, in order).
// Chain and shapes as spikes/decode/gemv_rand.cu: 28 layers of qkv, o, gate_up, down + lm_head, random
// weight values, whole step as one CUDA graph.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e) { printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

__device__ __forceinline__ void mma_bf16(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// CT = column tiles of 8 x-rows. Grid: ceil(N / 128), 256 threads. K a multiple of 64.
template <int CT>
__global__ void tc_gemv(const __nv_bfloat16* __restrict__ W, const __nv_bfloat16* __restrict__ x, float* __restrict__ y, int N, int K) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int n0 = warp * 16;
    if (n0 >= N) return;
    const int r_lo = min(n0 + g, N - 1), r_hi = min(n0 + g + 8, N - 1);
    const uint4* w_lo = reinterpret_cast<const uint4*>(W + (size_t)r_lo * K) + 2 * t;
    const uint4* w_hi = reinterpret_cast<const uint4*>(W + (size_t)r_hi * K) + 2 * t;
    float d[CT][4];
#pragma unroll
    for (int c = 0; c < CT; c++) d[c][0] = d[c][1] = d[c][2] = d[c][3] = 0;
    for (int b = 0; b < K / 64; b++) {
        const uint4 l0 = __ldcs(w_lo + b * 8), l1 = __ldcs(w_lo + b * 8 + 1), h0 = __ldcs(w_hi + b * 8), h1 = __ldcs(w_hi + b * 8 + 1);
        const uint32_t* L0 = reinterpret_cast<const uint32_t*>(&l0); const uint32_t* L1 = reinterpret_cast<const uint32_t*>(&l1);
        const uint32_t* H0 = reinterpret_cast<const uint32_t*>(&h0); const uint32_t* H1 = reinterpret_cast<const uint32_t*>(&h1);
#pragma unroll
        for (int c = 0; c < CT; c++) {
            const uint4* xr = reinterpret_cast<const uint4*>(x + (size_t)(c * 8 + g) * K) + 2 * t + b * 8;
            const uint4 x0 = xr[0], x1 = xr[1];
            const uint32_t* X0 = reinterpret_cast<const uint32_t*>(&x0); const uint32_t* X1 = reinterpret_cast<const uint32_t*>(&x1);
#pragma unroll
            for (int j = 0; j < 4; j++) mma_bf16(d[c], L0[j], H0[j], L1[j], H1[j], X0[j], X1[j]);
        }
    }
#pragma unroll
    for (int c = 0; c < CT; c++) {
        // d[c][0], d[c][1]: weight row n0 + g, x-rows 8c + 2t, +1;  d[c][2], d[c][3]: weight row n0 + g + 8.
        if (n0 + g < N) { y[(size_t)(c * 8 + 2 * t) * N + n0 + g] = d[c][0]; y[(size_t)(c * 8 + 2 * t + 1) * N + n0 + g] = d[c][1]; }
        if (n0 + g + 8 < N) { y[(size_t)(c * 8 + 2 * t) * N + n0 + g + 8] = d[c][2]; y[(size_t)(c * 8 + 2 * t + 1) * N + n0 + g + 8] = d[c][3]; }
    }
}

static void launch(int CT, const __nv_bfloat16* W, const __nv_bfloat16* x, float* y, int N, int K, cudaStream_t s) {
    int blocks = (N + 127) / 128;
    switch (CT) {
    case 1: tc_gemv<1><<<blocks, 256, 0, s>>>(W, x, y, N, K); break;
    case 2: tc_gemv<2><<<blocks, 256, 0, s>>>(W, x, y, N, K); break;
    case 3: tc_gemv<3><<<blocks, 256, 0, s>>>(W, x, y, N, K); break;
    case 4: tc_gemv<4><<<blocks, 256, 0, s>>>(W, x, y, N, K); break;
    }
}

static unsigned long long st = 88172645463325252ull;
static void fill(void* dev, size_t bytes, float amp) {
    static std::vector<__nv_bfloat16> host;
    host.resize(bytes / 2);
    for (auto& v : host) { st ^= st << 13; st ^= st >> 7; st ^= st << 17; v = __float2bfloat16(((int)(st >> 40) % 2001 - 1000) * amp); }
    CK(cudaMemcpy(dev, host.data(), bytes, cudaMemcpyHostToDevice));
}

struct Shape { const char* name; int N, K; };

int main(int argc, char** argv) {
    const int STEPS = argc > 1 ? atoi(argv[1]) : 200;
    const int H = 1536, I = 8960, KV = 256, V = 151936, L = 28;
    Shape layer[] = {{"qkv", H + 2 * KV, H}, {"o", H, H}, {"gate_up", 2 * I, H}, {"down", H, I}};
    cudaStream_t s; CK(cudaStreamCreate(&s));
    __nv_bfloat16* x; float* y;
    CK(cudaMalloc(&x, (size_t)32 * I * 2)); CK(cudaMalloc(&y, (size_t)32 * V * 4));
    fill(x, (size_t)32 * I * 2, 1e-3f);
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));

    // correctness of one product against the CPU
    {
        const int N = 300, K = 1536;
        std::vector<__nv_bfloat16> hw((size_t)N * K), hx((size_t)32 * I);
        __nv_bfloat16* W; CK(cudaMalloc(&W, hw.size() * 2)); fill(W, hw.size() * 2, 6e-5f);
        CK(cudaMemcpy(hw.data(), W, hw.size() * 2, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(hx.data(), x, hx.size() * 2, cudaMemcpyDeviceToHost));
        launch(4, W, x, y, N, K, s);
        std::vector<float> hy((size_t)32 * N); CK(cudaMemcpyAsync(hy.data(), y, hy.size() * 4, cudaMemcpyDeviceToHost, s)); CK(cudaStreamSynchronize(s));
        double worst = 0, scale = 0;
        for (int r = 0; r < 32; r++) for (int n = 0; n < N; n++) {
            double ref = 0;
            for (int k = 0; k < K; k++) ref += (double)__bfloat162float(hw[(size_t)n * K + k]) * __bfloat162float(hx[(size_t)r * K + k]);
            worst = fmax(worst, fabs(hy[(size_t)r * N + n] - ref)); scale = fmax(scale, fabs(ref));
        }
        // the same x in two different rows must give bit-identical results whatever the row
        CK(cudaMemcpy(x + (size_t)19 * K, x + (size_t)2 * K, K * 2, cudaMemcpyDeviceToDevice));
        launch(4, W, x, y, N, K, s);
        CK(cudaMemcpyAsync(hy.data(), y, hy.size() * 4, cudaMemcpyDeviceToHost, s)); CK(cudaStreamSynchronize(s));
        int differ = 0;
        for (int n = 0; n < N; n++) differ += hy[(size_t)19 * N + n] != hy[(size_t)2 * N + n];
        launch(1, W, x, y, N, K, s);
        std::vector<float> h1((size_t)8 * N); CK(cudaMemcpyAsync(h1.data(), y, h1.size() * 4, cudaMemcpyDeviceToHost, s)); CK(cudaStreamSynchronize(s));
        int differ1 = 0;
        for (int n = 0; n < N; n++) differ1 += h1[(size_t)2 * N + n] != hy[(size_t)2 * N + n];
        printf("correctness: max |ours - exact| %.2e (largest value %.2e); same x in rows 2 and 19: %d of %d results differ; row 2 with 8 rows vs with 32: %d differ\n",
               worst, scale, differ, N, differ1);
        CK(cudaFree(W));
    }

    size_t per_layer = 0;
    for (auto& sh : layer) per_layer += (size_t)sh.N * sh.K * 2;
    size_t total = per_layer * L + (size_t)V * H * 2;
    std::vector<__nv_bfloat16*> lw(L * 4);
    for (int l = 0; l < L; l++)
        for (int j = 0; j < 4; j++) {
            size_t bytes = (size_t)layer[j].N * layer[j].K * 2;
            CK(cudaMalloc(&lw[l * 4 + j], bytes)); fill(lw[l * 4 + j], bytes, 6e-5f);
        }
    __nv_bfloat16* hw; CK(cudaMalloc(&hw, (size_t)V * H * 2)); fill(hw, (size_t)V * H * 2, 6e-5f);
    printf("whole step, %.2f GB of random weights, %d steps:\n", total / 1e9, STEPS);
    for (int CT = 1; CT <= 4; CT++) {
        cudaGraph_t g; cudaGraphExec_t ge;
        CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
        for (int l = 0; l < L; l++)
            for (int j = 0; j < 4; j++) launch(CT, lw[l * 4 + j], x, y, layer[j].N, layer[j].K, s);
        launch(CT, hw, x, y, V, H, s);
        CK(cudaStreamEndCapture(s, &g)); CK(cudaGraphInstantiate(&ge, g, 0));
        for (int i = 0; i < 3; i++) CK(cudaGraphLaunch(ge, s));
        CK(cudaEventRecord(a, s));
        for (int i = 0; i < STEPS; i++) CK(cudaGraphLaunch(ge, s));
        CK(cudaEventRecord(b, s)); CK(cudaEventSynchronize(b));
        float ms; CK(cudaEventElapsedTime(&ms, a, b)); ms /= STEPS;
        printf("  %2d rows: step %.2f ms  %5.0f GB/s\n", CT * 8, ms, total / (ms / 1e3) / 1e9);
        CK(cudaGraphExecDestroy(ge)); CK(cudaGraphDestroy(g));
    }
    return 0;
}
