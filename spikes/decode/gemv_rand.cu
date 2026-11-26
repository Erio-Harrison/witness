// Spike: how much of the ~250 GB/s read bandwidth can a bf16 GEMV reach at Qwen2.5-1.5B shapes,
// per shape and for a whole decode step (28 layers + lm_head), ours vs cuBLAS, batch 1..4.
//
//   nvcc -O3 -arch=sm_89 -lcublas -o gemv gemv.cu && ./gemv
//
// y[b][n] = sum_k W[n][k] * x[b][k], W row-major [N][K] bf16, x/y bf16, fp32 accumulate.
// Weights are cycled over many copies so the 32 MB L2 cannot serve them.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e) { printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)
#define CB(x) do { cublasStatus_t st_ = (x); if (st_) { printf("%s:%d cublas %d\n", __FILE__, __LINE__, (int)st_); exit(1); } } while (0)

// One warp per output row; each lane reads 16 B (8 bf16) per step along K. B = batch (1..4).
template <int B>
__global__ void gemv_bf16(const __nv_bfloat16* __restrict__ W, const __nv_bfloat16* __restrict__ x,
                          __nv_bfloat16* __restrict__ y, int N, int K) {
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (warp >= N) return;
    const uint4* w = reinterpret_cast<const uint4*>(W + (size_t)warp * K);
    float acc[B] = {};
    for (int i = lane; i < K / 8; i += 32) {
        uint4 wv = __ldcs(w + i);  // streaming: weights are read once per step
        const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(&wv);
#pragma unroll
        for (int b = 0; b < B; b++) {
            uint4 xv = __ldg(reinterpret_cast<const uint4*>(x + (size_t)b * K) + i);
            const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(&xv);
#pragma unroll
            for (int j = 0; j < 4; j++) {
                float2 wf = __bfloat1622float2(w2[j]), xf = __bfloat1622float2(x2[j]);
                acc[b] += wf.x * xf.x + wf.y * xf.y;
            }
        }
    }
#pragma unroll
    for (int b = 0; b < B; b++) {
        float v = acc[b];
        for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
        if (lane == 0) y[(size_t)b * N + warp] = __float2bfloat16(v);
    }
}

static void launch(int B, const __nv_bfloat16* W, const __nv_bfloat16* x, __nv_bfloat16* y, int N, int K, cudaStream_t s) {
    int threads = 256, blocks = (N * 32 + threads - 1) / threads;
    switch (B) {
    case 1: gemv_bf16<1><<<blocks, threads, 0, s>>>(W, x, y, N, K); break;
    case 2: gemv_bf16<2><<<blocks, threads, 0, s>>>(W, x, y, N, K); break;
    case 3: gemv_bf16<3><<<blocks, threads, 0, s>>>(W, x, y, N, K); break;
    case 4: gemv_bf16<4><<<blocks, threads, 0, s>>>(W, x, y, N, K); break;
    }
}

static void cublas_gemv(cublasHandle_t h, int B, const __nv_bfloat16* W, const __nv_bfloat16* x, __nv_bfloat16* y, int N, int K) {
    float one = 1, zero = 0;  // column-major view: W is K x N, y (N x B) = W^T (N x K) * x (K x B)
    CB(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, B, K, &one, W, CUDA_R_16BF, K, x, CUDA_R_16BF, K, &zero, y,
                    CUDA_R_16BF, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}

struct Shape { const char* name; int N, K; };

// Fill device memory with small random bf16 values (|v| < 0.06, like real weights) when FILL < 0.
static void fill(void* dev, size_t bytes, int FILL) {
    if (FILL >= 0) { CK(cudaMemset(dev, FILL, bytes)); return; }
    static std::vector<__nv_bfloat16> host;
    static unsigned long long st = 88172645463325252ull;
    host.resize(bytes / 2);
    for (auto& v : host) {
        st ^= st << 13; st ^= st >> 7; st ^= st << 17;
        v = __float2bfloat16(((int)(st >> 40) % 2001 - 1000) * 6e-5f);
    }
    CK(cudaMemcpy(dev, host.data(), bytes, cudaMemcpyHostToDevice));
}

int main(int argc, char** argv) {
    int STEPS = argc > 1 ? atoi(argv[1]) : 20; int FILL = argc > 2 ? atoi(argv[2]) : 0;
    const int H = 1536, I = 8960, KV = 256, V = 151936, L = 28;
    Shape layer[] = {{"qkv", H + 2 * KV, H}, {"o", H, H}, {"gate_up", 2 * I, H}, {"down", H, I}};
    Shape head = {"lm_head", V, H};
    cudaStream_t s; CK(cudaStreamCreate(&s));
    cublasHandle_t h; CB(cublasCreate(&h)); CB(cublasSetStream(h, s));
    __nv_bfloat16 *x, *y; CK(cudaMalloc(&x, 4 * I * 2)); CK(cudaMalloc(&y, 4 * (size_t)V * 2));
    fill(x, 4 * I * 2, FILL);
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));

    // ---- a whole decode step's weight reads: 28 layers + lm_head, all distinct, in one CUDA graph
    size_t per_layer = 0;
    for (auto& sh : layer) per_layer += (size_t)sh.N * sh.K * 2;
    size_t total = per_layer * L + (size_t)V * H * 2;
    std::vector<__nv_bfloat16*> lw(L * 4);
    for (int l = 0; l < L; l++)
        for (int j = 0; j < 4; j++) {
            size_t bytes = (size_t)layer[j].N * layer[j].K * 2;
            CK(cudaMalloc(&lw[l * 4 + j], bytes)); fill(lw[l * 4 + j], bytes, FILL);
        }
    __nv_bfloat16* hw; CK(cudaMalloc(&hw, (size_t)V * H * 2)); fill(hw, (size_t)V * H * 2, FILL);
    printf("\nwhole step: %.2f GB of weights (28 layers + lm_head), ideal at 250 GB/s = %.2f ms\n", total / 1e9, total / 250e9 * 1e3);
    for (int B = 1; B <= 4; B += 3) {
        for (int impl = 0; impl < 1; impl++) {
            cudaGraph_t g; cudaGraphExec_t ge;
            CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
            for (int l = 0; l < L; l++)
                for (int j = 0; j < 4; j++)
                    impl ? cublas_gemv(h, B, lw[l * 4 + j], x, y, layer[j].N, layer[j].K)
                         : launch(B, lw[l * 4 + j], x, y, layer[j].N, layer[j].K, s);
            impl ? cublas_gemv(h, B, hw, x, y, V, H) : launch(B, hw, x, y, V, H, s);
            CK(cudaStreamEndCapture(s, &g)); CK(cudaGraphInstantiate(&ge, g, 0));
            for (int i = 0; i < 3; i++) CK(cudaGraphLaunch(ge, s));
            CK(cudaEventRecord(a, s));
            for (int i = 0; i < STEPS; i++) CK(cudaGraphLaunch(ge, s));
            CK(cudaEventRecord(b, s)); CK(cudaEventSynchronize(b));
            float ms; CK(cudaEventElapsedTime(&ms, a, b)); ms /= STEPS;
            printf("  B=%d %-6s step %.2f ms  %5.0f GB/s  -> %5.1f tok/s per user, %5.1f tok/s total\n", B,
                   impl ? "cuBLAS" : "ours", ms, total / (ms / 1e3) / 1e9, 1e3 / ms, B * 1e3 / ms);
            CK(cudaGraphExecDestroy(ge)); CK(cudaGraphDestroy(g));
        }
    }
    return 0;
}
