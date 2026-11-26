// Spike: does a product cost more when its weights are not in the GPU's L2 cache, as in the engine
// (each layer's matrix is used once per chunk), than when the same matrix is used again and again?
//   nvcc -O3 -arch=sm_89 -o cold cold.cu -lcublas -lcublasLt && ./cold N K R
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CK(x) do { cudaError_t e = (x); if (e) { printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)
#define CB(x) do { cublasStatus_t st_ = (x); if (st_) { printf("%s:%d cublas %d\n", __FILE__, __LINE__, (int)st_); exit(1); } } while (0)
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
int main(int argc, char** argv) {
    const int N = atoi(argv[1]), K = atoi(argv[2]), R = atoi(argv[3]), COPIES = 28;
    __nv_bfloat16 *W, *X; float* Y;
    CK(cudaMalloc(&W, (size_t)COPIES * N * K * 2)); CK(cudaMalloc(&X, (size_t)COPIES * R * K * 2)); CK(cudaMalloc(&Y, (size_t)COPIES * R * N * 4));
    {
        std::vector<__nv_bfloat16> h((size_t)N * K);
        for (auto& v : h) v = __float2bfloat16((rand() % 2001 - 1000) * 2e-5f);
        for (int c = 0; c < COPIES; c++) CK(cudaMemcpy(W + (size_t)c * N * K, h.data(), h.size() * 2, cudaMemcpyHostToDevice));
        h.resize((size_t)R * K);
        for (auto& v : h) v = __float2bfloat16((rand() % 2001 - 1000) * 1e-3f);
        for (int c = 0; c < COPIES; c++) CK(cudaMemcpy(X + (size_t)c * R * K, h.data(), h.size() * 2, cudaMemcpyHostToDevice));
    }
    cublasLtHandle_t lt; CB(cublasLtCreate(&lt));
    cublasLtMatmulDesc_t desc; CB(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t ta = CUBLAS_OP_T;
    CB(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof ta));
    cublasLtMatrixLayout_t la, lb, ld;
    CB(cublasLtMatrixLayoutCreate(&la, CUDA_R_16BF, K, N, K)); CB(cublasLtMatrixLayoutCreate(&lb, CUDA_R_16BF, K, R, K)); CB(cublasLtMatrixLayoutCreate(&ld, CUDA_R_32F, N, R, N));
    cublasLtMatmulAlgo_t algo; int tile = 24, stages = 9;
    CB(cublasLtMatmulAlgoInit(lt, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_32F, CUDA_R_32F, 21, &algo));
    CB(cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof tile));
    CB(cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stages, sizeof stages));
    const float one = 1, zero = 0;
    // w, x, y: which of the three change from one product to the next
    for (int mode = 0; mode < 8; mode++) {
        double best = 1e9;
        for (int rep = 0; rep < 5; rep++) {
            CK(cudaDeviceSynchronize());
            double t0 = now();
            for (int i = 0; i < 4 * COPIES; i++) {
                int c = i % COPIES;
                CB(cublasLtMatmul(lt, desc, &one, W + (mode & 1 ? (size_t)c * N * K : 0), la, X + (mode & 2 ? (size_t)c * R * K : 0), lb, &zero,
                                  Y + (mode & 4 ? (size_t)c * R * N : 0), ld, Y + (mode & 4 ? (size_t)c * R * N : 0), ld, &algo, nullptr, 0, 0));
            }
            CK(cudaDeviceSynchronize());
            best = std::min(best, (now() - t0) / (4 * COPIES) * 1e3);
        }
        printf("N %d K %d R %d: weights %s, input %s, output %s: %.3f ms\n", N, K, R, mode & 1 ? "change" : "same  ", mode & 2 ? "change" : "same  ", mode & 4 ? "change" : "same  ", best);
    }
}
