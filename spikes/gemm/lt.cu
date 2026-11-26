// Spike: is cublasGemmEx's choice of kernel the fastest cuBLAS has for the wide path's products on
// this GPU? Times cublasGemmEx against every cublasLt algorithm configuration (algorithm x tile x
// stages x custom option) for y[R][N] = x[R][K] . W[N][K]^T, bf16 operands, fp32 or bf16 result.
//
//   nvcc -O3 -arch=sm_89 -o lt lt.cu -lcublas -lcublasLt && ./lt N K R [bf16out]
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e) { printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)
#define CB(x) do { cublasStatus_t st_ = (x); if (st_) { printf("%s:%d cublas %d\n", __FILE__, __LINE__, (int)st_); exit(1); } } while (0)

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

template <class F> static double time_ms(F f, double budget_s) {
    f(); CK(cudaDeviceSynchronize());
    int n = 0; double t0 = now();
    do { f(); CK(cudaDeviceSynchronize()); n++; } while (now() - t0 < budget_s);
    return (now() - t0) / n * 1e3;
}

struct Cand { double ms; int id, tile, stages, custom; cublasLtMatmulAlgo_t algo; size_t ws; };

int main(int argc, char** argv) {
    const int N = atoi(argv[1]), K = atoi(argv[2]), R = atoi(argv[3]);
    const bool bf_out = argc > 4;
    const cudaDataType out_t = bf_out ? CUDA_R_16BF : CUDA_R_32F;
    __nv_bfloat16 *W, *X; void *Y, *Y2, *ws;
    const size_t WS = 64u << 20;
    CK(cudaMalloc(&W, (size_t)N * K * 2)); CK(cudaMalloc(&X, (size_t)R * K * 2));
    CK(cudaMalloc(&Y, (size_t)R * N * 4)); CK(cudaMalloc(&Y2, (size_t)R * N * 4)); CK(cudaMalloc(&ws, WS));
    {
        std::vector<__nv_bfloat16> h((size_t)N * K);
        srand(1);
        for (auto& v : h) v = __float2bfloat16((rand() % 2001 - 1000) * 2e-5f);
        CK(cudaMemcpy(W, h.data(), h.size() * 2, cudaMemcpyHostToDevice));
        h.resize((size_t)R * K);
        for (auto& v : h) v = __float2bfloat16((rand() % 2001 - 1000) * 1e-3f);
        CK(cudaMemcpy(X, h.data(), h.size() * 2, cudaMemcpyHostToDevice));
    }
    cublasHandle_t blas; CB(cublasCreate(&blas));
    cublasLtHandle_t lt; CB(cublasLtCreate(&lt));
    const float one = 1, zero = 0;
    const double tflop = 2.0 * N * K * R / 1e12;
    auto ex = [&] { CB(cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, N, R, K, &one, W, CUDA_R_16BF, K, X, CUDA_R_16BF, K, &zero, Y, out_t, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP)); };
    double base = time_ms(ex, 1.0);
    printf("N %d K %d R %d %s out: cublasGemmEx %.3f ms (%.2f TFLOPS)\n", N, K, R, bf_out ? "bf16" : "fp32", base, tflop / base * 1e3);
    std::vector<float> ref((size_t)R * N * (bf_out ? 1 : 2) / 2);
    CK(cudaMemcpy(ref.data(), Y, ref.size() * 4, cudaMemcpyDeviceToHost));

    cublasLtMatmulDesc_t desc; CB(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t ta = CUBLAS_OP_T;
    CB(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof ta));
    cublasLtMatrixLayout_t la, lb, ld;
    CB(cublasLtMatrixLayoutCreate(&la, CUDA_R_16BF, K, N, K));
    CB(cublasLtMatrixLayoutCreate(&lb, CUDA_R_16BF, K, R, K));
    CB(cublasLtMatrixLayoutCreate(&ld, out_t, N, R, N));

    std::vector<Cand> cands;
    auto try_algo = [&](cublasLtMatmulAlgo_t algo, int id, int tile, int stages, int custom, double budget) {
        cublasLtMatmulHeuristicResult_t res;
        if (cublasLtMatmulAlgoCheck(lt, desc, la, lb, ld, ld, &algo, &res) != CUBLAS_STATUS_SUCCESS || res.state != CUBLAS_STATUS_SUCCESS || res.workspaceSize > WS) return;
        bool ok = true;
        auto run = [&] { if (cublasLtMatmul(lt, desc, &one, W, la, X, lb, &zero, Y2, ld, Y2, ld, &algo, ws, WS, 0) != CUBLAS_STATUS_SUCCESS) ok = false; };
        run(); if (!ok) return;
        double ms = time_ms(run, budget);
        cands.push_back({ms, id, tile, stages, custom, algo, res.workspaceSize});
    };
    // What the heuristic picks.
    {
        cublasLtMatmulPreference_t pref; CB(cublasLtMatmulPreferenceCreate(&pref));
        CB(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &WS, sizeof WS));
        cublasLtMatmulHeuristicResult_t hr[8]; int n = 0;
        CB(cublasLtMatmulAlgoGetHeuristic(lt, desc, la, lb, ld, ld, pref, 8, hr, &n));
        for (int i = 0; i < n; i++) {
            int id, tile, stages, custom;
            cublasLtMatmulAlgoConfigGetAttribute(&hr[i].algo, CUBLASLT_ALGO_CONFIG_ID, &id, sizeof id, nullptr);
            cublasLtMatmulAlgoConfigGetAttribute(&hr[i].algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof tile, nullptr);
            cublasLtMatmulAlgoConfigGetAttribute(&hr[i].algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stages, sizeof stages, nullptr);
            cublasLtMatmulAlgoConfigGetAttribute(&hr[i].algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &custom, sizeof custom, nullptr);
            size_t before = cands.size();
            try_algo(hr[i].algo, id, tile, stages, custom, 0.5);
            if (cands.size() > before) printf("  heuristic %d: algo %d tile %d stages %d custom %d: %.3f ms\n", i, id, tile, stages, custom, cands.back().ms);
        }
    }
    cands.clear();
    int ids[64], n_ids = 0;
    CB(cublasLtMatmulAlgoGetIds(lt, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16BF, CUDA_R_16BF, out_t, out_t, 64, ids, &n_ids));
    for (int a = 0; a < n_ids; a++) {
        cublasLtMatmulAlgo_t algo;
        if (cublasLtMatmulAlgoInit(lt, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16BF, CUDA_R_16BF, out_t, out_t, ids[a], &algo)) continue;
        size_t written = 0;
        int tiles[CUBLASLT_MATMUL_TILE_END], stages[CUBLASLT_MATMUL_STAGES_END], custom_max = 0;
        cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_TILE_IDS, tiles, sizeof tiles, &written);
        int nt = (int)(written / sizeof(int)); if (nt == 0) { tiles[0] = CUBLASLT_MATMUL_TILE_UNDEFINED; nt = 1; }
        cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_STAGES_IDS, stages, sizeof stages, &written);
        int ns = (int)(written / sizeof(int)); if (ns == 0) { stages[0] = CUBLASLT_MATMUL_STAGES_UNDEFINED; ns = 1; }
        cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX, &custom_max, sizeof custom_max, &written);
        for (int t = 0; t < nt; t++)
            for (int s = 0; s < ns; s++)
                for (int c = 0; c <= custom_max; c++) {
                    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tiles[t], sizeof(int));
                    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stages[s], sizeof(int));
                    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &c, sizeof c);
                    try_algo(algo, ids[a], tiles[t], stages[s], c, 0.05);
                }
    }
    std::sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b) { return a.ms < b.ms; });
    printf("  %zu configurations ran; best by a short run, timed again for 1 s each, alternating with cublasGemmEx:\n", cands.size());
    std::vector<float> got(ref.size());
    double best_ms = 1e9, best_base = 0; const Cand* best = nullptr;
    for (size_t i = 0; i < cands.size() && i < 8; i++) {
        Cand& c = cands[i];
        auto run = [&] { CB(cublasLtMatmul(lt, desc, &one, W, la, X, lb, &zero, Y2, ld, Y2, ld, &c.algo, ws, WS, 0)); };
        double ms = time_ms(run, 1.0), b = time_ms(ex, 1.0);
        CK(cudaMemcpy(got.data(), Y2, got.size() * 4, cudaMemcpyDeviceToHost));
        size_t diff = 0;
        for (size_t j = 0; j < got.size(); j++) diff += memcmp(&got[j], &ref[j], 4) != 0;
        printf("    algo %2d tile %2d stages %2d custom %2d: %.3f ms (%.2f TFLOPS), GemmEx now %.3f ms; workspace %zu, %zu of %zu words differ\n",
               c.id, c.tile, c.stages, c.custom, ms, tflop / ms * 1e3, b, c.ws, diff, got.size());
        if (diff == 0 && c.ws == 0 && c.custom == 0 && ms < best_ms) best_ms = ms, best_base = b, best = &c;
    }
    // The line for core/engine.cu's table: only configurations whose result has the same bits as cublasGemmEx's.
    if (best && best_ms < 0.98 * best_base)
        printf("TABLE    {%d, %d, %d, %d, %d, %d},   // %.3f ms, cublasGemmEx %.3f ms\n", N, K, R, best->id, best->tile, best->stages, best_ms, best_base);
    else
        printf("TABLE    // N %d K %d R %d: nothing 2%% faster than cublasGemmEx with the same bits\n", N, K, R);
}
