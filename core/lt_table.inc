    // {N, K, rows, algorithm, tile, stages} on the RTX 4060 Laptop (30 W), cuBLAS 12.
    // 1024 rows (a full chunk): spikes/gemm/lt.cu puts algorithm 21 / tile 24 and algorithm 6 / tile 23
    // within noise of each other for the smaller shapes; timed inside the engine (bench, 4096 tokens)
    // 21 / 24 / 9 for everything is the fastest: 818 ms against 838 for the default and for 6 / 23 / 0.
    {2240, 1536, 1024, 21, 24, 9},   // a slice of gate/up
    {2048, 1536, 1024, 21, 24, 9},   // q/k/v
    {1536, 1536, 1024, 21, 24, 9},   // o
    {1536, 8960, 1024, 21, 24, 9},   // down: 1.574 ms, cublasGemmEx 1.662 ms
    // 768 to 895 rows: what lt.cu found more than 3% ahead of cublasGemmEx.
    {2048, 1536, 768, 21, 23, 9},    // 0.283 ms, cublasGemmEx 0.304 ms
    {1536, 8960, 768, 21, 20, 12},   // 1.216 ms, cublasGemmEx 1.284 ms
