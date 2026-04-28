// file: tma_gemm.cu
// Build: nvcc -arch=sm_90a -std=c++17 -O3 -lcuda tma_gemm.cu -o tma_gemm
// Run  : ./tma_gemm
//        compute-sanitizer ./tma_gemm   (silent)

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda/barrier>
#include <cudaTypedefs.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>

#pragma nv_diag_suppress static_var_with_dynamic_init

#define CUDA_CHECK(x) do { cudaError_t _e=(x); if(_e){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(_e));std::exit(1);}} while(0)
#define CU_CHECK(x)   do { CUresult _r=(x);   if(_r){const char*_s; cuGetErrorString(_r,&_s); fprintf(stderr,"CU %s:%d %s\n",__FILE__,__LINE__,_s);std::exit(1);}} while(0)

using barrier_t = cuda::barrier<cuda::thread_scope_block>;

// ----- Tile sizes (compile-time) -----
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 32;
constexpr int THREADS = 256;

// Each thread owns BM*BN/THREADS = 64 C elements.
// We'll lay them out as an 8x8 sub-tile per thread, threads in a 16x16 grid.
constexpr int TY = 16;   // threads along M
constexpr int TX = 16;   // threads along N
static_assert(TY * TX == THREADS, "");
constexpr int RM = BM / TY;  // = 8 rows of C per thread
constexpr int RN = BN / TX;  // = 8 cols of C per thread

// ----- PTX wrappers (same as Phase 1) -----
__device__ __forceinline__
void tma_load_2d(void* smem_dst, const CUtensorMap* tmap,
                 int x, int y, barrier_t& bar) {
    uint64_t bar_addr = __cvta_generic_to_shared(
        cuda::device::barrier_native_handle(bar));
    uint32_t s = __cvta_generic_to_shared(smem_dst);
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(s), "l"(tmap), "r"(x), "r"(y), "l"(bar_addr) : "memory");
}

__device__ __forceinline__
void tma_store_2d(const void* smem_src, const CUtensorMap* tmap,
                  int x, int y) {
    uint32_t s = __cvta_generic_to_shared(smem_src);
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group"
        " [%0, {%1, %2}], [%3];"
        :: "l"(tmap), "r"(x), "r"(y), "r"(s) : "memory");
}

__device__ __forceinline__ void tma_commit() {
    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}
template<int N> __device__ __forceinline__ void tma_wait() {
    asm volatile("cp.async.bulk.wait_group %0;" :: "n"(N) : "memory");
}
__device__ __forceinline__ void fence_async() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

// ===========================================================
// Kernel
//   A is row-major [M,K] half
//   B is row-major [K,N] half
//   C is row-major [M,N] float (we'll write half via a small cast at the end)
// ===========================================================
__global__ void tma_gemm_kernel(
    const __grid_constant__ CUtensorMap tmap_A,
    const __grid_constant__ CUtensorMap tmap_B,
    const __grid_constant__ CUtensorMap tmap_C,
    int M, int N, int K)
{
    // ----- Shared memory -----
    __shared__ alignas(128) half  smem_A[BM * BK];
    __shared__ alignas(128) half  smem_B[BK * BN];
    __shared__ alignas(128) half  smem_C[BM * BN];   // half output tile for TMA store
    __shared__ barrier_t bar;

    int tid = threadIdx.x;
    int warp = tid / 32;
    int lane = tid % 32;
    (void)warp; (void)lane;  // silence unused warnings, we don't need warp info here

    int n_block = blockIdx.x;
    int m_block = blockIdx.y;

    // 2D coords of this thread inside the CTA
    int ty = tid / TX;   // 0..15
    int tx = tid % TX;   // 0..15

    // ----- Init mbarrier (1 expected arrival per K iter — the issuer) -----
    if (tid == 0) {
        init(&bar, 1);
        fence_async();
    }
    __syncthreads();

    // ----- Per-thread C accumulator (RM x RN = 8x8 = 64 floats) -----
    float c_reg[RM][RN];
    #pragma unroll
    for (int i = 0; i < RM; ++i)
        #pragma unroll
        for (int j = 0; j < RN; ++j)
            c_reg[i][j] = 0.0f;

    // ----- K-LOOP -----
    int num_k_iters = K / BK;
    constexpr size_t bytes_per_iter =
        sizeof(half) * (BM * BK + BK * BN);

    for (int k_iter = 0; k_iter < num_k_iters; ++k_iter) {
        // ISSUE both loads using ONE barrier with combined transaction count.
        if (tid == 0) {
            // A tile coords: (k * BK, m_block * BM)  — innermost is K axis
            tma_load_2d(smem_A, &tmap_A, k_iter * BK, m_block * BM, bar);
            // B tile coords: (n_block * BN, k * BK)  — innermost is N axis
            tma_load_2d(smem_B, &tmap_B, n_block * BN, k_iter * BK, bar);
            (void)cuda::device::barrier_arrive_tx(bar, 1, bytes_per_iter);
        }

        // WAIT on phase = k_iter & 1
        bar.wait_parity(k_iter & 1);

        // ----- COMPUTE: per-thread inner product over BK -----
        // Each thread computes its 8x8 block of the C tile.
        // smem_A is [BM][BK] row-major  → element (i, k) at smem_A[i*BK + k]
        // smem_B is [BK][BN] row-major  → element (k, j) at smem_B[k*BN + j]
        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            // Cache the A column slice and B row slice for this kk
            float a_frag[RM];
            float b_frag[RN];
            #pragma unroll
            for (int i = 0; i < RM; ++i)
                a_frag[i] = __half2float(smem_A[(ty * RM + i) * BK + kk]);
            #pragma unroll
            for (int j = 0; j < RN; ++j)
                b_frag[j] = __half2float(smem_B[kk * BN + tx * RN + j]);

            #pragma unroll
            for (int i = 0; i < RM; ++i)
                #pragma unroll
                for (int j = 0; j < RN; ++j)
                    c_reg[i][j] += a_frag[i] * b_frag[j];
        }

        // No __syncthreads() needed before next iter: bar.wait_parity already
        // ordered all consumers; the next arrive_tx writes to the same smem
        // regions but only AFTER all threads have left the wait, which is
        // guaranteed by the barrier semantics.
        // BUT: we need a syncthreads to make sure all threads finished
        // *reading* smem before the next load overwrites it. Without it,
        // a fast thread could race ahead while a slow thread is still reading.
        __syncthreads();
    }

    // ----- WRITE C: registers -> smem -> gmem (via TMA) -----
    #pragma unroll
    for (int i = 0; i < RM; ++i) {
        int row = ty * RM + i;
        #pragma unroll
        for (int j = 0; j < RN; ++j) {
            int col = tx * RN + j;
            smem_C[row * BN + col] = __float2half(c_reg[i][j]);
        }
    }
    __syncthreads();

    if (tid == 0) {
        fence_async();
        tma_store_2d(smem_C, &tmap_C, n_block * BN, m_block * BM);
        tma_commit();
        tma_wait<0>();
    }
}

// ===========================================================
// Host-side descriptor helpers
// ===========================================================
CUtensorMap make_tmap_2d_half(half* gmem, int rows, int cols, int box_rows, int box_cols) {
    CUtensorMap tmap{};
    cuuint64_t size[2]   = { (cuuint64_t)cols, (cuuint64_t)rows };
    cuuint64_t stride[1] = { (cuuint64_t)cols * sizeof(half) };
    cuuint32_t box[2]    = { (cuuint32_t)box_cols, (cuuint32_t)box_rows };
    cuuint32_t es[2]     = { 1, 1 };
    CU_CHECK(cuTensorMapEncodeTiled(
        &tmap, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2,
        gmem, size, stride, box, es,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    return tmap;
}

// ===========================================================
// Reference (CPU) and main
// ===========================================================
void gemm_ref(const std::vector<float>& A, const std::vector<float>& B,
              std::vector<float>& C, int M, int N, int K) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k)
                acc += A[i*K + k] * B[k*N + j];
            C[i*N + j] = acc;
        }
}

int main() {
    int dev = 0;
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("Device: %s  CC: %d.%d\n", prop.name, prop.major, prop.minor);
    if (prop.major < 9) { fprintf(stderr, "Need Hopper.\n"); return 1; }

    // Use a small-but-real shape for the correctness test, then a big one for perf.
    const int M = 1024, N = 1024, K = 1024;
    static_assert(BM == 128 && BN == 128 && BK == 32, "");
    if (M % BM || N % BN || K % BK) {
        fprintf(stderr, "Shapes must be divisible by tile sizes.\n");
        return 1;
    }

    // ---- Host data (FP32 for reference, cast to FP16 for device) ----
    std::vector<float> hA_f((size_t)M*K), hB_f((size_t)K*N), hC_ref((size_t)M*N), hC_got((size_t)M*N);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : hA_f) x = dist(rng);
    for (auto& x : hB_f) x = dist(rng);

    std::vector<half> hA((size_t)M*K), hB((size_t)K*N), hC((size_t)M*N);
    for (size_t i = 0; i < hA.size(); ++i) hA[i] = __float2half(hA_f[i]);
    for (size_t i = 0; i < hB.size(); ++i) hB[i] = __float2half(hB_f[i]);

    // ---- Device buffers ----
    half *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, sizeof(half)*M*K));
    CUDA_CHECK(cudaMalloc(&dB, sizeof(half)*K*N));
    CUDA_CHECK(cudaMalloc(&dC, sizeof(half)*M*N));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), sizeof(half)*M*K, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), sizeof(half)*K*N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dC, 0, sizeof(half)*M*N));

    // ---- Descriptors ----
    CUtensorMap tmap_A = make_tmap_2d_half(dA, M, K, BM, BK);
    CUtensorMap tmap_B = make_tmap_2d_half(dB, K, N, BK, BN);
    CUtensorMap tmap_C = make_tmap_2d_half(dC, M, N, BM, BN);

    // ---- Launch ----
    dim3 grid(N / BN, M / BM);
    dim3 block(THREADS);
    printf("Shape: %dx%dx%d  Tile: %dx%dx%d  Grid: %dx%d  Threads: %d\n",
           M, N, K, BM, BN, BK, grid.x, grid.y, THREADS);

    // Warmup + correctness
    tma_gemm_kernel<<<grid, block>>>(tmap_A, tmap_B, tmap_C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(hC.data(), dC, sizeof(half)*M*N, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < hC.size(); ++i) hC_got[i] = __half2float(hC[i]);

    gemm_ref(hA_f, hB_f, hC_ref, M, N, K);

    int errs = 0;
    double max_abs = 0, max_rel = 0;
    for (size_t i = 0; i < hC_ref.size(); ++i) {
        double a = hC_got[i], r = hC_ref[i];
        double ae = std::fabs(a - r);
        double re = ae / (std::fabs(r) + 1e-6);
        if (ae > max_abs) max_abs = ae;
        if (re > max_rel) max_rel = re;
        if (re > 5e-2 && ae > 1e-1) {
            if (errs < 5)
                fprintf(stderr, "[%zu] got=%.4f ref=%.4f abs=%.4f rel=%.4f\n",
                        i, a, r, ae, re);
            errs++;
        }
    }
    printf("Correctness: %s (errs=%d, max_abs=%.4f, max_rel=%.4f)\n",
           errs == 0 ? "PASS" : "FAIL", errs, max_abs, max_rel);

    // ---- Performance run on a bigger problem ----
    {
        const int Mp = 4096, Np = 4096, Kp = 4096;
        half *pA, *pB, *pC;
        CUDA_CHECK(cudaMalloc(&pA, sizeof(half)*Mp*Kp));
        CUDA_CHECK(cudaMalloc(&pB, sizeof(half)*Kp*Np));
        CUDA_CHECK(cudaMalloc(&pC, sizeof(half)*Mp*Np));
        CUDA_CHECK(cudaMemset(pA, 1, sizeof(half)*Mp*Kp));
        CUDA_CHECK(cudaMemset(pB, 1, sizeof(half)*Kp*Np));

        CUtensorMap pA_map = make_tmap_2d_half(pA, Mp, Kp, BM, BK);
        CUtensorMap pB_map = make_tmap_2d_half(pB, Kp, Np, BK, BN);
        CUtensorMap pC_map = make_tmap_2d_half(pC, Mp, Np, BM, BN);

        dim3 g(Np / BN, Mp / BM);
        // warmup
        tma_gemm_kernel<<<g, block>>>(pA_map, pB_map, pC_map, Mp, Np, Kp);
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaEvent_t s, e;
        cudaEventCreate(&s); cudaEventCreate(&e);
        cudaEventRecord(s);
        const int IT = 5;
        for (int i = 0; i < IT; ++i)
            tma_gemm_kernel<<<g, block>>>(pA_map, pB_map, pC_map, Mp, Np, Kp);
        cudaEventRecord(e); cudaEventSynchronize(e);
        float ms = 0; cudaEventElapsedTime(&ms, s, e); ms /= IT;

        double tflops = 2.0 * Mp * Np * Kp / 1e12 / (ms / 1e3);
        printf("Perf 4096^3 FP16 (no wgmma): %.2f ms  %.1f TFLOPS\n", ms, tflops);

        cudaFree(pA); cudaFree(pB); cudaFree(pC);
    }

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}
