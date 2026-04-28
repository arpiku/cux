// file: tma_wgmma_gemm.cu
// Build: nvcc -arch=sm_90a -std=c++17 -O3 -lcuda --expt-relaxed-constexpr tma_wgmma_gemm.cu -o tma_wgmma_gemm
// Run  : ./tma_wgmma_gemm
//
// Note: -arch=sm_90a is required even on Blackwell for wgmma instructions.
// On RTX 5070 (CC 12.0) the binary will run via PTX JIT.

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

// ===== Tile sizes =====
// BM x BN per CTA, BK per pipeline stage.
// One consumer warpgroup (128 threads) handles the BMxBN tile.
constexpr int BM     = 128;
constexpr int BN     = 128;
constexpr int BK     = 64;
constexpr int STAGES = 3;     // pipeline depth

// One producer warp (warp 0) + one consumer warpgroup (warps 4-7).
// We use warps 4-7 (not 0-3) so warp 0 is naturally separate.
constexpr int PROD_THREADS = 32;
constexpr int CONS_THREADS = 128;
constexpr int THREADS      = PROD_THREADS + CONS_THREADS;  // 160 total
constexpr int CONS_WG_OFFSET = 32;  // consumer warpgroup starts at thread 32

// Per-warpgroup wgmma covers M=64, N=128, K=16.
// Our BMxBN = 128x128, so we issue 2 wgmmas along M, 1 along N.
constexpr int WG_M = 64;
constexpr int WG_N = 128;
constexpr int WG_K = 16;
constexpr int M_ITERS = BM / WG_M;   // 2
constexpr int N_ITERS = BN / WG_N;   // 1
constexpr int K_ITERS = BK / WG_K;   // 4

// ===== TMA wrappers (same as before) =====
__device__ __forceinline__
void tma_load_2d(void* smem, const CUtensorMap* tmap, int x, int y, barrier_t& bar) {
    uint64_t bar_addr = __cvta_generic_to_shared(cuda::device::barrier_native_handle(bar));
    uint32_t s = __cvta_generic_to_shared(smem);
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global"
        ".mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
        :: "r"(s), "l"(tmap), "r"(x), "r"(y), "l"(bar_addr) : "memory");
}
__device__ __forceinline__
void tma_store_2d(const void* smem, const CUtensorMap* tmap, int x, int y) {
    uint32_t s = __cvta_generic_to_shared(smem);
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];"
        :: "l"(tmap), "r"(x), "r"(y), "r"(s) : "memory");
}
__device__ __forceinline__ void tma_commit() { asm volatile("cp.async.bulk.commit_group;" ::: "memory"); }
template<int N> __device__ __forceinline__ void tma_wait() { asm volatile("cp.async.bulk.wait_group %0;" :: "n"(N) : "memory"); }
__device__ __forceinline__ void fence_async() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }

// ===== WGMMA wrappers =====
// Build a 64-bit smem descriptor for wgmma operands.
// Layout: bits[ 0:13] = smem_addr >> 4
//         bits[16:29] = leading byte offset (LBO) >> 4   — stride to next K-block
//         bits[32:45] = stride byte offset    (SBO) >> 4 — stride between contiguous "core matrices"
//         bits[62:63] = swizzle (0=none, 1=128B, 2=64B, 3=32B)
//
// For SWIZZLE_128B with FP16 and BK=64:
//   - "core matrix" is 8x8 elements = 8 rows of 16 bytes.
//   - LBO = 1024 bytes (16 in 16-byte units): distance to next 8-row block in same column
//   - SBO = stride between contiguous core matrices
__device__ __forceinline__
uint64_t make_smem_desc_A(half* smem_ptr) {
    uint64_t addr = __cvta_generic_to_shared(smem_ptr);
    uint64_t desc = 0;
    desc |= (addr & 0x3FFFFULL) >> 4;            // smem offset / 16
    desc |= ((uint64_t)(BK * sizeof(half)) >> 4) << 16;   // LBO: bytes per row / 16
    desc |= ((uint64_t)(8 * BK * sizeof(half)) >> 4) << 32; // SBO: 8-row block stride / 16
    desc |= ((uint64_t)1) << 62;                  // swizzle = 128B
    return desc;
}
__device__ __forceinline__
uint64_t make_smem_desc_B(half* smem_ptr) {
    uint64_t addr = __cvta_generic_to_shared(smem_ptr);
    uint64_t desc = 0;
    desc |= (addr & 0x3FFFFULL) >> 4;
    desc |= ((uint64_t)(BN * sizeof(half)) >> 4) << 16;
    desc |= ((uint64_t)(8 * BN * sizeof(half)) >> 4) << 32;
    desc |= ((uint64_t)1) << 62;
    return desc;
}

__device__ __forceinline__ void wgmma_fence()      { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wgmma_commit()     { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template<int N> __device__ __forceinline__ void wgmma_wait() {
    asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(N) : "memory");
}

// One 64x128x16 wgmma, accumulating into d[64].
__device__ __forceinline__
void wgmma_m64n128k16(float d[64], uint64_t desc_a, uint64_t desc_b) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, "
        " %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, "
        " %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, "
        "%64, %65, 1, 1, 1, 0, 0;"
        : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),
          "+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),
          "+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),
          "+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31]),
          "+f"(d[32]),"+f"(d[33]),"+f"(d[34]),"+f"(d[35]),"+f"(d[36]),"+f"(d[37]),"+f"(d[38]),"+f"(d[39]),
          "+f"(d[40]),"+f"(d[41]),"+f"(d[42]),"+f"(d[43]),"+f"(d[44]),"+f"(d[45]),"+f"(d[46]),"+f"(d[47]),
          "+f"(d[48]),"+f"(d[49]),"+f"(d[50]),"+f"(d[51]),"+f"(d[52]),"+f"(d[53]),"+f"(d[54]),"+f"(d[55]),
          "+f"(d[56]),"+f"(d[57]),"+f"(d[58]),"+f"(d[59]),"+f"(d[60]),"+f"(d[61]),"+f"(d[62]),"+f"(d[63])
        : "l"(desc_a), "l"(desc_b));
}

// ===== The kernel =====
__global__ void tma_wgmma_kernel(
    const __grid_constant__ CUtensorMap tmap_A,
    const __grid_constant__ CUtensorMap tmap_B,
    const __grid_constant__ CUtensorMap tmap_C,
    int M, int N, int K)
{
    // Multi-stage smem buffers, one per pipeline stage.
    __shared__ alignas(128) half smem_A[STAGES][BM * BK];
    __shared__ alignas(128) half smem_B[STAGES][BK * BN];
    __shared__ alignas(128) half smem_C[BM * BN];

    // Two barriers PER stage:
    //   bar_full[s]  : producer arrives when load complete; consumer waits for it.
    //   bar_empty[s] : consumer arrives when buffer is free; producer waits for it.
    __shared__ barrier_t bar_full[STAGES];
    __shared__ barrier_t bar_empty[STAGES];

    int tid = threadIdx.x;

    // ----- Init barriers (thread 0 of CTA) -----
    if (tid == 0) {
        for (int s = 0; s < STAGES; ++s) {
            init(&bar_full[s],  1);                 // producer's single arrive_tx
            init(&bar_empty[s], CONS_THREADS);      // every consumer thread arrives
        }
        fence_async();
    }
    __syncthreads();

    int n_block = blockIdx.x;
    int m_block = blockIdx.y;
    int num_k_iters = K / BK;
    constexpr size_t bytes_per_stage = sizeof(half) * (BM * BK + BK * BN);

    // =========================================================
    // PRODUCER WARP: tid in [0, 32)
    // =========================================================
    if (tid < PROD_THREADS) {
        // Only thread 0 of the producer warp issues TMA.
        if (tid == 0) {
            for (int k_iter = 0; k_iter < num_k_iters; ++k_iter) {
                int stage = k_iter % STAGES;
                int phase = (k_iter / STAGES) & 1;

                // After the first STAGES iters, wait for consumer to free this slot.
                if (k_iter >= STAGES) {
                    bar_empty[stage].wait_parity(phase ^ 1);
                    // ^1 because empty barrier flips on the *previous* trip through this stage
                }

                // Issue the two TMA loads into this stage's smem.
                tma_load_2d(smem_A[stage], &tmap_A, k_iter * BK, m_block * BM, bar_full[stage]);
                tma_load_2d(smem_B[stage], &tmap_B, n_block * BN, k_iter * BK, bar_full[stage]);
                (void)cuda::device::barrier_arrive_tx(bar_full[stage], 1, bytes_per_stage);
            }
        }
    }
    // =========================================================
    // CONSUMER WARPGROUP: tid in [32, 160)
    // =========================================================
    else {
        // Per-thread accumulator for the M_ITERS x N_ITERS sub-tiles.
        // Each sub-tile is 64 floats per thread (m64n128).
        float c_reg[M_ITERS][64] = {};

        for (int k_iter = 0; k_iter < num_k_iters; ++k_iter) {
            int stage = k_iter % STAGES;
            int phase = (k_iter / STAGES) & 1;

            // Wait for producer to fill this stage.
            bar_full[stage].wait_parity(phase);

            // ----- WGMMA: loop over K in chunks of WG_K=16 -----
            // For each (M sub-tile m, K-chunk kk) we issue one wgmma.
            wgmma_fence();
            #pragma unroll
            for (int m = 0; m < M_ITERS; ++m) {
                #pragma unroll
                for (int kk = 0; kk < K_ITERS; ++kk) {
                    half* a_ptr = &smem_A[stage][m * WG_M * BK + kk * WG_K];
                    half* b_ptr = &smem_B[stage][kk * WG_K * BN];
                    uint64_t desc_a = make_smem_desc_A(a_ptr);
                    uint64_t desc_b = make_smem_desc_B(b_ptr);
                    wgmma_m64n128k16(c_reg[m], desc_a, desc_b);
                }
            }
            wgmma_commit();
            // Wait for THIS stage's wgmmas to complete before signaling empty.
            wgmma_wait<0>();

            // Signal that this stage's smem buffers are free.
            (void)bar_empty[stage].arrive();
        }

        // ===== Epilogue: write c_reg -> smem_C -> gmem via TMA =====
        // The wgmma output layout per warpgroup for m64n128 is well-defined:
        // each thread holds 64 FP32s arranged as 16 cols of FP32 pairs across 8 row-groups.
        // This is the canonical "C fragment" layout.
        // We write per-thread regs into smem_C in the matching pattern.
        int wg_tid = tid - CONS_WG_OFFSET;        // 0..127
        int warp_id = wg_tid / 32;                // 0..3
        int lane    = wg_tid % 32;                // 0..31

        #pragma unroll
        for (int m = 0; m < M_ITERS; ++m) {
            // Each warp owns 16 rows of the 64-row sub-tile.
            // Within the warp, row groups of 8, with thread layout (lane/4, lane%4 * 2).
            int row_base = m * WG_M + warp_id * 16 + (lane / 4);
            int col_base = (lane % 4) * 2;
            #pragma unroll
            for (int rgroup = 0; rgroup < 2; ++rgroup) {       // 8 + 8 rows per warp
                int row = row_base + rgroup * 8;
                #pragma unroll
                for (int col_chunk = 0; col_chunk < 16; ++col_chunk) {
                    int col = col_base + col_chunk * 8;
                    int reg_idx = rgroup * 32 + col_chunk * 2;
                    smem_C[row * BN + col + 0] = __float2half(c_reg[m][reg_idx + 0]);
                    smem_C[row * BN + col + 1] = __float2half(c_reg[m][reg_idx + 1]);
                }
            }
        }
        __syncthreads();   // sync only the consumer warpgroup against itself

        // Thread 32 (first consumer thread) issues the C store.
        if (tid == CONS_WG_OFFSET) {
            fence_async();
            tma_store_2d(smem_C, &tmap_C, n_block * BN, m_block * BM);
            tma_commit();
            tma_wait<0>();
        }
    }
}

// ===== Host =====
CUtensorMap make_tmap_2d_half(half* gmem, int rows, int cols, int box_rows, int box_cols,
                              CUtensorMapSwizzle sw = CU_TENSOR_MAP_SWIZZLE_128B) {
    CUtensorMap tmap{};
    cuuint64_t size[2]   = { (cuuint64_t)cols, (cuuint64_t)rows };
    cuuint64_t stride[1] = { (cuuint64_t)cols * sizeof(half) };
    cuuint32_t box[2]    = { (cuuint32_t)box_cols, (cuuint32_t)box_rows };
    cuuint32_t es[2]     = { 1, 1 };
    CU_CHECK(cuTensorMapEncodeTiled(
        &tmap, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2,
        gmem, size, stride, box, es,
        CU_TENSOR_MAP_INTERLEAVE_NONE, sw,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    return tmap;
}

void gemm_ref(const std::vector<float>& A, const std::vector<float>& B,
              std::vector<float>& C, int M, int N, int K) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float acc = 0;
            for (int k = 0; k < K; ++k) acc += A[i*K+k] * B[k*N+j];
            C[i*N+j] = acc;
        }
}

int main() {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s  CC: %d.%d\n", prop.name, prop.major, prop.minor);

    const int M = 1024, N = 1024, K = 1024;

    std::vector<float> hA_f((size_t)M*K), hB_f((size_t)K*N), hC_ref((size_t)M*N), hC_got((size_t)M*N);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : hA_f) x = dist(rng);
    for (auto& x : hB_f) x = dist(rng);
    std::vector<half> hA((size_t)M*K), hB((size_t)K*N), hC((size_t)M*N);
    for (size_t i = 0; i < hA.size(); ++i) hA[i] = __float2half(hA_f[i]);
    for (size_t i = 0; i < hB.size(); ++i) hB[i] = __float2half(hB_f[i]);

    half *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, sizeof(half)*M*K));
    CUDA_CHECK(cudaMalloc(&dB, sizeof(half)*K*N));
    CUDA_CHECK(cudaMalloc(&dC, sizeof(half)*M*N));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), sizeof(half)*M*K, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), sizeof(half)*K*N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dC, 0, sizeof(half)*M*N));

    CUtensorMap tmap_A = make_tmap_2d_half(dA, M, K, BM, BK);
    CUtensorMap tmap_B = make_tmap_2d_half(dB, K, N, BK, BN);
    CUtensorMap tmap_C = make_tmap_2d_half(dC, M, N, BM, BN);

    dim3 grid(N / BN, M / BM);
    dim3 block(THREADS);

    tma_wgmma_kernel<<<grid, block>>>(tmap_A, tmap_B, tmap_C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(hC.data(), dC, sizeof(half)*M*N, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < hC.size(); ++i) hC_got[i] = __half2float(hC[i]);
    gemm_ref(hA_f, hB_f, hC_ref, M, N, K);

    int errs = 0; double max_abs = 0;
    for (size_t i = 0; i < hC_ref.size(); ++i) {
        double ae = std::fabs((double)hC_got[i] - hC_ref[i]);
        if (ae > max_abs) max_abs = ae;
        if (ae > 2.0 && errs < 5) {  // FP16 GEMM tolerance is generous
            fprintf(stderr, "[%zu] got=%.3f ref=%.3f abs=%.3f\n", i, hC_got[i], hC_ref[i], ae);
            errs++;
        }
    }
    printf("Correctness 1024^3: %s (errs=%d, max_abs=%.3f)\n",
           errs == 0 ? "PASS" : "FAIL", errs, max_abs);

    // Performance run
    const int Mp = 4096, Np = 4096, Kp = 4096;
    half *pA, *pB, *pC;
    CUDA_CHECK(cudaMalloc(&pA, sizeof(half)*Mp*Kp));
    CUDA_CHECK(cudaMalloc(&pB, sizeof(half)*Kp*Np));
    CUDA_CHECK(cudaMalloc(&pC, sizeof(half)*Mp*Np));
    CUDA_CHECK(cudaMemset(pA, 0, sizeof(half)*Mp*Kp));
    CUDA_CHECK(cudaMemset(pB, 0, sizeof(half)*Kp*Np));
    CUtensorMap pA_map = make_tmap_2d_half(pA, Mp, Kp, BM, BK);
    CUtensorMap pB_map = make_tmap_2d_half(pB, Kp, Np, BK, BN);
    CUtensorMap pC_map = make_tmap_2d_half(pC, Mp, Np, BM, BN);

    dim3 g(Np / BN, Mp / BM);
    tma_wgmma_kernel<<<g, block>>>(pA_map, pB_map, pC_map, Mp, Np, Kp);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    const int IT = 10;
    for (int i = 0; i < IT; ++i)
        tma_wgmma_kernel<<<g, block>>>(pA_map, pB_map, pC_map, Mp, Np, Kp);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms = 0; cudaEventElapsedTime(&ms, s, e); ms /= IT;

    double tflops = 2.0 * Mp * Np * Kp / 1e12 / (ms / 1e3);
    printf("Perf 4096^3 wgmma+TMA pipelined: %.2f ms  %.1f TFLOPS\n", ms, tflops);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    cudaFree(pA); cudaFree(pB); cudaFree(pC);
    return 0;
}
