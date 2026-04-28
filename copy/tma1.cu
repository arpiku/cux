// file: tma_2d_copy.cu
// Build:  nvcc -arch=sm_90a -std=c++17 -O3 tma_2d_copy.cu -o tma_2d_copy
// Run  :  ./tma_2d_copy
//
// Note: -arch=sm_90a (NOT sm_90) is required for TMA PTX instructions on H100.
//       The 'a' means "architecture-specific" features.

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda/barrier>
#include <cudaTypedefs.h>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <vector>

#define CUDA_CHECK(x) do { \
    cudaError_t e = (x); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(e)); std::exit(1);} } while(0)

#define CU_CHECK(x) do { \
    CUresult r = (x); \
    if (r != CUDA_SUCCESS) { \
        const char* s; cuGetErrorString(r, &s); \
        fprintf(stderr, "CU error %s:%d: %s\n", __FILE__, __LINE__, s); \
        std::exit(1);} } while(0)

using barrier_t = cuda::barrier<cuda::thread_scope_block>;

// --------- Tile size ---------
constexpr int TILE_M = 64;
constexpr int TILE_N = 64;

// --------- PTX inline wrappers ---------
// Trick to memorize: TMA load uses .global.shared and needs an mbarrier.
// TMA store uses .shared.global and uses bulk_group commit/wait.

__device__ __forceinline__
void tma_load_2d(void* smem_dst, const CUtensorMap* tmap,
                 int x, int y, barrier_t& bar) {
    uint64_t bar_addr = __cvta_generic_to_shared(
        cuda::device::barrier_native_handle(bar));
    uint32_t smem_int = __cvta_generic_to_shared(smem_dst);
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(smem_int), "l"(tmap), "r"(x), "r"(y), "l"(bar_addr)
        : "memory");
}

__device__ __forceinline__
void tma_store_2d(const void* smem_src, const CUtensorMap* tmap,
                  int x, int y) {
    uint32_t smem_int = __cvta_generic_to_shared(smem_src);
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group"
        " [%0, {%1, %2}], [%3];"
        :: "l"(tmap), "r"(x), "r"(y), "r"(smem_int)
        : "memory");
}

__device__ __forceinline__ void tma_store_commit() {
    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}

// Wait until at most N pending store groups remain.
template<int N>
__device__ __forceinline__ void tma_store_wait() {
    asm volatile("cp.async.bulk.wait_group %0;" :: "n"(N) : "memory");
}

__device__ __forceinline__ void fence_proxy_async() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

// --------- The kernel ---------
__global__ void tma_2d_copy_kernel(
    const __grid_constant__ CUtensorMap tmap_src,
    const __grid_constant__ CUtensorMap tmap_dst,
    int M, int N)
{
    // 128-byte aligned smem — TMA hardware requirement.
    __shared__ alignas(128) float smem_tile[TILE_M * TILE_N];
    __shared__ barrier_t bar;

    int tid  = threadIdx.x;
    int tile_x = blockIdx.x * TILE_N;
    int tile_y = blockIdx.y * TILE_M;

    // ---- Init mbarrier (1 thread will arrive on the load) ----
    if (tid == 0) {
        init(&bar, 1);
        fence_proxy_async();   // make barrier visible to async proxy
    }
    __syncthreads();

    // ---- ISSUE: one thread launches the bulk copy ----
    barrier_t::arrival_token tok;
    if (tid == 0) {
        tma_load_2d(smem_tile, &tmap_src, tile_x, tile_y, bar);
        // Tell the barrier how many bytes we expect to land.
        tok = cuda::device::barrier_arrive_tx(
            bar, /*arrive_count=*/1,
            /*tx_count_bytes=*/sizeof(float) * TILE_M * TILE_N);
    } else {
        tok = bar.arrive();
    }

    // ---- WAIT: every thread blocks until data lands ----
    bar.wait(std::move(tok));

    // ---- (Optional) compute on the tile ----
    // For demonstration, just multiply by 2.0f in-place in smem.
    #pragma unroll
    for (int i = tid; i < TILE_M * TILE_N; i += blockDim.x) {
        smem_tile[i] = smem_tile[i] * 2.0f;
    }
    __syncthreads();

    // Before TMA store can read smem, fence.proxy.async ensures the
    // generic-proxy writes (the *= 2 above) are visible to the async proxy.
    if (tid == 0) {
        fence_proxy_async();
        tma_store_2d(smem_tile, &tmap_dst, tile_x, tile_y);
        tma_store_commit();
        tma_store_wait<0>();
    }
    // No __syncthreads() needed at the end; the kernel is done.
}

// --------- Host: build the descriptor ---------
CUtensorMap make_2d_tensormap(float* gmem, int M, int N) {
    CUtensorMap tmap{};
    // ATTENTION: innermost dimension first!
    cuuint64_t size[2]   = { (cuuint64_t)N, (cuuint64_t)M };
    // Stride array has rank-1 entries; innermost stride is implicit (= elem size).
    cuuint64_t stride[1] = { (cuuint64_t)N * sizeof(float) };
    cuuint32_t box[2]    = { TILE_N, TILE_M };
    cuuint32_t es[2]     = { 1, 1 };

    CU_CHECK(cuTensorMapEncodeTiled(
        &tmap,
        CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
        2, gmem, size, stride, box, es,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,           // start without swizzling
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE     // OOB reads -> 0
    ));
    return tmap;
}

// --------- Driver ---------
int main() {
    // Realistic problem: a 4096x4096 tile of an image / matrix
    const int M = 4096, N = 4096;
    const size_t bytes = (size_t)M * N * sizeof(float);

    std::vector<float> hA(M * N), hB(M * N, 0.0f);
    for (int i = 0; i < M * N; ++i) hA[i] = (float)(i % 1024) * 0.001f;

    float *dA, *dB;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dB, 0, bytes));

    CUtensorMap tmap_src = make_2d_tensormap(dA, M, N);
    CUtensorMap tmap_dst = make_2d_tensormap(dB, M, N);

    dim3 grid(N / TILE_N, M / TILE_M);
    dim3 block(128);  // 128 threads is plenty for this trivial copy

    // Time it
    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    tma_2d_copy_kernel<<<grid, block>>>(tmap_src, tmap_dst, M, N);
    cudaEventRecord(e);
    CUDA_CHECK(cudaEventSynchronize(e));
    float ms = 0; cudaEventElapsedTime(&ms, s, e);

    CUDA_CHECK(cudaMemcpy(hB.data(), dB, bytes, cudaMemcpyDeviceToHost));

    // Verify
    int errs = 0;
    for (int i = 0; i < M * N && errs < 5; ++i) {
        float expect = hA[i] * 2.0f;
        if (fabsf(hB[i] - expect) > 1e-5f) {
            fprintf(stderr, "Mismatch @ %d: got %f expected %f\n",
                    i, hB[i], expect); errs++;
        }
    }
    printf("Errors: %d  Time: %.3f ms  Bandwidth(R+W): %.1f GB/s\n",
           errs, ms, (2.0 * bytes / 1e9) / (ms / 1e3));

    cudaFree(dA); cudaFree(dB);
    return 0;
}
