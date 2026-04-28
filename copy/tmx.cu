// file: tma1.cu  (2D TMA load/store, fixed v2)
// Build: nvcc -arch=sm_90a -std=c++17 -O3 -lcuda tma1.cu -o tma1
// Run  : ./tma1
//        compute-sanitizer ./tma1   (should be silent)

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda/barrier>
#include <cudaTypedefs.h>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <vector>

#pragma nv_diag_suppress static_var_with_dynamic_init

#define CUDA_CHECK(x) do { \
    cudaError_t _err = (x); \
    if (_err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(_err)); std::exit(1);} } while(0)

#define CU_CHECK(x) do { \
    CUresult _r = (x); \
    if (_r != CUDA_SUCCESS) { \
        const char* _s; cuGetErrorString(_r, &_s); \
        fprintf(stderr, "CU error %s:%d: %s\n", __FILE__, __LINE__, _s); \
        std::exit(1);} } while(0)

using barrier_t = cuda::barrier<cuda::thread_scope_block>;

constexpr int TILE_M = 64;
constexpr int TILE_N = 64;

// --------- PTX inline wrappers ---------
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

template<int N>
__device__ __forceinline__ void tma_store_wait() {
    asm volatile("cp.async.bulk.wait_group %0;" :: "n"(N) : "memory");
}

__device__ __forceinline__ void fence_proxy_async() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

// --------- Kernel ---------
__global__ void tma_2d_copy_kernel(
    const __grid_constant__ CUtensorMap tmap_src,
    const __grid_constant__ CUtensorMap tmap_dst,
    int M, int N)
{
    __shared__ alignas(128) float smem_tile[TILE_M * TILE_N];
    __shared__ barrier_t bar;

    int tid  = threadIdx.x;
    int tile_x = blockIdx.x * TILE_N;
    int tile_y = blockIdx.y * TILE_M;

    if (tid == 0) {
        init(&bar, 1);
        fence_proxy_async();
    }
    __syncthreads();

    if (tid == 0) {
        tma_load_2d(smem_tile, &tmap_src, tile_x, tile_y, bar);
        // [[nodiscard]] — cast away the returned arrival token
        (void)cuda::device::barrier_arrive_tx(
            bar, /*arrive_count=*/1,
            /*tx_count_bytes=*/sizeof(float) * TILE_M * TILE_N);
    }

    bar.wait_parity(0);

    for (int i = tid; i < TILE_M * TILE_N; i += blockDim.x) {
        smem_tile[i] = smem_tile[i] * 2.0f;
    }
    __syncthreads();

    if (tid == 0) {
        fence_proxy_async();
        tma_store_2d(smem_tile, &tmap_dst, tile_x, tile_y);
        tma_store_commit();
        tma_store_wait<0>();
    }
}

// --------- Host: build descriptor ---------
CUtensorMap make_2d_tensormap(float* gmem, int M, int N) {
    CUtensorMap tmap{};
    cuuint64_t size[2]   = { (cuuint64_t)N, (cuuint64_t)M };
    cuuint64_t stride[1] = { (cuuint64_t)N * sizeof(float) };
    cuuint32_t box[2]    = { TILE_N, TILE_M };
    cuuint32_t es[2]     = { 1, 1 };

    CU_CHECK(cuTensorMapEncodeTiled(
        &tmap,
        CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
        2, gmem, size, stride, box, es,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    return tmap;
}

// --------- Driver ---------
int main() {
    int dev = 0;
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("Device: %s  CC: %d.%d\n", prop.name, prop.major, prop.minor);
    if (prop.major < 9) {
        fprintf(stderr, "TMA requires CC >= 9.0 (Hopper).\n");
        return 1;
    }

    const int M = 4096, N = 4096;
    const size_t bytes = (size_t)M * N * sizeof(float);

    std::vector<float> hA((size_t)M * N), hB((size_t)M * N, 0.0f);
    for (int i = 0; i < M * N; ++i) hA[i] = (float)(i % 1024) * 0.001f;

    float *dA, *dB;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dB, 0, bytes));

    CUtensorMap tmap_src = make_2d_tensormap(dA, M, N);
    CUtensorMap tmap_dst = make_2d_tensormap(dB, M, N);

    dim3 grid(N / TILE_N, M / TILE_M);
    dim3 block(128);

    // Warmup
    tma_2d_copy_kernel<<<grid, block>>>(tmap_src, tmap_dst, M, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Time (renamed events to avoid macro shadowing)
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    CUDA_CHECK(cudaEventRecord(ev_start));
    const int ITERS = 20;
    for (int i = 0; i < ITERS; ++i)
        tma_2d_copy_kernel<<<grid, block>>>(tmap_src, tmap_dst, M, N);
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
    ms /= ITERS;

    CUDA_CHECK(cudaMemcpy(hB.data(), dB, bytes, cudaMemcpyDeviceToHost));

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

    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    cudaFree(dA); cudaFree(dB);
    return 0;
}
