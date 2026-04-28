// file: tma2_baseline.cu
// Build: nvcc -arch=sm_90a -std=c++17 -O3 -lcuda tma1_baseline.cu -o tma1_baseline
// Run  : ./tma1_baseline

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda/pipeline>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

namespace cg = cooperative_groups;

#define CUDA_CHECK(x) do { \
    cudaError_t _err = (x); \
    if (_err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(_err)); std::exit(1);} } while(0)

constexpr int TILE_M = 64;
constexpr int TILE_N = 64;
constexpr int TILE_ELEMS = TILE_M * TILE_N;

// =====================================================================
// Kernel 1: plain coalesced copy (no smem at all)
//   gmem -> register -> gmem.  This is the absolute simplest baseline.
// =====================================================================
__global__ void copy_plain_kernel(const float* __restrict__ A,
                                  float* __restrict__ B,
                                  int M, int N)
{
    int tile_x = blockIdx.x * TILE_N;
    int tile_y = blockIdx.y * TILE_M;

    // 128 threads cover 64*64 = 4096 elements: 32 elements/thread
    for (int i = threadIdx.x; i < TILE_ELEMS; i += blockDim.x) {
        int local_y = i / TILE_N;
        int local_x = i % TILE_N;
        int gy = tile_y + local_y;
        int gx = tile_x + local_x;
        int gidx = gy * N + gx;
        B[gidx] = A[gidx] * 2.0f;
    }
}

// =====================================================================
// Kernel 2: synchronous smem-staged copy
//   gmem -> reg -> smem -> reg -> gmem
//   Same memory hierarchy as TMA, but no async machinery.
// =====================================================================
__global__ void copy_smem_sync_kernel(const float* __restrict__ A,
                                      float* __restrict__ B,
                                      int M, int N)
{
    __shared__ float smem_tile[TILE_ELEMS];

    int tile_x = blockIdx.x * TILE_N;
    int tile_y = blockIdx.y * TILE_M;

    // Stage 1: gmem -> smem (via registers, synchronous)
    for (int i = threadIdx.x; i < TILE_ELEMS; i += blockDim.x) {
        int ly = i / TILE_N, lx = i % TILE_N;
        smem_tile[i] = A[(tile_y + ly) * N + (tile_x + lx)];
    }
    __syncthreads();

    // Stage 2: compute in smem
    for (int i = threadIdx.x; i < TILE_ELEMS; i += blockDim.x) {
        smem_tile[i] *= 2.0f;
    }
    __syncthreads();

    // Stage 3: smem -> gmem
    for (int i = threadIdx.x; i < TILE_ELEMS; i += blockDim.x) {
        int ly = i / TILE_N, lx = i % TILE_N;
        B[(tile_y + ly) * N + (tile_x + lx)] = smem_tile[i];
    }
}

// =====================================================================
// Kernel 3: cp.async-staged copy (Ampere-style async)
//   gmem -> smem (async, bypasses registers) -> reg -> gmem
//   This is the closest pre-TMA comparison.
// =====================================================================
__global__ void copy_cp_async_kernel(const float* __restrict__ A,
                                     float* __restrict__ B,
                                     int M, int N)
{
    __shared__ float smem_tile[TILE_ELEMS];

    auto block = cg::this_thread_block();
    int tile_x = blockIdx.x * TILE_N;
    int tile_y = blockIdx.y * TILE_M;

    // Async copy a contiguous row at a time — for a row-major
    // tile we issue TILE_M async copies of TILE_N floats each.
    // memcpy_async splits work across the block automatically.
    for (int row = 0; row < TILE_M; ++row) {
        const float* src = A + (tile_y + row) * N + tile_x;
        float*       dst = smem_tile + row * TILE_N;
        cg::memcpy_async(block, dst, src, sizeof(float) * TILE_N);
    }

    // One wait covers all queued async copies in this block
    cg::wait(block);

    // Compute in smem
    for (int i = threadIdx.x; i < TILE_ELEMS; i += blockDim.x) {
        smem_tile[i] *= 2.0f;
    }
    __syncthreads();

    // Write back synchronously
    for (int i = threadIdx.x; i < TILE_ELEMS; i += blockDim.x) {
        int ly = i / TILE_N, lx = i % TILE_N;
        B[(tile_y + ly) * N + (tile_x + lx)] = smem_tile[i];
    }
}

// =====================================================================
// Bench harness: shared verification + timing
// =====================================================================
template <typename KernelFn>
double bench(const char* name, KernelFn launch_kernel,
             const std::vector<float>& hA,
             float* dA, float* dB, std::vector<float>& hB,
             size_t bytes, int iters)
{
    // Reset output
    CUDA_CHECK(cudaMemset(dB, 0, bytes));

    // Warmup
    launch_kernel();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Verify (after warmup, dB has the result)
    CUDA_CHECK(cudaMemcpy(hB.data(), dB, bytes, cudaMemcpyDeviceToHost));
    int errs = 0;
    for (size_t i = 0; i < hA.size() && errs < 5; ++i) {
        float expect = hA[i] * 2.0f;
        if (fabsf(hB[i] - expect) > 1e-5f) {
            fprintf(stderr, "[%s] mismatch @ %zu: got %f expected %f\n",
                    name, i, hB[i], expect); errs++;
        }
    }

    // Time
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));
    CUDA_CHECK(cudaEventRecord(ev_start));
    for (int i = 0; i < iters; ++i) launch_kernel();
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
    ms /= iters;

    double bw = (2.0 * (double)bytes / 1e9) / (ms / 1e3);  // R + W
    printf("%-22s  errs=%d  time=%.3f ms  BW(R+W)=%7.1f GB/s\n",
           name, errs, ms, bw);

    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    return bw;
}

int main() {
    int dev = 0;
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("Device: %s  CC: %d.%d  HBM peak ~ %.0f GB/s\n\n",
           prop.name, prop.major, prop.minor,
        (prop.memoryBusWidth / 8) / 1e6);

    const int M = 4096, N = 4096;
    const size_t bytes = (size_t)M * N * sizeof(float);
    const int ITERS = 50;

    std::vector<float> hA((size_t)M * N), hB((size_t)M * N, 0.0f);
    for (int i = 0; i < M * N; ++i) hA[i] = (float)(i % 1024) * 0.001f;

    float *dA, *dB;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice));

    dim3 grid(N / TILE_N, M / TILE_M);
    dim3 block(128);

    printf("Problem: %dx%d FP32 (= %.1f MB), tile %dx%d, grid %dx%d, block %d, iters=%d\n\n",
           M, N, bytes / 1e6, TILE_M, TILE_N, grid.x, grid.y, block.x, ITERS);

    bench("plain coalesced",
          [&]{ copy_plain_kernel<<<grid, block>>>(dA, dB, M, N); },
          hA, dA, dB, hB, bytes, ITERS);

    bench("smem synchronous",
          [&]{ copy_smem_sync_kernel<<<grid, block>>>(dA, dB, M, N); },
          hA, dA, dB, hB, bytes, ITERS);

    bench("cp.async (Ampere)",
          [&]{ copy_cp_async_kernel<<<grid, block>>>(dA, dB, M, N); },
          hA, dA, dB, hB, bytes, ITERS);

    cudaFree(dA); cudaFree(dB);
    return 0;
}
