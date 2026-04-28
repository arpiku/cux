
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdint>
#include <iostream>

namespace cg = cooperative_groups;

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                          \
        if (err__ != cudaSuccess) {                                          \
            std::cerr << "CUDA error: " << cudaGetErrorString(err__)         \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

// Cooperative Groups version: block -> warp -> 16-thread tile.
template <int BLOCK_THREADS>
__global__ void cg_example0(const int* in, int* out, int n)
{
    __shared__ int smem[BLOCK_THREADS];

    cg::thread_block block = cg::this_thread_block();
    auto tile = cg::tiled_partition<16>(block);

    const int tid   = blockIdx.x * BLOCK_THREADS + threadIdx.x;
    const int value = (tid < n) ? in[tid] : 0;

    smem[threadIdx.x] = value;
    tile.sync();

    const int lane = tile.thread_rank();
    const int tile_id = tile.meta_group_rank();
    const int base = tile_id * tile.size();

    for (int offset = tile.size() / 2; offset > 0; offset >>= 1) {
        if (lane < offset) {
            smem[base + lane] += smem[base + lane + offset];
        }
        tile.sync();
    }

    if (lane == 0) {
        const int out_idx = blockIdx.x * (BLOCK_THREADS / 16) + tile_id;
        out[out_idx] = smem[base];
    }
}

// Manual/legacy version: same logical work, but explicit indexing and __syncthreads().
template <int BLOCK_THREADS>
__global__ void legacy_example0(const int* in, int* out, int n)
{
    __shared__ int smem[BLOCK_THREADS];

    const int tid    = threadIdx.x;
    const int global = blockIdx.x * BLOCK_THREADS + tid;

    const int warp_id   = tid / 32;
    const int subtile_id = (tid % 32) / 16;   // 0 or 1 within a warp
    const int lane      = tid % 16;           // 0..15 within the tile

    smem[tid] = (global < n) ? in[global] : 0;
    __syncthreads();

    const int base = warp_id * 32 + subtile_id * 16;

    for (int offset = 8; offset > 0; offset >>= 1) {
        if (lane < offset) {
            smem[base + lane] += smem[base + lane + offset];
        }
        __syncthreads();
    }

    if (lane == 0) {
        out[blockIdx.x * (BLOCK_THREADS / 16) + warp_id * 2 + subtile_id] = smem[base];
    }
}

template <typename Launcher>
float time_kernel(Launcher launcher, cudaStream_t stream, int iters)
{
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warm-up launch (optional but good practice).
    launcher();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < iters; ++i) {
        launcher();
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return ms / iters;
}

int main()
{
    constexpr int BLOCK_THREADS = 256;
    constexpr int N = 1 << 20;                 // 1M elements
    constexpr int OUT_PER_BLOCK = BLOCK_THREADS / 16;
    constexpr int GRID = (N + BLOCK_THREADS - 1) / BLOCK_THREADS;
    constexpr int ITERS = 100;

    int* h_in = new int[N];
    for (int i = 0; i < N; ++i) h_in[i] = i;

    int *d_in = nullptr, *d_out_cg = nullptr, *d_out_legacy = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out_cg, GRID * OUT_PER_BLOCK * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out_legacy, GRID * OUT_PER_BLOCK * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    auto cg_launcher = [&]() {
        cg_example0<BLOCK_THREADS><<<GRID, BLOCK_THREADS, 0, stream>>>(
            d_in, d_out_cg, N);
    };

    auto legacy_launcher = [&]() {
        legacy_example0<BLOCK_THREADS><<<GRID, BLOCK_THREADS, 0, stream>>>(
            d_in, d_out_legacy, N);
    };

    float cg_ms = time_kernel(cg_launcher, stream, ITERS);

    float legacy_ms = time_kernel(legacy_launcher, stream, ITERS);

    const double cg_s = static_cast<double>(cg_ms) * 1e-3;
    const double legacy_s = static_cast<double>(legacy_ms) * 1e-3;

    std::cout << std::scientific;
    std::cout << "cg_example0:    " << cg_s << " s/launch\n";
    std::cout << "legacy_example0: " << legacy_s << " s/launch\n";
    std::cout << "diff:           " << (cg_s > legacy_s ? (cg_s - legacy_s) : (legacy_s - cg_s)) << " s/launch\n";

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out_cg));
    CUDA_CHECK(cudaFree(d_out_legacy));
    delete[] h_in;

    return 0;
}
