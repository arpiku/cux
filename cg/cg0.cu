
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdint>
#include <iostream>

namespace cg = cooperative_groups;

// Each block is partitioned into 32-thread warps,
// then each warp is sub-partitioned into 16-thread tiles.
template <int BLOCK_THREADS>
__global__ void cg_example0(const int* in, int* out, int n)
{
    static_assert(BLOCK_THREADS % 32 == 0, "BLOCK_THREADS must be a multiple of 32");
    static_assert(32 % 16 == 0, "warp size must be divisible by tile size");

    extern __shared__ int smem[];

    cg::thread_block block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);   // warp-sized group
    auto tile = cg::tiled_partition<16>(warp);     // sub-tile inside the warp

    const int tid    = blockIdx.x * BLOCK_THREADS + threadIdx.x;
    const int value  = (tid < n) ? in[tid] : 0;

    // One shared-memory slot per thread in the block.
    smem[threadIdx.x] = value;

    // Only the 16 threads in this tile need to synchronize.
    tile.sync();

    // Reduction inside the 16-thread tile.
    const int lane = tile.thread_rank();
    const int base =
        warp.meta_group_rank() * warp.size() +
        tile.meta_group_rank() * tile.size();

    for (int offset = tile.size() / 2; offset > 0; offset >>= 1) {
        if (lane < offset) {
            smem[base + lane] += smem[base + lane + offset];
        }
        tile.sync();
    }

    // Lane 0 of each 16-thread tile writes one result.
    if (lane == 0) {
        const int tiles_per_warp = warp.size() / tile.size(); // 2
        const int out_idx =
            blockIdx.x * (BLOCK_THREADS / tile.size()) +
            warp.meta_group_rank() * tiles_per_warp +
            tile.meta_group_rank();

        out[out_idx] = smem[base];
    }
}

// Older/manual equivalent: same logical result, but you do the
// partitioning yourself and use __syncthreads() for the whole block.
template <int BLOCK_THREADS>
__global__ void legacy_example0(const int* in, int* out, int n)
{
    __shared__ int smem[BLOCK_THREADS];

    const int tid = threadIdx.x;
    const int global = blockIdx.x * BLOCK_THREADS + tid;

    const int warp_id = tid / 32;
    const int subtile_id = (tid % 32) / 16;   // 0 or 1 inside a warp
    const int lane = tid % 16;                // 0..15 inside the subtile

    smem[tid] = (global < n) ? in[global] : 0;
    __syncthreads();

    const int base = warp_id * 32 + subtile_id * 16;

    for (int offset = 8; offset > 0; offset >>= 1) {
        if (lane < offset) {
            smem[base + lane] += smem[base + lane + offset];
        }
        __syncthreads(); // heavier than needed: syncs the whole block
    }

    if (lane == 0) {
        out[blockIdx.x * (BLOCK_THREADS / 16) + warp_id * 2 + subtile_id] = smem[base];
    }
}

int main()
{
    constexpr int BLOCK_THREADS = 256;
    constexpr int N = 1024;
    constexpr int OUT_PER_BLOCK = BLOCK_THREADS / 16;
    constexpr int GRID = N / BLOCK_THREADS;

    int *h_in = new int[N];
    for (int i = 0; i < N; ++i) h_in[i] = 1;

    int *d_in = nullptr, *d_out0 = nullptr, *d_out1 = nullptr;
    cudaMalloc(&d_in,   N * sizeof(int));
    cudaMalloc(&d_out0, GRID * OUT_PER_BLOCK * sizeof(int));
    cudaMalloc(&d_out1, GRID * OUT_PER_BLOCK * sizeof(int));

    cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice);

    cg_example0<BLOCK_THREADS><<<GRID, BLOCK_THREADS, BLOCK_THREADS * sizeof(int)>>>(d_in, d_out0, N);
    legacy_example0<BLOCK_THREADS><<<GRID, BLOCK_THREADS>>>(d_in, d_out1, N);

    cudaDeviceSynchronize();

    cudaFree(d_in);
    cudaFree(d_out0);
    cudaFree(d_out1);
    delete[] h_in;

    return 0;
}
