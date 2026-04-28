#include <cuda.h>
#include <cuda/barrier>
#include <cudaTypedefs.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;
using barrier = cuda::barrier<cuda::thread_scope_block>;

#define BLOCK_M 128
#define BLOCK_N 128

// PTX wrapper for TMA load (cp.async.bulk.tensor.2d)
__device__ inline void tma_load_2d(void* smem_ptr,
                                    const CUtensorMap* tmap,
                                    int crd0, int crd1,
                                    barrier& bar) {
    uint64_t* bar_ptr = reinterpret_cast<uint64_t*>(
        cuda::device::barrier_native_handle(bar));

    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :
        : "r"(__cvta_generic_to_shared(smem_ptr)),
          "l"(tmap),
          "r"(crd0), "r"(crd1),
          "r"(__cvta_generic_to_shared(bar_ptr))
        : "memory");
}

__global__ void tma_gemm_load_kernel(const __grid_constant__ CUtensorMap tmap_A,
                                      float* output, int M, int N) {
    // Shared memory tile, must be 128-byte aligned for TMA
    __shared__ alignas(128) float smem_A[BLOCK_M * BLOCK_N];

    // mbarrier in shared memory
    __shared__ barrier bar;

    if (threadIdx.x == 0) {
        // Initialize barrier with expected arrival count = 1 (only thread 0 arrives)
        init(&bar, 1);
        // Fence to make barrier visible
        cuda::device::experimental::fence_proxy_async_shared_cta();
    }
    __syncthreads();

    // Compute tile coordinates
    int tile_x = blockIdx.x * BLOCK_N;
    int tile_y = blockIdx.y * BLOCK_M;

    barrier::arrival_token token;
    if (threadIdx.x == 0) {
        // Issue the bulk TMA load
        tma_load_2d(smem_A, &tmap_A, tile_x, tile_y, bar);

        // Tell the barrier how many bytes to expect
        token = cuda::device::barrier_arrive_tx(
            bar, 1, sizeof(float) * BLOCK_M * BLOCK_N);
    } else {
        token = bar.arrive();
    }

    // All threads wait for the data to land
    bar.wait(std::move(token));

    // Now the entire tile is in smem_A — use it
    int tid = threadIdx.x;
    if (tid < BLOCK_M * BLOCK_N) {
        output[blockIdx.y * gridDim.x * BLOCK_M * BLOCK_N +
               blockIdx.x * BLOCK_M * BLOCK_N + tid] = smem_A[tid] * 2.0f;
    }
}
