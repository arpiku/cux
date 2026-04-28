#include <cuda.h>

CUtensorMap make_tensor_map_2d(float* gmem_ptr, int M, int N) {
    CUtensorMap tmap{};

    // Tensor dimensions in elements (innermost first!)
    cuuint64_t size[2]    = {(cuuint64_t)N, (cuuint64_t)M};
    // Strides in BYTES, skipping the innermost (innermost is implicit)
    cuuint64_t stride[1]  = {(cuuint64_t)N * sizeof(float)};
    // Box (tile) dimensions in elements
    cuuint32_t box_size[2]   = {BLOCK_N, BLOCK_M};
    // Element strides within the box (usually 1)
    cuuint32_t elem_stride[2] = {1, 1};

    CUresult res = cuTensorMapEncodeTiled(
        &tmap,
        CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
        2,                                 // rank
        gmem_ptr,
        size,
        stride,
        box_size,
        elem_stride,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,        // smem swizzling
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );

    assert(res == CUDA_SUCCESS);
    return tmap;
}

// Launch
CUtensorMap tmap = make_tensor_map_2d(d_A, M, N);
tma_gemm_load_kernel<<<grid, block>>>(tmap, d_out, M, N);


// Pipelined TMA
template<int STAGES>
__global__ void tma_pipelined_kernel(const __grid_constant__ CUtensorMap tmap_A,
                                      const __grid_constant__ CUtensorMap tmap_B,
                                      float* C, int M, int N, int K) {
    // Multi-stage smem buffers
    __shared__ alignas(128) float smem_A[STAGES][BLOCK_M * BLOCK_K];
    __shared__ alignas(128) float smem_B[STAGES][BLOCK_K * BLOCK_N];

    // One barrier per stage, separated into "full" and "empty"
    __shared__ barrier bar_full[STAGES];
    __shared__ barrier bar_empty[STAGES];

    if (threadIdx.x == 0) {
        for (int s = 0; s < STAGES; s++) {
            init(&bar_full[s], 1);   // producer arrives when load completes
            init(&bar_empty[s], blockDim.x); // consumer arrives when tile consumed
        }
        cuda::device::experimental::fence_proxy_async_shared_cta();
    }
    __syncthreads();

    int num_k_tiles = K / BLOCK_K;
    int phase[STAGES] = {0};  // mbarrier phase tracking

    // PRODUCER warp (warp 0)
    if (threadIdx.x / 32 == 0 && threadIdx.x % 32 == 0) {
        for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
            int stage = k_tile % STAGES;

            // Wait for consumer to finish this stage (only after first cycle)
            if (k_tile >= STAGES) {
                bar_empty[stage].wait_parity(phase[stage]);
                phase[stage] ^= 1;
            }

            // Issue TMA loads
            tma_load_2d(&smem_A[stage][0], &tmap_A,
                        k_tile * BLOCK_K, blockIdx.y * BLOCK_M, bar_full[stage]);
            tma_load_2d(&smem_B[stage][0], &tmap_B,
                        blockIdx.x * BLOCK_N, k_tile * BLOCK_K, bar_full[stage]);

            cuda::device::barrier_arrive_tx(
                bar_full[stage], 1,
                sizeof(float) * (BLOCK_M * BLOCK_K + BLOCK_K * BLOCK_N));
        }
    }
    // CONSUMER warps (warp 1+)
    else {
        float accum = 0.0f;
        int consumer_phase[STAGES] = {0};

        for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
            int stage = k_tile % STAGES;

            // Wait for producer to fill this stage
            bar_full[stage].wait_parity(consumer_phase[stage]);
            consumer_phase[stage] ^= 1;

            // ... do mma using smem_A[stage] and smem_B[stage] ...
            // (would use wgmma instructions in real code)

            // Signal that we're done with this stage
            bar_empty[stage].arrive();
        }

        // Write out C
    }
}

__device__ inline void tma_store_2d(const void* smem_ptr,
                                     const CUtensorMap* tmap,
                                     int crd0, int crd1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group"
        " [%0, {%1, %2}], [%3];"
        :
        : "l"(tmap), "r"(crd0), "r"(crd1),
          "r"(__cvta_generic_to_shared(smem_ptr))
        : "memory");
}

__device__ inline void tma_store_commit() {
    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}

__device__ inline void tma_store_wait() {
    asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
}


// TMA Multicast
__device__ inline void tma_load_2d_multicast(void* smem_ptr,
                                              const CUtensorMap* tmap,
                                              int crd0, int crd1,
                                              barrier& bar,
                                              uint16_t cta_mask) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
        ".multicast::cluster"
        " [%0], [%1, {%2, %3}], [%4], %5;"
        :
        : "r"(__cvta_generic_to_shared(smem_ptr)),
          "l"(tmap),
          "r"(crd0), "r"(crd1),
          "r"(__cvta_generic_to_shared(
              cuda::device::barrier_native_handle(bar))),
          "h"(cta_mask)
        : "memory");
}
