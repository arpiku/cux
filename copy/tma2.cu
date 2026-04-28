// file: tma_3d_copy.cu
// Build: nvcc -arch=sm_90a -std=c++17 -O3 tma_3d_copy.cu -o tma_3d_copy

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda/barrier>
#include <cudaTypedefs.h>
#include <cstdio>
#include <vector>

#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e){fprintf(stderr,"%s\n",cudaGetErrorString(e)); std::exit(1);}} while(0)
#define CU_CHECK(x)   do { CUresult r=(x);  if(r){const char*s; cuGetErrorString(r,&s); fprintf(stderr,"%s\n",s); std::exit(1);}} while(0)

using barrier_t = cuda::barrier<cuda::thread_scope_block>;

// Real-world: attention K cache shape [B, S, D]
// We tile over (S, D) per batch.
constexpr int TILE_B = 1;     // one batch at a time
constexpr int TILE_S = 64;    // 64 sequence positions
constexpr int TILE_D = 64;    // 64 head_dim elements

__device__ __forceinline__
void tma_load_3d(void* smem, const CUtensorMap* tmap,
                 int c0, int c1, int c2, barrier_t& bar) {
    uint64_t bar_addr = __cvta_generic_to_shared(
        cuda::device::barrier_native_handle(bar));
    uint32_t s = __cvta_generic_to_shared(smem);
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4}], [%5];"
        :: "r"(s), "l"(tmap), "r"(c0), "r"(c1), "r"(c2), "l"(bar_addr)
        : "memory");
}

__device__ __forceinline__
void tma_store_3d(const void* smem, const CUtensorMap* tmap,
                  int c0, int c1, int c2) {
    uint32_t s = __cvta_generic_to_shared(smem);
    asm volatile(
        "cp.async.bulk.tensor.3d.global.shared::cta.bulk_group"
        " [%0, {%1, %2, %3}], [%4];"
        :: "l"(tmap), "r"(c0), "r"(c1), "r"(c2), "r"(s)
        : "memory");
}

__device__ __forceinline__ void tma_commit() { asm volatile("cp.async.bulk.commit_group;" ::: "memory"); }
template<int N> __device__ __forceinline__ void tma_wait() { asm volatile("cp.async.bulk.wait_group %0;" :: "n"(N) : "memory"); }
__device__ __forceinline__ void fence_async() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }

__global__ void tma_3d_copy_kernel(
    const __grid_constant__ CUtensorMap tmap_src,
    const __grid_constant__ CUtensorMap tmap_dst,
    int B, int S, int D)
{
    // Layout in smem mirrors box: [TILE_B][TILE_S][TILE_D]
    __shared__ alignas(128) float smem_tile[TILE_B * TILE_S * TILE_D];
    __shared__ barrier_t bar;

    int tid = threadIdx.x;
    // grid dims: x = D-tiles, y = S-tiles, z = batch
    int d0 = blockIdx.x * TILE_D;   // innermost
    int s0 = blockIdx.y * TILE_S;
    int b0 = blockIdx.z * TILE_B;

    if (tid == 0) { init(&bar, 1); fence_async(); }
    __syncthreads();

    barrier_t::arrival_token tok;
    if (tid == 0) {
        // KEY: coords are passed innermost-first to match the descriptor.
        tma_load_3d(smem_tile, &tmap_src, d0, s0, b0, bar);
        tok = cuda::device::barrier_arrive_tx(bar, 1,
              sizeof(float) * TILE_B * TILE_S * TILE_D);
    } else {
        tok = bar.arrive();
    }
    bar.wait(std::move(tok));

    // Pretend-compute: scale by batch index (just to exercise per-batch logic)
    float scale = 1.0f + 0.5f * b0;
    for (int i = tid; i < TILE_B * TILE_S * TILE_D; i += blockDim.x)
        smem_tile[i] *= scale;
    __syncthreads();

    if (tid == 0) {
        fence_async();
        tma_store_3d(smem_tile, &tmap_dst, d0, s0, b0);
        tma_commit();
        tma_wait<0>();
    }
}

CUtensorMap make_3d_tensormap(float* gmem, int B, int S, int D) {
    CUtensorMap tmap{};
    // INNERMOST FIRST: D, S, B
    cuuint64_t size[3]   = { (cuuint64_t)D, (cuuint64_t)S, (cuuint64_t)B };
    // strides for the 2 outer dims, in BYTES
    cuuint64_t stride[2] = {
        (cuuint64_t)D     * sizeof(float),         // bytes per S row
        (cuuint64_t)S * D * sizeof(float)          // bytes per B slab
    };
    cuuint32_t box[3]    = { TILE_D, TILE_S, TILE_B };
    cuuint32_t es[3]     = { 1, 1, 1 };

    CU_CHECK(cuTensorMapEncodeTiled(
        &tmap, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 3,
        gmem, size, stride, box, es,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    return tmap;
}

int main() {
    // Real-world sized: 8 batch, 2048 seq, 128 head_dim
    const int B = 8, S = 2048, D = 128;
    const size_t bytes = (size_t)B * S * D * sizeof(float);

    std::vector<float> hA((size_t)B * S * D), hB((size_t)B * S * D, 0);
    for (size_t i = 0; i < hA.size(); ++i) hA[i] = (float)((i * 7) % 1024) * 0.001f;

    float *dA, *dB;
    CUDA_CHECK(cudaMalloc(&dA, bytes)); CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dB, 0, bytes));

    CUtensorMap tmap_src = make_3d_tensormap(dA, B, S, D);
    CUtensorMap tmap_dst = make_3d_tensormap(dB, B, S, D);

    // Grid: (D-tiles, S-tiles, B-tiles)
    dim3 grid(D / TILE_D, S / TILE_S, B / TILE_B);
    dim3 block(128);

    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    tma_3d_copy_kernel<<<grid, block>>>(tmap_src, tmap_dst, B, S, D);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms = 0; cudaEventElapsedTime(&ms, s, e);

    CUDA_CHECK(cudaMemcpy(hB.data(), dB, bytes, cudaMemcpyDeviceToHost));

    int errs = 0;
    for (size_t b = 0; b < (size_t)B && errs < 5; ++b) {
        float scale = 1.0f + 0.5f * (b * TILE_B);
        for (size_t i = 0; i < (size_t)S * D && errs < 5; ++i) {
            size_t idx = b * S * D + i;
            float exp = hA[idx] * scale;
            if (fabsf(hB[idx] - exp) > 1e-4f) {
                fprintf(stderr, "Mismatch [b=%zu i=%zu]: got %f expected %f\n",
                        b, i, hB[idx], exp); errs++;
            }
        }
    }
    printf("3D copy: errors=%d  time=%.3f ms  BW(R+W)=%.1f GB/s\n",
           errs, ms, (2.0 * bytes / 1e9) / (ms / 1e3));

    cudaFree(dA); cudaFree(dB);
    return 0;
}
