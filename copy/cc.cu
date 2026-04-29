#include <cuda_pipeline.h>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

namespace cg = cooperative_groups;

__global__ void cp_async_kernel(const float* __restrict__ gmem,
                                 float* __restrict__ out, int N) {
    extern __shared__ float smem[];
    auto block = cg::this_thread_block();

    // Async copy gmem -> smem (skips registers)
    cg::memcpy_async(block, smem, gmem, sizeof(float) * blockDim.x);

    // Wait for the copy to complete
    cg::wait(block);

    // Now use smem
    int tid = threadIdx.x;
    out[blockIdx.x * blockDim.x + tid] = smem[tid] * 2.0f;
}

__global__ void cp_async_k1(const float* __restrict__ gmem,
                                 float* __restrict__ out, int N) {
     extern __shared__ float smem[];
     auto block = cg::this_thread_block();
     auto grid = cg::this_grid();
     auto  = cg::this_thread_block().num_threads();




}


int main() {

   return 0;
}
