#include <cuda_runtime.h>
#include <stdio.h>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/ptx>


#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

namespace cg = cooperative_groups;
namespace ptx = cuda::ptx;

__global__ void cooperativeGroupsKernel() {
    auto this_block = cg::this_thread_block();
    int tot_threads = this_block.num_threads();
    dim3 thread_coordinate = this_block.thread_index();
    dim3 block_dim = this_block.group_dim();
    auto this_grid = cg::this_grid();
    ptx::mbarrier_init(&bar, 1);




    printf("tot_threads : %d, block_coordinate: (%d, %d, %d)\n", tot_threads, thread_coordinate.x, thread_coordinate.y, thread_coordinate.z);
    printf("Block_dims : (%d, %d, %d)\n", block_dim.x, block_dim.y, block_dim.z);
}

__global__ void cooperativeGroupsKernelTiled() {
    auto this_block = cg::this_thread_block();
    int tot_threads = this_block.num_threads();
    dim3 thread_coordinate = this_block.thread_index();
    dim3 block_dim = this_block.group_dim();


    printf("tot_threads : %d, block_coordinate: (%d, %d, %d)\n", tot_threads, thread_coordinate.x, thread_coordinate.y, thread_coordinate.z);
    printf("Block_dims : (%d, %d, %d)\n", block_dim.x, block_dim.y, block_dim.z);
}

// Kernel definition
// No compile time attribute attached to the kernel
__global__ void cluster_kernel(float *input, float* output)
{
    auto this_block = cg::this_thread_block();
    int tot_threads = this_block.num_threads();
    dim3 thread_coordinate = this_block.thread_index();
    dim3 block_dim = this_block.group_dim();
    dim3 block_coordinate = this_block.group_index();

    auto this_cluster = cg::this_cluster();

    auto active_threads = cg::coalesced_threads();


    printf("N_threads : %d, thread_coordinate: (%d, %d, %d)\n", tot_threads, thread_coordinate.x, thread_coordinate.y, thread_coordinate.z);
    printf("Block_dims : (%d, %d, %d)\n", block_dim.x, block_dim.y, block_dim.z);
    printf("Block_coordinate : (%d, %d, %d)\n", block_coordinate.x, block_coordinate.y, block_coordinate.z);
    printf("_______________");
    printf("Cluster_thread_rank : (%d), Block_thread_rank : (%d)\n", this_cluster.thread_rank(), this_block.thread_rank());
    printf("Threads_in_cluster : (%d), Threads_in_block : (%d)\n", this_cluster.num_threads(), this_block.num_threads());
}

int main() {
    const int N = 128;
    dim3 blockDim(4, 2, 2);
    dim3 gridDim(2, 2, 1);


    printf("\n--- Cooperative Groups Kernel (blockDim: %dx%dx%d, gridDim: %dx%dx%d) ---\n",
           blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y, gridDim.z);
    cooperativeGroupsKernel<<<gridDim, blockDim>>>();
    CHECK_CUDA(cudaDeviceSynchronize());

   //
   //
   float *input, *output;
   dim3 threadsPerBlock(16, 16);
   dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);

   // Kernel invocation with runtime cluster size
       cudaLaunchConfig_t config = {0};
       // The grid dimension is not affected by cluster launch, and is still enumerated
       // using number of blocks.
       // The grid dimension should be a multiple of cluster size.
       config.gridDim = numBlocks;
       config.blockDim = threadsPerBlock;

       cudaLaunchAttribute attribute[1];
       attribute[0].id = cudaLaunchAttributeClusterDimension;
       attribute[0].val.clusterDim.x = 2; // Cluster size in X-dimension
       attribute[0].val.clusterDim.y = 1;
       attribute[0].val.clusterDim.z = 1;
       config.attrs = attribute;
       config.numAttrs = 1;

       cudaLaunchKernelEx(&config, cluster_kernel, input, output);

       CHECK_CUDA(cudaDeviceSynchronize());
    return 0;
}
