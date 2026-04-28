#include <cuda_runtime.h>
#include <stdio.h>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

__global__ void traditionalKernel() {
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    int tid_z = threadIdx.z;

    int bid_x = blockIdx.x;
    int bid_y = blockIdx.y;
    int bid_z = blockIdx.z;

    int bdim_x = blockDim.x;
    int bdim_y = blockDim.y;
    int bdim_z = blockDim.z;

    int gdim_x = gridDim.x;
    int gdim_y = gridDim.y;
    int gdim_z = gridDim.z;

    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z) +
                  (threadIdx.z * (blockDim.x * blockDim.y)) +
                  (threadIdx.y * blockDim.x) + threadIdx.x;

    printf("[Traditional] block(%d,%d,%d) thread(%d,%d,%d) blockDim(%d,%d,%d) gridDim(%d,%d,%d) - globalId=%d\n",
           bid_x, bid_y, bid_z, tid_x, tid_y, tid_z,
           bdim_x, bdim_y, bdim_z, gdim_x, gdim_y, gdim_z, threadId);
}

__global__ void cooperativeGroupsKernel() {
    int local_id_x = threadIdx.x;
    int local_id_y = threadIdx.y;
    int local_id_z = threadIdx.z;

    unsigned int local_rank = threadIdx.x +
                             threadIdx.y * blockDim.x +
                             threadIdx.z * blockDim.x * blockDim.y;

    unsigned int tile_rank = (blockIdx.x * blockDim.x * blockDim.y * blockDim.z + local_rank) % 32;

    unsigned int global_tid = blockIdx.x * blockDim.x * blockDim.y * blockDim.z +
                             local_rank;

    printf("[CooperativeGroups] blockIdx(%d,%d,%d) threadIdx(%d,%d,%d) blockDim(%d,%d,%d) gridDim(%d,%d,%d) tileRank=%u globalId=%u\n",
           blockIdx.x, blockIdx.y, blockIdx.z,
           local_id_x, local_id_y, local_id_z,
           blockDim.x, blockDim.y, blockDim.z,
           gridDim.x, gridDim.y, gridDim.z,
           tile_rank, global_tid);
}

__device__ __noinline__ void getPTXRegs(int* tid, int* ctaid, int* ntid, int* nctaid) {
    int reg_tid, reg_ctaid, reg_ntid, reg_nctaid;

    asm volatile("mov.u32 %0, %%tid.x;" : "=r"(reg_tid));
    asm volatile("mov.u32 %0, %%ctaid.x;" : "=r"(reg_ctaid));
    asm volatile("mov.u32 %0, %%ntid.x;" : "=r"(reg_ntid));
    asm volatile("mov.u32 %0, %%nctaid.x;" : "=r"(reg_nctaid));

    *tid = reg_tid;
    *ctaid = reg_ctaid;
    *ntid = reg_ntid;
    *nctaid = reg_nctaid;
}

__global__ void ptxAsmKernel() {
    int tid, ctaid, ntid, nctaid;

    getPTXRegs(&tid, &ctaid, &ntid, &nctaid);

    int tid_y, ctaid_y, ntid_y, nctaid_y;
    asm volatile("mov.u32 %0, %%tid.y;" : "=r"(tid_y));
    asm volatile("mov.u32 %0, %%ctaid.y;" : "=r"(ctaid_y));
    asm volatile("mov.u32 %0, %%ntid.y;" : "=r"(ntid_y));
    asm volatile("mov.u32 %0, %%nctaid.y;" : "=r"(nctaid_y));

    int tid_z, ctaid_z, nctaid_z;
    asm volatile("mov.u32 %0, %%tid.z;" : "=r"(tid_z));
    asm volatile("mov.u32 %0, %%ctaid.z;" : "=r"(ctaid_z));
    asm volatile("mov.u32 %0, %%nctaid.z;" : "=r"(nctaid_z));

    int clusterid;
    asm volatile("mov.u32 %0, %%clusterid.x;" : "=r"(clusterid));

    printf("[PTX ASM] %%tid=(%d,%d,%d) %%ctaid=(%d,%d,%d) %%ntid=(%d,%d) %%nctaid=(%d,%d,%d) %%clusterid=%d\n",
           tid, tid_y, tid_z, ctaid, ctaid_y, ctaid_z,
           ntid, ntid_y, nctaid, nctaid_y, nctaid_z, clusterid);
}

__global__ void multiDimKernel() {
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    int tid_z = threadIdx.z;

    int bid_x = blockIdx.x;
    int bid_y = blockIdx.y;
    int bid_z = blockIdx.z;

    int bdim_x = blockDim.x;
    int bdim_y = blockDim.y;
    int bdim_z = blockDim.z;

    int gdim_x = gridDim.x;
    int gdim_y = gridDim.y;
    int gdim_z = gridDim.z;

    int global_tid_3d = (bid_z * gdim_y * gdim_x * bdim_z * bdim_y * bdim_x +
                        bid_y * gdim_x * bdim_z * bdim_y * bdim_x +
                        bid_x * bdim_z * bdim_y * bdim_x) +
                       (tid_z * bdim_y * bdim_x +
                        tid_y * bdim_x +
                        tid_x);

    int linear_thread = threadIdx.x +
                        threadIdx.y * blockDim.x +
                        threadIdx.z * blockDim.x * blockDim.y;

    int linear_block = blockIdx.x +
                       blockIdx.y * gridDim.x +
                       blockIdx.z * gridDim.x * gridDim.y;

    printf("[MultiDim 3D] Block(%d,%d,%d) Thread(%d,%d,%d) BlockDim(%d,%d,%d) GridDim(%d,%d,%d) globalId=%d\n",
           bid_x, bid_y, bid_z, tid_x, tid_y, tid_z,
           bdim_x, bdim_y, bdim_z, gdim_x, gdim_y, gdim_z,
           global_tid_3d);

    printf("[MultiDim] linearBlock=%d linearThread=%d flatId=%d\n",
           linear_block, linear_thread, linear_block * (blockDim.x * blockDim.y * blockDim.z) + linear_thread);
}

__global__ void clusterKernel() {
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    int tid_z = threadIdx.z;

    int bid_x = blockIdx.x;
    int bid_y = blockIdx.y;
    int bid_z = blockIdx.z;

    int cluster_ndim_x = gridDim.x;
    int cluster_ndim_y = gridDim.y;
    int cluster_ndim_z = gridDim.z;

    int cluster_ctaid_x, cluster_ctaid_y, cluster_ctaid_z;
    asm volatile("mov.u32 %0, %%clusterid.x;" : "=r"(cluster_ctaid_x));
    asm volatile("mov.u32 %0, %%clusterid.y;" : "=r"(cluster_ctaid_y));
    asm volatile("mov.u32 %0, %%clusterid.z;" : "=r"(cluster_ctaid_z));

    unsigned int group_cluster_idx = cluster_ctaid_x +
                                   cluster_ctaid_y * gridDim.x +
                                   cluster_ctaid_z * gridDim.x * gridDim.y;

    printf("[Cluster] blockIdx(%d,%d,%d) threadIdx(%d,%d,%d) gridDim(%d,%d,%d) clusterNCtaId=(%d,%d,%d) clusterGroupIdx=%u\n",
           bid_x, bid_y, bid_z, tid_x, tid_y, tid_z,
           cluster_ndim_x, cluster_ndim_y, cluster_ndim_z,
           cluster_ctaid_x, cluster_ctaid_y, cluster_ctaid_z,
           group_cluster_idx);
}

int main(int argc, char* argv[]) {
    printf("=== CUDA Dimensioning Demo ===\n\n");

    dim3 blockDim(4, 2, 2);
    dim3 gridDim(2, 2, 1);

    printf("--- Traditional Kernel (blockDim: %dx%dx%d, gridDim: %dx%dx%d) ---\n",
           blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y, gridDim.z);
    traditionalKernel<<<gridDim, blockDim>>>();
    CHECK_CUDA(cudaDeviceSynchronize());

    printf("\n--- Cooperative Groups Kernel (blockDim: %dx%dx%d, gridDim: %dx%dx%d) ---\n",
           blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y, gridDim.z);
    cooperativeGroupsKernel<<<gridDim, blockDim>>>();
    CHECK_CUDA(cudaDeviceSynchronize());

    printf("\n--- PTX ASM Kernel (blockDim: %dx%dx%d, gridDim: %dx%dx%d) ---\n",
           blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y, gridDim.z);
    ptxAsmKernel<<<gridDim, blockDim>>>();
    CHECK_CUDA(cudaDeviceSynchronize());

    printf("\n--- Multi-Dim Kernel (blockDim: %dx%dx%d, gridDim: %dx%dx%d) ---\n",
           blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y, gridDim.z);
    multiDimKernel<<<gridDim, blockDim>>>();
    CHECK_CUDA(cudaDeviceSynchronize());

    printf("\n--- Cluster Kernel (blockDim: %dx%dx%d, gridDim: %dx%dx%d) ---\n",
           blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y, gridDim.z);
    clusterKernel<<<gridDim, blockDim>>>();
    CHECK_CUDA(cudaDeviceSynchronize());

    printf("\n=== Demo Complete ===\n");

    return 0;
}