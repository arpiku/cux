#include <cuda_runtime.h>

#include <cuda.h>
#include <cstdint>
#include <sys/types.h>
#include <iostream>

__global__ void cu_vec_add(const uint32_t* a, const uint32_t* b, uint32_t* c, uint N) {
    int tid = threadIdx.x + blockDim.x*blockIdx.x;

    if(tid >= N) return;

    c[tid] = a[tid] + b[tid];
}

int main() {
    return 0;
}
