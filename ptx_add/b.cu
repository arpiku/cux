

#include <cuda_runtime.h>

#include <cuda.h>
#include <cstdint>
#include <sys/types.h>
#include <iostream>


 __device__ uint32_t ptx_add(uint32_t a, uint32_t b) {
     uint32_t c;
     asm("add.u32 %0, %1, %2;" : "=r"(c) : "r"(a),"r"(b));
     return c;
 }


__global__ void asm_vec_add(const uint32_t* a, const uint32_t* b, uint32_t* c, uint N) {
    int tid = threadIdx.x + blockDim.x*blockIdx.x;

    if(tid >= N) return;
    c[tid] = ptx_add(a[tid], b[tid]);
}

int main() {
    return 0;
}
