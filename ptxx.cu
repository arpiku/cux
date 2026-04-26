#include <cuda_runtime.h>
#include <cuda.h>


__global__ void asm_vec_add(const uint* a, const uint* b, uint* c, uint N) {
    int tid = threadIdx.x + blockDim.x*blockIdx.x;

    uint a_val = a[tid];
    uint b_val = b[tid];
    uint c_val;

    if(tid >= N) return;
    asm("add.u32 %0, %1, %2" : "=r"(c_val) : "r"(a_val),"r"(b_val));

    c[tid] = c_val;
}

int main() {
    return 0;
}
