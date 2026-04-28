#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>
#include <sys/types.h>
#include <iostream>

/*
"h" = .u16 reg
"r" = .u32 reg
"l" = .u64 reg
"q" = .u128 reg // Only on platforms that support __int128
"f" = .f32 reg
"d" = .f64 reg
*/

//asm("cvt.f32.s64 %0, %1;" : "=f"(x) : "l"(y));

/*
ld.s64 rd1, [y];
cvt.f32.s64 f1, rd1;
st.f32 [x], f1;
*/

/*
 asm("add.s32 %0, %1, %2;" : "=r"(i) : "r"(j), "r"(k));
 Conceptually the above is like:

 add.s32 i, j, k;

 asm("add.s32 %0, %1, %1;" : "=r"(i) : "r"(k));
 add.s32 i, k, k;

 Three parts to it:
 ("The PTX COMMAND ": "OUTPUT" : "INPUT" )
    def func(a, b) -> c
    c foo(a,b)
    func : c : a,b
 */


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

__global__ void cu_vec_add(const uint32_t* a, const uint32_t* b, uint32_t* c, uint N) {
    int tid = threadIdx.x + blockDim.x*blockIdx.x;
    if(tid >= N) return;
    c[tid] = a[tid] + b[tid];
}

// asm("mov.u32 %0, %%clock;" : "=r"(x));
__device__ uint32_t ptx_clk() {
    uint32_t x;
    asm("mov.u32 %0, %%clock;" : "=r"(x));
    return x;
}

__global__ void print_clk() {
    uint32_t clk = ptx_clk();
    printf("Clock: %u\n", clk);
}



__device__ int cube (int x)
{
  int y;
  asm(".reg .u32 t1;\n\t"              // temp reg t1
      " mul.lo.u32 t1, %1, %1;\n\t"    // t1 = x * x
      " mul.lo.u32 %0, t1, %1;"        // y = t1 * x
      : "=r"(y) : "r" (x));
  return y;
}

__device__ int cond (int x)
{
  int y = 0;
  asm("{\n\t"
      " .reg .pred %p;\n\t"
      " setp.eq.s32 %p, %1, 34;\n\t" // x == 34?
      " @%p mov.s32 %0, 1;\n\t"      // set y to 1 if true
      "}"                            // conceptually y = (x==34)?1:y
      : "+r"(y) : "r" (x));
  return y;
}




int main() {
    uint32_t* hxa, *hxb, *hxc;
    uint N = 10;

    hxa = new uint32_t[N];
    hxb = new uint32_t[N];
    hxc = new uint32_t[N];

    for (int i = 0; i < N; i++) {
        hxa[i] = 1;
        hxb[i] = 1;
        hxc[i] = 0;
    }


    uint32_t* dxa, *dxb, *dxc;

    cudaMalloc(&dxa, N*sizeof(uint32_t));
    cudaMalloc(&dxb, N*sizeof(uint32_t));
    cudaMalloc(&dxc, N*sizeof(uint32_t));

    cudaMemcpy(dxa, hxa, N*sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dxb, hxb, N*sizeof(uint32_t), cudaMemcpyHostToDevice);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    asm_vec_add<<<1, 10, 0 ,stream>>>(dxa, dxb, dxc, N);
    cudaEventRecord(stop);
    cudaStreamSynchronize(stream);

    float time0 = 0.0f;
    cudaEventElapsedTime(&time0, start, stop);

    std::cout << "Time: " << time0 << " ms" << std::endl;

    free(hxa);
    free(hxb);
    free(hxc);
    cudaFree(dxa);
    cudaFree(dxb);
    cudaFree(dxc);
    cudaStreamDestroy(stream);

    print_clk<<<1,1>>>();

    // cu_vec_add<<<1, N>>>(dxa, dxb, dxc, N);

    return 0;
}
