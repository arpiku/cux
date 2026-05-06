#include "main.cuh"

int main() {

  GemmShape shape({512, 512, 512});

  auto x_gemm = [&](int m, int n, int k, const __nv_bfloat16 *a_,
                    const __nv_bfloat16 *b_, float *c_, cudaStream_t stream) {
    xgemms::x_bf16(m, n, k, a_, b_, c_, stream);
  };

  auto cu_gemm = [&](cublasHandle_t handle, int m, int n, int k,
                     const __nv_bfloat16 *a_, const __nv_bfloat16 *b_,
                     float *c_, cudaStream_t stream) {
    cugemms::bf16_tc_nn(handle, m, n, k, a_, b_, c_, stream);
  };

  auto res = runner<__nv_bfloat16>(shape, x_gemm, cu_gemm, 1, 2);

  print_comparison(res);

  return 0;
}
