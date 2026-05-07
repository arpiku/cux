#include "main.cuh"

int main() {
  std::vector<GemmShape> shapes = get_shape_list(5, 10);
  std::vector<ComparisonRow> rows;
  rows.reserve(shapes.size());

  for (const GemmShape &shape : shapes) {
    auto x_gemm = [&](int m, int n, int k, const __nv_bfloat16 *a_,
                      const __nv_bfloat16 *b_, float *c_,
                      cudaStream_t stream) {
      xgemms::x0_bf16_d(m, n, k, a_, b_, c_, stream);
    };

    auto cu_gemm = [&](cublasHandle_t handle, int m, int n, int k,
                       const __nv_bfloat16 *a_, const __nv_bfloat16 *b_,
                       float *c_, cudaStream_t stream) {
      cugemms::bf16_tc_nn(handle, m, n, k, a_, b_, c_, stream);
    };

    auto res = runner<__nv_bfloat16>(shape, x_gemm, cu_gemm, 1, 2, false);
    rows.push_back(make_comparison_row(res));
  }

  print_comparison_table(rows);

  return 0;
}
