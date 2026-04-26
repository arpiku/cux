#pragma once

#include <cuda_runtime.h>

namespace xgemms {

template <int TILE>
__global__ void gemm_bf16_naive_kernel(const __nv_bfloat16 *__restrict__ A,
                                       const __nv_bfloat16 *__restrict__ B,
                                       float *__restrict__ C, int M, int N,
                                       int K) {
  const int row = blockIdx.y * TILE + threadIdx.y;
  const int col = blockIdx.x * TILE + threadIdx.x;
  if (row >= M || col >= N) {
    return;
  }

  const int lda = M;
  const int ldb = K;
  const int ldc = M;

  float acc = 0.0f;
  for (int k = 0; k < K; ++k) {
    const float a = __bfloat162float(A[row + k * lda]);
    const float b = __bfloat162float(B[k + col * ldb]);
    acc += a * b;
  }

  C[row + col * ldc] = acc;
}

void x_bf16(int M, int N, int K, const __nv_bfloat16 *d_A,
            const __nv_bfloat16 *d_B, float *d_C,
            cudaStream_t stream) {

  constexpr int TILE = 16;
  dim3 block(TILE, TILE, 1);
  dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE, 1);
  gemm_bf16_naive_kernel<TILE>
      <<<grid, block, 0, stream>>>(d_A, d_B, d_C, M, N, K);
  return;
}

} // namespace xgemms
