#pragma once


template <int TILE>
__global__ void gemm_bf16_r1(const __nv_bfloat16 *__restrict__ A,
                                         const __nv_bfloat16 *__restrict__ B,
                                         float *__restrict__ C, const int M,
                                         const int N, const int K) {
  __shared__ __nv_bfloat16 tile_A[TILE][TILE];
  __shared__ __nv_bfloat16 tile_B[TILE][TILE];

  const int row = blockIdx.y * TILE + threadIdx.y;
  const int col = blockIdx.x * TILE + threadIdx.x;
  if (row >= M || col >= N) {
    return;
  }

  float acc = 0.0f;
  for (int k = 0; k < K; ++k) {
    acc += static_cast<float>(__bfloat162float(A[row + k * M]) *
                              __bfloat162float(B[k + col * K]));
  }

  C[row + col * M] = acc;
}
