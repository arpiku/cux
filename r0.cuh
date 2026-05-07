#pragma once

template <int TILE>
__global__ void gemm_bf16_naive_kernel_a(const __nv_bfloat16 *__restrict__ A,
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

// The removal of extra variables lead to observable speed improvements
template <int TILE>
__global__ void gemm_bf16_naive_kernel_b(const __nv_bfloat16 *__restrict__ A,
                                         const __nv_bfloat16 *__restrict__ B,
                                         float *__restrict__ C, const int M,
                                         const int N, const int K) {
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

// In the following kernels the error jumps significantly due to bfloat16
// compute Simple naive kernel with 1d block, memory coalescing attempt, but
// results similar to previous kernels
template <int BLOCK_LEN>
__global__ void gemm_bf16_naive_kernel_c(const __nv_bfloat16 *__restrict__ A,
                                         const __nv_bfloat16 *__restrict__ B,
                                         float *__restrict__ C, const int M,
                                         const int N, const int K) {
  const int row = blockIdx.y * BLOCK_LEN + (threadIdx.x / BLOCK_LEN);
  const int col = blockIdx.x * BLOCK_LEN + (threadIdx.x % BLOCK_LEN);
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

// uses __ldg for improved memory access, unroll for better performance
template <int TILE>
__global__ void gemm_bf16_naive_kernel_d(const __nv_bfloat16 *__restrict__ A,
                                         const __nv_bfloat16 *__restrict__ B,
                                         float *__restrict__ C, const int M,
                                         const int N, const int K) {
  int row = blockIdx.y * TILE + threadIdx.y;
  int col = blockIdx.x * TILE + threadIdx.x;

  if (row >= M || col >= N)
    return;

  float sum = 0.0f;
#pragma unroll
  // Pragma unroll is not effective for bf16 data types, infact perfomace is
  // lost ldg also ineffective, perfomance similar to above
  for (int k = 0; k < K; ++k) {
    sum += static_cast<float>(__bfloat162float(*(&A[row + k * M])) *
                              __bfloat162float(*(&B[k + col * K])));
  }
  C[row * N + col] = sum;
}
