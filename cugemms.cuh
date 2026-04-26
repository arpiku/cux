#pragma once

#include "common.cuh"
#include <cublas_v2.h>

namespace cugemms {

inline void fp32_pedantic_nn(cublasHandle_t handle,
                               int m,
                               int n,
                               int k,
                               const float* a,
                               const float* b,
                               float* c,
                               cudaStream_t stream) {
    CUDA_CHECK(cudaGetLastError());
    CUBLAS_CHECK(cublasSetStream(handle, stream));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    CUBLAS_CHECK(cublasGemmEx(handle,
                              CUBLAS_OP_N,
                              CUBLAS_OP_N,
                              m,
                              n,
                              k,
                              &alpha,
                              a,
                              CUDA_R_32F,
                              m,
                              b,
                              CUDA_R_32F,
                              k,
                              &beta,
                              c,
                              CUDA_R_32F,
                              m,
                              CUBLAS_COMPUTE_32F_PEDANTIC,
                              CUBLAS_GEMM_DEFAULT));
}


inline void bf16_tc_nn(cublasHandle_t handle,
                                    int m,
                                    int n,
                                    int k,
                                    const __nv_bfloat16* a,
                                    const __nv_bfloat16* b,
                                    float* c,
                                    cudaStream_t stream) {
    CUDA_CHECK(cudaGetLastError());
    CUBLAS_CHECK(cublasSetStream(handle, stream));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    CUBLAS_CHECK(cublasGemmEx(handle,
                              CUBLAS_OP_N,
                              CUBLAS_OP_N,
                              m,
                              n,
                              k,
                              &alpha,
                              a,
                              CUDA_R_16BF,
                              m,
                              b,
                              CUDA_R_16BF,
                              k,
                              &beta,
                              c,
                              CUDA_R_32F,
                              m,
                              CUBLAS_COMPUTE_32F_FAST_16BF,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

} // namespace cugemms
