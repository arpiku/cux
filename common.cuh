#pragma once

#include <cuda_runtime.h>

inline void cuda_check_impl(cudaError_t status,
                            const char* expr,
                            const char* file,
                            int line) {
    if (status != cudaSuccess) {
        std::fprintf(stderr,
                     "CUDA_CHECK failed: %s\n  status: %s\n  at %s:%d\n",
                     expr,
                     cudaGetErrorString(status),
                     file,
                     line);
        std::fflush(stderr);
        std::abort();
    }
}
inline void cublas_check_impl(cublasStatus_t status,
                              const char* expr,
                              const char* file,
                              int line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr,
                     "CUBLAS_CHECK failed: %s\n  status: %d\n  at %s:%d\n",
                     expr,
                     static_cast<int>(status),
                     file,
                     line);
        std::fflush(stderr);
        std::abort();
    }
}

#define CUDA_CHECK(expr) cuda_check_impl((expr), #expr, __FILE__, __LINE__)
#define CUBLAS_CHECK(expr) cublas_check_impl((expr), #expr, __FILE__, __LINE__)
