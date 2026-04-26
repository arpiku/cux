#include <memory>
#include <new>
#include <vector>
#include <random>
#include <thread>
#include <iostream>

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>

#include "cugemms.cuh"
#include "common.cuh"

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;

    explicit DeviceBuffer(std::size_t count) {
        allocate(count);
    }

    void allocate(std::size_t count) {
        T* raw = nullptr;
        CUDA_CHECK(cudaMalloc(&raw, count * sizeof(T)));
        ptr_.reset(raw);
        size_ = count;
    }

    T* get() noexcept { return ptr_.get(); }
    const T* get() const noexcept { return ptr_.get(); }

    std::size_t size() const noexcept { return size_; }

private:
    struct Deleter {
        void operator()(T* ptr) const noexcept {
            cudaFree(ptr);
        }
    };

    std::unique_ptr<T, Deleter> ptr_;
    std::size_t size_ = 0;
};


struct CublasHandle {
    CublasHandle() {
        CUBLAS_CHECK(cublasCreate(&handle_));
    }

    CublasHandle(const CublasHandle&) = delete;
    CublasHandle& operator=(const CublasHandle&) = delete;
    CublasHandle(CublasHandle&&) = delete;
    CublasHandle& operator=(CublasHandle&&) = delete;



    ~CublasHandle() {
        if (handle_ != nullptr) {
            cublasDestroy(handle_);
        }
    }

    cublasHandle_t get() const {
        return handle_;
    }

private:
    cublasHandle_t handle_ = nullptr;
};


template<typename T>
void parallel_random_fill(std::vector<T>& vec, std::uint32_t seed, float lo, float hi) {
    const std::size_t n = vec.size();
    if (n == 0) return;

    const unsigned int num_threads = std::max(1u, std::thread::hardware_concurrency());
    const std::size_t block_size = (n + num_threads - 1) / num_threads;

    std::vector<std::thread> workers;
    workers.reserve(num_threads);

    for (unsigned int t = 0; t < num_threads; ++t) {
        const std::size_t start = t * block_size;
        const std::size_t end   = std::min(start + block_size, n);
        if (start >= end) break;

        workers.emplace_back([&, start, end, t]() {
            // Unique seed per thread: combine user seed with thread index
            std::mt19937 rng(seed + t);
            std::uniform_real_distribution<float> dist(lo, hi);
            for (std::size_t i = start; i < end; ++i) {
                vec[i] = static_cast<T>(dist(rng));
            }
        });
    }

    for (auto& w : workers) {
        w.join();
    }
}

struct GemmShape {
    unsigned int m, n, k;
};


struct BenchmarkResult {
    GemmShape shape{};
    int warmup_iters = 0;
    int bench_iters = 0;

    float total_ms = 0.0f;
    float avg_ms = 0.0f;

    double tflops = 0.0;
};


template <typename LaunchFn>
inline BenchmarkResult benchmark_kernel(const GemmShape& shape,
                                   LaunchFn&& launch,
                                   int warmup_iters,
                                   int bench_iters,
                                   cudaStream_t stream) {
    BenchmarkResult result{};
    result.shape = shape;
    result.warmup_iters = warmup_iters;
    result.bench_iters = bench_iters;

    for (int i = 0; i < warmup_iters; ++i) {
        launch(shape, stream);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaEvent_t start{};
    cudaEvent_t stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < bench_iters; ++i) {
        launch(shape, stream);
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    auto gemm_tflops = [](const GemmShape& shape, double elapsed_ms) {
        if (elapsed_ms <= 0.0) {
            return 0.0;
        }
        auto gemm_flops = [](const GemmShape& shape) {
            return 2.0 * static_cast<double>(shape.m) *
                   static_cast<double>(shape.n) *
                   static_cast<double>(shape.k);
        };

        return gemm_flops(shape) / (elapsed_ms * 1.0e9);
    };

    result.total_ms = total_ms;
    result.avg_ms = (bench_iters > 0) ? (total_ms / static_cast<float>(bench_iters)) : 0.0f;
    result.tflops = gemm_tflops(shape, static_cast<double>(result.avg_ms));

    return result;
}

struct BenchresultAndOutput {
    BenchmarkResult result;
    std::vector<float> output;
};

template <typename InputT, typename LaunchFn>
BenchresultAndOutput run_gemm_case(const GemmShape& shape,
                              const std::vector<InputT>& h_a,
                              const std::vector<InputT>& h_b,
                              cublasHandle_t handle,
                              LaunchFn&& launch,
                              int warmup_iters,
                              int bench_iters,
                              cudaStream_t stream) {

    auto size = [] (uint r, uint c) { return r * c; };

    DeviceBuffer<InputT> d_a(size(shape.m, shape.k));
    DeviceBuffer<InputT> d_b(size(shape.k, shape.n));
    DeviceBuffer<float> d_c(size(shape.m, shape.n));

    CUDA_CHECK(cudaMemcpy(d_a.get(), h_a.data(), d_a.size() * sizeof(InputT), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b.get(), h_b.data(), d_b.size() * sizeof(InputT), cudaMemcpyHostToDevice));

    auto bench_launch = [&](const GemmShape&, cudaStream_t stream) {
        launch(handle,
               shape.m,
               shape.n,
               shape.k,
               d_a.get(),
               d_b.get(),
               d_c.get(),
               stream);
    };

    BenchmarkResult result =
        benchmark_kernel(shape, bench_launch, warmup_iters, bench_iters, stream);

    std::vector<float> output(d_c.size());
    CUDA_CHECK(cudaMemcpy(output.data(),
                          d_c.get(),
                          d_c.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));

    return {result, std::move(output)};
}

/* More generic test function:
 * run_gemm_case(ProblemShape, kernel_to_launch)
 */


// Column Major CPU parallelized Mat Mul
template <typename T>
std::vector<T> cpu_matmul_nn(const std::vector<T>& A,
                                   const std::vector<T>& B,
                                   std::size_t m,
                                   std::size_t n,
                                   std::size_t k) {
    std::vector<T> C(m * n, T{});

    if (A.size() != m * k || B.size() != k * n) {
        return C;
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for (std::size_t j = 0; j < n; ++j) {
        for (std::size_t i = 0; i < m; ++i) {
            T sum = T{};
            for (std::size_t p = 0; p < k; ++p) {
                sum += A[i + p * m] * B[p + j * k];
            }
            C[i + j * m] = sum;
        }
    }

    return C;
}

enum TestMode {
    BF16,
    FP32,
    TF32
};




int main () {
    const int N = 10;
    std::vector<float> hxA(N*N);
    std::vector<float> hxB(N*N);

    std::vector<float> refxC(N*N);

    parallel_random_fill(hxA, 0, 0.0f, 1.0f);
    parallel_random_fill(hxB, 1, 0.0f, 1.0f);


    refxC = cpu_matmul_nn(hxA, hxB, N, N, N);

    const GemmShape gshape({N, N, N});

    CublasHandle cublas_handle;
    int warmup_iters = 20;
    int bench_iters = 100;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    auto result = run_gemm_case(gshape, hxA, hxB, cublas_handle.get(), cugemms::fp32_pedantic_nn,warmup_iters,bench_iters, stream);

    for(auto& val : result.output) {
        std::cout << val << " ";
    }

    std::cout << "----------------------" <<std::endl;
    for(auto& val : refxC) {
        std::cout << val << " ";
    }


    // cudaMemcpy(dxA.get(), const void *src, size_t count, enum cudaMemcpyKind kind)






    return 0;
}
