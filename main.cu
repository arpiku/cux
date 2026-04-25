#include <cuda_runtime.h>
#include <memory>
#include <vector>
#include <cuda_bf16.h>
#include <random>
#include <thread>
#include <iostream>
#include <cublas_v2.h>

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
    double perf_pct = 0.0;

    float error = 0.0f;
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


struct CaseRun {
    BenchmarkResult result;
    std::vector<float> output;
};








int main () {
    const int N = 512;
    std::vector<__nv_bfloat16> hxA(N);
    std::vector<__nv_bfloat16> hxB(N);

    parallel_random_fill(hxA, 0, 0.0f, 1.0f);
    parallel_random_fill(hxB, 1, 0.0f, 1.0f);


    DeviceBuffer<__nv_bfloat16> dxA(1024);
    DeviceBuffer<__nv_bfloat16> dxB(1024);
    DeviceBuffer<__nv_bfloat16> dxC(1024);

    // cudaMemcpy(dxA.get(), const void *src, size_t count, enum cudaMemcpyKind kind)






    return 0;
}
