#include <memory>
#include <new>
#include <vector>
#include <random>
#include <thread>
#include <iostream>
#include <variant>

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
                                   LaunchFn&& kernel,
                                   int warmup_iters,
                                   int bench_iters
                                   ) {


    cudaStream_t stream;
    cudaStreamCreate(&stream);

    BenchmarkResult result{};
    result.shape = shape;
    result.warmup_iters = warmup_iters;
    result.bench_iters = bench_iters;

    for (int i = 0; i < warmup_iters; ++i) {
        kernel(stream);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaEvent_t start{};
    cudaEvent_t stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < bench_iters; ++i) {
        kernel(stream);
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

    cudaStreamDestroy(stream);
    return result;
}

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

struct Report {
    BenchmarkResult res;
    double l2_err = 0.0f;
};

template<typename InputT,  typename Xkernel, typename CuKernel>
std::tuple<Report, Report> runner (const GemmShape& shape, Xkernel&& x_kernel, CuKernel&& cu_kernel, unsigned int seed_a, unsigned int seed_b, float lo = 0.0f, float hi = 1.0f, int warm_up_iters = 10, int bench_iters = 100) {

    auto l2_error = [](const std::vector<float>& ref,
                       const std::vector<float>& got) -> double {
        if (ref.size() != got.size()) {
            throw std::runtime_error("l2_error: size mismatch");
        }

        double sum_sq = 0.0;

        #pragma omp parallel for reduction(+:sum_sq) schedule(static)
        for (std::size_t i = 0; i < ref.size(); ++i) {
            const double diff = static_cast<double>(got[i]) - static_cast<double>(ref[i]);
            sum_sq += diff * diff;
        }

        return std::sqrt(sum_sq);
    };

    std::vector<InputT> hxA(shape.m * shape.k);
    std::vector<InputT> hxB(shape.k * shape.n);
    std::vector<float>  refxC(shape.m * shape.n);

    DeviceBuffer<InputT> dxA(hxA.size());
    DeviceBuffer<InputT> dxB(hxB.size());
    DeviceBuffer<float>  dxC(refxC.size());


    std::vector<float>  cublasxC(0,shape.m * shape.n); // cublas result
    std::vector<float>  customxC(0,shape.m * shape.n);   // custom kernel result


    parallel_random_fill(hxA, seed_a, lo, hi);
    parallel_random_fill(hxB, seed_b, lo, hi);

    {   // Reusing the handle shouldn't be an issue, but still, keeping things fresh for the benchmark section anyways
        CublasHandle ref_cublas_handle;
        cudaStream_t s;
        cudaStreamCreate(&s);

        CUDA_CHECK(cudaMemcpy(dxA.get(), hxA.data(), dxA.size() * sizeof(InputT), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dxB.get(), hxB.data(), dxB.size() * sizeof(InputT), cudaMemcpyHostToDevice));
        cugemms::fp32_pedantic_nn(ref_cublas_handle.get(), shape.m, shape.n, shape.k, dxA.get(), dxB.get(), dxC.get(),s); // Reference fp32 pendantic kernel
        CUDA_CHECK(cudaMemcpy(refxC.data(), dxC.get(), dxC.size() * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemset(dxC.get(),0, dxC.size() * sizeof(float)));
        cudaStreamDestroy(s);
    }



    CublasHandle cublas_handle;

    auto loaded_cu_kernel =  [&](cudaStream_t stream) {
        cu_kernel(cublas_handle.get(),
            shape.m,
            shape.n,
            shape.k,
            dxA.get(),
            dxB.get(),
            dxC.get(),
            stream);
        };

    CUDA_CHECK(cudaMemcpy(cublasxC.data(), dxC.get(), dxC.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemset(dxC.get(),0, dxC.size() * sizeof(float)));

    CUDA_CHECK(cudaDeviceSynchronize());


    auto loaded_x_kernel =  [&](cudaStream_t stream) {
        x_kernel(
            shape.m,
            shape.n,
            shape.k,
            dxA.get(),
            dxB.get(),
            dxC.get(),
            stream);
        };

    CUDA_CHECK(cudaMemcpy(customxC.data(), dxC.get(), dxC.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemset(dxC.get(),0, dxC.size() * sizeof(float)));


    BenchmarkResult cu_bench_result = benchmark_kernel(shape, loaded_cu_kernel, warm_up_iters, bench_iters);
    BenchmarkResult x_bench_result = benchmark_kernel(shape, loaded_x_kernel, warm_up_iters, bench_iters);

    double cublas_l2 = l2_error(refxC, cublasxC);
    double custom_l2 = l2_error(refxC, customxC);

    return {{cu_bench_result, cublas_l2}, {x_bench_result, custom_l2}};
}









int main () {


    return 0;
}
