#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <thread>
#include <vector>

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "common.cuh"
#include "cugemms.cuh"
#include "xgemms.cuh"

template <typename T>
class DeviceBuffer {
public:
  DeviceBuffer() = default;

  explicit DeviceBuffer(std::size_t count) { allocate(count); }

  void allocate(std::size_t count) {
    T *raw = nullptr;
    CUDA_CHECK(cudaMalloc(&raw, count * sizeof(T)));
    ptr_.reset(raw);
    size_ = count;
  }

  T *get() noexcept { return ptr_.get(); }
  const T *get() const noexcept { return ptr_.get(); }

  std::size_t size() const noexcept { return size_; }

private:
  struct Deleter {
    void operator()(T *ptr) const noexcept { cudaFree(ptr); }
  };

  std::unique_ptr<T, Deleter> ptr_;
  std::size_t size_ = 0;
};

struct CublasHandle {
  CublasHandle() { CUBLAS_CHECK(cublasCreate(&handle_)); }

  CublasHandle(const CublasHandle &) = delete;
  CublasHandle &operator=(const CublasHandle &) = delete;
  CublasHandle(CublasHandle &&) = delete;
  CublasHandle &operator=(CublasHandle &&) = delete;

  ~CublasHandle() {
    if (handle_ != nullptr) {
      cublasDestroy(handle_);
    }
  }

  cublasHandle_t get() const { return handle_; }

private:
  cublasHandle_t handle_ = nullptr;
};

template <typename T>
void parallel_random_fill(std::vector<T> &vec, std::uint32_t seed, float lo,
                          float hi) {
  const std::size_t n = vec.size();
  if (n == 0)
    return;

  const unsigned int num_threads =
      std::max(1u, std::thread::hardware_concurrency());
  const std::size_t block_size = (n + num_threads - 1) / num_threads;

  std::vector<std::thread> workers;
  workers.reserve(num_threads);

  for (unsigned int t = 0; t < num_threads; ++t) {
    const std::size_t start = t * block_size;
    const std::size_t end = std::min(start + block_size, n);
    if (start >= end)
      break;

    workers.emplace_back([&, start, end, t]() {
      // Unique seed per thread: combine user seed with thread index
      std::mt19937 rng(seed + t);
      std::uniform_real_distribution<float> dist(lo, hi);
      for (std::size_t i = start; i < end; ++i) {
        vec[i] = static_cast<T>(dist(rng));
      }
    });
  }

  for (auto &w : workers) {
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
inline BenchmarkResult benchmark_kernel(const GemmShape &shape,
                                        LaunchFn &&kernel, int warmup_iters,
                                        int bench_iters) {

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

  auto gemm_tflops = [](const GemmShape &shape, double elapsed_ms) {
    if (elapsed_ms <= 0.0) {
      return 0.0;
    }
    auto gemm_flops = [](const GemmShape &shape) {
      return 2.0 * static_cast<double>(shape.m) * static_cast<double>(shape.n) *
             static_cast<double>(shape.k);
    };

    return gemm_flops(shape) / (elapsed_ms * 1.0e9);
  };

  result.total_ms = total_ms;
  result.avg_ms =
      (bench_iters > 0) ? (total_ms / static_cast<float>(bench_iters)) : 0.0f;
  result.tflops = gemm_tflops(shape, static_cast<double>(result.avg_ms));

  cudaStreamDestroy(stream);
  return result;
}

// Column Major CPU parallelized Mat Mul
template <typename T>
std::vector<T> cpu_matmul_nn(const std::vector<T> &A, const std::vector<T> &B,
                             std::size_t m, std::size_t n, std::size_t k) {
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

template <typename InputT, typename Xkernel, typename CuKernel>
std::tuple<Report, Report>
runner(const GemmShape &shape, Xkernel &&x_kernel, CuKernel &&cu_kernel,
       unsigned int seed_a, unsigned int seed_b, float lo = 0.0f,
       float hi = 1.0f, int warm_up_iters = 10, int bench_iters = 100) {

  auto l2_error = [](const std::vector<float> &ref,
                     const std::vector<float> &got) -> double {
    if (ref.size() != got.size()) {
      throw std::runtime_error("l2_error: size mismatch");
    }

    double sum_sq = 0.0;

#pragma omp parallel for reduction(+ : sum_sq) schedule(static)
    for (std::size_t i = 0; i < ref.size(); ++i) {
      const double diff =
          static_cast<double>(got[i]) - static_cast<double>(ref[i]);
      sum_sq += diff * diff;
    }

    return std::sqrt(sum_sq);
  };

  std::vector<InputT> hxA(shape.m * shape.k);
  std::vector<InputT> hxB(shape.k * shape.n);
  std::vector<float> refxC(shape.m * shape.n);

  DeviceBuffer<InputT> dxA(hxA.size());
  DeviceBuffer<InputT> dxB(hxB.size());
  DeviceBuffer<float> dxC(refxC.size());

  std::vector<float> cublasxC(shape.m * shape.n); // cublas result
  std::vector<float> customxC(shape.m * shape.n); // custom kernel result

  parallel_random_fill(hxA, seed_a, lo, hi);
  parallel_random_fill(hxB, seed_b, lo, hi);

  { // Reusing the handle shouldn't be an issue, but still, keeping things fresh
    // for the benchmark section anyways
    CublasHandle ref_cublas_handle;
    cudaStream_t s;
    cudaStreamCreate(&s);

    std::vector<float> refA(shape.m * shape.k);
    std::vector<float> refB(shape.k * shape.n);

    for (std::size_t i = 0; i < refA.size(); ++i)
      refA[i] = static_cast<float>(hxA[i]);
    for (std::size_t i = 0; i < refB.size(); ++i)
      refB[i] = static_cast<float>(hxB[i]);

    DeviceBuffer<float> refDxA(refA.size());
    DeviceBuffer<float> refDxB(refB.size());
    DeviceBuffer<float> refDxC(refxC.size());

    CUDA_CHECK(cudaMemcpy(refDxA.get(), refA.data(),
                          refA.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(refDxB.get(), refB.data(),
                          refB.size() * sizeof(float), cudaMemcpyHostToDevice));

    cugemms::fp32_pedantic_nn(ref_cublas_handle.get(), shape.m, shape.n,
                              shape.k, refDxA.get(), refDxB.get(), refDxC.get(),
                              s);
    CUDA_CHECK(cudaMemcpy(refxC.data(), dxC.get(), dxC.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemset(dxC.get(), 0, dxC.size() * sizeof(float)));
    cudaStreamDestroy(s);
  }

  CublasHandle cublas_handle;

  auto loaded_cu_kernel = [&](cudaStream_t stream) {
    cu_kernel(cublas_handle.get(), shape.m, shape.n, shape.k, dxA.get(),
              dxB.get(), dxC.get(), stream);
  };

  CUDA_CHECK(cudaMemcpy(cublasxC.data(), dxC.get(), dxC.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemset(dxC.get(), 0, dxC.size() * sizeof(float)));

  CUDA_CHECK(cudaDeviceSynchronize());

  auto loaded_x_kernel = [&](cudaStream_t stream) {
    x_kernel(shape.m, shape.n, shape.k, dxA.get(), dxB.get(), dxC.get(),
             stream);
  };

  CUDA_CHECK(cudaMemcpy(customxC.data(), dxC.get(), dxC.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemset(dxC.get(), 0, dxC.size() * sizeof(float)));

  BenchmarkResult cu_bench_result =
      benchmark_kernel(shape, loaded_cu_kernel, warm_up_iters, bench_iters);
  BenchmarkResult x_bench_result =
      benchmark_kernel(shape, loaded_x_kernel, warm_up_iters, bench_iters);

  double cublas_l2 = l2_error(refxC, cublasxC);
  double custom_l2 = l2_error(refxC, customxC);

  return {{cu_bench_result, cublas_l2}, {x_bench_result, custom_l2}};
}

void print_comparison(const std::tuple<Report, Report> &res) {
  const auto &[cublas_report, custom_report] = res;

  const auto &shape = cublas_report.res.shape;

  // cuBLAS results
  float cublas_avg_ms = cublas_report.res.avg_ms;
  double cublas_tflops = cublas_report.res.tflops;
  double cublas_l2 = cublas_report.l2_err;

  // Custom kernel results
  float custom_avg_ms = custom_report.res.avg_ms;
  double custom_tflops = custom_report.res.tflops;
  double custom_l2 = custom_report.l2_err;

  // Derived metrics
  float cublas_avg_sec = cublas_avg_ms / 1000.0f;
  float custom_avg_sec = custom_avg_ms / 1000.0f;
  double speed_pct = (cublas_avg_ms > 0.0f)
                         ? (static_cast<double>(cublas_avg_ms) /
                            static_cast<double>(custom_avg_ms)) *
                               100.0
                         : 0.0;

  constexpr int W = 18;

  std::cout << "\n" << std::string(80, '=') << "\n";
  std::cout << "  GEMM Benchmark Comparison\n";
  std::cout << "  Shape:  M=" << shape.m << ", N=" << shape.n
            << ", K=" << shape.k << "\n";
  std::cout << "  Warmup: " << cublas_report.res.warmup_iters
            << " iters  |  Bench: " << cublas_report.res.bench_iters
            << " iters\n";
  std::cout << std::string(80, '=') << "\n";

  auto print_row = [](const std::string &label, float time_s, float time_ms,
                      double tflops, double l2, double speed_pct,
                      bool is_cublas) {
    std::cout << "  " << std::left << std::setw(12) << label;
    std::cout << "  " << std::right << std::fixed << std::setprecision(6)
              << std::setw(W) << time_s << " s";
    std::cout << "  " << std::fixed << std::setprecision(3) << std::setw(W)
              << time_ms << " ms";
    std::cout << "  " << std::fixed << std::setprecision(2) << std::setw(W)
              << tflops << " TFLOPS";
    std::cout << "  " << std::scientific << std::setprecision(2) << std::setw(W)
              << l2 << " L2 err";
    if (is_cublas) {
      std::cout << "  " << std::string(W, ' ') << " (baseline)";
    } else {
      std::cout << "  " << std::fixed << std::setprecision(1) << std::setw(W)
                << speed_pct << " % speed";
    }
    std::cout << "\n";
  };

  print_row("cuBLAS", cublas_avg_sec, cublas_avg_ms, cublas_tflops, cublas_l2,
            0.0, true);
  print_row("Custom", custom_avg_sec, custom_avg_ms, custom_tflops, custom_l2,
            speed_pct, false);

  std::cout << std::string(80, '-') << "\n";

  if (speed_pct >= 100.0) {
    std::cout << "  >>> Custom kernel is " << std::fixed << std::setprecision(1)
              << (speed_pct - 100.0) << "% FASTER than cuBLAS.\n";
  } else {
    std::cout << "  >>> Custom kernel achieves " << std::fixed
              << std::setprecision(1) << speed_pct
              << "% of cuBLAS performance."
              << "  (" << std::fixed << std::setprecision(1)
              << (100.0 - speed_pct) << "% slower)\n";
  }
  std::cout << std::string(80, '=') << "\n" << std::endl;
}
