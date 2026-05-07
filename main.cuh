#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>
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

__global__ void l2_error_kernel(const float *ref, const float *got,
                                 std::size_t n, double *sum_sq) {
  const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                          static_cast<std::size_t>(threadIdx.x);
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;

  double local_sum = 0.0;
  for (std::size_t i = idx; i < n; i += stride) {
    const double diff = static_cast<double>(got[i]) - static_cast<double>(ref[i]);
    local_sum += diff * diff;
  }

  if (local_sum != 0.0) {
    atomicAdd(sum_sq, local_sum);
  }
}

inline double cuda_l2_error(const std::vector<float> &ref,
                            const std::vector<float> &got) {
  if (ref.size() != got.size()) {
    throw std::runtime_error("l2_error: size mismatch");
  }

  const std::size_t n = ref.size();
  if (n == 0) {
    return 0.0;
  }

  DeviceBuffer<double> d_sum(1);
  CUDA_CHECK(cudaMemset(d_sum.get(), 0, sizeof(double)));

  DeviceBuffer<float> d_ref(n);
  DeviceBuffer<float> d_got(n);

  CUDA_CHECK(cudaMemcpy(d_ref.get(), ref.data(), n * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_got.get(), got.data(), n * sizeof(float),
                        cudaMemcpyHostToDevice));

  constexpr int threads_per_block = 256;
  const int blocks = static_cast<int>((n + threads_per_block - 1) / threads_per_block);

  l2_error_kernel<<<blocks, threads_per_block>>>(d_ref.get(), d_got.get(), n,
                                                 d_sum.get());
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  double sum_sq = 0.0;
  CUDA_CHECK(cudaMemcpy(&sum_sq, d_sum.get(), sizeof(double),
                        cudaMemcpyDeviceToHost));
  return std::sqrt(sum_sq);
}

struct Report {
  BenchmarkResult res;
  double l2_err = 0.0f;
};

struct ComparisonRow {
  GemmShape shape{};
  float cublas_avg_ms = 0.0f;
  float custom_avg_ms = 0.0f;
  double cublas_tflops = 0.0;
  double custom_tflops = 0.0;
  double cublas_l2 = 0.0;
  double custom_l2 = 0.0;
  double speed_pct = 0.0;
  double speed_delta_pct = 0.0;
  int warmup_iters = 0;
  int bench_iters = 0;
};

template <typename InputT, typename Xkernel, typename CuKernel>
std::tuple<Report, Report> runner(const GemmShape &shape, Xkernel &&x_kernel, CuKernel &&cu_kernel,
       unsigned int seed_a, unsigned int seed_b, float lo = 0.0f,
       float hi = 1.0f, int warm_up_iters = 10, int bench_iters = 50, bool random = true) {

  std::vector<InputT> hxA(shape.m * shape.k, 1);
  std::vector<InputT> hxB(shape.k * shape.n, 1);
  std::vector<float> refxC(shape.m * shape.n);

  DeviceBuffer<InputT> dxA(hxA.size());
  DeviceBuffer<InputT> dxB(hxB.size());
  DeviceBuffer<float> dxC(refxC.size());

  std::vector<float> cublasxC(shape.m * shape.n);
  std::vector<float> customxC(shape.m * shape.n);

if (random) {
  parallel_random_fill(hxA, seed_a, lo, hi);
  parallel_random_fill(hxB, seed_b, lo, hi);
}
  // Upload benchmark inputs once so both kernels read the same data.
  CUDA_CHECK(cudaMemcpy(dxA.get(), hxA.data(), hxA.size() * sizeof(InputT),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dxB.get(), hxB.data(), hxB.size() * sizeof(InputT),
                        cudaMemcpyHostToDevice));

  { // Reusing the handle shouldn't be an issue, but still, keeping things fresh
    // for the benchmark section anyways
    CublasHandle ref_cublas_handle;
    cudaStream_t s;
    CUDA_CHECK(cudaStreamCreate(&s));

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
                          refA.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(refDxB.get(), refB.data(),
                          refB.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    cugemms::fp32_pedantic_nn(ref_cublas_handle.get(), shape.m, shape.n,
                              shape.k, refDxA.get(), refDxB.get(),
                              refDxC.get(), s);
    CUDA_CHECK(cudaStreamSynchronize(s));
    CUDA_CHECK(cudaMemcpy(refxC.data(), refDxC.get(),
                          refDxC.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaStreamDestroy(s));
  }

  CublasHandle cublas_handle;

  auto loaded_cu_kernel = [&](cudaStream_t stream) {
    cu_kernel(cublas_handle.get(), shape.m, shape.n, shape.k, dxA.get(),
              dxB.get(), dxC.get(), stream);
  };

  CUDA_CHECK(cudaMemset(dxC.get(), 0, dxC.size() * sizeof(float)));
  BenchmarkResult cu_bench_result =
      benchmark_kernel(shape, loaded_cu_kernel, warm_up_iters, bench_iters);

  CUDA_CHECK(cudaMemcpy(cublasxC.data(), dxC.get(),
                        dxC.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaMemset(dxC.get(), 0, dxC.size() * sizeof(float)));

  auto loaded_x_kernel = [&](cudaStream_t stream) {
    x_kernel(shape.m, shape.n, shape.k, dxA.get(), dxB.get(), dxC.get(),
             stream);
  };

  BenchmarkResult x_bench_result =
      benchmark_kernel(shape, loaded_x_kernel, warm_up_iters, bench_iters);

  CUDA_CHECK(cudaMemcpy(customxC.data(), dxC.get(),
                        dxC.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));

  double cublas_l2 = cuda_l2_error(refxC, cublasxC);
  double custom_l2 = cuda_l2_error(refxC, customxC);

  return {{cu_bench_result, cublas_l2}, {x_bench_result, custom_l2}};
}

inline std::string format_scientific(double value, int precision = 2) {
  std::ostringstream oss;
  oss << std::scientific << std::setprecision(precision) << value;
  return oss.str();
}

inline std::string format_seconds(double seconds, int precision = 3) {
  std::ostringstream oss;
  oss << format_scientific(seconds, precision) << " s";
  return oss.str();
}

inline std::string format_speed_pct(double pct) {
  std::ostringstream oss;
  const double delta = pct - 100.0;
  oss << std::fixed << std::setprecision(1) << pct << "% [";
  if (delta >= 0.0) {
    oss << "+";
  }
  oss << std::fixed << std::setprecision(1) << delta << "%]";
  return oss.str();
}

inline std::string format_shape(const GemmShape &shape) {
  std::ostringstream oss;
  oss << shape.m << "x" << shape.n << "x" << shape.k;
  return oss.str();
}

inline ComparisonRow make_comparison_row(const std::tuple<Report, Report> &res) {
  const auto &[cublas_report, custom_report] = res;
  const auto &shape = cublas_report.res.shape;

  ComparisonRow row{};
  row.shape = shape;
  row.cublas_avg_ms = cublas_report.res.avg_ms;
  row.custom_avg_ms = custom_report.res.avg_ms;
  row.cublas_tflops = cublas_report.res.tflops;
  row.custom_tflops = custom_report.res.tflops;
  row.cublas_l2 = cublas_report.l2_err;
  row.custom_l2 = custom_report.l2_err;
  row.speed_pct = (row.cublas_avg_ms > 0.0f && row.custom_avg_ms > 0.0f)
                      ? (static_cast<double>(row.cublas_avg_ms) /
                         static_cast<double>(row.custom_avg_ms)) * 100.0
                      : 0.0;
  row.speed_delta_pct = row.speed_pct - 100.0;
  row.warmup_iters = cublas_report.res.warmup_iters;
  row.bench_iters = cublas_report.res.bench_iters;
  return row;
}

inline void print_comparison_table(const std::vector<ComparisonRow> &rows) {
  if (rows.empty()) {
    std::cout << "No benchmark rows to print.\n";
    return;
  }

  constexpr int W_SHAPE = 12;
  constexpr int W_SECS = 16;
  constexpr int W_PCT = 22;
  constexpr int W_L2 = 18;
  constexpr int W_TF = 14;

  std::cout << std::string(132, '=') << "\n";
  std::cout << "  GEMM Benchmark Summary\n";
  std::cout << "  Warmup iters: " << rows.front().warmup_iters
            << " | Bench iters: " << rows.front().bench_iters << "\n";
  std::cout << std::string(132, '-') << "\n";

  std::cout << std::left << std::setw(W_SHAPE) << "Shape"
            << std::right << std::setw(W_SECS) << "cuBLAS (s)"
            << std::setw(W_SECS) << "Custom (s)"
            << std::setw(W_PCT) << "Custom % of cuBLAS"
            << std::setw(W_L2) << "cuBLAS L2"
            << std::setw(W_L2) << "Custom L2"
            << std::setw(W_TF) << "cuBLAS TF"
            << std::setw(W_TF) << "Custom TF"
            << "\n";
  std::cout << std::string(132, '-') << "\n";

  double sum_cublas_sec = 0.0;
  double sum_custom_sec = 0.0;
  double sum_speed_pct = 0.0;

  const ComparisonRow *best_row = &rows.front();
  const ComparisonRow *worst_row = &rows.front();

  for (const auto &row : rows) {
    const double cublas_sec = static_cast<double>(row.cublas_avg_ms) / 1000.0;
    const double custom_sec = static_cast<double>(row.custom_avg_ms) / 1000.0;

    sum_cublas_sec += cublas_sec;
    sum_custom_sec += custom_sec;
    sum_speed_pct += row.speed_pct;

    if (row.speed_pct > best_row->speed_pct) {
      best_row = &row;
    }
    if (row.speed_pct < worst_row->speed_pct) {
      worst_row = &row;
    }

    std::cout << std::left << std::setw(W_SHAPE) << format_shape(row.shape)
              << std::right << std::setw(W_SECS) << format_seconds(cublas_sec)
              << std::setw(W_SECS) << format_seconds(custom_sec)
              << std::setw(W_PCT) << format_speed_pct(row.speed_pct)
              << std::setw(W_L2) << format_scientific(row.cublas_l2)
              << std::setw(W_L2) << format_scientific(row.custom_l2)
              << std::setw(W_TF) << format_scientific(row.cublas_tflops, 2)
              << std::setw(W_TF) << format_scientific(row.custom_tflops, 2)
              << "\n";
  }

  const double avg_cublas_sec = sum_cublas_sec / static_cast<double>(rows.size());
  const double avg_custom_sec = sum_custom_sec / static_cast<double>(rows.size());
  const double avg_speed_pct = sum_speed_pct / static_cast<double>(rows.size());

  std::cout << std::string(132, '-') << "\n";
  std::cout << "  Overall avg custom speed: " << format_speed_pct(avg_speed_pct)
            << " | Avg cuBLAS: " << format_seconds(avg_cublas_sec)
            << " | Avg custom: " << format_seconds(avg_custom_sec) << "\n";
  std::cout << "  Best  custom speed:      "
            << format_speed_pct(best_row->speed_pct) << " @ "
            << format_shape(best_row->shape) << " ("
            << format_seconds(static_cast<double>(best_row->cublas_avg_ms) / 1000.0)
            << " vs "
            << format_seconds(static_cast<double>(best_row->custom_avg_ms) / 1000.0)
            << ")\n";
  std::cout << "  Worst custom speed:      "
            << format_speed_pct(worst_row->speed_pct) << " @ "
            << format_shape(worst_row->shape) << " ("
            << format_seconds(static_cast<double>(worst_row->cublas_avg_ms) / 1000.0)
            << " vs "
            << format_seconds(static_cast<double>(worst_row->custom_avg_ms) / 1000.0)
            << ")\n";
  std::cout << std::string(132, '=') << "\n";
}

std::vector<GemmShape> get_shape_list(uint32_t n0, uint32_t n1) {
  std::vector<GemmShape> shapes(n1 - n0 + 1);
  for (uint32_t i = n0; i <= n1; ++i) {
    uint32_t dim = 1u << i;
    shapes[i - n0] = GemmShape{dim, dim, dim};
  }
  return shapes;
}
