
#include <cuda_pipeline.h>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace cg = cooperative_groups;

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                         cudaGetErrorString(err));                             \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

__global__ void cp_async_kernel(const float* __restrict__ gmem,
                                float* __restrict__ out, int N) {
    (void)N;

    extern __shared__ float smem[];
    auto block = cg::this_thread_block();

    const int tid = threadIdx.x;
    const int base = blockIdx.x * blockDim.x;

    cg::memcpy_async(block, smem, gmem + base, sizeof(float) * blockDim.x);
    cg::wait(block);

    out[base + tid] = smem[tid] * 2.0f;
}

__global__ void direct_copy_kernel(const float* __restrict__ gmem,
                                   float* __restrict__ out, int N) {
    (void)N;

    extern __shared__ float smem[];
    const int tid = threadIdx.x;
    const int base = blockIdx.x * blockDim.x;

    smem[tid] = gmem[base + tid];
    __syncthreads();

    out[base + tid] = smem[tid] * 2.0f;
}

int main() {
    constexpr int threads = 256;
    constexpr dim3 blocks = dim3(128, 128, 1);
    constexpr int N = threads * blocks.x;
    constexpr int repeats = 100;

    const size_t bytes = sizeof(float) * N;
    const size_t shmem = sizeof(float) * threads;

    std::vector<float> h_in(N, 1.0f);

    float* d_in = nullptr;
    float* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

    auto bench = [&](auto launch) {
        cudaEvent_t start{}, stop{};
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        launch();
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < repeats; ++i) {
            launch();
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));

        return static_cast<double>(ms) * 1e-3 / repeats;
    };

    auto launch_cp_async = [&] {
        cp_async_kernel<<<blocks, threads, shmem>>>(d_in, d_out, N);
    };

    auto launch_direct = [&] {
        direct_copy_kernel<<<blocks, threads, shmem>>>(d_in, d_out, N);
    };

    const double t_cp_async = bench(launch_cp_async);
    const double t_direct = bench(launch_direct);

    std::printf("cp.async time: %.6e s\n", t_cp_async);
    std::printf("direct time:   %.6e s\n", t_direct);
    std::printf("difference (direct - cp.async): %.6e s\n",
                t_direct - t_cp_async);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    return 0;
}
