#include <cstdint>
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void matmul(uint32_t* A, uint32_t* B, uint32_t* C, uint32_t N) {
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        uint32_t sum = 0;
        for (uint32_t k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
};

void matmul_cpu(uint32_t* A, uint32_t* B, uint32_t* C, uint32_t N) {
    for (uint32_t row = 0; row < N; ++row) {
        for (uint32_t col = 0; col < N; ++col) {
            uint32_t sum = 0;
            for (uint32_t k = 0; k < N; ++k) {
                sum += A[row * N + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
        }
    }
};



int main() {
    uint32_t N = 32;
    uint32_t* hA, *hB, *hC, *refC;
    hA = new uint32_t[N * N];
    hB = new uint32_t[N * N];
    hC = new uint32_t[N * N];
    refC = new uint32_t[N * N];

    for (uint32_t i = 0; i < N * N; i++) {
        hA[i] = 1;
        hB[i] = i+1;
        hC[i] = 0;
    };

    uint32_t* dA, *dB, *dC;
    cudaMalloc(&dA, N * N * sizeof(uint32_t));
    cudaMalloc(&dB, N * N * sizeof(uint32_t));
    cudaMalloc(&dC, N * N * sizeof(uint32_t));

    dim3 block(8,8);
    dim3 grid(16,16);

    cudaMemcpy(dA, hA, N * N * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, N * N * sizeof(uint32_t), cudaMemcpyHostToDevice);
    matmul<<<grid, block>>>(dA, dB, dC, N);
    cudaMemcpy(hC, dC, N * N * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    matmul_cpu(hA, hB, refC, N);

    for (uint32_t i = 0; i < N * N; i++) {
        printf("%u - %u\n", hC[i], refC[i]);
        if (hC[i] != refC[i]) {
            printf("Mismatch at index %u: %u != %u\n", i, hC[i], refC[i]);
        }
    }

    return 0;
}
