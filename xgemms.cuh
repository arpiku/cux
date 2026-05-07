#pragma once

#include "common.cuh"

#include "r0.cuh"

namespace xgemms {
    // Simple Naive kernel
    void x0_bf16_0(int M, int N, int K, const __nv_bfloat16 *d_A,
        const __nv_bfloat16 *d_B, float *d_C,
        cudaStream_t stream) {
            constexpr int TILE = 16;
            dim3 block(TILE, TILE, 1);
            dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE, 1);
            gemm_bf16_naive_kernel_a<TILE>
            <<<grid, block, 0, stream>>>(d_A, d_B, d_C, M, N, K);
            return;
        }
    // Simple Naive kernel, with tweaked tile size
    void x0_bf16_1(int M, int N, int K, const __nv_bfloat16 *d_A,
        const __nv_bfloat16 *d_B, float *d_C,
        cudaStream_t stream) {
            constexpr int TILE = 8;
            dim3 block(TILE, TILE, 1);
            dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE, 1);
            gemm_bf16_naive_kernel_a<TILE>
            <<<grid, block, 0, stream>>>(d_A, d_B, d_C, M, N, K);
            return;
        }
    // Simple Naive kernel, with tweaked tile size and bfloat16 type
    void x0_bf16_2(int M, int N, int K, const __nv_bfloat16 *d_A,
            const __nv_bfloat16 *d_B, float *d_C,
            cudaStream_t stream) {
                constexpr int TILE = 8;
                dim3 block(TILE, TILE, 1);
                dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE, 1);
                gemm_bf16_naive_kernel_b<TILE>
                <<<grid, block, 0, stream>>>(d_A, d_B, d_C, M, N, K);
                return;
            }

    // Tile size = 8 produces best result, at 32 we are only at ~32 %
    void x0_bf16_3(int M, int N, int K, const __nv_bfloat16 *d_A,
            const __nv_bfloat16 *d_B, float *d_C,
            cudaStream_t stream) {
                constexpr int TILE = 8;
                dim3 block(TILE * TILE);
                dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE, 1);
                gemm_bf16_naive_kernel_c<TILE>
                <<<grid, block, 0, stream>>>(d_A, d_B, d_C, M, N, K);
                return;
            }

    void x0_bf16_4(int M, int N, int K, const __nv_bfloat16 *d_A,
            const __nv_bfloat16 *d_B, float *d_C,
            cudaStream_t stream) {
                constexpr int TILE = 8;
                dim3 block(TILE, TILE, 1);
                dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE, 1);
                gemm_bf16_naive_kernel_d<TILE>
                <<<grid, block, 0, stream>>>(d_A, d_B, d_C, M, N, K);
                return;
            }





} // namespace xgemms
