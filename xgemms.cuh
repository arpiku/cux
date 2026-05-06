#pragma once

#include "common.cuh"

#include "r0.cuh"

namespace xgemms {

    void x0_bf16(int M, int N, int K, const __nv_bfloat16 *d_A,
        const __nv_bfloat16 *d_B, float *d_C,
        cudaStream_t stream) {

            constexpr int TILE = 16;
            dim3 block(TILE, TILE, 1);
            dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE, 1);
            gemm_bf16_naive_kernel<TILE>
            <<<grid, block, 0, stream>>>(d_A, d_B, d_C, M, N, K);
            return;
        }

        void x0a_bf16(int M, int N, int K, const __nv_bfloat16 *d_A,
            const __nv_bfloat16 *d_B, float *d_C,
            cudaStream_t stream) {
                constexpr int TILE = 8;
                dim3 block(TILE, TILE, 1);
                dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE, 1);
                gemm_bf16_naive_kernel<TILE>
                <<<grid, block, 0, stream>>>(d_A, d_B, d_C, M, N, K);
                return;
            }

} // namespace xgemms
