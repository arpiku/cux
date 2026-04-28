#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>
#include <sys/types.h>
#include <iostream>

//

constexpr int mode_rz = 0;
constexpr int mode_rn = 1;

template <int mode>
struct helper;

template<> struct helper<mode_rz> {
    static constexpr const char mode[] = ".rz";
};

template<> struct helper<mode_rn> {
    static constexpr const char mode[] = ".rn";
};

template <int rounding_mode>
__device__ float compute_add(float a, float b) {
    float result;
    asm ("add.f32%1 %0,%2,%3;" : "=f"(result)
                            : "C"(helper<rounding_mode>::mode),
                              "f"(a), "f"(b));
    return result;
}

__global__ void kern(float *result, float a, float b) {
    *result++ = compute_add<mode_rn>(a,b); // generates add.f32.rn
    *result   = compute_add<mode_rz>(a,b); // generates add.f32.rz
}



int main() {
    return 0;
}
