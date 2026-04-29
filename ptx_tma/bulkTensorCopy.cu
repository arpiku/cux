
__device__ __forceinline__
void tma_store_2d(const void* smem, const CUtensorMap* tmap, int x, int y) {
    uint32_t s = __cvta_generic_to_shared(smem);
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];"
        :: "l"(tmap), "r"(x), "r"(y), "r"(s) : "memory");
}

int main() {
    return 0;
}
