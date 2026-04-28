#include <cooperative_groups.h>


#include <cuda/barrier>

#include <cuda/ptx>

#include <cuda_awbarrier_primitives.h>

__global__ void init_barrier0()
{
  __shared__ cuda::barrier<cuda::thread_scope_block> bar;
  auto block = cooperative_groups::this_thread_block();

  if (block.thread_rank() == 0)
  {
    // A single thread initializes the total expected arrival count.
    init(&bar, block.size());
  }
  block.sync();
}


__global__ void init_barrier1()
{
  __shared__ uint64_t bar;
  auto block = cooperative_groups::this_thread_block();

  if (block.thread_rank() == 0)
  {
    // A single thread initializes the total expected arrival count.
    cuda::ptx::mbarrier_init(&bar, block.size());
  }
  block.sync();
}

__global__ void init_barrier2()
{
  __shared__ uint64_t bar;
  auto block = cooperative_groups::this_thread_block();

  if (block.thread_rank() == 0)
  {
    // A single thread initializes the total expected arrival count.
    __mbarrier_init(&bar, block.size());
  }
  block.sync();
}


int main() {
    return 0;
}
