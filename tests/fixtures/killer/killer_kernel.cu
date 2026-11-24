// Tiny synthetic killer project (fixture, Tier C/D expected).
// Contains a warp-shuffle kernel AND a dependency with no ROCm equivalent.
#include <cuda_runtime.h>

__global__ void warp_reduce(float *data) {
    // Warp shuffle: warp-size dependent, HIPIFY cannot translate this.
    float v = data[threadIdx.x];
    v += __shfl_down_sync(0xffffffff, v, 16);
    v += __shfl_down_sync(0xffffffff, v, 8);
    data[threadIdx.x] = v;
}
