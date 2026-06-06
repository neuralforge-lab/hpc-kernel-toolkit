#include <hip/hip_runtime.h>
#define BLOCK_SIZE 256
__global__ void reduce_sum(float* input, float* output, int n) {
    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    int i = blockIdx.x * BLOCK_SIZE * 2 + threadIdx.x;
    sdata[tid] = (i < n ? input[i] : 0) + (i + BLOCK_SIZE < n ? input[i + BLOCK_SIZE] : 0);
    __syncthreads();
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) output[blockIdx.x] = sdata[0];
}
