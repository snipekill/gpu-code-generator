#define THREADS_PER_BLOCK 1024
#include "gpu_run.cuh"
__device__ void singleWarpReduce(volatile floatt* input, int idx) {
	input[idx] += input[idx + 64];
	input[idx] += input[idx + 32];
	input[idx] += input[idx + 16];
	input[idx] += input[idx + 8];
	input[idx] += input[idx + 4];
	input[idx] += input[idx + 2];
	input[idx] += input[idx + 1];
}
__global__ void kernel(floatt* input1, floatt* input2, floatt* output) {
	__shared__ float partial_sum[THREADS_PER_BLOCK];
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	partial_sum[threadIdx.x] = input1[idx]*input2[idx];
	__syncthreads();
	for (int s = THREADS_PER_BLOCK/2; s > 64; s = s>>1) {
		if (threadIdx.x < s) {
			partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
		}
		__syncthreads();
	}
	if (threadIdx.x < 64) {
		singleWarpReduce(partial_sum, threadIdx.x);
	}
}
__global__ void kernel(floatt* input, floatt* output) {
	__shared__ float partial_sum[THREADS_PER_BLOCK];
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	partial_sum[threadIdx.x] = input[idx];
	__syncthreads();
	for (int s = THREADS_PER_BLOCK/2; s > 64; s = s>>1) {
		if (threadIdx.x < s) {
			partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
		}
		__syncthreads();
	}
	if (threadIdx.x < 64) {
		singleWarpReduce(partial_sum, threadIdx.x);
	}
	if (threadIdx.x == 0) {
		output[blockIdx.x] = partial_sum[0];
	}
}
