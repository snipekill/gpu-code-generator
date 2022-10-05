// ------------------------------- STATIC (except datatype) -------------------------------
#include "gpu_run.cuh"
__device__ void singleWarpReduce(volatile float* input, int idx) {
// ------------------------------- STATIC (except datatype) -------------------------------

// ------------------------------- PROGRAM -------------------------------
// Here WRAP_SIZE = 32 is assumed. For 64, the sequence is [idx + 64], [idx + 32], ... [idx + 1]
	input[idx] += input[idx + 32];
	input[idx] += input[idx + 16];
	input[idx] += input[idx + 8];
	input[idx] += input[idx + 4];
	input[idx] += input[idx + 2];
	input[idx] += input[idx + 1];
// ------------------------------- PROGRAM -------------------------------

// ------------------------------- STATIC (except datatype) -------------------------------
}
__global__ void kernel(float* input1, float* input2, float* output) {
	__shared__ float partial_sum[THREADS_PER_BLOCK];	
	int idx = blockIdx.x * blockDim.x + threadIdx.x;	
	partial_sum[threadIdx.x] = input1[idx]*input2[idx];
	__syncthreads();
// ------------------------------- STATIC (except datatype) -------------------------------

// ------------------------------- PROGRAM -------------------------------
// Here WRAP_SIZE = 32 is assumed. For 64, the condition changed to s > 64
	for (int s = THREADS_PER_BLOCK/2; s > 32; s = s>>1) {
// ------------------------------- PROGRAM -------------------------------

// ------------------------------- STATIC (except datatype) -------------------------------	
		if (threadIdx.x < s) {
			partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
		}
		__syncthreads();
	}
// ------------------------------- STATIC (except datatype) -------------------------------

// ------------------------------- PROGRAM -------------------------------
// Here WRAP_SIZE = 32 is assumed. For 64, the condition changed to threadIdx.x < 64	
	if (threadIdx.x < 32) {
// ------------------------------- PROGRAM -------------------------------

// ------------------------------- STATIC (except datatype) -------------------------------	
		singleWarpReduce(partial_sum, threadIdx.x);
	}	
	if (threadIdx.x == 0) {
		output[blockIdx.x] = partial_sum[0];
	}
}
__global__ void kernel(float* input, float* output) {
	__shared__ float partial_sum[THREADS_PER_BLOCK];
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	partial_sum[threadIdx.x] = input[idx];
	__syncthreads();
	for (int s = THREADS_PER_BLOCK/2; s > 32; s = s>>1) {
		if (threadIdx.x < s) {
			partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
		}
		__syncthreads();
	}
// ------------------------------- STATIC (except datatype) -------------------------------

// ------------------------------- PROGRAM -------------------------------
// Here WRAP_SIZE = 32 is assumed. For 64, the condition changed to s > 64		
	if (threadIdx.x < 32) {
// ------------------------------- PROGRAM -------------------------------

// ------------------------------- STATIC (except datatype) -------------------------------	
		singleWarpReduce(partial_sum, threadIdx.x);
	}	
	if (threadIdx.x == 0) {
		output[blockIdx.x] = partial_sum[0];
	}
}
// ------------------------------- STATIC (except datatype) -------------------------------