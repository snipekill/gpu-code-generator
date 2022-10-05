#define M 96
#define K 64
#define N 32
#define TILE_X 64
#define TILE_Y 16
#define THREADS_PER_BLOCK 16
//---------main body---------
#include "gpu_run.cuh"
__global__ void kernel(floatt* input1, floatt* input2, floatt* output) {
	__shared__ float input1_s[TILE_X*TILE_Y];
	__shared__ float input2_s[TILE_X*TILE_Y];
	float temp = 0;
	int col_idx = blockIdx.x*blockDim.x + threadIdx.x;
	int row_idx = blockIdx.y*blockDim.y + threadIdx.y;
	for (int i=0; i<K; i+=TILE_X) {
		for (int t=0; t<TILE_X; t+=THREADS_PER_BLOCK) {
			input1_s[threadIdx.y*TILE_X + threadIdx.x + t] = input1[row_idx*K + i + threadIdx.x + t];
			input2_s[(threadIdx.y+t)*TILE_Y + threadIdx.x] = input2[(i+threadIdx.y+t)*N + col_idx];
		}
		__syncthreads();
		for (int j=0; j<TILE_X; j++) {
			temp += input1_s[threadIdx.y*TILE_X + j] * input2_s[j*TILE_Y + threadIdx.x];
		}
		__syncthreads();
	}
	if (row_idx < M && col_idx < N) {
		output[row_idx*N + col_idx] = temp;
	}
}