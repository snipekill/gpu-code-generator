// ------------------------------- STATIC (except datatype) -------------------------------
#include "gpu_run.cuh"
__global__ void reformat_input(float* input, float* output) {
	int row_idx = threadIdx.z*FS*FS + threadIdx.y*FS + threadIdx.x;
	int col_idx = blockIdx.z*OS_X*OS_Y + blockIdx.y*OS_X + blockIdx.x;
	if ( (threadIdx.y+blockIdx.y)<IS_Y && (threadIdx.x+blockIdx.x)<IS_X )
		output[row_idx*N + col_idx] = input[blockIdx.z*IC*IS_X*IS_X + threadIdx.z*IS_X*IS_Y + (threadIdx.y+blockIdx.y)*IS_X + (threadIdx.x+blockIdx.x)];
}
__global__ void reformat_filter(float* input, float* output) {
	int idx_z = blockIdx.z*blockDim.z + threadIdx.z;
	int idx_y = blockIdx.y*blockDim.y + threadIdx.y;
	int idx_x = blockIdx.x*blockDim.x + threadIdx.x;
	int idx_all = idx_z*FS*FS + idx_y*FS + idx_x;
	int row_idx = idx_z/IC;
	int col_idx = idx_all - row_idx*FS*FS*IC;
	output[row_idx*K + col_idx] = input[idx_all];
}
__global__ void kernel(float* input1, float* input2, float* output) {
	__shared__ float input1_s[TILE_X*TILE_Y];
	__shared__ float input2_s[TILE_X*TILE_Y];
	float temp = 0;
	int col_idx = blockIdx.x*blockDim.x + threadIdx.x;
	int row_idx = blockIdx.y*blockDim.y + threadIdx.y;
	for (int i=0; i<K; i+=TILE_X) {
		for (int t=0; t<TILE_X; t+=THREADS_PER_BLOCK_GEMM) {
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
// ------------------------------- STATIC (except datatype) -------------------------------