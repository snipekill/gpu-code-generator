#define M 96
#define K 64
#define N 32
#define THREADS_PER_BLOCK 16
//---------main body---------
#define BLOCKS_X ((N + THREADS_PER_BLOCK -1)/THREADS_PER_BLOCK)
#define BLOCKS_Y ((M + THREADS_PER_BLOCK -1)/THREADS_PER_BLOCK)
#include "gpu_run.cuh"
void gpu_run(floatt* input1, floatt* input2, floatt* output) {
	floatt* input1_d;
	size_t s_input1 = M*K*sizeof(floatt)
	floatt* input2_d;
	size_t s_input2 = K*N*sizeof(floatt)
	floatt* output_d;
	size_t s_output = M*N*sizeof(floatt)
	cudaMalloc(&input1_d, s_input1);
	cudaMalloc(&input2_d, s_input2);
	cudaMalloc(&output_d, s_output);
	cudaMemcpy(input1_d, input1, s_input1, cudaMemcpyHostToDevice);
	cudaMemcpy(input2_d, input2, s_input2, cudaMemcpyHostToDevice);
	dim3 block_dim(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
	dim3 grid_dim(BLOCKS_X, BLOCKS_Y);
	kernel<<<grid_dim, block_dim>>>(input1_d, input2_d, output_d);
	cudaMemcpy(output, output_d, s_output, cudaMemcpyDeviceToHost);
	cudaFree(input1_d);
	cudaFree(input2_d);
	cudaFree(output_d);
}
