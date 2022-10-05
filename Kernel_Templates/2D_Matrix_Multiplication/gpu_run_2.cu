#define INPUT_SIZE 4096
#define THREADS_PER_BLOCK 1024
#define BLOCKS ((INPUT_SIZE + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK)
#include "gpu_run.cuh"
void gpu_run(floatt* input1, floatt* input2, floatt* output) {
	floatt* input1_d;
	size_t s_input1 = INPUT_SIZE*sizeof(floatt);
	floatt* input2_d;
	size_t s_input2 = INPUT_SIZE*sizeof(floatt);
	floatt* output_d;
	size_t s_output = INPUT_SIZE*sizeof(floatt);
	cudaMalloc(&input1_d, s_input1);
	cudaMalloc(&input2_d, s_input2);
	cudaMalloc(&output_d, s_output);
	cudaMemcpy(input1_d, input1, s_input1, cudaMemcpyHostToDevice);
	cudaMemcpy(input2_d, input2, s_input2, cudaMemcpyHostToDevice);
	dim3 block_dim(THREADS_PER_BLOCK);
	dim3 grid_dim(BLOCKS);
	kernel<<<grid_dim, block_dim>>>(input1_d, input2_d, output_d);
	kernel<<<1, block_dim>>>(output_d, output_d);
	floatt *output_all = new floatt[INPUT_SIZE];
	cudaMemcpy(output_all, output_d, s_output, cudaMemcpyDeviceToHost);
	cudaFree(input1_d);
	cudaFree(input2_d);
	cudaFree(output_d);
	output[0] = output_all[0];
}